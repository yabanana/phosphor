#include "platform/metal/direct_lighting_passes.h"
#include "platform/metal/lighting_dispatch.h"
#include "platform/metal/scene_renderer.h"
#include "platform/metal/mesh_renderer.h"
#include "platform/metal/visibility_renderer.h"
#include "platform/metal/acceleration_structures.h"
#include "platform/metal/metal_graph_executor.h"
#include "renderer/local_light_scene.h"
#include "renderer/emissive_domain.h"
#include "renderer/transform_reference.h"
#include "renderer/stochastic_sampling.h"
#include "renderer/scene_store.h"
#include "renderer/gpu_scene.h"
#include "rendergraph/pass_context.h"
#include <glm/gtc/type_ptr.hpp>
namespace phosphor {
struct DirectLightingPasses::Impl {
    enum Pass { Emit, Guide, Pack, Cluster, Candidate, Temporal, Spatial, Shade, Snapshot,SignalClear,CheckClear,CheckNegative,Check,Count };
    MetalContext& c;PipelineCache& p;SceneRenderer& scene;MeshRenderer& mesh;VisibilityRenderer& vis;AccelerationStructures& rt;ShadowPasses& shadow;
    LaunchOptions options;LocalLightScene lightScene;HistoryRegistry history;ShadowPasses::Frame frame;
    di::EmissiveDomainTracker emissiveDomain;std::vector<float> emissiveWorlds;
    std::array<pipe::PipelineHandle,Count> kernels{};pipe::PipelineHandle receiver{};
    MTL::DepthStencilState* equalDepth=nullptr;
    RtConsumer restirConsumer,clusterConsumer;
    struct Slot {
        MTL::Buffer *source=nullptr,*lights=nullptr,*emitters=nullptr,*alias=nullptr,*surface=nullptr;
        MTL::Buffer *fresh=nullptr,*temporal=nullptr,*spatial=nullptr,*cells=nullptr,*indices=nullptr,*check=nullptr,*signalErrors=nullptr;
        std::array<MTL4::ArgumentTable*,Count> tables{};
    };std::array<Slot,METAL_FRAMES_IN_FLIGHT> slots{};
    struct View { MTL::Buffer* reservoir=nullptr;MTL::Buffer* surface=nullptr; };std::array<View,4> views{};
    di::StbnMask stbn;MTL::Buffer* ranks=nullptr;
    u64 pixelCapacity=0,lightCapacity=0,graphVersion=1;u64 sceneSignal=0;
    struct Signal {u64 scene, domain, geometry;u32 light;bool operator==(const Signal&)const=default;};
    Signal lastSignal{~u64{0},0,0,0};u64 signalEpoch=1;
    GPUDIParams params{};GPULightClusterParams clusterParams{};GPUEmissiveUpdateParams emitterParams{};
    MTL::GPUAddress paramsAddress=0,clusterAddress=0,emitterAddress=0;
    rg::BufferRef sourceRef{},lightsRef{},emittersRef{},aliasRef{},surfaceRef{},freshRef{},temporalRef{},spatialRef{},cellRef{},indexRef{},historyRef{},previousSurfaceRef{},rankRef{},poseRef{};
    rg::BufferRef checkRef{},signalErrorsRef{};pipe::PipelineHandle signalClear{},corruptFinite{};GPULightingCheckParams checkParams{};MTL::GPUAddress checkAddress=0;
    pipe::PipelineHandle checkClear{},checkDi{},corruptAlias{},corruptGeneration{};
    rg::TextureRef visibilityRef{},depthRef{},motionRef{},fallbackShading{},fallbackAlbedo{},output{};
    Impl(MetalContext& context,PipelineCache& pipelines,SceneRenderer& s,MeshRenderer& m,VisibilityRenderer& v,AccelerationStructures& a,ShadowPasses& sh,const LaunchOptions& o)
        :c(context),p(pipelines),scene(s),mesh(m),vis(v),rt(a),shadow(sh),options(o),restirConsumer(c,p,a),clusterConsumer(c,p,a) {
        const char* names[]={"light_emissive_update","material_lighting_guides","di_receiver_pack","light_cluster_build",
                             "restir_di_candidates","restir_di_temporal","restir_di_spatial","restir_di_shade_rt"};
        for(u32 i=0;i<Snapshot;++i)kernels[i]=p.request(lighting::kernel(names[i],i==Shade));
        clusterShade=p.request(lighting::kernel("light_cluster_shade_rt",true));
        pipe::PipelineDesc d;d.kind=pipe::PipelineKind::Render;d.label="Indexed overflow DI guides";d.functions={"forward_surface_vs","forward_di_receiver_fs",""};
        d.indirectCommandBuffers=true;d.output(0,rg::Format::RGBA16Float).output(1,rg::Format::RGBA16Float).output(2,rg::Format::RG32Float);receiver=p.request(d);
        auto* dd=MTL::DepthStencilDescriptor::alloc()->init();dd->setDepthCompareFunction(MTL::CompareFunctionGreaterEqual);dd->setDepthWriteEnabled(false);equalDepth=c.device()->newDepthStencilState(dd);dd->release();
        for(auto& f:slots) {for(auto*& t:f.tables)t=lighting::table(c);if(o.debugLighting)f.check=lighting::buffer(c,32,"Lighting independent state check",true);f.signalErrors=lighting::buffer(c,16,"DI numerical errors before sanitization",true);}
        if(o.debugLighting){checkClear=p.request(lighting::kernel("lighting_check_clear"));checkDi=p.request(lighting::kernel("lighting_check_di"));
            corruptAlias=p.request(lighting::kernel("lighting_corrupt_alias"));corruptGeneration=p.request(lighting::kernel("lighting_corrupt_reservoir_generation"));}
        signalClear=p.request(lighting::kernel("di_clear_signal_errors"));corruptFinite=p.request(lighting::kernel("di_corrupt_finite_radiance"));
        // An original, defined STBN generator runs at loading time only. Its
        // spectrum, period and cost are NOT accepted until tester evidence.
        di::StbnConfig cfg;cfg.seed=o.lightingSeed;stbn=di::generateStbn(cfg);
        ranks=lighting::buffer(c,stbn.ranks.size()*4,"Generated STBN ranks",true);std::memcpy(ranks->contents(),stbn.ranks.data(),stbn.ranks.size()*4);
    }
    pipe::PipelineHandle clusterShade{};
    ~Impl(){c.waitIdle();for(auto& f:slots){for(auto* b:{f.source,f.lights,f.emitters,f.alias,f.surface,f.fresh,f.temporal,f.spatial,f.cells,f.indices,f.check,f.signalErrors})c.memory().release(b,MemoryCategory::RayTracing);for(auto* t:f.tables)if(t)t->release();}for(auto& v:views){c.memory().release(v.reservoir,MemoryCategory::RayTracing);c.memory().release(v.surface,MemoryCategory::RayTracing);}c.memory().release(ranks,MemoryCategory::RayTracing);if(equalDepth)equalDepth->release();}
    void load(const GpuScene& g,const SceneStore& s){lightScene.rebuild(g,s,g.lights());for(u32 v=0;v<4;++v)history.invalidate(v,"scene load");reserve(std::max<size_t>(1,lightScene.lights.size()),pixelCapacity);++graphVersion;}
    void reserve(u64 lights,u64 pixels){
        if(lights>lightCapacity){lightCapacity=std::max<u64>(lights,lightCapacity+lightCapacity/2);for(auto& f:slots){for(auto* b:{f.source,f.lights,f.emitters,f.alias})c.memory().release(b,MemoryCategory::RayTracing);f.source=lighting::buffer(c,lightCapacity*sizeof(GPUSampledLight),"Local source lights",true);f.lights=lighting::buffer(c,lightCapacity*sizeof(GPUSampledLight),"Local world lights");f.emitters=lighting::buffer(c,lightCapacity*sizeof(GPUEmissiveSurface),"Full scene emitter records",true);f.alias=lighting::buffer(c,lightCapacity*sizeof(GPUAliasEntry),"Light proposal aliases",true);}++graphVersion;}
        if(pixels>pixelCapacity){for(u32 v=0;v<4;++v)history.invalidate(v,"DI allocation growth");pixelCapacity=pixels;for(auto& f:slots){for(auto* b:{f.surface,f.fresh,f.temporal,f.spatial})c.memory().release(b,MemoryCategory::RayTracing);f.surface=lighting::buffer(c,pixels*sizeof(GPUDISurface),"DI material surfaces");f.fresh=lighting::buffer(c,pixels*sizeof(GPUDIReservoir),"DI candidates");f.temporal=lighting::buffer(c,pixels*sizeof(GPUDIReservoir),"DI temporal");f.spatial=lighting::buffer(c,pixels*sizeof(GPUDIReservoir),"DI spatial");}for(auto& v:views){c.memory().release(v.reservoir,MemoryCategory::RayTracing);c.memory().release(v.surface,MemoryCategory::RayTracing);v.reservoir=lighting::buffer(c,pixels*sizeof(GPUDIReservoir),"DI reservoir per view");v.surface=lighting::buffer(c,pixels*sizeof(GPUDISurface),"DI surfaces per view");}++graphVersion;}
        const auto preset=(options.reducedLighting||options.forceApple9)?di::reducedPreset:di::fullPreset;
        const u64 cells=u64(preset.clusterX)*preset.clusterY*preset.clusterZ;
        for(auto& f:slots)if(!f.cells){f.cells=lighting::buffer(c,cells*sizeof(GPULightCluster),"3D light cluster cells");f.indices=lighting::buffer(c,cells*preset.clusterCapacity*4,"Bounded cluster indices");}
    }
    void prepare(const GpuScene& g,const SceneStore& s,std::span<const GPULight> l,const ShadowPasses::Frame& f){
        frame=f;lightScene.rebuild(g,s,l);reserve(std::max<size_t>(1,lightScene.lights.size()),u64(f.backingWidth)*f.backingHeight);
        auto& slot=slots[f.slot];const size_t n=lightScene.lights.size();
        if(n){std::memcpy(slot.source->contents(),lightScene.lights.data(),n*sizeof(GPUSampledLight));std::memcpy(slot.emitters->contents(),lightScene.emitters.data(),n*sizeof(GPUEmissiveSurface));std::memcpy(slot.alias->contents(),lightScene.alias.entries.data(),n*sizeof(GPUAliasEntry));}
        const bool hasEmissive=std::any_of(lightScene.emitters.begin(),lightScene.emitters.end(),[](const auto& e){return e.valid!=0;});
        if(hasEmissive) {
            if(!s.motionSlots().empty()&&!f.motionSinCosValid)throw std::logic_error("DI emitter domain requires exact uploaded motion phases");
            referenceWorlds(s,f.motionSinCos.data(),emissiveWorlds);
        } else emissiveWorlds.clear();
        if(emissiveDomain.update(lightScene.emitters,emissiveWorlds))++sceneSignal;
        if(s.stats().structure||s.stats().fullMaterials||!s.materialDeltas().empty())++sceneSignal;
        const Signal signal{f.scene,sceneSignal,rt.geometryRevision(),lightScene.revision};
        if(!(signal==lastSignal)){++signalEpoch;lastSignal=signal;}
        const u64 revision=signalEpoch;
        auto decision=history.begin(f.view,{f.width,f.height,f.backingWidth,f.backingHeight},revision,f.cut,f.reset);
        const auto preset=(options.reducedLighting||options.forceApple9)?di::reducedPreset:di::fullPreset;
        params={};params.width=f.width;params.height=f.height;params.lightCount=n;params.frameIndex=f.index;params.viewID=f.view;params.historyEpoch=history.get(f.view).generation;params.lightRevision=lightScene.revision;
        params.candidateCount=(options.reducedLighting||options.forceApple9)?std::min(options.lightingCandidates,preset.candidates):options.lightingCandidates;
        params.spatialCount=(options.reducedLighting||options.forceApple9)?std::min(options.lightingSpatialSamples,preset.neighbors):options.lightingSpatialSamples;
        params.spatialRadius=preset.radius;params.maxHistoryM=preset.maxHistoryM;params.maxHistoryAge=preset.maxHistoryAge;
        params.flags=DI_ENABLE_TEMPORAL|DI_ENABLE_SPATIAL|DI_ENABLE_VISIBILITY|DI_USE_STBN|(decision.reset?DI_RESET_HISTORY:0u);
        params.stbnWidth=stbn.config.width;params.stbnHeight=stbn.config.height;params.stbnFrames=stbn.config.frames;params.stbnDimensions=stbn.config.dimensions;params.stbnSeed=options.lightingSeed;
        params.depthRelativeThreshold=0.02f;params.normalThreshold=0.95f;params.targetFloor=1e-6f;params.slotCount=s.slotCapacity();params.pad1=options.directLighting==DirectLightingMode::BruteForce;
        checkParams={f.width,f.height,options.directLighting==DirectLightingMode::ReSTIR?1u:0u,u32(n)};checkAddress=lighting::upload(c,checkParams);
        paramsAddress=lighting::upload(c,params);emitterParams={u32(n),s.slotCapacity(),u32(s.materials().size()),u32(g.vertices().size())};emitterAddress=lighting::upload(c,emitterParams);
        clusterParams={};std::memcpy(clusterParams.view,f.constants.view,64);clusterParams.nearPlane=f.nearPlane;clusterParams.farPlane=1000;
        auto projection=glm::make_mat4(f.unjitteredVP)*glm::inverse(glm::make_mat4(f.constants.view));clusterParams.tanHalfFovX=1.0f/projection[0][0];clusterParams.tanHalfFovY=1.0f/projection[1][1];
        clusterParams.gridX=preset.clusterX;clusterParams.gridY=preset.clusterY;clusterParams.gridZ=preset.clusterZ;clusterParams.capacity=preset.clusterCapacity;clusterParams.lightCount=n;clusterAddress=lighting::upload(c,clusterParams);
        restirConsumer.prepare(f.slot,kernels[Shade]);clusterConsumer.prepare(f.slot,clusterShade);
    }
    void bindCommon(MTL4::ArgumentTable* t){auto& f=slots[frame.slot];auto& v=views[frame.view];t->setAddress(paramsAddress,0);t->setAddress(f.surface->gpuAddress(),1);t->setAddress(f.lights->gpuAddress(),2);t->setAddress(f.alias->gpuAddress(),3);t->setAddress(f.fresh->gpuAddress(),4);t->setAddress(v.reservoir->gpuAddress(),5);t->setAddress(v.surface->gpuAddress(),6);t->setAddress(ranks->gpuAddress(),8);t->setAddress(f.emitters->gpuAddress(),12);t->setAddress(scene.buffers().materials()->gpuAddress(),13);t->setAddress(scene.textureTableAddress(),14);t->setAddress(f.signalErrors->gpuAddress(),15);}
    void tex(MTL4::ArgumentTable* t,rg::PassContext& ctx,rg::TextureRef r,u32 b){t->setTexture(static_cast<MTL::Texture*>(ctx.texture(r))->gpuResourceID(),b);}
    void add(rg::RenderGraph& g,rg::TextureRef visibility,rg::TextureRef depth){using namespace rg;visibilityRef=visibility;depthRef=depth;auto& f=slots[frame.slot];auto& view=views[frame.view];
        sourceRef=g.importBuffer("Local source light records",{f.source->length()},ImportPerFrame|ImportContentsDefined);
        lightsRef=g.importBuffer("Local world lights",{f.lights->length()},ImportPerFrame);
        emittersRef=g.importBuffer("Full emitter geometry UV records",{f.emitters->length()},ImportPerFrame|ImportContentsDefined);
        aliasRef=g.importBuffer("Alias light proposal",{f.alias->length()},ImportPerFrame|ImportContentsDefined);
        rankRef=g.importBuffer("STBN ranks",{ranks->length()},ImportContentsDefined);
        surfaceRef=g.importBuffer("DI surfaces",{f.surface->length()},ImportPerFrame);
        freshRef=g.importBuffer("DI candidates",{f.fresh->length()},ImportPerFrame);temporalRef=g.importBuffer("DI temporal",{f.temporal->length()},ImportPerFrame);spatialRef=g.importBuffer("DI spatial",{f.spatial->length()},ImportPerFrame);
        cellRef=g.importBuffer("Local light clusters",{f.cells->length()},ImportPerFrame);indexRef=g.importBuffer("Local cluster indices",{f.indices->length()},ImportPerFrame);
        historyRef=g.importBuffer("DI history reservoirs per view",{view.reservoir->length()},ImportContentsDefined|ImportOutput);previousSurfaceRef=g.importBuffer("DI history surfaces per view",{view.surface->length()},ImportContentsDefined|ImportOutput);poseRef=vis.importPoseHistory(g);
        g.addPass("World emissive lights",PassType::Compute,[&](PassBuilder& b){b.read(sourceRef,Usage::ShaderRead,StageDispatch);b.read(emittersRef,Usage::ShaderRead,StageDispatch);emittersRef=b.write(emittersRef,Usage::ShaderWrite,StageDispatch);b.read(scene.dataRef(),Usage::ShaderRead,StageDispatch);b.read(rt.geometryBufferRef(),Usage::ShaderRead,StageDispatch);lightsRef=b.write(lightsRef,Usage::ShaderWrite,StageDispatch);b.setProfileShaders("light_emissive_update");},[this](PassContext& ctx){auto& f=slots[frame.slot];auto* t=f.tables[Emit];t->setAddress(emitterAddress,0);t->setAddress(f.emitters->gpuAddress(),1);t->setAddress(scene.buffers().instances()->gpuAddress(),2);t->setAddress(scene.buffers().materials()->gpuAddress(),3);t->setAddress(f.source->gpuAddress(),4);t->setAddress(f.lights->gpuAddress(),5);t->setAddress(scene.vertexBuffer()->gpuAddress(),6);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),p,kernels[Emit],t,std::max(1u,params.lightCount));});
        g.addPass("Pre-resolve material lighting guides",PassType::Compute,[&](PassBuilder& b){b.read(visibilityRef,Usage::ShaderRead,StageDispatch);b.read(shadow.surfaces(),Usage::ShaderRead,StageDispatch);b.read(scene.dataRef(),Usage::ShaderRead,StageDispatch);b.read(mesh.frameListsRef(),Usage::ShaderRead,StageDispatch);b.read(poseRef,Usage::ShaderRead,StageDispatch);surfaceRef=b.write(surfaceRef,Usage::ShaderWrite,StageDispatch);motionRef=b.createTexture("Pre-resolve pixel motion",{Format::RG32Float,frame.width,frame.height});motionRef=b.write(motionRef,Usage::ShaderWrite,StageDispatch);fallbackShading=b.createTexture("Indexed DI shading normal roughness",{Format::RGBA16Float,frame.width,frame.height});fallbackShading=b.write(fallbackShading,Usage::ShaderWrite,StageDispatch);fallbackAlbedo=b.createTexture("Indexed DI albedo metallic",{Format::RGBA16Float,frame.width,frame.height});fallbackAlbedo=b.write(fallbackAlbedo,Usage::ShaderWrite,StageDispatch);b.setProfileShaders("material_lighting_guides");},[this](PassContext& ctx){auto* t=slots[frame.slot].tables[Guide];t->setAddress(scene.frameConstantsAddress(),0);t->setAddress(scene.vertexBuffer()->gpuAddress(),1);t->setAddress(scene.buffers().instances()->gpuAddress(),2);t->setAddress(scene.buffers().materials()->gpuAddress(),3);t->setAddress(scene.lightsAddress(),4);t->setAddress(scene.textureTableAddress(),5);t->setAddress(mesh.meshletBuffer()->gpuAddress(),6);t->setAddress(mesh.meshletVertexBuffer()->gpuAddress(),7);t->setAddress(mesh.meshletTriangleBuffer()->gpuAddress(),8);t->setAddress(mesh.frame(frame.slot).candidates->gpuAddress(),9);t->setAddress(mesh.frame(frame.slot).bList->gpuAddress(),10);t->setAddress(vis.paramsAddress(),11);t->setAddress(vis.previousPoseAddress(),14);t->setAddress(vis.temporalAddress(),15);t->setAddress(slots[frame.slot].surface->gpuAddress(),16);t->setAddress(static_cast<MTL::Buffer*>(ctx.buffer(shadow.surfaces()))->gpuAddress(),17);t->setAddress(paramsAddress,18);tex(t,ctx,visibilityRef,0);tex(t,ctx,motionRef,7);tex(t,ctx,fallbackShading,8);tex(t,ctx,fallbackAlbedo,9);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),p,kernels[Guide],t,frame.width,frame.height);});
        g.addPass("Indexed overflow material receivers",PassType::Raster,[&](PassBuilder& b){fallbackShading=b.writeColor(fallbackShading,0,LoadIntent::Preserve);fallbackAlbedo=b.writeColor(fallbackAlbedo,1,LoadIntent::Preserve);motionRef=b.writeColor(motionRef,2,LoadIntent::Preserve);b.readDepth(depthRef);scene.declareDrawReads(b);b.read(poseRef,Usage::ShaderRead,StageVertex);b.setProfileShaders("forward_surface_vs,forward_di_receiver_fs");},[this](PassContext& ctx){scene.encodeOverlay(static_cast<MTL4::RenderCommandEncoder*>(ctx.encoder()),receiver,true,equalDepth);});
        g.addPass("Indexed DI receiver pack",PassType::Compute,[&](PassBuilder& b){for(auto r:{shadow.worldPosition(),shadow.geometricNormal(),shadow.receiverKeys(),fallbackShading,fallbackAlbedo})b.read(r,Usage::ShaderRead,StageDispatch);b.read(surfaceRef,Usage::ShaderRead,StageDispatch);surfaceRef=b.write(surfaceRef,Usage::ShaderWrite,StageDispatch);b.setProfileShaders("di_receiver_pack");},[this](PassContext& ctx){auto* t=slots[frame.slot].tables[Pack];t->setAddress(scene.frameConstantsAddress(),0);t->setAddress(paramsAddress,1);t->setAddress(slots[frame.slot].surface->gpuAddress(),2);tex(t,ctx,shadow.worldPosition(),5);tex(t,ctx,shadow.geometricNormal(),6);tex(t,ctx,shadow.receiverKeys(),10);tex(t,ctx,fallbackShading,11);tex(t,ctx,fallbackAlbedo,12);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),p,kernels[Pack],t,frame.width,frame.height);});
        signalErrorsRef=g.importBuffer("DI numerical error words",{16},ImportPerFrame|ImportOutput);
        g.addPass("DI numerical errors clear",PassType::Compute,[&](PassBuilder& b){signalErrorsRef=b.write(signalErrorsRef,Usage::ShaderWrite,StageDispatch);},[this](PassContext& ctx){auto* t=slots[frame.slot].tables[SignalClear];t->setAddress(slots[frame.slot].signalErrors->gpuAddress(),15);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),p,signalClear,t,4);});
        if(options.debugLightingCorrupt==7)g.addPass("Negative: finite radiance overflow",PassType::Compute,[&](PassBuilder& b){b.read(surfaceRef,Usage::ShaderRead,StageDispatch);b.read(lightsRef,Usage::ShaderRead,StageDispatch);lightsRef=b.write(lightsRef,Usage::ShaderWrite,StageDispatch);},[this](PassContext& ctx){auto* t=slots[frame.slot].tables[CheckNegative];t->setAddress(paramsAddress,0);t->setAddress(slots[frame.slot].surface->gpuAddress(),1);t->setAddress(slots[frame.slot].lights->gpuAddress(),2);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),p,corruptFinite,t,1);});
        auto reads=[&](PassBuilder& b){for(auto r:{surfaceRef,lightsRef,emittersRef,aliasRef,rankRef})b.read(r,Usage::ShaderRead,StageDispatch);b.read(scene.dataRef(),Usage::ShaderRead,StageDispatch);};
        if(options.debugLightingCorrupt==5 && options.directLighting==DirectLightingMode::ReSTIR)
            g.addPass("Negative: alias PDF",PassType::Compute,[&](PassBuilder& b){b.read(aliasRef,Usage::ShaderRead,StageDispatch);aliasRef=b.write(aliasRef,Usage::ShaderWrite,StageDispatch);b.setProfileShaders("lighting_corrupt_alias");},[this](PassContext& ctx){auto* t=slots[frame.slot].tables[CheckNegative];t->setAddress(checkAddress,0);t->setAddress(slots[frame.slot].alias->gpuAddress(),1);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),p,corruptAlias,t,std::max(1u,params.lightCount));});
        if(options.directLighting==DirectLightingMode::ReSTIR){
            g.addPass("DI candidates",PassType::Compute,[&](PassBuilder& b){reads(b);freshRef=b.write(freshRef,Usage::ShaderWrite,StageDispatch);b.setProfileShaders("restir_di_candidates");},[this](PassContext& ctx){auto* t=slots[frame.slot].tables[Candidate];bindCommon(t);t->setAddress(slots[frame.slot].fresh->gpuAddress(),7);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),p,kernels[Candidate],t,params.width*params.height);});
            g.addPass("DI temporal",PassType::Compute,[&](PassBuilder& b){reads(b);b.read(freshRef,Usage::ShaderRead,StageDispatch);b.read(historyRef,Usage::ShaderRead,StageDispatch);b.read(previousSurfaceRef,Usage::ShaderRead,StageDispatch);b.read(motionRef,Usage::ShaderRead,StageDispatch);temporalRef=b.write(temporalRef,Usage::ShaderWrite,StageDispatch);b.setProfileShaders("restir_di_temporal");},[this](PassContext& ctx){auto* t=slots[frame.slot].tables[Temporal];bindCommon(t);t->setAddress(slots[frame.slot].temporal->gpuAddress(),7);tex(t,ctx,motionRef,0);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),p,kernels[Temporal],t,params.width*params.height);});
            g.addPass("DI spatial",PassType::Compute,[&](PassBuilder& b){reads(b);b.read(temporalRef,Usage::ShaderRead,StageDispatch);spatialRef=b.write(spatialRef,Usage::ShaderWrite,StageDispatch);b.setProfileShaders("restir_di_spatial");},[this](PassContext& ctx){auto* t=slots[frame.slot].tables[Spatial];bindCommon(t);t->setAddress(slots[frame.slot].temporal->gpuAddress(),4);t->setAddress(slots[frame.slot].spatial->gpuAddress(),7);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),p,kernels[Spatial],t,params.width*params.height);});
        } else {
            g.addPass("3D light clusters",PassType::Compute,[&](PassBuilder& b){b.read(lightsRef,Usage::ShaderRead,StageDispatch);cellRef=b.write(cellRef,Usage::ShaderWrite,StageDispatch);indexRef=b.write(indexRef,Usage::ShaderWrite,StageDispatch);b.setProfileShaders("light_cluster_build");},[this](PassContext& ctx){auto& f=slots[frame.slot];auto* t=f.tables[Cluster];t->setAddress(clusterAddress,0);t->setAddress(f.lights->gpuAddress(),1);t->setAddress(f.cells->gpuAddress(),2);t->setAddress(f.indices->gpuAddress(),3);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),p,kernels[Cluster],t,clusterParams.gridX*clusterParams.gridY*clusterParams.gridZ);});
        }
        const bool ris=options.directLighting==DirectLightingMode::ReSTIR;
        g.addPass(ris?"Local ReSTIR RT shading":options.directLighting==DirectLightingMode::BruteForce?"Local brute-force RT shading":"Local clustered RT shading",PassType::Compute,[&](PassBuilder& b){reads(b);b.read(signalErrorsRef,Usage::ShaderRead,StageDispatch);signalErrorsRef=b.write(signalErrorsRef,Usage::ShaderWrite,StageDispatch);rt.declareTraceReads(b);if(ris)b.read(spatialRef,Usage::ShaderRead,StageDispatch);else{b.read(cellRef,Usage::ShaderRead,StageDispatch);b.read(indexRef,Usage::ShaderRead,StageDispatch);}output=b.createTexture("Local direct radiance",{Format::RGBA32Float,frame.width,frame.height});output=b.write(output,Usage::ShaderWrite,StageDispatch);b.setProfileShaders(ris?"restir_di_shade_rt":"light_cluster_shade_rt");},[this,ris](PassContext& ctx){auto& f=slots[frame.slot];auto* t=f.tables[Shade];bindCommon(t);if(ris){t->setAddress(f.spatial->gpuAddress(),3);t->setAddress(scene.buffers().instances()->gpuAddress(),4);restirConsumer.bind(t,5,6);}else{t->setAddress(clusterAddress,3);t->setAddress(f.cells->gpuAddress(),4);t->setAddress(f.indices->gpuAddress(),5);t->setAddress(ranks->gpuAddress(),6);t->setAddress(scene.buffers().instances()->gpuAddress(),7);clusterConsumer.bind(t,8,9);}tex(t,ctx,output,0);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),p,ris?kernels[Shade]:clusterShade,t,params.width*params.height);});
        if(options.debugLighting){
            checkRef=g.importBuffer("Lighting state check counters",{32},ImportPerFrame|ImportOutput);
            g.addPass("Lighting state check clear",PassType::Compute,[&](PassBuilder& b){checkRef=b.write(checkRef,Usage::ShaderWrite,StageDispatch);},[this](PassContext& ctx){auto* t=slots[frame.slot].tables[CheckClear];t->setAddress(slots[frame.slot].check->gpuAddress(),0);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),p,checkClear,t,8);});
            if(options.debugLightingCorrupt==6 && ris)g.addPass("Negative: light generation",PassType::Compute,[&](PassBuilder& b){b.read(spatialRef,Usage::ShaderRead,StageDispatch);spatialRef=b.write(spatialRef,Usage::ShaderWrite,StageDispatch);},[this](PassContext& ctx){auto* t=slots[frame.slot].tables[CheckNegative];t->setAddress(checkAddress,0);t->setAddress(slots[frame.slot].spatial->gpuAddress(),1);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),p,corruptGeneration,t,params.width*params.height);});
            g.addPass("Lighting independent state check",PassType::Compute,[&](PassBuilder& b){reads(b);if(ris)b.read(spatialRef,Usage::ShaderRead,StageDispatch);b.read(output,Usage::ShaderRead,StageDispatch);b.read(checkRef,Usage::ShaderRead,StageDispatch);checkRef=b.write(checkRef,Usage::ShaderWrite,StageDispatch);},[this](PassContext& ctx){auto& f=slots[frame.slot];auto* t=f.tables[Check];t->setAddress(checkAddress,0);t->setAddress(f.surface->gpuAddress(),1);t->setAddress(f.spatial->gpuAddress(),2);t->setAddress(f.lights->gpuAddress(),3);t->setAddress(f.check->gpuAddress(),4);tex(t,ctx,output,0);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),p,checkDi,t,params.width*params.height);});
        }
        if(ris)g.addPass("DI history snapshot",PassType::Blit,[&](PassBuilder& b){b.read(temporalRef,Usage::CopySrc,StageBlit);b.read(surfaceRef,Usage::CopySrc,StageBlit);historyRef=b.write(historyRef,Usage::CopyDst,StageBlit);previousSurfaceRef=b.write(previousSurfaceRef,Usage::CopyDst,StageBlit);},[this](PassContext& ctx){auto* e=static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder());const u64 count=u64(frame.width)*frame.height;auto& f=slots[frame.slot];auto& v=views[frame.view];e->copyFromBuffer(f.temporal,0,v.reservoir,0,count*sizeof(GPUDIReservoir));e->copyFromBuffer(f.surface,0,v.surface,0,count*sizeof(GPUDISurface));history.read(frame.view,frame.index+1);history.write(frame.view,frame.index+1,frame.constants.viewProjection);});
    }
    void bind(MetalGraphExecutor& e){auto& f=slots[frame.slot];auto& v=views[frame.view];e.bindBuffer(sourceRef,f.source);e.bindBuffer(lightsRef,f.lights);e.bindBuffer(emittersRef,f.emitters);e.bindBuffer(aliasRef,f.alias);e.bindBuffer(surfaceRef,f.surface);e.bindBuffer(freshRef,f.fresh);e.bindBuffer(temporalRef,f.temporal);e.bindBuffer(spatialRef,f.spatial);e.bindBuffer(cellRef,f.cells);e.bindBuffer(indexRef,f.indices);e.bindBuffer(historyRef,v.reservoir);e.bindBuffer(previousSurfaceRef,v.surface);e.bindBuffer(rankRef,ranks);if(options.debugLighting)e.bindBuffer(checkRef,f.check);e.bindBuffer(signalErrorsRef,f.signalErrors);}
};
DirectLightingPasses::DirectLightingPasses(MetalContext& c,PipelineCache& p,SceneRenderer& s,MeshRenderer& m,VisibilityRenderer& v,AccelerationStructures& a,ShadowPasses& sh,const LaunchOptions& o):impl_(std::make_unique<Impl>(c,p,s,m,v,a,sh,o)){}
DirectLightingPasses::~DirectLightingPasses()=default;
void DirectLightingPasses::loadScene(const GpuScene& g,const SceneStore& s){impl_->load(g,s);}
void DirectLightingPasses::prepareFrame(const GpuScene& g,const SceneStore& s,std::span<const GPULight> l,const ShadowPasses::Frame& f){impl_->prepare(g,s,l,f);}
void DirectLightingPasses::addToGraph(rg::RenderGraph& g,rg::TextureRef v,rg::TextureRef d){impl_->add(g,v,d);}
void DirectLightingPasses::bindFrame(MetalGraphExecutor& e){impl_->bind(e);}
rg::TextureRef DirectLightingPasses::direct()const{return impl_->output;}rg::TextureRef DirectLightingPasses::motion()const{return impl_->motionRef;}
rg::BufferRef DirectLightingPasses::surfaceRef()const{return impl_->surfaceRef;}rg::BufferRef DirectLightingPasses::lightsRef()const{return impl_->lightsRef;}rg::BufferRef DirectLightingPasses::emittersRef()const{return impl_->emittersRef;}
MTL::Buffer* DirectLightingPasses::lightsBuffer()const{return impl_->slots[impl_->frame.slot].lights;}MTL::Buffer* DirectLightingPasses::emittersBuffer()const{return impl_->slots[impl_->frame.slot].emitters;}
u32 DirectLightingPasses::lightCount()const{return impl_->params.lightCount;}u32 DirectLightingPasses::lightRevision()const{return impl_->lightScene.radianceRevision;}u64 DirectLightingPasses::version()const{return impl_->graphVersion;}
} // namespace phosphor

namespace phosphor {
bool DirectLightingPasses::check(u32 slot)const {
    if(!impl_->options.debugLighting)return true;const auto* c=static_cast<const u32*>(impl_->slots.at(slot).check->contents());
    const auto* errors=static_cast<const u32*>(impl_->slots.at(slot).signalErrors->contents());for(u32 i=0;i<4;++i)if(errors[i])return false;
    if(!c[0])return false;for(u32 i=3;i<8;++i)if(c[i])return false;return true;
}
}
