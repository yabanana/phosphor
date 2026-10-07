#include "platform/metal/shadow_passes.h"
#include "platform/metal/lighting_dispatch.h"
#include "platform/metal/acceleration_structures.h"
#include "platform/metal/scene_renderer.h"
#include "platform/metal/mesh_renderer.h"
#include "platform/metal/visibility_renderer.h"
#include "platform/metal/metal_graph_executor.h"
#include "renderer/scene_store.h"
#include "renderer/gpu_scene.h"
#include "renderer/shadow_layout.h"
#include "rendergraph/pass_context.h"
#include <glm/gtc/type_ptr.hpp>
#include <bit>
#include <vector>
namespace phosphor {
struct ShadowPasses::Impl {
    enum Pass { Clear, Guide, Pack, Cull, Pcss, Sun, Temporal, Filter, Contact, PassCount };
    MetalContext& c; PipelineCache& pipelines; SceneRenderer& scene; MeshRenderer& mesh; VisibilityRenderer& visibility;
    AccelerationStructures* rt;
    LaunchOptions options; ShadowSettings settings;
    std::unique_ptr<RtConsumer> sunConsumer;
    Frame frame{}; GPUShadowParams params{}; MTL::GPUAddress paramsAddress=0;
    std::array<pipe::PipelineHandle,PassCount> kernels{};
    pipe::PipelineHandle indexed, meshDepth, receiver;
    MTL::DepthStencilState* depthState=nullptr;
    MTL::DepthStencilState* receiverDepthState=nullptr;
    std::vector<GPUMeshInfo> geometry;
    struct Slot {
        MTL::Buffer *surfaces=nullptr,*flags=nullptr,*counters=nullptr;
        std::array<MTL4::ArgumentTable*,PassCount> tables{};
        MTL4::ArgumentTable* receiverTable=nullptr;
        std::vector<MTL4::ArgumentTable*> rasterTables;
        std::vector<MTL::GPUAddress> drawParams;
    };
    std::array<Slot,METAL_FRAMES_IN_FLIGHT> slots{};
    struct View {
        MTL::Buffer *previous=nullptr,*next=nullptr;
        std::array<MTL::Texture*,4> cached{};
        std::array<u64,4> revisions{};
        u32 width=0,height=0;
    };
    std::array<View,HistoryRegistry::MaxViews> views{};
    HistoryRegistry history;
    u64 capacity=0, casterCapacity=0, graphVersion=1, casterRevision=1, materialRevision=1, lightRevision=1;
    u64 previousLightHash=0;
    u32 selectedSun=~0u;
    rg::BufferRef surfaceRef{},flagsRef{},counterRef{},previousRef{},nextRef{},poseRef{};
    rg::TextureRef visibilityRef{},depthRef{},pointRef{},normalRef{},keyRef{},maskRef{},zeroRef{};
    rg::TextureRef rawMask{},temporalOutput{},filterOutput{},contactOutput{};
    std::array<rg::TextureRef,4> maps{};
    Impl(MetalContext& context,PipelineCache& p,SceneRenderer& s,MeshRenderer& m,VisibilityRenderer& v,
         AccelerationStructures* a,const LaunchOptions& o):c(context),pipelines(p),scene(s),mesh(m),visibility(v),rt(a),options(o) {
        settings.mapResolution=o.shadowMapResolution; settings.contact=o.contactShadows; settings.staticCache=o.shadowCache;
        settings.mode=o.shadows==ShadowMode::RT?ShadowTechnique::RayTraced:
                      o.shadows==ShadowMode::CSM?ShadowTechnique::Cascaded:ShadowTechnique::Off;
        const char* names[]={"shadow_receiver_clear","shadow_surface_guides","shadow_receiver_pack","shadow_caster_flags",
                             "shadow_csm_pcss","shadow_sun_rt","shadow_temporal","shadow_filter","shadow_contact"};
        for (u32 i=0;i<PassCount;++i) {
            if (i==Sun && o.shadows!=ShadowMode::RT) continue;
            kernels[i]=p.request(lighting::kernel(names[i],i==Sun));
        }
        pipe::PipelineDesc d; d.kind=pipe::PipelineKind::Render;d.label="CSM indexed casters";
        d.functions={"shadow_depth_vertex","shadow_depth_fragment",""}; indexed=p.request(d);
        d.kind=pipe::PipelineKind::Mesh;d.label="CSM mesh casters";
        d.functions={"","shadow_depth_mesh","shadow_depth_fragment"};d.mesh.meshThreads=128; meshDepth=p.request(d);
        d=pipe::PipelineDesc{};d.kind=pipe::PipelineKind::Render;d.label="Indexed overflow shadow receivers";
        d.functions={"forward_surface_vs","forward_shadow_receiver_fs",""};d.indirectCommandBuffers=true;
        d.output(0,rg::Format::RGBA32Float).output(1,rg::Format::RGBA16Float).output(2,rg::Format::RGBA32Uint);
        receiver=p.request(d);
        auto* dd=MTL::DepthStencilDescriptor::alloc()->init();dd->setDepthCompareFunction(MTL::CompareFunctionGreater);
        dd->setDepthWriteEnabled(true);depthState=c.device()->newDepthStencilState(dd);dd->release();
        dd=MTL::DepthStencilDescriptor::alloc()->init();dd->setDepthCompareFunction(MTL::CompareFunctionGreaterEqual);dd->setDepthWriteEnabled(true);receiverDepthState=c.device()->newDepthStencilState(dd);dd->release();
        if(!depthState || !receiverDepthState) throw std::runtime_error("CSM depth state allocation failed");
        for(auto& f:slots) { for(auto*& t:f.tables)t=lighting::table(c); f.receiverTable=lighting::table(c);f.counters=lighting::buffer(c,sizeof(GPUShadowCounters),"Shadow counters",true); }
        if(o.shadows==ShadowMode::RT) { if(!rt)throw std::logic_error("Sun RT without AS");sunConsumer=std::make_unique<RtConsumer>(c,p,*rt); }
    }
    ~Impl() {
        c.waitIdle();
        for(auto& f:slots) {
            for(auto* b:{f.surfaces,f.flags,f.counters})c.memory().release(b,MemoryCategory::RayTracing);
            for(auto* t:f.tables)if(t)t->release();if(f.receiverTable)f.receiverTable->release();
            for(auto* t:f.rasterTables)if(t)t->release();
        }
        for(auto& v:views) {c.memory().release(v.previous,MemoryCategory::RayTracing);c.memory().release(v.next,MemoryCategory::RayTracing);for(auto* t:v.cached)c.memory().release(t,MemoryCategory::RayTracing);}
        if(depthState)depthState->release();if(receiverDepthState)receiverDepthState->release();
    }
    void load(const GpuScene& g,const SceneStore& store) {
        c.waitIdle();geometry.assign(g.meshInfos().begin(),g.meshInfos().end());
        for(auto& f:slots) {
            for(auto* t:f.rasterTables)if(t)t->release(); f.rasterTables.clear();
            f.rasterTables.resize(4*(geometry.size()+1));f.drawParams.resize(f.rasterTables.size());
            for(auto*& t:f.rasterTables)t=lighting::table(c);
        }
        ++casterRevision;++materialRevision;++graphVersion;
        for(u32 v=0;v<views.size();++v)history.invalidate(v,"scene load");
        reserve(store.slotCapacity(),frame.backingWidth,frame.backingHeight);
    }
    void reserve(u32 slotsCount,u32 w,u32 h) {
        const u64 pixels=u64(w)*h;
        if(pixels>capacity) {
            capacity=pixels;
            for(auto& f:slots) {c.memory().release(f.surfaces,MemoryCategory::RayTracing);f.surfaces=lighting::buffer(c,capacity*sizeof(GPUShadowSurface),"Shadow geometric surface");}
            for(auto& v:views) {
                c.memory().release(v.previous,MemoryCategory::RayTracing);c.memory().release(v.next,MemoryCategory::RayTracing);
                v.previous=lighting::buffer(c,capacity*sizeof(GPUShadowHistory),"Solar visibility history read");
                v.next=lighting::buffer(c,capacity*sizeof(GPUShadowHistory),"Solar visibility history write");
            }
            ++graphVersion;
        }
        if(slotsCount>casterCapacity) {
            casterCapacity=slotsCount;
            for(auto& f:slots){c.memory().release(f.flags,MemoryCategory::RayTracing);f.flags=lighting::buffer(c,std::max<u64>(1,casterCapacity)*4,"CSM full-scene caster flags");}
            ++graphVersion;
        }
    }
    void prepare(const SceneStore& store,std::span<const GPULight> lights,const Frame& f) {
        frame=f;reserve(store.slotCapacity(),f.backingWidth,f.backingHeight);
        if(store.stats().structure || store.stats().fullInstances || !store.instanceDeltas().empty() || !store.motionSlots().empty() || !store.dirtyRoots().empty())++casterRevision;
        if(store.stats().fullMaterials || !store.materialDeltas().empty())++materialRevision;
        selectedSun=~0u;u64 hash=1469598103934665603ull;
        for(u32 i=0;i<lights.size();++i)if(lights[i].type==LIGHT_DIRECTIONAL){selectedSun=i;break;}
        if(selectedSun!=~0u)for(u32 b:std::bit_cast<std::array<u32,sizeof(GPULight)/4>>(lights[selectedSun])){hash^=b;hash*=1099511628211ull;}
        if(hash!=previousLightHash){++lightRevision;previousLightHash=hash;}
        const u64 signalRevision=(f.scene*1099511628211ull)^casterRevision^(materialRevision<<21)^(lightRevision<<42);
        auto decision=history.begin(f.view,{f.width,f.height,f.backingWidth,f.backingHeight},signalRevision,f.cut,f.reset);
        params={};
        const auto inverse=glm::inverse(glm::make_mat4(f.constants.viewProjection));
        std::memcpy(params.inverseViewProjection,glm::value_ptr(inverse),64);std::memcpy(params.viewProjection,f.constants.viewProjection,64);
        std::memcpy(params.previousViewProjection,history.get(f.view).previousViewProjection.data(),64);
        std::memcpy(params.view,f.constants.view,64);std::memcpy(params.cameraPosition,f.constants.cameraPosition,16);
        glm::vec3 toward(0,1,0);if(selectedSun!=~0u)toward=-glm::normalize(glm::make_vec3(lights[selectedSun].direction));
        std::memcpy(params.lightDirection,glm::value_ptr(toward),12);params.lightDirection[3]=settings.sunAngularRadius;
        ShadowCamera camera;camera.inverseViewProjection=glm::inverse(glm::make_mat4(f.unjitteredVP));camera.position=glm::make_vec3(f.constants.cameraPosition);
        camera.forward=-glm::vec3(glm::inverse(glm::make_mat4(f.constants.view))[2]);camera.nearPlane=f.nearPlane;
        const auto cascades=makeShadowCascades(camera,toward,settings);std::copy(cascades.begin(),cascades.end(),params.cascades);
        params.width=f.width;params.height=f.height;params.slotCount=store.slotCapacity();params.meshCount=geometry.size();
        params.candidateCapacity=mesh.capacity();params.materialCount=store.materials().size();params.frameIndex=f.index;
        params.flags=(options.shadows==ShadowMode::CSM?SHADOW_FLAG_CSM:options.shadows==ShadowMode::RT?SHADOW_FLAG_RT:0u) |
                     (!decision.reset?SHADOW_FLAG_HISTORY_VALID:0u) | (settings.contact?SHADOW_FLAG_CONTACT:0u) |
                     (selectedSun!=~0u?SHADOW_FLAG_LIGHT_VALID:0u);
        params.mapResolution=settings.mapResolution;params.pcssSearchSamples=settings.blockerSamples;params.pcssFilterSamples=settings.filterSamples;
        params.historyMaxSamples=settings.historySamples;params.viewID=f.view;params.lightID=selectedSun;params.lightRevision=lightRevision;params.sceneRevision=signalRevision;
        params.temporalNormalThreshold=settings.temporalNormalThreshold;params.temporalPositionThreshold=settings.temporalPositionThreshold;
        params.contactDistance=settings.contactDistance;params.contactThickness=settings.contactThickness;params.contactStrength=settings.contactStrength;params.contactSteps=settings.contactSteps;
        params.depthBiasWorld=settings.depthBiasWorld;params.normalBiasWorld=settings.normalBiasWorld;params.maxTraceDistance=1e6f;
        params.corruption=options.debugLightingCorrupt<=4?options.debugLightingCorrupt:0;paramsAddress=lighting::upload(c,params);
        auto& slot=slots[f.slot];
        for(u32 cascade=0;cascade<4;++cascade) {
            auto draw=params;draw.cascadeIndex=cascade;draw.casterSlot=~0u;draw.meshletFirst=0;draw.meshletCount=mesh.meshletCount();
            slot.drawParams[cascade*(geometry.size()+1)]=lighting::upload(c,draw);
            for(u32 m=0;m<geometry.size();++m){draw.pad=m;slot.drawParams[cascade*(geometry.size()+1)+m+1]=lighting::upload(c,draw);}
        }
        if(sunConsumer)sunConsumer->prepare(f.slot,kernels[Sun]);
    }
    void common(MTL4::ArgumentTable* t,rg::PassContext& ctx) {
        auto& f=slots[frame.slot];t->setAddress(paramsAddress,SB_PARAMS);t->setAddress(f.surfaces->gpuAddress(),SB_SURFACE);
        t->setAddress(scene.buffers().instances()->gpuAddress(),SB_INSTANCES);t->setAddress(scene.meshBuffer()->gpuAddress(),SB_MESHES);
        t->setAddress(scene.vertexBuffer()->gpuAddress(),SB_VERTICES);t->setAddress(mesh.meshletBuffer()->gpuAddress(),SB_MESHLETS);
        t->setAddress(mesh.meshletVertexBuffer()->gpuAddress(),SB_MESHLET_VERTICES);t->setAddress(mesh.meshletTriangleBuffer()->gpuAddress(),SB_MESHLET_TRIANGLES);
        t->setAddress(mesh.frame(frame.slot).candidates->gpuAddress(),SB_CANDIDATES_A);t->setAddress(mesh.frame(frame.slot).bList->gpuAddress(),SB_CANDIDATES_B);
        t->setAddress(f.flags->gpuAddress(),SB_CASTER_FLAGS);t->setAddress(f.counters->gpuAddress(),SB_COUNTERS);
        t->setAddress(scene.buffers().materials()->gpuAddress(),SB_MATERIALS);t->setAddress(scene.textureTableAddress(),SB_TEXTURES);
    }
    void texture(MTL4::ArgumentTable* t,rg::PassContext& ctx,rg::TextureRef ref,u32 binding) {
        t->setTexture(static_cast<MTL::Texture*>(ctx.texture(ref))->gpuResourceID(),binding);
    }
    void add(rg::RenderGraph& g,rg::TextureRef v,rg::TextureRef d) {
        using namespace rg;visibilityRef=v;depthRef=d;auto& f=slots[frame.slot];poseRef=visibility.importPoseHistory(g);
        surfaceRef=g.importBuffer("Pre-resolve shadow surface",{capacity*sizeof(GPUShadowSurface)},ImportPerFrame);
        flagsRef=g.importBuffer("Independent CSM caster flags",{std::max<u64>(1,casterCapacity)*4},ImportPerFrame);
        counterRef=g.importBuffer("Shadow counters",{sizeof(GPUShadowCounters)},ImportPerFrame|ImportOutput);
        g.addPass("Shadow receiver clear",PassType::Compute,[&](PassBuilder& b){
            counterRef=b.write(counterRef,Usage::ShaderWrite,StageDispatch);keyRef=b.createTexture("Indexed receiver keys",{Format::RGBA32Uint,frame.width,frame.height});
            keyRef=b.write(keyRef,Usage::ShaderWrite,StageDispatch);b.setProfileShaders("shadow_receiver_clear");
        },[this](PassContext& ctx){auto* t=slots[frame.slot].tables[Clear];common(t,ctx);texture(t,ctx,keyRef,10);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),pipelines,kernels[Clear],t,frame.width,frame.height);});
        g.addPass("Pre-resolve geometric guides",PassType::Compute,[&](PassBuilder& b){
            b.read(visibilityRef,Usage::ShaderRead,StageDispatch);b.read(depthRef,Usage::ShaderRead,StageDispatch);b.read(scene.dataRef(),Usage::ShaderRead,StageDispatch);b.read(mesh.frameListsRef(),Usage::ShaderRead,StageDispatch);
            b.read(counterRef,Usage::ShaderRead,StageDispatch);counterRef=b.write(counterRef,Usage::ShaderWrite,StageDispatch);
            surfaceRef=b.write(surfaceRef,Usage::ShaderWrite,StageDispatch);
            pointRef=b.createTexture("Receiver world position",{Format::RGBA32Float,frame.width,frame.height});pointRef=b.write(pointRef,Usage::ShaderWrite,StageDispatch);
            normalRef=b.createTexture("Receiver geometric normal",{Format::RGBA16Float,frame.width,frame.height});normalRef=b.write(normalRef,Usage::ShaderWrite,StageDispatch);
            b.setProfileShaders("shadow_surface_guides");
        },[this](PassContext& ctx){auto* t=slots[frame.slot].tables[Guide];common(t,ctx);texture(t,ctx,visibilityRef,ST_VISIBILITY);texture(t,ctx,depthRef,ST_DEPTH);texture(t,ctx,pointRef,ST_WORLD_POSITION);texture(t,ctx,normalRef,ST_GEOMETRIC_NORMAL);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),pipelines,kernels[Guide],t,frame.width,frame.height);});
        g.addPass("Indexed overflow receiver prepass",PassType::Raster,[&](PassBuilder& b){
            pointRef=b.writeColor(pointRef,0,LoadIntent::Preserve);normalRef=b.writeColor(normalRef,1,LoadIntent::Preserve);keyRef=b.writeColor(keyRef,2,LoadIntent::Preserve);
            depthRef=b.writeDepth(depthRef,LoadIntent::Preserve);scene.declareDrawReads(b);b.read(poseRef,Usage::ShaderRead,StageVertex);b.setProfileShaders("forward_surface_vs,forward_shadow_receiver_fs");
        },[this](PassContext& ctx){scene.encodeOverlay(static_cast<MTL4::RenderCommandEncoder*>(ctx.encoder()),receiver,true,receiverDepthState);});
        g.addPass("Indexed receiver pack",PassType::Compute,[&](PassBuilder& b){
            for(auto r:{pointRef,normalRef,keyRef,depthRef})b.read(r,Usage::ShaderRead,StageDispatch);b.read(surfaceRef,Usage::ShaderRead,StageDispatch);surfaceRef=b.write(surfaceRef,Usage::ShaderWrite,StageDispatch);
            b.setProfileShaders("shadow_receiver_pack");
        },[this](PassContext& ctx){auto* t=slots[frame.slot].tables[Pack];common(t,ctx);texture(t,ctx,pointRef,5);texture(t,ctx,normalRef,6);texture(t,ctx,keyRef,10);texture(t,ctx,depthRef,0);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),pipelines,kernels[Pack],t,frame.width,frame.height);});
        g.addPass("CSM caster selection",PassType::Compute,[&](PassBuilder& b){
            b.read(scene.dataRef(),Usage::ShaderRead,StageDispatch);flagsRef=b.write(flagsRef,Usage::ShaderWrite,StageDispatch);
            b.read(counterRef,Usage::ShaderRead,StageDispatch);counterRef=b.write(counterRef,Usage::ShaderWrite,StageDispatch);b.setProfileShaders("shadow_caster_flags");
        },[this](PassContext& ctx){auto* t=slots[frame.slot].tables[Cull];common(t,ctx);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),pipelines,kernels[Cull],t,std::max(1u,params.slotCount));});
        for(u32 cascade=0;cascade<4;++cascade)g.addPass("CSM cascade "+std::to_string(cascade),PassType::Raster,[&,cascade](PassBuilder& b){
            ClearValue clear;clear.depth=0;maps[cascade]=b.createTexture("CSM depth "+std::to_string(cascade),{Format::Depth32Float,settings.mapResolution,settings.mapResolution});maps[cascade]=b.writeDepth(maps[cascade],LoadIntent::Clear,clear);
            b.read(scene.dataRef(),Usage::ShaderRead,StageVertex|StageMesh|StageFragment);b.read(flagsRef,Usage::ShaderRead,StageVertex|StageMesh);b.setProfileShaders("shadow_depth_mesh,shadow_depth_vertex,shadow_depth_fragment");
        },[this,cascade](PassContext& ctx){
            auto* e=static_cast<MTL4::RenderCommandEncoder*>(ctx.encoder());auto& slot=slots[frame.slot];e->setDepthStencilState(depthState);
            const u32 base=cascade*(geometry.size()+1);
            if(!options.forceApple9 && mesh.meshletCount() && params.slotCount && mesh.meshletCount()<=65535 && params.slotCount<=65535) {
                auto* t=slot.rasterTables[base];common(t,ctx);t->setAddress(slot.drawParams[base],1);e->setRenderPipelineState(pipelines.render(meshDepth));e->setArgumentTable(t,MTL::RenderStageMesh|MTL::RenderStageFragment);
                e->drawMeshThreadgroups(MTL::Size::Make(mesh.meshletCount(),params.slotCount,1),MTL::Size::Make(1,1,1),MTL::Size::Make(128,1,1));
            } else {
                e->setRenderPipelineState(pipelines.render(indexed));
                for(u32 m=0;m<geometry.size();++m) {
                    if(!geometry[m].indexCount || !params.slotCount)continue;
                    auto* t=slot.rasterTables[base+m+1];common(t,ctx);t->setAddress(slot.drawParams[base+m+1],1);e->setArgumentTable(t,MTL::RenderStageVertex|MTL::RenderStageFragment);
                    e->drawIndexedPrimitives(MTL::PrimitiveTypeTriangle,geometry[m].indexCount,MTL::IndexTypeUInt32,
                        scene.indexBuffer()->gpuAddress()+u64(geometry[m].indexOffset)*4,u64(geometry[m].indexCount)*4,params.slotCount,geometry[m].vertexOffset,0);
                }
            }
            e->setDepthStencilState(scene.depthState());
        });
        const bool ray=options.shadows==ShadowMode::RT;
        g.addPass(ray?"Solar RT visibility":"CSM PCSS visibility",PassType::Compute,[&](PassBuilder& b){
            b.read(surfaceRef,Usage::ShaderRead,StageDispatch);if(ray)rt->declareTraceReads(b);else for(auto r:maps)b.read(r,Usage::ShaderRead,StageDispatch);
            b.read(counterRef,Usage::ShaderRead,StageDispatch);counterRef=b.write(counterRef,Usage::ShaderWrite,StageDispatch);
            rawMask=b.createTexture("Sun visibility raw",{Format::R16Float,frame.width,frame.height});rawMask=b.write(rawMask,Usage::ShaderWrite,StageDispatch);
            b.setProfileShaders(ray?"shadow_sun_rt":"shadow_csm_pcss");
        },[this,ray](PassContext& ctx){auto* t=slots[frame.slot].tables[ray?Sun:Pcss];common(t,ctx);texture(t,ctx,rawMask,4);if(ray)sunConsumer->bind(t,0,4);else for(u32 i=0;i<4;++i)texture(t,ctx,maps[i],i?6+i:2);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),pipelines,kernels[ray?Sun:Pcss],t,frame.width,frame.height);});
        previousRef=g.importBuffer("Solar history previous per view",{capacity*sizeof(GPUShadowHistory)},ImportContentsDefined|ImportOutput);
        nextRef=g.importBuffer("Solar history next per view",{capacity*sizeof(GPUShadowHistory)},ImportOutput);
        auto raw=rawMask;
        g.addPass("Solar temporal moments",PassType::Compute,[&](PassBuilder& b){b.read(surfaceRef,Usage::ShaderRead,StageDispatch);b.read(raw,Usage::ShaderRead,StageDispatch);b.read(previousRef,Usage::ShaderRead,StageDispatch);nextRef=b.write(nextRef,Usage::ShaderWrite,StageDispatch);
            b.read(counterRef,Usage::ShaderRead,StageDispatch);counterRef=b.write(counterRef,Usage::ShaderWrite,StageDispatch);temporalOutput=b.createTexture("Sun temporal visibility",{Format::R16Float,frame.width,frame.height});temporalOutput=b.write(temporalOutput,Usage::ShaderWrite,StageDispatch);b.setProfileShaders("shadow_temporal");
        },[this,raw](PassContext& ctx){auto* t=slots[frame.slot].tables[Temporal];common(t,ctx);auto& view=views[frame.view];t->setAddress(view.previous->gpuAddress(),12);t->setAddress(view.next->gpuAddress(),13);texture(t,ctx,raw,3);texture(t,ctx,temporalOutput,4);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),pipelines,kernels[Temporal],t,frame.width,frame.height);history.read(frame.view,frame.index+1);history.write(frame.view,frame.index+1,params.viewProjection);});
        auto temporalMask=temporalOutput;
        g.addPass("Solar spatial filter",PassType::Compute,[&](PassBuilder& b){b.read(surfaceRef,Usage::ShaderRead,StageDispatch);b.read(temporalMask,Usage::ShaderRead,StageDispatch);b.read(nextRef,Usage::ShaderRead,StageDispatch);filterOutput=b.createTexture("Sun filtered visibility",{Format::R16Float,frame.width,frame.height});filterOutput=b.write(filterOutput,Usage::ShaderWrite,StageDispatch);b.setProfileShaders("shadow_filter");
        },[this,temporalMask](PassContext& ctx){auto* t=slots[frame.slot].tables[Filter];common(t,ctx);t->setAddress(views[frame.view].next->gpuAddress(),13);texture(t,ctx,temporalMask,3);texture(t,ctx,filterOutput,4);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),pipelines,kernels[Filter],t,frame.width,frame.height);});
        maskRef=filterOutput;
        if(settings.contact){auto primary=filterOutput;g.addPass("Solar contact detail",PassType::Compute,[&](PassBuilder& b){b.read(surfaceRef,Usage::ShaderRead,StageDispatch);b.read(depthRef,Usage::ShaderRead,StageDispatch);b.read(primary,Usage::ShaderRead,StageDispatch);b.read(counterRef,Usage::ShaderRead,StageDispatch);counterRef=b.write(counterRef,Usage::ShaderWrite,StageDispatch);contactOutput=b.createTexture("Sun visibility with contact",{Format::R16Float,frame.width,frame.height});contactOutput=b.write(contactOutput,Usage::ShaderWrite,StageDispatch);b.setProfileShaders("shadow_contact");},[this,primary](PassContext& ctx){auto* t=slots[frame.slot].tables[Contact];common(t,ctx);texture(t,ctx,primary,3);texture(t,ctx,contactOutput,4);texture(t,ctx,depthRef,0);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),pipelines,kernels[Contact],t,frame.width,frame.height);});}
        if(settings.contact)maskRef=contactOutput;
        g.addPass("Lighting zero fallback",PassType::Compute,[&](PassBuilder& b){zeroRef=b.createTexture("Zero direct and indirect fallback",{Format::RGBA16Float,frame.width,frame.height});zeroRef=b.write(zeroRef,Usage::ShaderWrite,StageDispatch);b.setProfileShaders("lighting_zero");},[this](PassContext& ctx){auto* t=slots[frame.slot].receiverTable;texture(t,ctx,zeroRef,0);auto* e=static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder());lighting::dispatch(e,pipelines,zeroPipeline,t,frame.width,frame.height);});
    }
    pipe::PipelineHandle zeroPipeline=pipe::INVALID_PIPELINE;
    void bind(MetalGraphExecutor& e) {
        auto& f=slots[frame.slot];auto& v=views[frame.view];e.bindBuffer(surfaceRef,f.surfaces);e.bindBuffer(flagsRef,f.flags);e.bindBuffer(counterRef,f.counters);e.bindBuffer(previousRef,v.previous);e.bindBuffer(nextRef,v.next);
    }
};
ShadowPasses::ShadowPasses(MetalContext& c,PipelineCache& p,SceneRenderer& s,MeshRenderer& m,VisibilityRenderer& v,AccelerationStructures* rt,const LaunchOptions& o):impl_(std::make_unique<Impl>(c,p,s,m,v,rt,o)){impl_->zeroPipeline=p.request(lighting::kernel("lighting_zero"));}
ShadowPasses::~ShadowPasses()=default;
void ShadowPasses::loadScene(const GpuScene& g,const SceneStore& s){impl_->load(g,s);}
void ShadowPasses::prepareFrame(const SceneStore& s,std::span<const GPULight> l,const Frame& f){
    auto& v=impl_->views[f.view];if(impl_->history.get(f.view).valid)std::swap(v.previous,v.next);
    impl_->prepare(s,l,f);
}
void ShadowPasses::addToGraph(rg::RenderGraph& g,rg::TextureRef v,rg::TextureRef d){impl_->add(g,v,d);}
void ShadowPasses::bindFrame(MetalGraphExecutor& e){impl_->bind(e);}
rg::BufferRef ShadowPasses::surfaces()const{return impl_->surfaceRef;}
rg::TextureRef ShadowPasses::worldPosition()const{return impl_->pointRef;}
rg::TextureRef ShadowPasses::geometricNormal()const{return impl_->normalRef;}
rg::TextureRef ShadowPasses::receiverKeys()const{return impl_->keyRef;}
rg::TextureRef ShadowPasses::mask()const{return impl_->maskRef;}
rg::TextureRef ShadowPasses::zeroLighting()const{return impl_->zeroRef;}
u32 ShadowPasses::sunIndex()const{return impl_->selectedSun;}
u64 ShadowPasses::version()const{return impl_->graphVersion;}
GPUShadowCounters ShadowPasses::counters(u32 slot)const{return *static_cast<const GPUShadowCounters*>(impl_->slots.at(slot).counters->contents());}
} // namespace phosphor
