#include "platform/metal/gi_passes.h"
#include "platform/metal/lighting_dispatch.h"
#include "platform/metal/acceleration_structures.h"
#include "platform/metal/scene_renderer.h"
#include "platform/metal/metal_graph_executor.h"
#include "renderer/probe_grid.h"
#include "renderer/radiance_cache.h"
#include "renderer/gpu_scene.h"
#include "renderer/scene_store.h"
#include "renderer/cull_math.h"
#include "renderer/transform_reference.h"
#include "renderer/transform_math.h"
#include "rendergraph/pass_context.h"
#include <glm/gtc/type_ptr.hpp>
#include <cstdio>
namespace phosphor {
struct GiPasses::Impl {
    enum Pass { Trace,Classify,Blend,Resolve,Cache,Candidates,Temporal,Spatial,Shade,Snapshot,CheckClear,CheckNegative,Check,Count };
    MetalContext& c;PipelineCache& p;SceneRenderer& scene;AccelerationStructures& rt;ShadowPasses& shadow;DirectLightingPasses& direct;LaunchOptions options;
    GiLightingEpoch lightingEpoch;GiEnvironment environment;
    ProbeGridConfig config;std::unique_ptr<ProbeGrid> oracle;
    ShadowPasses::Frame frame;GPUProbeGridParams params{};GPUProbeTraceExtra extra{};
    MTL::GPUAddress paramsAddress=0,extraAddress=0,checkAddress=0;
    GPUGiCheckParams checkParams{};pipe::PipelineHandle checkClear{},checkKernel{},checkNegative{};rg::BufferRef checkRef{};
    std::array<pipe::PipelineHandle,Count> kernels{};
    RtConsumer traceConsumer,candidateConsumer,temporalConsumer,spatialConsumer;
    struct Slot {
        MTL::Buffer *rays=nullptr,*cacheCandidates=nullptr,*fresh=nullptr,*temporal=nullptr,*spatial=nullptr,*check=nullptr;
        u32 expectedPixels=0;
        std::array<MTL4::ArgumentTable*,Count> tables{};
    };std::array<Slot,METAL_FRAMES_IN_FLIGHT> slots{};
    MTL::Buffer *states=nullptr,*cache=nullptr;std::array<MTL::Buffer*,4> histories{};
    MTL::Texture *previousIrr=nullptr,*nextIrr=nullptr,*previousDist=nullptr,*nextDist=nullptr;
    u32 probeCount=0,rayCount=0,irrWidth=0,irrHeight=0,distWidth=0,distHeight=0;
    u64 pixels=0,graphVersion=1,sceneRevision=1,materialRevision=1;u32 generation=1;struct Signal {u64 scene,geometry,materials,rtGeometry;u64 lights;bool operator==(const Signal&)const=default;};Signal lastSignal{~u64{0},0,0,0,0};
    bool atlasValid=false;HistoryRegistry history;
    rg::BufferRef stateRef{},cacheRef{},raysRef{},cacheCandidatesRef{},freshRef{},temporalRef{},spatialRef{},historyRef{};
    rg::TextureRef previousIrrRef{},nextIrrRef{},previousDistRef{},nextDistRef{},output{},referenceDiffuse{};
    pipe::PipelineHandle referencePipeline{};MTL4::ArgumentTable* referenceTable=nullptr;
    Impl(MetalContext& context,PipelineCache& pipelines,SceneRenderer& s,AccelerationStructures& a,ShadowPasses& sh,DirectLightingPasses& d,const LaunchOptions& o)
        :c(context),p(pipelines),scene(s),rt(a),shadow(sh),direct(d),options(o),traceConsumer(c,p,a),candidateConsumer(c,p,a),temporalConsumer(c,p,a),spatialConsumer(c,p,a) {
        const char* names[]={"ddgi_trace","ddgi_classify","ddgi_blend","ddgi_resolve","radiance_cache_update","gi_candidates","gi_temporal","gi_spatial","gi_shade"};
        for(u32 i=0;i<Snapshot;++i)kernels[i]=p.request(lighting::kernel(names[i],i==Trace||i==Candidates||i==Temporal||i==Spatial));
        for(auto& f:slots){for(auto*& t:f.tables)t=lighting::table(c);if(o.debugLighting)f.check=lighting::buffer(c,32,"GI invariant checker",true);}
        if(o.debugLighting){checkClear=p.request(lighting::kernel("gi_check_clear"));checkKernel=p.request(lighting::kernel("gi_check_state"));checkNegative=p.request(lighting::kernel("gi_corrupt_state"));}
        referencePipeline=p.request(lighting::kernel("gi_reference_diffuse"));referenceTable=lighting::table(c);
    }
    ~Impl(){c.waitIdle();if(referenceTable)referenceTable->release();releaseVolume();for(auto& f:slots){for(auto* b:{f.fresh,f.temporal,f.spatial,f.check})c.memory().release(b,MemoryCategory::RayTracing);for(auto* t:f.tables)if(t)t->release();}for(auto* b:histories)c.memory().release(b,MemoryCategory::RayTracing);}
    void releaseVolume(){for(auto* b:{states,cache})c.memory().release(b,MemoryCategory::RayTracing);states=cache=nullptr;for(auto* t:{previousIrr,nextIrr,previousDist,nextDist})c.memory().release(t,MemoryCategory::RayTracing);previousIrr=nextIrr=previousDist=nextDist=nullptr;for(auto& f:slots){c.memory().release(f.rays,MemoryCategory::RayTracing);c.memory().release(f.cacheCandidates,MemoryCategory::RayTracing);f.rays=f.cacheCandidates=nullptr;}}
    void load(const GpuScene& g,const SceneStore& s){
        c.waitIdle();releaseVolume();config=ProbeGridConfig{};config.raysPerProbe=options.giRays;
        if(options.reducedLighting||options.forceApple9){config.counts={4,3,4};config.raysPerProbe=std::min(32u,options.giRays);}
        // Freeze a scene-volume preset at load. The API stays bounded and the
        // tester must sweep grid spacing/ray budget before quality acceptance.
        glm::vec3 low(1e30f),high(-1e30f);bool any=false;
        std::vector<float> worlds;float phase[SCENE_MOTION_CLASSES*2];motionSinCosTable(0,phase);referenceWorlds(s,phase,worlds);
        u32 worldSlot=0;for(const auto& i:s.instances()) { const u32 slot=worldSlot++;if((i.flags&INSTANCE_FLAG_VALID)&&i.meshIndex<g.meshInfos().size()) {
            const auto sphere=cullWorldSphere(worlds.data()+size_t(slot)*16,g.meshInfos()[i.meshIndex].boundingSphere);
            low=glm::min(low,glm::vec3(sphere.x,sphere.y,sphere.z)-sphere.r);high=glm::max(high,glm::vec3(sphere.x,sphere.y,sphere.z)+sphere.r);any=true;
            u32 root=slot;while(s.nodes()[root].depth && s.nodes()[root].parentSlot<s.slotCapacity())root=s.nodes()[root].parentSlot;
            if(std::find(s.motionSlots().begin(),s.motionSlots().end(),root)!=s.motionSlots().end()) {
                const auto& m=s.motions()[root];const glm::vec3 center(m.centre[0],m.centre[1]+m.height,m.centre[2]);
                const float reach=glm::length(glm::vec3(sphere.x,sphere.y,sphere.z)-center)+sphere.r+std::abs(m.radius);
                low=glm::min(low,center-glm::vec3(reach));high=glm::max(high,center+glm::vec3(reach));
            }
        }}
        if(any){config.origin=low-glm::vec3(0.25f);config.spacing=glm::max((high-low+glm::vec3(0.5f))/glm::vec3(config.counts-glm::uvec3(1)),glm::vec3(0.1f));}
        if(options.giGrid[0])config.counts={options.giGrid[0],options.giGrid[1],options.giGrid[2]};
        if(options.giSpacing>0)config.spacing=glm::vec3(options.giSpacing);
        if(options.giProbeAnchor)config.origin=glm::vec3(options.giAnchor[0],options.giAnchor[1],options.giAnchor[2])-glm::vec3(config.counts/2u)*config.spacing;
        oracle=std::make_unique<ProbeGrid>(config);probeCount=oracle->probeCount();rayCount=probeCount*config.raysPerProbe;
        states=lighting::buffer(c,probeCount*sizeof(GPUProbeState),"DDGI probe state",true);std::memcpy(states->contents(),oracle->states().data(),probeCount*sizeof(GPUProbeState));
        cache=lighting::buffer(c,16384*sizeof(GPURadianceCacheEntry),"Bounded directional radiance cache",true);std::memset(cache->contents(),0,cache->length());
        irrWidth=config.counts.x*(config.irradianceTexels+2);irrHeight=config.counts.y*config.counts.z*(config.irradianceTexels+2);
        distWidth=config.counts.x*(config.distanceTexels+2);distHeight=config.counts.y*config.counts.z*(config.distanceTexels+2);
        previousIrr=lighting::texture(c,irrWidth,irrHeight,MTL::PixelFormatRGBA32Float,"DDGI previous irradiance");nextIrr=lighting::texture(c,irrWidth,irrHeight,MTL::PixelFormatRGBA32Float,"DDGI next irradiance");
        previousDist=lighting::texture(c,distWidth,distHeight,MTL::PixelFormatRG32Float,"DDGI previous distance moments");nextDist=lighting::texture(c,distWidth,distHeight,MTL::PixelFormatRG32Float,"DDGI next distance moments");
        for(auto& f:slots){f.rays=lighting::buffer(c,rayCount*sizeof(GPUProbeRay),"DDGI radiance distance rays");f.cacheCandidates=lighting::buffer(c,rayCount*sizeof(GPUGiReservoir),"DDGI cache radiance candidates");}
        atlasValid=false;++sceneRevision;++materialRevision;++generation;++graphVersion;lastSignal={~u64{0},0,0,0,0};for(u32 v=0;v<4;++v)history.invalidate(v,"GI scene load");
    }
    void reserve(u64 capacity){if(capacity<=pixels)return;for(u32 v=0;v<4;++v)history.invalidate(v,"GI allocation growth");pixels=capacity;for(auto& f:slots){for(auto* b:{f.fresh,f.temporal,f.spatial})c.memory().release(b,MemoryCategory::RayTracing);f.fresh=lighting::buffer(c,pixels*sizeof(GPUGiReservoir),"GI candidate reservoirs");f.temporal=lighting::buffer(c,pixels*sizeof(GPUGiReservoir),"GI temporal reservoirs");f.spatial=lighting::buffer(c,pixels*sizeof(GPUGiReservoir),"GI spatial reservoirs");}for(auto*& b:histories){c.memory().release(b,MemoryCategory::RayTracing);b=lighting::buffer(c,pixels*sizeof(GPUGiReservoir),"GI reservoir per view");}++graphVersion;}
    void prepare(const GpuScene& g,const SceneStore& s,std::span<const GPULight> lights,const ShadowPasses::Frame& f){
        frame=f;reserve(u64(f.backingWidth)*f.backingHeight);
        if(atlasValid){std::swap(previousIrr,nextIrr);std::swap(previousDist,nextDist);}
        if(s.stats().structure||s.stats().fullInstances||!s.instanceDeltas().empty()||!s.motionSlots().empty()||!s.dirtyRoots().empty())++sceneRevision;
        if(s.stats().fullMaterials||!s.materialDeltas().empty())++materialRevision;
        const u64 allLightEpoch=lightingEpoch.update(lights,environment,direct.lightRevision());
        const Signal signal{f.scene,sceneRevision,materialRevision,rt.geometryRevision(),allLightEpoch};
        if(!(signal==lastSignal)){++generation;lastSignal=signal;atlasValid=false;}
        const auto decision=history.begin(f.view,{f.width,f.height,f.backingWidth,f.backingHeight},generation,f.cut,f.reset);
        params=oracle->parameters(f.index,generation);params.width=f.width;params.height=f.height;
        params.slotCount=s.slotCapacity();params.meshCount=g.getMeshCount();params.materialCount=s.materials().size();params.lightCount=f.constants.lightCount;
        params.mode=options.gi==GiMode::DDGI?GI_MODE_DDGI:options.gi==GiMode::Cache?GI_MODE_CACHE:GI_MODE_RESTIR;
        // Probe/cache reset is independent of view/camera history reset. Since
        // the shader ABI uses one bit, a view cut may conservatively clear probes.
        params.reset=(!atlasValid||decision.reset)?1u:0u;
        params.cacheCapacity=16384;params.cacheProbeLimit=8;params.cacheMaxAge=60;params.cacheGeneration=generation;
        params.geometryRevision=sceneRevision;params.lightRevision=u32(allLightEpoch);params.materialRevision=materialRevision;
        params.viewRevision=(f.view<<28)|(u32(history.get(f.view).generation)&0x0fffffffu);
        extra={direct.lightCount(),std::min(256u,rayCount),rayCount,options.lightingSeed};extra.sunAngularRadius=environment.sunAngularRadius;
        for(u32 i=0;i<3;++i)params.skyRadiance[i]=environment.skyRadiance[i];
        checkParams={f.width,f.height,params.mode,options.debugGiCorrupt};checkAddress=lighting::upload(c,checkParams);slots[f.slot].expectedPixels=f.width*f.height;
        paramsAddress=lighting::upload(c,params);extraAddress=lighting::upload(c,extra);
        traceConsumer.prepare(f.slot,kernels[Trace]);candidateConsumer.prepare(f.slot,kernels[Candidates]);temporalConsumer.prepare(f.slot,kernels[Temporal]);spatialConsumer.prepare(f.slot,kernels[Spatial]);
    }
    void tex(MTL4::ArgumentTable* t,rg::PassContext& ctx,rg::TextureRef r,u32 b){t->setTexture(static_cast<MTL::Texture*>(ctx.texture(r))->gpuResourceID(),b);}
    void traceTable(MTL4::ArgumentTable* t){auto r=rt.traceResources(frame.slot);t->setAddress(paramsAddress,1);t->setAddress(states->gpuAddress(),2);t->setAddress(r.instances->gpuAddress(),5);t->setAddress(r.meshes->gpuAddress(),6);t->setAddress(r.vertices->gpuAddress(),7);t->setAddress(r.indices->gpuAddress(),8);t->setAddress(r.materials->gpuAddress(),9);t->setAddress(r.textures->gpuAddress(),10);t->setAddress(scene.lightsAddress(),11);t->setAddress(direct.lightsBuffer()->gpuAddress(),13);t->setAddress(extraAddress,14);t->setAddress(cache->gpuAddress(),15);t->setAddress(direct.emittersBuffer()->gpuAddress(),16);}
    void add(rg::RenderGraph& g){using namespace rg;auto& f=slots[frame.slot];
        stateRef=g.importBuffer("DDGI states",{states->length()},ImportContentsDefined|ImportOutput);cacheRef=g.importBuffer("Directional radiance cache",{cache->length()},ImportContentsDefined|ImportOutput);
        raysRef=g.importBuffer("DDGI rays",{f.rays->length()},ImportPerFrame);cacheCandidatesRef=g.importBuffer("DDGI outgoing cache candidates",{f.cacheCandidates->length()},ImportPerFrame);
        freshRef=g.importBuffer("GI candidate reservoirs",{f.fresh->length()},ImportPerFrame);temporalRef=g.importBuffer("GI temporal reservoirs",{f.temporal->length()},ImportPerFrame);spatialRef=g.importBuffer("GI spatial reservoirs",{f.spatial->length()},ImportPerFrame);
        historyRef=g.importBuffer("GI reservoirs per view",{histories[frame.view]->length()},ImportContentsDefined|ImportOutput);
        previousIrrRef=g.importTexture("DDGI previous irradiance",{Format::RGBA32Float,irrWidth,irrHeight},ImportContentsDefined|ImportOutput);
        previousDistRef=g.importTexture("DDGI previous distance",{Format::RG32Float,distWidth,distHeight},ImportContentsDefined|ImportOutput);
        nextIrrRef=g.importTexture("DDGI current irradiance",{Format::RGBA32Float,irrWidth,irrHeight},ImportOutput);
        nextDistRef=g.importTexture("DDGI current distance",{Format::RG32Float,distWidth,distHeight},ImportOutput);
        auto traceReads=[&](PassBuilder& b){rt.declareTraceReads(b);b.read(stateRef,Usage::ShaderRead,StageDispatch);b.read(direct.lightsRef(),Usage::ShaderRead,StageDispatch);b.read(direct.emittersRef(),Usage::ShaderRead,StageDispatch);};
        g.addPass("DDGI radiance trace",PassType::Compute,[&](PassBuilder& b){traceReads(b);b.read(previousIrrRef,Usage::ShaderRead,StageDispatch);b.read(previousDistRef,Usage::ShaderRead,StageDispatch);raysRef=b.write(raysRef,Usage::ShaderWrite,StageDispatch);cacheCandidatesRef=b.write(cacheCandidatesRef,Usage::ShaderWrite,StageDispatch);b.setProfileShaders("ddgi_trace");},[this](PassContext& ctx){auto* t=slots[frame.slot].tables[Trace];traceTable(t);t->setAddress(slots[frame.slot].rays->gpuAddress(),3);t->setAddress(slots[frame.slot].cacheCandidates->gpuAddress(),12);traceConsumer.bind(t,0,4);tex(t,ctx,previousIrrRef,0);tex(t,ctx,previousDistRef,1);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),p,kernels[Trace],t,rayCount);});
        g.addPass("DDGI classify and relocate",PassType::Compute,[&](PassBuilder& b){b.read(raysRef,Usage::ShaderRead,StageDispatch);b.read(stateRef,Usage::ShaderRead,StageDispatch);stateRef=b.write(stateRef,Usage::ShaderWrite,StageDispatch);b.setProfileShaders("ddgi_classify");},[this](PassContext& ctx){auto* t=slots[frame.slot].tables[Classify];t->setAddress(paramsAddress,0);t->setAddress(states->gpuAddress(),1);t->setAddress(slots[frame.slot].rays->gpuAddress(),2);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),p,kernels[Classify],t,probeCount);});
        g.addPass("DDGI irradiance and moments atlases",PassType::Compute,[&](PassBuilder& b){b.read(raysRef,Usage::ShaderRead,StageDispatch);b.read(stateRef,Usage::ShaderRead,StageDispatch);b.read(previousIrrRef,Usage::ShaderRead,StageDispatch);b.read(previousDistRef,Usage::ShaderRead,StageDispatch);nextIrrRef=b.write(nextIrrRef,Usage::ShaderWrite,StageDispatch);nextDistRef=b.write(nextDistRef,Usage::ShaderWrite,StageDispatch);b.setProfileShaders("ddgi_blend");},[this](PassContext& ctx){auto* t=slots[frame.slot].tables[Blend];t->setAddress(paramsAddress,0);t->setAddress(states->gpuAddress(),1);t->setAddress(slots[frame.slot].rays->gpuAddress(),2);tex(t,ctx,previousIrrRef,0);tex(t,ctx,previousDistRef,1);tex(t,ctx,nextIrrRef,2);tex(t,ctx,nextDistRef,3);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),p,kernels[Blend],t,std::max(irrWidth,distWidth),std::max(irrHeight,distHeight));atlasValid=true;});
        g.addPass("Bounded radiance cache update",PassType::Compute,[&](PassBuilder& b){b.read(cacheCandidatesRef,Usage::ShaderRead,StageDispatch);b.read(cacheRef,Usage::ShaderRead,StageDispatch);cacheRef=b.write(cacheRef,Usage::ShaderWrite,StageDispatch);b.setProfileShaders("radiance_cache_update");},[this](PassContext& ctx){auto* t=slots[frame.slot].tables[Cache];t->setAddress(paramsAddress,0);t->setAddress(cache->gpuAddress(),1);t->setAddress(slots[frame.slot].cacheCandidates->gpuAddress(),2);t->setAddress(extraAddress,3);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),p,kernels[Cache],t,1);});
        const bool reuse=options.gi==GiMode::ReSTIR;
        if(options.gi!=GiMode::DDGI){
            g.addPass("GI cosine candidates",PassType::Compute,[&](PassBuilder& b){traceReads(b);b.read(cacheRef,Usage::ShaderRead,StageDispatch);for(auto r:{shadow.worldPosition(),shadow.geometricNormal(),nextIrrRef,nextDistRef})b.read(r,Usage::ShaderRead,StageDispatch);freshRef=b.write(freshRef,Usage::ShaderWrite,StageDispatch);b.setProfileShaders("gi_candidates");},[this](PassContext& ctx){auto* t=slots[frame.slot].tables[Candidates];traceTable(t);t->setAddress(slots[frame.slot].fresh->gpuAddress(),3);candidateConsumer.bind(t,0,4);tex(t,ctx,shadow.worldPosition(),0);tex(t,ctx,shadow.geometricNormal(),1);tex(t,ctx,nextIrrRef,2);tex(t,ctx,nextDistRef,3);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),p,kernels[Candidates],t,frame.width,frame.height);});
            if(reuse){
                g.addPass("GI temporal reconnection",PassType::Compute,[&](PassBuilder& b){rt.declareTraceReads(b);b.read(freshRef,Usage::ShaderRead,StageDispatch);b.read(historyRef,Usage::ShaderRead,StageDispatch);for(auto r:{shadow.worldPosition(),shadow.geometricNormal(),direct.motion()})b.read(r,Usage::ShaderRead,StageDispatch);temporalRef=b.write(temporalRef,Usage::ShaderWrite,StageDispatch);b.setProfileShaders("gi_temporal");},[this](PassContext& ctx){auto* t=slots[frame.slot].tables[Temporal];temporalConsumer.bind(t,0,5);t->setAddress(paramsAddress,1);t->setAddress(slots[frame.slot].fresh->gpuAddress(),2);t->setAddress(histories[frame.view]->gpuAddress(),3);t->setAddress(slots[frame.slot].temporal->gpuAddress(),4);t->setAddress(scene.buffers().instances()->gpuAddress(),6);tex(t,ctx,shadow.worldPosition(),0);tex(t,ctx,shadow.geometricNormal(),1);tex(t,ctx,direct.motion(),2);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),p,kernels[Temporal],t,frame.width,frame.height);});
                g.addPass("GI spatial reconnection",PassType::Compute,[&](PassBuilder& b){rt.declareTraceReads(b);b.read(temporalRef,Usage::ShaderRead,StageDispatch);for(auto r:{shadow.worldPosition(),shadow.geometricNormal()})b.read(r,Usage::ShaderRead,StageDispatch);spatialRef=b.write(spatialRef,Usage::ShaderWrite,StageDispatch);b.setProfileShaders("gi_spatial");},[this](PassContext& ctx){auto* t=slots[frame.slot].tables[Spatial];spatialConsumer.bind(t,0,4);t->setAddress(paramsAddress,1);t->setAddress(slots[frame.slot].temporal->gpuAddress(),2);t->setAddress(slots[frame.slot].spatial->gpuAddress(),3);t->setAddress(scene.buffers().instances()->gpuAddress(),5);tex(t,ctx,shadow.worldPosition(),0);tex(t,ctx,shadow.geometricNormal(),1);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),p,kernels[Spatial],t,frame.width,frame.height);});
            }
        }
        g.addPass(options.gi==GiMode::DDGI?"DDGI indirect irradiance":"GI reservoir irradiance",PassType::Compute,[&](PassBuilder& b){b.read(stateRef,Usage::ShaderRead,StageDispatch);if(options.gi!=GiMode::DDGI)b.read(reuse?spatialRef:freshRef,Usage::ShaderRead,StageDispatch);for(auto r:{shadow.worldPosition(),shadow.geometricNormal(),nextIrrRef,nextDistRef})b.read(r,Usage::ShaderRead,StageDispatch);output=b.createTexture("Indirect diffuse irradiance",{Format::RGBA32Float,frame.width,frame.height});output=b.write(output,Usage::ShaderWrite,StageDispatch);b.setProfileShaders(options.gi==GiMode::DDGI?"ddgi_resolve":"gi_shade");},[this,reuse](PassContext& ctx){const bool ddgi=options.gi==GiMode::DDGI;auto* t=slots[frame.slot].tables[ddgi?Resolve:Shade];t->setAddress(paramsAddress,0);if(ddgi)t->setAddress(states->gpuAddress(),1);else{t->setAddress((reuse?slots[frame.slot].spatial:slots[frame.slot].fresh)->gpuAddress(),1);t->setAddress(states->gpuAddress(),2);}tex(t,ctx,shadow.worldPosition(),0);tex(t,ctx,shadow.geometricNormal(),1);tex(t,ctx,nextIrrRef,2);tex(t,ctx,nextDistRef,3);tex(t,ctx,output,4);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),p,kernels[ddgi?Resolve:Shade],t,frame.width,frame.height);if(!reuse)history.write(frame.view,frame.index+1,frame.constants.viewProjection);});
        if(options.debugLighting) {
            checkRef=g.importBuffer("GI invariant check counters",{32},ImportPerFrame|ImportOutput);
            g.addPass("GI invariant check clear",PassType::Compute,[&](PassBuilder& b){checkRef=b.write(checkRef,Usage::ShaderWrite,StageDispatch);},[this](PassContext& ctx){auto* t=slots[frame.slot].tables[CheckClear];t->setAddress(slots[frame.slot].check->gpuAddress(),0);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),p,checkClear,t,8);});
            auto checkBind=[this,reuse](MTL4::ArgumentTable* t){t->setAddress(paramsAddress,0);t->setAddress(states->gpuAddress(),1);t->setAddress(cache->gpuAddress(),2);t->setAddress((reuse?slots[frame.slot].spatial:slots[frame.slot].fresh)->gpuAddress(),3);t->setAddress(slots[frame.slot].check->gpuAddress(),4);t->setAddress(checkAddress,5);};
            if(options.debugGiCorrupt)g.addPass("Negative: GI state",PassType::Compute,[&](PassBuilder& b){b.read(stateRef,Usage::ShaderRead,StageDispatch);stateRef=b.write(stateRef,Usage::ShaderWrite,StageDispatch);b.read(cacheRef,Usage::ShaderRead,StageDispatch);cacheRef=b.write(cacheRef,Usage::ShaderWrite,StageDispatch);if(options.gi!=GiMode::DDGI){const auto ref=reuse?spatialRef:freshRef;b.read(ref,Usage::ShaderRead,StageDispatch);if(reuse)spatialRef=b.write(ref,Usage::ShaderWrite,StageDispatch);else freshRef=b.write(ref,Usage::ShaderWrite,StageDispatch);}},[this,checkBind](PassContext& ctx){auto* t=slots[frame.slot].tables[CheckNegative];checkBind(t);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),p,checkNegative,t,1);});
            g.addPass("GI independent state check",PassType::Compute,[&](PassBuilder& b){b.read(stateRef,Usage::ShaderRead,StageDispatch);b.read(cacheRef,Usage::ShaderRead,StageDispatch);if(options.gi!=GiMode::DDGI)b.read(reuse?spatialRef:freshRef,Usage::ShaderRead,StageDispatch);b.read(output,Usage::ShaderRead,StageDispatch);b.read(checkRef,Usage::ShaderRead,StageDispatch);checkRef=b.write(checkRef,Usage::ShaderWrite,StageDispatch);},[this,checkBind](PassContext& ctx){auto* t=slots[frame.slot].tables[Check];checkBind(t);tex(t,ctx,output,0);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),p,checkKernel,t,std::max({frame.width*frame.height,probeCount,params.cacheCapacity}));});
        }
        if(options.captureLinearSignal==1)g.addPass("Independent indirect diffuse signal",PassType::Compute,[&](PassBuilder& b){
            b.read(output,Usage::ShaderRead,StageDispatch);b.read(direct.surfaceRef(),Usage::ShaderRead,StageDispatch);
            referenceDiffuse=b.createTexture("Unoccluded indirect diffuse reflected radiance",{Format::RGBA32Float,frame.width,frame.height});
            referenceDiffuse=b.write(referenceDiffuse,Usage::ShaderWrite,StageDispatch);b.setProfileShaders("gi_reference_diffuse");
        },[this](PassContext& ctx){referenceTable->setAddress(paramsAddress,0);referenceTable->setAddress(static_cast<MTL::Buffer*>(ctx.buffer(direct.surfaceRef()))->gpuAddress(),1);tex(referenceTable,ctx,output,0);tex(referenceTable,ctx,referenceDiffuse,1);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),p,referencePipeline,referenceTable,frame.width,frame.height);});
        if(reuse)g.addPass("GI history snapshot",PassType::Blit,[&](PassBuilder& b){b.read(temporalRef,Usage::CopySrc,StageBlit);historyRef=b.write(historyRef,Usage::CopyDst,StageBlit);},[this](PassContext& ctx){auto* e=static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder());e->copyFromBuffer(slots[frame.slot].temporal,0,histories[frame.view],0,u64(frame.width)*frame.height*sizeof(GPUGiReservoir));history.read(frame.view,frame.index+1);history.write(frame.view,frame.index+1,frame.constants.viewProjection);});

    }
    void bind(MetalGraphExecutor& e){auto& f=slots[frame.slot];e.bindBuffer(stateRef,states);e.bindBuffer(cacheRef,cache);e.bindBuffer(raysRef,f.rays);e.bindBuffer(cacheCandidatesRef,f.cacheCandidates);e.bindBuffer(freshRef,f.fresh);e.bindBuffer(temporalRef,f.temporal);e.bindBuffer(spatialRef,f.spatial);e.bindBuffer(historyRef,histories[frame.view]);if(options.debugLighting)e.bindBuffer(checkRef,f.check);e.bindTexture(previousIrrRef,previousIrr);e.bindTexture(previousDistRef,previousDist);e.bindTexture(nextIrrRef,nextIrr);e.bindTexture(nextDistRef,nextDist);}
};
GiPasses::GiPasses(MetalContext& c,PipelineCache& p,SceneRenderer& s,AccelerationStructures& a,ShadowPasses& sh,DirectLightingPasses& d,const LaunchOptions& o):impl_(std::make_unique<Impl>(c,p,s,a,sh,d,o)){}
GiPasses::~GiPasses()=default;
void GiPasses::loadScene(const GpuScene& g,const SceneStore& s){impl_->load(g,s);}void GiPasses::prepareFrame(const GpuScene& g,const SceneStore& s,std::span<const GPULight> lights,const ShadowPasses::Frame& f){impl_->prepare(g,s,lights,f);}
void GiPasses::setEnvironment(const GiEnvironment& value){impl_->environment=value;}void GiPasses::addToGraph(rg::RenderGraph& g){impl_->add(g);}void GiPasses::bindFrame(MetalGraphExecutor& e){impl_->bind(e);}
rg::TextureRef GiPasses::irradiance()const{return impl_->output;}u64 GiPasses::version()const{return impl_->graphVersion;}
} // namespace phosphor

namespace phosphor { rg::TextureRef GiPasses::referenceDiffuse()const{return impl_->referenceDiffuse;} }

namespace phosphor {bool GiPasses::check(u32 slot)const {
    if(!impl_->options.debugLighting)return true;const auto& f=impl_->slots.at(slot);const auto* count=static_cast<const u32*>(f.check->contents());
    bool okay=count[0]==f.expectedPixels && count[1]==impl_->probeCount && count[2]==16384;
    for(u32 i=3;i<8;++i)okay &= count[i]==0;
    if(!okay)std::printf("GI diagnostics slot %u | inspected %u/%u pixels %u/%u probes %u/16384 cache | problems output %u probe %u cache %u reservoir %u duplicate %u\n",
        slot,count[0],f.expectedPixels,count[1],impl_->probeCount,count[2],count[3],count[4],count[5],count[6],count[7]);
    return okay;
}}
