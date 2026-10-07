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
#include "renderer/shadow_dispatch.h"
#include "renderer/shadow_math.h"
#include "renderer/cull_math.h"
#include "renderer/transform_math.h"
#include "rendergraph/pass_context.h"
#include <glm/gtc/type_ptr.hpp>
#include <bit>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
namespace phosphor {
struct ShadowPasses::Impl {
    enum Pass { Clear, Guide, Pack, Cull, Pcss, Sun, Temporal, Filter, Contact, PassCount };
    MetalContext& c; PipelineCache& pipelines; SceneRenderer& scene; MeshRenderer& mesh; VisibilityRenderer& visibility;
    AccelerationStructures* rt;
    LaunchOptions options; ShadowSettings settings;
    std::unique_ptr<RtConsumer> sunConsumer;
    Frame frame{}; GPUShadowParams params{}; MTL::GPUAddress paramsAddress=0;
    std::array<pipe::PipelineHandle,PassCount> kernels{};
    pipe::PipelineHandle indexed, meshDepth, receiver, cacheIndexed, cacheMesh, cacheInitialize, cachePublish, cacheValidate, cacheComposite;
    MTL::DepthStencilState* depthState=nullptr;
    MTL::DepthStencilState* receiverDepthState=nullptr;
    MTL::DepthStencilState* compositeDepthState=nullptr;
    std::vector<GPUMeshInfo> geometry;
    struct Slot {
        MTL::Buffer *surfaces=nullptr,*flags=nullptr,*counters=nullptr;
        std::vector<u32> expectedFlags;u32 expectedCount=0;
        std::array<MTL4::ArgumentTable*,PassCount> tables{};
        MTL4::ArgumentTable* receiverTable=nullptr;
        std::vector<MTL4::ArgumentTable*> rasterTables;
        std::vector<MTL::GPUAddress> drawParams;
        std::vector<ShadowCasterDraw> casterDraws;
        std::array<MTL4::ArgumentTable*,4> initializeTables{},publishTables{},validateTables{},compositeTables{};
        std::array<MTL::GPUAddress,4> staticParams{},dynamicParams{},expectedAddress{};
        std::array<MTL::Buffer*,4> expectedBuffer{};
        MTL::GPUAddress classificationAddress=0;
        MTL::Buffer* classificationBuffer=nullptr;
    };
    std::array<Slot,METAL_FRAMES_IN_FLIGHT> slots{};
    struct View {
        MTL::Buffer *previous=nullptr,*next=nullptr;
        std::array<MTL::Texture*,4> cached{};
        std::array<MTL::Buffer*,4> cacheTiles{};
        std::array<std::array<ShadowCacheRevision,64>,4> desired{},published{};
        std::array<std::array<bool,64>,4> valid{};
        std::array<GPUShadowCascade,4> projection{};
        std::array<bool,4> hasProjection{};
        std::array<bool,4> initialized{};
        std::array<u64,4> projectionRevision{};
        std::array<u64,4> readyMask{},currentMask{},updateMask{};
        u32 nextUpdate=0;
        u32 width=0,height=0;
    };
    std::array<View,HistoryRegistry::MaxViews> views{};
    HistoryRegistry history;
    struct StaticSnapshot { GPUInstance instance{}; ShadowBounds bounds{}; bool valid=false; };
    std::vector<StaticSnapshot> oldStatic;
    std::vector<float> worldMatrices;
    std::vector<u32> staticClasses,movingSlots;
    std::vector<ShadowBounds> casterBounds,changedBounds;
    u64 staticCasterRevision=1;
    u64 capacity=0, casterCapacity=0, graphVersion=1, casterRevision=1, materialRevision=1, lightRevision=1;
    struct Signal {u64 scene,casters,materials,light,geometry;bool operator==(const Signal&)const=default;};
    Signal lastSignal{~u64{0},0,0,0,0};u64 signalEpoch=1;
    u64 previousLightHash=0;
    u32 cachedPipelineGeneration=0;
    u32 selectedSun=~0u;
    rg::BufferRef surfaceRef{},flagsRef{},counterRef{},previousRef{},nextRef{},poseRef{};
    rg::TextureRef visibilityRef{},depthRef{},pointRef{},normalRef{},keyRef{},maskRef{},zeroRef{};
    rg::TextureRef rawMask{},temporalOutput{},filterOutput{},contactOutput{};
    std::array<rg::TextureRef,4> maps{},cacheRefs{},currentStaticRefs{},dynamicRefs{};
    std::array<rg::BufferRef,4> tileRefs{},expectedRefs{};
    rg::BufferRef classificationRef{};
    Impl(MetalContext& context,PipelineCache& p,SceneRenderer& s,MeshRenderer& m,VisibilityRenderer& v,
         AccelerationStructures* a,const LaunchOptions& o):c(context),pipelines(p),scene(s),mesh(m),visibility(v),rt(a),options(o) {
        settings.mapResolution=o.shadowMapResolution; settings.contact=o.contactShadows; settings.staticCache=o.shadowCache;
        settings.mode=o.shadows==ShadowMode::RT?ShadowTechnique::RayTraced:
                      o.shadows==ShadowMode::CSM?ShadowTechnique::Cascaded:ShadowTechnique::Off;
        validateShadowSettings(settings);
        if(settings.staticCache && o.shadows!=ShadowMode::CSM)
            throw std::invalid_argument("--shadow-cache on requires --shadows csm");
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
        if(settings.staticCache) {
            d=pipe::PipelineDesc{};d.kind=pipe::PipelineKind::Render;d.label="Regional static/dynamic CSM indexed";
            d.functions={"shadow_cache_depth_vertex","shadow_cache_depth_fragment",""};cacheIndexed=p.request(d);
            d.kind=pipe::PipelineKind::Mesh;d.label="Regional static/dynamic CSM mesh";
            d.functions={"","shadow_cache_depth_mesh","shadow_cache_depth_fragment"};d.mesh.meshThreads=128;cacheMesh=p.request(d);
            cachePublish=p.request(lighting::kernel("shadow_cache_publish"));
            cacheInitialize=p.request(lighting::kernel("shadow_cache_initialize"));
            cacheValidate=p.request(lighting::kernel("shadow_cache_validate"));
            d=pipe::PipelineDesc{};d.kind=pipe::PipelineKind::Render;d.label="Compose nearest static/dynamic shadow depth";
            d.functions={"shadow_cache_composite_vertex","shadow_cache_composite_fragment",""};cacheComposite=p.request(d);
        }
        auto* dd=MTL::DepthStencilDescriptor::alloc()->init();dd->setDepthCompareFunction(MTL::CompareFunctionGreater);
        dd->setDepthWriteEnabled(true);depthState=c.device()->newDepthStencilState(dd);dd->release();
        dd=MTL::DepthStencilDescriptor::alloc()->init();dd->setDepthCompareFunction(MTL::CompareFunctionGreaterEqual);dd->setDepthWriteEnabled(true);receiverDepthState=c.device()->newDepthStencilState(dd);dd->release();
        dd=MTL::DepthStencilDescriptor::alloc()->init();dd->setDepthCompareFunction(MTL::CompareFunctionAlways);
        dd->setDepthWriteEnabled(true);compositeDepthState=c.device()->newDepthStencilState(dd);dd->release();
        if(!depthState || !receiverDepthState || !compositeDepthState) throw std::runtime_error("CSM depth state allocation failed");
        for(auto& f:slots) { for(auto*& t:f.tables)t=lighting::table(c); f.receiverTable=lighting::table(c);f.counters=lighting::buffer(c,sizeof(GPUShadowCounters),"Shadow counters",true); }
        if(settings.staticCache)for(auto& f:slots)for(u32 i=0;i<4;++i){f.initializeTables[i]=lighting::table(c);f.publishTables[i]=lighting::table(c);f.validateTables[i]=lighting::table(c);f.compositeTables[i]=lighting::table(c);}
        if(o.shadows==ShadowMode::RT) { if(!rt)throw std::logic_error("Sun RT without AS");sunConsumer=std::make_unique<RtConsumer>(c,p,*rt); }
    }
    ~Impl() {
        c.waitIdle();
        for(auto& f:slots) {
            for(auto* b:{f.surfaces,f.flags,f.counters})c.memory().release(b,MemoryCategory::RayTracing);
            for(auto* t:f.tables)if(t)t->release();if(f.receiverTable)f.receiverTable->release();
            for(auto* t:f.rasterTables)if(t)t->release();
            for(auto* t:f.initializeTables)if(t)t->release();for(auto* t:f.publishTables)if(t)t->release();for(auto* t:f.validateTables)if(t)t->release();for(auto* t:f.compositeTables)if(t)t->release();
        }
        for(auto& v:views) {c.memory().release(v.previous,MemoryCategory::RayTracing);c.memory().release(v.next,MemoryCategory::RayTracing);for(auto* t:v.cached)c.memory().release(t,MemoryCategory::RayTracing);for(auto* b:v.cacheTiles)c.memory().release(b,MemoryCategory::RayTracing);}
        if(depthState)depthState->release();if(receiverDepthState)receiverDepthState->release();if(compositeDepthState)compositeDepthState->release();
    }
    void prepareCasterDraws(Slot& slot,const SceneStore& store) {
        planShadowCasterDraws(store.buckets(),geometry,store.slotCapacity(),!options.forceApple9,slot.casterDraws);
        const size_t required=4u*(settings.staticCache?2u:1u)*slot.casterDraws.size();
        const size_t old=slot.rasterTables.size();
        // This slot has completed before prepareFrame. Never rewrite another
        // frame's argument tables, and keep a distinct table per recorded draw.
        if(required>old) {
            slot.rasterTables.resize(required,nullptr);
            for(size_t i=old;i<required;++i)slot.rasterTables[i]=lighting::table(c);
        }
        slot.drawParams.resize(required);
    }
    void load(const GpuScene& g,const SceneStore& store) {
        c.waitIdle();geometry.assign(g.meshInfos().begin(),g.meshInfos().end());
        for(auto& f:slots) {
            for(auto* t:f.rasterTables)if(t)t->release(); f.rasterTables.clear();
            f.drawParams.clear();f.casterDraws.clear();
            if(settings.mode==ShadowTechnique::Cascaded)prepareCasterDraws(f,store);
        }
        ++casterRevision;++staticCasterRevision;++materialRevision;++graphVersion;
        oldStatic.assign(store.slotCapacity(),{});
        for(auto& v:views) {
            v.valid={};v.hasProjection={};v.initialized={};
            if(settings.staticCache)for(u32 i=0;i<4;++i) {
                if(!v.cached[i])v.cached[i]=lighting::texture(c,settings.mapResolution,settings.mapResolution,MTL::PixelFormatR32Float,"Static shadow regional depth cache");
                if(!v.cacheTiles[i])v.cacheTiles[i]=lighting::buffer(c,64*sizeof(GPUShadowCacheTile),"Static shadow exact tile revisions");
            }
        }
        for(u32 v=0;v<views.size();++v)history.invalidate(v,"scene load");
        reserve(store.slotCapacity(),frame.backingWidth,frame.backingHeight);
    }
    void reserve(u32 slotsCount,u32 w,u32 h) {
        if(worldMatrices.size()<size_t(slotsCount)*16)worldMatrices.resize(size_t(slotsCount)*16);
        if(staticClasses.size()<slotsCount)staticClasses.resize(slotsCount);
        if(movingSlots.size()<slotsCount)movingSlots.resize(slotsCount);
        if(oldStatic.size()<slotsCount)oldStatic.resize(slotsCount);
        casterBounds.reserve(slotsCount);changedBounds.reserve(size_t(slotsCount)*2);
        const u64 pixels=u64(w)*h;
        if(pixels>capacity) {
            for(u32 view=0;view<views.size();++view)history.invalidate(view,"Solar allocation growth");
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
            for(auto& f:slots){c.memory().release(f.flags,MemoryCategory::RayTracing);f.flags=lighting::buffer(c,std::max<u64>(1,casterCapacity)*4,"CSM full-scene caster flags",options.debugLighting>0);f.expectedFlags.resize(casterCapacity);}
            ++graphVersion;
        }
    }
    static GPUShadowCacheTile gpuTile(ShadowCacheRevision r) {
        GPUShadowCacheTile t{};
        t.lightLo=u32(r.light);t.lightHi=u32(r.light>>32);t.casterLo=u32(r.caster);t.casterHi=u32(r.caster>>32);
        t.materialLo=u32(r.material);t.materialHi=u32(r.material>>32);t.projectionLo=u32(r.projection);t.projectionHi=u32(r.projection>>32);t.valid=1;return t;
    }
    void fullCasterBounds(const SceneStore& store,const Frame& f) {
        const u32 count=store.slotCapacity();const auto instances=store.instances();
        std::fill_n(movingSlots.begin(),count,0u);
        for(u32 slot:store.motionSlots())movingSlots.at(slot)=1u;
        if(!store.motionSlots().empty() && !f.motionSinCosValid)
            throw std::logic_error("Full-scene shadow bounds need the renderer's exact uploaded motion sin/cos table");
        for(u32 slot=0;slot<count;++slot) {
            std::memcpy(worldMatrices.data()+size_t(slot)*16,instances[slot].modelMatrix,64);
            if(movingSlots[slot]) {
                const auto& motion=store.motions()[slot];const u32 k=std::min(motion.speedClass,SCENE_MOTION_CLASSES-1);
                motionWorld(motion,f.motionSinCos[k*2],f.motionSinCos[k*2+1],worldMatrices.data()+size_t(slot)*16);
            }
        }
        // Same parent-before-child order and arithmetic as scene_hier_level.
        // Scratch is preallocated; there is no GPU readback or guessed motion.
        for(u32 depth=1;depth<SCENE_MAX_LEVELS;++depth)for(u32 slot=0;slot<count;++slot) {
            const auto& node=store.nodes()[slot];if(node.depth!=depth || !(instances[slot].flags&INSTANCE_FLAG_VALID))continue;
            if(node.parentSlot>=count)throw std::logic_error("Shadow hierarchy parent out of range");
            float matrix[16];mat4Mul(worldMatrices.data()+size_t(node.parentSlot)*16,node.local,matrix);
            std::memcpy(worldMatrices.data()+size_t(slot)*16,matrix,64);
        }
        casterBounds.clear();changedBounds.clear();
        for(u32 slot=0;slot<count;++slot) {
            const auto& instance=instances[slot];const bool casts=(instance.flags&INSTANCE_FLAG_VALID) && (instance.flags&2u) && instance.meshIndex<geometry.size();
            const bool isStatic=casts && (instance.flags&4u) && !movingSlots[slot] && store.nodes()[slot].depth==0;
            staticClasses[slot]=isStatic?1u:0u;
            ShadowBounds bounds{};
            if(casts) {
                const auto sphere=cullWorldSphere(worldMatrices.data()+size_t(slot)*16,geometry[instance.meshIndex].boundingSphere);
                const glm::vec3 center(sphere.x,sphere.y,sphere.z);
                const float radius=sphere.r+std::max(0.001f,sphere.r*1e-4f);
                if(!std::isfinite(radius)||!std::isfinite(center.x)||!std::isfinite(center.y)||!std::isfinite(center.z))
                    throw std::logic_error("Nonfinite full-scene shadow caster bounds");
                bounds={center-glm::vec3(radius),center+glm::vec3(radius)};casterBounds.push_back(bounds);
            }
            auto& old=oldStatic[slot];
            const bool changed=old.valid!=isStatic || (isStatic && std::memcmp(&old.instance,&instance,sizeof(instance))!=0);
            if(changed) { if(old.valid)changedBounds.push_back(old.bounds);if(isStatic)changedBounds.push_back(bounds); }
            old.instance=instance;old.bounds=bounds;old.valid=isStatic;
        }
        for(size_t slot=count;slot<oldStatic.size();++slot)if(oldStatic[slot].valid){changedBounds.push_back(oldStatic[slot].bounds);oldStatic[slot].valid=false;}
        if(!changedBounds.empty())++staticCasterRevision;
        // Update every view's tile revisions against its last projection, so
        // a view that was dormant during the change cannot reuse old depth.
        if(settings.staticCache)for(auto& view:views)for(u32 cascade=0;cascade<4;++cascade) {
            u64 dirty=0;
            if(!view.hasProjection[cascade])dirty=~u64(0);
            else for(auto bounds:changedBounds) {
                const float footprint=view.projection[cascade].radius*0.5f;
                bounds.minimum-=glm::vec3(footprint);bounds.maximum+=glm::vec3(footprint);
                dirty|=shadowDirtyTiles(view.projection[cascade],bounds);
            }
            for(u32 tile=0;tile<64;++tile)if(dirty&(u64(1)<<tile))view.desired[cascade][tile].caster=staticCasterRevision;
        }
    }
    void planCache() {
        auto& view=views[frame.view];auto& slot=slots[frame.slot];
        auto classification=c.frameUploads().allocate(std::max<u64>(1,params.slotCount)*sizeof(u32));
        if(params.slotCount)std::memcpy(classification.cpu,staticClasses.data(),params.slotCount*sizeof(u32));
        slot.classificationAddress=classification.gpu;slot.classificationBuffer=classification.buffer;
        for(u32 cascade=0;cascade<4;++cascade) {
            const auto& projection=params.cascades[cascade];
            if(!view.hasProjection[cascade] || std::memcmp(&view.projection[cascade],&projection,sizeof(projection))) {
                ++view.projectionRevision[cascade];view.projection[cascade]=projection;view.hasProjection[cascade]=true;
            }
            view.readyMask[cascade]=0;view.updateMask[cascade]=0;
            for(u32 tile=0;tile<64;++tile) {
                auto& revision=view.desired[cascade][tile];revision.light=lightRevision;revision.material=materialRevision;
                revision.projection=view.projectionRevision[cascade];
                if(view.valid[cascade][tile] && view.published[cascade][tile]==revision)view.readyMask[cascade]|=u64(1)<<tile;
            }
            view.currentMask[cascade]=~view.readyMask[cascade];
        }
        u32 reserved=0;
        for(u32 offset=0;offset<256 && reserved<settings.cacheUpdateBudget;++offset) {
            const u32 index=(view.nextUpdate+offset)%256,cascade=index/64,tile=index%64;
            if(!(view.currentMask[cascade]&(u64(1)<<tile)))continue;
            view.updateMask[cascade]|=u64(1)<<tile;++reserved;
        }
        view.nextUpdate=(view.nextUpdate+std::max(reserved,1u))%256;
        for(u32 cascade=0;cascade<4;++cascade) {
            GPUShadowCacheParams p{};p.resolution=settings.mapResolution;p.cascade=cascade;p.slotCount=params.slotCount;
            p.currentLo=u32(view.currentMask[cascade]);p.currentHi=u32(view.currentMask[cascade]>>32);
            p.updateLo=u32(view.updateMask[cascade]);p.updateHi=u32(view.updateMask[cascade]>>32);
            p.readyLo=u32(view.readyMask[cascade]);p.readyHi=u32(view.readyMask[cascade]>>32);p.corruption=params.corruption;
            p.pad=view.initialized[cascade]?0u:1u;
            if(params.corruption==SHADOW_CORRUPT_CACHE) { p.readyLo|=1u;p.updateLo&=~1u; } // GPU exact tuple oracle must reject an unpopulated/stale tile
            p.casterClass=0;slot.staticParams[cascade]=lighting::upload(c,p);
            p.casterClass=1;slot.dynamicParams[cascade]=lighting::upload(c,p);
            auto expected=c.frameUploads().allocate(64*sizeof(GPUShadowCacheTile));
            auto* out=reinterpret_cast<GPUShadowCacheTile*>(expected.cpu);
            for(u32 tile=0;tile<64;++tile)out[tile]=gpuTile(view.desired[cascade][tile]);
            if(params.corruption==SHADOW_CORRUPT_CACHE)out[0].projectionLo^=0x80000000u;
            slot.expectedAddress[cascade]=expected.gpu;slot.expectedBuffer[cascade]=expected.buffer;
        }
    }
    void prepare(const SceneStore& store,std::span<const GPULight> lights,const Frame& f) {
        frame=f;reserve(store.slotCapacity(),f.backingWidth,f.backingHeight);
        if(store.stats().structure || store.stats().fullInstances || !store.instanceDeltas().empty() || !store.motionSlots().empty() || !store.dirtyRoots().empty())++casterRevision;
        if(store.stats().fullMaterials || !store.materialDeltas().empty())++materialRevision;
        if(cachedPipelineGeneration!=pipelines.generation()) {
            cachedPipelineGeneration=pipelines.generation();++materialRevision; // alpha shader hot reload also invalidates static depth
        }
        selectedSun=~0u;u64 hash=1469598103934665603ull;
        for(u32 i=0;i<lights.size();++i)if(lights[i].type==LIGHT_DIRECTIONAL){selectedSun=i;break;}
        if(selectedSun!=~0u)for(u32 b:std::bit_cast<std::array<u32,sizeof(GPULight)/4>>(lights[selectedSun])){hash^=b;hash*=1099511628211ull;}
        if(hash!=previousLightHash){++lightRevision;previousLightHash=hash;}
        const Signal signal{f.scene,casterRevision,materialRevision,lightRevision,rt?rt->geometryRevision():0};
        if(!(signal==lastSignal)){++signalEpoch;lastSignal=signal;}const u64 signalRevision=signalEpoch;
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
        fullCasterBounds(store,f);
        const auto cascades=makeShadowCascades(camera,toward,settings,casterBounds);std::copy(cascades.begin(),cascades.end(),params.cascades);
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
        if(settings.mode==ShadowTechnique::Cascaded) {
            prepareCasterDraws(slot,store);
            const u32 classes=settings.staticCache?2u:1u;
            for(u32 cascade=0;cascade<4;++cascade)for(u32 cls=0;cls<classes;++cls) {
                const size_t base=(cascade*classes+cls)*slot.casterDraws.size();
                for(size_t i=0;i<slot.casterDraws.size();++i) {
                    const auto& range=slot.casterDraws[i];
                    auto draw=params;draw.cascadeIndex=cascade;draw.casterSlot=range.firstSlot;
                    draw.meshletFirst=range.meshletFirst;draw.meshletCount=range.meshletCount;
                    draw.meshletGridWidth=range.meshletCount;draw.pad=range.mesh;
                    slot.drawParams[base+i]=lighting::upload(c,draw);
                }
            }
        }
        if(options.debugLighting) {
            auto& slot=slots[f.slot];slot.expectedCount=params.slotCount;
            for(u32 i=0;i<params.slotCount;++i) {
                u32 flags=0;const auto& instance=store.instances()[i];
                if((instance.flags&INSTANCE_FLAG_VALID)&&(instance.flags&2u)&&instance.meshIndex<geometry.size()) {
                    const auto sphere=cullWorldSphere(worldMatrices.data()+size_t(i)*16,geometry[instance.meshIndex].boundingSphere);
                    for(u32 cascade=0;cascade<4;++cascade)if(shadowCasterIntersects(params.cascades[cascade],sphere.x,sphere.y,sphere.z,sphere.r))flags|=1u<<cascade;
                }
                slot.expectedFlags[i]=flags;
            }
        }
        if(settings.staticCache)planCache();
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
    void drawCascade(rg::PassContext& ctx,u32 cascade,u32 cls,bool cached) {
        auto* e=static_cast<MTL4::RenderCommandEncoder*>(ctx.encoder());auto& slot=slots[frame.slot];
        e->setDepthStencilState(depthState);
        const size_t base=(cascade*(settings.staticCache?2u:1u)+cls)*slot.casterDraws.size();
        // Shadow depth remains conservatively two-sided, as before this change.
        // Mesh/cull-class buckets stay separate; mirrored transforms and MASK
        // materials keep the exact same vertex/fragment handling.
        for(size_t i=0;i<slot.casterDraws.size();++i) {
            const auto& range=slot.casterDraws[i];
            auto* t=slot.rasterTables[base+i];common(t,ctx);t->setAddress(slot.drawParams[base+i],SB_PARAMS);
            if(cached){t->setAddress(cls==0?slot.staticParams[cascade]:slot.dynamicParams[cascade],18);t->setAddress(slot.classificationAddress,19);}
            if(range.meshShader) {
                e->setRenderPipelineState(pipelines.render(cached?cacheMesh:meshDepth));
                e->setArgumentTable(t,MTL::RenderStageMesh|MTL::RenderStageFragment);
                e->drawMeshThreadgroups(MTL::Size::Make(range.meshletCount,range.slotCount,1),
                    MTL::Size::Make(1,1,1),MTL::Size::Make(128,1,1));
            } else {
                const auto& info=geometry[range.mesh];
                e->setRenderPipelineState(pipelines.render(cached?cacheIndexed:indexed));
                e->setArgumentTable(t,MTL::RenderStageVertex|MTL::RenderStageFragment);
                // instance_id starts at zero. Shader adds range.firstSlot,
                // explicitly, so the offset cannot be applied twice.
                e->drawIndexedPrimitives(MTL::PrimitiveTypeTriangle,info.indexCount,MTL::IndexTypeUInt32,
                    scene.indexBuffer()->gpuAddress()+u64(info.indexOffset)*4,u64(info.indexCount)*4,
                    range.slotCount,info.vertexOffset,0);
            }
        }
        e->setDepthStencilState(scene.depthState());
    }
    void addCacheGraph(rg::RenderGraph& g) {
        using namespace rg;
        classificationRef=g.importBuffer("Conservative static caster classes",{std::max<u64>(1,casterCapacity)*4},ImportContentsDefined|ImportPerFrame);
        for(u32 cascade=0;cascade<4;++cascade) {
            const std::string suffix=std::to_string(cascade);
            cacheRefs[cascade]=g.importTexture("Persistent static shadow tiles "+suffix,{Format::R32Float,settings.mapResolution,settings.mapResolution},ImportContentsDefined|ImportOutput);
            tileRefs[cascade]=g.importBuffer("Persistent static tile revisions "+suffix,{64*sizeof(GPUShadowCacheTile)},ImportContentsDefined|ImportOutput);
            expectedRefs[cascade]=g.importBuffer("Current exact tile revisions "+suffix,{64*sizeof(GPUShadowCacheTile)},ImportContentsDefined|ImportPerFrame);
            g.addPass("Initialize static tile metadata "+suffix,PassType::Compute,[&,cascade](PassBuilder& b){
                tileRefs[cascade]=b.write(tileRefs[cascade],Usage::ShaderWrite,StageDispatch);b.setProfileShaders("shadow_cache_initialize");
            },[this,cascade](PassContext& ctx){
                auto& slot=slots[frame.slot];auto* t=slot.initializeTables[cascade];t->setAddress(slot.staticParams[cascade],18);
                t->setAddress(views[frame.view].cacheTiles[cascade]->gpuAddress(),21);
                lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),pipelines,cacheInitialize,t,64);
                views[frame.view].initialized[cascade]=true;
            });
            g.addPass("Current stale static regions "+suffix,PassType::Raster,[&,cascade](PassBuilder& b){
                ClearValue clear;clear.depth=0;
                currentStaticRefs[cascade]=b.createTexture("Current static fallback depth "+suffix,{Format::Depth32Float,settings.mapResolution,settings.mapResolution});
                currentStaticRefs[cascade]=b.writeDepth(currentStaticRefs[cascade],LoadIntent::Clear,clear);
                b.read(scene.dataRef(),Usage::ShaderRead,StageVertex|StageMesh|StageFragment);b.read(flagsRef,Usage::ShaderRead,StageVertex|StageMesh);
                b.read(classificationRef,Usage::ShaderRead,StageVertex|StageMesh);b.setProfileShaders("shadow_cache_depth_vertex,shadow_cache_depth_mesh,shadow_cache_depth_fragment");
            },[this,cascade](PassContext& ctx){drawCascade(ctx,cascade,0,true);});
            g.addPass("Current dynamic shadow depth "+suffix,PassType::Raster,[&,cascade](PassBuilder& b){
                ClearValue clear;clear.depth=0;
                dynamicRefs[cascade]=b.createTexture("Current dynamic caster depth "+suffix,{Format::Depth32Float,settings.mapResolution,settings.mapResolution});
                dynamicRefs[cascade]=b.writeDepth(dynamicRefs[cascade],LoadIntent::Clear,clear);
                b.read(scene.dataRef(),Usage::ShaderRead,StageVertex|StageMesh|StageFragment);b.read(flagsRef,Usage::ShaderRead,StageVertex|StageMesh);
                b.read(classificationRef,Usage::ShaderRead,StageVertex|StageMesh);b.setProfileShaders("shadow_cache_depth_vertex,shadow_cache_depth_mesh,shadow_cache_depth_fragment");
            },[this,cascade](PassContext& ctx){drawCascade(ctx,cascade,1,true);});
            g.addPass("Publish admitted static tiles "+suffix,PassType::Compute,[&,cascade](PassBuilder& b){
                b.read(currentStaticRefs[cascade],Usage::ShaderRead,StageDispatch);b.read(expectedRefs[cascade],Usage::ShaderRead,StageDispatch);
                cacheRefs[cascade]=b.write(cacheRefs[cascade],Usage::ShaderWrite,StageDispatch);tileRefs[cascade]=b.write(tileRefs[cascade],Usage::ShaderWrite,StageDispatch);
                b.setProfileShaders("shadow_cache_publish");
            },[this,cascade](PassContext& ctx){
                auto& slot=slots[frame.slot];auto* t=slot.publishTables[cascade];
                t->setAddress(slot.staticParams[cascade],18);t->setAddress(slot.expectedAddress[cascade],20);t->setAddress(views[frame.view].cacheTiles[cascade]->gpuAddress(),21);
                texture(t,ctx,currentStaticRefs[cascade],0);texture(t,ctx,cacheRefs[cascade],1);
                lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),pipelines,cachePublish,t,settings.mapResolution,settings.mapResolution);
                // CPU publication names scheduled writes. Persistent graph imports
                // order their completion before the next frame can read/rewrite.
                auto& view=views[frame.view];
                for(u32 tile=0;tile<64;++tile)if((view.updateMask[cascade]&(u64(1)<<tile)) && !(params.corruption==SHADOW_CORRUPT_CACHE && tile==0)) {
                    view.published[cascade][tile]=view.desired[cascade][tile];view.valid[cascade][tile]=true;
                }
            });
            g.addPass("Validate cached tile revisions "+suffix,PassType::Compute,[&,cascade](PassBuilder& b){
                b.read(tileRefs[cascade],Usage::ShaderRead,StageDispatch);b.read(expectedRefs[cascade],Usage::ShaderRead,StageDispatch);
                b.read(counterRef,Usage::ShaderRead,StageDispatch);counterRef=b.write(counterRef,Usage::ShaderWrite,StageDispatch);b.setProfileShaders("shadow_cache_validate");
            },[this,cascade](PassContext& ctx){
                auto& slot=slots[frame.slot];auto* t=slot.validateTables[cascade];t->setAddress(slot.staticParams[cascade],18);
                t->setAddress(slot.expectedAddress[cascade],20);t->setAddress(views[frame.view].cacheTiles[cascade]->gpuAddress(),21);t->setAddress(slot.counters->gpuAddress(),15);
                lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),pipelines,cacheValidate,t,64);
            });
            g.addPass("Compose nearest shadow depth "+suffix,PassType::Raster,[&,cascade](PassBuilder& b){
                b.read(cacheRefs[cascade],Usage::ShaderRead,StageFragment);b.read(tileRefs[cascade],Usage::ShaderRead,StageFragment);b.read(expectedRefs[cascade],Usage::ShaderRead,StageFragment);
                b.read(currentStaticRefs[cascade],Usage::ShaderRead,StageFragment);b.read(dynamicRefs[cascade],Usage::ShaderRead,StageFragment);
                ClearValue clear;clear.depth=0;maps[cascade]=b.createTexture("Composed current shadow depth "+suffix,{Format::Depth32Float,settings.mapResolution,settings.mapResolution});
                maps[cascade]=b.writeDepth(maps[cascade],LoadIntent::Discard,clear);b.setProfileShaders("shadow_cache_composite_vertex,shadow_cache_composite_fragment");
            },[this,cascade](PassContext& ctx){
                auto& slot=slots[frame.slot];auto* t=slot.compositeTables[cascade];t->setAddress(slot.staticParams[cascade],18);
                t->setAddress(slot.expectedAddress[cascade],20);t->setAddress(views[frame.view].cacheTiles[cascade]->gpuAddress(),21);t->setAddress(slot.counters->gpuAddress(),15);
                texture(t,ctx,cacheRefs[cascade],0);texture(t,ctx,currentStaticRefs[cascade],1);texture(t,ctx,dynamicRefs[cascade],2);
                auto* e=static_cast<MTL4::RenderCommandEncoder*>(ctx.encoder());e->setDepthStencilState(compositeDepthState);
                e->setRenderPipelineState(pipelines.render(cacheComposite));e->setArgumentTable(t,MTL::RenderStageVertex|MTL::RenderStageFragment);
                e->drawPrimitives(MTL::PrimitiveTypeTriangle,0,3);e->setDepthStencilState(scene.depthState());
            });
        }
    }
    void add(rg::RenderGraph& g,rg::TextureRef v,rg::TextureRef d) {
        using namespace rg;visibilityRef=v;depthRef=d;auto& f=slots[frame.slot];poseRef=visibility.importPoseHistory(g);
        // These compute-written guides also become raster MRTs. Match the
        // physical depth attachment, not the smaller active DRS rectangle.
        // Buffer indexing, viewport, dispatch and readback keep frame.width/height.
        const auto receiverDepth=g.resources().at(depthRef.resource).texture;
        if(receiverDepth.width<frame.width||receiverDepth.height<frame.height)
            throw std::logic_error("Lighting receiver active extent exceeds its depth attachment");
        surfaceRef=g.importBuffer("Pre-resolve shadow surface",{capacity*sizeof(GPUShadowSurface)},ImportPerFrame);
        flagsRef=g.importBuffer("Independent CSM caster flags",{std::max<u64>(1,casterCapacity)*4},ImportPerFrame);
        counterRef=g.importBuffer("Shadow counters",{sizeof(GPUShadowCounters)},ImportPerFrame|ImportOutput);
        g.addPass("Shadow receiver clear",PassType::Compute,[&](PassBuilder& b){
            counterRef=b.write(counterRef,Usage::ShaderWrite,StageDispatch);keyRef=b.createTexture("Indexed receiver keys",{Format::RGBA32Uint,receiverDepth.width,receiverDepth.height});
            keyRef=b.write(keyRef,Usage::ShaderWrite,StageDispatch);b.setProfileShaders("shadow_receiver_clear");
        },[this](PassContext& ctx){auto* t=slots[frame.slot].tables[Clear];common(t,ctx);texture(t,ctx,keyRef,10);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),pipelines,kernels[Clear],t,frame.width,frame.height);});
        g.addPass("Pre-resolve geometric guides",PassType::Compute,[&](PassBuilder& b){
            b.read(visibilityRef,Usage::ShaderRead,StageDispatch);b.read(depthRef,Usage::ShaderRead,StageDispatch);b.read(scene.dataRef(),Usage::ShaderRead,StageDispatch);b.read(mesh.frameListsRef(),Usage::ShaderRead,StageDispatch);
            b.read(counterRef,Usage::ShaderRead,StageDispatch);counterRef=b.write(counterRef,Usage::ShaderWrite,StageDispatch);
            surfaceRef=b.write(surfaceRef,Usage::ShaderWrite,StageDispatch);
            pointRef=b.createTexture("Receiver world position",{Format::RGBA32Float,receiverDepth.width,receiverDepth.height});pointRef=b.write(pointRef,Usage::ShaderWrite,StageDispatch);
            normalRef=b.createTexture("Receiver geometric normal",{Format::RGBA16Float,receiverDepth.width,receiverDepth.height});normalRef=b.write(normalRef,Usage::ShaderWrite,StageDispatch);
            b.setProfileShaders("shadow_surface_guides");
        },[this](PassContext& ctx){auto* t=slots[frame.slot].tables[Guide];common(t,ctx);texture(t,ctx,visibilityRef,ST_VISIBILITY);texture(t,ctx,depthRef,ST_DEPTH);texture(t,ctx,pointRef,ST_WORLD_POSITION);texture(t,ctx,normalRef,ST_GEOMETRIC_NORMAL);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),pipelines,kernels[Guide],t,frame.width,frame.height);});
        g.addPass("Indexed overflow receiver prepass",PassType::Raster,[&](PassBuilder& b){
            pointRef=b.writeColor(pointRef,0,LoadIntent::Preserve);normalRef=b.writeColor(normalRef,1,LoadIntent::Preserve);keyRef=b.writeColor(keyRef,2,LoadIntent::Preserve);
            depthRef=b.writeDepth(depthRef,LoadIntent::Preserve);scene.declareDrawReads(b);b.read(poseRef,Usage::ShaderRead,StageVertex);b.setProfileShaders("forward_surface_vs,forward_shadow_receiver_fs");
        },[this](PassContext& ctx){scene.encodeOverlay(static_cast<MTL4::RenderCommandEncoder*>(ctx.encoder()),receiver,true,receiverDepthState);});
        g.addPass("Indexed receiver pack",PassType::Compute,[&](PassBuilder& b){
            for(auto r:{pointRef,normalRef,keyRef,depthRef})b.read(r,Usage::ShaderRead,StageDispatch);b.read(surfaceRef,Usage::ShaderRead,StageDispatch);surfaceRef=b.write(surfaceRef,Usage::ShaderWrite,StageDispatch);
            // scene.dataRef also versions vertex updates; preserve untouched
            // V-buffer pixels while publishing canonical indexed guides.
            b.read(scene.dataRef(),Usage::ShaderRead,StageDispatch);
            pointRef=b.write(pointRef,Usage::ShaderWrite,StageDispatch);normalRef=b.write(normalRef,Usage::ShaderWrite,StageDispatch);
            b.read(counterRef,Usage::ShaderRead,StageDispatch);counterRef=b.write(counterRef,Usage::ShaderWrite,StageDispatch);
            b.setProfileShaders("shadow_receiver_pack");
        },[this](PassContext& ctx){auto* t=slots[frame.slot].tables[Pack];common(t,ctx);
            // Entry-local buffer18: original mesh-local index stream. Slots
            //3/5/6/15 are instances/meshes/vertices/counters from common().
            t->setAddress(scene.indexBuffer()->gpuAddress(),18);texture(t,ctx,pointRef,5);texture(t,ctx,normalRef,6);texture(t,ctx,keyRef,10);texture(t,ctx,depthRef,0);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),pipelines,kernels[Pack],t,frame.width,frame.height);});
        g.addPass("CSM caster selection",PassType::Compute,[&](PassBuilder& b){
            b.read(scene.dataRef(),Usage::ShaderRead,StageDispatch);flagsRef=b.write(flagsRef,Usage::ShaderWrite,StageDispatch);
            b.read(counterRef,Usage::ShaderRead,StageDispatch);counterRef=b.write(counterRef,Usage::ShaderWrite,StageDispatch);b.setProfileShaders("shadow_caster_flags");
        },[this](PassContext& ctx){auto* t=slots[frame.slot].tables[Cull];common(t,ctx);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),pipelines,kernels[Cull],t,std::max(1u,params.slotCount));});
        if(settings.staticCache)addCacheGraph(g);
        else for(u32 cascade=0;cascade<4;++cascade)g.addPass("CSM cascade "+std::to_string(cascade),PassType::Raster,[&,cascade](PassBuilder& b){
            ClearValue clear;clear.depth=0;maps[cascade]=b.createTexture("CSM depth "+std::to_string(cascade),{Format::Depth32Float,settings.mapResolution,settings.mapResolution});maps[cascade]=b.writeDepth(maps[cascade],LoadIntent::Clear,clear);
            b.read(scene.dataRef(),Usage::ShaderRead,StageVertex|StageMesh|StageFragment);b.read(flagsRef,Usage::ShaderRead,StageVertex|StageMesh);b.setProfileShaders("shadow_depth_mesh,shadow_depth_vertex,shadow_depth_fragment");
        },[this,cascade](PassContext& ctx){drawCascade(ctx,cascade,0,false);});
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
            b.read(counterRef,Usage::ShaderRead,StageDispatch);counterRef=b.write(counterRef,Usage::ShaderWrite,StageDispatch);temporalOutput=b.createTexture("Sun temporal visibility",{Format::RGBA16Float,frame.width,frame.height});temporalOutput=b.write(temporalOutput,Usage::ShaderWrite,StageDispatch);b.setProfileShaders("shadow_temporal");
        },[this,raw](PassContext& ctx){auto* t=slots[frame.slot].tables[Temporal];common(t,ctx);auto& view=views[frame.view];t->setAddress(view.previous->gpuAddress(),12);t->setAddress(view.next->gpuAddress(),13);texture(t,ctx,raw,3);texture(t,ctx,temporalOutput,4);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),pipelines,kernels[Temporal],t,frame.width,frame.height);history.read(frame.view,frame.index+1);history.write(frame.view,frame.index+1,params.viewProjection);});
        auto temporalMask=temporalOutput;
        g.addPass("Solar spatial filter",PassType::Compute,[&](PassBuilder& b){b.read(surfaceRef,Usage::ShaderRead,StageDispatch);b.read(temporalMask,Usage::ShaderRead,StageDispatch);b.read(nextRef,Usage::ShaderRead,StageDispatch);filterOutput=b.createTexture("Sun filtered visibility",{Format::RGBA16Float,frame.width,frame.height});filterOutput=b.write(filterOutput,Usage::ShaderWrite,StageDispatch);b.setProfileShaders("shadow_filter");
        },[this,temporalMask](PassContext& ctx){auto* t=slots[frame.slot].tables[Filter];common(t,ctx);t->setAddress(views[frame.view].next->gpuAddress(),13);texture(t,ctx,temporalMask,3);texture(t,ctx,filterOutput,4);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),pipelines,kernels[Filter],t,frame.width,frame.height);});
        maskRef=filterOutput;
        if(settings.contact){auto primary=filterOutput;g.addPass("Solar contact detail",PassType::Compute,[&](PassBuilder& b){b.read(surfaceRef,Usage::ShaderRead,StageDispatch);b.read(depthRef,Usage::ShaderRead,StageDispatch);b.read(primary,Usage::ShaderRead,StageDispatch);b.read(counterRef,Usage::ShaderRead,StageDispatch);counterRef=b.write(counterRef,Usage::ShaderWrite,StageDispatch);contactOutput=b.createTexture("Sun visibility with contact",{Format::RGBA16Float,frame.width,frame.height});contactOutput=b.write(contactOutput,Usage::ShaderWrite,StageDispatch);b.setProfileShaders("shadow_contact");},[this,primary](PassContext& ctx){auto* t=slots[frame.slot].tables[Contact];common(t,ctx);texture(t,ctx,primary,3);texture(t,ctx,contactOutput,4);texture(t,ctx,depthRef,0);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),pipelines,kernels[Contact],t,frame.width,frame.height);});}
        if(settings.contact)maskRef=contactOutput;
        g.addPass("Lighting zero fallback",PassType::Compute,[&](PassBuilder& b){zeroRef=b.createTexture("Zero direct and indirect fallback",{Format::RGBA16Float,frame.width,frame.height});zeroRef=b.write(zeroRef,Usage::ShaderWrite,StageDispatch);b.setProfileShaders("lighting_zero");},[this](PassContext& ctx){auto* t=slots[frame.slot].receiverTable;texture(t,ctx,zeroRef,0);auto* e=static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder());lighting::dispatch(e,pipelines,zeroPipeline,t,frame.width,frame.height);});
    }
    pipe::PipelineHandle zeroPipeline=pipe::INVALID_PIPELINE;
    void bind(MetalGraphExecutor& e) {
        auto& f=slots[frame.slot];auto& v=views[frame.view];
        if(settings.staticCache){e.bindBuffer(classificationRef,f.classificationBuffer);for(u32 i=0;i<4;++i){e.bindTexture(cacheRefs[i],v.cached[i]);e.bindBuffer(tileRefs[i],v.cacheTiles[i]);e.bindBuffer(expectedRefs[i],f.expectedBuffer[i]);}}
        e.bindBuffer(surfaceRef,f.surfaces);e.bindBuffer(flagsRef,f.flags);e.bindBuffer(counterRef,f.counters);e.bindBuffer(previousRef,v.previous);e.bindBuffer(nextRef,v.next);
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
rg::TextureRef ShadowPasses::depth()const{return impl_->depthRef;}
rg::TextureRef ShadowPasses::zeroLighting()const{return impl_->zeroRef;}
u32 ShadowPasses::sunIndex()const{return impl_->selectedSun;}
u64 ShadowPasses::version()const{return impl_->graphVersion;}
GPUShadowCounters ShadowPasses::counters(u32 slot)const{return *static_cast<const GPUShadowCounters*>(impl_->slots.at(slot).counters->contents());}
ShadowPasses::CacheState ShadowPasses::cacheState(u32 viewID)const {
    CacheState state{};state.enabled=impl_->settings.staticCache;state.updateBudget=impl_->settings.cacheUpdateBudget;
    if(!state.enabled)return state;
    const auto& view=impl_->views.at(viewID);
    for(u32 c=0;c<4;++c){state.readyTiles+=std::popcount(view.readyMask[c]);state.currentTiles+=std::popcount(view.currentMask[c]);state.updates+=std::popcount(view.updateMask[c]);}
    for(u32 slot=0;slot<impl_->params.slotCount;++slot) {
        const auto& instance=impl_->oldStatic[slot].instance;
        if(!(instance.flags&INSTANCE_FLAG_VALID) || !(instance.flags&2u) || instance.meshIndex>=impl_->geometry.size())continue;
        if(impl_->staticClasses[slot])++state.staticCasters;else ++state.dynamicCasters;
    }
    return state;
}
} // namespace phosphor

namespace phosphor {
bool ShadowPasses::check(u32 slot)const {
    if(!impl_->options.debugLighting)return true;const auto& f=impl_->slots.at(slot);
    if(static_cast<const GPUShadowCounters*>(f.counters->contents())->errors)return false;
    const auto* actual=static_cast<const u32*>(f.flags->contents());
    for(u32 i=0;i<f.expectedCount;++i)if(actual[i]!=f.expectedFlags[i])return false;return true;
}
}

namespace phosphor {ShadowPasses::ReadResources ShadowPasses::readResources()const{return {impl_->maps,impl_->paramsAddress,impl_->params};}}
