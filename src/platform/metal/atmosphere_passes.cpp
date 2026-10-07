#include "platform/metal/atmosphere_passes.h"
#include "platform/metal/lighting_dispatch.h"
#include "platform/metal/scene_renderer.h"
#include "platform/metal/direct_lighting_passes.h"
#include "platform/metal/acceleration_structures.h"
#include "platform/metal/gi_passes.h"
#include "platform/metal/metal_graph_executor.h"
#include "platform/metal/volume_diagnostics.h"
#include "renderer/atmosphere.h"
#include "renderer/fog_settings.h"
#include "renderer/cloud_settings.h"
#include "renderer/atmosphere_bindings.h"
#include "renderer/history_registry.h"
#include "rendergraph/pass_context.h"
#include "core/log.h"
#include <glm/gtc/type_ptr.hpp>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace phosphor {
namespace {
MTL::Buffer* volumeBuffer(MetalContext& c,u64 bytes,const char* label,bool shared=false) {
    auto* result=c.memory().newBuffer(std::max<u64>(bytes,16),shared?MTL::ResourceStorageModeShared:MTL::ResourceStorageModePrivate,
                                       MemoryCategory::RenderTargets,label);
    if(!result)throw std::runtime_error("F14 volume buffer allocation failed");return result;
}
MTL::Texture* lutTexture(MetalContext& c,u32 w,u32 h,const char* label) {
    auto* descriptor=MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatRGBA32Float,w,h,false);
    descriptor->setStorageMode(MTL::StorageModePrivate);descriptor->setUsage(MTL::TextureUsageShaderRead|MTL::TextureUsageShaderWrite);
    auto* result=c.memory().newTexture(descriptor,MemoryCategory::RenderTargets,label);
    if(!result)throw std::runtime_error("F14 LUT allocation failed");return result;
}
void copy3(float* target,glm::dvec3 value){target[0]=float(value.x);target[1]=float(value.y);target[2]=float(value.z);}
void dispatch3D(MTL4::ComputeCommandEncoder* encoder,PipelineCache& pipelines,pipe::PipelineHandle handle,MTL4::ArgumentTable* table,
                u32 x,u32 y,u32 z) {
    auto* state=pipelines.compute(handle);if(!state||!x||!y||!z)throw std::logic_error("F14 dispatch missing ready pipeline or work");
    encoder->setComputePipelineState(state);encoder->setArgumentTable(table);
    encoder->dispatchThreads(MTL::Size::Make(x,y,z),MTL::Size::Make(std::min(x,4u),std::min(y,4u),std::min(z,4u)));
}
void key(std::vector<u64>& out,double value){out.push_back(std::bit_cast<u64>(value));}
void key(std::vector<u64>& out,glm::dvec3 value){key(out,value.x);key(out,value.y);key(out,value.z);}
}
struct AtmospherePasses::Impl {
    enum Pass { Clear,Transmittance,Multiscattering,Sky,AtmosphereApply,FogInject,FogTemporal,FogIntegrate,FogApply,
                CloudMarch,CloudTemporal,CloudApply,FogSnapshot,CloudSnapshot,Count };
    MetalContext& c;PipelineCache& pipelines;SceneRenderer& scene;
    DirectLightingPasses* direct;AccelerationStructures* rt;GiPasses* gi;ShadowPasses* shadows;
    LaunchOptions options;
    AtmosphereSettings atmosphere;FogSettings fog;CloudSettings clouds;DayNightSettings clockSettings;DayNightClock clock;
    DayNightState celestial;bool lightingPrepared=false;u64 lightingCalls=0;
    std::array<pipe::PipelineHandle,Count> handles{};pipe::PipelineHandle fogRtHandle=pipe::INVALID_PIPELINE;
    std::unique_ptr<RtConsumer> fogConsumer;
    std::unique_ptr<VolumeDiagnostics> diagnostics;
    bool diagnosticSelected=false,previousDiagnosticSelected=false,diagnosticArmed=false,hadFogHistory=false,hadCloudHistory=false;
    MTL::Texture *transmittance=nullptr,*multiscattering=nullptr;
    MTL::Buffer *dummyState=nullptr,*dummyLight=nullptr,*dummyEmitter=nullptr;
    HistoryRegistry fogHistory,cloudHistory;
    AtmosphereVersions physicalVersions;AtmosphereUpdate physicalUpdate;
    struct View {
        MTL::Texture* sky=nullptr;
        MTL::Buffer *fog=nullptr,*cloud=nullptr;
        AtmosphereVersions skyVersions;
        u32 pipelineGeneration=~u32(0);
        bool skyReady=false;
        std::array<float,16> fogPreviousView{};
        double cloudPreviousTime=0;
    };
    std::array<View,HistoryRegistry::MaxViews> views{};
    struct Slot {
        MTL::Buffer *counter=nullptr,*fogFresh=nullptr,*fogFiltered=nullptr,*fogIntegrated=nullptr,*cloudFresh=nullptr,*cloudFiltered=nullptr,*alias=nullptr;
        std::array<MTL4::ArgumentTable*,Count> tables{};
        u64 recordedIndex=0;bool recorded=false;
    };
    std::array<Slot,METAL_FRAMES_IN_FLIGHT> slots{};
    ShadowPasses::Frame frame;
    GPUAtmosphereParams atmosphereParams{};GPUFogParams fogParams{};GPUCloudParams cloudParams{};
    MTL::GPUAddress atmosphereAddress=0,fogAddress=0,cloudAddress=0,dummyProbeAddress=0,dummyShadowAddress=0;
    GPUProbeGridParams dummyProbes{};GPUShadowParams dummyShadows{};
    u64 cloudCapacity=0,aliasCapacity=0,fogCells=0,graphVersion=1;
    u32 cloudWidth=0,cloudHeight=0,lutMask=0,previousMask=~0u,lastPipelineGeneration=~0u;
    u32 lastWidth=0,lastHeight=0,lastView=~0u;
    bool fogUsesRt=false,previousFogRt=false,active=false,transmittanceReady=false,multiscatteringReady=false;
    struct VolumeSignal {
        u64 scene=0,geometry=0,material=0,clock=0,environment=0;
        u32 localLight=0,pipeline=0,nearBits=0,farBits=0;
        bool rayVisibility=false;
        bool operator==(const VolumeSignal&)const=default;
    };
    VolumeSignal previousSignal{~u64(0),0,0,0,0,0,0,0,0,false};u64 volumeEpoch=1;
    GiEnvironment environmentValue;std::vector<u64> environmentKey;u64 environmentEpoch=0;
    rg::TextureRef transRef{},multiRef{},skyRef{},sourceRef{},depthRef{},atmosphereOutput{},fogOutput{},cloudRaw{},cloudGuide{},cloudFiltered{},cloudOutput{},outputRef{};
    rg::BufferRef counterRef{},fogFreshRef{},fogFilteredRef{},fogIntegratedRef{},fogHistoryRef{},cloudFreshRef{},cloudFilteredRef{},cloudHistoryRef{},aliasRef{},dummyStateRef{},dummyLightRef{},dummyEmitterRef{};

    static DayNightSettings celestialSettings(const LaunchOptions& o) {
        DayNightSettings settings;settings.dayLengthSeconds=o.atmoDayLength;settings.startDayFraction=double(o.atmoStartHour)/24;return settings;
    }
    Impl(MetalContext& context,PipelineCache& p,SceneRenderer& s,DirectLightingPasses* d,AccelerationStructures* a,GiPasses* g,ShadowPasses* sh,const LaunchOptions& o)
        :c(context),pipelines(p),scene(s),direct(d),rt(a),gi(g),shadows(sh),options(o),clockSettings(celestialSettings(o)),clock(clockSettings) {
        active=o.atmosphere||o.fog||o.clouds;
        if(o.cloudFullRate&&!o.clouds)throw std::invalid_argument("Cloud full-rate reference requires clouds enabled");
        if(o.forceApple9){atmosphere.skyWidth=128;atmosphere.skyHeight=72;atmosphere.marchSteps=48;atmosphere.multiDirections=32;fog.gridX=32;fog.gridY=18;}
        validateAtmosphere(atmosphere);validateFog(fog);validateClouds(clouds);
        const char* names[]={"volume_clear_counters","atmosphere_transmittance","atmosphere_multiscattering","atmosphere_sky_view","atmosphere_apply",
                             "fog_inject","fog_temporal","fog_integrate","fog_apply","clouds_march","clouds_temporal","clouds_apply"};
        if(active)for(u32 i=0;i<FogSnapshot;++i)handles[i]=p.request(lighting::kernel(names[i]));
        if(o.fog&&a){fogRtHandle=p.request(lighting::kernel("fog_inject_rt",true));fogConsumer=std::make_unique<RtConsumer>(c,p,*a);}
        if(!o.volumeOracle.empty()||o.debugVolumeCorrupt){
            if(!o.debugLighting)throw std::invalid_argument("Volume oracle requires --debug-lighting N");
            if(o.fogHomogeneous&&(!o.fog||o.volumeOracle.empty()))throw std::invalid_argument("Homogeneous fog requires fog and a volume oracle directory");
            if(o.debugVolumeCorrupt==VOLUME_CORRUPT_HISTORY&&!o.fog&&(!o.clouds||o.cloudFullRate))throw std::invalid_argument("Foreign-history control needs fog or low-rate clouds");
            diagnostics=std::make_unique<VolumeDiagnostics>(c,p,VolumeDiagnostics::Config{o.volumeOracle,o.debugLighting,o.fogHomogeneous,o.debugVolumeCorrupt});
        }
        try {if(active){
            transmittance=lutTexture(c,atmosphere.transmittanceWidth,atmosphere.transmittanceHeight,"Atmosphere transmittance SI LUT");
            multiscattering=lutTexture(c,atmosphere.multiWidth,atmosphere.multiHeight,"Atmosphere isotropic multiple-scattering SI LUT");
            for(auto& v:views)v.sky=lutTexture(c,atmosphere.skyWidth,atmosphere.skyHeight,"Sky view linear radiance per view");
            dummyState=volumeBuffer(c,sizeof(GPUProbeState),"Disabled DDGI state binding",true);std::memset(dummyState->contents(),0,sizeof(GPUProbeState));
            dummyLight=volumeBuffer(c,sizeof(GPUSampledLight),"Disabled local-light binding",true);std::memset(dummyLight->contents(),0,sizeof(GPUSampledLight));
            dummyEmitter=volumeBuffer(c,sizeof(GPUEmissiveSurface),"Disabled emitter binding",true);std::memset(dummyEmitter->contents(),0,sizeof(GPUEmissiveSurface));
            for(auto& slot:slots){for(auto*& table:slot.tables)table=lighting::table(c);slot.counter=volumeBuffer(c,sizeof(GPUVolumeCounters),"F14 volume diagnostics",true);}
        }
        if(o.fog){fogCells=u64(fog.gridX)*fog.gridY*fog.gridZ;
            for(auto& slot:slots){slot.fogFresh=volumeBuffer(c,fogCells*sizeof(GPUFogCell),"Current froxel source");slot.fogFiltered=volumeBuffer(c,fogCells*sizeof(GPUFogCell),"Filtered froxel source");slot.fogIntegrated=volumeBuffer(c,fogCells*sizeof(GPUFogIntegrated),"Integrated front-to-back froxels");}
            for(auto& view:views)view.fog=volumeBuffer(c,fogCells*sizeof(GPUFogCell),"Persistent froxel history per view");}
        }catch(...){releaseOwned();throw;}
    }
    void releaseOwned(){
        for(auto& slot:slots){for(auto* b:{slot.counter,slot.fogFresh,slot.fogFiltered,slot.fogIntegrated,slot.cloudFresh,slot.cloudFiltered,slot.alias})c.memory().release(b,MemoryCategory::RenderTargets);for(auto* t:slot.tables)if(t)t->release();}
        for(auto& view:views){c.memory().release(view.sky,MemoryCategory::RenderTargets);c.memory().release(view.fog,MemoryCategory::RenderTargets);c.memory().release(view.cloud,MemoryCategory::RenderTargets);}
        c.memory().release(transmittance,MemoryCategory::RenderTargets);c.memory().release(multiscattering,MemoryCategory::RenderTargets);
        for(auto* b:{dummyState,dummyLight,dummyEmitter})c.memory().release(b,MemoryCategory::RenderTargets);
    }
    ~Impl(){c.waitIdle();releaseOwned();}
    glm::dvec3 groundReference()const{return atmosphere.planetCenter+glm::dvec3(0,atmosphere.bottomRadius+2,0);}
    void updateEnvironment() {
        // Exact parameter bit tuple. Camera and frame age are deliberately not
        // part of the world radiance domain; no XOR or hash can cancel changes.
        std::vector<u64> tuple;tuple.reserve(48);
        key(tuple,atmosphere.planetCenter);key(tuple,atmosphere.bottomRadius);key(tuple,atmosphere.topRadius);
        key(tuple,atmosphere.rayleighScattering);key(tuple,atmosphere.mieScattering);key(tuple,atmosphere.ozoneAbsorption);
        key(tuple,atmosphere.rayleighScaleHeight);key(tuple,atmosphere.mieScaleHeight);key(tuple,atmosphere.mieAbsorption);key(tuple,atmosphere.mieG);
        key(tuple,atmosphere.ozoneCenterHeight);key(tuple,atmosphere.ozoneHalfWidth);key(tuple,atmosphere.groundAlbedo);
        key(tuple,celestial.sunDirection);key(tuple,celestial.moonDirection);key(tuple,celestial.sunIrradiance);key(tuple,celestial.moonIrradiance);
        key(tuple,clockSettings.starIntensity);key(tuple,clockSettings.sunAngularRadius);tuple.push_back(celestial.epoch);
        tuple.push_back(pipelines.generation());
        if(tuple==environmentKey)return;
        environmentKey=std::move(tuple);++environmentEpoch;
        // Fixed, bounded hemispherical quadrature at ground, independent of the
        // currently rendered camera. This is an explicit isotropic GI adapter.
        glm::dvec3 L(0);const glm::dvec3 point=groundReference();
        const std::array<glm::dvec3,5> rays={glm::dvec3(0,1,0),glm::dvec3(1,0.25,0),glm::dvec3(-1,0.25,0),glm::dvec3(0,0.25,1),glm::dvec3(0,0.25,-1)};
        for(auto ray:rays){ray=glm::normalize(ray);L+=atmosphereSingleScattering(atmosphere,point,ray,celestial.sunDirection,celestial.sunIrradiance,24).radiance;
            L+=atmosphereSingleScattering(atmosphere,point,ray,celestial.moonDirection,celestial.moonIrradiance,24).radiance;}
        L=L/5.0+glm::dvec3(clockSettings.starIntensity);
        for(u32 i=0;i<3;++i)environmentValue.skyRadiance[i]=float(L[i]);
        environmentValue.sunAngularRadius=float(clockSettings.sunAngularRadius);environmentValue.externalRevision=environmentEpoch;
    }
    std::array<GPULight,2> lighting(double seconds,glm::dvec3 camera) {
        if(!std::isfinite(camera.x)||!std::isfinite(camera.y)||!std::isfinite(camera.z))throw std::invalid_argument("F14 camera position is not finite");
        const bool jump=options.timeJumpEveryN&&lightingCalls&&lightingCalls%options.timeJumpEveryN==0;
        ++lightingCalls;celestial=clock.sample(seconds,jump);lightingPrepared=true;updateEnvironment();
        return atmosphereDirectionalLights(atmosphere,celestial,groundReference());
    }
    void reserveCloud(u64 pixels) {
        if(pixels<=cloudCapacity)return;cloudCapacity=pixels;++graphVersion;hadCloudHistory=false;
        // Any replacement invalidates ALL view states, including dormant views.
        for(u32 i=0;i<views.size();++i)cloudHistory.invalidate(i,"Cloud allocation growth");
        for(auto& slot:slots){c.memory().release(slot.cloudFresh,MemoryCategory::RenderTargets);slot.cloudFresh=volumeBuffer(c,pixels*sizeof(GPUCloudHistory),"Current cloud world/depth state");
            if(!options.cloudFullRate){c.memory().release(slot.cloudFiltered,MemoryCategory::RenderTargets);slot.cloudFiltered=volumeBuffer(c,pixels*sizeof(GPUCloudHistory),"Filtered cloud state for snapshot");}}
        if(!options.cloudFullRate)for(auto& view:views){c.memory().release(view.cloud,MemoryCategory::RenderTargets);view.cloud=volumeBuffer(c,pixels*sizeof(GPUCloudHistory),"Persistent cloud history per view");}
    }
    void reserveAlias(u32 count) {
        const u64 capacity=std::max(1u,count);if(capacity<=aliasCapacity)return;aliasCapacity=capacity;++graphVersion;
        for(auto& slot:slots){c.memory().release(slot.alias,MemoryCategory::RenderTargets);slot.alias=volumeBuffer(c,capacity*sizeof(GPUAliasEntry),"Fog complete-list uniform light proposal",true);}
    }
    void prepare(const ShadowPasses::Frame& f,u64 geometry,u64 material) {
        if(!active)return;if(!lightingPrepared)throw std::logic_error("F14 prepareLighting must precede scene/volume preparation");
        if(f.slot>=slots.size()||f.view>=views.size()||!f.width||!f.height||f.index==std::numeric_limits<u64>::max())throw std::invalid_argument("Invalid F14 frame/view or completion timeline");
        auto& slot=slots[f.slot];if(slot.recorded&&c.frameEvent()->signaledValue()<=slot.recordedIndex)throw std::logic_error("F14 frame slot still GPU-owned");
        frame=f;lightingPrepared=false;
        hadFogHistory=fogHistory.get(f.view).valid;hadCloudHistory=cloudHistory.get(f.view).valid;
        const bool hadLuts=transmittanceReady&&multiscatteringReady&&views[f.view].skyReady;
        if(options.fog&&!direct&&f.constants.lightCount>2)throw std::logic_error("Fog local lighting requires the F11 world-light producer even with legacy visible shading");
        if(lastWidth!=f.width||lastHeight!=f.height||lastView!=f.view){lastWidth=f.width;lastHeight=f.height;lastView=f.view;++graphVersion;}
        const u32 pipelineGeneration=pipelines.generation();const bool newPipelines=lastPipelineGeneration!=pipelineGeneration;lastPipelineGeneration=pipelineGeneration;
        if(newPipelines){transmittanceReady=false;multiscatteringReady=false;for(auto& view:views)view.skyReady=false;}
        physicalUpdate=physicalVersions.update(atmosphere,celestial,groundReference(),newPipelines);
        const auto skyUpdate=views[f.view].skyVersions.update(atmosphere,celestial,glm::dvec3(glm::make_vec3(f.constants.cameraPosition)),views[f.view].pipelineGeneration!=pipelineGeneration);
        views[f.view].pipelineGeneration=pipelineGeneration;
        if(physicalUpdate.transmittance)transmittanceReady=false;if(physicalUpdate.multiscattering)multiscatteringReady=false;if(skyUpdate.skyView)views[f.view].skyReady=false;
        lutMask=(!transmittanceReady?1u:0u)|(!multiscatteringReady?2u:0u)|(!views[f.view].skyReady?4u:0u);
        if(previousMask!=lutMask){previousMask=lutMask;++graphVersion;}
        const glm::mat4 inverse=glm::inverse(glm::make_mat4(f.constants.viewProjection));
        AtmosphereUpdate update=physicalUpdate;update.skyRevision=skyUpdate.skyRevision;
        atmosphereParams=makeAtmosphereParams(atmosphere,clockSettings,celestial,update,glm::dvec3(glm::make_vec3(f.constants.cameraPosition)),
                                               glm::value_ptr(inverse),f.constants.viewProjection,f.width,f.height,u32(f.index),f.view);
        const auto originalAtmosphere=atmosphereParams;
        diagnosticSelected=diagnostics&&(f.index+1)%options.debugLighting==0;diagnosticArmed=false;
        if(diagnostics&&diagnosticSelected){const u32 fault=diagnostics->corruption();
            if(fault==VOLUME_CORRUPT_UNITS){atmosphereParams.corruption=fault;lutMask|=7u;diagnosticArmed=true;}
            if(fault==VOLUME_CORRUPT_LIGHT){for(auto& value:atmosphereParams.sunIrradiance)value*=8;diagnosticArmed=true;}
            if(fault==VOLUME_CORRUPT_OMIT_LUT&&hadLuts){for(auto& value:atmosphereParams.rayleighScattering)value*=1.5f;++atmosphereParams.parameterRevision;++atmosphereParams.skyRevision;lutMask=0;diagnosticArmed=true;}
        }
        if(previousMask!=lutMask){previousMask=lutMask;++graphVersion;}
        if(previousDiagnosticSelected!=diagnosticSelected){previousDiagnosticSelected=diagnosticSelected;++graphVersion;}
        atmosphereAddress=lighting::upload(c,atmosphereParams);dummyProbeAddress=lighting::upload(c,dummyProbes);dummyShadowAddress=lighting::upload(c,dummyShadows);
        fogUsesRt=options.fog&&rt&&rt->active();if(previousFogRt!=fogUsesRt){previousFogRt=fogUsesRt;++graphVersion;}
        const u32 localRevision=direct?direct->lightRevision():0;
        const float fogNear=float(fog.nearDistance);float fogFar=float(fog.farDistance);
        if(!fogUsesRt&&shadows){const float coverage=shadows->readResources().parameters.cascades[3].splitFar;
            if(!(coverage>fogNear))throw std::logic_error("CSM fog requires a prepared positive shadow receiver extent");
            fogFar=std::min(fogFar,std::min(coverage,120.0f));}
        const VolumeSignal signal{f.scene,geometry,material,celestial.epoch,environmentEpoch,localRevision,pipelineGeneration,std::bit_cast<u32>(fogNear),std::bit_cast<u32>(fogFar),fogUsesRt};
        if(previousSignal!=signal){previousSignal=signal;++volumeEpoch;}
        const bool reset=f.cut||f.reset||celestial.reset;
        if(options.fog){
            auto fogDecision=fogHistory.begin(f.view,{fog.gridX,fog.gridY,f.width,f.height},volumeEpoch,reset,reset);
            fogParams={};std::memcpy(fogParams.inverseViewProjection,glm::value_ptr(inverse),64);std::memcpy(fogParams.view,f.constants.view,64);
            std::memcpy(fogParams.previousViewProjection,fogHistory.get(f.view).previousViewProjection.data(),64);std::memcpy(fogParams.previousView,views[f.view].fogPreviousView.data(),64);
            fogParams.gridX=fog.gridX;fogParams.gridY=fog.gridY;fogParams.gridZ=fog.gridZ;fogParams.maxLocalLights=fog.maxLocalLights;
            fogParams.outputWidth=f.width;fogParams.outputHeight=f.height;fogParams.frameIndex=u32(f.index);fogParams.viewID=f.view;
            fogParams.nearDistance=fogNear;fogParams.farDistance=fogFar;
            fogParams.densityAtBase=float(fog.densityAtBase);fogParams.heightBase=float(fog.heightBase);fogParams.heightFalloff=float(fog.heightFalloff);fogParams.maxDensity=float(fog.maxDensity);
            copy3(fogParams.albedo,fog.albedo);fogParams.anisotropy=float(fog.anisotropy);fogParams.historyWeight=float(fog.historyWeight);fogParams.positionThreshold=float(fog.positionThreshold);
            fogParams.maxHistoryAge=fog.maxHistoryAge;fogParams.depthRelativeThreshold=float(fog.depthRelativeThreshold);fogParams.maxTraceDistance=1e6f;
            copy3(fogParams.cameraPosition,glm::dvec3(glm::make_vec3(f.constants.cameraPosition)));fogParams.timeSeconds=float(celestial.seconds);
            copy3(fogParams.sunDirection,celestial.sunDirection);copy3(fogParams.sunIrradiance,celestial.sunIrradiance);
            copy3(fogParams.moonDirection,celestial.moonDirection);copy3(fogParams.moonIrradiance,celestial.moonIrradiance);
            fogParams.slotCount=scene.buffers().slotCapacity();fogParams.lightCount=direct?direct->lightCount():0;fogParams.lightRevision=localRevision;
            fogParams.generation=u32(fogDecision.generation);fogParams.previousViewID=f.view;
            fogParams.flags=(!fogDecision.reset?VOLUME_HISTORY_VALID:0u)|(fogUsesRt?VOLUME_SHADOW_RT:shadows?VOLUME_SHADOW_CSM:0u)|
                             (gi?VOLUME_ENABLE_GI:0u)|(fogParams.lightCount?VOLUME_ENABLE_LOCAL_LIGHTS:0u);
            if(diagnostics&&diagnostics->homogeneousFog()){fogParams.heightFalloff=0;fogParams.densityAtBase=0.01f;fogParams.flags&=~VOLUME_HISTORY_VALID;}
            if(diagnostics&&diagnosticSelected&&diagnostics->corruption()==VOLUME_CORRUPT_HISTORY&&hadFogHistory){fogParams.corruption=VOLUME_CORRUPT_HISTORY;fogParams.flags|=VOLUME_HISTORY_VALID;diagnosticArmed=true;}
            fogAddress=lighting::upload(c,fogParams);reserveAlias(fogParams.lightCount);
            auto* aliases=static_cast<GPUAliasEntry*>(slot.alias->contents());const u32 count=std::max(fogParams.lightCount,1u);
            for(u32 i=0;i<count;++i)aliases[i]={1.0f,1.0f/float(count),i,i};
            if(fogUsesRt)fogConsumer->prepare(f.slot,fogRtHandle);
        }
        if(options.clouds){
            const u32 divisor=options.cloudFullRate?1u:options.forceApple9?4u:2u;cloudWidth=(f.width+divisor-1)/divisor;cloudHeight=(f.height+divisor-1)/divisor;
            const u64 pixels=u64((f.backingWidth+divisor-1)/divisor)*((f.backingHeight+divisor-1)/divisor);reserveCloud(std::max(pixels,u64(cloudWidth)*cloudHeight));
            auto decision=cloudHistory.begin(f.view,{cloudWidth,cloudHeight,f.width,f.height},volumeEpoch,reset,reset||options.cloudFullRate);
            cloudParams={};std::memcpy(cloudParams.inverseViewProjection,glm::value_ptr(inverse),64);std::memcpy(cloudParams.previousViewProjection,cloudHistory.get(f.view).previousViewProjection.data(),64);
            copy3(cloudParams.cameraPosition,glm::dvec3(glm::make_vec3(f.constants.cameraPosition)));copy3(cloudParams.wind,clouds.wind);
            cloudParams.timeSeconds=float(celestial.seconds);cloudParams.previousTimeSeconds=float(views[f.view].cloudPreviousTime);
            cloudParams.baseHeight=float(clouds.baseHeight);cloudParams.topHeight=float(clouds.topHeight);cloudParams.coverage=float(clouds.coverage);cloudParams.densityScale=float(clouds.densityScale);
            cloudParams.noiseScale=float(clouds.noiseScale);cloudParams.erosionScale=float(clouds.erosionScale);cloudParams.extinction=float(clouds.extinction);cloudParams.albedo=float(clouds.albedo);
            cloudParams.anisotropy=float(clouds.anisotropy);cloudParams.maxDistance=float(clouds.maxDistance);cloudParams.terminationTransmittance=float(clouds.terminationTransmittance);
            cloudParams.historyWeight=options.cloudFullRate?0:float(clouds.historyWeight);cloudParams.maxHistorySamples=clouds.maxHistorySamples;
            cloudParams.depthRelativeThreshold=float(clouds.depthRelativeThreshold);cloudParams.positionThreshold=float(clouds.positionThreshold);cloudParams.lightStepDistance=float(clouds.lightStepDistance);
            cloudParams.width=cloudWidth;cloudParams.height=cloudHeight;cloudParams.outputWidth=f.width;cloudParams.outputHeight=f.height;
            cloudParams.marchSteps=clouds.marchSteps;cloudParams.lightSteps=clouds.lightSteps;cloudParams.seed=clouds.seed;cloudParams.generation=u32(decision.generation);
            cloudParams.frameIndex=u32(f.index);cloudParams.viewID=f.view;cloudParams.flags=!options.cloudFullRate&&!decision.reset?VOLUME_HISTORY_VALID:0u;
            if(options.atmosphere)cloudParams.flags|=CLOUD_SCENE_HAS_ATMOSPHERE;
            if(diagnostics&&diagnosticSelected&&diagnostics->corruption()==VOLUME_CORRUPT_HISTORY&&hadCloudHistory&&!options.cloudFullRate){cloudParams.corruption=VOLUME_CORRUPT_HISTORY;cloudParams.flags|=VOLUME_HISTORY_VALID;diagnosticArmed=true;}
            cloudAddress=lighting::upload(c,cloudParams);
        }
        if(diagnostics){GPUAtmosphereParams expected=originalAtmosphere;
            if(diagnosticSelected&&diagnostics->corruption()==VOLUME_CORRUPT_OMIT_LUT&&diagnosticArmed)expected=atmosphereParams;
            diagnostics->prepare(f.slot,f.index,f.view,expected,atmosphereParams,fogParams,diagnosticArmed);}
        if(f.index==0)LOG_INFO("F14 SI: atmosphere=%u fog=%u range=%.1fm shadow=%s locals=%u GI=%u clouds=%u full-reference=%u physical-HDR=RGBA32Float",
                              unsigned(options.atmosphere),unsigned(options.fog),double(fogFar),fogUsesRt?"RT-own-IFT":shadows?"CSM-bounded120m":"unshadowed",
                              fogParams.lightCount,unsigned(gi!=nullptr),unsigned(options.clouds),unsigned(options.cloudFullRate));
    }
    void texture(MTL4::ArgumentTable* table,rg::PassContext& ctx,rg::TextureRef ref,u32 index) {
        if(!ref.valid())throw std::logic_error("F14 texture dependency is missing");
        table->setTexture(static_cast<MTL::Texture*>(ctx.texture(ref))->gpuResourceID(),index);
    }
    void atmosphereBindings(MTL4::ArgumentTable* table){table->setAddress(atmosphereAddress,0);table->setAddress(slots[frame.slot].counter->gpuAddress(),15);}
    void physicalSource(rg::PassContext& ctx,rg::TextureRef input)const {
        const auto* texture=static_cast<MTL::Texture*>(ctx.texture(input));
        if(!texture||texture->width()<frame.width||texture->height()<frame.height||
           (texture->pixelFormat()!=MTL::PixelFormatRGBA16Float&&texture->pixelFormat()!=MTL::PixelFormatRGBA32Float))
            throw std::invalid_argument("F14 input must be matching linear floating HDR before exposure/upscale/UI");
    }
    void fogBindings(MTL4::ArgumentTable* table,rg::PassContext& ctx) {
        auto& slot=slots[frame.slot];table->setAddress(fogAddress,0);table->setAddress(slot.counter->gpuAddress(),15);table->setAddress(atmosphereAddress,14);
        const auto g=gi?gi->readResources():GiPasses::ReadResources{};
        table->setAddress(g.params?g.params:dummyProbeAddress,3);table->setAddress(g.states?g.states->gpuAddress():dummyState->gpuAddress(),4);
        texture(table,ctx,g.irradiance.valid()?g.irradiance:transRef,0);texture(table,ctx,g.moments.valid()?g.moments:transRef,1);
        table->setAddress(direct?direct->lightsBuffer()->gpuAddress():dummyLight->gpuAddress(),5);table->setAddress(slot.alias->gpuAddress(),6);
        table->setAddress(direct?direct->emittersBuffer()->gpuAddress():dummyEmitter->gpuAddress(),7);
        table->setAddress(scene.buffers().materials()->gpuAddress(),8);table->setAddress(scene.textureTableAddress(),9);
        table->setAddress(scene.buffers().instances()->gpuAddress(),10);texture(table,ctx,transRef,12);
        if(fogUsesRt)fogConsumer->bind(table,11,12);
        else {const auto s=shadows?shadows->readResources():ShadowPasses::ReadResources{};table->setAddress(s.params?s.params:dummyShadowAddress,13);
            for(u32 i=0;i<4;++i)texture(table,ctx,s.maps[i].valid()?s.maps[i]:depthRef,8+i);}
    }
    void cloudBindings(MTL4::ArgumentTable* table){table->setAddress(cloudAddress,0);table->setAddress(atmosphereAddress,1);table->setAddress(slots[frame.slot].counter->gpuAddress(),15);}
    void counterAccess(rg::PassBuilder& b){b.read(counterRef,rg::Usage::ShaderRead,rg::StageDispatch);counterRef=b.write(counterRef,rg::Usage::ShaderWrite,rg::StageDispatch);}
    void fogInputAccess(rg::PassBuilder& b) {
        using namespace rg;b.read(transRef,Usage::ShaderRead,StageDispatch);b.read(scene.dataRef(),Usage::ShaderRead,StageDispatch);b.read(aliasRef,Usage::ShaderRead,StageDispatch);
        if(direct){b.read(direct->lightsRef(),Usage::ShaderRead,StageDispatch);b.read(direct->emittersRef(),Usage::ShaderRead,StageDispatch);}else{b.read(dummyLightRef,Usage::ShaderRead,StageDispatch);b.read(dummyEmitterRef,Usage::ShaderRead,StageDispatch);}
        const auto g=gi?gi->readResources():GiPasses::ReadResources{};
        if(g.stateRef.valid())b.read(g.stateRef,Usage::ShaderRead,StageDispatch);else b.read(dummyStateRef,Usage::ShaderRead,StageDispatch);
        if(g.irradiance.valid())b.read(g.irradiance,Usage::ShaderRead,StageDispatch);if(g.moments.valid())b.read(g.moments,Usage::ShaderRead,StageDispatch);
        if(fogUsesRt)rt->declareTraceReads(b);else if(shadows){for(auto r:shadows->readResources().maps)b.read(r,Usage::ShaderRead,StageDispatch);}else b.read(depthRef,Usage::ShaderRead,StageDispatch);
    }
    rg::TextureRef add(rg::RenderGraph& graph,rg::TextureRef hdr,rg::TextureRef depth) {
        using namespace rg;if(!active)return hdr;sourceRef=hdr;depthRef=depth;
        if(!hdr.valid()||!depth.valid())throw std::invalid_argument("F14 physical composition requires linear HDR and current scene depth");
        if(options.fog&&direct&&(!direct->lightsRef().valid()||!direct->emittersRef().valid()))throw std::logic_error("F14 fog local-light graph producer must be declared first");
        if(options.fog&&gi){const auto r=gi->readResources();if(!r.stateRef.valid()||!r.irradiance.valid()||!r.moments.valid())throw std::logic_error("F14 fog DDGI graph producer must be declared first");}
        counterRef=graph.importBuffer("F14 per-slot diagnostics",{sizeof(GPUVolumeCounters)},ImportPerFrame|ImportOutput);
        transRef=graph.importTexture("Persistent atmosphere transmittance",{Format::RGBA32Float,atmosphere.transmittanceWidth,atmosphere.transmittanceHeight},ImportContentsDefined|ImportOutput);
        multiRef=graph.importTexture("Persistent atmosphere multiscattering",{Format::RGBA32Float,atmosphere.multiWidth,atmosphere.multiHeight},ImportContentsDefined|ImportOutput);
        skyRef=graph.importTexture("Persistent sky view per camera",{Format::RGBA32Float,atmosphere.skyWidth,atmosphere.skyHeight},ImportContentsDefined|ImportOutput);
        if(diagnostics)diagnostics->beginGraph(graph);
        graph.addPass("F14 diagnostics clear",PassType::Compute,[&](PassBuilder& b){counterRef=b.write(counterRef,Usage::ShaderWrite,StageDispatch);},[this](PassContext& ctx){auto& s=slots[frame.slot];auto* t=s.tables[Clear];t->setAddress(s.counter->gpuAddress(),15);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),pipelines,handles[Clear],t,8);s.recorded=true;s.recordedIndex=frame.index;});
        if(lutMask&1u)graph.addPass("Physical atmosphere transmittance LUT",PassType::Compute,[&](PassBuilder& b){transRef=b.write(transRef,Usage::ShaderWrite,StageDispatch);counterAccess(b);b.setProfileShaders("atmosphere_transmittance");},[this](PassContext& ctx){auto* t=slots[frame.slot].tables[Transmittance];atmosphereBindings(t);texture(t,ctx,transRef,5);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),pipelines,handles[Transmittance],t,atmosphere.transmittanceWidth,atmosphere.transmittanceHeight);transmittanceReady=true;});
        if(diagnostics&&(lutMask&1u))diagnostics->stamp(graph,transRef,VOLUME_NUMERIC_TRANS);
        if(lutMask&2u)graph.addPass("Physical atmosphere multiscattering LUT",PassType::Compute,[&](PassBuilder& b){b.read(transRef,Usage::ShaderRead,StageDispatch);multiRef=b.write(multiRef,Usage::ShaderWrite,StageDispatch);counterAccess(b);b.setProfileShaders("atmosphere_multiscattering");},[this](PassContext& ctx){auto* t=slots[frame.slot].tables[Multiscattering];atmosphereBindings(t);texture(t,ctx,transRef,0);texture(t,ctx,multiRef,5);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),pipelines,handles[Multiscattering],t,atmosphere.multiWidth,atmosphere.multiHeight);multiscatteringReady=true;});
        if(diagnostics&&(lutMask&2u))diagnostics->stamp(graph,multiRef,VOLUME_NUMERIC_MULTI);
        if(lutMask&4u)graph.addPass("Physical sky-view LUT",PassType::Compute,[&](PassBuilder& b){b.read(transRef,Usage::ShaderRead,StageDispatch);b.read(multiRef,Usage::ShaderRead,StageDispatch);skyRef=b.write(skyRef,Usage::ShaderWrite,StageDispatch);counterAccess(b);b.setProfileShaders("atmosphere_sky_view");},[this](PassContext& ctx){auto* t=slots[frame.slot].tables[Sky];atmosphereBindings(t);texture(t,ctx,transRef,0);texture(t,ctx,multiRef,1);texture(t,ctx,skyRef,5);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),pipelines,handles[Sky],t,atmosphere.skyWidth,atmosphere.skyHeight);views[frame.view].skyReady=true;});
        if(diagnostics&&(lutMask&4u))diagnostics->stamp(graph,skyRef,VOLUME_NUMERIC_SKY);
        outputRef=sourceRef;
        if(options.atmosphere){const TextureRef input=outputRef;graph.addPass("Physical sky and aerial perspective",PassType::Compute,[&,input](PassBuilder& b){for(auto r:{input,depthRef,transRef,multiRef,skyRef})b.read(r,Usage::ShaderRead,StageDispatch);counterAccess(b);atmosphereOutput=b.createTexture("Physical atmosphere RGBA32 HDR",{Format::RGBA32Float,frame.width,frame.height});atmosphereOutput=b.write(atmosphereOutput,Usage::ShaderWrite,StageDispatch);b.setProfileShaders("atmosphere_apply");},[this,input](PassContext& ctx){physicalSource(ctx,input);auto* t=slots[frame.slot].tables[AtmosphereApply];atmosphereBindings(t);texture(t,ctx,transRef,0);texture(t,ctx,multiRef,1);texture(t,ctx,skyRef,2);texture(t,ctx,input,3);texture(t,ctx,depthRef,4);texture(t,ctx,atmosphereOutput,5);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),pipelines,handles[AtmosphereApply],t,frame.width,frame.height);});outputRef=atmosphereOutput;}
        // Clouds are the farther medium in the declared preset; near-ground
        // fog is then composed front-to-back over cloud+scene radiance.
        if(options.clouds)addClouds(graph);
        if(options.fog)addFog(graph);
        if(diagnostics){VolumeDiagnostics::Sources input;input.transmittance=transRef;input.multiple=multiRef;input.sky=skyRef;input.counters=slots[frame.slot].counter;
            if(options.fog){input.fogCells=fogFilteredRef;input.fogIntegrated=fogIntegratedRef;input.fogCellsBuffer=slots[frame.slot].fogFiltered;input.fogIntegratedBuffer=slots[frame.slot].fogIntegrated;}
            diagnostics->collect(graph,input);}
        return outputRef;
    }
    void addClouds(rg::RenderGraph& graph) {
        using namespace rg;
        cloudFreshRef=graph.importBuffer("Current cloud world/depth tags",{cloudCapacity*sizeof(GPUCloudHistory)},ImportPerFrame|ImportOutput);
        graph.addPass(options.cloudFullRate?"Cloud full-rate reference":"Cloud low-rate ray march",PassType::Compute,[&](PassBuilder& b){for(auto r:{depthRef,transRef,multiRef})b.read(r,Usage::ShaderRead,StageDispatch);counterAccess(b);cloudFreshRef=b.write(cloudFreshRef,Usage::ShaderWrite,StageDispatch);
            cloudRaw=b.createTexture("Cloud raw linear radiance and T",{Format::RGBA32Float,cloudWidth,cloudHeight});cloudRaw=b.write(cloudRaw,Usage::ShaderWrite,StageDispatch);
            cloudGuide=b.createTexture("Cloud centroid and opaque metre depths",{Format::RG32Float,cloudWidth,cloudHeight});cloudGuide=b.write(cloudGuide,Usage::ShaderWrite,StageDispatch);b.setProfileShaders("clouds_march");
        },[this](PassContext& ctx){auto* t=slots[frame.slot].tables[CloudMarch];cloudBindings(t);t->setAddress(slots[frame.slot].cloudFresh->gpuAddress(),2);texture(t,ctx,depthRef,0);texture(t,ctx,transRef,1);texture(t,ctx,multiRef,2);texture(t,ctx,cloudRaw,5);texture(t,ctx,cloudGuide,7);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),pipelines,handles[CloudMarch],t,cloudWidth,cloudHeight);});
        TextureRef cloudInput=cloudRaw;
        if(!options.cloudFullRate){
            cloudFilteredRef=graph.importBuffer("Filtered cloud world/depth tags",{cloudCapacity*sizeof(GPUCloudHistory)},ImportPerFrame);
            cloudHistoryRef=graph.importBuffer("Persistent cloud history for view",{cloudCapacity*sizeof(GPUCloudHistory)},ImportContentsDefined|ImportOutput);
            if(diagnostics&&hadCloudHistory)diagnostics->foreignHistory(graph,cloudHistoryRef,views[frame.view].cloud,u32(u64(cloudWidth)*cloudHeight),true);
            const TextureRef raw=cloudRaw,guide=cloudGuide;
            graph.addPass("Cloud wind-advection temporal reconstruction",PassType::Compute,[&,raw,guide](PassBuilder& b){b.read(raw,Usage::ShaderRead,StageDispatch);b.read(guide,Usage::ShaderRead,StageDispatch);b.read(cloudHistoryRef,Usage::ShaderRead,StageDispatch);cloudFilteredRef=b.write(cloudFilteredRef,Usage::ShaderWrite,StageDispatch);counterAccess(b);cloudFiltered=b.createTexture("Cloud reconstructed radiance and T",{Format::RGBA32Float,cloudWidth,cloudHeight});cloudFiltered=b.write(cloudFiltered,Usage::ShaderWrite,StageDispatch);b.setProfileShaders("clouds_temporal");
            },[this,raw,guide](PassContext& ctx){auto* t=slots[frame.slot].tables[CloudTemporal];cloudBindings(t);t->setAddress(slots[frame.slot].cloudFiltered->gpuAddress(),2);t->setAddress(views[frame.view].cloud->gpuAddress(),3);texture(t,ctx,raw,3);texture(t,ctx,guide,4);texture(t,ctx,cloudFiltered,5);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),pipelines,handles[CloudTemporal],t,cloudWidth,cloudHeight);});
            graph.addPass("Cloud history snapshot",PassType::Blit,[&](PassBuilder& b){b.read(cloudFilteredRef,Usage::CopySrc,StageBlit);cloudHistoryRef=b.write(cloudHistoryRef,Usage::CopyDst,StageBlit);},[this](PassContext& ctx){auto* e=static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder());e->copyFromBuffer(slots[frame.slot].cloudFiltered,0,views[frame.view].cloud,0,u64(cloudWidth)*cloudHeight*sizeof(GPUCloudHistory));cloudHistory.read(frame.view,frame.index+1);cloudHistory.write(frame.view,frame.index+1,frame.constants.viewProjection);views[frame.view].cloudPreviousTime=celestial.seconds;});
            cloudInput=cloudFiltered;
        }
        const TextureRef input=outputRef,cloud=cloudInput,guide=cloudGuide;
        graph.addPass("Cloud depth-aware physical composition",PassType::Compute,[&,input,cloud,guide](PassBuilder& b){for(auto r:{input,cloud,guide,depthRef,transRef,multiRef})b.read(r,Usage::ShaderRead,StageDispatch);counterAccess(b);cloudOutput=b.createTexture("Cloud physical RGBA32 HDR",{Format::RGBA32Float,frame.width,frame.height});cloudOutput=b.write(cloudOutput,Usage::ShaderWrite,StageDispatch);b.setProfileShaders("clouds_apply");},[this,input,cloud,guide](PassContext& ctx){physicalSource(ctx,input);auto* t=slots[frame.slot].tables[CloudApply];cloudBindings(t);texture(t,ctx,depthRef,0);texture(t,ctx,transRef,1);texture(t,ctx,multiRef,2);texture(t,ctx,cloud,3);texture(t,ctx,guide,4);texture(t,ctx,cloudOutput,5);texture(t,ctx,input,6);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),pipelines,handles[CloudApply],t,frame.width,frame.height);});outputRef=cloudOutput;
    }
    void addFog(rg::RenderGraph& graph) {
        using namespace rg;
        fogFreshRef=graph.importBuffer("Current froxel injection",{fogCells*sizeof(GPUFogCell)},ImportPerFrame);
        fogFilteredRef=graph.importBuffer("Filtered froxel injection",{fogCells*sizeof(GPUFogCell)},ImportPerFrame);
        fogIntegratedRef=graph.importBuffer("Integrated froxel volume",{fogCells*sizeof(GPUFogIntegrated)},ImportPerFrame);
        fogHistoryRef=graph.importBuffer("Persistent froxel history for view",{fogCells*sizeof(GPUFogCell)},ImportContentsDefined|ImportOutput);
        aliasRef=graph.importBuffer("Fog complete local-light uniform proposal",{aliasCapacity*sizeof(GPUAliasEntry)},ImportPerFrame|ImportContentsDefined);
        dummyStateRef=graph.importBuffer("Disabled volumetric GI state",{sizeof(GPUProbeState)},ImportContentsDefined);
        dummyLightRef=graph.importBuffer("Disabled volumetric local light",{sizeof(GPUSampledLight)},ImportContentsDefined);
        dummyEmitterRef=graph.importBuffer("Disabled volumetric emitter",{sizeof(GPUEmissiveSurface)},ImportContentsDefined);
        graph.addPass(fogUsesRt?"Froxel source with independent RT visibility":"Froxel source with bounded CSM visibility",PassType::Compute,[&](PassBuilder& b){fogInputAccess(b);counterAccess(b);fogFreshRef=b.write(fogFreshRef,Usage::ShaderWrite,StageDispatch);b.setProfileShaders(fogUsesRt?"fog_inject_rt":"fog_inject");},[this](PassContext& ctx){auto* t=slots[frame.slot].tables[FogInject];fogBindings(t,ctx);t->setAddress(slots[frame.slot].fogFresh->gpuAddress(),1);if(diagnostics&&diagnostics->homogeneousFog()){diagnostics->homogeneous(ctx,slots[frame.slot].fogFresh,fogAddress);return;}dispatch3D(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),pipelines,fogUsesRt?fogRtHandle:handles[FogInject],t,fog.gridX,fog.gridY,fog.gridZ);});
        if(diagnostics&&hadFogHistory)diagnostics->foreignHistory(graph,fogHistoryRef,views[frame.view].fog,u32(fogCells),false);
        graph.addPass("Froxel temporal source",PassType::Compute,[&](PassBuilder& b){b.read(fogFreshRef,Usage::ShaderRead,StageDispatch);b.read(fogHistoryRef,Usage::ShaderRead,StageDispatch);counterAccess(b);fogFilteredRef=b.write(fogFilteredRef,Usage::ShaderWrite,StageDispatch);b.setProfileShaders("fog_temporal");},[this](PassContext& ctx){auto* t=slots[frame.slot].tables[FogTemporal];t->setAddress(fogAddress,0);t->setAddress(slots[frame.slot].fogFiltered->gpuAddress(),1);t->setAddress(slots[frame.slot].fogFresh->gpuAddress(),2);t->setAddress(views[frame.view].fog->gpuAddress(),16);t->setAddress(slots[frame.slot].counter->gpuAddress(),15);dispatch3D(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),pipelines,handles[FogTemporal],t,fog.gridX,fog.gridY,fog.gridZ);});
        graph.addPass("Froxel exponential front-to-back integration",PassType::Compute,[&](PassBuilder& b){b.read(fogFilteredRef,Usage::ShaderRead,StageDispatch);fogIntegratedRef=b.write(fogIntegratedRef,Usage::ShaderWrite,StageDispatch);b.setProfileShaders("fog_integrate");},[this](PassContext& ctx){auto* t=slots[frame.slot].tables[FogIntegrate];t->setAddress(fogAddress,0);t->setAddress(slots[frame.slot].fogIntegrated->gpuAddress(),1);t->setAddress(slots[frame.slot].fogFiltered->gpuAddress(),2);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),pipelines,handles[FogIntegrate],t,fog.gridX,fog.gridY);});
        graph.addPass("Froxel history snapshot",PassType::Blit,[&](PassBuilder& b){b.read(fogFilteredRef,Usage::CopySrc,StageBlit);fogHistoryRef=b.write(fogHistoryRef,Usage::CopyDst,StageBlit);},[this](PassContext& ctx){auto* e=static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder());e->copyFromBuffer(slots[frame.slot].fogFiltered,0,views[frame.view].fog,0,fogCells*sizeof(GPUFogCell));fogHistory.read(frame.view,frame.index+1);fogHistory.write(frame.view,frame.index+1,frame.constants.viewProjection);std::copy_n(frame.constants.view,16,views[frame.view].fogPreviousView.begin());});
        const TextureRef input=outputRef;
        graph.addPass("Near-volume physical fog composition",PassType::Compute,[&,input](PassBuilder& b){b.read(fogFilteredRef,Usage::ShaderRead,StageDispatch);b.read(fogIntegratedRef,Usage::ShaderRead,StageDispatch);b.read(input,Usage::ShaderRead,StageDispatch);b.read(depthRef,Usage::ShaderRead,StageDispatch);fogOutput=b.createTexture("Fog physical RGBA32 HDR",{Format::RGBA32Float,frame.width,frame.height});fogOutput=b.write(fogOutput,Usage::ShaderWrite,StageDispatch);b.setProfileShaders("fog_apply");},[this,input](PassContext& ctx){physicalSource(ctx,input);auto* t=slots[frame.slot].tables[FogApply];t->setAddress(fogAddress,0);t->setAddress(slots[frame.slot].fogIntegrated->gpuAddress(),1);t->setAddress(slots[frame.slot].fogFiltered->gpuAddress(),2);texture(t,ctx,input,2);texture(t,ctx,depthRef,3);texture(t,ctx,fogOutput,4);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),pipelines,handles[FogApply],t,frame.width,frame.height);});outputRef=fogOutput;
    }
    void bind(MetalGraphExecutor& executor) {
        if(!active)return;if(diagnostics)diagnostics->bindFrame(executor);auto& slot=slots[frame.slot];auto& view=views[frame.view];
        executor.bindTexture(transRef,transmittance);executor.bindTexture(multiRef,multiscattering);executor.bindTexture(skyRef,view.sky);executor.bindBuffer(counterRef,slot.counter);
        if(options.clouds){executor.bindBuffer(cloudFreshRef,slot.cloudFresh);if(!options.cloudFullRate){executor.bindBuffer(cloudFilteredRef,slot.cloudFiltered);executor.bindBuffer(cloudHistoryRef,view.cloud);}}
        if(options.fog){executor.bindBuffer(fogFreshRef,slot.fogFresh);executor.bindBuffer(fogFilteredRef,slot.fogFiltered);executor.bindBuffer(fogIntegratedRef,slot.fogIntegrated);executor.bindBuffer(fogHistoryRef,view.fog);executor.bindBuffer(aliasRef,slot.alias);executor.bindBuffer(dummyStateRef,dummyState);executor.bindBuffer(dummyLightRef,dummyLight);executor.bindBuffer(dummyEmitterRef,dummyEmitter);}
    }
    bool checkSlot(u32 index)const {
        const auto& slot=slots.at(index);if(!slot.recorded)return true;
        if(c.frameEvent()->signaledValue()<=slot.recordedIndex)throw std::logic_error("F14 check read before GPU frame completion");
        const auto& counters=*static_cast<const GPUVolumeCounters*>(slot.counter->contents());
        if(counters.nonfinite||counters.invalidUnits||counters.invalidHistory){LOG_ERROR("F14 invalid source: nonfinite=%u units=%u history=%u",counters.nonfinite,counters.invalidUnits,counters.invalidHistory);return false;}
        return true;
    }
};
AtmospherePasses::AtmospherePasses(MetalContext& c,PipelineCache& p,SceneRenderer& s,DirectLightingPasses* d,AccelerationStructures* a,GiPasses* g,ShadowPasses* sh,const LaunchOptions& o)
    :impl_(std::make_unique<Impl>(c,p,s,d,a,g,sh,o)){}
AtmospherePasses::~AtmospherePasses()=default;
std::array<GPULight,2> AtmospherePasses::prepareLighting(double seconds,glm::dvec3 camera){return impl_->lighting(seconds,camera);}
void AtmospherePasses::prepareFrame(const ShadowPasses::Frame& f,u64 geometry,u64 material){impl_->prepare(f,geometry,material);}
rg::TextureRef AtmospherePasses::addToGraph(rg::RenderGraph& g,rg::TextureRef hdr,rg::TextureRef depth){return impl_->add(g,hdr,depth);}
void AtmospherePasses::bindFrame(MetalGraphExecutor& e){impl_->bind(e);}
u64 AtmospherePasses::version()const{return impl_->graphVersion;}
const GiEnvironment& AtmospherePasses::environment()const{return impl_->environmentValue;}
u64 AtmospherePasses::clockEpoch()const{return impl_->celestial.epoch;}
bool AtmospherePasses::clockReset()const{return impl_->celestial.reset;}
float AtmospherePasses::exposureEv100()const{return float(impl_->celestial.exposureEv100);}
AtmospherePasses::ReadResources AtmospherePasses::readResources()const {
    ReadResources out;out.transmittance=impl_->transmittance;out.multiscattering=impl_->multiscattering;out.sky=impl_->views.at(impl_->frame.view).sky;
    out.transmittanceRef=impl_->transRef;out.multiscatteringRef=impl_->multiRef;out.skyRef=impl_->skyRef;out.output=impl_->outputRef;
    out.atmosphere=impl_->atmosphereParams;out.fog=impl_->fogParams;out.clouds=impl_->cloudParams;return out;
}
bool AtmospherePasses::check(u32 slot)const{return impl_->checkSlot(slot);}
bool AtmospherePasses::consumeDiagnostics(u32 slot){return !impl_->diagnostics||impl_->diagnostics->consume(slot);}
} // namespace phosphor
