#include "platform/metal/reflection_passes.h"
#include "platform/metal/denoise_passes.h"
#include "platform/metal/gi_passes.h"
#include "platform/metal/lighting_dispatch.h"
#include "platform/metal/acceleration_structures.h"
#include "platform/metal/scene_renderer.h"
#include "platform/metal/metal_graph_executor.h"
#include "renderer/reflection_settings.h"
#include "renderer/reflection_probe.h"
#include "renderer/scene_store.h"
#include "renderer/gpu_scene.h"
#include "renderer/cull_math.h"
#include "renderer/transform_reference.h"
#include "rendergraph/pass_context.h"
#include <glm/gtc/type_ptr.hpp>
#include <array>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <string>
#include <vector>
namespace phosphor {
namespace {
constexpr u32 ProbeSide=64,ProbeMips=7;
u16 probeHalf(float value){const u32 bits=std::bit_cast<u32>(value);const u32 sign=(bits>>16)&0x8000u;const i32 exponent=i32((bits>>23)&255)-127+15;u32 mantissa=bits&0x7fffffu;
    if(exponent<-10)return u16(sign);if(exponent<=0){mantissa|=0x800000u;const u32 shift=u32(14-exponent),round=(1u<<(shift-1))-1u;return u16(sign|((mantissa+round+((mantissa>>shift)&1u))>>shift));}
    u32 rounded=mantissa+0xfffu+((mantissa>>13)&1u),e=u32(exponent);if(rounded&0x800000u){rounded=0;++e;}if(e>=31)throw std::runtime_error("Probe HDR exceeds active half format");return u16(sign|(e<<10)|(rounded>>13));}
std::vector<float> probePFM(const std::filesystem::path& path){std::ifstream in(path,std::ios::binary);std::string magic;u32 width=0,height=0;float scale=0;
    if(!(in>>magic>>width>>height>>scale)||magic!="PF"||width!=ProbeSide||height!=ProbeSide||!std::isfinite(scale)||scale==0)throw std::runtime_error("Expected 64x64 linear RGB PFM: "+path.string());
    char separator=0;in.get(separator);if(separator=='\r'&&in.peek()=='\n')in.get();else if(separator!='\n'&&separator!=' ')throw std::runtime_error("Invalid PFM separator");
    std::vector<u32> words(size_t(width)*height*3);in.read(reinterpret_cast<char*>(words.data()),std::streamsize(words.size()*4));if(!in||in.peek()!=std::char_traits<char>::eof())throw std::runtime_error("PFM payload size mismatch");
    const bool little=scale<0,hostLittle=std::endian::native==std::endian::little;std::vector<float> result(size_t(width)*height*4);
    for(u32 y=0;y<height;++y)for(u32 x=0;x<width;++x){for(u32 c=0;c<3;++c){u32 bits=words[(size_t(height-1-y)*width+x)*3+c];if(little!=hostLittle)bits=(bits>>24)|((bits>>8)&0xff00)|((bits<<8)&0xff0000)|(bits<<24);
        const float v=std::bit_cast<float>(bits)*std::abs(scale);if(!std::isfinite(v)||v<0)throw std::runtime_error("PFM contains invalid radiance");result[(size_t(y)*width+x)*4+c]=v;}result[(size_t(y)*width+x)*4+3]=1;}
    return result;
}
}
struct ReflectionPasses::Impl {
    MetalContext& c;PipelineCache& p;SceneRenderer& scene;DirectLightingPasses& direct;AccelerationStructures* rt;GiPasses* gi;
    LaunchOptions options;DenoisePasses denoise;ShadowPasses::Frame frame{};u64 graphVersion=1,pixels=0,sceneEpoch=0;u32 probeGeneration=1,lastPipelineGeneration=~0u;
    bool custom=false,probeDirty=true,probeNeedsCapture=false,rtReflection=false,rtAO=false,rtCapture=false,outputHalf=false,probePlacementDirty=true,probePlaced=false;u64 staticStructure=0,lastSignalEpoch=~u64{0};
    std::string source="analytic-environment";
    u64 geometryEpoch=0,instanceRevision=0,nodeRevision=0,materialRevision=0,motionRevision=0;
    struct GeometryContent {u64 scene,structure,rtGeometry,instances,nodes,materials,motion,pipeline;bool operator==(const GeometryContent&)const=default;};
    GeometryContent lastGeometry{};bool geometryKnown=false;
    std::pair<u64,u64> publishedComponents{};u64 publishedVersion=1;
    ReflectionSettings settings;AOSettings aoSettings;GPUReflectionProbe probe{};GPUReflectionParams params{};GPUAOParams aoParams{};
    GPUReflectionReduceParams reduceParams{};GPUReflectionComposeParams composeParams{},preComposeParams{};
    MTL::GPUAddress paramsAddress=0,aoAddress=0,reduceAddress=0,composeAddress=0,preComposeAddress=0,probeAddress=0,giDummyAddress=0,extraDummyAddress=0;
    std::array<MTL::GPUAddress,6> captureAddress{},rasterCaptureAddress{};std::array<MTL::GPUAddress,ProbeMips> filterAddress{},validateAddress{};MTL::GPUAddress rawValidateAddress=0;
    pipe::PipelineHandle reflectionSSR{},reflectionRT{},probeOnly{},captureRT{},captureRaster{},prefilter{},validateProbe{},publishProbe{};
    pipe::PipelineHandle rtao{},gtao{},reduce{},compose{},zero{},clear{},checker{};
    pipe::PipelineHandle inputPoison{},inputCheck{},filteredIndirectExport{};
    std::unique_ptr<RtConsumer> reflectionConsumer,aoConsumer,captureConsumer;
    MTL::DepthStencilState* captureDepthState=nullptr;
    MTL::Texture *rawCube=nullptr,*filteredCube=nullptr,*rawArray=nullptr,*dummyTexture=nullptr;
    std::array<MTL::Texture*,6> faceViews{};std::array<MTL::Texture*,ProbeMips> mipViews{};
    MTL::Buffer *probeMetadata=nullptr,*probeFault=nullptr,*staticSlots=nullptr,*dummyBuffer=nullptr;
    MTL::PixelFormat probeFormat=MTL::PixelFormatRGBA16Float;rg::Format graphProbeFormat=rg::Format::RGBA16Float;
    struct Draw {GPUMeshInfo mesh;u32 staticIndex;CullClass cull;};std::vector<Draw> draws;std::vector<u32> staticIDs;std::vector<GPUMeshInfo> geometry;std::vector<float> probeWorlds;
    struct Slot {
        MTL::Buffer *samples=nullptr,*metadata=nullptr,*errors=nullptr,*testSurfaces=nullptr;bool customUsed=false;u32 expected=0;
        std::array<MTL4::ArgumentTable*,8> sampleTables{};std::array<MTL4::ArgumentTable*,6> captureTables{};
        std::array<MTL4::ArgumentTable*,ProbeMips> filterTables{},validateTables{};
        std::vector<MTL4::ArgumentTable*> drawTables;
        MTL4::ArgumentTable *reduceTable=nullptr,*aoTable=nullptr,*composeTable=nullptr,*preComposeTable=nullptr,*zeroTable=nullptr,*clearTable=nullptr,*checkTable=nullptr,*publishTable=nullptr,*rawValidateTable=nullptr;
        MTL4::ArgumentTable *inputPoisonTable=nullptr,*inputCheckTable=nullptr,*filteredIndirectTable=nullptr;
    };std::array<Slot,METAL_FRAMES_IN_FLIGHT> slots{};
    rg::BufferRef sampleRef{},metadata{},errorsRef{},probeRef{},probeFaultRef{},staticRef{},dummyRef{};
    rg::BufferRef receivers{};rg::TextureRef motion{};
    rg::TextureRef residual{},base{},depth{},specular{},ao{},distance{},specSelected{},aoSelected{},directSelected{},giSelected{},filteredIndirect{},output{},rawRef{},filteredRef{},dummyTexRef{};
    std::array<rg::TextureRef,6> faceRef{};std::array<rg::TextureRef,ProbeMips> mipRef{};
    Impl(MetalContext& context,PipelineCache& pipelines,SceneRenderer& s,DirectLightingPasses& d,AccelerationStructures* a,GiPasses* g,const LaunchOptions& o)
        :c(context),p(pipelines),scene(s),direct(d),rt(a),gi(g),options(o),denoise(c,p,o){
        if(o.reflectionSamples<1||o.reflectionSamples>8||!std::isfinite(o.aoRadius)||o.aoRadius<=0)throw std::invalid_argument("Invalid F13 source sampling preset");
        reflectionSSR=p.request(lighting::kernel("reflection_ssr"));probeOnly=p.request(lighting::kernel("reflection_probe_only"));gtao=p.request(lighting::kernel("ao_gtao"));
        if(rt){reflectionRT=p.request(lighting::kernel("reflection_rt",true));rtao=p.request(lighting::kernel("ao_rtao",true));captureRT=p.request(lighting::kernel("reflection_capture_rt",true));
            reflectionConsumer=std::make_unique<RtConsumer>(c,p,*rt);aoConsumer=std::make_unique<RtConsumer>(c,p,*rt);captureConsumer=std::make_unique<RtConsumer>(c,p,*rt);}
        pipe::PipelineDesc raster;raster.kind=pipe::PipelineKind::Render;raster.label="Static reflection probe scene capture";raster.functions={"reflection_probe_capture_vs","reflection_probe_capture_fs",""};
        if(c.device()->supports32BitFloatFiltering()&&!o.forceApple9&&!o.reducedLighting){probeFormat=MTL::PixelFormatRGBA32Float;graphProbeFormat=rg::Format::RGBA32Float;}
        raster.output(0,graphProbeFormat);captureRaster=p.request(raster);
        prefilter=p.request(lighting::kernel("reflection_probe_prefilter"));validateProbe=p.request(lighting::kernel("reflection_probe_validate"));publishProbe=p.request(lighting::kernel("reflection_probe_ready"));
        reduce=p.request(lighting::kernel("reflection_reduce"));compose=p.request(lighting::kernel("reflection_composite"));zero=p.request(lighting::kernel("reflection_signal_zero"));clear=p.request(lighting::kernel("reflection_counter_clear"));checker=p.request(lighting::kernel("reflection_check"));
        inputCheck=p.request(lighting::kernel("reflection_input_check"));if(o.debugReflectionCorrupt>=2)inputPoison=p.request(lighting::kernel("reflection_input_poison"));
        if(o.captureLinearSignal==8)filteredIndirectExport=p.request(lighting::kernel("reflection_filtered_indirect_diffuse"));
        auto* dd=MTL::DepthStencilDescriptor::alloc()->init();dd->setDepthCompareFunction(MTL::CompareFunctionGreater);dd->setDepthWriteEnabled(true);captureDepthState=c.device()->newDepthStencilState(dd);dd->release();
        if(!captureDepthState)throw std::runtime_error("Probe depth state creation failed");
        dummyBuffer=lighting::buffer(c,256,"F13 safe disabled GI buffer",true);std::memset(dummyBuffer->contents(),0,256);
        dummyTexture=lighting::texture(c,1,1,MTL::PixelFormatRGBA32Float,"F13 disabled GI zero");
        probeMetadata=lighting::buffer(c,sizeof(GPUReflectionProbe),"F13 GPU-published probe metadata");
        probeFault=lighting::buffer(c,16,"F13 persistent probe validation epoch",true);std::memset(probeFault->contents(),0,16);
        for(auto& f:slots){for(auto*& t:f.sampleTables)t=lighting::table(c);for(auto*& t:f.captureTables)t=lighting::table(c);for(auto*& t:f.filterTables)t=lighting::table(c);for(auto*& t:f.validateTables)t=lighting::table(c);
            for(auto** t:{&f.reduceTable,&f.aoTable,&f.composeTable,&f.preComposeTable,&f.zeroTable,&f.clearTable,&f.checkTable,&f.publishTable,&f.rawValidateTable,&f.inputPoisonTable,&f.inputCheckTable})*t=lighting::table(c);
            if(o.captureLinearSignal==8)f.filteredIndirectTable=lighting::table(c);
            f.errors=lighting::buffer(c,64,"F13 numerical state counters",true);std::memset(f.errors->contents(),0,64);}
        aoSettings.radius=o.aoRadius;aoSettings.rays=(o.reducedLighting||o.forceApple9)?2:4;
    }
    ~Impl(){c.waitIdle();releaseProbe();c.memory().release(probeMetadata,MemoryCategory::RayTracing);c.memory().release(probeFault,MemoryCategory::RayTracing);c.memory().release(staticSlots,MemoryCategory::RayTracing);c.memory().release(dummyBuffer,MemoryCategory::RayTracing);c.memory().release(dummyTexture,MemoryCategory::RayTracing);
        for(auto& f:slots){for(auto* b:{f.samples,f.metadata,f.errors,f.testSurfaces})c.memory().release(b,MemoryCategory::RayTracing);
            for(auto* t:f.sampleTables)t->release();for(auto* t:f.captureTables)t->release();for(auto* t:f.filterTables)t->release();for(auto* t:f.validateTables)t->release();for(auto* t:f.drawTables)t->release();
            for(auto* t:{f.reduceTable,f.aoTable,f.composeTable,f.preComposeTable,f.zeroTable,f.clearTable,f.checkTable,f.publishTable,f.rawValidateTable,f.inputPoisonTable,f.inputCheckTable,f.filteredIndirectTable})if(t)t->release();}if(captureDepthState)captureDepthState->release();}
    void releaseProbe(){for(auto*& t:faceViews){c.memory().release(t,MemoryCategory::RayTracing);t=nullptr;}for(auto*& t:mipViews){c.memory().release(t,MemoryCategory::RayTracing);t=nullptr;}
        c.memory().release(rawArray,MemoryCategory::RayTracing);c.memory().release(rawCube,MemoryCategory::RayTracing);c.memory().release(filteredCube,MemoryCategory::RayTracing);rawArray=rawCube=filteredCube=nullptr;}
    MTL::Texture* cube(u32 mips,const char* label){auto* d=MTL::TextureDescriptor::alloc()->init();d->setTextureType(MTL::TextureTypeCubeArray);d->setPixelFormat(probeFormat);d->setWidth(ProbeSide);d->setHeight(ProbeSide);d->setDepth(1);d->setArrayLength(1);d->setMipmapLevelCount(mips);
        d->setStorageMode(MTL::StorageModePrivate);d->setUsage(MTL::TextureUsageShaderRead|MTL::TextureUsageShaderWrite|MTL::TextureUsageRenderTarget|MTL::TextureUsagePixelFormatView);
        auto* texture=c.memory().newTexture(d,MemoryCategory::RayTracing,label);d->release();if(!texture)throw std::runtime_error("Probe texture allocation failed");return texture;}
    void uploadFace(u32 face,std::span<const float> rgba){const u64 pixelBytes=probeFormat==MTL::PixelFormatRGBA32Float?16:8,rowBytes=ProbeSide*pixelBytes;auto slice=c.stagingAllocate(rowBytes*ProbeSide);
        if(pixelBytes==16)std::memcpy(slice.cpu,rgba.data(),rgba.size()*4);else{auto* words=reinterpret_cast<u16*>(slice.cpu);for(size_t i=0;i<rgba.size();++i)words[i]=probeHalf(rgba[i]);}
        auto* target=rawCube;c.enqueueUpload([slice,target,face,rowBytes](MTL4::ComputeCommandEncoder* e){e->copyFromBuffer(slice.buffer,slice.offset,rowBytes,rowBytes*ProbeSide,MTL::Size::Make(ProbeSide,ProbeSide,1),target,face,0,MTL::Origin::Make(0,0,0));});}
    void buildStatic(const SceneStore& store){
        std::vector<u32> eligible;reflectionProbeStaticSlots(store.instances(),store.nodes(),store.motionSlots(),eligible);
        std::vector<u32> nextIDs;std::vector<Draw> nextDraws;nextIDs.reserve(eligible.size());nextDraws.reserve(eligible.size());
        const auto instances=store.instances();for(u32 slot:eligible){const auto& i=instances[slot];
            if(i.meshIndex>=geometry.size()||i.materialIndex>=store.materials().size())continue;
            const auto& m=store.materials()[i.materialIndex];const CullClass cull=(m.flags&MATERIAL_FLAG_DOUBLE_SIDED)?CullClass::None:(i.flags&INSTANCE_FLAG_MIRRORED)?CullClass::BackMirrored:CullClass::Back;
            nextDraws.push_back({geometry[i.meshIndex],u32(nextIDs.size()),cull});nextIDs.push_back(slot);}
        // Cull/mesh records are consumed while encoding. The GPU slot buffer and
        // per-slot tables only need replacement when membership actually changes.
        draws=std::move(nextDraws);
        if(!staticSlots||nextIDs!=staticIDs){c.waitIdle();staticIDs=std::move(nextIDs);
            c.memory().release(staticSlots,MemoryCategory::RayTracing);staticSlots=lighting::buffer(c,std::max<size_t>(1,staticIDs.size())*4,"Full static scene probe slots",true);
            if(!staticIDs.empty())std::memcpy(staticSlots->contents(),staticIDs.data(),staticIDs.size()*4);
            for(auto& f:slots){for(auto* t:f.drawTables)t->release();f.drawTables.clear();for(size_t i=0;i<6*draws.size();++i)f.drawTables.push_back(lighting::table(c));}
            ++graphVersion;
        }
        staticStructure=store.structureVersion();
    }
    bool updateProbePlacement(const SceneStore& store){
        if(!store.motionSlots().empty()&&!frame.motionSinCosValid)throw std::logic_error("Probe bounds require the capture frame motion phases");
        referenceWorlds(store,frame.motionSinCos.data(),probeWorlds);
        std::optional<ReflectionProbeBounds> bounds;
        if(options.reflectionCaptureProbe&&!rtCapture){
            if(!staticIDs.empty())bounds=reflectionProbeWorldBounds(store.instances(),geometry,probeWorlds,staticIDs);
        }else bounds=reflectionProbeWorldBounds(store.instances(),geometry,probeWorlds);
        const glm::vec3 low=bounds?bounds->minimum:glm::vec3(-10),high=bounds?bounds->maximum:glm::vec3(10),center=(low+high)*.5f;
        bool changed=!probePlaced;for(u32 i=0;i<3;++i){const float minimum=low[i]-.25f,maximum=high[i]+.25f;
            changed|=probe.boxMin[i]!=minimum||probe.boxMax[i]!=maximum||probe.capturePosition[i]!=center[i];
            probe.boxMin[i]=minimum;probe.boxMax[i]=maximum;probe.capturePosition[i]=center[i];}
        probePlaced=true;probePlacementDirty=false;return changed;
    }
    void load(const GpuScene& geometrySource,const SceneStore& store){c.waitIdle();releaseProbe();geometry.assign(geometrySource.meshInfos().begin(),geometrySource.meshInfos().end());buildStatic(store);
        rawCube=cube(1,"Raw reflection probe cube array");filteredCube=cube(ProbeMips,"Filtered reflection probe mip array");
        rawArray=c.memory().newTextureView(rawCube,probeFormat,MTL::TextureType2DArray,NS::Range::Make(0,1),NS::Range::Make(0,6),MemoryCategory::RayTracing,"Raw probe array validation view");
        for(u32 face=0;face<6;++face)faceViews[face]=c.memory().newTextureView(rawCube,probeFormat,MTL::TextureType2D,NS::Range::Make(0,1),NS::Range::Make(face,1),MemoryCategory::RayTracing,"Probe capture face view");
        for(u32 mip=0;mip<ProbeMips;++mip)mipViews[mip]=c.memory().newTextureView(filteredCube,probeFormat,MTL::TextureType2DArray,NS::Range::Make(mip,1),NS::Range::Make(0,6),MemoryCategory::RayTracing,"Filtered probe mip array view");
        if(!rawArray||std::any_of(faceViews.begin(),faceViews.end(),[](auto* t){return !t;})||std::any_of(mipViews.begin(),mipViews.end(),[](auto* t){return !t;}))throw std::runtime_error("Probe texture view creation failed");
        // World placement waits for prepareFrame's exact GPU motion phases.
        // No identity child placeholder can enter capture bounds or parallax.
        probe={};probePlacementDirty=true;probePlaced=false;
        probe.blendDistance=1;probe.mipCount=ProbeMips;probe.generation=++probeGeneration;probe.enabled=1;
        static constexpr const char* faces[]={"px.pfm","nx.pfm","py.pfm","ny.pfm","pz.pfm","nz.pfm"};
        for(u32 face=0;face<6;++face){std::vector<float> rgba;
            if(!options.reflectionProbePath.empty())rgba=probePFM(std::filesystem::path(options.reflectionProbePath)/faces[face]);
            else{rgba.resize(ProbeSide*ProbeSide*4);for(u32 y=0;y<ProbeSide;++y)for(u32 x=0;x<ProbeSide;++x){const auto d=reflectionCubeDirection(face,{(x+.5f)/ProbeSide,(y+.5f)/ProbeSide});const auto L=glm::mix(glm::vec3(.02f,.025f,.03f),glm::vec3(.1f,.12f,.16f),d.y*.5f+.5f);
                    const size_t at=(size_t(y)*ProbeSide+x)*4;for(u32 k=0;k<3;++k)rgba[at+k]=L[k];rgba[at+3]=1;}}
            uploadFace(face,rgba);}
        auto zeroSlice=c.stagingAllocate(16);std::memset(zeroSlice.cpu,0,16);auto* zeroTarget=dummyTexture;c.enqueueUpload([zeroSlice,zeroTarget](MTL4::ComputeCommandEncoder* e){e->copyFromBuffer(zeroSlice.buffer,zeroSlice.offset,16,16,MTL::Size::Make(1,1,1),zeroTarget,0,0,MTL::Origin::Make(0,0,0));});c.flushUploads();
        source=options.reflectionCaptureProbe?(rt?"actual-scene-rt":"actual-static-raster-unshadowed"):options.reflectionProbePath.empty()?"analytic-environment":"cooked-linear-pfm";
        probeDirty=true;probeNeedsCapture=options.reflectionCaptureProbe;sceneEpoch=0;denoise.invalidateAll("F13 scene load");++graphVersion;
    }
    void reserve(){const u64 count=u64(frame.backingWidth)*frame.backingHeight;if(count<=pixels)return;pixels=count;
        for(auto& f:slots){c.memory().release(f.samples,MemoryCategory::RayTracing);c.memory().release(f.metadata,MemoryCategory::RayTracing);
            f.samples=lighting::buffer(c,pixels*options.reflectionSamples*sizeof(GPUSpecularSample),"F13 per-sample raw reflection records");f.metadata=lighting::buffer(c,pixels*sizeof(GPUSpecularSample),"F13 reduced reflection hit metadata");}
        if(options.debugReflectionCorrupt==3)for(auto& f:slots){c.memory().release(f.testSurfaces,MemoryCategory::RayTracing);f.testSurfaces=lighting::buffer(c,pixels*sizeof(GPUDISurface),"F13 independent negative normal guides");}
        denoise.invalidateAll("F13 signal allocation growth");++graphVersion;
    }
    void prepare(const SceneStore& store,const ShadowPasses::Frame& f,bool useCustom,u64 signalEpoch){
        if(!f.width||!f.height||f.width>f.backingWidth||f.height>f.backingHeight||f.slot>=METAL_FRAMES_IN_FLIGHT||f.view>=4||
            u64(f.backingWidth)*f.backingHeight>std::numeric_limits<u32>::max())throw std::invalid_argument("Invalid F13 frame extent or view");
        frame=f;if(!rawCube)throw std::logic_error("F13 loadScene must precede prepareFrame");
        if(custom!=useCustom){custom=useCustom;denoise.invalidateAll("F13 denoiser path change");++graphVersion;}reserve();
        const bool recordsChanged=staticStructure!=store.structureVersion()||store.stats().fullInstances||store.stats().fullNodes||store.stats().fullMaterials||
            !store.instanceDeltas().empty()||!store.nodeDeltas().empty()||!store.materialDeltas().empty();
        if(recordsChanged){buildStatic(store);probePlacementDirty=true;}
        const bool nextRT=rt&&rt->active();const bool nextReflection=options.reflections==ReflectionMode::RT&&nextRT;
        const bool nextAO=options.ao==AoMode::RTAO&&nextRT,nextCapture=options.reflectionCaptureProbe&&nextRT;
        const bool captureChanged=nextCapture!=rtCapture;if(captureChanged)probePlacementDirty=true;
        if(nextReflection!=rtReflection||nextAO!=rtAO||captureChanged){rtReflection=nextReflection;rtAO=nextAO;rtCapture=nextCapture;++graphVersion;}
        // RT captures live geometry. A moving root moves every descendant even
        // when the child's CPU instance remains an unchanged identity record.
        const bool placementChanged=(probePlacementDirty||(rtCapture&&!store.motionSlots().empty()))?updateProbePlacement(store):false;
        if(placementChanged||(options.reflectionCaptureProbe&&(sceneEpoch!=f.scene||lastSignalEpoch!=signalEpoch||recordsChanged||captureChanged))||lastPipelineGeneration!=p.generation()){
            if(!probeDirty)++graphVersion;probeDirty=true;probeNeedsCapture=options.reflectionCaptureProbe;probe.generation=++probeGeneration;}
        sceneEpoch=f.scene;lastSignalEpoch=signalEpoch;lastPipelineGeneration=p.generation();
        source=options.reflectionCaptureProbe?(rtCapture?"actual-scene-rt":"actual-static-raster-unshadowed"):options.reflectionProbePath.empty()?"analytic-environment":"cooked-linear-pfm";
        params={};std::memcpy(params.viewProjection,f.constants.viewProjection,64);const auto inverse=glm::inverse(glm::make_mat4(f.constants.viewProjection));std::memcpy(params.inverseViewProjection,glm::value_ptr(inverse),64);
        std::memcpy(params.view,f.constants.view,64);std::memcpy(params.cameraPosition,f.constants.cameraPosition,16);
        params.width=f.width;params.height=f.height;params.frameIndex=u32(f.index);params.maxDistance=settings.maxDistance;params.ssrThickness=settings.ssrThickness;
        params.rtRoughnessLow=settings.rtRoughnessLow;params.rtRoughnessHigh=settings.rtRoughnessHigh;params.slotCount=store.slotCapacity();params.meshCount=u32(geometry.size());params.materialCount=u32(store.materials().size());params.lightCount=f.constants.lightCount;
        params.ssrSteps=settings.ssrSteps;params.ssrBinarySteps=settings.ssrBinarySteps;params.probeCount=1;params.seed=options.lightingSeed;params.maxProbeMip=ProbeMips-1;
        params.environment[0]=.03f;params.environment[1]=.035f;params.environment[2]=.045f;
        params.flags=REFLECTION_ENABLE_PROBES;if(options.reflections!=ReflectionMode::Probes)params.flags|=REFLECTION_ENABLE_SSR;
        if(rtReflection)params.flags|=REFLECTION_ENABLE_RT;if(gi&&options.gi!=GiMode::Off)params.flags|=REFLECTION_ENABLE_GI|REFLECTION_ENABLE_CACHE;
        paramsAddress=lighting::upload(c,params);probeAddress=lighting::upload(c,probe);
        for(u32 i=0;i<options.reflectionSamples;++i){auto sample=params;sample.seed=options.lightingSeed+i*0x9e3779b9u;sampleAddress[i]=lighting::upload(c,sample);}
        aoParams={};std::memcpy(aoParams.viewProjection,params.viewProjection,64);std::memcpy(aoParams.inverseViewProjection,params.inverseViewProjection,64);
        aoParams.radius=aoSettings.radius;aoParams.originBias=aoSettings.originBias;aoParams.thickness=aoSettings.thickness;aoParams.width=f.width;aoParams.height=f.height;
        aoParams.samples=aoSettings.rays;aoParams.slices=aoSettings.slices;aoParams.steps=aoSettings.steps;aoParams.frameIndex=u32(f.index);aoParams.slotCount=store.slotCapacity();
        const auto projection=glm::make_mat4(f.unjitteredVP)*glm::inverse(glm::make_mat4(f.constants.view));aoParams.pixelScale=std::abs(projection[1][1])*f.height*.5f;aoAddress=lighting::upload(c,aoParams);
        reduceParams={f.width,f.height,options.reflectionSamples,u32(pixels)};reduceAddress=lighting::upload(c,reduceParams);
        composeParams={f.width,f.height,f.backingWidth,f.backingHeight,0,{0,options.debugReflectionCorrupt,0}};
        if(gi&&options.gi!=GiMode::Off)composeParams.flags|=REFLECT_COMPOSE_GI;if(custom)composeParams.flags|=REFLECT_COMPOSE_CUSTOM;
        if(options.directLighting!=DirectLightingMode::Legacy)composeParams.flags|=REFLECT_COMPOSE_DI;
        if(options.reflections!=ReflectionMode::Off)composeParams.flags|=REFLECT_COMPOSE_SPEC;if(options.ao!=AoMode::Off)composeParams.flags|=REFLECT_COMPOSE_AO;composeParams.pad[0]=0;composeAddress=lighting::upload(c,composeParams);
        preComposeParams=composeParams;preComposeParams.flags&=REFLECT_COMPOSE_DI|REFLECT_COMPOSE_GI;preComposeAddress=lighting::upload(c,preComposeParams);
        GPUProbeGridParams dummy{};dummy.reset=1;giDummyAddress=lighting::upload(c,dummy);GPUProbeTraceExtra extra{};extra.sampledLightCount=direct.lightCount();extra.frameSeed=options.lightingSeed;extra.sunAngularRadius=.00465f;extraDummyAddress=lighting::upload(c,extra);
        for(u32 face=0;face<6;++face){const auto vp=reflectionProbeViewProjection(probe,face,.05f,settings.maxDistance),inv=glm::inverse(vp);auto capture=params;
            std::memcpy(capture.viewProjection,glm::value_ptr(vp),64);std::memcpy(capture.inverseViewProjection,glm::value_ptr(inv),64);for(u32 k=0;k<3;++k)capture.cameraPosition[k]=probe.capturePosition[k];capture.width=capture.height=ProbeSide;captureAddress[face]=lighting::upload(c,capture);
            GPUProbeCaptureParams raster{};std::memcpy(raster.viewProjection,glm::value_ptr(vp),64);for(u32 k=0;k<3;++k){raster.capturePosition[k]=probe.capturePosition[k];raster.environment[k]=params.environment[k];}
            raster.slotCount=store.slotCapacity();raster.materialCount=u32(store.materials().size());raster.lightCount=f.constants.lightCount;raster.sampledLightCount=direct.lightCount();raster.seed=options.lightingSeed;rasterCaptureAddress[face]=lighting::upload(c,raster);}
        for(u32 mip=0;mip<ProbeMips;++mip){GPUProbeFilterParams filter{std::max(1u,ProbeSide>>mip),mip,128,0,float(mip)/float(ProbeMips-1),float(ProbeSide),{0,0}};
            filterAddress[mip]=lighting::upload(c,filter);validateAddress[mip]=filterAddress[mip];}GPUProbeFilterParams rawCheck{ProbeSide,~0u,0,0,0,float(ProbeSide),{0,0}};rawValidateAddress=lighting::upload(c,rawCheck);
        if(rtReflection)reflectionConsumer->prepare(frame.slot,reflectionRT);if(rtAO)aoConsumer->prepare(frame.slot,rtao);if(rtCapture&&probeNeedsCapture)captureConsumer->prepare(frame.slot,captureRT);
        auto& slot=slots[frame.slot];slot.customUsed=custom;slot.expected=frame.width*frame.height;
        if(store.stats().fullInstances||!store.instanceDeltas().empty())++instanceRevision;
        if(store.stats().fullNodes||!store.nodeDeltas().empty())++nodeRevision;
        if(store.stats().fullMaterials||!store.materialDeltas().empty())++materialRevision;
        if(!store.motionSlots().empty()||!store.dirtyRoots().empty())++motionRevision;
        const GeometryContent geometryContent{f.scene,store.structureVersion(),rt?rt->geometryRevision():0,instanceRevision,nodeRevision,materialRevision,motionRevision,p.generation()};
        if(!geometryKnown||lastGeometry!=geometryContent){lastGeometry=geometryContent;geometryKnown=true;++geometryEpoch;}
        if(custom){const std::array<u64,4> revisions{direct.lightRevision(),gi&&options.gi!=GiMode::Off?gi->readResources().parameters.cacheGeneration:0,signalEpoch,geometryEpoch};denoise.prepareFrame(frame,signalEpoch,revisions);}
    }
    void texture(MTL4::ArgumentTable* table,rg::PassContext& ctx,rg::TextureRef ref,u32 index){table->setTexture(static_cast<MTL::Texture*>(ctx.texture(ref))->gpuResourceID(),index);}
    void commonReads(rg::PassBuilder& b,rg::Stages stages=rg::StageDispatch){b.read(receivers,rg::Usage::ShaderRead,stages);b.read(direct.lightsRef(),rg::Usage::ShaderRead,stages);b.read(direct.emittersRef(),rg::Usage::ShaderRead,stages);b.read(scene.dataRef(),rg::Usage::ShaderRead,stages);
        if(gi&&options.gi!=GiMode::Off){const auto r=gi->readResources();if(r.cacheRef.valid())b.read(r.cacheRef,rg::Usage::ShaderRead,stages);if(r.stateRef.valid())b.read(r.stateRef,rg::Usage::ShaderRead,stages);
            if(r.irradiance.valid())b.read(r.irradiance,rg::Usage::ShaderRead,stages);if(r.moments.valid())b.read(r.moments,rg::Usage::ShaderRead,stages);}else b.read(dummyTexRef,rg::Usage::ShaderRead,stages);}
    void commonTable(MTL4::ArgumentTable* table,rg::PassContext& ctx,MTL::GPUAddress address){table->setAddress(address,0);table->setAddress(static_cast<MTL::Buffer*>(ctx.buffer(receivers))->gpuAddress(),1);
        table->setAddress(scene.buffers().materials()->gpuAddress(),8);table->setAddress(scene.textureTableAddress(),9);table->setAddress(scene.lightsAddress(),10);
        table->setAddress(direct.lightsBuffer()->gpuAddress(),11);table->setAddress(direct.emittersBuffer()->gpuAddress(),12);table->setAddress(probeMetadata->gpuAddress(),18);
        if(gi&&options.gi!=GiMode::Off){const auto r=gi->readResources();table->setAddress(r.cache->gpuAddress(),13);table->setAddress(r.params,14);table->setAddress(r.states->gpuAddress(),15);table->setAddress(r.extra,16);texture(table,ctx,r.irradiance,3);texture(table,ctx,r.moments,4);}
        else{table->setAddress(dummyBuffer->gpuAddress(),13);table->setAddress(giDummyAddress,14);table->setAddress(dummyBuffer->gpuAddress(),15);table->setAddress(extraDummyAddress,16);texture(table,ctx,dummyTexRef,3);texture(table,ctx,dummyTexRef,4);}
        if(rt&&rt->active()){const auto r=rt->traceResources(frame.slot);table->setAddress(r.instances->gpuAddress(),4);table->setAddress(r.meshes->gpuAddress(),5);table->setAddress(r.vertices->gpuAddress(),6);table->setAddress(r.indices->gpuAddress(),7);}
    }
    void addProbe(rg::RenderGraph& graph){using namespace rg;
        rawRef=graph.importTexture("F13 raw cube parent",{graphProbeFormat,ProbeSide,ProbeSide,6,1},ImportContentsDefined);
        filteredRef=graph.importTexture("F13 filtered cube parent",{graphProbeFormat,ProbeSide,ProbeSide,6,ProbeMips},ImportContentsDefined|ImportOutput);
        probeRef=graph.importBuffer("F13 GPU-published probe descriptor",{sizeof(GPUReflectionProbe)},ImportOutput);
        probeFaultRef=graph.importBuffer("F13 sticky probe validation state",{16},ImportContentsDefined|ImportOutput);
        staticRef=graph.importBuffer("F13 full static probe slots",{staticSlots->length()},ImportContentsDefined);
        for(u32 face=0;face<6;++face)faceRef[face]=graph.importTexture("F13 raw probe face "+std::to_string(face),{graphProbeFormat,ProbeSide,ProbeSide},ImportContentsDefined);
        for(u32 mip=0;mip<ProbeMips;++mip)mipRef[mip]=graph.importTexture("F13 filtered probe mip "+std::to_string(mip),{graphProbeFormat,std::max(1u,ProbeSide>>mip),std::max(1u,ProbeSide>>mip),6},ImportContentsDefined);
        if(probeDirty){
            if(probeNeedsCapture)for(u32 face=0;face<6;++face){
                if(rtCapture)graph.addPass("F13 actual RT probe face "+std::to_string(face),PassType::Compute,[this,face](PassBuilder& b){commonReads(b);rt->declareTraceReads(b);faceRef[face]=b.write(faceRef[face],Usage::ShaderWrite,StageDispatch);b.setProfileShaders("reflection_capture_rt");},[this,face](PassContext& ctx){auto* table=slots[frame.slot].captureTables[face];commonTable(table,ctx,captureAddress[face]);captureConsumer->bind(table,2,3);texture(table,ctx,faceRef[face],5);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),p,captureRT,table,ProbeSide*ProbeSide);});
                else graph.addPass("F13 actual static raster probe face "+std::to_string(face),PassType::Raster,[this,face](PassBuilder& b){commonReads(b,StageVertex|StageFragment);b.read(staticRef,Usage::ShaderRead,StageVertex);
                    ClearValue clear;clear.color[0]=params.environment[0];clear.color[1]=params.environment[1];clear.color[2]=params.environment[2];clear.color[3]=1;faceRef[face]=b.writeColor(faceRef[face],0,LoadIntent::Clear,clear);
                    auto depth=b.createTexture("F13 probe depth face "+std::to_string(face),{Format::Depth32Float,ProbeSide,ProbeSide});b.writeDepth(depth,LoadIntent::Clear);b.setProfileShaders("reflection_probe_capture_vs,reflection_probe_capture_fs");
                },[this,face](PassContext& ctx){auto* encoder=static_cast<MTL4::RenderCommandEncoder*>(ctx.encoder());auto* state=p.render(captureRaster);if(!state)throw std::runtime_error("Probe raster PSO is not ready");
                    encoder->setRenderPipelineState(state);encoder->setDepthStencilState(captureDepthState);encoder->setViewport(MTL::Viewport{0,0,ProbeSide,ProbeSide,0,1});
                    // Capture VP flips Y to cube convention: ordinary object
                    // fronts are clockwise; mirrored fronts reverse cull mode.
                    MTL::CullMode cull=MTL::CullModeNone;
                    for(u32 draw=0;draw<draws.size();++draw){const auto& d=draws[draw];auto* table=slots[frame.slot].drawTables[size_t(face)*draws.size()+draw];
                        table->setAddress(rasterCaptureAddress[face],0);table->setAddress(scene.vertexBuffer()->gpuAddress(),1);table->setAddress(scene.buffers().instances()->gpuAddress(),2);table->setAddress(scene.buffers().materials()->gpuAddress(),3);
                        table->setAddress(scene.lightsAddress(),4);table->setAddress(scene.textureTableAddress(),5);table->setAddress(staticSlots->gpuAddress(),6);table->setAddress(direct.lightsBuffer()->gpuAddress(),7);table->setAddress(direct.emittersBuffer()->gpuAddress(),8);
                        encoder->setArgumentTable(table,MTL::RenderStageVertex|MTL::RenderStageFragment);const auto desired=d.cull==CullClass::None?MTL::CullModeNone:d.cull==CullClass::BackMirrored?MTL::CullModeFront:MTL::CullModeBack;
                        if(desired!=cull){encoder->setCullMode(desired);cull=desired;}const u64 offset=u64(d.mesh.indexOffset)*4;
                        encoder->drawIndexedPrimitives(MTL::PrimitiveTypeTriangle,d.mesh.indexCount,MTL::IndexTypeUInt32,scene.indexBuffer()->gpuAddress()+offset,scene.indexBuffer()->length()-offset,1,d.mesh.vertexOffset,d.staticIndex);
                    }
                    if(cull!=MTL::CullModeNone)encoder->setCullMode(MTL::CullModeNone);
                });}
            graph.addPass("F13 raw probe numerical validation",PassType::Compute,[this](PassBuilder& b){b.read(rawRef,Usage::ShaderRead,StageDispatch);for(auto r:faceRef)b.read(r,Usage::ShaderRead,StageDispatch);b.read(errorsRef,Usage::ShaderRead,StageDispatch);errorsRef=b.write(errorsRef,Usage::ShaderWrite,StageDispatch);},[this](PassContext& ctx){auto* t=slots[frame.slot].rawValidateTable;t->setAddress(rawValidateAddress,0);t->setAddress(slots[frame.slot].errors->gpuAddress(),1);t->setTexture(rawArray->gpuResourceID(),0);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),p,validateProbe,t,ProbeSide*ProbeSide*6);});
            for(u32 mip=0;mip<ProbeMips;++mip){graph.addPass("F13 probe GGX mip "+std::to_string(mip),PassType::Compute,[this,mip](PassBuilder& b){b.read(rawRef,Usage::ShaderRead,StageDispatch);for(auto r:faceRef)b.read(r,Usage::ShaderRead,StageDispatch);
                    if(mip)b.read(filteredRef,Usage::ShaderRead,StageDispatch);filteredRef=b.write(filteredRef,Usage::ShaderWrite,StageDispatch);mipRef[mip]=b.write(mipRef[mip],Usage::ShaderWrite,StageDispatch);b.setProfileShaders("reflection_probe_prefilter");
                },[this,mip](PassContext& ctx){auto* t=slots[frame.slot].filterTables[mip];t->setAddress(filterAddress[mip],0);texture(t,ctx,rawRef,0);t->setTexture(mipViews[mip]->gpuResourceID(),1);
                    auto* encoder=static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder());auto* state=p.compute(prefilter);if(!state)throw std::runtime_error("Probe filter PSO is not ready");encoder->setComputePipelineState(state);encoder->setArgumentTable(t);
                    const u32 side=std::max(1u,ProbeSide>>mip);encoder->dispatchThreads(MTL::Size::Make(side,side,6),MTL::Size::Make(std::min(8u,side),std::min(8u,side),1));});
                graph.addPass("F13 filtered probe validate "+std::to_string(mip),PassType::Compute,[this,mip](PassBuilder& b){b.read(mipRef[mip],Usage::ShaderRead,StageDispatch);b.read(errorsRef,Usage::ShaderRead,StageDispatch);errorsRef=b.write(errorsRef,Usage::ShaderWrite,StageDispatch);},[this,mip](PassContext& ctx){auto* t=slots[frame.slot].validateTables[mip];t->setAddress(validateAddress[mip],0);t->setAddress(slots[frame.slot].errors->gpuAddress(),1);t->setTexture(mipViews[mip]->gpuResourceID(),0);const u32 side=std::max(1u,ProbeSide>>mip);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),p,validateProbe,t,side*side*6);});}
        }
        graph.addPass("F13 publish completed probe generation",PassType::Compute,[this](PassBuilder& b){b.read(filteredRef,Usage::ShaderRead,StageDispatch);for(auto r:mipRef)b.read(r,Usage::ShaderRead,StageDispatch);b.read(errorsRef,Usage::ShaderRead,StageDispatch);errorsRef=b.write(errorsRef,Usage::ShaderWrite,StageDispatch);
            b.read(probeFaultRef,Usage::ShaderRead,StageDispatch);probeFaultRef=b.write(probeFaultRef,Usage::ShaderWrite,StageDispatch);probeRef=b.write(probeRef,Usage::ShaderWrite,StageDispatch);b.setProfileShaders("reflection_probe_ready");},[this](PassContext& ctx){auto* t=slots[frame.slot].publishTable;t->setAddress(probeAddress,0);t->setAddress(probeMetadata->gpuAddress(),1);t->setAddress(slots[frame.slot].errors->gpuAddress(),2);t->setAddress(probeFault->gpuAddress(),3);texture(t,ctx,filteredRef,0);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),p,publishProbe,t,1);
            if(probeDirty){probeDirty=false;probeNeedsCapture=false;++graphVersion;}
        });
    }
    rg::TextureRef add(rg::RenderGraph& graph,rg::TextureRef baseHDR,rg::TextureRef sceneDepth){using namespace rg;
        residual=baseHDR;depth=sceneDepth;receivers=direct.surfaceRef();motion=direct.motion();filteredIndirect={};auto& slot=slots[frame.slot];
        if(!residual.valid()||!depth.valid())throw std::invalid_argument("F13 needs resolved residual HDR and depth");
        const auto hdrDescriptor=graph.resources().at(residual.resource).texture;
        if(hdrDescriptor.format!=Format::RGBA16Float&&hdrDescriptor.format!=Format::RGBA32Float)throw std::invalid_argument("F13 base must be linear floating HDR");
        outputHalf=false;composeParams.pad[0]=0;composeAddress=lighting::upload(c,composeParams);
        preComposeParams=composeParams;preComposeParams.flags&=REFLECT_COMPOSE_DI|REFLECT_COMPOSE_GI;preComposeAddress=lighting::upload(c,preComposeParams);
        errorsRef=graph.importBuffer("F13 completed signal error words",{64},ImportPerFrame|ImportOutput);
        dummyRef=graph.importBuffer("F13 disabled GI safe buffer",{dummyBuffer->length()},ImportContentsDefined);
        dummyTexRef=graph.importTexture("F13 disabled GI zero texture",{Format::RGBA32Float,1,1},ImportContentsDefined);
        sampleRef=graph.importBuffer("F13 per-sample specular records",{slot.samples->length()},ImportPerFrame);
        metadata=graph.importBuffer("F13 selected secondary hit metadata",{slot.metadata->length()},ImportPerFrame);
        graph.addPass("F13 numerical counters clear",PassType::Compute,[this](PassBuilder& b){errorsRef=b.write(errorsRef,Usage::ShaderWrite,StageDispatch);},[this](PassContext& ctx){auto* t=slots[frame.slot].clearTable;t->setAddress(slots[frame.slot].errors->gpuAddress(),0);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),p,clear,t,16);});
        if(options.debugReflectionCorrupt>=2){if(options.debugReflectionCorrupt==3)receivers=graph.importBuffer("F13 owned poisoned receiver guides",{slot.testSurfaces->length()},ImportPerFrame);
            graph.addPass("F13 authored input negative",PassType::Compute,[this](PassBuilder& b){b.read(direct.surfaceRef(),Usage::ShaderRead,StageDispatch);b.read(direct.motion(),Usage::ShaderRead,StageDispatch);
                if(options.debugReflectionCorrupt==3)receivers=b.write(receivers,Usage::ShaderWrite,StageDispatch);motion=b.createTexture("F13 owned negative pixel motion",{Format::RG32Float,frame.width,frame.height});motion=b.write(motion,Usage::ShaderWrite,StageDispatch);b.setProfileShaders("reflection_input_poison");
            },[this](PassContext& ctx){auto* t=slots[frame.slot].inputPoisonTable;t->setAddress(composeAddress,0);t->setAddress(static_cast<MTL::Buffer*>(ctx.buffer(direct.surfaceRef()))->gpuAddress(),1);
                t->setAddress(options.debugReflectionCorrupt==3?slots[frame.slot].testSurfaces->gpuAddress():dummyBuffer->gpuAddress(),2);texture(t,ctx,direct.motion(),0);texture(t,ctx,motion,1);
                lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),p,inputPoison,t,frame.width*frame.height);});}
        graph.addPass("F13 independent receiver motion input check",PassType::Compute,[this](PassBuilder& b){b.read(receivers,Usage::ShaderRead,StageDispatch);b.read(motion,Usage::ShaderRead,StageDispatch);
            b.read(errorsRef,Usage::ShaderRead,StageDispatch);errorsRef=b.write(errorsRef,Usage::ShaderWrite,StageDispatch);},[this](PassContext& ctx){auto* t=slots[frame.slot].inputCheckTable;t->setAddress(composeAddress,0);
                t->setAddress(static_cast<MTL::Buffer*>(ctx.buffer(receivers))->gpuAddress(),1);t->setAddress(slots[frame.slot].errors->gpuAddress(),2);texture(t,ctx,motion,0);
                lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),p,inputCheck,t,frame.width*frame.height);});
        addProbe(graph);
        const auto rawSourceDescriptor=hdrDescriptor;
        graph.addPass("F13 positive RAW pre-reflection lighting",PassType::Compute,[this,rawSourceDescriptor](PassBuilder& b){b.read(residual,Usage::ShaderRead,StageDispatch);b.read(receivers,Usage::ShaderRead,StageDispatch);
            b.read(direct.direct(),Usage::ShaderRead,StageDispatch);b.read(gi&&options.gi!=GiMode::Off?gi->irradiance():dummyTexRef,Usage::ShaderRead,StageDispatch);b.read(dummyTexRef,Usage::ShaderRead,StageDispatch);
            b.read(errorsRef,Usage::ShaderRead,StageDispatch);errorsRef=b.write(errorsRef,Usage::ShaderWrite,StageDispatch);auto descriptor=rawSourceDescriptor;descriptor.format=Format::RGBA32Float;
            base=b.createTexture("F13 complete raw SSR source Lo",descriptor);base=b.write(base,Usage::ShaderWrite,StageDispatch);b.setProfileShaders("reflection_composite");
        },[this](PassContext& ctx){auto* t=slots[frame.slot].preComposeTable;t->setAddress(preComposeAddress,0);t->setAddress(static_cast<MTL::Buffer*>(ctx.buffer(receivers))->gpuAddress(),1);t->setAddress(slots[frame.slot].errors->gpuAddress(),2);
            texture(t,ctx,residual,0);texture(t,ctx,direct.direct(),2);texture(t,ctx,gi&&options.gi!=GiMode::Off?gi->irradiance():dummyTexRef,4);texture(t,ctx,dummyTexRef,5);texture(t,ctx,dummyTexRef,6);texture(t,ctx,base,7);
            lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),p,compose,t,frame.backingWidth*frame.backingHeight);});
        graph.addPass("F13 inactive signals clear",PassType::Compute,[this](PassBuilder& b){
            metadata=b.write(metadata,Usage::ShaderWrite,StageDispatch);
            specular=b.createTexture("F13 raw specular Lo",{Format::RGBA32Float,frame.width,frame.height});specular=b.write(specular,Usage::ShaderWrite,StageDispatch);
            ao=b.createTexture("F13 raw ambient visibility",{Format::RGBA32Float,frame.width,frame.height});ao=b.write(ao,Usage::ShaderWrite,StageDispatch);
            distance=b.createTexture("F13 world specular hit distance",{Format::R32Float,frame.width,frame.height});distance=b.write(distance,Usage::ShaderWrite,StageDispatch);
        },[this](PassContext& ctx){auto* t=slots[frame.slot].zeroTable;t->setAddress(composeAddress,0);t->setAddress(slots[frame.slot].metadata->gpuAddress(),1);texture(t,ctx,specular,0);texture(t,ctx,ao,1);texture(t,ctx,distance,2);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),p,zero,t,frame.width*frame.height);});
        if(options.reflections!=ReflectionMode::Off){rg::TextureRef scratch{},scratchDistance{};
            // Scratch storage is reused, but each dispatch owns its argument
            // table and output slice. Aggregate record versions preserve all
            // earlier slices and prevent the graph culling prior proposals.
            for(u32 sample=0;sample<options.reflectionSamples;++sample){auto sampleParams=params;sampleParams.seed=options.lightingSeed+sample*0x9e3779b9u;const auto address=lighting::upload(c,sampleParams);
                // Updated in prepareFrame for cached graph callbacks below.
                sampleAddress[sample]=address;
                graph.addPass("F13 reflection sample "+std::to_string(sample),PassType::Compute,[this,sample,&scratch,&scratchDistance](PassBuilder& b){
                    commonReads(b);b.read(base,Usage::ShaderRead,StageDispatch);b.read(depth,Usage::ShaderRead,StageDispatch);b.read(filteredRef,Usage::ShaderRead,StageDispatch);b.read(probeRef,Usage::ShaderRead,StageDispatch);
                    if(rtReflection)rt->declareTraceReads(b);if(sample)b.read(sampleRef,Usage::ShaderRead,StageDispatch);sampleRef=b.write(sampleRef,Usage::ShaderWrite,StageDispatch);
                    if(sample==0){scratch=b.createTexture("F13 raw sample scratch",{Format::RGBA32Float,frame.width,frame.height});scratchDistance=b.createTexture("F13 sample distance scratch",{Format::R32Float,frame.width,frame.height});}
                    scratch=b.write(scratch,Usage::ShaderWrite,StageDispatch);scratchDistance=b.write(scratchDistance,Usage::ShaderWrite,StageDispatch);sampleScratch[sample]=scratch;sampleDistance[sample]=scratchDistance;
                    b.setProfileShaders(options.reflections==ReflectionMode::Probes?"reflection_probe_only":rtReflection?"reflection_rt":"reflection_ssr");
                },[this,sample](PassContext& ctx){auto* t=slots[frame.slot].sampleTables[sample];commonTable(t,ctx,sampleAddress[sample]);
                    t->setAddress(slots[frame.slot].samples->gpuAddress()+u64(sample)*pixels*sizeof(GPUSpecularSample),17);
                    if(rtReflection)reflectionConsumer->bind(t,2,3);texture(t,ctx,base,0);texture(t,ctx,depth,1);texture(t,ctx,filteredRef,2);texture(t,ctx,sampleScratch[sample],5);texture(t,ctx,sampleDistance[sample],6);
                    lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),p,options.reflections==ReflectionMode::Probes?probeOnly:rtReflection?reflectionRT:reflectionSSR,t,frame.width*frame.height);
                });}
            graph.addPass("F13 independent raw specular reduction",PassType::Compute,[this](PassBuilder& b){b.read(sampleRef,Usage::ShaderRead,StageDispatch);metadata=b.write(metadata,Usage::ShaderWrite,StageDispatch);
                b.read(errorsRef,Usage::ShaderRead,StageDispatch);errorsRef=b.write(errorsRef,Usage::ShaderWrite,StageDispatch);specular=b.write(specular,Usage::ShaderWrite,StageDispatch);distance=b.write(distance,Usage::ShaderWrite,StageDispatch);b.setProfileShaders("reflection_reduce");
            },[this](PassContext& ctx){auto* t=slots[frame.slot].reduceTable;t->setAddress(reduceAddress,0);t->setAddress(slots[frame.slot].samples->gpuAddress(),1);t->setAddress(slots[frame.slot].metadata->gpuAddress(),2);t->setAddress(slots[frame.slot].errors->gpuAddress(),3);texture(t,ctx,specular,0);texture(t,ctx,distance,1);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),p,reduce,t,frame.width*frame.height);});
        }
        if(options.ao!=AoMode::Off)graph.addPass(rtAO?"F13 world-radius RTAO":"F13 world-radius GTAO",PassType::Compute,[this](PassBuilder& b){b.read(receivers,Usage::ShaderRead,StageDispatch);b.read(depth,Usage::ShaderRead,StageDispatch);if(rtAO)rt->declareTraceReads(b);ao=b.write(ao,Usage::ShaderWrite,StageDispatch);b.setProfileShaders(rtAO?"ao_rtao":"ao_gtao");},[this](PassContext& ctx){auto* t=slots[frame.slot].aoTable;t->setAddress(aoAddress,0);t->setAddress(static_cast<MTL::Buffer*>(ctx.buffer(receivers))->gpuAddress(),1);
            if(rtAO){aoConsumer->bind(t,2,3);t->setAddress(rt->traceResources(frame.slot).instances->gpuAddress(),4);}texture(t,ctx,depth,0);texture(t,ctx,ao,1);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),p,rtAO?rtao:gtao,t,frame.width*frame.height);});
        directSelected=direct.direct();giSelected=gi&&options.gi!=GiMode::Off?gi->irradiance():dummyTexRef;specSelected=specular;aoSelected=ao;
        if(custom){if(options.directLighting!=DirectLightingMode::Legacy)directSelected=denoise.addSignal(graph,DENOISE_SIGNAL_DI,direct.direct(),motion,receivers);
            if(gi&&options.gi!=GiMode::Off)giSelected=denoise.addSignal(graph,DENOISE_SIGNAL_GI,gi->irradiance(),motion,receivers);
            if(options.reflections!=ReflectionMode::Off)specSelected=denoise.addSignal(graph,DENOISE_SIGNAL_SPECULAR,specular,motion,receivers,metadata);
            if(options.ao!=AoMode::Off)aoSelected=denoise.addSignal(graph,DENOISE_SIGNAL_AO,ao,motion,receivers);}
        if(options.captureLinearSignal==8){
            if(!custom||!gi||options.gi==GiMode::Off)throw std::logic_error("Filtered indirect capture requires GI and custom denoising");
            graph.addPass("F13 actual filtered indirect diffuse Lo export",PassType::Compute,[this](PassBuilder& b){
                b.read(giSelected,Usage::ShaderRead,StageDispatch);b.read(receivers,Usage::ShaderRead,StageDispatch);
                filteredIndirect=b.createTexture("Actual filtered indirect diffuse reflected radiance",{Format::RGBA32Float,frame.width,frame.height});
                filteredIndirect=b.write(filteredIndirect,Usage::ShaderWrite,StageDispatch);b.setProfileShaders("reflection_filtered_indirect_diffuse");
            },[this](PassContext& ctx){auto* t=slots[frame.slot].filteredIndirectTable;t->setAddress(composeAddress,0);
                t->setAddress(static_cast<MTL::Buffer*>(ctx.buffer(receivers))->gpuAddress(),1);texture(t,ctx,giSelected,0);texture(t,ctx,filteredIndirect,1);
                lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),p,filteredIndirectExport,t,frame.width*frame.height);});
        }
        auto finalDescriptor=hdrDescriptor;finalDescriptor.format=Format::RGBA32Float;
        graph.addPass("F13 single-count positive lighting composite",PassType::Compute,[this,finalDescriptor](PassBuilder& b){b.read(residual,Usage::ShaderRead,StageDispatch);b.read(receivers,Usage::ShaderRead,StageDispatch);
            for(auto ref:{directSelected,giSelected,specSelected,aoSelected})b.read(ref,Usage::ShaderRead,StageDispatch);
            b.read(errorsRef,Usage::ShaderRead,StageDispatch);errorsRef=b.write(errorsRef,Usage::ShaderWrite,StageDispatch);output=b.createTexture("F13 composed linear HDR",finalDescriptor);output=b.write(output,Usage::ShaderWrite,StageDispatch);b.setProfileShaders("reflection_composite");
        },[this](PassContext& ctx){auto* t=slots[frame.slot].composeTable;t->setAddress(composeAddress,0);t->setAddress(static_cast<MTL::Buffer*>(ctx.buffer(receivers))->gpuAddress(),1);t->setAddress(slots[frame.slot].errors->gpuAddress(),2);
            texture(t,ctx,residual,0);texture(t,ctx,directSelected,2);texture(t,ctx,giSelected,4);texture(t,ctx,specSelected,5);texture(t,ctx,aoSelected,6);texture(t,ctx,output,7);
            lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),p,compose,t,frame.backingWidth*frame.backingHeight);});
        if(options.debugLighting)graph.addPass("F13 raw and composed independent checks",PassType::Compute,[this](PassBuilder& b){for(auto ref:{output,specular,ao,distance})b.read(ref,Usage::ShaderRead,StageDispatch);if(options.reflections!=ReflectionMode::Off)b.read(metadata,Usage::ShaderRead,StageDispatch);
            b.read(errorsRef,Usage::ShaderRead,StageDispatch);errorsRef=b.write(errorsRef,Usage::ShaderWrite,StageDispatch);
        },[this](PassContext& ctx){auto* t=slots[frame.slot].checkTable;t->setAddress(composeAddress,0);t->setAddress(slots[frame.slot].metadata->gpuAddress(),1);t->setAddress(slots[frame.slot].errors->gpuAddress(),2);texture(t,ctx,output,0);texture(t,ctx,specular,1);texture(t,ctx,ao,2);texture(t,ctx,distance,3);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),p,checker,t,frame.width*frame.height);});
        return output;
    }
    std::array<MTL::GPUAddress,8> sampleAddress{};std::array<rg::TextureRef,8> sampleScratch{},sampleDistance{};
    void bind(MetalGraphExecutor& executor){auto& slot=slots[frame.slot];executor.bindBuffer(errorsRef,slot.errors);executor.bindBuffer(sampleRef,slot.samples);executor.bindBuffer(metadata,slot.metadata);executor.bindBuffer(probeRef,probeMetadata);executor.bindBuffer(probeFaultRef,probeFault);executor.bindBuffer(staticRef,staticSlots);executor.bindBuffer(dummyRef,dummyBuffer);
        executor.bindTexture(rawRef,rawCube);executor.bindTexture(filteredRef,filteredCube);executor.bindTexture(dummyTexRef,dummyTexture);if(options.debugReflectionCorrupt==3)executor.bindBuffer(receivers,slot.testSurfaces);
        for(u32 i=0;i<6;++i)executor.bindTexture(faceRef[i],faceViews[i]);for(u32 i=0;i<ProbeMips;++i)executor.bindTexture(mipRef[i],mipViews[i]);if(custom)denoise.bindFrame(executor);
    }
};
ReflectionPasses::ReflectionPasses(MetalContext& c,PipelineCache& p,SceneRenderer& s,DirectLightingPasses& d,AccelerationStructures* a,GiPasses* g,const LaunchOptions& o):impl_(std::make_unique<Impl>(c,p,s,d,a,g,o)){}
ReflectionPasses::~ReflectionPasses()=default;
void ReflectionPasses::loadScene(const GpuScene& g,const SceneStore& s){impl_->load(g,s);}
void ReflectionPasses::prepareFrame(const SceneStore& s,const ShadowPasses::Frame& f,bool custom,u64 epoch){impl_->prepare(s,f,custom,epoch);}
rg::TextureRef ReflectionPasses::addToGraph(rg::RenderGraph& g,rg::TextureRef base,rg::TextureRef depth){return impl_->add(g,base,depth);}
void ReflectionPasses::bindFrame(MetalGraphExecutor& e){impl_->bind(e);}
u64 ReflectionPasses::version()const{const std::pair<u64,u64> components{impl_->graphVersion,impl_->denoise.version()};if(impl_->publishedComponents!=components){impl_->publishedComponents=components;++impl_->publishedVersion;}return impl_->publishedVersion;}
bool ReflectionPasses::check(u32 slot)const{const auto& f=impl_->slots.at(slot);const auto* errors=static_cast<const u32*>(f.errors->contents());bool okay=!impl_->options.debugLighting||errors[0]==f.expected;for(u32 i=1;i<8;++i)okay=okay&&!errors[i];if(!okay){std::fprintf(stderr,"REFLECTION check slot %u expected %u counts %u %u %u %u %u %u %u %u\n",slot,f.expected,errors[0],errors[1],errors[2],errors[3],errors[4],errors[5],errors[6],errors[7]);
    if(errors[8])std::fprintf(stderr,"REFLECTION motion raw slot %u count %u first_tid %u bits %08x %08x\n",slot,errors[8],errors[9],errors[10],errors[11]);return false;}return !f.customUsed||impl_->denoise.check(slot);}
bool ReflectionPasses::ready()const{const auto& i=*impl_;const auto& o=i.options;
    if(!i.p.compute(i.compose)||!i.p.compute(i.zero)||!i.p.compute(i.clear)||!i.p.compute(i.prefilter)||!i.p.compute(i.validateProbe)||!i.p.compute(i.publishProbe))return false;
    if(!i.p.compute(i.inputCheck)||(o.debugReflectionCorrupt>=2&&!i.p.compute(i.inputPoison)))return false;
    if(o.captureLinearSignal==8&&!i.p.compute(i.filteredIndirectExport))return false;
    if(o.reflections!=ReflectionMode::Off&&(!i.p.compute(i.reduce)||!i.p.compute(o.reflections==ReflectionMode::Probes?i.probeOnly:i.reflectionSSR)))return false;
    if(o.reflections==ReflectionMode::RT&&i.rt&&!i.p.compute(i.reflectionRT))return false;
    if(o.ao!=AoMode::Off&&!i.p.compute(i.gtao))return false;if(o.ao==AoMode::RTAO&&i.rt&&!i.p.compute(i.rtao))return false;
    if(o.reflectionCaptureProbe&&!(i.rt?bool(i.p.compute(i.captureRT)):bool(i.p.render(i.captureRaster))))return false;
    if(o.debugLighting&&!i.p.compute(i.checker))return false;return o.lightingDenoise==LightingDenoiseMode::Off||i.denoise.ready();}
rg::TextureRef ReflectionPasses::hitDistance()const{return impl_->distance;}rg::TextureRef ReflectionPasses::rawSpecular()const{return impl_->specular;}rg::TextureRef ReflectionPasses::rawAO()const{return impl_->ao;}
rg::TextureRef ReflectionPasses::filteredSpecular()const{return impl_->specSelected;}rg::TextureRef ReflectionPasses::filteredAO()const{return impl_->aoSelected;}rg::BufferRef ReflectionPasses::metadataRef()const{return impl_->metadata;}
rg::TextureRef ReflectionPasses::filteredIndirectDiffuse()const{return impl_->filteredIndirect;}
const char* ReflectionPasses::probeSource()const{return impl_->source.c_str();}
} // namespace phosphor
