#include "platform/metal/metalfx_denoise_fixture.h"
#include "platform/metal/pipeline_cache.h"
#include "platform/metal/gpu_memory.h"
#include "platform/metal/metal_graph_executor.h"
#include "renderer/metalfx_fixture_oracle.h"
#include "renderer/offline_reference.h"
#include "renderer/history_registry.h"
#include "core/log.h"
#include <chrono>
#include "rendergraph/pass_context.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace phosphor {
namespace {
constexpr std::array<std::string_view,5> scenarios{"constant","impulse","channels","lifecycle","wide-hdr"};
// Analytic camera exposure maps the physical red target to 0.5. The documented
// R16Float exposure texture rounds this subnormal to 23 units of 2^-24.
constexpr float WideManualExposure=.5f/368640.f;
constexpr float WideManualExposureR16=23.f*0x1p-24f;
enum Input : u32 { Color,Normal,Rough,Diffuse,Specular,Motion,Hit,Reactive,Strength,Depth,InputCount };
constexpr std::array<rg::Format,InputCount> formats{rg::Format::RGBA32Float,rg::Format::RGBA16Float,rg::Format::R16Float,
    rg::Format::RGBA16Float,rg::Format::RGBA16Float,rg::Format::RG32Float,rg::Format::R32Float,rg::Format::R8Unorm,rg::Format::R8Unorm,rg::Format::Depth32Float};
std::string quote(std::string_view input) {
    std::ostringstream out;out<<'"';for(unsigned char c:input){if(c=='"'||c=='\\')out<<'\\'<<c;
        else if(c=='\n')out<<"\\n";else if(c=='\r')out<<"\\r";else if(c=='\t')out<<"\\t";
        else if(c<32)out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<unsigned(c)<<std::dec;else out<<c;}out<<'"';return out.str();
}
std::string supplied(const char* key){const auto* value=std::getenv(key);return value?quote(value):"null";}
std::string number(double value){if(!std::isfinite(value))return "null";std::ostringstream out;out<<std::setprecision(17)<<value;return out.str();}
pipe::PipelineDesc compute(const char* name){pipe::PipelineDesc d;d.kind=pipe::PipelineKind::Compute;d.label=name;d.functions={name,"",""};return d;}
}
struct MetalfxDenoiseFixture::Impl {
    MetalContext& c;PipelineCache& p;Options options;std::unique_ptr<MetalfxDenoise> adapter;
    pipe::PipelineHandle generate{},depthPipeline{},readback{};
    MTL::DepthStencilState* depthState=nullptr;
    MetalfxDenoise::Frame frame{};GPUFXFixtureParams params{};MTL::GPUAddress paramsAddress=0;
    struct Slot {
        std::array<MTL::Texture*,InputCount> textures{};
        MTL::Buffer* images=nullptr;MTL::Buffer* samples=nullptr;
        MTL4::ArgumentTable *generate=nullptr,*depth=nullptr,*readback=nullptr;
        u32 width=0,height=0,outWidth=0,outHeight=0;bool pending=false;
        GPUFXFixtureParams tag{};u64 frame=0;u32 view=0,shaderGeneration=0;bool steady=false;
        u64 inputSignalEpoch=0,signalEpoch=0,sdkResetsBefore=0,sdkEncodesBefore=0,sdkEncodeDelta=0;
        bool requestedReset=false,requestedCut=false,channelsWrap=false,sdkResetSubmitted=false;
    };
    std::array<std::array<Slot,METAL_FRAMES_IN_FLIGHT>,HistoryRegistry::MaxViews> slots{};
    std::array<rg::TextureRef,InputCount> refs{};rg::TextureRef sdk{},physical{};
    static constexpr std::array<metalfx_denoise::Channel,11> packedChannels{metalfx_denoise::Channel::Color,metalfx_denoise::Channel::Normal,
        metalfx_denoise::Channel::Roughness,metalfx_denoise::Channel::Motion,metalfx_denoise::Channel::Depth,metalfx_denoise::Channel::DiffuseAlbedo,
        metalfx_denoise::Channel::SpecularAlbedo,metalfx_denoise::Channel::Exposure,metalfx_denoise::Channel::HitDistance,metalfx_denoise::Channel::Reactive,metalfx_denoise::Channel::Strength};
    std::array<rg::TextureRef,packedChannels.size()> packed{};
    rg::BufferRef imageRef{},sampleRef{};bool nativeGraph=false;
    u64 ownVersion=1,nativeFrames=0,checkedFrames=0,steadyFrames=0,packChecks=0,failures=0;
    u32 capturedViews=0;u64 suppliedCuts=0;std::vector<metalfx_denoise::Extent> capturedExtents;
    bool reloadRequested=false;std::vector<u32> capturedGenerations;
    mutable u64 publishedVersion=1;mutable std::pair<u64,u64> published{};
    struct ScaledPair {std::vector<float> unit,scaled;u32 width=0,height=0;FXMetamorphicComparison result{};bool checked=false;};
    std::map<u32,ScaledPair> pairs;
    bool finishCalled=false,prewarmAttempted=false,prewarmReady=false;double prewarmElapsedMs=0;
    std::string prewarmStatus="NOT_ATTEMPTED";
    Impl(MetalContext& context,PipelineCache& pipelines,MetalfxDenoise::Factory factory,Options o):c(context),p(pipelines),options(std::move(o)) {
        if(!MetalfxDenoiseFixture::validScenario(options.scenario)||options.outputDirectory.empty())throw std::invalid_argument("Invalid denoised fixture scenario/output");
        if(!options.activeViews||options.activeViews>HistoryRegistry::MaxViews||!options.prewarmTimeoutMs||options.prewarmTimeoutMs>120000)
            throw std::invalid_argument("Invalid SDK fixture active views/prewarm budget");
        MetalfxDenoise::Options sdkOptions;sdkOptions.enabled=true;sdkOptions.views=options.activeViews;
        sdkOptions.autoExposure=options.autoExposure;
        // Supply a binary16-representable float so native texture-write rounding
        // cannot choose a different texel. Preserve the analytic request in logs.
        sdkOptions.manualExposure=options.manualExposureControl?WideManualExposureR16:1.f;
        sdkOptions.resizeSettleFrames=0; // controlled fixture extents; production keeps its async settling
        sdkOptions.reactiveMask=true;sdkOptions.specularHitDistance=false;sdkOptions.strengthMask=true;
        sdkOptions.sdkOutputScale=options.preExposedPolicy?MetalfxDenoise::Options::OutputScale::PreExposed:MetalfxDenoise::Options::OutputScale::Unverified;
        adapter=std::make_unique<MetalfxDenoise>(c,p,sdkOptions,std::move(factory));
        generate=p.request(compute("fx_fixture_generate"));readback=p.request(compute("fx_fixture_readback"));
        pipe::PipelineDesc depth;depth.kind=pipe::PipelineKind::Render;depth.label="SDK fixture actual Depth32 writer";
        depth.functions={"fx_fixture_depth_vs","fx_fixture_depth_fs",""};depthPipeline=p.request(depth);
        auto* d=MTL::DepthStencilDescriptor::alloc()->init();d->setDepthCompareFunction(MTL::CompareFunctionAlways);d->setDepthWriteEnabled(true);
        depthState=c.device()->newDepthStencilState(d);d->release();if(!depthState)throw std::runtime_error("SDK fixture depth state allocation failed");
    }
    ~Impl(){c.waitIdle();for(u32 s=0;s<METAL_FRAMES_IN_FLIGHT;++s){
            try{consume(s);}catch(...){++failures;for(auto& view:slots)view[s].pending=false;}
        }
        adapter.reset();for(auto& view:slots)for(auto& s:view){release(s);for(auto* t:{s.generate,s.depth,s.readback})if(t)t->release();}
        if(depthState)depthState->release();}
    std::string exposureMetadata(const GPUFXFixtureSample* packedSamples=nullptr)const {
        const auto& stats=adapter->stats();std::ostringstream out;
        out<<",\"auto_exposure_requested\":"<<(options.autoExposure?"true":"false")
           <<",\"exposure_descriptor_configured\":"<<(stats.descriptorConfigured?"true":"false")
           <<",\"auto_exposure_enabled\":"<<(stats.autoExposureEnabled?"true":"false")
           <<",\"exposure_mode\":"<<quote(!stats.descriptorConfigured?"not-configured":stats.autoExposureEnabled?"sdk-auto":"manual")
           <<",\"manual_exposure_control\":"<<(options.manualExposureControl?"true":"false")
           <<",\"requested_manual_exposure_fp32\":"<<number(options.manualExposureControl?WideManualExposure:1.f)
           <<",\"provided_manual_exposure_fp32\":"<<number(options.manualExposureControl?WideManualExposureR16:1.f)
           <<",\"manual_exposure_prequantized\":"<<(options.manualExposureControl?"true":"false")
           <<",\"expected_manual_exposure_r16\":"<<number(options.manualExposureControl?WideManualExposureR16:1.f)
           <<",\"provided_manual_exposure_texture_value\":"<<number(packedSamples?packedSamples[0].exposure:options.manualExposureControl?WideManualExposureR16:1.f)
           <<",\"actual_manual_exposure_readback\":"<<(packedSamples?"true":"false")
           <<",\"actual_provided_manual_exposure_texture_value\":"<<(packedSamples?number(packedSamples[0].exposure):"null")
           <<",\"manual_exposure_texture_ignored\":"<<(stats.autoExposureEnabled?"true":"false")
           <<",\"packed_exposure_is_provided_manual_value\":true";
        // Before GPU readback, provided_manual_exposure_texture_value is the
        // expected R16 value; actual_manual_exposure_readback explicitly says so.
        // Per-frame packed_samples.exposure comes from the supplied 1x1 texture.
        // Auto exposure ignores it; the SDK internal exposure is not exposed.
        return out.str();
    }
    void release(Slot& s) {
        if(s.pending)throw std::logic_error("SDK fixture readback cannot retire before consumption");
        for(auto*& t:s.textures){c.memory().release(t,MemoryCategory::RenderTargets);t=nullptr;}
        c.memory().release(s.images,MemoryCategory::Other);c.memory().release(s.samples,MemoryCategory::Other);s.images=s.samples=nullptr;
    }
    MTL4::ArgumentTable* table(u32 buffers,u32 textures) {
        auto* d=MTL4::ArgumentTableDescriptor::alloc()->init();d->setMaxBufferBindCount(buffers);d->setMaxTextureBindCount(textures);
        NS::Error* error=nullptr;auto* value=c.device()->newArgumentTable(d,&error);d->release();
        if(!value)throw std::runtime_error("SDK fixture argument table allocation failed");return value;
    }
    void reserve(Slot& s) {
        const auto e=frame.extent;
        if(s.width==e.inputWidth&&s.height==e.inputHeight&&s.outWidth==e.outputWidth&&s.outHeight==e.outputHeight)return;
        release(s);s.width=e.inputWidth;s.height=e.inputHeight;s.outWidth=e.outputWidth;s.outHeight=e.outputHeight;
        for(u32 i=0;i<InputCount;++i) {
            auto* d=MTL::TextureDescriptor::texture2DDescriptor(toMetalFormat(formats[i]),s.width,s.height,false);
            d->setStorageMode(MTL::StorageModePrivate);
            d->setUsage(i==Depth?MTL::TextureUsageShaderRead|MTL::TextureUsageRenderTarget:MTL::TextureUsageShaderRead|MTL::TextureUsageShaderWrite);
            s.textures[i]=c.memory().newTexture(d,MemoryCategory::RenderTargets,"SDK fixture generated source");
            if(!s.textures[i])throw std::runtime_error("SDK fixture texture allocation failed");
        }
        s.images=c.memory().newBuffer(u64(s.outWidth)*s.outHeight*6*sizeof(float),MTL::ResourceStorageModeShared,MemoryCategory::Other,"SDK actual half and physical readback");
        s.samples=c.memory().newBuffer(8*sizeof(GPUFXFixtureSample),MTL::ResourceStorageModeShared,MemoryCategory::Other,"SDK actual authored and packed guide probes");
        if(!s.images||!s.samples)throw std::runtime_error("SDK fixture readback allocation failed");
        if(!s.generate)s.generate=table(1,9);if(!s.depth)s.depth=table(1,0);if(!s.readback)s.readback=table(4,20);++ownVersion;
    }
    void prepare(const MetalfxDenoise::Frame& source) {
        if(source.view>=HistoryRegistry::MaxViews||source.slot>=METAL_FRAMES_IN_FLIGHT)throw std::invalid_argument("Invalid SDK fixture view/slot");
        if(!metalfx_denoise::validExtent(source.extent)||source.extent.inputWidth<16||source.extent.inputHeight<16)
            throw std::invalid_argument("SDK fixture needs valid extents and at least 16 input pixels per axis");
        consume(source.slot);frame=source;if(source.cut||source.reset)++suppliedCuts;
        // Exercise the existing cache gateway after real native GPU work, rather
        // than merely racing an initial request at the renderer's frame5 hook.
        // The cache owns its compiler, atomic generation and old PSO retirement.
        if(options.scenario=="lifecycle"&&!reloadRequested&&nativeFrames>=8&&!p.reloadPending())
            reloadRequested=p.reload(p.library());
        params={};params.width=source.extent.inputWidth;params.height=source.extent.inputHeight;params.outputWidth=source.extent.outputWidth;params.outputHeight=source.extent.outputHeight;
        params.scenario=u32(std::find(scenarios.begin(),scenarios.end(),options.scenario)-scenarios.begin());params.frame=u32(source.index);
        const u32 phase=u32(source.index/48u)%2u;
        params.phase=params.scenario==FX_FIXTURE_CHANNELS?u32(source.index%std::max(1u,params.width/2u)):phase;
        params.color[0]=.5f;params.color[1]=.25f;params.color[2]=.125f;
        params.nearPlane=.1f;params.planeDistance=4;params.motionPixels=1;params.impulseAmplitude=.5f;
        frame.preExposure=options.preExposedPolicy&&phase?1.f/64.f:1.f;
        if(params.scenario==FX_FIXTURE_WIDE_HDR){params.color[0]=368640;params.color[1]=128;params.color[2]=64;frame.preExposure=1.f/64.f;}
        params.preExposure=frame.preExposure;
        // This is an actual analytic plane camera, independent of whatever scene
        // the renderer displays behind the fixture. Depth=.1/4 is reverse-Z.
        frame.worldToView={1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
        const float f=1.7320508075688772f,aspect=float(source.extent.outputWidth)/source.extent.outputHeight;
        frame.viewToClip={f/aspect,0,0,0,0,f,0,0,0,0,0,-1,0,0,.1f,0};frame.jitterPixels={0,0};
        // A modulo wrap replaces the authored signal without a motion-vector
        // correspondence. Invalidate that known source change once; ordinary
        // one-pixel motion keeps temporal history, as in production.
        const auto history=fxFixtureHistoryPolicy(params.scenario,source.index,params.width,source.signalEpoch,source.reset);
        frame.signalEpoch=history.signalEpoch;frame.reset=history.reset;
        auto& slot=slots[frame.view][frame.slot];reserve(slot);
        slot.inputSignalEpoch=source.signalEpoch;slot.signalEpoch=frame.signalEpoch;slot.requestedReset=frame.reset;
        slot.requestedCut=frame.cut;slot.channelsWrap=history.channelsWrap;slot.steady=history.steady;
        slot.sdkResetsBefore=adapter->stats().resets;slot.sdkEncodesBefore=adapter->stats().encodedFrames;
        auto slice=c.frameUploads().allocate(sizeof(params));std::memcpy(slice.cpu,&params,sizeof(params));paramsAddress=slice.gpu;
        adapter->prepareFrame(frame);finishCalled=false;
        if(!prewarmAttempted) {
            const auto started=std::chrono::steady_clock::now();prewarmAttempted=true;
            const bool pending=adapter->status()==MetalfxDenoise::Status::Pending||adapter->status()==MetalfxDenoise::Status::Settling||adapter->status()==MetalfxDenoise::Status::Ready;
            prewarmReady=pending&&adapter->prewarmPreparedFixture(options.activeViews,std::chrono::milliseconds(options.prewarmTimeoutMs));
            prewarmElapsedMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
            prewarmStatus=prewarmReady?"READY":pending?"INITIAL_PREWARM_FAILED":"SKIPPED_TERMINAL_CONTRACT";
            std::filesystem::create_directories(options.outputDirectory);
            const auto path=std::filesystem::path(options.outputDirectory)/"prewarm.json";
            if(std::filesystem::exists(path))throw std::runtime_error("Refuse SDK fixture initial prewarm evidence overwrite");
            std::ofstream out(path);out<<"{\"schema\":\"phosphor.metalfx-prewarm.v1\",\"state\":"<<quote(prewarmStatus)<<exposureMetadata()
                <<",\"ready\":"<<(prewarmReady?"true":"false")<<",\"budget_ms\":"<<options.prewarmTimeoutMs<<",\"elapsed_ms\":"<<number(prewarmElapsedMs)
                <<",\"active_views\":"<<options.activeViews<<",\"frame\":"<<frame.index<<",\"phase\":"<<params.phase
                <<",\"input_width\":"<<frame.extent.inputWidth<<",\"input_height\":"<<frame.extent.inputHeight<<",\"output_width\":"<<frame.extent.outputWidth<<",\"output_height\":"<<frame.extent.outputHeight
                <<",\"shader_generation\":"<<p.generation()<<",\"actual_encoded_before_counting\":"<<adapter->stats().encodedFrames<<",\"cancelled\":false,\"reason\":"<<quote(adapter->fallbackReason())<<"}\n";
            if(!out)throw std::runtime_error("SDK initial prewarm evidence write failed");
            LOG_INFO("SDK fixture initial prewarm %s: %.1f ms / %u ms, %u views, frame %llu phase %u",prewarmStatus.c_str(),prewarmElapsedMs,options.prewarmTimeoutMs,options.activeViews,static_cast<unsigned long long>(frame.index),params.phase);
            if(pending&&!prewarmReady)throw std::runtime_error("SDK fixture initial prewarm failed before counting frames: "+adapter->fallbackReason());
        }
    }
    bool pipelinesReady()const{return p.compute(generate)&&p.compute(readback)&&p.render(depthPipeline);}
    rg::TextureRef add(rg::RenderGraph& graph) {
        using namespace rg;nativeGraph=false;if(!pipelinesReady())return {};
        auto& slot=slots[frame.view][frame.slot];
        for(u32 i=0;i<InputCount;++i)refs[i]=graph.importTexture("SDK fixture source "+std::to_string(i),{formats[i],slot.width,slot.height},ImportPerFrame|ImportOutput);
        graph.addPass("SDK fixture real channel generation",PassType::Compute,[&](PassBuilder& b){
            for(u32 i=0;i<Depth;++i)refs[i]=b.write(refs[i],Usage::ShaderWrite,StageDispatch);b.setProfileShaders("fx_fixture_generate");
        },[this](PassContext& ctx){
            auto& s=slots[frame.view][frame.slot];s.generate->setAddress(paramsAddress,0);
            for(u32 i=0;i<Depth;++i)s.generate->setTexture(s.textures[i]->gpuResourceID(),i);
            auto* e=static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder());e->setComputePipelineState(p.compute(generate));e->setArgumentTable(s.generate);
            e->dispatchThreads(MTL::Size::Make(params.width,params.height,1),MTL::Size::Make(8,8,1));
        });
        graph.addPass("SDK fixture legitimate reverse-Z depth",PassType::Raster,[&](PassBuilder& b){
            ClearValue clear;clear.depth=0;refs[Depth]=b.writeDepth(refs[Depth],LoadIntent::Clear,clear);b.setProfileShaders("fx_fixture_depth_vs,fx_fixture_depth_fs");
        },[this](PassContext& ctx){
            auto& s=slots[frame.view][frame.slot];s.depth->setAddress(paramsAddress,0);
            auto* e=static_cast<MTL4::RenderCommandEncoder*>(ctx.encoder());e->setRenderPipelineState(p.render(depthPipeline));e->setDepthStencilState(depthState);
            e->setViewport(MTL::Viewport{0,0,double(params.width),double(params.height),0,1});e->setArgumentTable(s.depth,MTL::RenderStageFragment);
            e->drawPrimitives(MTL::PrimitiveTypeTriangle,0,3);
        });
        MetalfxDenoise::Inputs inputs{refs[Color],refs[Depth],refs[Motion],refs[Diffuse],refs[Specular],refs[Normal],refs[Rough],
                                     refs[Hit],refs[Reactive],refs[Strength],refs[Color]};
        physical=adapter->addToGraph(graph,inputs);sdk=adapter->sdkOutputRef();
        if(!adapter->ready()||!sdk.valid())return physical;
        for(u32 i=0;i<packed.size();++i)packed[i]=adapter->packedChannelRef(packedChannels[i]);
        imageRef=graph.importBuffer("SDK fixture actual images",{slot.images->length()},ImportPerFrame|ImportOutput);
        sampleRef=graph.importBuffer("SDK fixture actual input sample probes",{slot.samples->length()},ImportPerFrame|ImportOutput);
        graph.addPass("SDK fixture actual output and guide readback",PassType::Compute,[&](PassBuilder& b){
            b.read(sdk,Usage::ShaderRead,StageDispatch);b.read(physical,Usage::ShaderRead,StageDispatch);
            for(auto i:{Color,Normal,Rough,Motion,Depth,Diffuse,Specular})b.read(refs[i],Usage::ShaderRead,StageDispatch);
            for(auto guide:packed)b.read(guide,Usage::ShaderRead,StageDispatch);
            imageRef=b.write(imageRef,Usage::ShaderWrite,StageDispatch);sampleRef=b.write(sampleRef,Usage::ShaderWrite,StageDispatch);b.setProfileShaders("fx_fixture_readback");
        },[this](PassContext& ctx){
            auto& s=slots[frame.view][frame.slot];auto* t=s.readback;t->setAddress(paramsAddress,0);t->setAddress(s.images->gpuAddress(),1);t->setAddress(s.samples->gpuAddress(),2);
            t->setAddress(s.samples->gpuAddress()+4*sizeof(GPUFXFixtureSample),3);
            t->setTexture(static_cast<MTL::Texture*>(ctx.texture(sdk))->gpuResourceID(),0);t->setTexture(static_cast<MTL::Texture*>(ctx.texture(physical))->gpuResourceID(),1);
            const std::array<Input,7> authored{Color,Normal,Rough,Motion,Depth,Diffuse,Specular};
            for(u32 i=0;i<authored.size();++i)t->setTexture(s.textures[authored[i]]->gpuResourceID(),i+2);
            for(u32 i=0;i<packed.size();++i)t->setTexture(static_cast<MTL::Texture*>(ctx.texture(packed[i]))->gpuResourceID(),i+9);
            auto* e=static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder());e->setComputePipelineState(p.compute(readback));e->setArgumentTable(t);
            e->dispatchThreads(MTL::Size::Make(std::max(4u,params.outputWidth*params.outputHeight),1,1),MTL::Size::Make(32,1,1));
            // These counters advance in the actual SDK encoding callback,
            // after setShouldResetHistory(reset) and encodeToCommandBuffer.
            s.sdkEncodeDelta=adapter->stats().encodedFrames-s.sdkEncodesBefore;
            s.sdkResetSubmitted=adapter->stats().resets==s.sdkResetsBefore+1u;
            s.tag=params;s.frame=frame.index;s.view=frame.view;s.shaderGeneration=p.generation();s.pending=true;++nativeFrames;
            capturedViews|=1u<<frame.view;
            if(std::find(capturedExtents.begin(),capturedExtents.end(),frame.extent)==capturedExtents.end())capturedExtents.push_back(frame.extent);
            if(std::find(capturedGenerations.begin(),capturedGenerations.end(),s.shaderGeneration)==capturedGenerations.end())capturedGenerations.push_back(s.shaderGeneration);
        });
        nativeGraph=true;return physical;
    }
    void bind(MetalGraphExecutor& e) {
        if(!pipelinesReady())return;const auto& s=slots[frame.view][frame.slot];
        for(u32 i=0;i<InputCount;++i)e.bindTexture(refs[i],s.textures[i]);
        adapter->bindFrame(e);if(nativeGraph){e.bindBuffer(imageRef,s.images);e.bindBuffer(sampleRef,s.samples);}
    }
    bool channels(const Slot& s,const GPUFXFixtureSample* points,bool packedSample=false)const {
        bool pass=true;const auto& t=s.tag;
        for(u32 i=0;i<4;++i) {
            const u32 x=i==0?t.width/4u:i==1?3u*t.width/4u:i==2?t.width/2u:std::min(t.width/4u+t.phase,t.width-1u);
            const auto& sample=points[i];const bool right=x>=t.width/2u;
            const float expectedR=t.scenario==FX_FIXTURE_CHANNELS?(right?.9f:.04f):.5f;
            const std::array<float,3> expectedN=t.scenario==FX_FIXTURE_CHANNELS&&right?std::array<float,3>{.6f,0,.8f}:std::array<float,3>{0,0,1};
            for(u32 j=0;j<3;++j)pass=pass&&std::isfinite(sample.normal[j])&&std::abs(sample.normal[j]-expectedN[j])<=.002f;
            pass=pass&&std::isfinite(sample.roughness)&&std::abs(sample.roughness-expectedR)<=.001f&&
                std::abs(sample.depth-.025f)<=1e-6f&&std::abs(sample.diffuseR-.6f)<=.001f&&std::abs(sample.specularR-.04f)<=.001f;
            const u32 center=t.width/4u+t.phase;const bool moving=t.scenario==FX_FIXTURE_CHANNELS&&t.phase>0&&std::abs(int(x)-int(center))<=3;
            pass=pass&&std::isfinite(sample.motion[0])&&std::isfinite(sample.motion[1])&&sample.motion[0]==(moving?-1.f:0.f)&&sample.motion[1]==0;
            std::array<float,3> expectedColor{t.color[0],t.color[1],t.color[2]};
            if(t.scenario==FX_FIXTURE_IMPULSE)expectedColor=std::abs(int(x)-int(t.width/2u))<=3?std::array<float,3>{t.impulseAmplitude,t.impulseAmplitude,t.impulseAmplitude}:std::array<float,3>{0,0,0};
            if(t.scenario==FX_FIXTURE_CHANNELS)expectedColor=std::abs(int(x)-int(center))<=3?std::array<float,3>{.75f,.5f,.25f}:std::array<float,3>{.125f,.125f,.125f};
            for(u32 j=0;j<3;++j){if(packedSample)expectedColor[j]*=t.preExposure;
                pass=pass&&std::isfinite(sample.color[j])&&std::abs(sample.color[j]-expectedColor[j])<=(packedSample?.001f:1e-6f)*std::max(1.f,std::abs(expectedColor[j]));}
            if(packedSample)pass=pass&&sample.exposure==(options.manualExposureControl?WideManualExposureR16:1.f)&&sample.hitDistance==0&&sample.reactive==0&&sample.strength==0;
        }
        return pass;
    }
    bool consume(u32 index) {
        if(index>=METAL_FRAMES_IN_FLIGHT)throw std::invalid_argument("SDK fixture slot outside ring");
        bool all=true;for(auto& view:slots){auto& s=view[index];if(!s.pending)continue;
            if(c.frameEvent()->signaledValue()<=s.frame)throw std::logic_error("SDK fixture readback before real GPU completion");
            const size_t count=size_t(s.tag.outputWidth)*s.tag.outputHeight*3;
            const auto* data=static_cast<const float*>(s.images->contents());const auto* samples=static_cast<const GPUFXFixtureSample*>(s.samples->contents());
            const std::span<const float> sdkImage(data,count),physicalImage(data+count,count);
            bool finite=std::all_of(data,data+count*2,[](float value){return std::isfinite(value)&&value>=0;});
            const bool channelPass=channels(s,samples)&&channels(s,samples+4,true);bool constantPass=true;FXConstantComparison constant{};
            if(s.tag.scenario==FX_FIXTURE_CONSTANT||s.tag.scenario==FX_FIXTURE_LIFECYCLE||s.tag.scenario==FX_FIXTURE_WIDE_HDR) {
                constant=compareFXConstant(sdkImage,physicalImage,{s.tag.color[0],s.tag.color[1],s.tag.color[2]},s.tag.preExposure);
                constantPass=!s.steady||constant.restored;
            }
            if(s.steady&&s.tag.scenario==FX_FIXTURE_IMPULSE) {
                auto& pair=pairs[s.view];if(pair.width!=s.tag.outputWidth||pair.height!=s.tag.outputHeight){pair={};pair.width=s.tag.outputWidth;pair.height=s.tag.outputHeight;}
                if(s.tag.preExposure==1)pair.unit.assign(sdkImage.begin(),sdkImage.end());else pair.scaled.assign(sdkImage.begin(),sdkImage.end());
                if(!pair.unit.empty()&&!pair.scaled.empty()){pair.result=compareFXScaledPair(pair.unit,pair.scaled,1.f/64);pair.checked=true;}
            }
            const bool historyPass=s.sdkEncodeDelta==1u&&(!(s.requestedReset||s.requestedCut)||s.sdkResetSubmitted);
            const bool pass=finite&&channelPass&&constantPass&&historyPass;if(!pass)++failures;++checkedFrames;if(s.steady)++steadyFrames;all=all&&pass;
            std::ostringstream stem;stem<<"frame-"<<std::setw(6)<<std::setfill('0')<<s.frame<<"-view-"<<s.view;
            std::filesystem::create_directories(options.outputDirectory);
            const auto base=std::filesystem::path(options.outputDirectory)/stem.str();
            if(std::filesystem::exists(base.string()+".json")||std::filesystem::exists(base.string()+"-sdk.pfm")||std::filesystem::exists(base.string()+"-physical.pfm"))
                throw std::runtime_error("Refuse SDK fixture evidence overwrite");
            std::string error;if(!writeLinearPfm(base.string()+"-sdk.pfm",s.tag.outputWidth,s.tag.outputHeight,sdkImage,error)||
                !writeLinearPfm(base.string()+"-physical.pfm",s.tag.outputWidth,s.tag.outputHeight,physicalImage,error))throw std::runtime_error(error);
            std::ofstream out(base.string()+".json");out<<std::setprecision(17);
            out<<"{\"schema\":\"phosphor.metalfx-fixture.v1\",\"kind\":\"f13-sdk\",\"actual_sdk_encoded\":true,\"frame\":"<<s.frame
               <<",\"view\":"<<s.view<<",\"slot\":"<<index<<",\"scenario\":"<<quote(options.scenario)<<",\"preExposure\":"<<s.tag.preExposure<<exposureMetadata(samples+4)
               <<",\"steady\":"<<(s.steady?"true":"false")<<",\"finite\":"<<(finite?"true":"false")<<",\"channels_passed\":"<<(channelPass?"true":"false")
               <<",\"requested_history_reset\":"<<(s.requestedReset?"true":"false")<<",\"requested_camera_cut\":"<<(s.requestedCut?"true":"false")
               <<",\"input_signal_epoch\":"<<s.inputSignalEpoch<<",\"source_signal_epoch\":"<<s.signalEpoch
               <<",\"channels_wrap\":"<<(s.channelsWrap?"true":"false")<<",\"sdk_reset_submitted\":"<<(s.sdkResetSubmitted?"true":"false")
               <<",\"sdk_encode_delta\":"<<s.sdkEncodeDelta<<",\"history_hint_passed\":"<<(historyPass?"true":"false")
               <<",\"restored_passed\":"<<(constant.restored?"true":"false")<<",\"sdk_preexposed_hypothesis\":"<<(constant.preExposed?"true":"false")
               <<",\"sdk_physical_hypothesis\":"<<(constant.physical?"true":"false")<<",\"restored_relative_error\":"<<number(constant.restoredRelativeError)
               <<",\"input_width\":"<<s.tag.width<<",\"input_height\":"<<s.tag.height<<",\"output_width\":"<<s.tag.outputWidth<<",\"output_height\":"<<s.tag.outputHeight
               <<",\"phase\":"<<s.tag.phase<<",\"constant_physical\":["<<s.tag.color[0]<<','<<s.tag.color[1]<<','<<s.tag.color[2]<<"],\"input_samples\":[";
            auto writeSamples=[&](u32 first){for(u32 i=0;i<4;++i){if(i)out<<',';const auto& a=samples[first+i];
                out<<"{\"color\":["<<number(a.color[0])<<','<<number(a.color[1])<<','<<number(a.color[2])<<"],\"normal\":["<<number(a.normal[0])<<','<<number(a.normal[1])<<','<<number(a.normal[2])
                   <<"],\"roughness\":"<<number(a.roughness)<<",\"depth\":"<<number(a.depth)<<",\"motion\":["<<number(a.motion[0])<<','<<number(a.motion[1])
                   <<"],\"diffuseR\":"<<number(a.diffuseR)<<",\"specularR\":"<<number(a.specularR)<<",\"exposure\":"<<number(a.exposure)
                   <<",\"hitDistance\":"<<number(a.hitDistance)<<",\"reactive\":"<<number(a.reactive)<<",\"strength\":"<<number(a.strength)<<'}';}};
            writeSamples(0);out<<"],\"packed_samples\":[";writeSamples(4);
            out<<"],\"passed\":"<<(pass?"true":"false")<<",\"policy_promoted\":false,\"provenance\":{\"shader_generation\":"<<s.shaderGeneration
               <<",\"source_sha\":"<<supplied("PHOSPHOR_SOURCE_SHA")<<",\"binary_sha\":"<<supplied("PHOSPHOR_BINARY_SHA")
               <<",\"manifest_sha\":"<<supplied("PHOSPHOR_MANIFEST_SHA")<<"}}\n";
            if(!out)throw std::runtime_error("SDK fixture JSON write failed");s.pending=false;
        }
        for(const auto& result:adapter->drainPackChecks())if(result.available){++packChecks;if(!result.ok){++failures;all=false;}}
        return all;
    }
    bool lifecyclePassed()const {
        const auto& stats=adapter->stats();
        return options.scenario!="lifecycle"||((capturedViews&(capturedViews-1u))!=0&&capturedExtents.size()>1&&suppliedCuts>0&&stats.requests>1&&stats.retirements>0&&stats.resets>1&&reloadRequested&&capturedGenerations.size()>1);
    }
    bool passed()const {
        bool pairPass=true;if(options.scenario=="impulse"){
            pairPass=!pairs.empty();for(const auto& [view,pair]:pairs){(void)view;pairPass=pairPass&&pair.checked&&pair.result.passed;}
        }
        return finishCalled&&nativeFrames>0&&checkedFrames==nativeFrames&&steadyFrames>0&&packChecks==nativeFrames&&
               adapter->stats().encodedFrames==nativeFrames&&!failures&&pairPass&&lifecyclePassed();
    }
    bool finish() {
        if(finishCalled)return passed();
        for(u32 slot=0;slot<METAL_FRAMES_IN_FLIGHT;++slot)consume(slot);
        finishCalled=true;std::filesystem::create_directories(options.outputDirectory);
        const auto path=std::filesystem::path(options.outputDirectory)/"summary.json";
        if(std::filesystem::exists(path))throw std::runtime_error("Refuse SDK fixture summary overwrite");
        std::ofstream out(path);out<<report()<<'\n';if(!out)throw std::runtime_error("SDK fixture summary write failed");return passed();
    }
    std::string report()const {
        const bool pass=passed();const auto& stats=adapter->stats();
        std::ostringstream out;out<<"{\"schema\":\"phosphor.metalfx-fixture.v1\",\"scenario\":"<<quote(options.scenario)<<exposureMetadata()
            <<",\"state\":"<<quote(!nativeFrames?"NOT_EXECUTED_NATIVE":!finishCalled?"GPU_RECORDS_PENDING":pass?"FIXTURE_CHECKS_PASSED":"FIXTURE_CHECKS_FAILED")
            <<",\"native_frames\":"<<nativeFrames<<",\"checked_frames\":"<<checkedFrames<<",\"steady_frames\":"<<steadyFrames<<",\"pack_checks\":"<<packChecks<<",\"failures\":"<<failures<<",\"passed\":"<<(pass?"true":"false")
            <<",\"actual_encoded_frames\":"<<stats.encodedFrames<<",\"factory_requests\":"<<stats.requests<<",\"retirements_submitted\":"<<stats.retirements<<",\"obsolete_requests\":"<<stats.discardedRequests
            <<",\"sdk_available\":"<<(stats.sdkAvailable?"true":"false")<<",\"device_supported\":"<<(stats.deviceSupported?"true":"false")<<",\"factory_installed\":"<<(stats.factoryInstalled?"true":"false")
            <<",\"history_resets\":"<<stats.resets<<",\"supplied_cuts\":"<<suppliedCuts<<",\"captured_view_mask\":"<<capturedViews<<",\"captured_extent_count\":"<<capturedExtents.size()<<",\"lifecycle_passed\":"<<(lifecyclePassed()?"true":"false")
            <<",\"reload_requested_after_native_work\":"<<(reloadRequested?"true":"false")<<",\"captured_generation_count\":"<<capturedGenerations.size()
            <<",\"retirement_is_destruction_proof\":false,\"final_destruction_verified\":false"
            <<",\"fallback_reason\":"<<quote(adapter->fallbackReason())<<",\"preexposed_policy_experiment\":"<<(options.preExposedPolicy?"true":"false")
            <<",\"initial_prewarm_attempted\":"<<(prewarmAttempted?"true":"false")<<",\"initial_prewarm_ready\":"<<(prewarmReady?"true":"false")
            <<",\"initial_prewarm_state\":"<<quote(prewarmStatus)<<",\"initial_prewarm_budget_ms\":"<<options.prewarmTimeoutMs<<",\"initial_prewarm_elapsed_ms\":"<<number(prewarmElapsedMs)<<",\"initial_active_views\":"<<options.activeViews
            <<",\"phase_accepted\":false,\"production_policy_promoted\":false,\"metamorphic\":[";
        bool first=true;for(const auto& [view,pair]:pairs){if(!first)out<<',';first=false;out<<"{\"view\":"<<view<<",\"checked\":"<<(pair.checked?"true":"false")
            <<",\"normalized_gain\":"<<number(pair.result.normalizedGain)<<",\"shape_relative_error\":"<<number(pair.result.relativeShapeError)
            <<",\"support_disagreement\":"<<number(pair.result.supportDisagreement)<<",\"passed\":"<<(pair.result.passed?"true":"false")<<'}';}
        out<<"]}";return out.str();
    }
};
bool MetalfxDenoiseFixture::validScenario(std::string_view name){return std::find(scenarios.begin(),scenarios.end(),name)!=scenarios.end();}
MetalfxDenoiseFixture::MetalfxDenoiseFixture(MetalContext& c,PipelineCache& p,MetalfxDenoise::Factory f,Options o):impl_(std::make_unique<Impl>(c,p,std::move(f),std::move(o))){}
MetalfxDenoiseFixture::~MetalfxDenoiseFixture()=default;
void MetalfxDenoiseFixture::prepareFrame(const MetalfxDenoise::Frame& f){impl_->prepare(f);}
bool MetalfxDenoiseFixture::initialPrewarmAttempted()const{return impl_->prewarmAttempted;}
rg::TextureRef MetalfxDenoiseFixture::addToGraph(rg::RenderGraph& g){return impl_->add(g);}
void MetalfxDenoiseFixture::bindFrame(MetalGraphExecutor& e){impl_->bind(e);}
bool MetalfxDenoiseFixture::consume(u32 slot){return impl_->consume(slot);}
bool MetalfxDenoiseFixture::finish(){return impl_->finish();}
bool MetalfxDenoiseFixture::ready()const{return impl_->pipelinesReady()&&impl_->adapter->ready();}
u64 MetalfxDenoiseFixture::version()const {const std::pair<u64,u64> key{impl_->ownVersion,impl_->adapter->version()};if(impl_->published!=key){impl_->published=key;++impl_->publishedVersion;}return impl_->publishedVersion;}
std::string MetalfxDenoiseFixture::status()const{return impl_->adapter->ready()?"ready":impl_->adapter->fallbackReason();}
std::string MetalfxDenoiseFixture::reportJSON()const{return impl_->report();}
} // namespace phosphor
