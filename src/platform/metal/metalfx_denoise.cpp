#include "platform/metal/metalfx_denoise.h"
#include "platform/metal/gpu_memory.h"
#include "platform/metal/pipeline_cache.h"
#include "platform/metal/metal_graph_executor.h"
#include "renderer/history_registry.h"
#include "rendergraph/pass_context.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

#if !defined(PHOSPHOR_DISABLE_METALFX_DENOISED) && \
    __has_include(<MetalFX/MTL4FXTemporalDenoisedScaler.hpp>) && \
    __has_include(<MetalFX/MTLFXTemporalDenoisedScaler.hpp>)
#define PHOSPHOR_HAS_METALFX_DENOISED 1
#include <MetalFX/MTL4FXTemporalDenoisedScaler.hpp>
#include <MetalFX/MTLFXTemporalDenoisedScaler.hpp>
#else
#define PHOSPHOR_HAS_METALFX_DENOISED 0
#endif

namespace phosphor {
namespace {
using Channel=metalfx_denoise::Channel;
constexpr size_t ChannelCount=size_t(Channel::Count);
constexpr size_t channelIndex(Channel c){return size_t(c);}
bool finiteMatrix(const std::array<float,16>& m) {
    return std::all_of(m.begin(),m.end(),[](float x){return std::isfinite(x);});
}
pipe::PipelineDesc packPipeline(const char* name) {
    pipe::PipelineDesc d;d.kind=pipe::PipelineKind::Compute;d.label=name;d.functions={name,"",""};return d;
}
}

struct MetalfxDenoise::Impl {
    MetalContext& context;PipelineCache& pipelines;Options options;Factory factory;Frame frame{};
    Stats stats{};Status currentStatus=Status::Disabled;std::string reason="MetalFX denoised was not requested";
    float minimumScale=1,maximumScale=1;u64 graphVersion=1;
    pipe::PipelineHandle packHandle{},clearHandle{},restoreHandle{};
    HistoryRegistry histories;
    struct Packed {
        std::array<MTL::Texture*,ChannelCount> textures{};
        MTL::Buffer* counters=nullptr;MTL4::ArgumentTable* table=nullptr;MTL4::ArgumentTable* clearTable=nullptr;MTL4::ArgumentTable* restoreTable=nullptr;
        u64 recordedFrame=0;u32 expectedPixels=0;bool recorded=false,consumed=false;
    };
    struct View {
        Scaler scaler;std::future<Scaler> pending;
        metalfx_denoise::RequestKey desired{},requested{};
        bool desiredValid=false,failed=false,used=false;
        Status failureStatus=Status::FactoryRejected;std::string failureReason;
        u32 stableFrames=0;
        std::array<MTL::TextureUsage,ChannelCount> usage{};
        std::array<Packed,METAL_FRAMES_IN_FLIGHT> slots{};
        MTL::Buffer* historyToken=nullptr;
    };
    std::array<View,HistoryRegistry::MaxViews> views{};
    MTL::Texture* neutral=nullptr;
    rg::TextureRef neutralRef{};
    Inputs inputs{};
    std::array<rg::TextureRef,ChannelCount> packedRefs{};
    rg::BufferRef counterRef{},historyRef{};
    MTL::GPUAddress packParamsAddress=0,restoreParamsAddress=0;
    bool reset=true,graphNative=false;
    struct RetiredCheck {MTL::Buffer* buffer;u64 frame;u32 view,slot,expectedPixels;};
    std::vector<RetiredCheck> retiredChecks;
    std::vector<PackCheck> completedChecks;

    Impl(MetalContext& c,PipelineCache& p,Options o,Factory f)
        :context(c),pipelines(p),options(o),factory(std::move(f)) {
        if(!o.views||o.views>HistoryRegistry::MaxViews)throw std::invalid_argument("MetalFX denoise views outside 1..4");
        stats.requested=o.enabled;stats.sdkAvailable=PHOSPHOR_HAS_METALFX_DENOISED!=0;
        stats.factoryInstalled=bool(factory.request)&&bool(factory.retire);
#if PHOSPHOR_HAS_METALFX_DENOISED
        // Safe selectors in the local SDK return false when the runtime class/API
        // is unavailable. This does not certify another physical Apple device.
        stats.deviceSupported=MTLFX::TemporalDenoisedScalerDescriptor::supportsDevice(c.device()) &&
            MTLFX::TemporalDenoisedScalerDescriptor::supportsMetal4FX(c.device());
        if(stats.deviceSupported) {
            minimumScale=MTLFX::TemporalDenoisedScalerDescriptor::supportedInputContentMinScale(c.device());
            maximumScale=MTLFX::TemporalDenoisedScalerDescriptor::supportedInputContentMaxScale(c.device());
        }
#endif
        if(o.enabled && stats.sdkAvailable && stats.deviceSupported && stats.factoryInstalled) {
            packHandle=p.request(packPipeline("denoise_pack"));clearHandle=p.request(packPipeline("denoise_pack_clear"));
            restoreHandle=p.request(packPipeline("denoise_restore_radiance"));
            auto* d=MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatRGBA16Float,1,1,false);
            d->setStorageMode(MTL::StorageModeShared);d->setUsage(MTL::TextureUsageShaderRead);
            neutral=c.memory().newTexture(d,MemoryCategory::RenderTargets,"MetalFX absent optional guide");
            if(!neutral)throw std::runtime_error("MetalFX neutral guide allocation failed");
            const u16 zero[4]{};neutral->replaceRegion(MTL::Region::Make2D(0,0,1,1),0,zero,sizeof(zero));
        }
    }
    ~Impl() {
        context.waitIdle();collect(true,false);
        for(auto& v:views) {
            retire(std::move(v.scaler),v.used);
            releaseTargets(v);
            context.memory().release(v.historyToken,MemoryCategory::RenderTargets);
            for(auto& slot:v.slots) {
                if(slot.table)slot.table->release();
                if(slot.clearTable)slot.clearTable->release();
                if(slot.restoreTable)slot.restoreTable->release();
            }
        }
        collectChecks(); // waitIdle above makes every archived readback available.
        context.memory().release(neutral,MemoryCategory::RenderTargets);
        context.collectGarbage(); // Deferred callbacks still have a live gateway.
    }
    void setStatus(Status s,std::string why) {
        if(currentStatus!=s){currentStatus=s;++graphVersion;}
        reason=std::move(why);
    }
    void fail(View& v,Status s,std::string why) {
        v.failed=true;v.failureStatus=s;v.failureReason=why;setStatus(s,std::move(why));
    }
    void retire(Scaler s,bool used) {
        if(!s)return;
        ++stats.retirements;
        // Plain ownership only. Do NOT call adoptTemporalScaler, inspect private
        // references, or apply the F8 TemporalScaler cycle workaround here.
        if(used) context.deferCall([retire=factory.retire,s=std::move(s)]()mutable{retire(std::move(s));});
        else factory.retire(std::move(s)); // Never encoded: utility release is safe.
    }
    void releaseTargets(View& v) {
        for(auto& slot:v.slots) {
            for(auto*& t:slot.textures){context.memory().release(t,MemoryCategory::RenderTargets);t=nullptr;}
            if(slot.counters&&slot.recorded&&!slot.consumed)
                retiredChecks.push_back({slot.counters,slot.recordedFrame,u32(&v-views.data()),u32(&slot-v.slots.data()),slot.expectedPixels});
            else context.memory().release(slot.counters,MemoryCategory::Other);
            slot.counters=nullptr;slot.recorded=false;slot.consumed=false;
        }
    }
    u32 flags()const {
        return (options.specularHitDistance?METALFX_PACK_HIT_DISTANCE:0u) |
            (options.reactiveMask?METALFX_PACK_REACTIVE:0u) |
            (options.strengthMask?METALFX_PACK_STRENGTH:0u);
    }
    u32 descriptorFlags()const {
        // Request identity includes SDK exposure mode; shader pack flags do not.
        return flags()|(options.autoExposure?1u<<31:0u);
    }
    bool frameValid()const {
        return frame.slot<METAL_FRAMES_IN_FLIGHT && frame.view<options.views &&
            metalfx_denoise::validSemantics(frame.semantics) &&
            metalfx_denoise::validExtent(frame.extent,minimumScale,maximumScale) &&
            finiteMatrix(frame.worldToView)&&finiteMatrix(frame.viewToClip) &&
            std::isfinite(frame.jitterPixels.x)&&std::isfinite(frame.jitterPixels.y) &&
            std::isfinite(frame.preExposure)&&frame.preExposure>0 &&
            std::isfinite(options.manualExposure)&&options.manualExposure>=0x1p-24f&&options.manualExposure<=65504.f;
    }
#if PHOSPHOR_HAS_METALFX_DENOISED
    void request(View& v) {
        auto* d=MTLFX::TemporalDenoisedScalerDescriptor::alloc();
        if(!d){fail(v,Status::MissingSDK,"Denoised runtime descriptor class unavailable");return;}
        d=d->init();
        if(!d){fail(v,Status::FactoryRejected,"Denoised descriptor initialization failed");return;}
        const auto e=v.desired.extent;
        d->setInputWidth(e.inputWidth);d->setInputHeight(e.inputHeight);
        d->setOutputWidth(e.outputWidth);d->setOutputHeight(e.outputHeight);
        d->setColorTextureFormat(toMetalFormat(metalfx_denoise::format(Channel::Color)));
        d->setDepthTextureFormat(toMetalFormat(metalfx_denoise::format(Channel::Depth)));
        d->setMotionTextureFormat(toMetalFormat(metalfx_denoise::format(Channel::Motion)));
        d->setDiffuseAlbedoTextureFormat(toMetalFormat(metalfx_denoise::format(Channel::DiffuseAlbedo)));
        d->setSpecularAlbedoTextureFormat(toMetalFormat(metalfx_denoise::format(Channel::SpecularAlbedo)));
        d->setNormalTextureFormat(toMetalFormat(metalfx_denoise::format(Channel::Normal)));
        d->setRoughnessTextureFormat(toMetalFormat(metalfx_denoise::format(Channel::Roughness)));
        d->setOutputTextureFormat(toMetalFormat(metalfx_denoise::format(Channel::Output)));
        d->setReactiveMaskTextureEnabled(options.reactiveMask);
        d->setReactiveMaskTextureFormat(toMetalFormat(metalfx_denoise::format(Channel::Reactive)));
        d->setSpecularHitDistanceTextureEnabled(options.specularHitDistance);
        d->setSpecularHitDistanceTextureFormat(toMetalFormat(metalfx_denoise::format(Channel::HitDistance)));
        d->setDenoiseStrengthMaskTextureEnabled(options.strengthMask);
        d->setDenoiseStrengthMaskTextureFormat(toMetalFormat(metalfx_denoise::format(Channel::Strength)));
        d->setTransparencyOverlayTextureEnabled(false);d->setAutoExposureEnabled(options.autoExposure);
        stats.descriptorConfigured=true;stats.autoExposureEnabled=d->isAutoExposureEnabled();
        if(stats.autoExposureEnabled!=options.autoExposure) {
            d->release();fail(v,Status::FactoryRejected,"SDK descriptor did not retain requested exposure mode");return;
        }
        // Full synchronous framework compilation belongs on the gateway worker,
        // never on this render thread. The callback returns a queue future.
        d->setRequiresSynchronousInitialization(true);
        // Deliberately no ordinary-scaler inputContentWidth/Height or descriptor
        // guessed dynamic-content setters: absent from the denoised SDK contract.
        try {
            v.pending=factory.request(d);v.requested=v.desired;++stats.requests;
            if(!v.pending.valid())fail(v,Status::FactoryRejected,"Gateway returned an invalid denoised future");
        } catch(const std::exception& e) {fail(v,Status::FactoryRejected,e.what());}
        catch(...) {fail(v,Status::FactoryRejected,"Denoised factory request failed");}
        d->release();
    }
    void queryUsage(View& v) {
        auto* s=v.scaler.get();
        v.usage[channelIndex(Channel::Color)]=s->colorTextureUsage();
        v.usage[channelIndex(Channel::Depth)]=s->depthTextureUsage();
        v.usage[channelIndex(Channel::Motion)]=s->motionTextureUsage();
        v.usage[channelIndex(Channel::DiffuseAlbedo)]=s->diffuseAlbedoTextureUsage();
        v.usage[channelIndex(Channel::SpecularAlbedo)]=s->specularAlbedoTextureUsage();
        v.usage[channelIndex(Channel::Normal)]=s->normalTextureUsage();
        v.usage[channelIndex(Channel::Roughness)]=s->roughnessTextureUsage();
        v.usage[channelIndex(Channel::HitDistance)]=options.specularHitDistance?s->specularHitDistanceTextureUsage():MTL::TextureUsageUnknown;
        v.usage[channelIndex(Channel::Reactive)]=options.reactiveMask?s->reactiveTextureUsage():MTL::TextureUsageUnknown;
        v.usage[channelIndex(Channel::Strength)]=options.strengthMask?s->denoiseStrengthMaskTextureUsage():MTL::TextureUsageUnknown;
        v.usage[channelIndex(Channel::Exposure)]=MTL::TextureUsageShaderRead;
        v.usage[channelIndex(Channel::Output)]=s->outputTextureUsage();
        v.usage[channelIndex(Channel::RestoredOutput)]=MTL::TextureUsageShaderRead|MTL::TextureUsageShaderWrite;
        // Depth32Float cannot be a generic compute storage destination. If an
        // unexpected framework asks that usage, choose the custom path.
        if(v.usage[channelIndex(Channel::Depth)]&MTL::TextureUsageShaderWrite)
            throw std::runtime_error("Denoised depth usage requires unsupported Depth32Float shader writes");
        const auto e=v.desired.extent;
        if(s->inputWidth()!=e.inputWidth||s->inputHeight()!=e.inputHeight||
           s->outputWidth()!=e.outputWidth||s->outputHeight()!=e.outputHeight)
            throw std::runtime_error("Gateway returned a denoised scaler with mismatched dimensions");
        const std::array<MTL::PixelFormat,ChannelCount> actual{
            s->colorTextureFormat(),s->depthTextureFormat(),s->motionTextureFormat(),
            s->diffuseAlbedoTextureFormat(),s->specularAlbedoTextureFormat(),s->normalTextureFormat(),
            s->roughnessTextureFormat(),s->specularHitDistanceTextureFormat(),s->reactiveMaskTextureFormat(),
            s->denoiseStrengthMaskTextureFormat(),MTL::PixelFormatR16Float,s->outputTextureFormat(),MTL::PixelFormatRGBA32Float};
        for(size_t i=0;i<ChannelCount;++i) {
            if((i==channelIndex(Channel::HitDistance)&&!options.specularHitDistance) ||
               (i==channelIndex(Channel::Reactive)&&!options.reactiveMask) ||
               (i==channelIndex(Channel::Strength)&&!options.strengthMask))continue;
            if(actual[i]!=toMetalFormat(metalfx_denoise::Formats[i]))throw std::runtime_error("Denoised scaler format mismatch");
        }
    }
#endif
    void allocate(View& v) {
        const auto e=v.desired.extent;
        for(auto& slot:v.slots) {
            for(size_t i=0;i<ChannelCount;++i) {
                const Channel ch=Channel(i);const bool output=ch==Channel::Output||ch==Channel::RestoredOutput,exposure=ch==Channel::Exposure;
                const u32 w=exposure?1u:output?e.outputWidth:e.inputWidth;
                const u32 h=exposure?1u:output?e.outputHeight:e.inputHeight;
                auto* d=MTL::TextureDescriptor::texture2DDescriptor(toMetalFormat(metalfx_denoise::Formats[i]),w,h,false);
                d->setStorageMode(MTL::StorageModePrivate);
                auto usage=v.usage[i]|MTL::TextureUsageShaderRead;
                if(ch!=Channel::Depth)usage|=MTL::TextureUsageShaderWrite;
                else usage|=MTL::TextureUsageRenderTarget;
                d->setUsage(usage);
                slot.textures[i]=context.memory().newTexture(d,MemoryCategory::RenderTargets,metalfx_denoise::Names[i].data());
                if(!slot.textures[i])throw std::runtime_error("Denoised packed channel allocation failed");
                if((slot.textures[i]->usage()&v.usage[i])!=v.usage[i])throw std::runtime_error("Denoised texture usage mismatch");
            }
            slot.counters=context.memory().newBuffer(sizeof(GPUMetalfxDenoisePackCounters),MTL::ResourceStorageModeShared,
                                                      MemoryCategory::Other,"MetalFX pack readback checks");
            if(!slot.counters)throw std::runtime_error("Denoised check allocation failed");
            std::memset(slot.counters->contents(),0,sizeof(GPUMetalfxDenoisePackCounters));
            if(!slot.table) {
                auto* d=MTL4::ArgumentTableDescriptor::alloc()->init();d->setMaxBufferBindCount(2);d->setMaxTextureBindCount(20);
                NS::Error* error=nullptr;slot.table=context.device()->newArgumentTable(d,&error);d->release();
                if(!slot.table)throw std::runtime_error("Denoised pack argument table allocation failed");
            }
            if(!slot.clearTable) {
                auto* d=MTL4::ArgumentTableDescriptor::alloc()->init();d->setMaxBufferBindCount(1);
                NS::Error* error=nullptr;slot.clearTable=context.device()->newArgumentTable(d,&error);d->release();
                if(!slot.clearTable)throw std::runtime_error("Denoised check clear argument table allocation failed");
            }
            if(!slot.restoreTable) {
                auto* d=MTL4::ArgumentTableDescriptor::alloc()->init();d->setMaxBufferBindCount(1);d->setMaxTextureBindCount(2);
                NS::Error* error=nullptr;slot.restoreTable=context.device()->newArgumentTable(d,&error);d->release();
                if(!slot.restoreTable)throw std::runtime_error("Denoised radiance restore argument table allocation failed");
            }
        }
        if(!v.historyToken) {
            v.historyToken=context.memory().newBuffer(16,MTL::ResourceStorageModeShared,MemoryCategory::RenderTargets,"MetalFX opaque per-view history anchor");
            if(!v.historyToken)throw std::runtime_error("Denoised history anchor allocation failed");
            std::memset(v.historyToken->contents(),0,16);
        }
        ++graphVersion;
    }
    void collect(bool wait,bool install=true) {
        for(auto& v:views) {
            if(!v.pending.valid())continue;
            if(!wait&&v.pending.wait_for(std::chrono::seconds(0))!=std::future_status::ready)continue;
            Scaler s;
            const bool current=metalfx_denoise::acceptsResult(v.requested,v.desired)&&v.requested.pipelineGeneration==pipelines.generation();
            try{s=v.pending.get();}
            catch(const std::exception& e){if(!current)++stats.discardedRequests;else fail(v,Status::FactoryRejected,e.what());continue;}
            catch(...){if(!current)++stats.discardedRequests;else fail(v,Status::FactoryRejected,"Denoised gateway future failed");continue;}
            if(!current||!install) {
                ++stats.discardedRequests;retire(std::move(s),false);continue;
            }
            if(!s){fail(v,Status::FactoryRejected,"Denoised descriptor/factory rejected candidate formats");continue;}
            retire(std::move(v.scaler),v.used);v.scaler=std::move(s);v.used=false;
            releaseTargets(v);
#if PHOSPHOR_HAS_METALFX_DENOISED
            try{queryUsage(v);allocate(v);}
            catch(const std::exception& e){retire(std::move(v.scaler),false);releaseTargets(v);fail(v,Status::UsageRejected,e.what());}
#endif
        }
    }
    void prepare(const Frame& f) {
        frame=f;collectChecks();
        if(f.view<options.views&&f.slot<METAL_FRAMES_IN_FLIGHT) {
            auto& old=views[f.view].slots[f.slot];
            if(old.recorded&&!old.consumed) {
                const auto completed=checks(f.view,f.slot);
                if(!completed.available)throw std::logic_error("Denoised frame slot recycled before its GPU checks completed");
                completedChecks.push_back(completed);old.consumed=true;
            }
        }
        if(!options.enabled){setStatus(Status::Disabled,"MetalFX denoised disabled; caller custom signal path");return;}
        if(!stats.sdkAvailable){setStatus(Status::MissingSDK,"Built without Metal4FX TemporalDenoisedScaler SDK headers");++stats.fallbackFrames;return;}
        if(!stats.deviceSupported){setStatus(Status::UnsupportedDevice,"Runtime/device does not support Metal4FX denoised");++stats.fallbackFrames;return;}
        if(!stats.factoryInstalled){setStatus(Status::MissingFactory,"PipelineCache denoised gateway is not installed; custom fallback");++stats.fallbackFrames;return;}
        if(!frameValid()){setStatus(Status::InvalidContract,"Invalid SDK extent/channel-space/matrix/exposure contract");++stats.fallbackFrames;return;}
        if(f.preExposure!=1.0f&&options.sdkOutputScale==Options::OutputScale::Unverified) {
            setStatus(Status::UnverifiedExposureMapping,"Nonunit pre-exposure requires an explicit SDK output-unit scalar/impulse fixture; custom fallback");
            ++stats.fallbackFrames;return;
        }
        auto& v=views[f.view];
        const metalfx_denoise::RequestKey desired{f.extent,pipelines.generation(),descriptorFlags()};
        if(!v.desiredValid || !(desired==v.desired)) {
            retire(std::move(v.scaler),v.used);v.used=false;releaseTargets(v);
            v.desired=desired;v.desiredValid=true;v.stableFrames=0;v.failed=false;
            histories.invalidate(f.view,"MetalFX denoised extent/pipeline change");++graphVersion;
        } else if(v.stableFrames<options.resizeSettleFrames)++v.stableFrames;
        collect(false);
        const auto d=histories.begin(f.view,{f.extent.inputWidth,f.extent.inputHeight,f.extent.outputWidth,f.extent.outputHeight},
                                     f.signalEpoch,f.cut,f.reset);
        reset=d.reset;
#if PHOSPHOR_HAS_METALFX_DENOISED
        if(!v.scaler&&!v.pending.valid()&&!v.failed && v.stableFrames>=options.resizeSettleFrames)request(v);
#endif
        publishPrepared(v);
    }
    void publishPrepared(View& v) {
        if(v.failed){setStatus(v.failureStatus,v.failureReason);++stats.fallbackFrames;return;}
        if(!v.scaler) {
            setStatus(v.pending.valid()?Status::Pending:Status::Settling,v.pending.valid()?"Denoised gateway request pending":"Denoised input/output extent settling");
            ++stats.fallbackFrames;return;
        }
        if(!pipelines.compute(packHandle)||!pipelines.compute(clearHandle)||!pipelines.compute(restoreHandle)){setStatus(Status::Pending,"Denoised guide-pack/restore pipelines pending");++stats.fallbackFrames;return;}
        setStatus(Status::Ready,"");
        GPUMetalfxDenoisePackParams pp{frame.extent.inputWidth,frame.extent.inputHeight,flags(),0,options.manualExposure,0.002f,frame.preExposure,0};
        auto slice=context.frameUploads().allocate(sizeof(pp));std::memcpy(slice.cpu,&pp,sizeof(pp));packParamsAddress=slice.gpu;
        GPUMetalfxRestoreParams rp{frame.extent.outputWidth,frame.extent.outputHeight,1.0f/frame.preExposure,0};
        auto restore=context.frameUploads().allocate(sizeof(rp));std::memcpy(restore.cpu,&rp,sizeof(rp));restoreParamsAddress=restore.gpu;
    }
    bool prewarm(u32 activeViews,std::chrono::milliseconds budget) {
        if(!activeViews||activeViews>options.views||frame.view>=activeViews||budget.count()<=0||budget.count()>120000)
            throw std::invalid_argument("Invalid initial fixture prewarm views/budget");
        // First real prepare has already applied SDK/device/factory, semantic,
        // matrix and output-policy validation. Terminal diagnostics are retained.
        if(currentStatus!=Status::Pending&&currentStatus!=Status::Settling&&currentStatus!=Status::Ready)return false;
        if(!frameValid()||stats.encodedFrames)throw std::logic_error("Fixture prewarm requires an unencoded canonical frame");
        const auto deadline=std::chrono::steady_clock::now()+budget;
        const metalfx_denoise::RequestKey key{frame.extent,pipelines.generation(),descriptorFlags()};
        for(u32 i=0;i<activeViews;++i) {
            auto& v=views[i];
            if(v.used||(v.desiredValid&&v.desired!=key)||(v.pending.valid()&&v.requested!=key)) {
                setStatus(Status::FixturePrewarmSuperseded,"Fixture initial prewarm key was superseded before waiting");return false;
            }
            if(v.failed){setStatus(v.failureStatus,v.failureReason);return false;}
            if(!v.desiredValid){v.desired=key;v.desiredValid=true;histories.invalidate(i,"Fixture initial prewarm");++graphVersion;}
#if PHOSPHOR_HAS_METALFX_DENOISED
            if(!v.scaler&&!v.pending.valid())request(v);
#endif
            if(v.failed){setStatus(v.failureStatus,v.failureReason);return false;}
        }
        for(u32 i=0;i<activeViews;++i) {
            auto& v=views[i];
            if(v.pending.valid()&&v.pending.wait_until(deadline)!=std::future_status::ready) {
                setStatus(Status::FixturePrewarmTimeout,"Fixture initial denoised future exceeded bounded prewarm deadline; initialization is not cancelled");return false;
            }
        }
        // Use ordinary stale-result/usage/format validation. This never invokes
        // the unbounded collect(true) or queue waitAllFinal/drain APIs.
        pipelines.beginFrame();
        if(pipelines.generation()!=key.pipelineGeneration){setStatus(Status::FixturePrewarmSuperseded,"Fixture initial prewarm generation was superseded before result collection");return false;}
        collect(false);
        for(u32 i=0;i<activeViews;++i) {
            const auto& v=views[i];
            if(v.desired!=key||pipelines.generation()!=key.pipelineGeneration) {
                setStatus(Status::FixturePrewarmSuperseded,"Fixture initial prewarm generation/extent was superseded");return false;
            }
            if(v.failed){setStatus(v.failureStatus,v.failureReason);return false;}
            if(!v.scaler){setStatus(Status::FactoryRejected,"Fixture initial prewarm produced no native scaler");return false;}
        }
        // Refresh ONLY the already prepared frame's uniforms/status. Do not
        // repeat history.begin, settle counts, phase/cut bookkeeping or uploads
        // for synthetic rendered frames while the model compiles.
        publishPrepared(views[frame.view]);return currentStatus==Status::Ready;
    }
    bool sourceValid(const rg::RenderGraph& g,rg::TextureRef ref,Channel ch)const {
        if(!ref.valid()||ref.resource>=g.resources().size())return false;
        const auto& r=g.resources()[ref.resource];const auto& t=r.texture;
        if(r.kind!=rg::ResourceKind::Texture||t.width<frame.extent.inputWidth||t.height<frame.extent.inputHeight||
           t.sampleCount!=1||t.depth!=1)return false;
        if(ch==Channel::Depth)return t.format==rg::Format::Depth32Float;
        if(ch==Channel::Reactive||ch==Channel::Strength)return t.format==rg::Format::R8Unorm||t.format==rg::Format::R16Float||t.format==rg::Format::R32Float;
        if(ch==Channel::Roughness||ch==Channel::HitDistance)return t.format==rg::Format::R16Float||t.format==rg::Format::R32Float;
        if(ch==Channel::Motion)return t.format==rg::Format::RG16Float||t.format==rg::Format::RG32Float;
        return t.format==rg::Format::RGBA16Float||t.format==rg::Format::RGBA32Float;
    }
    rg::TextureRef add(rg::RenderGraph& g,const Inputs& in) {
        inputs=in;graphNative=false;
        if(!in.customFallback.valid())throw std::invalid_argument("Denoised adapter requires an actual custom fallback texture");
        if(currentStatus!=Status::Ready)return in.customFallback;
        bool valid=sourceValid(g,in.noisyColor,Channel::Color)&&sourceValid(g,in.worldNormal,Channel::Normal)&&sourceValid(g,in.roughness,Channel::Roughness)&&
            sourceValid(g,in.diffuseAlbedo,Channel::DiffuseAlbedo)&&sourceValid(g,in.specularAlbedo,Channel::SpecularAlbedo)&&
            sourceValid(g,in.motion,Channel::Motion)&&sourceValid(g,in.depth,Channel::Depth);
        if(options.specularHitDistance)valid=valid&&sourceValid(g,in.hitDistance,Channel::HitDistance);
        if(options.reactiveMask)valid=valid&&sourceValid(g,in.reactiveMask,Channel::Reactive);
        if(options.strengthMask)valid=valid&&sourceValid(g,in.strengthMask,Channel::Strength);
        if(!valid) {
            histories.invalidate(frame.view,"MetalFX missing/invalid channel");
            setStatus(Status::InvalidContract,"Missing SDK guide or incompatible format/active region; custom fallback");
            ++stats.fallbackFrames;return in.customFallback;
        }
        using namespace rg;const auto e=frame.extent;auto& v=views[frame.view];auto& slot=v.slots[frame.slot];
        neutralRef=g.importTexture("MetalFX optional neutral guide",{Format::RGBA16Float,1,1},ImportContentsDefined);
        for(size_t i=0;i<ChannelCount;++i) {
            const bool output=i==channelIndex(Channel::Output)||i==channelIndex(Channel::RestoredOutput),exposure=i==channelIndex(Channel::Exposure);
            packedRefs[i]=g.importTexture(std::string(metalfx_denoise::Names[i]),{metalfx_denoise::Formats[i],
                exposure?1u:output?e.outputWidth:e.inputWidth,exposure?1u:output?e.outputHeight:e.inputHeight},ImportPerFrame|ImportOutput);
        }
        counterRef=g.importBuffer("MetalFX guide pack check counters",{sizeof(GPUMetalfxDenoisePackCounters)},ImportPerFrame|ImportOutput);
        historyRef=g.importBuffer("MetalFX denoised opaque history anchor",{16},ImportContentsDefined|ImportOutput);
        g.addPass("MetalFX guide check clear",PassType::Compute,[&](PassBuilder& b){
            counterRef=b.write(counterRef,Usage::ShaderWrite,StageDispatch);b.setProfileShaders("denoise_pack_clear");
        },[this](PassContext& ctx){
            auto& slot=views[frame.view].slots[frame.slot];auto* enc=static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder());
            slot.clearTable->setAddress(slot.counters->gpuAddress(),0);enc->setComputePipelineState(pipelines.compute(clearHandle));
            enc->setArgumentTable(slot.clearTable);enc->dispatchThreads(MTL::Size::Make(8,1,1),MTL::Size::Make(8,1,1));
        });
        g.addPass("MetalFX exact active depth crop",PassType::Blit,[&](PassBuilder& b){
            b.read(inputs.depth,Usage::CopySrc,StageBlit);packedRefs[channelIndex(Channel::Depth)]=b.write(packedRefs[channelIndex(Channel::Depth)],Usage::CopyDst,StageBlit);
        },[this](PassContext& ctx){
            auto* enc=static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder());
            enc->copyFromTexture(static_cast<MTL::Texture*>(ctx.texture(inputs.depth)),0,0,MTL::Origin::Make(0,0,0),
                MTL::Size::Make(frame.extent.inputWidth,frame.extent.inputHeight,1),
                views[frame.view].slots[frame.slot].textures[channelIndex(Channel::Depth)],0,0,MTL::Origin::Make(0,0,0));
        });
        g.addPass("MetalFX exact active guide pack",PassType::Compute,[&](PassBuilder& b){
            for(auto t:{inputs.noisyColor,inputs.worldNormal,inputs.roughness,inputs.diffuseAlbedo,inputs.specularAlbedo,inputs.motion,inputs.depth,neutralRef})
                b.read(t,Usage::ShaderRead,StageDispatch);
            if(options.specularHitDistance)b.read(inputs.hitDistance,Usage::ShaderRead,StageDispatch);
            if(options.reactiveMask)b.read(inputs.reactiveMask,Usage::ShaderRead,StageDispatch);
            if(options.strengthMask)b.read(inputs.strengthMask,Usage::ShaderRead,StageDispatch);
            b.read(counterRef,Usage::ShaderRead,StageDispatch);counterRef=b.write(counterRef,Usage::ShaderWrite,StageDispatch);
            for(size_t i=0;i<channelIndex(Channel::Output);++i)if(i!=channelIndex(Channel::Depth))
                packedRefs[i]=b.write(packedRefs[i],Usage::ShaderWrite,StageDispatch);
            b.setProfileShaders("denoise_pack");
        },[this](PassContext& ctx){encodePack(ctx);});
        g.addPass("MetalFX temporal denoised HDR",PassType::External,[&](PassBuilder& b){
            for(size_t i=0;i<channelIndex(Channel::Output);++i)b.read(packedRefs[i],Usage::ShaderRead,StageExternal);
            b.read(historyRef,Usage::ShaderRead,StageExternal);historyRef=b.write(historyRef,Usage::ShaderWrite,StageExternal);
            packedRefs[channelIndex(Channel::Output)]=b.write(packedRefs[channelIndex(Channel::Output)],Usage::ShaderWrite,StageExternal);
        },[this](PassContext& ctx){encodeEffect(ctx);});
        g.addPass("MetalFX physical radiance restore",PassType::Compute,[&](PassBuilder& b){
            b.read(packedRefs[channelIndex(Channel::Output)],Usage::ShaderRead,StageDispatch);
            packedRefs[channelIndex(Channel::RestoredOutput)]=b.write(packedRefs[channelIndex(Channel::RestoredOutput)],Usage::ShaderWrite,StageDispatch);
            b.setProfileShaders("denoise_restore_radiance");
        },[this](PassContext& ctx){
            auto& slot=views[frame.view].slots[frame.slot];auto* t=slot.restoreTable;
            t->setAddress(restoreParamsAddress,0);t->setTexture(slot.textures[channelIndex(Channel::Output)]->gpuResourceID(),0);
            t->setTexture(slot.textures[channelIndex(Channel::RestoredOutput)]->gpuResourceID(),1);
            auto* enc=static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder());enc->setComputePipelineState(pipelines.compute(restoreHandle));enc->setArgumentTable(t);
            enc->dispatchThreads(MTL::Size::Make(frame.extent.outputWidth,frame.extent.outputHeight,1),MTL::Size::Make(8,8,1));
        });
        graphNative=true;return packedRefs[channelIndex(Channel::RestoredOutput)];
    }
    void encodePack(rg::PassContext& ctx) {
        auto& slot=views[frame.view].slots[frame.slot];auto* t=slot.table;
        t->setAddress(packParamsAddress,0);t->setAddress(slot.counters->gpuAddress(),1);
        auto input=[&](rg::TextureRef r,u32 binding){auto* texture=r.valid()?static_cast<MTL::Texture*>(ctx.texture(r)):neutral;t->setTexture(texture->gpuResourceID(),binding);};
        input(inputs.noisyColor,0);input(inputs.worldNormal,1);input(inputs.roughness,2);input(inputs.diffuseAlbedo,3);
        input(inputs.specularAlbedo,4);input(inputs.motion,5);input(options.specularHitDistance?inputs.hitDistance:rg::TextureRef{},6);
        input(options.reactiveMask?inputs.reactiveMask:rg::TextureRef{},7);input(options.strengthMask?inputs.strengthMask:rg::TextureRef{},16);input(inputs.depth,19);
        const std::array<std::pair<Channel,u32>,10> outputs{{{Channel::Color,8},{Channel::Normal,9},{Channel::Roughness,10},
            {Channel::DiffuseAlbedo,11},{Channel::SpecularAlbedo,12},{Channel::Motion,13},{Channel::HitDistance,14},
            {Channel::Reactive,15},{Channel::Strength,17},{Channel::Exposure,18}}};
        for(auto [ch,binding]:outputs)t->setTexture(slot.textures[channelIndex(ch)]->gpuResourceID(),binding);
        auto* enc=static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder());enc->setComputePipelineState(pipelines.compute(packHandle));enc->setArgumentTable(t);
        enc->dispatchThreads(MTL::Size::Make(frame.extent.inputWidth,frame.extent.inputHeight,1),MTL::Size::Make(8,8,1));
        slot.recorded=true;slot.consumed=false;slot.recordedFrame=frame.index;
        slot.expectedPixels=frame.extent.inputWidth*frame.extent.inputHeight;
    }
    void encodeEffect(rg::PassContext& ctx) {
#if PHOSPHOR_HAS_METALFX_DENOISED
        auto& v=views[frame.view];auto& slot=v.slots[frame.slot];auto* s=v.scaler.get();
        if(!s)throw std::logic_error("Denoised graph executed without its per-view scaler");
        auto get=[&](Channel c){return slot.textures[channelIndex(c)];};
        s->setColorTexture(get(Channel::Color));s->setDepthTexture(get(Channel::Depth));s->setMotionTexture(get(Channel::Motion));
        s->setDiffuseAlbedoTexture(get(Channel::DiffuseAlbedo));s->setSpecularAlbedoTexture(get(Channel::SpecularAlbedo));
        s->setNormalTexture(get(Channel::Normal));s->setRoughnessTexture(get(Channel::Roughness));s->setOutputTexture(get(Channel::Output));
        s->setSpecularHitDistanceTexture(options.specularHitDistance?get(Channel::HitDistance):nullptr);
        s->setReactiveMaskTexture(options.reactiveMask?get(Channel::Reactive):nullptr);
        s->setDenoiseStrengthMaskTexture(options.strengthMask?get(Channel::Strength):nullptr);
        s->setTransparencyOverlayTexture(nullptr);s->setExposureTexture(get(Channel::Exposure));s->setPreExposure(frame.preExposure);
        const auto scale=metalfx_denoise::sdkMotionScale(),jitter=metalfx_denoise::sdkJitter(frame.jitterPixels);
        s->setMotionVectorScaleX(scale.x);s->setMotionVectorScaleY(scale.y);s->setJitterOffsetX(jitter.x);s->setJitterOffsetY(jitter.y);
        s->setDepthReversed(true);s->setShouldResetHistory(reset);
        simd::float4x4 worldToView{},viewToClip{};std::memcpy(&worldToView,frame.worldToView.data(),64);std::memcpy(&viewToClip,frame.viewToClip.data(),64);
        s->setWorldToViewMatrix(worldToView);s->setViewToClipMatrix(viewToClip);
        auto* cmd=static_cast<MTL4::CommandBuffer*>(ctx.commandBuffer());auto* fence=static_cast<MTL::Fence*>(ctx.externalFence());
        if(!cmd||!fence)throw std::logic_error("Denoised effect requires the render graph External boundary");
        s->setFence(fence);s->encodeToCommandBuffer(cmd);v.used=true;
        histories.read(frame.view,frame.index+1);histories.write(frame.view,frame.index+1,frame.viewToClip.data());
        ++stats.encodedFrames;if(reset)++stats.resets;
#else
        (void)ctx;throw std::logic_error("Denoised native graph cannot be active without SDK headers");
#endif
    }
    void bind(MetalGraphExecutor& e) {
        if(!graphNative||currentStatus!=Status::Ready)return;
        auto& v=views[frame.view];auto& slot=v.slots[frame.slot];
        e.bindTexture(neutralRef,neutral);
        for(size_t i=0;i<ChannelCount;++i)e.bindTexture(packedRefs[i],slot.textures[i]);
        e.bindBuffer(counterRef,slot.counters);e.bindBuffer(historyRef,v.historyToken);
    }
    PackCheck checks(u32 view,u32 slot)const {
        const auto& p=views.at(view).slots.at(slot);PackCheck out;
        if(!p.counters||!p.recorded||context.frameEvent()->signaledValue()<=p.recordedFrame)return out;
        out.available=true;out.frame=p.recordedFrame;out.view=view;out.slot=slot;std::memcpy(&out.counters,p.counters->contents(),sizeof(out.counters));
        const auto& n=out.counters;
        out.ok=n.pixels==p.expectedPixels &&
            !n.colorErrors&&!n.normalErrors&&!n.albedoErrors&&!n.roughnessErrors&&!n.motionErrors&&!n.hitErrors&&!n.maskErrors;
        return out;
    }
    PackCheck archiveCheck(const RetiredCheck& r)const {
        PackCheck out;out.available=true;out.frame=r.frame;out.view=r.view;out.slot=r.slot;
        std::memcpy(&out.counters,r.buffer->contents(),sizeof(out.counters));const auto& n=out.counters;
        out.ok=n.pixels==r.expectedPixels&&!n.colorErrors&&!n.normalErrors&&!n.albedoErrors&&
            !n.roughnessErrors&&!n.motionErrors&&!n.hitErrors&&!n.maskErrors;return out;
    }
    void collectChecks() {
        std::erase_if(retiredChecks,[&](const RetiredCheck& r){
            if(context.frameEvent()->signaledValue()<=r.frame)return false;
            completedChecks.push_back(archiveCheck(r));context.memory().release(r.buffer,MemoryCategory::Other);return true;
        });
    }
    std::vector<PackCheck> drainChecks() {
        collectChecks();
        for(u32 v=0;v<options.views;++v)for(u32 slot=0;slot<METAL_FRAMES_IN_FLIGHT;++slot) {
            auto& p=views[v].slots[slot];if(p.consumed)continue;
            auto result=checks(v,slot);if(result.available){completedChecks.push_back(result);p.consumed=true;}
        }
        std::vector<PackCheck> out;out.swap(completedChecks);return out;
    }
};

MetalfxDenoise::MetalfxDenoise(MetalContext& c,PipelineCache& p,Options o,Factory f):impl_(std::make_unique<Impl>(c,p,o,std::move(f))){}
MetalfxDenoise::~MetalfxDenoise()=default;
void MetalfxDenoise::prepareFrame(const Frame& f){impl_->prepare(f);}
bool MetalfxDenoise::prewarmPreparedFixture(u32 views,std::chrono::milliseconds budget){return impl_->prewarm(views,budget);}
rg::TextureRef MetalfxDenoise::addToGraph(rg::RenderGraph& g,const Inputs& in){return impl_->add(g,in);}
void MetalfxDenoise::bindFrame(MetalGraphExecutor& e){impl_->bind(e);}
bool MetalfxDenoise::ready()const{return impl_->currentStatus==Status::Ready;}
MetalfxDenoise::Status MetalfxDenoise::status()const{return impl_->currentStatus;}
const std::string& MetalfxDenoise::fallbackReason()const{return impl_->reason;}
const MetalfxDenoise::Stats& MetalfxDenoise::stats()const{return impl_->stats;}
u64 MetalfxDenoise::version()const{return impl_->graphVersion;}
MetalfxDenoise::PackCheck MetalfxDenoise::readPackChecks(u32 view,u32 slot)const{return impl_->checks(view,slot);}
std::vector<MetalfxDenoise::PackCheck> MetalfxDenoise::drainPackChecks(){return impl_->drainChecks();}
rg::TextureRef MetalfxDenoise::sdkOutputRef()const{return impl_->graphNative&&ready()?impl_->packedRefs[channelIndex(Channel::Output)]:rg::TextureRef{};}
rg::TextureRef MetalfxDenoise::packedChannelRef(metalfx_denoise::Channel channel)const {
    return impl_->graphNative&&ready()&&channelIndex(channel)<ChannelCount?impl_->packedRefs[channelIndex(channel)]:rg::TextureRef{};
}
MTL::Texture* MetalfxDenoise::sdkOutputTexture(u32 view,u32 slot)const{return impl_->views.at(view).slots.at(slot).textures[channelIndex(Channel::Output)];}
} // namespace phosphor
