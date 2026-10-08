#pragma once
#include "platform/metal/metal_context.h"
#include "renderer/metalfx_denoise_contract.h"
#include "rendergraph/render_graph.h"
#include <array>
#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <string>
#include <vector>

namespace MTLFX {class TemporalDenoisedScalerDescriptor;}
namespace MTL4FX {class TemporalDenoisedScaler;}
namespace phosphor {
class PipelineCache;
class MetalGraphExecutor;
class MetalfxDenoise {
public:
    using Scaler=std::shared_ptr<MTL4FX::TemporalDenoisedScaler>;
    struct Factory {
        // Must route creation through PipelineCache's existing utility queue and
        // compiler; copy descriptor before returning. No std::async/compiler
        // created by the adapter, and no cast from ordinary TemporalScaler.
        std::function<std::future<Scaler>(MTLFX::TemporalDenoisedScalerDescriptor*)> request;
        // Called after GPU use completes; schedule standard release on workers.
        std::function<void(Scaler)> retire;
    };
    struct Options {
        enum class OutputScale : u8 { Unverified,PreExposed };
        bool enabled=false, reactiveMask=true, specularHitDistance=false, strengthMask=false;
        // Diagnostic fixture opt-in; production retains manual exposure.
        bool autoExposure=false;
        float manualExposure=1.0f; // SDK exposure hint; does not alter color packing or output restoration
        OutputScale sdkOutputScale=OutputScale::Unverified;
        metalfx_denoise::RadiometricDomain radiometricDomain=metalfx_denoise::RadiometricDomain::UnqualifiedSceneLinear;
        u32 views=1,resizeSettleFrames=4;
    };
    struct Frame {
        u32 slot=0,view=0;u64 index=0,signalEpoch=0;
        metalfx_denoise::Extent extent{};
        std::array<float,16> worldToView{1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
        std::array<float,16> viewToClip{1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1}; // unjittered, reverse-Z
        glm::vec2 jitterPixels{};float preExposure=1;
        bool cut=false,reset=false;
        metalfx_denoise::Semantics semantics{};
    };
    struct Inputs {
        rg::TextureRef noisyColor{},depth{},motion{},diffuseAlbedo{},specularAlbedo{},worldNormal{},roughness{};
        rg::TextureRef hitDistance{},reactiveMask{},strengthMask{},customFallback{};
    };
    enum class Status : u8 { Disabled,MissingSDK,UnsupportedDevice,MissingFactory,InvalidContract,
                            Settling,Pending,FactoryRejected,UsageRejected,UnverifiedExposureMapping,Ready,FixturePrewarmTimeout,FixturePrewarmSuperseded,UnqualifiedRadiometricDomain };
    struct Stats {
        bool requested=false,sdkAvailable=false,deviceSupported=false,factoryInstalled=false;
        // Read back from the actual descriptor, not inferred from caller intent.
        bool descriptorConfigured=false,autoExposureEnabled=false;
        metalfx_denoise::RadiometricDomain radiometricDomain=metalfx_denoise::RadiometricDomain::UnqualifiedSceneLinear;
        u64 requests=0,encodedFrames=0,fallbackFrames=0,resets=0,discardedRequests=0,retirements=0;
    };
    struct PackCheck {
        bool available=false,ok=false;u64 frame=0;u32 view=0,slot=0;
        GPUMetalfxDenoisePackCounters counters{};
    };
    MetalfxDenoise(MetalContext&,PipelineCache&,Options,Factory={});
    ~MetalfxDenoise();
    MetalfxDenoise(const MetalfxDenoise&)=delete;
    MetalfxDenoise& operator=(const MetalfxDenoise&)=delete;
    void prepareFrame(const Frame&);
    // Diagnostic-only initial prewarm AFTER one real prepare inside beginFrame.
    // Uses its exact current extent/phase; active view futures share one deadline.
    // No GPU frame/history advance, compiler creation or production-path wait.
    bool prewarmPreparedFixture(u32 activeViews,std::chrono::milliseconds budget);
    // Diagnostic lifecycle fixture ONLY: wait for actual reload/resize futures
    // under one total deadline, preserving the current logical frame/history.
    // Requires resizeSettleFrames=0. Production never calls this method.
    bool waitPreparedLifecycleFixture(u32 activeViews,std::chrono::milliseconds budget);
    // If not ready, this returns the caller's actual custom-denoised composite.
    // No phantom native pass, and no second energy contribution is introduced.
    [[nodiscard]] rg::TextureRef addToGraph(rg::RenderGraph&,const Inputs&);
    void bindFrame(MetalGraphExecutor&);
    [[nodiscard]] bool ready()const;
    [[nodiscard]] Status status()const;
    [[nodiscard]] const std::string& fallbackReason()const;
    [[nodiscard]] const Stats& stats()const;
    [[nodiscard]] u64 version()const;
    [[nodiscard]] PackCheck readPackChecks(u32 view,u32 slot)const; // only after GPU completion
    [[nodiscard]] std::vector<PackCheck> drainPackChecks(); // includes retired resize/view records
    // Borrowed current-frame SDK output only; consumers declare its graph read
    // and preserve frame/view/slot tags before copying to completed readback.
    [[nodiscard]] rg::TextureRef sdkOutputRef()const;
    [[nodiscard]] rg::TextureRef packedChannelRef(metalfx_denoise::Channel)const;
    [[nodiscard]] MTL::Texture* sdkOutputTexture(u32 view,u32 slot)const;
private:
    struct Impl;std::unique_ptr<Impl> impl_;
};
} // namespace phosphor
