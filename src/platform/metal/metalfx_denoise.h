#pragma once
#include "platform/metal/metal_context.h"
#include "renderer/metalfx_denoise_contract.h"
#include "rendergraph/render_graph.h"
#include <array>
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
        OutputScale sdkOutputScale=OutputScale::Unverified;
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
                            Settling,Pending,FactoryRejected,UsageRejected,UnverifiedExposureMapping,Ready };
    struct Stats {
        bool requested=false,sdkAvailable=false,deviceSupported=false,factoryInstalled=false;
        u64 requests=0,encodedFrames=0,fallbackFrames=0,resets=0,discardedRequests=0;
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
private:
    struct Impl;std::unique_ptr<Impl> impl_;
};
} // namespace phosphor
