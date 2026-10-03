#pragma once
#include "platform/metal/metal_context.h"
#include "pipeline/pipeline_registry.h"
#include "renderer/history_registry.h"
#include "renderer/exposure.h"
#include "renderer/post_layout.h"
#include "renderer/temporal_layout.h"
#include "rendergraph/render_graph.h"
#include <MetalFX/MetalFX.hpp>
#include <array>
#include <future>
#include <memory>

namespace phosphor {
class PipelineCache;
class VisibilityRenderer;
class PostProcessor {
  public:
    struct Options {
        bool temporal = false, autoExposure = false, corruptExposure = false, forceReset = false;
        u32 tonemap = 0, views = 1, jitterVariant = 0;
        float sharpening = 0, whitePoint = 4, debugMotionScale = 1;
    };
    PostProcessor(MetalContext &, PipelineCache &, const Options &);
    ~PostProcessor();
    void prewarm(u32 width, u32 height);
    GPUTemporalParams prepareFrame(u32 slot, u64 frame, u32 view, u32 inputWidth, u32 inputHeight, u32 width,
                                   u32 height, const float *matrix, u64 scene, bool cut, bool forceReset, float dt,
                                   float exposure, float headroom = 1);
    void finishFrame(const float *matrix);
    rg::TextureRef addToGraph(rg::RenderGraph &, VisibilityRenderer &, rg::TextureRef drawable,
                              rg::Format outputFormat = rg::Format::BGRA8Srgb);
    rg::TextureRef addSDRCapture(rg::RenderGraph &, rg::TextureRef display);
    void bindFrame(class MetalGraphExecutor &);
    [[nodiscard]] u32 inputWidth() const { return params_.inputWidth; }
    [[nodiscard]] u32 inputHeight() const { return params_.inputHeight; }
    [[nodiscard]] const HistoryRegistry &histories() const { return histories_; }
    [[nodiscard]] bool temporalReady() const;
    [[nodiscard]] u64 temporalFrames() const { return temporalFrames_; }
    [[nodiscard]] u64 fallbackFrames() const { return fallbackFrames_; }
    [[nodiscard]] u64 resetCount() const { return resets_; }
    [[nodiscard]] float lastExposure() const;
    [[nodiscard]] float manualExposure() const { return params_.manualExposure; }
    bool checkExposure(const std::array<u32, ExposureBins> &referenceHistogram,
                       const std::array<u32, ExposureBins> &low, const std::array<u32, ExposureBins> &high) const;
    [[nodiscard]] const char *effectiveUpscaler() const;

  private:
    void configure(u32 width, u32 height);
    void requestScaler(u32 view);
    void collectScalers(bool wait);
    void releaseTargets();
    void bind(rg::PassContext &, MTL4::ArgumentTable *);
    void encodeUpscale(rg::PassContext &);
    MetalContext &context_;
    PipelineCache &pipelines_;
    Options options_;
    HistoryRegistry histories_;
    struct View {
        std::shared_ptr<MTL4FX::TemporalScaler> scaler;
        std::future<std::shared_ptr<MTL4FX::TemporalScaler>> pending;
        u32 requestedWidth = 0, requestedHeight = 0;
        MTL::Buffer *exposureState = nullptr;
        MTL::Texture *exposure = nullptr;
        bool failed = false, exposureValid = false;
        u64 exposureScene = ~u64{0};
    };
    std::array<View, HistoryRegistry::MaxViews> views_{};
    struct Slot {
        MTL::Texture *output = nullptr;
        MTL::Buffer *histogram = nullptr;
    };
    std::array<Slot, METAL_FRAMES_IN_FLIGHT> slots_{};
    std::array<MTL4::ArgumentTable *, 6> tables_{};
    pipe::PipelineHandle clear_, histogram_, reduce_, native_, presentSDR_, presentEDR_, capture_;
    u32 width_ = 0, height_ = 0, slot_ = 0, view_ = 0;
    u64 frame_ = 0, temporalFrames_ = 0, fallbackFrames_ = 0, resets_ = 0;
    bool supported_ = false, reset_ = true;
    GPUTemporalParams temporal_{};
    GPUPostParams params_{};
    MTL::GPUAddress paramsAddress_ = 0;
    rg::TextureRef input_, depth_, motion_, reactive_, output_, exposure_;
    rg::BufferRef histogramRef_, stateRef_;
};
} // namespace phosphor
