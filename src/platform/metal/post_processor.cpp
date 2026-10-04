#include "platform/metal/post_processor.h"
#include "platform/metal/temporal_worker.h"
#include "platform/metal/pipeline_cache.h"
#include "platform/metal/visibility_renderer.h"
#include "platform/metal/metal_graph_executor.h"
#include "platform/metal/gpu_memory.h"
#include "platform/metal/upload_ring.h"
#include "rendergraph/pass_context.h"
#include "core/log.h"
#include "renderer/exposure.h"
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstring>
#include <stdexcept>

namespace phosphor {
namespace {
pipe::PipelineDesc kernel(const char *name) {
    pipe::PipelineDesc d;
    d.kind = pipe::PipelineKind::Compute;
    d.label = name;
    d.functions = {name, "", ""};
    return d;
}
MTL::Texture *texture(MetalContext &c, u32 w, u32 h, MTL::PixelFormat format, const char *label) {
    auto *d = MTL::TextureDescriptor::texture2DDescriptor(format, w, h, false);
    d->setStorageMode(MTL::StorageModePrivate);
    d->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite | MTL::TextureUsageRenderTarget);
    return c.memory().newTexture(d, MemoryCategory::RenderTargets, label);
}
std::array<double, 3> curveReference(u32 sample, u32 curve, double headroom) {
    const double value = std::exp2((double(sample) - 32) / 4);
    std::array<double, 3> input = sample & 1u ? std::array<double, 3>{value, value * 0.3, value * 0.05}
                                              : std::array<double, 3>{value, value, value};
    if (sample == 0 || sample == 5)
        input = {0, 0, 0};
    else if (sample <= 3) {
        input = {0, 0, 0};
        input[sample - 1] = 1;
    } else if (sample == 4)
        input = {65504, 65504, 65504};
    std::array<double, 3> result{};
    if (curve == 1) {
        constexpr double inset[3][3] = {{0.8424790623, 0.0784336, 0.0792237451},
                                        {0.0423282423, 0.8784686365, 0.0791661275},
                                        {0.0423756549, 0.0784336, 0.8791429738}};
        constexpr double outset[3][3] = {{1.1968790051, -0.0980208811, -0.0990297441},
                                         {-0.0528968518, 1.1519031299, -0.0989611768},
                                         {-0.0529716355, -0.0980434501, 1.1510736726}};
        double y[3]{};
        for (u32 c = 0; c < 3; ++c) {
            double mixed = 0;
            for (u32 k = 0; k < 3; ++k)
                mixed += inset[c][k] * input[k];
            const double x = std::clamp((std::log2(std::max(mixed, 1e-10)) + 12.47393) / 16.5, 0.0, 1.0);
            y[c] = 15.5 * std::pow(x, 6) - 40.14 * std::pow(x, 5) + 31.96 * std::pow(x, 4) - 6.868 * std::pow(x, 3) +
                   0.4298 * x * x + 0.1191 * x - 0.00232;
        }
        for (u32 c = 0; c < 3; ++c) {
            double mixed = 0;
            for (u32 k = 0; k < 3; ++k)
                mixed += outset[c][k] * y[k];
            result[c] = std::clamp(std::pow(std::max(mixed, 0.0), 2.2), 0.0, 1.0);
        }
    } else
        for (u32 c = 0; c < 3; ++c) {
            const double x = input[c];
            result[c] = std::clamp(curve == 0 ? (2.51 * x * x + 0.03 * x) / (2.43 * x * x + 0.59 * x + 0.14)
                                              : (x + x * x / 16) / (1 + x),
                                   0.0, 1.0);
        }
    for (u32 c = 0; c < 3; ++c) {
        const double highlight = std::max(input[c] - 1, 0.0);
        result[c] = std::clamp(result[c] + (headroom - 1) * highlight / (highlight + headroom), 0.0, headroom);
    }
    return result;
}
} // namespace
PostProcessor::PostProcessor(MetalContext &c, PipelineCache &p, const Options &o)
    : context_(c), pipelines_(p), options_(o) {
    if (!o.views || o.views > HistoryRegistry::MaxViews)
        throw std::invalid_argument("Invalid temporal view count");
    clear_ = p.request(kernel("exposure_clear"));
    histogram_ = p.request(kernel("exposure_histogram"));
    reduce_ = p.request(kernel("exposure_reduce"));
    native_ = p.request(kernel("post_native"));
    if (o.checkCurves || p.harvesting())
        curveProbe_ = p.request(kernel("post_curve_probe"));
    if (o.checkCurves)
        curveReadback_ = c.memory().newBuffer(576 * 16, MTL::ResourceStorageModeShared, MemoryCategory::Other,
                                              "Tone curve chart readback");
    for (u32 i = 0; i < 2; ++i) {
        pipe::PipelineDesc d;
        d.kind = pipe::PipelineKind::Render;
        d.label = i ? "EDR tone map" : "SDR tone map";
        d.functions = {"post_vs", "post_fs", ""};
        d.output(0, i ? rg::Format::RGBA16Float : rg::Format::BGRA8Srgb);
        (i ? presentEDR_ : presentSDR_) = p.request(d);
    }
    {
        pipe::PipelineDesc d;
        d.kind = pipe::PipelineKind::Render;
        d.label = "EDR capture to SDR";
        d.functions = {"post_vs", "post_capture_fs", ""};
        d.output(0, rg::Format::BGRA8Srgb);
        capture_ = p.request(d);
    }
    for (auto *&t : tables_) {
        auto *d = MTL4::ArgumentTableDescriptor::alloc()->init();
        d->setMaxBufferBindCount(3);
        d->setMaxTextureBindCount(6);
        NS::Error *e = nullptr;
        t = c.device()->newArgumentTable(d, &e);
        d->release();
        if (!t)
            throw std::runtime_error("Post argument table creation failed");
    }
    supported_ = MTLFX::TemporalScalerDescriptor::supportsMetal4FX(c.device());
    if (o.temporal && !supported_)
        LOG_WARN("MetalFX Metal 4 temporal unsupported; using native/spatial fallback");
    for (u32 i = 0; i < o.views; ++i) {
        auto &v = views_[i];
        v.exposureState =
            c.memory().newBuffer(16, MTL::ResourceStorageModeShared, MemoryCategory::Other, "Exposure state per view");
        const float initial[4] = {1, 1, 0, 0};
        std::memcpy(v.exposureState->contents(), initial, 16);
        v.exposure = texture(c, 1, 1, MTL::PixelFormatR16Float, "Exposure texture per view");
    }
    for (auto &s : slots_)
        s.histogram = c.memory().newBuffer(256 * sizeof(u32), MTL::ResourceStorageModeShared, MemoryCategory::Other,
                                           "Luminance histogram");
}
PostProcessor::~PostProcessor() {
    context_.waitIdle();
    context_.collectGarbage(); // runs the deferred scaler retirements, which use pipelines_
    context_.memory().release(curveReadback_, MemoryCategory::Other);
    releaseTargets();
    for (auto &v : views_) {
        v.scaler.reset();
        if (v.worker)
            v.worker->cancelPending();
        context_.memory().release(v.workerBuffer, MemoryCategory::Other);
        v.workerBuffer = nullptr;
        v.worker.reset();
        context_.memory().release(v.exposure, MemoryCategory::RenderTargets);
        context_.memory().release(v.exposureState, MemoryCategory::Other);
    }
    for (auto &s : slots_)
        context_.memory().release(s.histogram, MemoryCategory::Other);
    for (auto *t : tables_)
        if (t)
            t->release();
}
void PostProcessor::releaseTargets() {
    for (auto &s : slots_) {
        context_.memory().release(s.output, MemoryCategory::RenderTargets);
        s.output = nullptr;
    }
}
void PostProcessor::configure(u32 w, u32 h) {
    if (width_ == w && height_ == h) {
        if (stableFrames_ < settleFrames())
            ++stableFrames_;
        if (stableFrames_ >= settleFrames())
            for (u32 i = 0; i < options_.views; ++i)
                if (views_[i].requestDeferred && !views_[i].pending.valid())
                    requestScaler(i);
        return;
    }
    // The first size is requested at once (prewarm waits for it); later ones
    // only after settleFrames() unchanged frames: a scaler per intermediate
    // live-resize size would be superseded before use (5.1 GB peak measured).
    stableFrames_ = width_ == 0 ? settleFrames() : 0;
    width_ = w;
    height_ = h;
    releaseTargets();
    for (auto &s : slots_)
        s.output = texture(context_, w, h, MTL::PixelFormatRGBA16Float, "Reconstructed HDR");
    for (u32 i = 0; i < options_.views; ++i) {
        auto &v = views_[i];
        retireAfterFrames(std::move(v.scaler));
        if (usesWorker()) {
            if (v.worker) {
                v.worker->cancelPending();
                retiredWorkers_.push_back(v.worker);
            }
            context_.memory().release(v.workerBuffer, MemoryCategory::Other);
            v.workerBuffer = nullptr;
            v.worker = std::make_shared<TemporalWorker>(context_.device(), w, h);
            const auto keep = v.worker;
            v.workerBuffer = context_.memory().newSharedBuffer(
                keep->mapping(), keep->layout().mappedBytes,
                ^(void *, NS::UInteger) {
                  keep->retire();
                },
                MemoryCategory::Other, "MetalFX per-view shared bridge");
        }
        v.failed = false;
        histories_.invalidate(i, "output resize");
        if (!v.pending.valid() && !v.pendingWorker.valid())
            scheduleScaler(i);
    }
}
void PostProcessor::retireAfterFrames(std::shared_ptr<MTL4FX::TemporalScaler> scaler) {
    if (scaler) // destroyed off the render thread once its frames have completed
        context_.deferCall([cache = &pipelines_, retired = std::move(scaler)]() mutable {
            cache->retireTemporalScaler(std::move(retired));
        });
}
void PostProcessor::scheduleScaler(u32 index) {
    if (stableFrames_ >= settleFrames())
        requestScaler(index);
    else
        views_[index].requestDeferred = true;
}
void PostProcessor::requestScaler(u32 index) {
    if (!options_.temporal || !supported_)
        return;
    auto &v = views_[index];
    v.requestDeferred = false;
    if (usesWorker()) {
        v.pendingWorker = pipelines_.requestTemporalWorker(v.worker);
        return;
    }
    auto *d = MTLFX::TemporalScalerDescriptor::alloc()->init();
    d->setInputWidth(width_);
    d->setInputHeight(height_);
    d->setOutputWidth(width_);
    d->setOutputHeight(height_);
    d->setColorTextureFormat(MTL::PixelFormatRGBA16Float);
    d->setOutputTextureFormat(MTL::PixelFormatRGBA16Float);
    d->setDepthTextureFormat(MTL::PixelFormatDepth32Float);
    d->setMotionTextureFormat(MTL::PixelFormatRG16Float);
    d->setReactiveMaskTextureEnabled(true);
    d->setReactiveMaskTextureFormat(MTL::PixelFormatR8Unorm);
    d->setAutoExposureEnabled(false);
    d->setInputContentPropertiesEnabled(true);
    d->setInputContentMinScale(1.0f);
    d->setInputContentMaxScale(2.0f);
    d->setRequiresSynchronousInitialization(true); // factory and compilation on utility workers
    v.requestedWidth = width_;
    v.requestedHeight = height_;
    v.pending = pipelines_.requestTemporalScaler(d);
    d->release();
}
void PostProcessor::collectScalers(bool wait) {
    std::erase_if(retiredWorkers_, [](const auto &w) { return w->finished(); });
    if (TemporalWorker::failureCount())
        throw std::runtime_error("MetalFX worker failed; aborting frame");
    for (u32 i = 0; i < options_.views; ++i) {
        auto &v = views_[i];
        if (usesWorker()) {
            if (!v.pendingWorker.valid())
                continue;
            if (!wait && v.pendingWorker.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
                continue;
            auto ready = v.pendingWorker.get();
            if (ready != v.worker) {
                requestScaler(i);
                continue;
            }
            if (!ready->ready()) {
                requestScaler(i);
                continue;
            }
            histories_.invalidate(i, "MetalFX worker ready");
            LOG_INFO("MetalFX isolated temporal view %u ready (%ux%u)", i, width_, height_);
            continue;
        }
        if (!v.pending.valid())
            continue;
        if (!wait && v.pending.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
            continue;
        auto scaler = v.pending.get();
        if (v.requestedWidth != width_ || v.requestedHeight != height_) {
            pipelines_.retireTemporalScaler(std::move(scaler)); // never encoded
            scheduleScaler(i);
            continue;
        }
        retireAfterFrames(std::move(v.scaler)); // never drop one frames in flight may use
        v.scaler = std::move(scaler);
        v.failed = !v.scaler;
        histories_.invalidate(i, v.failed ? "MetalFX unavailable" : "MetalFX ready");
        LOG_INFO("MetalFX temporal view %u: %s (%ux%u)", i, v.scaler ? "ready" : "native fallback", width_, height_);
    }
}
void PostProcessor::prewarm(u32 w, u32 h) {
    configure(w, h);
    collectScalers(true);
}
bool PostProcessor::temporalReady() const {
    return options_.temporal &&
           (usesWorker() ? views_[view_].worker && views_[view_].worker->ready() : bool(views_[view_].scaler));
}
const char *PostProcessor::effectiveUpscaler() const {
    return temporalReady() ? (usesWorker() ? "metalfx-temporal-isolated" : "metalfx-temporal") : "native-spatial";
}
float PostProcessor::lastExposure() const {
    return static_cast<const float *>(views_[view_].exposureState->contents())[0];
}
GPUTemporalParams PostProcessor::prepareFrame(u32 slot, u64 frame, u32 view, u32 iw, u32 ih, u32 w, u32 h,
                                              const float *matrix, u64 scene, bool cut, bool forceReset, float dt,
                                              float exposure, float headroom) {
    slot_ = slot;
    view_ = view;
    frame_ = frame;
    configure(w, h);
    collectScalers(false);
    const auto decision = histories_.begin(view, {iw, ih, w, h}, scene, cut, forceReset);
    reset_ = decision.reset;
    if (reset_)
        ++resets_;
    temporal_ = {};
    std::copy_n(matrix, 16, temporal_.currentViewProjection);
    const auto &state = histories_.get(view);
    std::copy_n(state.previousViewProjection.data(), 16, temporal_.previousViewProjection);
    temporal_.renderSize[0] = float(iw);
    temporal_.renderSize[1] = float(ih);
    temporal_.previousRenderSize[0] = float(state.extent.inputWidth);
    temporal_.previousRenderSize[1] = float(state.extent.inputHeight);
    temporal_.deltaTime = dt;
    temporal_.historyValid = !reset_;
    temporal_.viewIndex = view;
    temporal_.frameIndex = static_cast<u32>(frame);
    temporal_.manualExposure = exposure;
    temporal_.mipBias = options_.neutralMipBias ? 0.0f : std::min(0.0f, std::log2(float(iw) / float(w)));
    if (temporalReady()) {
        const auto j = temporalJitter(state.sample);
        temporal_.jitter[0] = j[0];
        temporal_.jitter[1] = j[1];
    }
    auto &viewState = views_[view];
    const bool exposureReset = !viewState.exposureValid || viewState.exposureScene != scene || cut;
    viewState.exposureValid = true;
    viewState.exposureScene = scene;
    params_ = {iw,
               ih,
               w,
               h,
               options_.tonemap,
               (options_.autoExposure ? 1u : 0u) | (options_.corruptExposure ? 2u : 0u) |
                   (options_.corruptCurves ? 4u : 0u),
               exposureReset ? 1u : 0u,
               temporalReady() ? 1u : 0u,
               dt,
               exposure,
               3.0f,
               options_.sharpening,
               std::max(1.0f, headroom),
               1.0f / 1024.0f,
               1024.0f,
               options_.whitePoint};
    const auto p = context_.frameUploads().allocate(sizeof(params_));
    std::memcpy(p.cpu, &params_, sizeof(params_));
    paramsAddress_ = p.gpu;
    return temporal_;
}
bool PostProcessor::checkExposure(const std::array<u32, ExposureBins> &referenceHistogram,
                                  const std::array<u32, ExposureBins> &low,
                                  const std::array<u32, ExposureBins> &high) const {
    std::array<u32, ExposureBins> histogram{};
    std::memcpy(histogram.data(), slots_[slot_].histogram->contents(), sizeof(histogram));
    const float expected = exposureTarget(histogram);
    const auto *state = static_cast<const float *>(views_[view_].exposureState->contents());
    bool histogramPass = true;
    u32 actual = 0, upper = 0, lower = 0;
    for (u32 i = 0; i < ExposureBins; ++i) {
        actual += histogram[i];
        upper += low[i];
        lower += high[i];
        histogramPass &= actual >= lower && actual <= upper;
    }
    const bool pass = histogramPass && std::isfinite(state[0]) && state[0] > 0 && std::isfinite(state[1]) &&
                      std::abs(state[1] - expected) <= std::max(0.00001f, std::abs(expected) * 0.0001f);
    std::printf("EXPOSURE view %u | target %.6f reference %.6f adapted %.6f | histogram %s | %s\n", view_,
                double(state[1]), double(expected), double(state[0]),
                histogram == referenceHistogram ? "exact"
                : histogramPass                 ? "bin-boundary rounding"
                                                : "mismatch",
                pass ? "PASS" : "FAIL");
    return pass;
}
void PostProcessor::finishFrame(const float *matrix) {
    histories_.read(view_, frame_ + 1);
    histories_.write(view_, frame_ + 1, matrix);
}
void PostProcessor::bind(rg::PassContext &ctx, MTL4::ArgumentTable *table) {
    table->setAddress(slots_[slot_].histogram->gpuAddress(), 0);
    table->setAddress(paramsAddress_, 1);
    table->setAddress(views_[view_].exposureState->gpuAddress(), 2);
    table->setTexture(static_cast<MTL::Texture *>(ctx.texture(input_))->gpuResourceID(), 0);
    table->setTexture(views_[view_].exposure->gpuResourceID(), 1);
    table->setTexture(static_cast<MTL::Texture *>(ctx.texture(depth_))->gpuResourceID(), 2);
}
void PostProcessor::encodeUpscale(rg::PassContext &ctx) {
    auto *cmd = static_cast<MTL4::CommandBuffer *>(ctx.commandBuffer());
    auto *fence = static_cast<MTL::Fence *>(ctx.externalFence());
    auto &v = views_[view_];
    auto *input = static_cast<MTL::Texture *>(ctx.texture(input_));
    if (temporalReady() && usesWorker()) {
        auto worker = v.worker;
        const auto &layout = worker->layout();
        auto *bridge = static_cast<MTL::Buffer *>(ctx.buffer(workerBridge_));
        auto *copy = cmd->computeCommandEncoder();
        copy->waitForFence(fence, MTL::StageBlit);
        const std::array<rg::TextureRef, 5> inputs = {input_, depth_, motion_, reactive_, exposure_};
        for (u32 i = 0; i < inputs.size(); ++i) {
            const bool exposure = i == temporal_worker::Exposure;
            copy->copyFromTexture(static_cast<MTL::Texture *>(ctx.texture(inputs[i])), 0, 0, MTL::Origin::Make(0, 0, 0),
                                  MTL::Size::Make(exposure ? 1 : width_, exposure ? 1 : height_, 1), bridge,
                                  slot_ * layout.slotBytes + layout.offsets[i], layout.rows[i],
                                  layout.rows[i] * (exposure ? 1 : height_));
        }
        copy->endEncoding();
        temporal_worker::Request request;
        request.delayMs = options_.debugWorkerDelayMs;
        request.slot = slot_;
        request.inputWidth = params_.inputWidth;
        request.inputHeight = params_.inputHeight;
        request.reset = reset_ || options_.forceReset;
        request.motionScale = options_.debugMotionScale;
        request.jitterX = temporal_.jitter[0] * (options_.jitterVariant & 1u ? -1.0f : 1.0f);
        request.jitterY = temporal_.jitter[1] * (options_.jitterVariant & 2u ? -1.0f : 1.0f);
        if (options_.debugWorkerCrash && frame_ + 1 == options_.debugWorkerCrash)
            request.command = temporal_worker::Command::CrashForTest;
        const u64 ticket = worker->enqueue(request);
        ctx.externalDependency(
            {worker->inputReady(), worker->outputReady(), ticket,
             [](rg::PassContext &c, void *p) { static_cast<PostProcessor *>(p)->finishWorkerCopy(c); }, this,
             [](void *p, u64 value) { static_cast<TemporalWorker *>(p)->submitted(value); }, worker.get()});
        ++temporalFrames_;
    } else if (temporalReady()) {
        auto *s = v.scaler.get();
        s->setColorTexture(input);
        s->setDepthTexture(static_cast<MTL::Texture *>(ctx.texture(depth_)));
        s->setMotionTexture(static_cast<MTL::Texture *>(ctx.texture(motion_)));
        s->setReactiveMaskTexture(static_cast<MTL::Texture *>(ctx.texture(reactive_)));
        s->setOutputTexture(slots_[slot_].output);
        s->setExposureTexture(v.exposure);
        s->setPreExposure(1.0f);
        s->setInputContentWidth(params_.inputWidth);
        s->setInputContentHeight(params_.inputHeight);
        s->setMotionVectorScaleX(options_.debugMotionScale);
        s->setMotionVectorScaleY(options_.debugMotionScale);
        s->setJitterOffsetX(temporal_.jitter[0] * (options_.jitterVariant & 1u ? -1.0f : 1.0f));
        s->setJitterOffsetY(temporal_.jitter[1] * (options_.jitterVariant & 2u ? -1.0f : 1.0f));
        s->setDepthReversed(true);
        s->setReset(reset_ || options_.forceReset);
        s->setFence(fence);
        s->encodeToCommandBuffer(cmd);
        ++temporalFrames_;
    } else {
        auto *enc = cmd->computeCommandEncoder();
        enc->waitForFence(fence, MTL::StageDispatch);
        bind(ctx, tables_[3]);
        tables_[3]->setTexture(slots_[slot_].output->gpuResourceID(), 1);
        enc->setComputePipelineState(pipelines_.compute(native_));
        enc->setArgumentTable(tables_[3]);
        enc->dispatchThreadgroups(MTL::Size::Make((width_ + 15) / 16, (height_ + 15) / 16, 1),
                                  MTL::Size::Make(16, 16, 1));
        enc->updateFence(fence, MTL::StageDispatch);
        enc->endEncoding();
        ++fallbackFrames_;
    }
}
void PostProcessor::finishWorkerCopy(rg::PassContext &ctx) {
    const auto &layout = views_[view_].worker->layout();
    auto *command = static_cast<MTL4::CommandBuffer *>(ctx.commandBuffer());
    auto *copy = command->computeCommandEncoder();
    copy->copyFromBuffer(static_cast<MTL::Buffer *>(ctx.buffer(workerBridge_)),
                         slot_ * layout.slotBytes + layout.offsets[temporal_worker::Output],
                         layout.rows[temporal_worker::Output], layout.rows[temporal_worker::Output] * height_,
                         MTL::Size::Make(width_, height_, 1), slots_[slot_].output, 0, 0, MTL::Origin::Make(0, 0, 0));
    copy->updateFence(static_cast<MTL::Fence *>(ctx.externalFence()), MTL::StageBlit);
    copy->endEncoding();
}
u64 PostProcessor::workerDeviceBytes() const {
    u64 n = 0;
    for (const auto &v : views_)
        if (v.worker)
            n += v.worker->deviceBytes();
    for (const auto &w : retiredWorkers_)
        n += w->deviceBytes();
    return n;
}
u64 PostProcessor::workerPhysicalFootprint() const {
    u64 n = 0;
    for (const auto &v : views_)
        if (v.worker)
            n += v.worker->physicalFootprint();
    for (const auto &w : retiredWorkers_)
        n += w->physicalFootprint();
    return n;
}
u64 PostProcessor::workerBridgeBytes() const {
    return TemporalWorker::mappedBytes();
}
rg::TextureRef PostProcessor::addToGraph(rg::RenderGraph &g, VisibilityRenderer &scene, rg::TextureRef drawable,
                                         rg::Format format) {
    using namespace rg;
    input_ = scene.color();
    depth_ = scene.depth();
    motion_ = scene.motion();
    reactive_ = scene.reactiveMask();
    if (usesWorker())
        workerBridge_ =
            g.importBuffer("MetalFX worker transfer", {temporal_worker::makeLayout(width_, height_).mappedBytes},
                           ImportPerFrame | ImportContentsDefined);
    histogramRef_ = g.importBuffer("Luminance histogram", {256 * sizeof(u32)}, ImportPerFrame);
    stateRef_ = g.importBuffer("Exposure and temporal state per view", {16}, ImportContentsDefined | ImportOutput);
    exposure_ = g.importTexture("Exposure", {Format::R16Float, 1, 1}, ImportOutput);
    output_ =
        g.importTexture("Reconstructed HDR", {Format::RGBA16Float, width_, height_}, ImportOutput | ImportPerFrame);
    g.addPass(
        "Histogram clear", PassType::Compute,
        [&](PassBuilder &b) { histogramRef_ = b.write(histogramRef_, Usage::ShaderWrite, StageDispatch); },
        [this](PassContext &ctx) {
            auto *enc = static_cast<MTL4::ComputeCommandEncoder *>(ctx.encoder());
            bind(ctx, tables_[0]);
            enc->setComputePipelineState(pipelines_.compute(clear_));
            enc->setArgumentTable(tables_[0]);
            enc->dispatchThreadgroups(MTL::Size::Make(1, 1, 1), MTL::Size::Make(256, 1, 1));
        });
    g.addPass(
        "Luminance histogram", PassType::Compute,
        [&](PassBuilder &b) {
            b.read(input_, Usage::ShaderRead, StageDispatch);
            b.read(depth_, Usage::ShaderRead, StageDispatch);
            b.read(histogramRef_, Usage::ShaderRead, StageDispatch);
            histogramRef_ = b.write(histogramRef_, Usage::ShaderWrite, StageDispatch);
        },
        [this](PassContext &ctx) {
            auto *enc = static_cast<MTL4::ComputeCommandEncoder *>(ctx.encoder());
            bind(ctx, tables_[1]);
            enc->setComputePipelineState(pipelines_.compute(histogram_));
            enc->setArgumentTable(tables_[1]);
            enc->dispatchThreadgroups(
                MTL::Size::Make((params_.inputWidth + 15) / 16, (params_.inputHeight + 15) / 16, 1),
                MTL::Size::Make(16, 16, 1));
        });
    g.addPass(
        "Exposure", PassType::Compute,
        [&](PassBuilder &b) {
            b.read(histogramRef_, Usage::ShaderRead, StageDispatch);
            b.read(stateRef_, Usage::ShaderRead, StageDispatch);
            stateRef_ = b.write(stateRef_, Usage::ShaderWrite, StageDispatch);
            exposure_ = b.write(exposure_, Usage::ShaderWrite, StageDispatch);
        },
        [this](PassContext &ctx) {
            auto *enc = static_cast<MTL4::ComputeCommandEncoder *>(ctx.encoder());
            bind(ctx, tables_[2]);
            enc->setComputePipelineState(pipelines_.compute(reduce_));
            enc->setArgumentTable(tables_[2]);
            enc->dispatchThreadgroups(MTL::Size::Make(1, 1, 1), MTL::Size::Make(256, 1, 1));
        });
    g.addPass(
        "Temporal reconstruction", PassType::External,
        [&](PassBuilder &b) {
            if (usesWorker()) {
                b.read(workerBridge_, Usage::CopySrc, StageExternal);
                workerBridge_ = b.write(workerBridge_, Usage::CopyDst, StageExternal);
            }
            for (auto t : {input_, depth_, motion_, reactive_, exposure_})
                b.read(t, Usage::ShaderRead, StageExternal);
            b.read(stateRef_, Usage::ShaderRead, StageExternal);
            stateRef_ = b.write(stateRef_, Usage::ShaderWrite, StageExternal);
            output_ = b.write(output_, Usage::ShaderWrite, StageExternal);
        },
        [this](PassContext &ctx) { encodeUpscale(ctx); });
    g.addPass(
        "Display tone map", PassType::Raster,
        [&](PassBuilder &b) {
            b.read(output_, Usage::ShaderRead, StageFragment);
            b.read(exposure_, Usage::ShaderRead, StageFragment);
            drawable = b.writeColor(drawable, 0, LoadIntent::Clear);
        },
        [this, format](PassContext &ctx) {
            auto *enc = static_cast<MTL4::RenderCommandEncoder *>(ctx.encoder());
            auto *t = tables_[4];
            t->setAddress(paramsAddress_, 1);
            t->setTexture(slots_[slot_].output->gpuResourceID(), 0);
            t->setTexture(views_[view_].exposure->gpuResourceID(), 1);
            enc->setRenderPipelineState(pipelines_.render(format == Format::RGBA16Float ? presentEDR_ : presentSDR_));
            enc->setArgumentTable(t, MTL::RenderStageFragment);
            enc->drawPrimitives(MTL::PrimitiveTypeTriangle, 0, 3);
        });
    if (options_.checkCurves) {
        auto check = g.importBuffer("Tone curve chart", {576 * 16}, ImportOutput);
        g.addPass(
            "Tone curve chart probe", PassType::Compute,
            [&](PassBuilder &b) {
                check = b.write(check, Usage::ShaderWrite, StageDispatch);
                b.setSideEffect();
            },
            [this](PassContext &ctx) {
                auto *enc = static_cast<MTL4::ComputeCommandEncoder *>(ctx.encoder());
                tables_[6]->setAddress(curveReadback_->gpuAddress(), 0);
                tables_[6]->setAddress(paramsAddress_, 1);
                enc->setArgumentTable(tables_[6]);
                enc->setComputePipelineState(pipelines_.compute(curveProbe_));
                enc->dispatchThreadgroups(MTL::Size::Make(3, 1, 1), MTL::Size::Make(256, 1, 1));
            });
    }
    return drawable;
}
bool PostProcessor::checkCurves() const {
    const auto *actual = static_cast<const float *>(curveReadback_->contents());
    constexpr double headrooms[3] = {1, 2, 8};
    u32 errors = 0;
    double worst = 0;
    for (u32 i = 0; i < 576; ++i) {
        const double headroom = headrooms[(i / 64) % 3];
        const auto expected = curveReference(i % 64, i / 192, headroom);
        for (u32 c = 0; c < 3; ++c) {
            const double value = actual[i * 4 + c], error = std::abs(value - expected[c]);
            worst = std::max(worst, error);
            if (!std::isfinite(value) || value < 0 || value > headroom || error > 5e-5 * headroom)
                ++errors;
        }
    }
    std::printf("POST-CURVES 576 HDR/color samples, headrooms 1/2/8 | errors %u max %.8f | %s\n", errors, worst,
                errors ? "FAIL" : "PASS");
    return errors == 0;
}
rg::TextureRef PostProcessor::addSDRCapture(rg::RenderGraph &g, rg::TextureRef display) {
    using namespace rg;
    TextureRef result;
    g.addPass(
        "EDR capture to SDR", PassType::Raster,
        [&](PassBuilder &b) {
            b.read(display, Usage::ShaderRead, StageFragment);
            result = b.createTexture("SDR screenshot", {Format::BGRA8Srgb, width_, height_});
            result = b.writeColor(result, 0, LoadIntent::Clear);
        },
        [this, display](PassContext &ctx) {
            auto *enc = static_cast<MTL4::RenderCommandEncoder *>(ctx.encoder());
            tables_[5]->setTexture(static_cast<MTL::Texture *>(ctx.texture(display))->gpuResourceID(), 0);
            enc->setRenderPipelineState(pipelines_.render(capture_));
            enc->setArgumentTable(tables_[5], MTL::RenderStageFragment);
            enc->drawPrimitives(MTL::PrimitiveTypeTriangle, 0, 3);
        });
    return result;
}
void PostProcessor::bindFrame(MetalGraphExecutor &e) {
    if (usesWorker())
        e.bindBuffer(workerBridge_, views_[view_].workerBuffer);
    e.bindTexture(output_, slots_[slot_].output);
    e.bindTexture(exposure_, views_[view_].exposure);
}
} // namespace phosphor
