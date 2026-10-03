#include "platform/metal/post_processor.h"
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
} // namespace
PostProcessor::PostProcessor(MetalContext &c, PipelineCache &p, const Options &o)
    : context_(c), pipelines_(p), options_(o) {
    if (!o.views || o.views > HistoryRegistry::MaxViews)
        throw std::invalid_argument("Invalid temporal view count");
    clear_ = p.request(kernel("exposure_clear"));
    histogram_ = p.request(kernel("exposure_histogram"));
    reduce_ = p.request(kernel("exposure_reduce"));
    native_ = p.request(kernel("post_native"));
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
        v.exposure = texture(c, 1, 1, MTL::PixelFormatR32Float, "Exposure texture per view");
    }
    for (auto &s : slots_)
        s.histogram = c.memory().newBuffer(256 * sizeof(u32), MTL::ResourceStorageModeShared, MemoryCategory::Other,
                                           "Luminance histogram");
}
PostProcessor::~PostProcessor() {
    context_.waitIdle();
    releaseTargets();
    for (auto &v : views_) {
        v.scaler.reset();
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
    if (width_ == w && height_ == h)
        return;
    width_ = w;
    height_ = h;
    releaseTargets();
    for (auto &s : slots_)
        s.output = texture(context_, w, h, MTL::PixelFormatRGBA16Float, "Reconstructed HDR");
    for (u32 i = 0; i < options_.views; ++i) {
        auto &v = views_[i];
        if (v.scaler) {
            v.scaler->retain();
            context_.deferRelease(v.scaler.get());
            v.scaler.reset();
        }
        v.failed = false;
        histories_.invalidate(i, "output resize");
        if (!v.pending.valid())
            requestScaler(i);
    }
}
void PostProcessor::requestScaler(u32 index) {
    if (!options_.temporal || !supported_)
        return;
    auto &v = views_[index];
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
    d->setRequiresSynchronousInitialization(true); // runs on PipelineCache's utility workers
    v.requestedWidth = width_;
    v.requestedHeight = height_;
    v.pending = pipelines_.requestTemporalScaler(d);
    d->release();
}
void PostProcessor::collectScalers(bool wait) {
    for (u32 i = 0; i < options_.views; ++i) {
        auto &v = views_[i];
        if (!v.pending.valid())
            continue;
        if (!wait && v.pending.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
            continue;
        auto scaler = v.pending.get();
        if (v.requestedWidth != width_ || v.requestedHeight != height_) {
            requestScaler(i);
            continue;
        }
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
    return options_.temporal && bool(views_[view_].scaler);
}
const char *PostProcessor::effectiveUpscaler() const {
    return temporalReady() ? "metalfx-temporal" : "native-spatial";
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
    temporal_.mipBias = std::min(0.0f, std::log2(float(iw) / float(w)));
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
               (options_.autoExposure ? 1u : 0u) | (options_.corruptExposure ? 2u : 0u),
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
bool PostProcessor::checkExposure() const {
    std::array<u32, ExposureBins> histogram{};
    std::memcpy(histogram.data(), slots_[slot_].histogram->contents(), sizeof(histogram));
    const float expected = exposureTarget(histogram);
    const auto *state = static_cast<const float *>(views_[view_].exposureState->contents());
    const bool pass = std::isfinite(state[0]) && state[0] > 0 && std::isfinite(state[1]) &&
                      std::abs(state[1] - expected) <= std::max(0.00001f, std::abs(expected) * 0.0001f);
    std::printf("EXPOSURE view %u | target %.6f reference %.6f adapted %.6f | %s\n", view_, double(state[1]),
                double(expected), double(state[0]), pass ? "PASS" : "FAIL");
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
    if (temporalReady()) {
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
rg::TextureRef PostProcessor::addToGraph(rg::RenderGraph &g, VisibilityRenderer &scene, rg::TextureRef drawable,
                                         rg::Format format) {
    using namespace rg;
    input_ = scene.color();
    depth_ = scene.depth();
    motion_ = scene.motion();
    reactive_ = scene.reactiveMask();
    histogramRef_ = g.importBuffer("Luminance histogram", {256 * sizeof(u32)}, ImportPerFrame);
    stateRef_ = g.importBuffer("Exposure and temporal state per view", {16}, ImportContentsDefined | ImportOutput);
    exposure_ = g.importTexture("Exposure", {Format::R32Float, 1, 1}, ImportOutput);
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
            enc->setDepthStencilState(nullptr);
            enc->setViewport(MTL::Viewport{0, 0, double(width_), double(height_), 0, 1});
            enc->setArgumentTable(t, MTL::RenderStageFragment);
            enc->drawPrimitives(MTL::PrimitiveTypeTriangle, 0, 3);
        });
    return drawable;
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
            enc->setDepthStencilState(nullptr);
            enc->setViewport(MTL::Viewport{0, 0, double(width_), double(height_), 0, 1});
            enc->setArgumentTable(tables_[5], MTL::RenderStageFragment);
            enc->drawPrimitives(MTL::PrimitiveTypeTriangle, 0, 3);
        });
    return result;
}
void PostProcessor::bindFrame(MetalGraphExecutor &e) {
    e.bindTexture(output_, slots_[slot_].output);
    e.bindTexture(exposure_, views_[view_].exposure);
}
} // namespace phosphor
