#include "platform/metal/hiz_builder.h"

#include "core/log.h"
#include "platform/metal/gpu_memory.h"
#include "platform/metal/pipeline_cache.h"
#include "renderer/gpu_types.h"
#include "renderer/meshlet_cull_math.h"
#include "renderer/meshlet_layout.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace phosphor {

namespace {

constexpr u64 kParamStride = 256;

NS::String* str(const char* s) { return NS::String::string(s, NS::UTF8StringEncoding); }

pipe::PipelineDesc kernelDesc(const char* function) {
    pipe::PipelineDesc d;
    d.kind      = pipe::PipelineKind::Compute;
    d.label     = function;
    d.functions = {function, "", ""};
    return d;
}

MTL::Size groups(u32 w, u32 h) { return MTL::Size::Make((w + HIZ_GROUP - 1) / HIZ_GROUP, (h + HIZ_GROUP - 1) / HIZ_GROUP, 1); }

} // namespace

HiZBuilder::HiZBuilder(MetalContext& context, PipelineCache& pipelines, Backend backend)
    : context_(context), pipelines_(pipelines), backend_(backend) {
    if (backend_ == Backend::Sampler && !context_.effectiveApple10()) {
        throw std::runtime_error("Hi-Z sampler backend needs Apple10 sampler min reduction (effective family is " +
                                 std::string(context_.effectiveFamilyName()) + ")");
    }
    kLevel0_ = pipelines_.request(kernelDesc(KERNEL_HIZ_LEVEL0));
    kReduce_ = pipelines_.request(kernelDesc(backend_ == Backend::Sampler ? KERNEL_HIZ_REDUCE_SAMPLER : "hiz_reduce_simd"));
    for (u32 i = 0; i < 2; ++i) {
        NS::Error* error = nullptr;
        MTL4::ArgumentTableDescriptor* d = MTL4::ArgumentTableDescriptor::alloc()->init();
        d->setMaxBufferBindCount(1);
        d->setMaxTextureBindCount(2);
        d->setLabel(str(i == 0 ? "Hi-Z A arguments" : "Hi-Z final arguments"));
        tables_[i] = context_.device()->newArgumentTable(d, &error);
        d->release();
        if (!tables_[i]) throw std::runtime_error("Failed to create the Hi-Z argument table");
    }
    LOG_INFO("Hi-Z backend: %s (effective family %s)", backendName(), context_.effectiveFamilyName());
}

HiZBuilder::~HiZBuilder() {
    context_.waitIdle();
    release();
    for (MTL4::ArgumentTable* t : tables_) t->release();
}

void HiZBuilder::release() {
    GpuMemory& m = context_.memory();
    for (MTL::Texture** t : {&current_, &history_}) {
        if (*t) m.release(*t, MemoryCategory::RenderTargets);
        *t = nullptr;
    }
    if (params_) m.release(params_, MemoryCategory::RenderTargets);
    params_ = nullptr;
    bytes_  = 0;
}

bool HiZBuilder::ready() const { return pipelines_.compute(kLevel0_) && pipelines_.compute(kReduce_); }

bool HiZBuilder::resize(u32 width, u32 height) {
    if (width == width_ && height == height_ && current_) return false;
    release();
    width_   = width;
    height_  = height;
    width0_  = hizLevel0Size(width);
    height0_ = hizLevel0Size(height);
    levels_  = hizLevelCount(width0_, height0_);
    MTL::TextureDescriptor* d = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatR32Float, width0_, height0_, true);
    d->setMipmapLevelCount(levels_);
    d->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
    d->setStorageMode(MTL::StorageModePrivate);
    GpuMemory& m = context_.memory();
    current_ = m.newTexture(d, MemoryCategory::RenderTargets, "Hi-Z current");
    history_ = m.newTexture(d, MemoryCategory::RenderTargets, "Hi-Z history");
    bytes_   = 2 * current_->allocatedSize();

    // Parameters: level 0, then one entry per reduce dispatch; the last entry
    // is level 0 with the corruption flag (self-check negative control).
    const u32 step = backend_ == Backend::Sampler ? 1u : 5u;
    dispatches_ = levels_ > 1 ? (levels_ - 1 + step - 1) / step : 0;
    const u32 entries = 2 + dispatches_;
    params_ = m.newBuffer(entries * kParamStride, MTL::ResourceStorageModeShared, MemoryCategory::RenderTargets,
                          "Hi-Z parameters");
    auto* base = static_cast<u8*>(params_->contents());
    auto put = [&](u32 index, const GPUHiZParams& p) { std::memcpy(base + index * kParamStride, &p, sizeof(p)); };
    GPUHiZParams l0{{width, height}, {width0_, height0_}, 0, levels_, {0, 0}};
    put(0, l0);
    u32 e = 1;
    for (u32 l = 1; l < levels_; l += step) {
        GPUHiZParams p{{hizLevelSize(width0_, l - 1), hizLevelSize(height0_, l - 1)},
                       {hizLevelSize(width0_, l), hizLevelSize(height0_, l)}, l, levels_, {0, 0}};
        put(e++, p);
    }
    l0.pad[0] = 1; // corrupt
    put(e, l0);
    LOG_INFO("Hi-Z pyramids %ux%u (%u levels, %s) for %ux%u: %.2f MiB", width0_, height0_, levels_, backendName(),
             width, height, static_cast<double>(bytes_) / (1 << 20));
    return true;
}

u64 HiZBuilder::readbackBytes() const {
    u64 total = 0;
    for (u32 l = 0; l < levels_; ++l) total += u64(hizLevelSize(width0_, l)) * hizLevelSize(height0_, l) * 4;
    return total;
}

void HiZBuilder::encodeReadback(MTL4::ComputeCommandEncoder* enc, MTL::Texture* pyramid, MTL::Buffer* dst) const {
    if (!pyramid || !dst || dst->length() < readbackBytes()) return;
    u64 offset = 0;
    for (u32 l = 0; l < levels_; ++l) {
        const u32 w = hizLevelSize(width0_, l), h = hizLevelSize(height0_, l);
        enc->copyFromTexture(pyramid, 0, l, MTL::Origin::Make(0, 0, 0), MTL::Size::Make(w, h, 1), dst, offset, u64(w) * 4,
                             u64(w) * h * 4);
        offset += u64(w) * h * 4;
    }
}

u32 HiZBuilder::commandCount() const {
    // level 0: pso + table + dispatch; reduce: pso once, then barrier + table + dispatch each
    return 3 + (dispatches_ ? 1 + 3 * dispatches_ : 0) + 1;
}

void HiZBuilder::encode(MTL4::ComputeCommandEncoder* enc, MTL::Texture* depth, MTL::Texture* dst, u32 table,
                        bool corrupt) const {
    if (!ready() || !depth || !dst || !params_) return;
    MTL4::ArgumentTable* t = tables_[table & 1u];
    const MTL::GPUAddress params = params_->gpuAddress();
    const MTL::Size tg = MTL::Size::Make(HIZ_GROUP, HIZ_GROUP, 1);
    t->setAddress(params + (corrupt ? (1 + dispatches_) * kParamStride : 0), HZ_PARAMS);
    t->setTexture(depth->gpuResourceID(), HZ_TEX_SRC);
    t->setTexture(dst->gpuResourceID(), HZ_TEX_DST);
    enc->setComputePipelineState(pipelines_.compute(kLevel0_));
    enc->setArgumentTable(t);
    enc->dispatchThreadgroups(groups(width0_, height0_), tg);
    if (dispatches_ == 0) return;
    // The pyramid is its own source from here on (other mip levels).
    t->setTexture(dst->gpuResourceID(), HZ_TEX_SRC);
    enc->setComputePipelineState(pipelines_.compute(kReduce_));
    const u32 step = backend_ == Backend::Sampler ? 1u : 5u;
    u32 e = 1;
    for (u32 l = 1; l < levels_; l += step, ++e) {
        enc->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
        t->setAddress(params + e * kParamStride, HZ_PARAMS);
        enc->setArgumentTable(t);
        enc->dispatchThreadgroups(groups(hizLevelSize(width0_, l), hizLevelSize(height0_, l)), tg);
    }
}

} // namespace phosphor
