#include "platform/metal/graph_debug_passes.h"
#include "platform/metal/gpu_memory.h"
#include "platform/metal/metal_graph_executor.h"
#include "platform/metal/pipeline_cache.h"
#include "rendergraph/pass_context.h"
#include "core/log.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

namespace phosphor {

namespace {

NS::String* str(const char* s) { return NS::String::string(s, NS::UTF8StringEncoding); }

// Argument table slots; must match shaders/graph_debug.metal.
constexpr NS::UInteger kBindFrame  = 0;
constexpr NS::UInteger kBindBuffer = 1;
constexpr NS::UInteger kBindTex    = 0;

MTL4::ArgumentTable* makeTable(MetalContext& context, NS::UInteger buffers, NS::UInteger textures,
                               const char* label) {
    MTL4::ArgumentTableDescriptor* d = MTL4::ArgumentTableDescriptor::alloc()->init();
    d->setMaxBufferBindCount(buffers);
    d->setMaxTextureBindCount(textures);
    d->setLabel(str(label));
    NS::Error* error = nullptr;
    MTL4::ArgumentTable* table = context.device()->newArgumentTable(d, &error);
    d->release();
    if (!table) throw std::runtime_error(std::string("Failed to create argument table ") + label);
    return table;
}

MTL4::ComputeCommandEncoder* computeEncoder(rg::PassContext& ctx) {
    return static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder());
}

MTL::ResourceID textureId(rg::PassContext& ctx, rg::TextureRef ref) {
    return static_cast<MTL::Texture*>(ctx.texture(ref))->gpuResourceID();
}

MTL::GPUAddress bufferAddress(rg::PassContext& ctx, rg::BufferRef ref) {
    return static_cast<MTL::Buffer*>(ctx.buffer(ref))->gpuAddress();
}

} // namespace

GraphDebugPasses::GraphDebugPasses(MetalContext& context, PipelineCache& pipelines)
    : context_(context), pipelines_(pipelines) {
    buildPipelines();
    computeArgs_ = makeTable(context_, 2, 1, "Graph debug compute arguments");
    rasterArgs_  = makeTable(context_, 1, 1, "Graph debug raster arguments");
    for (u32 i = 0; i < METAL_FRAMES_IN_FLIGHT; ++i) {
        const std::string label = "Debug readback " + std::to_string(i);
        readback_[i] = context_.memory().newBuffer(rg::kDebugReadbackSize, MTL::ResourceStorageModeShared,
                                                   MemoryCategory::Other, label.c_str());
        if (!readback_[i]) throw std::runtime_error("Failed to allocate the graph debug readback buffer");
        std::memset(readback_[i]->contents(), 0, rg::kDebugReadbackSize);
    }
}

GraphDebugPasses::~GraphDebugPasses() {
    context_.waitIdle();
    for (MTL::Buffer* b : readback_) context_.memory().release(b, MemoryCategory::Other);
    rasterArgs_->release();
    computeArgs_->release();
}

pipe::PipelineHandle GraphDebugPasses::computePipeline(const char* function) {
    pipe::PipelineDesc desc;
    desc.kind         = pipe::PipelineKind::Compute;
    desc.label        = function;
    desc.functions[0] = function;
    return pipelines_.request(desc);
}

void GraphDebugPasses::buildPipelines() {
    // Created before the first frame: the engine waits for every pipeline.
    fill_     = computePipeline("debug_fill");
    reduce_   = computePipeline("debug_reduce");
    expand_   = computePipeline("debug_expand");
    checksum_ = computePipeline("debug_checksum");

    pipe::PipelineDesc desc;
    desc.label        = "Graph debug raster";
    desc.functions[0] = "debug_vs";
    desc.functions[1] = "debug_fs";
    desc.output(0, rg::Format::R32Uint);
    raster_ = pipelines_.request(desc);
}

void GraphDebugPasses::addToGraph(rg::RenderGraph& graph) {
    rg::DebugChainExec exec;
    exec.fill = [this](rg::PassContext& ctx) {
        MTL4::ComputeCommandEncoder* enc = computeEncoder(ctx);
        // The frame index goes through the per-frame upload ring (no allocation).
        const UploadRing::Slice constants = context_.frameUploads().allocate(sizeof(u32));
        *reinterpret_cast<u32*>(constants.cpu) = static_cast<u32>(ctx.frameIndex());
        computeArgs_->setAddress(constants.gpu, kBindFrame);
        computeArgs_->setTexture(textureId(ctx, refs_.a), kBindTex);
        enc->setComputePipelineState(pipelines_.compute(fill_));
        enc->setArgumentTable(computeArgs_);
        enc->dispatchThreads(MTL::Size::Make(rg::kDebugSize, rg::kDebugSize, 1), MTL::Size::Make(16, 16, 1));
    };
    exec.reduce = [this](rg::PassContext& ctx) {
        MTL4::ComputeCommandEncoder* enc = computeEncoder(ctx);
        computeArgs_->setTexture(textureId(ctx, refs_.a), kBindTex);
        computeArgs_->setAddress(bufferAddress(ctx, refs_.b), kBindBuffer);
        enc->setComputePipelineState(pipelines_.compute(reduce_));
        enc->setArgumentTable(computeArgs_);
        enc->dispatchThreads(MTL::Size::Make(rg::kDebugSize, 1, 1), MTL::Size::Make(64, 1, 1));
    };
    exec.expand = [this](rg::PassContext& ctx) {
        MTL4::ComputeCommandEncoder* enc = computeEncoder(ctx);
        computeArgs_->setTexture(textureId(ctx, refs_.c), kBindTex);
        computeArgs_->setAddress(bufferAddress(ctx, refs_.b), kBindBuffer);
        enc->setComputePipelineState(pipelines_.compute(expand_));
        enc->setArgumentTable(computeArgs_);
        enc->dispatchThreads(MTL::Size::Make(rg::kDebugSize, rg::kDebugSize, 1), MTL::Size::Make(16, 16, 1));
    };
    exec.raster = [this](rg::PassContext& ctx) {
        auto* enc = static_cast<MTL4::RenderCommandEncoder*>(ctx.encoder());
        rasterArgs_->setTexture(textureId(ctx, refs_.c), kBindTex);
        enc->setRenderPipelineState(pipelines_.render(raster_));
        enc->setArgumentTable(rasterArgs_, MTL::RenderStageVertex | MTL::RenderStageFragment);
        enc->drawPrimitives(MTL::PrimitiveTypeTriangle, 0, 3);
    };
    exec.checksum = [this](rg::PassContext& ctx) {
        MTL4::ComputeCommandEncoder* enc = computeEncoder(ctx);
        computeArgs_->setTexture(textureId(ctx, refs_.d), kBindTex);
        computeArgs_->setAddress(bufferAddress(ctx, refs_.readback), kBindBuffer);
        enc->setComputePipelineState(pipelines_.compute(checksum_));
        enc->setArgumentTable(computeArgs_);
        enc->dispatchThreads(MTL::Size::Make(rg::kDebugSize, rg::kDebugSize, 1), MTL::Size::Make(16, 16, 1));
    };
    rg::addDebugChain(graph, refs_, exec);
}

void GraphDebugPasses::onCompiled(const rg::RenderGraph& graph, const rg::CompiledGraph& compiled) {
    alias_    = rg::summarizeAliasing(compiled);
    compiled_ = true;
    const auto& resources = graph.resources();
    LOG_INFO("Graph transients: heap %llu bytes vs %llu unaliased, %u aliased placement(s)",
             static_cast<unsigned long long>(alias_.heapSize), static_cast<unsigned long long>(alias_.unaliasedSize),
             alias_.aliasedFlags);
    for (const rg::Placement& p : compiled.aliasing.placements) {
        LOG_INFO("Graph transients:   %-10s offset %8llu size %8llu%s", resources[p.resource].name.c_str(),
                 static_cast<unsigned long long>(p.offset), static_cast<unsigned long long>(p.size),
                 p.aliased ? " (aliased)" : "");
    }
    for (const auto& [a, b] : alias_.sharedPairs) {
        LOG_INFO("Graph transients: %s shares memory with %s", resources[a].name.c_str(), resources[b].name.c_str());
    }
}

void GraphDebugPasses::verifySlot(u32 slot) {
    Pending& p = pending_[slot];
    if (!p.valid) return;
    p.valid = false;
    const u64 bad = rg::countMismatches(static_cast<u32>(p.frame),
                                        static_cast<const u32*>(readback_[slot]->contents()));
    ++checked_;
    if (bad != 0) {
        ++badFrames_;
        mismatches_ += bad;
        if (badFrames_ <= 5) {
            LOG_ERROR("Graph transients: frame %llu has %llu wrong value(s) out of %u",
                      static_cast<unsigned long long>(p.frame), static_cast<unsigned long long>(bad),
                      rg::kDebugSize * rg::kDebugSize);
        }
    }
}

void GraphDebugPasses::beginFrame(u32 slot) { verifySlot(slot); }

void GraphDebugPasses::bind(MetalGraphExecutor& executor, u32 slot) const {
    executor.bindBuffer(refs_.readback, readback_[slot]);
}

void GraphDebugPasses::frameEncoded(u32 slot, u64 frame) {
    pending_[slot] = {true, frame};
    ++frames_;
}

bool GraphDebugPasses::finish() {
    for (u32 s = 0; s < METAL_FRAMES_IN_FLIGHT; ++s) verifySlot(s);
    const bool pass = compiled_ && frames_ > 0 && checked_ == frames_ && mismatches_ == 0 &&
                      alias_.aliasedFlags > 0 && !alias_.sharedPairs.empty();
    std::printf("GRAPH-TRANSIENTS frames=%llu checked=%llu mismatches=%llu heap=%llu unaliased=%llu aliased=%u %s\n",
                static_cast<unsigned long long>(frames_), static_cast<unsigned long long>(checked_),
                static_cast<unsigned long long>(mismatches_), static_cast<unsigned long long>(alias_.heapSize),
                static_cast<unsigned long long>(alias_.unaliasedSize), alias_.aliasedFlags, pass ? "PASS" : "FAIL");
    std::fflush(stdout);
    return pass;
}

} // namespace phosphor
