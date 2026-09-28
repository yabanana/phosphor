#include "platform/metal/graph_debug_passes.h"
#include "platform/metal/gpu_memory.h"
#include "platform/metal/metal_graph_executor.h"
#include "rendergraph/pass_context.h"
#include "core/log.h"

#include <cstdio>
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

GraphDebugPasses::GraphDebugPasses(MetalContext& context) : context_(context) {
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
    raster_->release();
    for (auto* p : {fill_, reduce_, expand_, checksum_}) p->release();
}

MTL::ComputePipelineState* GraphDebugPasses::computePipeline(const char* function) {
    MTL4::LibraryFunctionDescriptor* fn = MTL4::LibraryFunctionDescriptor::alloc()->init();
    fn->setLibrary(context_.library());
    fn->setName(str(function));
    MTL4::ComputePipelineDescriptor* desc = MTL4::ComputePipelineDescriptor::alloc()->init();
    desc->setLabel(str(function));
    desc->setComputeFunctionDescriptor(fn);
    NS::Error* error = nullptr;
    MTL::ComputePipelineState* pso = context_.compiler()->newComputePipelineState(desc, nullptr, &error);
    desc->release();
    fn->release();
    if (!pso) {
        const char* reason = error ? error->localizedDescription()->utf8String() : "unknown error";
        throw std::runtime_error(std::string("Failed to build ") + function + ": " + reason);
    }
    return pso;
}

void GraphDebugPasses::buildPipelines() {
    fill_     = computePipeline("debug_fill");
    reduce_   = computePipeline("debug_reduce");
    expand_   = computePipeline("debug_expand");
    checksum_ = computePipeline("debug_checksum");

    MTL4::LibraryFunctionDescriptor* vs = MTL4::LibraryFunctionDescriptor::alloc()->init();
    vs->setLibrary(context_.library());
    vs->setName(str("debug_vs"));
    MTL4::LibraryFunctionDescriptor* fs = MTL4::LibraryFunctionDescriptor::alloc()->init();
    fs->setLibrary(context_.library());
    fs->setName(str("debug_fs"));
    MTL4::RenderPipelineDescriptor* desc = MTL4::RenderPipelineDescriptor::alloc()->init();
    desc->setLabel(str("Graph debug raster"));
    desc->setVertexFunctionDescriptor(vs);
    desc->setFragmentFunctionDescriptor(fs);
    desc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatR32Uint);
    NS::Error* error = nullptr;
    raster_ = context_.compiler()->newRenderPipelineState(desc, nullptr, &error);
    desc->release();
    fs->release();
    vs->release();
    if (!raster_) {
        const char* reason = error ? error->localizedDescription()->utf8String() : "unknown error";
        throw std::runtime_error(std::string("Failed to build the graph debug raster pipeline: ") + reason);
    }
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
        enc->setComputePipelineState(fill_);
        enc->setArgumentTable(computeArgs_);
        enc->dispatchThreads(MTL::Size::Make(rg::kDebugSize, rg::kDebugSize, 1), MTL::Size::Make(16, 16, 1));
    };
    exec.reduce = [this](rg::PassContext& ctx) {
        MTL4::ComputeCommandEncoder* enc = computeEncoder(ctx);
        computeArgs_->setTexture(textureId(ctx, refs_.a), kBindTex);
        computeArgs_->setAddress(bufferAddress(ctx, refs_.b), kBindBuffer);
        enc->setComputePipelineState(reduce_);
        enc->setArgumentTable(computeArgs_);
        enc->dispatchThreads(MTL::Size::Make(rg::kDebugSize, 1, 1), MTL::Size::Make(64, 1, 1));
    };
    exec.expand = [this](rg::PassContext& ctx) {
        MTL4::ComputeCommandEncoder* enc = computeEncoder(ctx);
        computeArgs_->setTexture(textureId(ctx, refs_.c), kBindTex);
        computeArgs_->setAddress(bufferAddress(ctx, refs_.b), kBindBuffer);
        enc->setComputePipelineState(expand_);
        enc->setArgumentTable(computeArgs_);
        enc->dispatchThreads(MTL::Size::Make(rg::kDebugSize, rg::kDebugSize, 1), MTL::Size::Make(16, 16, 1));
    };
    exec.raster = [this](rg::PassContext& ctx) {
        auto* enc = static_cast<MTL4::RenderCommandEncoder*>(ctx.encoder());
        rasterArgs_->setTexture(textureId(ctx, refs_.c), kBindTex);
        enc->setRenderPipelineState(raster_);
        enc->setArgumentTable(rasterArgs_, MTL::RenderStageVertex | MTL::RenderStageFragment);
        enc->drawPrimitives(MTL::PrimitiveTypeTriangle, 0, 3);
    };
    exec.checksum = [this](rg::PassContext& ctx) {
        MTL4::ComputeCommandEncoder* enc = computeEncoder(ctx);
        computeArgs_->setTexture(textureId(ctx, refs_.d), kBindTex);
        computeArgs_->setAddress(bufferAddress(ctx, refs_.readback), kBindBuffer);
        enc->setComputePipelineState(checksum_);
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
