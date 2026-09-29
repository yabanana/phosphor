#include "platform/metal/async_compute_probe.h"
#include "platform/metal/gpu_memory.h"
#include "platform/metal/metal_graph_executor.h"
#include "rendergraph/pass_context.h"
#include "core/log.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

namespace phosphor {

namespace {

NS::String* str(const char* s) { return NS::String::string(s, NS::UTF8StringEncoding); }

// Argument table slots; must match shaders/async_compute.metal.
constexpr NS::UInteger kBindFrame    = 0;
constexpr NS::UInteger kBindSeed     = 1;
constexpr NS::UInteger kBindResult   = 2;
constexpr NS::UInteger kBindReadback = 3;

MTL4::ArgumentTable* makeTable(MetalContext& context, const char* label) {
    MTL4::ArgumentTableDescriptor* d = MTL4::ArgumentTableDescriptor::alloc()->init();
    d->setMaxBufferBindCount(4);
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

MTL::GPUAddress bufferAddress(rg::PassContext& ctx, rg::BufferRef ref) {
    return static_cast<MTL::Buffer*>(ctx.buffer(ref))->gpuAddress();
}

} // namespace

AsyncComputeProbe::AsyncComputeProbe(MetalContext& context) : context_(context) {
    seed_        = computePipeline("async_seed");
    reduce_      = computePipeline("async_reduce");
    consume_     = computePipeline("async_consume");
    seedArgs_    = makeTable(context_, "Async probe seed arguments");
    reduceArgs_  = makeTable(context_, "Async probe reduce arguments");
    consumeArgs_ = makeTable(context_, "Async probe consume arguments");
    for (u32 i = 0; i < METAL_FRAMES_IN_FLIGHT; ++i) {
        const std::string label = "Async probe readback " + std::to_string(i);
        readback_[i] = context_.memory().newBuffer(rg::kAsyncReadbackSize, MTL::ResourceStorageModeShared,
                                                   MemoryCategory::Other, label.c_str());
        if (!readback_[i]) throw std::runtime_error("Failed to allocate the async probe readback buffer");
        std::memset(readback_[i]->contents(), 0, rg::kAsyncReadbackSize);
    }
}

AsyncComputeProbe::~AsyncComputeProbe() {
    context_.waitIdle();
    for (MTL::Buffer* b : readback_) context_.memory().release(b, MemoryCategory::Other);
    consumeArgs_->release();
    reduceArgs_->release();
    seedArgs_->release();
    for (auto* p : {seed_, reduce_, consume_}) p->release();
}

MTL::ComputePipelineState* AsyncComputeProbe::computePipeline(const char* function) {
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

void AsyncComputeProbe::addProducers(rg::RenderGraph& graph) {
    rg::AsyncProbeExec& exec = exec_;
    exec.seed = [this](rg::PassContext& ctx) {
        MTL4::ComputeCommandEncoder* enc = computeEncoder(ctx);
        // The frame index goes through the per-frame upload ring (no allocation).
        const UploadRing::Slice constants = context_.frameUploads().allocate(sizeof(u32));
        *reinterpret_cast<u32*>(constants.cpu) = static_cast<u32>(ctx.frameIndex());
        seedArgs_->setAddress(constants.gpu, kBindFrame);
        seedArgs_->setAddress(bufferAddress(ctx, refs_.s), kBindSeed);
        enc->setComputePipelineState(seed_);
        enc->setArgumentTable(seedArgs_);
        enc->dispatchThreads(MTL::Size::Make(rg::kAsyncSeedCount, 1, 1), MTL::Size::Make(256, 1, 1));
    };
    exec.reduce = [this](rg::PassContext& ctx) {
        MTL4::ComputeCommandEncoder* enc = computeEncoder(ctx);
        reduceArgs_->setAddress(bufferAddress(ctx, refs_.s), kBindSeed);
        reduceArgs_->setAddress(bufferAddress(ctx, refs_.r), kBindResult);
        enc->setComputePipelineState(reduce_);
        enc->setArgumentTable(reduceArgs_);
        enc->dispatchThreads(MTL::Size::Make(rg::kAsyncResultCount, 1, 1), MTL::Size::Make(64, 1, 1));
    };
    exec.consume = [this](rg::PassContext& ctx) {
        MTL4::ComputeCommandEncoder* enc = computeEncoder(ctx);
        const UploadRing::Slice constants = context_.frameUploads().allocate(sizeof(u32));
        *reinterpret_cast<u32*>(constants.cpu) = static_cast<u32>(ctx.frameIndex());
        consumeArgs_->setAddress(constants.gpu, kBindFrame);
        consumeArgs_->setAddress(bufferAddress(ctx, refs_.r), kBindResult);
        consumeArgs_->setAddress(bufferAddress(ctx, refs_.readback), kBindReadback);
        enc->setComputePipelineState(consume_);
        enc->setArgumentTable(consumeArgs_);
        enc->dispatchThreads(MTL::Size::Make(rg::kAsyncResultCount, 1, 1), MTL::Size::Make(64, 1, 1));
    };
    rg::addAsyncProbeProducers(graph, refs_, exec);
}

void AsyncComputeProbe::addConsumer(rg::RenderGraph& graph) {
    rg::addAsyncProbeConsumer(graph, refs_, exec_);
}

void AsyncComputeProbe::onCompiled(const rg::RenderGraph& graph, const rg::CompiledGraph& compiled) {
    queueSyncs_ = static_cast<u32>(compiled.queueSyncs.size());
    asyncPass_  = false;
    for (u32 pass = 0; pass < graph.passes().size(); ++pass) {
        const rg::PassNode& node = graph.passes()[pass];
        if (node.name == "Async reduce") {
            asyncPass_ = node.queue == rg::Queue::AsyncCompute && !compiled.culled[pass];
        }
    }
    compiled_ = true;
    LOG_INFO("Async probe: %u queue sync(s), reduce pass %s the async queue", queueSyncs_,
             asyncPass_ ? "on" : "NOT on");
    for (const rg::QueueSync& s : compiled.queueSyncs) {
        LOG_INFO("Async probe:   signal %u after position %u, wait before position %u", s.value,
                 s.signalAfterPosition, s.waitBeforePosition);
    }
}

void AsyncComputeProbe::verifySlot(u32 slot) {
    Pending& p = pending_[slot];
    if (!p.valid) return;
    p.valid = false;
    const u64 bad = rg::countAsyncMismatches(static_cast<u32>(p.frame),
                                             static_cast<const u32*>(readback_[slot]->contents()));
    ++checked_;
    if (bad != 0) {
        ++badFrames_;
        mismatches_ += bad;
        if (badFrames_ <= 5) {
            LOG_ERROR("Async probe: frame %llu has %llu wrong value(s) out of %u",
                      static_cast<unsigned long long>(p.frame), static_cast<unsigned long long>(bad),
                      rg::kAsyncResultCount);
        }
    }
}

void AsyncComputeProbe::beginFrame(u32 slot) { verifySlot(slot); }

void AsyncComputeProbe::bind(MetalGraphExecutor& executor, u32 slot) const {
    executor.bindBuffer(refs_.readback, readback_[slot]);
}

void AsyncComputeProbe::frameEncoded(u32 slot, u64 frame) {
    pending_[slot] = {true, frame};
    ++frames_;
}

bool AsyncComputeProbe::finish() {
    for (u32 s = 0; s < METAL_FRAMES_IN_FLIGHT; ++s) verifySlot(s);
    const bool pass = compiled_ && asyncPass_ && queueSyncs_ == 2 && frames_ > 0 && checked_ == frames_ &&
                      mismatches_ == 0;
    std::printf("ASYNC-COMPUTE frames=%llu checked=%llu mismatches=%llu %s\n",
                static_cast<unsigned long long>(frames_), static_cast<unsigned long long>(checked_),
                static_cast<unsigned long long>(mismatches_), pass ? "PASS" : "FAIL");
    std::fflush(stdout);
    return pass;
}

} // namespace phosphor
