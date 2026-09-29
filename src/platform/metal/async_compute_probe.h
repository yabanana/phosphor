#pragma once

#include "core/types.h"
#include "pipeline/pipeline_registry.h"
#include "platform/metal/metal_context.h"
#include "rendergraph/async_probe_reference.h"

#include <array>

namespace phosphor {

class MetalGraphExecutor;
class PipelineCache;

// ---------------------------------------------------------------------------
// AsyncComputeProbe -- --debug-async-compute (F2.6).
//
// Adds the synthetic seed -> reduce -> consume chain of rg::addAsyncProbeChain
// to the frame graph (the middle pass on the async compute queue) and checks
// its result on the CPU, exactly (every value), once the frame that wrote it
// has completed: when its slot is reused METAL_FRAMES_IN_FLIGHT frames later,
// and for the remaining frames at exit.  A missing event between the queues
// makes the consumer read stale data and shows up as mismatches.  Pipelines
// are built once; the readback buffers (shared storage, one per frame in
// flight) come from context.memory().
// ---------------------------------------------------------------------------

class AsyncComputeProbe {
public:
    AsyncComputeProbe(MetalContext& context, PipelineCache& pipelines);
    ~AsyncComputeProbe();

    AsyncComputeProbe(const AsyncComputeProbe&) = delete;
    AsyncComputeProbe& operator=(const AsyncComputeProbe&) = delete;

    /// Add the passes to `graph` (call once per graph build, before the
    /// passes the async work should overlap).
    /// Seed + async reduce; declare before the frame's main passes.
    void addProducers(rg::RenderGraph& graph);
    /// Graphics consume of the async result; declare after them, so that the
    /// async reduce overlaps the passes in between.
    void addConsumer(rg::RenderGraph& graph);

    /// Log the queue syncs of a freshly compiled graph and remember them.
    void onCompiled(const rg::RenderGraph& graph, const rg::CompiledGraph& compiled);

    /// Before recording a frame in `slot` (after MetalContext::beginFrame):
    /// checks the frame that used the slot before, which has completed.
    void beginFrame(u32 slot);
    /// Bind the slot's readback buffer to the executor (every frame).
    void bind(MetalGraphExecutor& executor, u32 slot) const;
    /// The frame was encoded into the slot.
    void frameEncoded(u32 slot, u64 frame);

    /// After MetalContext::waitIdle(): check the remaining frames, print the
    /// ASYNC-COMPUTE line.  Returns true on PASS.
    [[nodiscard]] bool finish();

private:
    void verifySlot(u32 slot);
    pipe::PipelineHandle computePipeline(const char* function);

    MetalContext& context_;
    PipelineCache& pipelines_;

    pipe::PipelineHandle seed_    = pipe::INVALID_PIPELINE;
    pipe::PipelineHandle reduce_  = pipe::INVALID_PIPELINE;
    pipe::PipelineHandle consume_ = pipe::INVALID_PIPELINE;
    // One table per pass: the passes are encoded into different command buffers.
    MTL4::ArgumentTable* seedArgs_    = nullptr;
    MTL4::ArgumentTable* reduceArgs_  = nullptr;
    MTL4::ArgumentTable* consumeArgs_ = nullptr;

    rg::AsyncProbeRefs refs_;
    rg::AsyncProbeExec exec_; // callbacks, built by addProducers()
    std::array<MTL::Buffer*, METAL_FRAMES_IN_FLIGHT> readback_{};
    struct Pending {
        bool valid = false;
        u64  frame = 0;
    };
    std::array<Pending, METAL_FRAMES_IN_FLIGHT> pending_{};

    u64 frames_     = 0;
    u64 checked_    = 0;
    u64 mismatches_ = 0;
    u64 badFrames_  = 0;
    u32 queueSyncs_ = 0;
    bool asyncPass_ = false; // the reduce pass is on the async queue in the compiled graph
    bool compiled_  = false;
};

} // namespace phosphor
