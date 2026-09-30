#pragma once

#include "core/types.h"
#include "pipeline/pipeline_registry.h"
#include "platform/metal/metal_context.h"
#include "rendergraph/render_graph.h"

namespace phosphor {

class PipelineCache;

// ---------------------------------------------------------------------------
// KnownCostPass -- --debug-gpu-cost N (F4.1 negative control).
//
// A compute pass ("Known cost", graphics queue, side effect) in which each of
// kThreads threads runs N LCG steps (shaders/known_cost.metal) and writes a
// transient buffer.  Its measured GPU time must grow linearly with N and the
// other passes must not move: a timing mechanism that attributes work to the
// wrong pass, or measures the encoder instead of the work, fails that.
// ---------------------------------------------------------------------------

class KnownCostPass {
public:
    static constexpr u32 kThreads = 1u << 20;

    KnownCostPass(MetalContext& context, PipelineCache& pipelines, u32 iterations);
    ~KnownCostPass();

    KnownCostPass(const KnownCostPass&) = delete;
    KnownCostPass& operator=(const KnownCostPass&) = delete;

    void addToGraph(rg::RenderGraph& graph);
    /// The pass's output: the forward pass declares a read of it so the two
    /// never overlap on the GPU (overlapping passes share their time).
    [[nodiscard]] rg::BufferRef output() const { return out_; }

private:
    MetalContext&        context_;
    PipelineCache&       pipelines_;
    u32                  iterations_ = 0;
    pipe::PipelineHandle pipeline_ = pipe::INVALID_PIPELINE;
    MTL4::ArgumentTable* args_ = nullptr;
    rg::BufferRef        out_;
};

} // namespace phosphor
