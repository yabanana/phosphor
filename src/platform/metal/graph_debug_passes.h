#pragma once

#include "core/types.h"
#include "platform/metal/metal_context.h"
#include "rendergraph/graph_debug_reference.h"

#include <array>

namespace phosphor {

class MetalGraphExecutor;

// ---------------------------------------------------------------------------
// GraphDebugPasses -- --debug-graph-transients (F2.2).
//
// Adds the synthetic pass chain of rg::addDebugChain to the frame graph and
// checks its result on the CPU, exactly (every value), once the frame that
// wrote it has completed: when its slot is reused METAL_FRAMES_IN_FLIGHT
// frames later, and for the remaining frames at exit.  Pipelines are built
// once, through the context's MTL4 compiler; the readback buffers (shared
// storage, one per frame in flight) come from context.memory().
// ---------------------------------------------------------------------------

class GraphDebugPasses {
public:
    explicit GraphDebugPasses(MetalContext& context);
    ~GraphDebugPasses();

    GraphDebugPasses(const GraphDebugPasses&) = delete;
    GraphDebugPasses& operator=(const GraphDebugPasses&) = delete;

    /// Add the passes to `graph` (call once per graph build).
    void addToGraph(rg::RenderGraph& graph);

    /// Log the aliasing plan of a freshly compiled graph and remember it.
    void onCompiled(const rg::RenderGraph& graph, const rg::CompiledGraph& compiled);

    /// Before recording a frame in `slot` (after MetalContext::beginFrame):
    /// checks the frame that used the slot before, which has completed.
    void beginFrame(u32 slot);
    /// Bind the slot's readback buffer to the executor (every frame).
    void bind(MetalGraphExecutor& executor, u32 slot) const;
    /// The frame was encoded into the slot.
    void frameEncoded(u32 slot, u64 frame);

    /// After MetalContext::waitIdle(): check the remaining frames, print the
    /// GRAPH-TRANSIENTS line.  Returns true on PASS.
    [[nodiscard]] bool finish();

private:
    void buildPipelines();
    void verifySlot(u32 slot);
    MTL::ComputePipelineState* computePipeline(const char* function);

    MetalContext& context_;

    MTL::ComputePipelineState* fill_     = nullptr;
    MTL::ComputePipelineState* reduce_   = nullptr;
    MTL::ComputePipelineState* expand_   = nullptr;
    MTL::ComputePipelineState* checksum_ = nullptr;
    MTL::RenderPipelineState*  raster_   = nullptr;
    MTL4::ArgumentTable*       computeArgs_ = nullptr;
    MTL4::ArgumentTable*       rasterArgs_  = nullptr;

    rg::DebugChainRefs refs_;
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
    rg::DebugAliasSummary alias_;
    bool compiled_ = false;
};

} // namespace phosphor
