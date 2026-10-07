#pragma once

#include "core/types.h"
#include "rendergraph/render_graph.h"

#include <string>
#include <vector>

namespace phosphor::rg {

// ---------------------------------------------------------------------------
// TimingPlan (F4.1) -- which GPU intervals of a compiled graph are timed.
//
// Measured on M5 Max (docs/opt-log.md, "F4 — Spike di osservabilità"):
//   * a timestamp written at the START of an encoder is written late when
//     the previous work is long, so only END timestamps are used: an
//     interval is end(unit) - end(previous unit on the same queue), or
//     - commit start, a command-buffer timestamp written at the beginning
//     of every commit that follows a cross-queue wait (and of the frame);
//   * inside a render encoder every timestamp lands at the end of the render
//     pass (at most 4 are written): passes fused into one render group are
//     ONE unit, its members are reported as `fused`;
//   * in a compute encoder a Precise timestamp after each pass is distinct,
//     so a compute encoder with several passes has one unit per pass.
//
// Units are listed in encode order.  Query layout of one frame slot:
// [0, units.size())                   end timestamp of unit i,
// [units.size(), queriesPerFrame())   commit-start timestamps, in the order
//                                     the executor writes them.
// ---------------------------------------------------------------------------

enum class TimestampKind : u8 {
    RenderEnd,       // end of a render encoder (afterStage Fragment, Relaxed)
    ComputeEnd,      // end of a single-pass compute encoder (Relaxed)
    ComputePassEnd,  // after one pass of a multi-pass compute encoder (Precise)
};

struct TimedUnit {
    std::string   name;              // pass name, or "A + B" for a fused render group
    Queue         queue = Queue::Graphics;
    TimestampKind kind  = TimestampKind::RenderEnd;
    u32           encoder       = 0; // index into CompiledGraph::encoders
    u32           firstPosition = 0; // positions (CompiledGraph::order) covered, inclusive
    u32           lastPosition  = 0;
    bool          fused         = false; // more than one pass in one render group
    std::vector<u32> passes;         // pass indices (RenderGraph::passes()) covered
    u64           dramBytes     = 0; // estimatePassBandwidth summed over `passes`
    // Union of explicitly declared read/write stages of the covered passes.
    // Resource kind and pass name do not imply a stage: tracing an AS at
    // Dispatch stays Dispatch; an AS build/refit carries StageAccelerationStructure.
    Stages        stages        = StageNone;
    // External frameworks declare StageExternal (a union containing AS), but
    // own their completion fence/dispatch. Only compute AS work needs this anchor.
    bool          needsAccelerationStructureAnchor = false;
};

struct TimingPlan {
    std::vector<TimedUnit> units;
    /// Per position: the unit that covers it (~0u for none).
    std::vector<u32> unitOfPosition;
    /// Commit-start queries reserved per frame (after the unit queries).
    u32 commitStartQueries = 0;

    [[nodiscard]] u32 queriesPerFrame() const { return static_cast<u32>(units.size()) + commitStartQueries; }
};

/// Build the plan of a successfully compiled graph.  `maxCommits` bounds the
/// commit-start queries (MetalContext::MAX_FRAME_SUBMISSIONS in the engine).
/// Culled passes are not covered.
[[nodiscard]] TimingPlan buildTimingPlan(const RenderGraph& graph, const CompiledGraph& compiled, u32 maxCommits);

/// Estimated DRAM bytes per pass (index = RenderGraph pass index; culled
/// passes 0).  Same model as estimateBandwidth() (graph_dump.h): attachment
/// load/store traffic of a render group is charged to the pass that performs
/// it (loads to the first member that touches the attachment, stores to the
/// last), memoryless resources cost nothing; shader reads/writes and copies
/// are charged to their pass.  The per-pass sum equals
/// estimateBandwidth().totalBytes().
struct PassTraffic {
    u64 readBytes  = 0;
    u64 writeBytes = 0;
    [[nodiscard]] u64 total() const { return readBytes + writeBytes; }
};
[[nodiscard]] std::vector<PassTraffic> estimatePassBandwidth(const RenderGraph& graph, const CompiledGraph& compiled);

} // namespace phosphor::rg
