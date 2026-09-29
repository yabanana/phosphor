#pragma once

#include "core/types.h"
#include "platform/metal/metal_context.h"
#include "rendergraph/timing_plan.h"

#include <array>
#include <memory>
#include <vector>

namespace phosphor {

// ---------------------------------------------------------------------------
// GpuTimestamps (F4.1) -- MTL4 counter-heap timestamps of the render graph's
// timed units (rg::TimingPlan).
//
// Measured rules (docs/opt-log.md, "F4 — Spike di osservabilità"): only END
// timestamps are reliable; a render encoder gets one timestamp after the
// Fragment stage (Relaxed), a compute encoder one after each timed pass
// (Precise when it holds several passes, Relaxed otherwise), and every
// commit (the frame's first command buffer and every command buffer that
// opens a new submission after a cross-queue wait) starts with a
// command-buffer timestamp.  A unit lasts from the latest of those points
// on its queue (previous unit end or commit start) to its own end.
//
// One heap holds METAL_FRAMES_IN_FLIGHT ranges of `stride_` queries; a slot's
// range is read on the CPU (resolveCounterRange) when the slot is reused,
// i.e. after its frame has completed.  Ranges are NOT invalidated per frame:
// measured, invalidateCounterRange on one slot while other slots are in
// flight also wipes some of the NEXT frame's writes into that slot (1 frame
// in 3 read back 0).  An entry the GPU did not write keeps a value from an
// older frame instead, so a unit is valid only if its end is not older than
// the frame's first commit start on its queue (always written).  The heap is created at
// graph compile time (bigger plan: a new heap, the old one released after
// the frames in flight); nothing is allocated per frame except the
// autoreleased NSData of resolveCounterRange (drained by the frame's pool).
// Timestamps are mach_absolute_time ticks (24 MHz on Apple silicon).
//
// Semantics: a unit's time is its EXCLUSIVE contribution to its queue's
// timeline (end minus the previous end), so the units of a frame add up to
// the queue's busy span.  Passes the GPU overlaps (no dependency between
// them, or consecutive frames) are charged to whichever ends later; a unit
// that ends before its predecessor is invalid for that frame.
// ---------------------------------------------------------------------------

class GpuTimestamps {
public:
    explicit GpuTimestamps(MetalContext& context);
    ~GpuTimestamps();

    GpuTimestamps(const GpuTimestamps&) = delete;
    GpuTimestamps& operator=(const GpuTimestamps&) = delete;

    /// A new graph was compiled: adopt its plan.  Frames still in flight
    /// were recorded with the previous plan and are discarded.
    void configure(const rg::TimingPlan& plan);

    /// Times of one completed frame, valid until the next beginFrame/drain.
    struct Resolved {
        bool        valid = false;
        u64         frame = 0;
        u32         units = 0;
        const u64*  startTicks = nullptr; // per unit (0 = unknown)
        const u64*  endTicks   = nullptr; // per unit (0 = not written)
        const float* ms        = nullptr; // per unit
        const bool* unitValid  = nullptr; // per unit
        float       sumMs  = 0.0f;        // valid units
        float       spanMs = -1.0f;       // graphics queue: first commit start -> last unit end
    };

    /// Before encoding frame `frameIndex` into `slot` (the slot's previous
    /// frame has completed): returns that previous frame's times, then
    /// prepares the slot's range for the new frame.
    Resolved beginFrame(u32 slot, u64 frameIndex);
    /// After MetalContext::waitIdle(): the times of the frame recorded in
    /// `slot` that nobody resolved yet (call for the slots in frame order).
    Resolved drain(u32 slot);
    /// Frame index recorded in `slot` and not resolved yet (~0 if none).
    [[nodiscard]] u64 pendingFrame(u32 slot) const { return slots_[slot].frame; }

    // --- Encoding (MetalGraphExecutor, render thread) -------------------------
    /// First command in `cmd`, which starts a commit on `queue`.
    void commitStart(MTL4::CommandBuffer* cmd, rg::Queue queue);
    /// End of timed unit `unit` (last command of a render encoder).
    void endUnit(MTL4::RenderCommandEncoder* encoder, u32 unit);
    /// End of timed unit `unit` in a compute encoder.
    void endUnit(MTL4::ComputeCommandEncoder* encoder, u32 unit);

    [[nodiscard]] bool   enabled() const { return heap_ != nullptr && unitCount_ > 0; }
    [[nodiscard]] double tickNs()  const { return tickNs_; }
    [[nodiscard]] const rg::TimingPlan& plan() const { return plan_; }

private:
    struct Slot {
        u64              frame = ~0ull;     // frame recorded in the slot (~0: none / stale)
        std::vector<u32> startQuery;        // per unit, relative to the slot base (~0u: none)
        u32              commitStarts = 0;
        u32              firstGraphicsStart = ~0u;
        std::array<u8, MetalContext::MAX_FRAME_SUBMISSIONS> commitQueue{}; // queue of each commit start
    };

    Resolved resolve(u32 slot);
    [[nodiscard]] u32 base(u32 slot) const { return slot * stride_; }
    [[nodiscard]] u32 queueIndex(rg::Queue queue) const { return queue == rg::Queue::AsyncCompute ? 1u : 0u; }

    MetalContext&        context_;
    MTL4::CounterHeap*   heap_ = nullptr;
    u32                  stride_ = 0;       // queries per slot range
    double               tickNs_ = 41.666666;
    rg::TimingPlan       plan_;
    u32                  unitCount_ = 0;
    std::vector<rg::TimestampKind> kinds_;  // per unit
    std::vector<u8>      unitQueue_;        // per unit (queueIndex)
    std::array<Slot, METAL_FRAMES_IN_FLIGHT> slots_{};

    // Recording state of the frame being encoded.
    u32 recording_ = ~0u;                   // slot
    std::array<u32, 2> lastQuery_{};        // per queue, relative query of the latest end/commit start
    // Latest unit end per queue of the last resolved frame: a commit may
    // start while the previous frame still runs on the GPU (frames overlap
    // without vsync), so a commit start is clamped to it (no double count).
    std::array<u64, 2> lastEndTick_{};

    // Resolution scratch (sized at configure).
    std::vector<u64>   ticks_;
    std::vector<u32>   endQuery_;
    std::vector<u64>   startTicks_;
    std::vector<u64>   endTicks_;
    std::vector<float> ms_;
    std::vector<u8>    validBytes_;
    std::unique_ptr<bool[]> valid_;
};

} // namespace phosphor
