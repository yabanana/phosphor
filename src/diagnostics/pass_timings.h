#pragma once

#include "core/types.h"
#include "diagnostics/bench_report.h"

#include <string>
#include <vector>

namespace phosphor {

namespace rg {
struct TimingPlan;
}

// ---------------------------------------------------------------------------
// PassTimings (F4.1) -- per-unit GPU times from resolved timestamps.
//
// One "unit" is a timed interval of rg::TimingPlan (a render group, or one
// compute pass).  The executor resolves the timestamps of a completed frame
// (raw ticks, 0 = entry never written) and records, per unit, which query
// holds its start (the previous unit's end on the same queue, or a
// commit-start query) -- see rg::TimingPlan for the layout.
//
// Two consumers:
//   * the rolling window (last kWindow frames, average and maximum) for the
//     Pass Timings panel and the overlay;
//   * the measured frames of a benchmark run (beginMeasure/endMeasure),
//     summarised into BenchReport::passes.
// configure() and beginMeasure() allocate; addFrame() never does (O7).
// ---------------------------------------------------------------------------

/// Duration in ms of every unit of one frame: end tick - start tick, where
/// ticks[] holds the frame's resolved queries.  A unit is invalid (valid[u] =
/// false, ms 0) if either tick is 0 or end < start.  `tickNs` = ns per tick
/// (1e9 / MTL::Device::queryTimestampFrequency()).
void computeUnitTimes(const u64* ticks, u32 tickCount, const u32* startQuery, const u32* endQuery, u32 unitCount,
                      double tickNs, float* outMs, bool* outValid);

class PassTimings {
public:
    static constexpr u32 kWindow = 60;

    struct UnitStats {
        float avgMs   = 0.0f;
        float maxMs   = 0.0f;
        u32   samples = 0;  // valid samples in the window
    };

    /// Reset for a new plan (graph recompiled): names, queues, fused flags,
    /// DRAM bytes and pass names are copied; windows and measurements cleared.
    void configure(const rg::TimingPlan& plan, const std::vector<std::string>& passNames,
                   const std::vector<std::string>& passShaders);

    /// One resolved frame: `ms`/`valid` have unitCount() entries.  `frameSpanMs`
    /// = graphics queue span of the frame (first commit start to last unit end),
    /// negative if unknown.
    void addFrame(u64 frameIndex, const float* ms, const bool* valid, float frameSpanMs);

    [[nodiscard]] u32 unitCount() const { return static_cast<u32>(units_.size()); }
    [[nodiscard]] const std::string& unitName(u32 unit) const { return units_[unit].name; }
    [[nodiscard]] UnitStats rolling(u32 unit) const;
    /// Average over the window of the per-frame sum of valid unit times.
    [[nodiscard]] float rollingSumMs() const;
    [[nodiscard]] float rollingSpanMs() const;
    /// Last frame index added (~0 if none).
    [[nodiscard]] u64 lastFrame() const { return lastFrame_; }

    /// Keep every frame added from now on (up to `frames`) for the report.
    void beginMeasure(u32 frames);
    /// Stop keeping frames; returns how many were kept.
    u32 endMeasure();
    /// Per-unit summaries of the measured frames (TimingSummary nearest rank,
    /// invalid samples skipped) plus the per-frame sum.
    void summarize(std::vector<PassReport>& out, TimingSummary& sum, TimingSummary& span) const;

private:
    struct Unit {
        std::string name;
        std::string queue;           // "graphics" / "async"
        bool        fused = false;
        u64         dramBytes = 0;
        std::vector<std::string> passes;
        std::vector<std::string> shaders;
        float window[kWindow] = {};
        bool  windowValid[kWindow] = {};
    };
    std::vector<Unit> units_;
    float sumWindow_[kWindow]  = {};
    float spanWindow_[kWindow] = {};
    u32   windowCount_ = 0;        // frames in the window (<= kWindow)
    u32   windowHead_  = 0;        // next slot
    u64   lastFrame_   = ~0ull;

    bool  measuring_ = false;
    u32   measureCapacity_ = 0;
    u32   measured_ = 0;
    std::vector<float> measuredMs_;    // [frame * units + unit], NaN = invalid
    std::vector<float> measuredSum_;
    std::vector<float> measuredSpan_;
};

} // namespace phosphor
