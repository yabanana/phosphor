#pragma once

#include "core/types.h"

#include <memory>

namespace phosphor {

namespace rg {
struct TimingPlan;
}

// ---------------------------------------------------------------------------
// TracyGpuZones (F4.2) -- GPU zones in Tracy from our own MTL4 timestamps.
//
// Tracy's Metal backend samples MTLCounterSampleBuffer, which MTL4 encoders
// do not support (opt-log, F4 spike 2), so zones are emitted by hand with
// Tracy's C API (serial variants): one GPU context per queue ("graphics",
// "async"), period = ns per tick; MTL4 timestamps are mach_absolute_time
// ticks, the same counter Tracy reads on Apple arm64 (CNTVCT_EL0), so the
// context needs no calibration.  Zones are emitted when a frame's
// timestamps are resolved (3 frames after encoding).
//
// Without PHOSPHOR_TRACY every method is an inline no-op.
// ---------------------------------------------------------------------------

class TracyGpuZones {
public:
    TracyGpuZones();
    ~TracyGpuZones();

    /// Create the contexts on first use (`tickNs` = ns per GPU tick, current
    /// GPU time in ticks as reference) and one source location per unit of
    /// `plan` (names copied into owned, stable storage).  Graph compile time
    /// only (allocates).
    void configure(const rg::TimingPlan& plan, double tickNs, u64 nowTicks);

    /// Emit the units of one resolved frame: raw start/end ticks per unit
    /// (unitCount = plan units; entries with start == 0 or end < start are
    /// skipped).  No allocation.
    void emitFrame(const u64* startTicks, const u64* endTicks, u32 unitCount);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#ifndef PHOSPHOR_TRACY
struct TracyGpuZones::Impl {};
inline TracyGpuZones::TracyGpuZones() = default;
inline TracyGpuZones::~TracyGpuZones() = default;
inline void TracyGpuZones::configure(const rg::TimingPlan&, double, u64) {}
inline void TracyGpuZones::emitFrame(const u64*, const u64*, u32) {}
#endif

} // namespace phosphor
