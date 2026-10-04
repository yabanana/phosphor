#pragma once
// MetalFX temporal scaler ownership (F8.4).
//
// MetalFX 40.9 (macOS 27.2, M5 Max) returns temporal scalers that hold one
// strong reference to themselves through their internal C++ filter (Metal 3
// and Metal 4 APIs, measured: retain count 2 right after creation, 1 owner).
// A plain release leaves the scaler and its GPU resources alive forever.
//
// adoptTemporalScaler() records that internal count once, after the creating
// autorelease pool has drained. releaseTemporalScaler() drops the caller's
// reference and, only when the object survives with exactly the recorded
// self-reference left, releases that one too; a weak reference then proves
// the deallocation. Any other shape (no internal reference, more than one,
// or an unexpected owner at release) gets a plain release and is counted, so
// a framework fix turns this into an ordinary release with no code change.
//
// Precondition for every release: no GPU work that uses the scaler is still
// pending (callers defer past the frames in flight or wait for idle).
#include "core/types.h"
#include <memory>

namespace MTL4FX {
class TemporalScaler;
}

namespace phosphor::metalfx {

struct LifetimeCounters {
    u64 adopted = 0;          // scalers created through adoptTemporalScaler
    u64 released = 0;         // destroyed by the caller's release alone
    u64 cycleReleased = 0;    // destroyed after dropping the internal self-reference
    u64 retained = 0;         // still alive after release (leaked)
    u64 unknownSignature = 0; // internal count other than 0 or 1 at adoption
    u64 maxReleaseMicros = 0; // longest release (deallocation runs on the caller's thread)
};

/// Wrap a +1 scaler; the deleter calls releaseTemporalScaler. Call after the
/// creating autorelease pool has drained. A null scaler yields an empty pointer.
std::shared_ptr<MTL4FX::TemporalScaler> adoptTemporalScaler(MTL4FX::TemporalScaler *scaler);

enum class ReleaseOutcome { Released, CycleReleased, Retained };
/// Release the caller's reference given the internal count recorded at adoption.
ReleaseOutcome releaseTemporalScaler(MTL4FX::TemporalScaler *scaler, u32 internalReferences);

LifetimeCounters counters();

} // namespace phosphor::metalfx
