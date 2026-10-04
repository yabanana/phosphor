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
// or an unexpected owner at release) gets a plain release and is counted.
//
// The extra release is also limited to MetalFX framework versions on which
// the defect was reproduced and the release verified (like a driver-bug
// list). On any other version every scaler gets a plain release: a fixed
// framework then needs no code change, and a still-broken one leaks visibly
// (`retained`, exit 1) until it is verified and listed, instead of risking an
// over-release when a transient owner happens to look like the old shape.
// PHOSPHOR_METALFX_UNVERIFIED=1 forces that path (negative control: FAIL).
//
// Under the GPU capture layer (MTL_CAPTURE_ENABLED, Xcode frame capture) the
// caller holds a capture wrapper around the real scaler: only the wrapper is
// observable, so the inner self-reference cannot be released and every
// recreation still leaks. Those releases are counted as `wrapped` and the
// lifetime result is unverified, never PASS.
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
    u64 wrapped = 0;          // capture wrappers destroyed; the wrapped scaler leaks
    u64 maxReleaseMicros = 0; // longest release (deallocation runs on the caller's thread)
};

/// Wrap a +1 scaler; the deleter calls releaseTemporalScaler. Call after the
/// creating autorelease pool has drained. A null scaler yields an empty pointer.
std::shared_ptr<MTL4FX::TemporalScaler> adoptTemporalScaler(MTL4FX::TemporalScaler *scaler);

enum class ReleaseOutcome { Released, CycleReleased, Wrapped, Retained };
/// Release the caller's reference given the internal count recorded at adoption;
/// `wrapped`: the scaler is a GPU capture wrapper (plain release only).
ReleaseOutcome releaseTemporalScaler(MTL4FX::TemporalScaler *scaler, u32 internalReferences, bool wrapped = false);

LifetimeCounters counters();
/// CFBundleVersion of the loaded MetalFX framework and whether the
/// self-reference release is enabled for it.
const char *frameworkVersion();
bool selfReferenceReleaseEnabled();

} // namespace phosphor::metalfx
