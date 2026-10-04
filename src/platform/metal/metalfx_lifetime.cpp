#include "platform/metal/metalfx_lifetime.h"
#include "core/log.h"
#include <MetalFX/MetalFX.hpp>
#include <atomic>
#include <chrono>

// Weak-reference entry points of the Objective-C runtime, specified by the
// Clang ARC ABI ("Runtime support"); the SDK has no C++ declaration of them.
extern "C" {
id objc_initWeak(id *location, id value);
id objc_loadWeakRetained(id *location);
void objc_destroyWeak(id *location);
}

namespace phosphor::metalfx {
namespace {
struct Counters {
    std::atomic<u64> adopted{0}, released{0}, cycleReleased{0}, retained{0}, unknownSignature{0}, maxReleaseMicros{0};
};
Counters &state() {
    static Counters c;
    return c;
}
id asId(MTL4FX::TemporalScaler *p) { return reinterpret_cast<id>(p); }
} // namespace

std::shared_ptr<MTL4FX::TemporalScaler> adoptTemporalScaler(MTL4FX::TemporalScaler *scaler) {
    if (!scaler)
        return {};
    // The caller owns exactly one reference (new... family), so anything above
    // one is held by the scaler's own implementation at creation time.
    const auto count = scaler->retainCount();
    const u32 internal = count > 1 ? static_cast<u32>(count - 1) : 0;
    state().adopted.fetch_add(1, std::memory_order_relaxed);
    if (internal > 1) {
        state().unknownSignature.fetch_add(1, std::memory_order_relaxed);
        LOG_WARN("MetalFX temporal scaler created with %u internal references; only one is a known self-reference, "
                 "releasing normally",
                 internal);
    }
    return std::shared_ptr<MTL4FX::TemporalScaler>(
        scaler, [internal](MTL4FX::TemporalScaler *p) { releaseTemporalScaler(p, internal); });
}

ReleaseOutcome releaseTemporalScaler(MTL4FX::TemporalScaler *scaler, u32 internalReferences) {
    if (!scaler)
        return ReleaseOutcome::Released;
    const auto begin = std::chrono::steady_clock::now();
    id weak = nullptr;
    objc_initWeak(&weak, asId(scaler));
    scaler->release();
    ReleaseOutcome outcome = ReleaseOutcome::Released;
    if (id alive = objc_loadWeakRetained(&weak)) {
        auto *object = reinterpret_cast<MTL4FX::TemporalScaler *>(alive);
        // Our probe plus, for the known defect, the single self-reference.
        const bool onlySelf = internalReferences == 1 && object->retainCount() == 2;
        object->release(); // the probe
        outcome = ReleaseOutcome::Retained;
        if (onlySelf) {
            // Nothing else owns the scaler: drop the reference its filter keeps
            // on it. Its later release during deallocation is a runtime no-op.
            object->release();
            id after = objc_loadWeakRetained(&weak);
            if (after)
                reinterpret_cast<MTL4FX::TemporalScaler *>(after)->release();
            else
                outcome = ReleaseOutcome::CycleReleased;
        }
    }
    objc_destroyWeak(&weak);
    const u64 micros = static_cast<u64>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin).count());
    for (u64 seen = state().maxReleaseMicros.load(std::memory_order_relaxed);
         micros > seen && !state().maxReleaseMicros.compare_exchange_weak(seen, micros, std::memory_order_relaxed);) {
    }
    switch (outcome) {
    case ReleaseOutcome::Released:
        state().released.fetch_add(1, std::memory_order_relaxed);
        break;
    case ReleaseOutcome::CycleReleased:
        state().cycleReleased.fetch_add(1, std::memory_order_relaxed);
        break;
    case ReleaseOutcome::Retained:
        state().retained.fetch_add(1, std::memory_order_relaxed);
        LOG_WARN("MetalFX temporal scaler still referenced after release (internal references at creation: %u)",
                 internalReferences);
        break;
    }
    return outcome;
}

LifetimeCounters counters() {
    const auto &c = state();
    return {c.adopted.load(),  c.released.load(),         c.cycleReleased.load(), c.retained.load(),
            c.unknownSignature.load(), c.maxReleaseMicros.load()};
}

} // namespace phosphor::metalfx
