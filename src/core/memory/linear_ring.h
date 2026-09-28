#pragma once

#include "core/types.h"

#include <optional>
#include <vector>

namespace phosphor {

// ---------------------------------------------------------------------------
// LinearRing -- per-frame linear suballocation in a ring of `capacity` bytes
// shared by `framesInFlight` frames.
//
// Each frame bump-allocates after the previous frame's data; nothing is freed
// individually.  When frame n begins, the frame that used the same slot
// (n - framesInFlight) has finished on the GPU -- the caller guarantees this,
// as MetalContext::beginFrame() does -- so its bytes become reusable.
// Allocations never straddle the end of the ring.  Offsets only; the owner
// maps them onto a buffer.
// ---------------------------------------------------------------------------

class LinearRing {
public:
    struct Stats {
        u64 capacity       = 0;
        u64 inFlightBytes  = 0; // bytes held by frames not yet recycled
        u64 frameBytes     = 0; // bytes allocated by the current frame
        u64 peakFrameBytes = 0;
        u32 overflows      = 0; // allocations that did not fit
    };

    LinearRing(u64 capacity, u32 framesInFlight);

    /// Start frame `frameIndex` (consecutive, starting at 0): recycles the
    /// bytes of frame `frameIndex - framesInFlight`.
    void beginFrame(u64 frameIndex);

    /// Offset of `size` bytes aligned to `alignment` (a power of two), or
    /// nullopt if the ring cannot hold it without overwriting frames in flight.
    [[nodiscard]] std::optional<u64> allocate(u64 size, u64 alignment);

    /// Close the current frame; its bytes stay reserved until recycled.
    void endFrame();

    /// Drop everything (only when no frame is in flight on the GPU).
    void reset();

    /// Change the capacity; implies reset() (only with the GPU idle).
    void resize(u64 capacity);

    [[nodiscard]] u64   capacity() const { return capacity_; }
    [[nodiscard]] Stats stats() const;

private:
    u64 capacity_ = 0;
    u32 framesInFlight_ = 0;
    // Monotonic byte positions: physical offset = position % capacity.
    u64 head_ = 0;
    u64 tail_ = 0;
    u64 frameStart_ = 0;
    std::vector<u64> frameEnds_; // head at endFrame(), per slot
    std::vector<bool> slotUsed_;
    u64 currentFrame_ = 0;
    u64 peakFrameBytes_ = 0;
    u32 overflows_ = 0;
};

} // namespace phosphor
