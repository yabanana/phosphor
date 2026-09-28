#include "core/memory/linear_ring.h"

#include <algorithm>

namespace phosphor {

LinearRing::LinearRing(u64 capacity, u32 framesInFlight)
    : capacity_(capacity), framesInFlight_(std::max(framesInFlight, 1u)),
      frameEnds_(framesInFlight_, 0), slotUsed_(framesInFlight_, false) {}

void LinearRing::reset() {
    head_ = tail_ = frameStart_ = 0;
    std::fill(frameEnds_.begin(), frameEnds_.end(), 0);
    std::fill(slotUsed_.begin(), slotUsed_.end(), false);
}

void LinearRing::resize(u64 capacity) {
    capacity_ = capacity;
    reset();
}

void LinearRing::beginFrame(u64 frameIndex) {
    currentFrame_ = frameIndex;
    const size_t slot = static_cast<size_t>(frameIndex % framesInFlight_);
    // The previous user of this slot has completed: everything it (and any
    // earlier frame) allocated is free again.
    if (slotUsed_[slot]) tail_ = std::max(tail_, frameEnds_[slot]);
    frameStart_ = head_;
}

std::optional<u64> LinearRing::allocate(u64 size, u64 alignment) {
    if (size == 0 || size > capacity_ || alignment == 0 || (alignment & (alignment - 1)) != 0) {
        ++overflows_;
        return std::nullopt;
    }
    // Align the physical offset (the capacity need not be a multiple of the
    // alignment), and never straddle the end of the ring: skip to the next
    // lap, whose offset 0 satisfies any alignment.
    u64 lapStart = head_ - head_ % capacity_;
    u64 offset = (head_ - lapStart + alignment - 1) & ~(alignment - 1);
    if (offset + size > capacity_) {
        lapStart += capacity_;
        offset = 0;
    }
    const u64 pos = lapStart + offset;
    if (pos + size - tail_ > capacity_) {
        ++overflows_;
        return std::nullopt;
    }
    head_ = pos + size;
    return offset;
}

void LinearRing::endFrame() {
    const size_t slot = static_cast<size_t>(currentFrame_ % framesInFlight_);
    frameEnds_[slot] = head_;
    slotUsed_[slot] = true;
    peakFrameBytes_ = std::max(peakFrameBytes_, head_ - frameStart_);
}

LinearRing::Stats LinearRing::stats() const {
    Stats s;
    s.capacity       = capacity_;
    s.inFlightBytes  = head_ - tail_;
    s.frameBytes     = head_ - frameStart_;
    s.peakFrameBytes = std::max(peakFrameBytes_, s.frameBytes);
    s.overflows      = overflows_;
    return s;
}

} // namespace phosphor
