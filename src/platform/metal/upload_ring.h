#pragma once

#include "core/memory/linear_ring.h"
#include "core/types.h"

#include <Metal/Metal.hpp>

namespace phosphor {

class GpuMemory;

// ---------------------------------------------------------------------------
// UploadRing -- CPU-written, GPU-read memory in one shared write-combined
// buffer (S-MEM-3), suballocated by a LinearRing.
//
// Two uses, both owned by MetalContext:
//   frame uploads  per-frame constants/instances/UI geometry; beginFrame()
//                  and endFrame() follow the frame loop, so a frame's slices
//                  stay valid until its GPU work completes.
//   staging        loading-time copies into private resources; the owner
//                  calls reset() once the GPU has consumed them.
//
// allocate() never fails: if the ring is full it returns a one-off buffer
// (released after the frame, counted by GpuMemory so the zero-allocation
// check sees it) and grows the ring at the next beginFrame().
// tryAllocate() fails instead, so the staging path can flush and retry.
// ---------------------------------------------------------------------------

class UploadRing {
public:
    struct Slice {
        u8*             cpu    = nullptr;
        MTL::GPUAddress gpu    = 0;
        MTL::Buffer*    buffer = nullptr;
        u64             offset = 0;

        explicit operator bool() const { return cpu != nullptr; }
    };

    UploadRing(GpuMemory& memory, u64 capacity, u32 framesInFlight, const char* label);
    ~UploadRing();

    UploadRing(const UploadRing&) = delete;
    UploadRing& operator=(const UploadRing&) = delete;

    void beginFrame(u64 frameIndex);
    void endFrame();

    /// Slice of `size` bytes, or an empty slice if the ring is full.
    [[nodiscard]] Slice tryAllocate(u64 size, u64 alignment = 256);
    /// Slice of `size` bytes; falls back to a one-off buffer when full.
    [[nodiscard]] Slice allocate(u64 size, u64 alignment = 256);

    /// Forget every slice (GPU idle, e.g. after a staging flush).
    void reset();

    [[nodiscard]] LinearRing::Stats stats() const { return ring_.stats(); }
    [[nodiscard]] u32 oneOffBuffers() const { return oneOffBuffers_; }

private:
    Slice sliceAt(MTL::Buffer* buffer, u64 offset) const;

    GpuMemory&   memory_;
    const char*  label_;
    LinearRing   ring_;
    MTL::Buffer* buffer_ = nullptr;
    u64          growTo_ = 0; // capacity requested by an overflow, applied at beginFrame()
    u32          oneOffBuffers_ = 0;
};

} // namespace phosphor
