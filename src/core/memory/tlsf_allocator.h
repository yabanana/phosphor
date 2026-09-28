#pragma once

#include "core/types.h"

#include <array>
#include <string>
#include <vector>

namespace phosphor {

// ---------------------------------------------------------------------------
// TlsfAllocator -- Two-Level Segregated Fit allocator over an abstract range
// [0, capacity) (Masmano et al., "TLSF: a new dynamic memory allocator for
// real-time systems", 2004).
//
// It hands out offsets, not memory: the owner maps them onto whatever backs
// the range (a placement MTLHeap, a large buffer, a sparse resource).  Both
// allocate() and free() are O(1): free blocks live in lists indexed by
// (first level = power of two, second level = 1 of 16 linear subdivisions),
// found through two bitmaps.  Physically adjacent free blocks are merged on
// free, so the range never fragments into unusable slivers of free space.
//
// Block metadata lives in an internal node pool (the range itself may not be
// CPU-visible), recycled through a free list: after warm-up, allocate/free
// do not touch the system allocator.
// ---------------------------------------------------------------------------

class TlsfAllocator {
public:
    using Handle = u32;
    static constexpr Handle INVALID_HANDLE = ~0u;

    struct Allocation {
        u64    offset = 0;
        u64    size   = 0; // requested size
        Handle handle = INVALID_HANDLE;

        [[nodiscard]] bool valid() const { return handle != INVALID_HANDLE; }
    };

    struct Stats {
        u64 capacity         = 0;
        u64 usedBytes        = 0; // including alignment padding kept in used blocks
        u64 freeBytes        = 0;
        u64 largestFreeBlock = 0;
        u32 allocationCount  = 0;
        u32 freeBlockCount   = 0;

        /// 0 = all free space is one block; towards 1 = free space is scattered.
        [[nodiscard]] float fragmentation() const {
            return freeBytes == 0 ? 0.0f
                                  : 1.0f - static_cast<float>(largestFreeBlock) / static_cast<float>(freeBytes);
        }
    };

    explicit TlsfAllocator(u64 capacity);

    TlsfAllocator(const TlsfAllocator&) = delete;
    TlsfAllocator& operator=(const TlsfAllocator&) = delete;
    TlsfAllocator(TlsfAllocator&&) = default;
    TlsfAllocator& operator=(TlsfAllocator&&) = default;

    /// Allocate `size` bytes at an offset that is a multiple of `alignment`
    /// (a power of two).  Returns an invalid allocation when no block fits.
    [[nodiscard]] Allocation allocate(u64 size, u64 alignment = 1);

    /// Release an allocation.  Invalid handles are ignored.
    void free(Handle handle);

    /// Release everything (the range becomes one free block).
    void reset();

    [[nodiscard]] u64   capacity() const { return capacity_; }
    /// Metadata nodes ever created (grows only while new peaks are reached).
    [[nodiscard]] size_t metadataNodeCount() const { return nodes_.size(); }
    [[nodiscard]] Stats stats() const;

    /// Check every structural invariant (tests and debug builds).  On failure
    /// returns false and describes the first violation in `error`.
    [[nodiscard]] bool validate(std::string* error = nullptr) const;

private:
    static constexpr u32 SL_BITS  = 4;              // 16 second-level lists
    static constexpr u32 SL_COUNT = 1u << SL_BITS;
    // Sizes >= SL_COUNT map to first level floorLog2(size) - SL_BITS + 1,
    // up to 63 - SL_BITS + 1 for the largest u64 values.
    static constexpr u32 FL_COUNT = 64 - SL_BITS + 1;
    static constexpr u32 NONE     = ~0u;

    struct Node {
        u64  offset   = 0;
        u64  size     = 0; // block size (>= requested size)
        u64  request  = 0; // requested size, for used blocks
        u32  prevPhys = NONE;
        u32  nextPhys = NONE;
        u32  prevFree = NONE;
        u32  nextFree = NONE;
        bool isFree   = false;
        bool live     = false; // node slot in use (block exists)
    };

    struct Index {
        u32 fl = 0;
        u32 sl = 0;
    };

    static Index mappingInsert(u64 size);
    static bool  mappingSearch(u64 size, Index& index);

    u32  newNode();
    void releaseNode(u32 node);
    void insertFree(u32 node);
    void removeFree(u32 node);
    u32  findFree(Index& index) const;
    /// Split `node` so it keeps its first `size` bytes; the rest becomes a new
    /// free block.  Returns the new block's node (or NONE if nothing remains).
    u32 splitTail(u32 node, u64 size);

    u64 capacity_ = 0;
    std::vector<Node> nodes_;
    std::vector<u32>  freeNodes_; // recycled node slots
    u64 flBitmap_ = 0;
    std::array<u32, FL_COUNT> slBitmaps_{};
    std::array<std::array<u32, SL_COUNT>, FL_COUNT> heads_{};
    u64 usedBytes_       = 0;
    u32 allocationCount_ = 0;
};

} // namespace phosphor
