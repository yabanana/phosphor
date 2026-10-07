#pragma once

#include "core/memory/memory_budget.h"
#include "core/memory/tlsf_allocator.h"
#include "core/types.h"
#include "platform/metal/residency_manager.h"

#include <Metal/Metal.hpp>

#include <array>
#include <unordered_map>
#include <vector>

namespace phosphor {

class MetalContext;

// ---------------------------------------------------------------------------
// GpuMemory -- the only place where the backend creates GPU buffers and
// textures.
//
// Private resources are placed in MTLHeapTypePlacement heaps and
// suballocated with a TLSF allocator (sizes/alignments from
// heap*SizeAndAlign): the residency set holds one entry per heap instead of
// one per resource, and creating a resource costs no kernel allocation once
// a heap has room.  Shared and memoryless resources are standalone.
//
// Level content (Geometry, Textures) and engine-lifetime data use separate
// heaps and residency sets (ResidencyClass), so a level change empties and
// trims whole heaps without touching the rest.
//
// Every allocation is labelled, made resident (except memoryless textures)
// and accounted per MemoryCategory.  allocationCount() is monotonic: the
// engine samples it around a benchmark run to prove that no GPU allocation
// happens once the frame loop is warm (rule O7).
//
// release() is deferred: the resource stays resident, alive and its heap
// range reserved until every frame in flight at release time has completed.
// ---------------------------------------------------------------------------

class GpuMemory {
public:
    struct CategoryStats {
        u64 bytes = 0;
        u32 count = 0;
    };

    struct HeapStats {
        u64            size = 0;
        ResidencyClass cls  = ResidencyClass::Static;
        TlsfAllocator::Stats tlsf;
    };

    /// Residency class of a category's allocations.
    [[nodiscard]] static ResidencyClass residencyClass(MemoryCategory category);

    explicit GpuMemory(MetalContext& context);
    ~GpuMemory();

    GpuMemory(const GpuMemory&) = delete;
    GpuMemory& operator=(const GpuMemory&) = delete;

    [[nodiscard]] MTL::Buffer* newBuffer(u64 length, MTL::ResourceOptions options, MemoryCategory category,
                                         const char* label);
    [[nodiscard]] MTL::Texture* newTexture(const MTL::TextureDescriptor* descriptor, MemoryCategory category,
                                           const char* label);
    // A view owns no new storage; its parent texture must remain owned through
    // all readers. Residency/accounting and deferred release still belong here.
    [[nodiscard]] MTL::Texture* newTextureView(MTL::Texture* source,MTL::PixelFormat format,MTL::TextureType type,
                                              NS::Range levels,NS::Range slices,MemoryCategory category,const char* label);
    /// F9: placement allocation with the device-reported AS alignment. The
    /// containing heap is resident; release() owns retirement like buffers.
    [[nodiscard]] MTL::AccelerationStructure* newAccelerationStructure(u64 size, MemoryCategory category,
                                                                       const char* label);
    [[nodiscard]] MTL::IntersectionFunctionTable* newIntersectionFunctionTable(MTL::ComputePipelineState* pipeline,
                                                                              u32 count, MemoryCategory category,
                                                                              const char* label);

    /// Import a page-aligned shared mapping. The callback owns its lifetime
    /// until the completed GPU buffer is actually destroyed.
    [[nodiscard]] MTL::Buffer *newSharedBuffer(void *mapping, u64 length, void (^deallocator)(void *, NS::UInteger),
                                               MemoryCategory category, const char *label);

    /// F5.3: an indirect command buffer (a device object, never placed in a
    /// heap): labelled, made resident and accounted like a standalone buffer;
    /// released with release().
    [[nodiscard]] MTL::IndirectCommandBuffer* newIndirectCommandBuffer(const MTL::IndirectCommandBufferDescriptor* descriptor,
                                                                       u32 maxCommands, MTL::ResourceOptions options,
                                                                       MemoryCategory category, const char* label);

    /// Release after the frames currently in flight complete.  Null is ignored.
    void release(MTL::Resource* resource, MemoryCategory category);

    /// A private placement heap owned by the caller (e.g. TransientHeap, which
    /// places aliasing resources at offsets it chooses).  Made resident as one
    /// allocation and accounted under `category` with its full size.
    [[nodiscard]] MTL::Heap* newPlacementHeap(u64 size, MemoryCategory category, const char* label);
    /// Deferred release of a heap from newPlacementHeap().
    void releaseHeap(MTL::Heap* heap, MemoryCategory category);

    /// Called by MetalContext once frame `completedFrame` has finished on the
    /// GPU (~0 = everything, at shutdown).
    void releaseCompleted(u64 completedFrame);

    /// Release placement heaps that hold no resources, keeping one empty
    /// spare per residency class unless `keepSpare` is false.  GPU must be
    /// idle.  Returns the bytes given back.
    u64 trimEmptyHeaps(bool keepSpare = true);

    /// Allocations made since the GpuMemory was created (monotonic).
    [[nodiscard]] u64 allocationCount() const { return allocationCount_; }
    [[nodiscard]] const CategoryStats& stats(MemoryCategory category) const {
        return stats_[static_cast<u32>(category)];
    }
    [[nodiscard]] u64 totalBytes() const;
    /// Fill `out` (cleared first; reuse it to avoid per-frame allocations).
    void heapStats(std::vector<HeapStats>& out) const;

private:
    struct Placement {
        u32                   heap   = 0;
        TlsfAllocator::Handle handle = TlsfAllocator::INVALID_HANDLE;
        u64                   offset = 0;
    };
    struct Heap {
        MTL::Heap*     heap = nullptr;
        TlsfAllocator  tlsf;
        ResidencyClass cls = ResidencyClass::Static;
    };
    struct Pending {
        MTL::Resource* resource   = nullptr;
        u64            afterFrame = 0;
    };

    /// Reserve `size` bytes aligned to `align` in a heap, creating one if none fits.
    bool place(u64 size, u64 align, ResidencyClass cls, Placement& out);
    void account(MTL::Resource* resource, MemoryCategory category);
    void unaccount(MTL::Resource* resource, MemoryCategory category);
    void destroy(MTL::Resource* resource);

    MetalContext& context_;
    std::array<CategoryStats, MEMORY_CATEGORY_COUNT> stats_{};
    std::array<MemoryBudget::Level, MEMORY_CATEGORY_COUNT> reportedLevel_{}; // last level logged
    u64 allocationCount_ = 0;
    std::vector<Heap> heaps_; // released heaps leave a null entry (indices stay stable)
    std::unordered_map<const MTL::Resource*, Placement> placements_;
    std::vector<Pending> pending_;
    std::unordered_map<MTL::Resource*,MTL::Texture*> viewParents_;
};

} // namespace phosphor
