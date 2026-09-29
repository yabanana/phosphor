#pragma once

#include "core/types.h"

#include <Metal/Metal.hpp>

namespace phosphor {

class MetalContext;

// ---------------------------------------------------------------------------
// TransientHeap -- one MTLHeapTypePlacement heap for resources that live only
// within a frame (render-graph intermediates, F1.1 / F2.2).
//
// Resources are placed at offsets chosen by the caller and may overlap: two
// intermediates whose lifetimes do not intersect share memory (aliasing).
// Metal 4 does not track this, so the first pass that uses memory previously
// used by another resource must be preceded by a barrier with
// MTL4::VisibilityOptionResourceAlias, and must not read the new resource
// before writing it (its contents are undefined).
//
// Resources are created once (when the frame graph is built), not per frame
// (O7).  The heap is accounted as one Transient allocation and is resident
// as a whole.
// ---------------------------------------------------------------------------

class TransientHeap {
public:
    explicit TransientHeap(MetalContext& context);
    ~TransientHeap();

    TransientHeap(const TransientHeap&) = delete;
    TransientHeap& operator=(const TransientHeap&) = delete;

    /// Ensure the heap holds at least `bytes`.  Growing replaces the heap
    /// (the old one is released after the frames in flight), which
    /// invalidates every resource created from it: call between frames,
    /// before (re)creating the resources.  Returns false on failure.
    bool reserve(u64 bytes);

    /// Heap footprint of a resource, for the caller's offset assignment.
    [[nodiscard]] MTL::SizeAndAlign sizeAndAlign(const MTL::TextureDescriptor* descriptor) const;
    [[nodiscard]] MTL::SizeAndAlign sizeAndAlign(u64 length) const;

    /// Resources at `offset` (must satisfy sizeAndAlign and fit in the heap).
    /// Returns null on invalid placement.
    [[nodiscard]] MTL::Texture* createTexture(const MTL::TextureDescriptor* descriptor, u64 offset,
                                              const char* label);
    [[nodiscard]] MTL::Buffer* createBuffer(u64 length, u64 offset, const char* label);

    /// Deferred release of a resource created here.
    /// `evict`: the resource was made resident on its own (see the executor).
    void release(MTL::Resource* resource, bool evict = false);

    [[nodiscard]] u64 size() const { return heap_ ? heap_->size() : 0; }

private:
    bool fits(u64 offset, const MTL::SizeAndAlign& sa, const char* label) const;

    MetalContext& context_;
    MTL::Heap*    heap_ = nullptr;
};

} // namespace phosphor
