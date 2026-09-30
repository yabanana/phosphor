#include "platform/metal/gpu_memory.h"
#include "platform/metal/metal_context.h"
#include "core/log.h"
#include "core/profile.h"

#include <algorithm>

namespace phosphor {

namespace {

NS::String* str(const char* s) {
    return NS::String::string(s, NS::UTF8StringEncoding);
}

bool isMemoryless(const MTL::Resource* resource) {
    return resource->storageMode() == MTL::StorageModeMemoryless;
}

u64 alignUp(u64 v, u64 alignment) { return (v + alignment - 1) & ~(alignment - 1); }

// Storage mode bits of MTLResourceOptions (MTLResourceStorageModeShift = 4);
// metal-cpp exposes the values but no mask.
constexpr MTL::ResourceOptions kStorageModeMask = 0xF0;

} // namespace

GpuMemory::GpuMemory(MetalContext& context) : context_(context) {}

ResidencyClass GpuMemory::residencyClass(MemoryCategory category) {
    return category == MemoryCategory::Geometry || category == MemoryCategory::Textures ? ResidencyClass::Streaming
                                                                                        : ResidencyClass::Static;
}

GpuMemory::~GpuMemory() {
    releaseCompleted(~u64{0});
    for (u32 c = 0; c < MEMORY_CATEGORY_COUNT; ++c) {
        if (stats_[c].count != 0) {
            LOG_WARN("GpuMemory: %u %s allocations (%llu bytes) still alive at shutdown", stats_[c].count,
                     memoryCategoryName(static_cast<MemoryCategory>(c)),
                     static_cast<unsigned long long>(stats_[c].bytes));
        }
    }
    for (Heap& h : heaps_) {
        if (!h.heap) continue;
        PH_FREE(h.heap, "GPU heap pages");
        h.heap->release();
    }
}

bool GpuMemory::place(u64 size, u64 align, ResidencyClass cls, Placement& out) {
    for (u32 i = 0; i < heaps_.size(); ++i) {
        if (!heaps_[i].heap || heaps_[i].cls != cls) continue;
        const bool wasEmpty = heaps_[i].tlsf.stats().allocationCount == 0;
        const TlsfAllocator::Allocation a = heaps_[i].tlsf.allocate(size, align);
        if (a.valid()) {
            // A spare heap may have been left volatile by trimEmptyHeaps():
            // take it back before placing anything in it.
            if (wasEmpty) heaps_[i].heap->setPurgeableState(MTL::PurgeableStateNonVolatile);
            out = {i, a.handle, a.offset};
            return true;
        }
    }

    MTL::HeapDescriptor* desc = MTL::HeapDescriptor::alloc()->init();
    desc->setType(MTL::HeapTypePlacement);
    desc->setStorageMode(MTL::StorageModePrivate);
    desc->setHazardTrackingMode(MTL::HazardTrackingModeUntracked);
    // Page size from the budget (MemoryBudget::heapPageSize); larger resources get their own heap.
    desc->setSize(std::max(context_.budget().heapPageSize(), alignUp(size, align)));
    MTL::Heap* heap = context_.device()->newHeap(desc);
    desc->release();
    if (!heap) {
        LOG_ERROR("GpuMemory: failed to create a placement heap for %llu bytes",
                  static_cast<unsigned long long>(size));
        return false;
    }
    heap->setLabel(str("Placement heap"));
    context_.makeResident(heap, cls); // one residency entry for everything placed in it
    ++allocationCount_;
    PH_ALLOC(heap, heap->size(), "GPU heap pages");

    // Reuse a slot left by a trimmed heap so indices stay small.
    u32 index = static_cast<u32>(heaps_.size());
    for (u32 i = 0; i < heaps_.size(); ++i) {
        if (!heaps_[i].heap) {
            index = i;
            break;
        }
    }
    Heap entry{heap, TlsfAllocator(heap->size()), cls};
    if (index == heaps_.size()) {
        heaps_.push_back(std::move(entry));
    } else {
        heaps_[index] = std::move(entry);
    }
    const TlsfAllocator::Allocation a = heaps_[index].tlsf.allocate(size, align);
    if (!a.valid()) return false;
    out = {index, a.handle, a.offset};
    return true;
}

namespace {

// Tracy pool names (static strings, one per MemoryCategory).  Every resource
// (standalone or placed in a heap page) is reported by its pointer and size
// in its category pool.  Heaps themselves are reported in separate pools
// (caller-owned heaps per category, internal pages in "GPU heap pages"), so
// a page and the resources placed in it are never counted in one pool.
[[maybe_unused]] const char* tracyPool(MemoryCategory category) {
    switch (category) {
    case MemoryCategory::Geometry:      return "GPU Geometry";
    case MemoryCategory::Textures:      return "GPU Textures";
    case MemoryCategory::Upload:        return "GPU Upload";
    case MemoryCategory::Transient:     return "GPU Transient";
    case MemoryCategory::RenderTargets: return "GPU Render targets";
    case MemoryCategory::Other:         return "GPU Other";
    case MemoryCategory::COUNT:         break;
    }
    return "GPU ?";
}

[[maybe_unused]] const char* tracyHeapPool(MemoryCategory category) {
    switch (category) {
    case MemoryCategory::Geometry:      return "GPU heap Geometry";
    case MemoryCategory::Textures:      return "GPU heap Textures";
    case MemoryCategory::Upload:        return "GPU heap Upload";
    case MemoryCategory::Transient:     return "GPU heap Transient";
    case MemoryCategory::RenderTargets: return "GPU heap Render targets";
    case MemoryCategory::Other:         return "GPU heap Other";
    case MemoryCategory::COUNT:         break;
    }
    return "GPU heap ?";
}

} // namespace

void GpuMemory::account(MTL::Resource* resource, MemoryCategory category) {
    const u32 c = static_cast<u32>(category);
    CategoryStats& s = stats_[c];
    s.bytes += resource->allocatedSize();
    ++s.count;
    ++allocationCount_;
    PH_ALLOC(resource, resource->allocatedSize(), tracyPool(category));

    // Log when a category crosses into Warning or Over (once per crossing).
    const MemoryBudget::Level level = context_.budget().level(category, s.bytes);
    if (level > reportedLevel_[c]) {
        LOG_WARN("GPU memory budget: %s at %.1f of %.1f MiB (%s)", memoryCategoryName(category),
                 static_cast<double>(s.bytes) / (1 << 20),
                 static_cast<double>(context_.budget().limit(category)) / (1 << 20),
                 level == MemoryBudget::Level::Over ? "over budget" : "warning");
    }
    reportedLevel_[c] = level;
}

void GpuMemory::unaccount(MTL::Resource* resource, MemoryCategory category) {
    const u32 c = static_cast<u32>(category);
    CategoryStats& s = stats_[c];
    s.bytes -= resource->allocatedSize();
    --s.count;
    PH_FREE(resource, tracyPool(category));
    reportedLevel_[c] = std::min(reportedLevel_[c], context_.budget().level(category, s.bytes));
}

MTL::Buffer* GpuMemory::newBuffer(u64 length, MTL::ResourceOptions options, MemoryCategory category,
                                  const char* label) {
    MTL::Buffer* buffer = nullptr;
    const bool isPrivate = (options & kStorageModeMask) == MTL::ResourceStorageModePrivate;
    if (isPrivate) {
        const MTL::SizeAndAlign sa = context_.device()->heapBufferSizeAndAlign(length, options);
        Placement p;
        if (place(sa.size, sa.align, residencyClass(category), p)) {
            buffer = heaps_[p.heap].heap->newBuffer(length, options, p.offset);
            if (buffer) {
                placements_[buffer] = p;
            } else {
                heaps_[p.heap].tlsf.free(p.handle);
            }
        }
    }
    if (!buffer) {
        buffer = context_.device()->newBuffer(length, options);
        if (!buffer) {
            LOG_ERROR("GpuMemory: failed to allocate %llu-byte buffer '%s'",
                      static_cast<unsigned long long>(length), label);
            return nullptr;
        }
        context_.makeResident(buffer, residencyClass(category));
    }
    buffer->setLabel(str(label));
    account(buffer, category);
    return buffer;
}

MTL::Texture* GpuMemory::newTexture(const MTL::TextureDescriptor* descriptor, MemoryCategory category,
                                    const char* label) {
    MTL::Texture* texture = nullptr;
    if (descriptor->storageMode() == MTL::StorageModePrivate) {
        const MTL::SizeAndAlign sa = context_.device()->heapTextureSizeAndAlign(descriptor);
        Placement p;
        if (place(sa.size, sa.align, residencyClass(category), p)) {
            texture = heaps_[p.heap].heap->newTexture(descriptor, p.offset);
            if (texture) {
                placements_[texture] = p;
            } else {
                heaps_[p.heap].tlsf.free(p.handle);
            }
        }
    }
    if (!texture) {
        texture = context_.device()->newTexture(descriptor);
        if (!texture) {
            LOG_ERROR("GpuMemory: failed to allocate %lux%lu texture '%s'",
                      static_cast<unsigned long>(descriptor->width()),
                      static_cast<unsigned long>(descriptor->height()), label);
            return nullptr;
        }
        // Memoryless textures live in tile memory only: nothing to make resident.
        if (!isMemoryless(texture)) context_.makeResident(texture, residencyClass(category));
    }
    texture->setLabel(str(label));
    account(texture, category);
    return texture;
}

void GpuMemory::release(MTL::Resource* resource, MemoryCategory category) {
    if (!resource) return;
    unaccount(resource, category);
    pending_.push_back({resource, context_.frameIndex()});
}

MTL::Heap* GpuMemory::newPlacementHeap(u64 size, MemoryCategory category, const char* label) {
    MTL::HeapDescriptor* desc = MTL::HeapDescriptor::alloc()->init();
    desc->setType(MTL::HeapTypePlacement);
    desc->setStorageMode(MTL::StorageModePrivate);
    desc->setHazardTrackingMode(MTL::HazardTrackingModeUntracked);
    desc->setSize(size);
    MTL::Heap* heap = context_.device()->newHeap(desc);
    desc->release();
    if (!heap) {
        LOG_ERROR("GpuMemory: failed to create a %llu-byte placement heap '%s'",
                  static_cast<unsigned long long>(size), label);
        return nullptr;
    }
    heap->setLabel(str(label));
    context_.makeResident(heap, residencyClass(category));
    CategoryStats& s = stats_[static_cast<u32>(category)];
    s.bytes += heap->size();
    ++s.count;
    ++allocationCount_;
    PH_ALLOC(heap, heap->size(), tracyHeapPool(category));
    return heap;
}

void GpuMemory::releaseHeap(MTL::Heap* heap, MemoryCategory category) {
    if (!heap) return;
    CategoryStats& s = stats_[static_cast<u32>(category)];
    s.bytes -= heap->size();
    --s.count;
    PH_FREE(heap, tracyHeapPool(category));
    context_.deferRelease(heap, heap);
}

void GpuMemory::destroy(MTL::Resource* resource) {
    const auto it = placements_.find(resource);
    if (it != placements_.end()) {
        // The heap is resident as a whole; the range is reusable once the
        // resource object is gone.
        resource->release();
        heaps_[it->second.heap].tlsf.free(it->second.handle);
        placements_.erase(it);
        return;
    }
    if (!isMemoryless(resource)) context_.evict(resource);
    resource->release();
}

void GpuMemory::releaseCompleted(u64 completedFrame) {
    size_t kept = 0;
    for (size_t i = 0; i < pending_.size(); ++i) {
        const Pending p = pending_[i];
        if (completedFrame != ~u64{0} && p.afterFrame > completedFrame) {
            pending_[kept++] = p;
            continue;
        }
        destroy(p.resource);
    }
    pending_.resize(kept);
}

u64 GpuMemory::trimEmptyHeaps(bool keepSpare) {
    u64 freed = 0;
    std::array<bool, static_cast<size_t>(ResidencyClass::COUNT)> keptSpare{};
    for (Heap& h : heaps_) {
        if (!h.heap) continue;
        bool& spare = keptSpare[static_cast<size_t>(h.cls)];
        if (h.tlsf.stats().allocationCount != 0) continue;
        if (keepSpare && !spare) { // keep one empty heap per class for the next level
            spare = true;
            // Empty and cheap to lose: let the OS reclaim it under memory
            // pressure (F1.5); place() makes it non-volatile before reuse.
            h.heap->setPurgeableState(MTL::PurgeableStateVolatile);
            continue;
        }
        freed += h.heap->size();
        PH_FREE(h.heap, "GPU heap pages");
        context_.evict(h.heap);
        h.heap->release();
        h.heap = nullptr;
        h.tlsf = TlsfAllocator(0);
    }
    if (freed > 0) {
        LOG_INFO("GpuMemory: trimmed %llu MiB of empty heaps", static_cast<unsigned long long>(freed >> 20));
    }
    return freed;
}

u64 GpuMemory::totalBytes() const {
    u64 total = 0;
    for (const CategoryStats& s : stats_) total += s.bytes;
    return total;
}

void GpuMemory::heapStats(std::vector<HeapStats>& out) const {
    out.clear();
    for (const Heap& h : heaps_) {
        if (h.heap) out.push_back({h.heap->size(), h.cls, h.tlsf.stats()});
    }
}

} // namespace phosphor
