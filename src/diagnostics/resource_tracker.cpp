#include "diagnostics/resource_tracker.h"
#include "rhi/vk_allocator.h"
#include <cstring>

namespace phosphor {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

ResourceTracker::ResourceTracker(GpuAllocator& allocator)
    : allocator_(allocator)
{
    std::memset(&stats_, 0, sizeof(stats_));
}

// ---------------------------------------------------------------------------
// update -- refresh VMA statistics
// ---------------------------------------------------------------------------

void ResourceTracker::update() {
    stats_ = allocator_.getStats();
}

// ---------------------------------------------------------------------------
// VMA-based queries
// ---------------------------------------------------------------------------

u32 ResourceTracker::getTotalAllocations() const {
    return static_cast<u32>(stats_.total.statistics.blockCount);
}

VkDeviceSize ResourceTracker::getTotalUsedBytes() const {
    return stats_.total.statistics.allocationBytes;
}

VkDeviceSize ResourceTracker::getTotalReservedBytes() const {
    return stats_.total.statistics.blockBytes;
}

} // namespace phosphor
