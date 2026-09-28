#pragma once

#include "core/types.h"
#include <string>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

namespace phosphor {

class GpuAllocator;

// ---------------------------------------------------------------------------
// TrackedResourceType / TrackedResource -- per-resource bookkeeping
// ---------------------------------------------------------------------------

enum class TrackedResourceType : u32 {
    Buffer = 0,
    Image,
    ImageView,
    Pipeline,
    DescriptorSet,
    Sampler,
    AccelerationStructure,
    COUNT
};

struct TrackedResource {
    TrackedResourceType type;
    u64                 handle;
    u64                 sizeBytes;
    std::string         name;
};

// ---------------------------------------------------------------------------
// ResourceTracker -- tracks live GPU resource allocations and VMA statistics
// for leak detection and the diagnostic UI panel.
// ---------------------------------------------------------------------------

class ResourceTracker {
public:
    explicit ResourceTracker(GpuAllocator& allocator);

    void track(TrackedResourceType type, u64 handle, u64 sizeBytes, const char* name = "") {
        TrackedResource r{type, handle, sizeBytes, name ? name : ""};
        resources_[handle] = r;
        totalAllocated_ += sizeBytes;
        typeCounts_[static_cast<u32>(type)]++;
    }

    void untrack(u64 handle) {
        auto it = resources_.find(handle);
        if (it != resources_.end()) {
            totalAllocated_ -= it->second.sizeBytes;
            typeCounts_[static_cast<u32>(it->second.type)]--;
            resources_.erase(it);
        }
    }

    /// Refresh VMA statistics from the allocator.  Call once per frame.
    void update();

    [[nodiscard]] u32 getTotalAllocations() const;
    [[nodiscard]] VkDeviceSize getTotalUsedBytes() const;
    [[nodiscard]] VkDeviceSize getTotalReservedBytes() const;

    [[nodiscard]] u64 getTotalAllocatedBytes() const { return totalAllocated_; }
    [[nodiscard]] u32 getResourceCount() const { return static_cast<u32>(resources_.size()); }

    [[nodiscard]] u32 getCountByType(TrackedResourceType type) const {
        return typeCounts_[static_cast<u32>(type)];
    }

    [[nodiscard]] float getTotalAllocatedMB() const {
        return static_cast<float>(totalAllocated_) / (1024.0f * 1024.0f);
    }

    [[nodiscard]] float getTotalUsedMB() const {
        return static_cast<float>(getTotalUsedBytes()) / (1024.0f * 1024.0f);
    }

    [[nodiscard]] float getTotalReservedMB() const {
        return static_cast<float>(getTotalReservedBytes()) / (1024.0f * 1024.0f);
    }

    [[nodiscard]] const std::unordered_map<u64, TrackedResource>& getAllResources() const {
        return resources_;
    }

    static const char* typeName(TrackedResourceType type) {
        switch (type) {
            case TrackedResourceType::Buffer:                return "Buffer";
            case TrackedResourceType::Image:                 return "Image";
            case TrackedResourceType::ImageView:             return "ImageView";
            case TrackedResourceType::Pipeline:              return "Pipeline";
            case TrackedResourceType::DescriptorSet:         return "DescriptorSet";
            case TrackedResourceType::Sampler:               return "Sampler";
            case TrackedResourceType::AccelerationStructure: return "AccelStruct";
            default:                                         return "Unknown";
        }
    }

private:
    GpuAllocator& allocator_;
    VmaTotalStatistics stats_{};

    std::unordered_map<u64, TrackedResource> resources_;
    u64 totalAllocated_ = 0;
    u32 typeCounts_[static_cast<u32>(TrackedResourceType::COUNT)]{};
};

} // namespace phosphor
