#pragma once

#include "core/types.h"
#include "rhi/vk_common.h"

#include <vk_mem_alloc.h>

namespace phosphor {

class VulkanDevice;

struct AllocatedBuffer {
    VkBuffer        buffer        = VK_NULL_HANDLE;
    VmaAllocation   allocation    = VK_NULL_HANDLE;
    VkDeviceAddress deviceAddress = 0;
    VkDeviceSize    size          = 0;
};

struct AllocatedImage {
    VkImage       image      = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkFormat      format     = VK_FORMAT_UNDEFINED;
    VkExtent3D    extent{};
    u32           mipLevels  = 1;
};

class GpuAllocator {
public:
    explicit GpuAllocator(VulkanDevice& device);
    ~GpuAllocator();

    GpuAllocator(const GpuAllocator&) = delete;
    GpuAllocator& operator=(const GpuAllocator&) = delete;
    GpuAllocator(GpuAllocator&&) = delete;
    GpuAllocator& operator=(GpuAllocator&&) = delete;

    AllocatedBuffer createBuffer(VkDeviceSize size,
                                 VkBufferUsageFlags usage,
                                 VmaMemoryUsage memUsage,
                                 VmaAllocationCreateFlags flags = 0);

    AllocatedImage createImage(const VkImageCreateInfo& info,
                               VmaMemoryUsage memUsage);

    VkImageView createImageView(VkImage image,
                                VkFormat format,
                                VkImageAspectFlags aspect,
                                u32 mipLevels = 1,
                                VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D,
                                u32 layerCount = 1);

    void* mapMemory(AllocatedBuffer& buffer);
    void  unmapMemory(AllocatedBuffer& buffer);

    void destroyBuffer(AllocatedBuffer& buffer);
    void destroyImage(AllocatedImage& image);
    void destroyImageView(VkImageView view);

    VmaAllocator       getAllocator() const { return allocator_; }
    VmaTotalStatistics getStats() const;

private:
    VulkanDevice& device_;
    VmaAllocator  allocator_ = VK_NULL_HANDLE;
};

} // namespace phosphor
