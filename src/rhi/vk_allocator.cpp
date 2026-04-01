// VMA implementation — this define must appear in exactly one translation unit
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include "rhi/vk_allocator.h"
#include "rhi/vk_device.h"
#include "core/log.h"

namespace phosphor {

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------
GpuAllocator::GpuAllocator(VulkanDevice& device) : device_(device) {
    VmaAllocatorCreateInfo allocatorInfo{};
    allocatorInfo.flags          = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    allocatorInfo.physicalDevice = device_.getPhysicalDevice();
    allocatorInfo.device         = device_.getDevice();
    allocatorInfo.instance       = device_.getInstance();
    allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_3;

    VK_CHECK(vmaCreateAllocator(&allocatorInfo, &allocator_));
    LOG_INFO("VMA allocator created");
}

GpuAllocator::~GpuAllocator() {
    if (allocator_ != VK_NULL_HANDLE) {
        vmaDestroyAllocator(allocator_);
        allocator_ = VK_NULL_HANDLE;
    }
}

// ---------------------------------------------------------------------------
// Buffer creation
// ---------------------------------------------------------------------------
AllocatedBuffer GpuAllocator::createBuffer(VkDeviceSize size,
                                            VkBufferUsageFlags usage,
                                            VmaMemoryUsage memUsage,
                                            VmaAllocationCreateFlags flags) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size  = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = memUsage;
    allocInfo.flags = flags;

    AllocatedBuffer result{};
    result.size = size;

    VK_CHECK(vmaCreateBuffer(allocator_, &bufferInfo, &allocInfo,
                              &result.buffer, &result.allocation, nullptr));

    // Retrieve device address if the buffer was created with that usage flag
    if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
        VkBufferDeviceAddressInfo addrInfo{};
        addrInfo.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        addrInfo.buffer = result.buffer;
        result.deviceAddress =
            vkGetBufferDeviceAddress(device_.getDevice(), &addrInfo);
    }

    return result;
}

// ---------------------------------------------------------------------------
// Image creation
// ---------------------------------------------------------------------------
AllocatedImage GpuAllocator::createImage(const VkImageCreateInfo& info,
                                          VmaMemoryUsage memUsage) {
    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = memUsage;

    AllocatedImage result{};
    result.format    = info.format;
    result.extent    = info.extent;
    result.mipLevels = info.mipLevels;

    VK_CHECK(vmaCreateImage(allocator_, &info, &allocInfo,
                             &result.image, &result.allocation, nullptr));

    return result;
}

// ---------------------------------------------------------------------------
// Image view creation
// ---------------------------------------------------------------------------
VkImageView GpuAllocator::createImageView(VkImage image,
                                           VkFormat format,
                                           VkImageAspectFlags aspect,
                                           u32 mipLevels,
                                           VkImageViewType viewType,
                                           u32 layerCount) {
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image    = image;
    viewInfo.viewType = viewType;
    viewInfo.format   = format;

    viewInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    viewInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    viewInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    viewInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

    viewInfo.subresourceRange.aspectMask     = aspect;
    viewInfo.subresourceRange.baseMipLevel   = 0;
    viewInfo.subresourceRange.levelCount     = mipLevels;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount     = layerCount;

    VkImageView view = VK_NULL_HANDLE;
    VK_CHECK(vkCreateImageView(device_.getDevice(), &viewInfo, nullptr, &view));
    return view;
}

// ---------------------------------------------------------------------------
// Memory mapping
// ---------------------------------------------------------------------------
void* GpuAllocator::mapMemory(AllocatedBuffer& buffer) {
    void* data = nullptr;
    VK_CHECK(vmaMapMemory(allocator_, buffer.allocation, &data));
    return data;
}

void GpuAllocator::unmapMemory(AllocatedBuffer& buffer) {
    vmaUnmapMemory(allocator_, buffer.allocation);
}

// ---------------------------------------------------------------------------
// Resource destruction
// ---------------------------------------------------------------------------
void GpuAllocator::destroyBuffer(AllocatedBuffer& buffer) {
    if (buffer.buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator_, buffer.buffer, buffer.allocation);
        buffer.buffer        = VK_NULL_HANDLE;
        buffer.allocation    = VK_NULL_HANDLE;
        buffer.deviceAddress = 0;
        buffer.size          = 0;
    }
}

void GpuAllocator::destroyImage(AllocatedImage& image) {
    if (image.image != VK_NULL_HANDLE) {
        vmaDestroyImage(allocator_, image.image, image.allocation);
        image.image      = VK_NULL_HANDLE;
        image.allocation = VK_NULL_HANDLE;
    }
}

void GpuAllocator::destroyImageView(VkImageView view) {
    if (view != VK_NULL_HANDLE) {
        vkDestroyImageView(device_.getDevice(), view, nullptr);
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------
VmaTotalStatistics GpuAllocator::getStats() const {
    VmaTotalStatistics stats{};
    vmaCalculateStatistics(allocator_, &stats);
    return stats;
}

} // namespace phosphor
