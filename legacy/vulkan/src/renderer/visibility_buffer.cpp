#include "renderer/visibility_buffer.h"
#include "rhi/vk_device.h"
#include "core/log.h"

namespace phosphor {

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------

VisibilityBuffer::VisibilityBuffer(VulkanDevice& device, GpuAllocator& allocator,
                                   VkExtent2D extent)
    : device_(device), allocator_(allocator), extent_(extent) {
    create();
    LOG_INFO("VisibilityBuffer created (%ux%u)", extent_.width, extent_.height);
}

VisibilityBuffer::~VisibilityBuffer() {
    cleanup();
}

// ---------------------------------------------------------------------------
// Recreate on resize
// ---------------------------------------------------------------------------

void VisibilityBuffer::recreate(VkExtent2D newExtent) {
    if (newExtent.width == extent_.width && newExtent.height == extent_.height) {
        return;
    }
    device_.waitIdle();
    cleanup();
    extent_ = newExtent;
    create();
    LOG_INFO("VisibilityBuffer recreated (%ux%u)", extent_.width, extent_.height);
}

// ---------------------------------------------------------------------------
// Internal: create images + views
// ---------------------------------------------------------------------------

void VisibilityBuffer::create() {
    // --- Visibility image: R32_UINT ---
    {
        VkImageCreateInfo imgInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        imgInfo.imageType   = VK_IMAGE_TYPE_2D;
        imgInfo.format      = VK_FORMAT_R32_UINT;
        imgInfo.extent      = {extent_.width, extent_.height, 1};
        imgInfo.mipLevels   = 1;
        imgInfo.arrayLayers = 1;
        imgInfo.samples     = VK_SAMPLE_COUNT_1_BIT;
        imgInfo.tiling      = VK_IMAGE_TILING_OPTIMAL;
        imgInfo.usage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                            | VK_IMAGE_USAGE_STORAGE_BIT
                            | VK_IMAGE_USAGE_SAMPLED_BIT;
        imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        visBuffer_ = allocator_.createImage(imgInfo, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        visView_   = allocator_.createImageView(
            visBuffer_.image, VK_FORMAT_R32_UINT, VK_IMAGE_ASPECT_COLOR_BIT);
    }

    // --- Depth image: D32_SFLOAT ---
    {
        VkImageCreateInfo imgInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        imgInfo.imageType   = VK_IMAGE_TYPE_2D;
        imgInfo.format      = VK_FORMAT_D32_SFLOAT;
        imgInfo.extent      = {extent_.width, extent_.height, 1};
        imgInfo.mipLevels   = 1;
        imgInfo.arrayLayers = 1;
        imgInfo.samples     = VK_SAMPLE_COUNT_1_BIT;
        imgInfo.tiling      = VK_IMAGE_TILING_OPTIMAL;
        imgInfo.usage       = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                            | VK_IMAGE_USAGE_SAMPLED_BIT;
        imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        depthBuffer_ = allocator_.createImage(imgInfo, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        depthView_   = allocator_.createImageView(
            depthBuffer_.image, VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT);
    }
}

// ---------------------------------------------------------------------------
// Internal: destroy images + views
// ---------------------------------------------------------------------------

void VisibilityBuffer::cleanup() {
    allocator_.destroyImageView(visView_);
    visView_ = VK_NULL_HANDLE;
    allocator_.destroyImage(visBuffer_);

    allocator_.destroyImageView(depthView_);
    depthView_ = VK_NULL_HANDLE;
    allocator_.destroyImage(depthBuffer_);
}

} // namespace phosphor
