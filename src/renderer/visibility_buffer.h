#pragma once

#include "core/types.h"
#include "rhi/vk_common.h"
#include "rhi/vk_allocator.h"

namespace phosphor {

class VulkanDevice;

// ---------------------------------------------------------------------------
// VisibilityBuffer — manages the R32_UINT visibility image and the
// D32_SFLOAT depth image used by the mesh shading pass.
// ---------------------------------------------------------------------------

class VisibilityBuffer {
public:
    VisibilityBuffer(VulkanDevice& device, GpuAllocator& allocator, VkExtent2D extent);
    ~VisibilityBuffer();

    VisibilityBuffer(const VisibilityBuffer&) = delete;
    VisibilityBuffer& operator=(const VisibilityBuffer&) = delete;

    void recreate(VkExtent2D newExtent);

    [[nodiscard]] VkImage     getVisImage()   const { return visBuffer_.image; }
    [[nodiscard]] VkImageView getVisView()    const { return visView_; }
    [[nodiscard]] VkImage     getDepthImage() const { return depthBuffer_.image; }
    [[nodiscard]] VkImageView getDepthView()  const { return depthView_; }
    [[nodiscard]] VkExtent2D  getExtent()     const { return extent_; }

private:
    void create();
    void cleanup();

    VulkanDevice&  device_;
    GpuAllocator&  allocator_;
    VkExtent2D     extent_;
    AllocatedImage visBuffer_{};    // R32_UINT
    VkImageView    visView_   = VK_NULL_HANDLE;
    AllocatedImage depthBuffer_{}; // D32_SFLOAT
    VkImageView    depthView_ = VK_NULL_HANDLE;
};

} // namespace phosphor
