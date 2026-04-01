#pragma once

#include "core/types.h"
#include "rhi/vk_common.h"
#include <vector>

namespace phosphor {

class VulkanDevice;

struct AcquireResult {
    u32  imageIndex;
    bool needsRecreate;
};

class Swapchain {
public:
    Swapchain(VulkanDevice& device, VkExtent2D extent);
    ~Swapchain();

    Swapchain(const Swapchain&)            = delete;
    Swapchain& operator=(const Swapchain&) = delete;
    Swapchain(Swapchain&&)                 = delete;
    Swapchain& operator=(Swapchain&&)      = delete;

    AcquireResult acquireNextImage(VkSemaphore signalSemaphore);
    bool          present(VkQueue queue, VkSemaphore waitSemaphore, u32 imageIndex);
    void          recreate(VkExtent2D newExtent);

    VkFormat     getFormat() const;
    VkExtent2D   getExtent() const;
    u32          getImageCount() const;
    VkImageView  getImageView(u32 index) const;
    VkImage      getImage(u32 index) const;

private:
    void               create(VkExtent2D extent);
    void               cleanup();
    VkSurfaceFormatKHR chooseSurfaceFormat();
    VkPresentModeKHR   choosePresentMode();
    VkExtent2D         chooseExtent(const VkSurfaceCapabilitiesKHR& caps,
                                    VkExtent2D requested);

    VulkanDevice&            device_;
    VkSwapchainKHR           swapchain_ = VK_NULL_HANDLE;
    VkFormat                 format_    = VK_FORMAT_UNDEFINED;
    VkExtent2D               extent_{};
    std::vector<VkImage>     images_;
    std::vector<VkImageView> imageViews_;
};

} // namespace phosphor
