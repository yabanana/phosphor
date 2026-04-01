#include "rhi/vk_swapchain.h"
#include "rhi/vk_device.h"
#include <algorithm>
#include <limits>

namespace phosphor {

Swapchain::Swapchain(VulkanDevice& device, VkExtent2D extent) : device_(device) {
    create(extent);
}

Swapchain::~Swapchain() {
    cleanup();
}

// ---------- public API --------------------------------------------------

AcquireResult Swapchain::acquireNextImage(VkSemaphore signalSemaphore) {
    u32 imageIndex = 0;
    VkResult result = vkAcquireNextImageKHR(
        device_.getDevice(), swapchain_, UINT64_MAX,
        signalSemaphore, VK_NULL_HANDLE, &imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        return {0, true};
    }

    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        LOG_ERROR("Swapchain: vkAcquireNextImageKHR failed (%d)", static_cast<int>(result));
        std::abort();
    }

    return {imageIndex, false};
}

bool Swapchain::present(VkQueue queue, VkSemaphore waitSemaphore, u32 imageIndex) {
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores    = &waitSemaphore;
    presentInfo.swapchainCount     = 1;
    presentInfo.pSwapchains        = &swapchain_;
    presentInfo.pImageIndices      = &imageIndex;

    VkResult result = vkQueuePresentKHR(queue, &presentInfo);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
        return false; // caller should recreate
    }

    if (result != VK_SUCCESS) {
        LOG_ERROR("Swapchain: vkQueuePresentKHR failed (%d)", static_cast<int>(result));
        std::abort();
    }

    return true;
}

void Swapchain::recreate(VkExtent2D newExtent) {
    device_.waitIdle();
    cleanup();
    create(newExtent);
    LOG_INFO("Swapchain: recreated (%ux%u)", newExtent.width, newExtent.height);
}

VkFormat    Swapchain::getFormat()     const { return format_; }
VkExtent2D  Swapchain::getExtent()     const { return extent_; }
u32         Swapchain::getImageCount() const { return static_cast<u32>(images_.size()); }

VkImageView Swapchain::getImageView(u32 index) const { return imageViews_[index]; }
VkImage     Swapchain::getImage(u32 index)     const { return images_[index]; }

// ---------- internals ---------------------------------------------------

void Swapchain::create(VkExtent2D extent) {
    VkPhysicalDevice physDev = device_.getPhysicalDevice();
    VkSurfaceKHR     surface = device_.getSurface();

    VkSurfaceCapabilitiesKHR caps{};
    VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physDev, surface, &caps));

    VkSurfaceFormatKHR surfaceFormat = chooseSurfaceFormat();
    VkPresentModeKHR   presentMode   = choosePresentMode();
    VkExtent2D         chosenExtent  = chooseExtent(caps, extent);

    // Request one more than the minimum so the driver has room to work.
    u32 imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount) {
        imageCount = caps.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface          = surface;
    createInfo.minImageCount    = imageCount;
    createInfo.imageFormat      = surfaceFormat.format;
    createInfo.imageColorSpace  = surfaceFormat.colorSpace;
    createInfo.imageExtent      = chosenExtent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                                | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    createInfo.preTransform     = caps.currentTransform;
    createInfo.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode      = presentMode;
    createInfo.clipped          = VK_TRUE;
    createInfo.oldSwapchain     = VK_NULL_HANDLE;

    const auto& families = device_.getQueueFamilyIndices();
    u32 queueFamilyIndices[] = {families.graphics.value(),
                                families.present.value()};

    if (families.graphics.value() != families.present.value()) {
        createInfo.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices   = queueFamilyIndices;
    } else {
        createInfo.imageSharingMode      = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.queueFamilyIndexCount = 0;
        createInfo.pQueueFamilyIndices   = nullptr;
    }

    VK_CHECK(vkCreateSwapchainKHR(device_.getDevice(), &createInfo, nullptr,
                                  &swapchain_));

    format_ = surfaceFormat.format;
    extent_ = chosenExtent;

    // Retrieve swapchain images.
    u32 count = 0;
    VK_CHECK(vkGetSwapchainImagesKHR(device_.getDevice(), swapchain_, &count, nullptr));
    images_.resize(count);
    VK_CHECK(vkGetSwapchainImagesKHR(device_.getDevice(), swapchain_, &count,
                                     images_.data()));

    // Create image views.
    imageViews_.resize(count);
    for (u32 i = 0; i < count; ++i) {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image                           = images_[i];
        viewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format                          = format_;
        viewInfo.components.r                    = VK_COMPONENT_SWIZZLE_IDENTITY;
        viewInfo.components.g                    = VK_COMPONENT_SWIZZLE_IDENTITY;
        viewInfo.components.b                    = VK_COMPONENT_SWIZZLE_IDENTITY;
        viewInfo.components.a                    = VK_COMPONENT_SWIZZLE_IDENTITY;
        viewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel   = 0;
        viewInfo.subresourceRange.levelCount     = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount     = 1;

        VK_CHECK(vkCreateImageView(device_.getDevice(), &viewInfo, nullptr,
                                   &imageViews_[i]));
    }

    LOG_INFO("Swapchain: created %ux%u, %u images, format %d, present mode %d",
             extent_.width, extent_.height, count,
             static_cast<int>(format_), static_cast<int>(presentMode));
}

void Swapchain::cleanup() {
    VkDevice dev = device_.getDevice();

    for (auto view : imageViews_) {
        if (view != VK_NULL_HANDLE) {
            vkDestroyImageView(dev, view, nullptr);
        }
    }
    imageViews_.clear();
    images_.clear();

    if (swapchain_ != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(dev, swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
    }
}

VkSurfaceFormatKHR Swapchain::chooseSurfaceFormat() {
    VkPhysicalDevice physDev = device_.getPhysicalDevice();
    VkSurfaceKHR     surface = device_.getSurface();

    u32 count = 0;
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physDev, surface, &count, nullptr));
    std::vector<VkSurfaceFormatKHR> formats(count);
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physDev, surface, &count,
                                                  formats.data()));

    for (const auto& fmt : formats) {
        if (fmt.format == VK_FORMAT_B8G8R8A8_SRGB &&
            fmt.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return fmt;
        }
    }

    // Fall back to whatever the driver offers first.
    return formats[0];
}

VkPresentModeKHR Swapchain::choosePresentMode() {
    VkPhysicalDevice physDev = device_.getPhysicalDevice();
    VkSurfaceKHR     surface = device_.getSurface();

    u32 count = 0;
    VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(physDev, surface, &count,
                                                       nullptr));
    std::vector<VkPresentModeKHR> modes(count);
    VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(physDev, surface, &count,
                                                       modes.data()));

    for (auto mode : modes) {
        if (mode == VK_PRESENT_MODE_MAILBOX_KHR) {
            return mode;
        }
    }

    // FIFO is guaranteed by the spec.
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D Swapchain::chooseExtent(const VkSurfaceCapabilitiesKHR& caps,
                                   VkExtent2D requested) {
    // If currentExtent is the special value 0xFFFFFFFF, the surface size
    // is determined by the swapchain extent; otherwise use it directly.
    if (caps.currentExtent.width != std::numeric_limits<u32>::max()) {
        return caps.currentExtent;
    }

    VkExtent2D actual{};
    actual.width  = std::clamp(requested.width,
                               caps.minImageExtent.width,
                               caps.maxImageExtent.width);
    actual.height = std::clamp(requested.height,
                               caps.minImageExtent.height,
                               caps.maxImageExtent.height);
    return actual;
}

} // namespace phosphor
