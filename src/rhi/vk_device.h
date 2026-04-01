#pragma once

#include "core/types.h"
#include "rhi/vk_common.h"
#include <optional>
#include <vector>

namespace phosphor {

class Window;

struct QueueFamilyIndices {
    std::optional<u32> graphics;
    std::optional<u32> compute;   // prefer dedicated (no graphics bit)
    std::optional<u32> transfer;  // prefer dedicated (no graphics/compute bit)
    std::optional<u32> present;

    bool isComplete() const {
        return graphics.has_value() && compute.has_value() &&
               transfer.has_value() && present.has_value();
    }
};

struct Queues {
    VkQueue graphics = VK_NULL_HANDLE;
    VkQueue compute  = VK_NULL_HANDLE;
    VkQueue transfer = VK_NULL_HANDLE;
    VkQueue present  = VK_NULL_HANDLE;
};

class VulkanDevice {
public:
    VulkanDevice(Window& window, bool enableValidation);
    ~VulkanDevice();

    // Non-copyable, non-movable (singleton-like)
    VulkanDevice(const VulkanDevice&) = delete;
    VulkanDevice& operator=(const VulkanDevice&) = delete;
    VulkanDevice(VulkanDevice&&) = delete;
    VulkanDevice& operator=(VulkanDevice&&) = delete;

    VkInstance                                      getInstance() const;
    VkPhysicalDevice                                getPhysicalDevice() const;
    VkDevice                                        getDevice() const;
    VkSurfaceKHR                                    getSurface() const;
    const Queues&                                   getQueues() const;
    const QueueFamilyIndices&                       getQueueFamilyIndices() const;
    const VkPhysicalDeviceProperties&               getProperties() const;
    const VkPhysicalDeviceMeshShaderPropertiesEXT&  getMeshShaderProperties() const;

    u32  findMemoryType(u32 typeFilter, VkMemoryPropertyFlags properties) const;
    void waitIdle() const;

private:
    void createInstance(bool enableValidation);
    void createSurface(Window& window);
    void pickPhysicalDevice();
    void findQueueFamilies();
    void createLogicalDevice(bool enableValidation);
    void setupDebugMessenger();

    u32  scorePhysicalDevice(VkPhysicalDevice device) const;
    bool checkDeviceExtensions(VkPhysicalDevice device) const;

    VkInstance                              instance_        = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT               debugMessenger_  = VK_NULL_HANDLE;
    VkPhysicalDevice                       physicalDevice_  = VK_NULL_HANDLE;
    VkDevice                               device_          = VK_NULL_HANDLE;
    VkSurfaceKHR                           surface_         = VK_NULL_HANDLE;

    QueueFamilyIndices                     queueFamilies_;
    Queues                                 queues_;

    VkPhysicalDeviceProperties             properties_{};
    VkPhysicalDeviceMeshShaderPropertiesEXT meshShaderProps_{};
    VkPhysicalDeviceMemoryProperties       memProperties_{};

    bool                                   validationEnabled_ = false;
};

} // namespace phosphor
