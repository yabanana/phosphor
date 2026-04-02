#include "rhi/vk_device.h"
#include "core/log.h"
#include "core/window.h"

#include <SDL3/SDL_vulkan.h>
#include <algorithm>
#include <cstring>
#include <set>
#include <vector>

namespace phosphor {

// ---------------------------------------------------------------------------
// Validation layer debug callback
// ---------------------------------------------------------------------------
static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT      severity,
    VkDebugUtilsMessageTypeFlagsEXT             /*type*/,
    const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
    void*                                       /*userData*/)
{
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        LOG_ERROR("[Vulkan Validation] %s", callbackData->pMessage);
    } else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        LOG_WARN("[Vulkan Validation] %s", callbackData->pMessage);
    } else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
        LOG_INFO("[Vulkan Validation] %s", callbackData->pMessage);
    } else {
        LOG_DEBUG("[Vulkan Validation] %s", callbackData->pMessage);
    }
    return VK_FALSE;
}

// ---------------------------------------------------------------------------
// Required device extensions
// ---------------------------------------------------------------------------
static const std::vector<const char*> kRequiredDeviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_EXT_MESH_SHADER_EXTENSION_NAME,
    VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
    VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
    VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
    VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
    VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
    VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
};

static const char* kValidationLayerName = "VK_LAYER_KHRONOS_validation";

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------
VulkanDevice::VulkanDevice(Window& window, bool enableValidation)
    : validationEnabled_(enableValidation)
{
    createInstance(enableValidation);
    createSurface(window);
    pickPhysicalDevice();
    findQueueFamilies();
    createLogicalDevice(enableValidation);

    if (validationEnabled_) {
        setupDebugMessenger();
    }

    LOG_INFO("VulkanDevice initialised  [GPU: %s]", properties_.deviceName);
}

VulkanDevice::~VulkanDevice() {
    if (device_ != VK_NULL_HANDLE) {
        vkDestroyDevice(device_, nullptr);
    }

    if (validationEnabled_ && debugMessenger_ != VK_NULL_HANDLE) {
        auto destroyFn = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
        if (destroyFn) {
            destroyFn(instance_, debugMessenger_, nullptr);
        }
    }

    if (surface_ != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
    }

    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
    }
}

// ---------------------------------------------------------------------------
// Instance creation
// ---------------------------------------------------------------------------
void VulkanDevice::createInstance(bool enableValidation) {
    VkApplicationInfo appInfo{};
    appInfo.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName   = "Phosphor";
    appInfo.applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
    appInfo.pEngineName        = "Phosphor Engine";
    appInfo.engineVersion      = VK_MAKE_API_VERSION(0, 1, 0, 0);
    appInfo.apiVersion         = VK_API_VERSION_1_3;

    // Gather extensions required by SDL surface + optional debug utils
    u32 sdlExtCount = 0;
    const char* const* sdlExts = SDL_Vulkan_GetInstanceExtensions(&sdlExtCount);
    std::vector<const char*> extensions(sdlExts, sdlExts + sdlExtCount);

    if (enableValidation) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    VkInstanceCreateInfo createInfo{};
    createInfo.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo        = &appInfo;
    createInfo.enabledExtensionCount   = static_cast<u32>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    // Validation layers
    std::vector<const char*> layers;
    if (enableValidation) {
        // Verify the validation layer is available
        u32 layerCount = 0;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        bool found = false;
        for (const auto& layer : availableLayers) {
            if (std::strcmp(layer.layerName, kValidationLayerName) == 0) {
                found = true;
                break;
            }
        }

        if (found) {
            layers.push_back(kValidationLayerName);
            LOG_INFO("Vulkan validation layer enabled");
        } else {
            LOG_WARN("Validation layer requested but %s not available",
                     kValidationLayerName);
        }
    }

    createInfo.enabledLayerCount   = static_cast<u32>(layers.size());
    createInfo.ppEnabledLayerNames = layers.data();

    VK_CHECK(vkCreateInstance(&createInfo, nullptr, &instance_));
    LOG_INFO("VkInstance created (API 1.3)");
}

// ---------------------------------------------------------------------------
// Surface creation via SDL
// ---------------------------------------------------------------------------
void VulkanDevice::createSurface(Window& window) {
    surface_ = window.createVulkanSurface(instance_);
    if (surface_ == VK_NULL_HANDLE) {
        LOG_ERROR("Failed to create Vulkan surface");
        std::abort();
    }
}

// ---------------------------------------------------------------------------
// Physical device selection
// ---------------------------------------------------------------------------
void VulkanDevice::pickPhysicalDevice() {
    u32 deviceCount = 0;
    vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr);
    if (deviceCount == 0) {
        LOG_ERROR("No GPUs with Vulkan support found");
        std::abort();
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data());

    u32 bestScore = 0;
    VkPhysicalDevice bestDevice = VK_NULL_HANDLE;

    for (auto dev : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(dev, &props);

        if (!checkDeviceExtensions(dev)) {
            LOG_DEBUG("GPU '%s' skipped: missing required extensions",
                      props.deviceName);
            continue;
        }

        // Quick queue family completeness check
        u32 qfCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qfCount, nullptr);
        std::vector<VkQueueFamilyProperties> qfProps(qfCount);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qfCount, qfProps.data());

        bool hasGraphics = false;
        bool hasPresent  = false;
        for (u32 i = 0; i < qfCount; ++i) {
            if (qfProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                hasGraphics = true;
            }
            VkBool32 presentSupport = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, surface_,
                                                  &presentSupport);
            if (presentSupport) {
                hasPresent = true;
            }
        }

        if (!hasGraphics || !hasPresent) {
            LOG_DEBUG("GPU '%s' skipped: incomplete queue families",
                      props.deviceName);
            continue;
        }

        u32 score = scorePhysicalDevice(dev);
        LOG_INFO("GPU candidate: '%s'  score=%u", props.deviceName, score);

        if (score > bestScore) {
            bestScore  = score;
            bestDevice = dev;
        }
    }

    if (bestDevice == VK_NULL_HANDLE) {
        LOG_ERROR("No suitable GPU found");
        std::abort();
    }

    physicalDevice_ = bestDevice;
    vkGetPhysicalDeviceProperties(physicalDevice_, &properties_);
    vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memProperties_);

    LOG_INFO("Selected GPU: '%s'", properties_.deviceName);
}

u32 VulkanDevice::scorePhysicalDevice(VkPhysicalDevice device) const {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(device, &props);

    u32 score = 0;

    switch (props.deviceType) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   score += 1000; break;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: score += 500;  break;
    default: break;
    }

    score += props.limits.maxImageDimension2D / 1000;

    return score;
}

bool VulkanDevice::checkDeviceExtensions(VkPhysicalDevice device) const {
    u32 extCount = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> available(extCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extCount,
                                          available.data());

    for (const char* required : kRequiredDeviceExtensions) {
        bool found = false;
        for (const auto& ext : available) {
            if (std::strcmp(ext.extensionName, required) == 0) {
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Queue family discovery
// ---------------------------------------------------------------------------
void VulkanDevice::findQueueFamilies() {
    u32 qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &qfCount,
                                              nullptr);
    std::vector<VkQueueFamilyProperties> families(qfCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &qfCount,
                                              families.data());

    for (u32 i = 0; i < qfCount; ++i) {
        const auto& qf = families[i];

        // Graphics queue
        if ((qf.queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
            !queueFamilies_.graphics.has_value()) {
            queueFamilies_.graphics = i;
        }

        // Prefer dedicated compute queue (no graphics bit)
        if ((qf.queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            !(qf.queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            queueFamilies_.compute = i;
        }

        // Prefer dedicated transfer queue (no graphics or compute bit)
        if ((qf.queueFlags & VK_QUEUE_TRANSFER_BIT) &&
            !(qf.queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
            !(qf.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
            queueFamilies_.transfer = i;
        }

        // Present support
        VkBool32 presentSupport = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice_, i, surface_,
                                              &presentSupport);
        if (presentSupport && !queueFamilies_.present.has_value()) {
            queueFamilies_.present = i;
        }
    }

    // Fallback: use graphics family for compute if no dedicated one found
    if (!queueFamilies_.compute.has_value()) {
        for (u32 i = 0; i < qfCount; ++i) {
            if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                queueFamilies_.compute = i;
                break;
            }
        }
    }

    // Fallback: use any family with transfer for transfer if no dedicated
    if (!queueFamilies_.transfer.has_value()) {
        for (u32 i = 0; i < qfCount; ++i) {
            if (families[i].queueFlags & VK_QUEUE_TRANSFER_BIT) {
                queueFamilies_.transfer = i;
                break;
            }
        }
    }

    LOG_INFO("Queue families  graphics=%u  compute=%u  transfer=%u  present=%u",
             queueFamilies_.graphics.value_or(~0u),
             queueFamilies_.compute.value_or(~0u),
             queueFamilies_.transfer.value_or(~0u),
             queueFamilies_.present.value_or(~0u));
}

// ---------------------------------------------------------------------------
// Logical device creation with feature pNext chain
// ---------------------------------------------------------------------------
void VulkanDevice::createLogicalDevice(bool enableValidation) {
    // Collect unique queue family indices
    std::set<u32> uniqueQueueFamilies = {
        queueFamilies_.graphics.value(),
        queueFamilies_.compute.value(),
        queueFamilies_.transfer.value(),
        queueFamilies_.present.value(),
    };

    float queuePriority = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    queueCreateInfos.reserve(uniqueQueueFamilies.size());

    for (u32 family : uniqueQueueFamilies) {
        VkDeviceQueueCreateInfo qci{};
        qci.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = family;
        qci.queueCount       = 1;
        qci.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(qci);
    }

    // Build feature pNext chain.
    // Vulkan 1.2/1.3 promoted features MUST use VkPhysicalDeviceVulkan1XFeatures
    // and NOT the old per-extension structs (they conflict in the pNext chain).

    // Vulkan 1.2 consolidated features (replaces DescriptorIndexing, BDA, Timeline, 8bit, etc.)
    VkPhysicalDeviceVulkan12Features vulkan12Features{};
    vulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    // Descriptor indexing (bindless)
    vulkan12Features.descriptorIndexing                                 = VK_TRUE;
    vulkan12Features.shaderSampledImageArrayNonUniformIndexing          = VK_TRUE;
    vulkan12Features.shaderStorageBufferArrayNonUniformIndexing         = VK_TRUE;
    vulkan12Features.shaderStorageImageArrayNonUniformIndexing          = VK_TRUE;
    vulkan12Features.descriptorBindingSampledImageUpdateAfterBind       = VK_TRUE;
    vulkan12Features.descriptorBindingStorageBufferUpdateAfterBind      = VK_TRUE;
    vulkan12Features.descriptorBindingStorageImageUpdateAfterBind       = VK_TRUE;
    vulkan12Features.descriptorBindingPartiallyBound                    = VK_TRUE;
    vulkan12Features.descriptorBindingVariableDescriptorCount           = VK_TRUE;
    vulkan12Features.runtimeDescriptorArray                             = VK_TRUE;
    // Buffer device address
    vulkan12Features.bufferDeviceAddress                                = VK_TRUE;
    // Timeline semaphores
    vulkan12Features.timelineSemaphore                                  = VK_TRUE;
    // Host query reset
    vulkan12Features.hostQueryReset                                     = VK_TRUE;
    // 8-bit storage + scalar block layout
    vulkan12Features.storageBuffer8BitAccess                            = VK_TRUE;
    vulkan12Features.shaderInt8                                         = VK_TRUE;
    vulkan12Features.scalarBlockLayout                                  = VK_TRUE;

    // Vulkan 1.3 consolidated features (replaces Sync2, DynamicRendering, Maintenance4)
    VkPhysicalDeviceVulkan13Features vulkan13Features{};
    vulkan13Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    vulkan13Features.synchronization2  = VK_TRUE;
    vulkan13Features.dynamicRendering  = VK_TRUE;
    vulkan13Features.maintenance4      = VK_TRUE;
    vulkan13Features.pNext = &vulkan12Features;

    // Extension features (NOT promoted — keep individual structs)

    // Acceleration structure
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeatures{};
    accelFeatures.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    accelFeatures.accelerationStructure = VK_TRUE;
    accelFeatures.pNext = &vulkan13Features;

    // Ray tracing pipeline
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtFeatures{};
    rtFeatures.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    rtFeatures.rayTracingPipeline = VK_TRUE;
    rtFeatures.pNext = &accelFeatures;

    // Mesh shader
    VkPhysicalDeviceMeshShaderFeaturesEXT meshShaderFeatures{};
    meshShaderFeatures.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT;
    meshShaderFeatures.taskShader = VK_TRUE;
    meshShaderFeatures.meshShader = VK_TRUE;
    meshShaderFeatures.pNext = &rtFeatures;

    // Base physical device features
    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &meshShaderFeatures;
    features2.features.samplerAnisotropy   = VK_TRUE;
    features2.features.fillModeNonSolid    = VK_TRUE;
    features2.features.wideLines           = VK_TRUE;
    features2.features.multiDrawIndirect   = VK_TRUE;
    features2.features.shaderInt64         = VK_TRUE;
    features2.features.geometryShader      = VK_TRUE;

    // Create the device
    VkDeviceCreateInfo createInfo{};
    createInfo.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pNext                   = &features2;
    createInfo.queueCreateInfoCount    = static_cast<u32>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos       = queueCreateInfos.data();
    createInfo.enabledExtensionCount   = static_cast<u32>(kRequiredDeviceExtensions.size());
    createInfo.ppEnabledExtensionNames = kRequiredDeviceExtensions.data();

    // Validation layers on device (deprecated but some loaders still use it)
    std::vector<const char*> layers;
    if (enableValidation) {
        layers.push_back(kValidationLayerName);
    }
    createInfo.enabledLayerCount   = static_cast<u32>(layers.size());
    createInfo.ppEnabledLayerNames = layers.data();

    VK_CHECK(vkCreateDevice(physicalDevice_, &createInfo, nullptr, &device_));

    // Retrieve queues
    vkGetDeviceQueue(device_, queueFamilies_.graphics.value(), 0,
                     &queues_.graphics);
    vkGetDeviceQueue(device_, queueFamilies_.compute.value(), 0,
                     &queues_.compute);
    vkGetDeviceQueue(device_, queueFamilies_.transfer.value(), 0,
                     &queues_.transfer);
    vkGetDeviceQueue(device_, queueFamilies_.present.value(), 0,
                     &queues_.present);

    // Query mesh shader properties
    meshShaderProps_ = {};
    meshShaderProps_.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT;

    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &meshShaderProps_;
    vkGetPhysicalDeviceProperties2(physicalDevice_, &props2);

    LOG_INFO("VkDevice created  (%u unique queue families)",
             static_cast<u32>(uniqueQueueFamilies.size()));
}

// ---------------------------------------------------------------------------
// Debug messenger
// ---------------------------------------------------------------------------
void VulkanDevice::setupDebugMessenger() {
    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;

    createInfo.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
#ifndef NDEBUG
    createInfo.messageSeverity |=
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT;
#endif

    createInfo.messageType =
        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;

    createInfo.pfnUserCallback = debugCallback;

    auto createFn = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
    if (createFn) {
        VK_CHECK(createFn(instance_, &createInfo, nullptr, &debugMessenger_));
        LOG_INFO("Vulkan debug messenger created");
    } else {
        LOG_WARN("vkCreateDebugUtilsMessengerEXT not available");
    }
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------
VkInstance       VulkanDevice::getInstance()       const { return instance_; }
VkPhysicalDevice VulkanDevice::getPhysicalDevice() const { return physicalDevice_; }
VkDevice         VulkanDevice::getDevice()         const { return device_; }
VkSurfaceKHR     VulkanDevice::getSurface()        const { return surface_; }

const Queues&             VulkanDevice::getQueues()             const { return queues_; }
const QueueFamilyIndices& VulkanDevice::getQueueFamilyIndices() const { return queueFamilies_; }

const VkPhysicalDeviceProperties& VulkanDevice::getProperties() const {
    return properties_;
}

const VkPhysicalDeviceMeshShaderPropertiesEXT&
VulkanDevice::getMeshShaderProperties() const {
    return meshShaderProps_;
}

// ---------------------------------------------------------------------------
// Memory type lookup
// ---------------------------------------------------------------------------
u32 VulkanDevice::findMemoryType(u32 typeFilter,
                                  VkMemoryPropertyFlags properties) const {
    for (u32 i = 0; i < memProperties_.memoryTypeCount; ++i) {
        if ((typeFilter & (1u << i)) &&
            (memProperties_.memoryTypes[i].propertyFlags & properties) ==
                properties) {
            return i;
        }
    }
    LOG_ERROR("Failed to find suitable memory type (filter=0x%x props=0x%x)",
              typeFilter, properties);
    std::abort();
}

void VulkanDevice::waitIdle() const {
    VkResult r = vkDeviceWaitIdle(device_);
    if (r != VK_SUCCESS && r != VK_ERROR_DEVICE_LOST) {
        LOG_ERROR("vkDeviceWaitIdle failed with %d", (int)r);
    }
}

} // namespace phosphor
