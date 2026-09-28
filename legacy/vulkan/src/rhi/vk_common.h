#pragma once

#include <vulkan/vulkan.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <functional>
#include "core/types.h"
#include "core/log.h"

#define VK_CHECK(x)                                                            \
    do {                                                                       \
        VkResult _r = (x);                                                     \
        if (_r != VK_SUCCESS) {                                                \
            LOG_ERROR("Vulkan error %d at %s:%d", (int)_r, __FILE__,           \
                      __LINE__);                                               \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

namespace phosphor {

// RAII wrapper for Vulkan handles destroyed via VkDevice
template <typename T, auto DestroyFn>
class VkHandle {
    VkDevice device_ = VK_NULL_HANDLE;
    T handle_ = VK_NULL_HANDLE;

public:
    VkHandle() = default;
    VkHandle(VkDevice d, T h) : device_(d), handle_(h) {}

    ~VkHandle() {
        if (handle_ != VK_NULL_HANDLE) {
            DestroyFn(device_, handle_, nullptr);
        }
    }

    VkHandle(VkHandle&& o) noexcept
        : device_(o.device_),
          handle_(std::exchange(o.handle_, VK_NULL_HANDLE)) {}

    VkHandle& operator=(VkHandle&& o) noexcept {
        if (this != &o) {
            if (handle_ != VK_NULL_HANDLE) {
                DestroyFn(device_, handle_, nullptr);
            }
            device_ = o.device_;
            handle_ = std::exchange(o.handle_, VK_NULL_HANDLE);
        }
        return *this;
    }

    VkHandle(const VkHandle&) = delete;
    VkHandle& operator=(const VkHandle&) = delete;

    operator T() const { return handle_; }
    T get() const { return handle_; }
    T* ptr() { return &handle_; }
    explicit operator bool() const { return handle_ != VK_NULL_HANDLE; }

    void reset() {
        if (handle_ != VK_NULL_HANDLE) {
            DestroyFn(device_, handle_, nullptr);
            handle_ = VK_NULL_HANDLE;
        }
    }
};

// RAII wrapper for Vulkan handles destroyed via VkInstance
template <typename T, auto DestroyFn>
class VkInstanceHandle {
    VkInstance instance_ = VK_NULL_HANDLE;
    T handle_ = VK_NULL_HANDLE;

public:
    VkInstanceHandle() = default;
    VkInstanceHandle(VkInstance i, T h) : instance_(i), handle_(h) {}

    ~VkInstanceHandle() {
        if (handle_ != VK_NULL_HANDLE) {
            DestroyFn(instance_, handle_, nullptr);
        }
    }

    VkInstanceHandle(VkInstanceHandle&& o) noexcept
        : instance_(o.instance_),
          handle_(std::exchange(o.handle_, VK_NULL_HANDLE)) {}

    VkInstanceHandle& operator=(VkInstanceHandle&& o) noexcept {
        if (this != &o) {
            if (handle_ != VK_NULL_HANDLE) {
                DestroyFn(instance_, handle_, nullptr);
            }
            instance_ = o.instance_;
            handle_ = std::exchange(o.handle_, VK_NULL_HANDLE);
        }
        return *this;
    }

    VkInstanceHandle(const VkInstanceHandle&) = delete;
    VkInstanceHandle& operator=(const VkInstanceHandle&) = delete;

    operator T() const { return handle_; }
    T get() const { return handle_; }
    T* ptr() { return &handle_; }
};

// Common type aliases
using UniquePipeline = VkHandle<VkPipeline, vkDestroyPipeline>;
using UniquePipelineLayout = VkHandle<VkPipelineLayout, vkDestroyPipelineLayout>;
using UniqueDescriptorSetLayout = VkHandle<VkDescriptorSetLayout, vkDestroyDescriptorSetLayout>;
using UniqueDescriptorPool = VkHandle<VkDescriptorPool, vkDestroyDescriptorPool>;
using UniqueSemaphore = VkHandle<VkSemaphore, vkDestroySemaphore>;
using UniqueFence = VkHandle<VkFence, vkDestroyFence>;
using UniqueCommandPool = VkHandle<VkCommandPool, vkDestroyCommandPool>;
using UniqueShaderModule = VkHandle<VkShaderModule, vkDestroyShaderModule>;
using UniqueSampler = VkHandle<VkSampler, vkDestroySampler>;
using UniqueImageView = VkHandle<VkImageView, vkDestroyImageView>;
using UniquePipelineCache = VkHandle<VkPipelineCache, vkDestroyPipelineCache>;

} // namespace phosphor
