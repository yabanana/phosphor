#pragma once

#include "rhi/vk_common.h"
#include "core/types.h"
#include <vector>
#include <mutex>

namespace phosphor {

class VulkanDevice;

class BindlessDescriptorManager {
public:
    BindlessDescriptorManager(VulkanDevice& device);
    ~BindlessDescriptorManager();
    BindlessDescriptorManager(const BindlessDescriptorManager&) = delete;
    BindlessDescriptorManager& operator=(const BindlessDescriptorManager&) = delete;

    u32 registerTexture(VkImageView view, VkImageLayout layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    void unregisterTexture(u32 index);

    u32 registerStorageBuffer(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize range);
    void unregisterStorageBuffer(u32 index);

    VkDescriptorSet getDescriptorSet() const { return descriptorSet_; }
    VkDescriptorSetLayout getLayout() const { return layout_; }
    VkPipelineLayout getBindlessPipelineLayout() const { return pipelineLayout_; }

    void bindToCommandBuffer(VkCommandBuffer cmd, VkPipelineBindPoint bindPoint) const;

private:
    u32 allocateTextureSlot();
    u32 allocateBufferSlot();

    VulkanDevice& device_;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout layout_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;

    std::vector<u32> freeTextureSlots_;
    std::vector<u32> freeBufferSlots_;
    u32 nextTextureSlot_ = 0;
    u32 nextBufferSlot_ = 0;

    VkSampler linearSampler_ = VK_NULL_HANDLE;
    VkSampler nearestSampler_ = VK_NULL_HANDLE;
    VkSampler shadowSampler_ = VK_NULL_HANDLE;
};

} // namespace phosphor
