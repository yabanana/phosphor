#include "rhi/vk_descriptors.h"
#include "rhi/vk_device.h"
#include "renderer/push_constants.h"
#include <array>
#include <cassert>

namespace phosphor {

BindlessDescriptorManager::BindlessDescriptorManager(VulkanDevice& device) : device_(device) {
    VkDevice dev = device_.getDevice();

    // Create immutable samplers
    {
        VkSamplerCreateInfo info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        info.magFilter = VK_FILTER_LINEAR;
        info.minFilter = VK_FILTER_LINEAR;
        info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        info.addressModeU = info.addressModeV = info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        info.maxAnisotropy = 16.0f;
        info.anisotropyEnable = VK_TRUE;
        info.maxLod = VK_LOD_CLAMP_NONE;
        VK_CHECK(vkCreateSampler(dev, &info, nullptr, &linearSampler_));

        info.magFilter = VK_FILTER_NEAREST;
        info.minFilter = VK_FILTER_NEAREST;
        info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        info.anisotropyEnable = VK_FALSE;
        VK_CHECK(vkCreateSampler(dev, &info, nullptr, &nearestSampler_));

        VkSamplerCreateInfo shadowInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        shadowInfo.magFilter = VK_FILTER_LINEAR;
        shadowInfo.minFilter = VK_FILTER_LINEAR;
        shadowInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        shadowInfo.addressModeU = shadowInfo.addressModeV = shadowInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        shadowInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        shadowInfo.compareEnable = VK_TRUE;
        shadowInfo.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        shadowInfo.maxLod = VK_LOD_CLAMP_NONE;
        VK_CHECK(vkCreateSampler(dev, &shadowInfo, nullptr, &shadowSampler_));
    }

    // Descriptor set layout: binding 0 = sampled images[], binding 1 = samplers[], binding 2 = storage buffers[]
    std::array<VkSampler, 3> immutableSamplers = {linearSampler_, nearestSampler_, shadowSampler_};

    std::array<VkDescriptorSetLayoutBinding, 3> bindings{};

    // Binding 0: sampled images (bindless texture array)
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    bindings[0].descriptorCount = MAX_BINDLESS_TEXTURES;
    bindings[0].stageFlags = VK_SHADER_STAGE_ALL;

    // Binding 1: samplers (immutable, small fixed set)
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    bindings[1].descriptorCount = static_cast<u32>(immutableSamplers.size());
    bindings[1].stageFlags = VK_SHADER_STAGE_ALL;
    bindings[1].pImmutableSamplers = immutableSamplers.data();

    // Binding 2: storage buffers (bindless buffer array)
    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].descriptorCount = MAX_BINDLESS_BUFFERS;
    bindings[2].stageFlags = VK_SHADER_STAGE_ALL;

    // Binding flags: UPDATE_AFTER_BIND + PARTIALLY_BOUND for bindings 0 and 2
    // Binding 1 (samplers) is fixed/immutable
    std::array<VkDescriptorBindingFlags, 3> bindingFlags{};
    bindingFlags[0] = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT
                    | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT
                    | VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT;
    bindingFlags[1] = 0; // immutable samplers, no special flags
    bindingFlags[2] = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT
                    | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;

    VkDescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
    flagsInfo.bindingCount = static_cast<u32>(bindingFlags.size());
    flagsInfo.pBindingFlags = bindingFlags.data();

    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.pNext = &flagsInfo;
    layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    layoutInfo.bindingCount = static_cast<u32>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    VK_CHECK(vkCreateDescriptorSetLayout(dev, &layoutInfo, nullptr, &layout_));

    // Descriptor pool
    std::array<VkDescriptorPoolSize, 3> poolSizes{};
    poolSizes[0] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, MAX_BINDLESS_TEXTURES};
    poolSizes[1] = {VK_DESCRIPTOR_TYPE_SAMPLER, static_cast<u32>(immutableSamplers.size())};
    poolSizes[2] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, MAX_BINDLESS_BUFFERS};

    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = static_cast<u32>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();

    VK_CHECK(vkCreateDescriptorPool(dev, &poolInfo, nullptr, &pool_));

    // Allocate the single global descriptor set
    u32 variableCount = MAX_BINDLESS_TEXTURES;
    VkDescriptorSetVariableDescriptorCountAllocateInfo variableInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO};
    variableInfo.descriptorSetCount = 1;
    variableInfo.pDescriptorCounts = &variableCount;

    VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.pNext = &variableInfo;
    allocInfo.descriptorPool = pool_;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &layout_;

    VK_CHECK(vkAllocateDescriptorSets(dev, &allocInfo, &descriptorSet_));

    // Pipeline layout with push constants
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_ALL;
    pushRange.offset = 0;
    pushRange.size = sizeof(PushConstants);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &layout_;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;

    VK_CHECK(vkCreatePipelineLayout(dev, &pipelineLayoutInfo, nullptr, &pipelineLayout_));

    LOG_INFO("Bindless descriptor manager initialized: %u textures, %u buffers",
             MAX_BINDLESS_TEXTURES, MAX_BINDLESS_BUFFERS);
}

BindlessDescriptorManager::~BindlessDescriptorManager() {
    VkDevice dev = device_.getDevice();
    if (pipelineLayout_) vkDestroyPipelineLayout(dev, pipelineLayout_, nullptr);
    if (pool_) vkDestroyDescriptorPool(dev, pool_, nullptr);
    if (layout_) vkDestroyDescriptorSetLayout(dev, layout_, nullptr);
    if (linearSampler_) vkDestroySampler(dev, linearSampler_, nullptr);
    if (nearestSampler_) vkDestroySampler(dev, nearestSampler_, nullptr);
    if (shadowSampler_) vkDestroySampler(dev, shadowSampler_, nullptr);
}

u32 BindlessDescriptorManager::allocateTextureSlot() {
    if (!freeTextureSlots_.empty()) {
        u32 slot = freeTextureSlots_.back();
        freeTextureSlots_.pop_back();
        return slot;
    }
    assert(nextTextureSlot_ < MAX_BINDLESS_TEXTURES && "Bindless texture limit exceeded");
    return nextTextureSlot_++;
}

u32 BindlessDescriptorManager::allocateBufferSlot() {
    if (!freeBufferSlots_.empty()) {
        u32 slot = freeBufferSlots_.back();
        freeBufferSlots_.pop_back();
        return slot;
    }
    assert(nextBufferSlot_ < MAX_BINDLESS_BUFFERS && "Bindless buffer limit exceeded");
    return nextBufferSlot_++;
}

u32 BindlessDescriptorManager::registerTexture(VkImageView view, VkImageLayout layout) {
    u32 index = allocateTextureSlot();

    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageView = view;
    imageInfo.imageLayout = layout;

    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = descriptorSet_;
    write.dstBinding = 0;
    write.dstArrayElement = index;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    write.pImageInfo = &imageInfo;

    vkUpdateDescriptorSets(device_.getDevice(), 1, &write, 0, nullptr);
    return index;
}

void BindlessDescriptorManager::unregisterTexture(u32 index) {
    freeTextureSlots_.push_back(index);
}

u32 BindlessDescriptorManager::registerStorageBuffer(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize range) {
    u32 index = allocateBufferSlot();

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = buffer;
    bufferInfo.offset = offset;
    bufferInfo.range = range;

    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = descriptorSet_;
    write.dstBinding = 2;
    write.dstArrayElement = index;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &bufferInfo;

    vkUpdateDescriptorSets(device_.getDevice(), 1, &write, 0, nullptr);
    return index;
}

void BindlessDescriptorManager::unregisterStorageBuffer(u32 index) {
    freeBufferSlots_.push_back(index);
}

void BindlessDescriptorManager::bindToCommandBuffer(VkCommandBuffer cmd, VkPipelineBindPoint bindPoint) const {
    vkCmdBindDescriptorSets(cmd, bindPoint, pipelineLayout_, 0, 1, &descriptorSet_, 0, nullptr);
}

} // namespace phosphor
