#include "rhi/vk_commands.h"
#include "rhi/vk_device.h"

namespace phosphor {

CommandManager::CommandManager(VulkanDevice& device) : device_(device) {
    VkDevice dev = device_.getDevice();
    u32 graphicsFamily = device_.getQueueFamilyIndices().graphics.value();

    // Per-frame command pools with individual buffer reset capability.
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = graphicsFamily;

    for (u32 i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        VK_CHECK(vkCreateCommandPool(dev, &poolInfo, nullptr, &graphicsPools_[i]));
    }

    // Allocate one primary command buffer per pool.
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    for (u32 i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        allocInfo.commandPool = graphicsPools_[i];
        VK_CHECK(vkAllocateCommandBuffers(dev, &allocInfo, &primaryBuffers_[i]));
    }

    // Dedicated pool for one-shot immediate submissions.
    VkCommandPoolCreateInfo immediatePoolInfo{};
    immediatePoolInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    immediatePoolInfo.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    immediatePoolInfo.queueFamilyIndex = graphicsFamily;
    VK_CHECK(vkCreateCommandPool(dev, &immediatePoolInfo, nullptr, &immediatePool_));

    // Fence for immediate submissions (created unsignaled).
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VK_CHECK(vkCreateFence(dev, &fenceInfo, nullptr, &immediateFence_));

    LOG_INFO("CommandManager: created %u frame pools + immediate pool",
             FRAMES_IN_FLIGHT);
}

CommandManager::~CommandManager() {
    VkDevice dev = device_.getDevice();

    if (immediateFence_ != VK_NULL_HANDLE)
        vkDestroyFence(dev, immediateFence_, nullptr);
    if (immediatePool_ != VK_NULL_HANDLE)
        vkDestroyCommandPool(dev, immediatePool_, nullptr);

    for (u32 i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        // Command buffers are implicitly freed when the pool is destroyed.
        if (graphicsPools_[i] != VK_NULL_HANDLE)
            vkDestroyCommandPool(dev, graphicsPools_[i], nullptr);
    }
}

VkCommandBuffer CommandManager::getCommandBuffer(u32 frameIndex) const {
    return primaryBuffers_[frameIndex];
}

void CommandManager::resetPools(u32 frameIndex) {
    VK_CHECK(vkResetCommandPool(device_.getDevice(), graphicsPools_[frameIndex], 0));
}

void CommandManager::immediateSubmit(std::function<void(VkCommandBuffer)> fn) {
    VkDevice dev = device_.getDevice();

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool        = immediatePool_;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(dev, &allocInfo, &cmd));

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

    fn(cmd);

    VK_CHECK(vkEndCommandBuffer(cmd));

    VkSubmitInfo submitInfo{};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &cmd;

    VkQueue graphicsQueue = device_.getQueues().graphics;
    VkResult sr = vkQueueSubmit(graphicsQueue, 1, &submitInfo, immediateFence_);
    if (sr == VK_ERROR_DEVICE_LOST) { LOG_WARN("Device lost in immediateSubmit (queue submit)"); return; }
    VK_CHECK(sr);
    VkResult wr = vkWaitForFences(dev, 1, &immediateFence_, VK_TRUE, UINT64_MAX);
    if (wr == VK_ERROR_DEVICE_LOST) { LOG_WARN("Device lost in immediateSubmit"); return; }
    VK_CHECK(wr);

    // Reset fence and pool for next immediate submit.
    VK_CHECK(vkResetFences(dev, 1, &immediateFence_));
    VK_CHECK(vkResetCommandPool(dev, immediatePool_, 0));
}

VkCommandPool CommandManager::getGraphicsPool(u32 frameIndex) const {
    return graphicsPools_[frameIndex];
}

} // namespace phosphor
