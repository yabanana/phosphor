#pragma once

#include "core/types.h"
#include "rhi/vk_common.h"
#include <array>

namespace phosphor {

class VulkanDevice;

class CommandManager {
public:
    explicit CommandManager(VulkanDevice& device);
    ~CommandManager();

    CommandManager(const CommandManager&)            = delete;
    CommandManager& operator=(const CommandManager&) = delete;
    CommandManager(CommandManager&&)                 = delete;
    CommandManager& operator=(CommandManager&&)      = delete;

    VkCommandBuffer getCommandBuffer(u32 frameIndex) const;
    void            resetPools(u32 frameIndex);

    /// Allocate a one-shot command buffer, record into it via `fn`,
    /// submit to the graphics queue, and block until completion.
    void immediateSubmit(std::function<void(VkCommandBuffer)> fn);

    VkCommandPool getGraphicsPool(u32 frameIndex) const;

private:
    VulkanDevice&                                   device_;
    std::array<VkCommandPool, FRAMES_IN_FLIGHT>     graphicsPools_{};
    std::array<VkCommandBuffer, FRAMES_IN_FLIGHT>   primaryBuffers_{};
    VkCommandPool                                   immediatePool_ = VK_NULL_HANDLE;
    VkFence                                         immediateFence_ = VK_NULL_HANDLE;
};

} // namespace phosphor
