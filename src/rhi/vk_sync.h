#pragma once

#include "core/types.h"
#include "rhi/vk_common.h"
#include <array>

namespace phosphor {

class VulkanDevice;

struct FrameSync {
    VkSemaphore imageAvailable = VK_NULL_HANDLE;
    VkSemaphore renderFinished = VK_NULL_HANDLE;
    VkSemaphore timelineGpu    = VK_NULL_HANDLE;
    u64         timelineValue  = 0;
    VkFence     inFlight       = VK_NULL_HANDLE;
};

class SyncManager {
public:
    explicit SyncManager(VulkanDevice& device);
    ~SyncManager();

    SyncManager(const SyncManager&)            = delete;
    SyncManager& operator=(const SyncManager&) = delete;
    SyncManager(SyncManager&&)                 = delete;
    SyncManager& operator=(SyncManager&&)      = delete;

    void       waitForFrame(u32 frameIndex);
    FrameSync& getFrameSync(u32 frameIndex);
    void       advanceTimeline(u32 frameIndex);

private:
    VulkanDevice&                           device_;
    std::array<FrameSync, FRAMES_IN_FLIGHT> frameSyncs_{};
};

} // namespace phosphor
