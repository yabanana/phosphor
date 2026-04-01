#include "rhi/vk_sync.h"
#include "rhi/vk_device.h"

namespace phosphor {

SyncManager::SyncManager(VulkanDevice& device) : device_(device) {
    VkDevice dev = device_.getDevice();

    VkSemaphoreCreateInfo binarySemInfo{};
    binarySemInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkSemaphoreTypeCreateInfo timelineInfo{};
    timelineInfo.sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    timelineInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    timelineInfo.initialValue  = 0;

    VkSemaphoreCreateInfo timelineSemInfo{};
    timelineSemInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    timelineSemInfo.pNext = &timelineInfo;

    // Create fences in signaled state so the first waitForFrame call
    // returns immediately instead of blocking forever.
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (u32 i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        VK_CHECK(vkCreateSemaphore(dev, &binarySemInfo, nullptr,
                                   &frameSyncs_[i].imageAvailable));
        VK_CHECK(vkCreateSemaphore(dev, &binarySemInfo, nullptr,
                                   &frameSyncs_[i].renderFinished));
        VK_CHECK(vkCreateSemaphore(dev, &timelineSemInfo, nullptr,
                                   &frameSyncs_[i].timelineGpu));
        VK_CHECK(vkCreateFence(dev, &fenceInfo, nullptr,
                               &frameSyncs_[i].inFlight));

        frameSyncs_[i].timelineValue = 0;
    }

    LOG_INFO("SyncManager: created %u frame sync sets", FRAMES_IN_FLIGHT);
}

SyncManager::~SyncManager() {
    VkDevice dev = device_.getDevice();

    for (u32 i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        if (frameSyncs_[i].imageAvailable != VK_NULL_HANDLE)
            vkDestroySemaphore(dev, frameSyncs_[i].imageAvailable, nullptr);
        if (frameSyncs_[i].renderFinished != VK_NULL_HANDLE)
            vkDestroySemaphore(dev, frameSyncs_[i].renderFinished, nullptr);
        if (frameSyncs_[i].timelineGpu != VK_NULL_HANDLE)
            vkDestroySemaphore(dev, frameSyncs_[i].timelineGpu, nullptr);
        if (frameSyncs_[i].inFlight != VK_NULL_HANDLE)
            vkDestroyFence(dev, frameSyncs_[i].inFlight, nullptr);
    }
}

void SyncManager::waitForFrame(u32 frameIndex) {
    VkDevice dev = device_.getDevice();
    VkFence  fence = frameSyncs_[frameIndex].inFlight;

    VK_CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
    VK_CHECK(vkResetFences(dev, 1, &fence));
}

FrameSync& SyncManager::getFrameSync(u32 frameIndex) {
    return frameSyncs_[frameIndex];
}

void SyncManager::advanceTimeline(u32 frameIndex) {
    ++frameSyncs_[frameIndex].timelineValue;
}

} // namespace phosphor
