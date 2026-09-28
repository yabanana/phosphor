#pragma once

#include "rhi/vk_common.h"
#include "rhi/vk_allocator.h"
#include "core/types.h"

namespace phosphor {

class VulkanDevice;
class CommandManager;
class SyncManager;

struct StagingAllocation {
    void* cpuPtr = nullptr;
    VkDeviceSize offset = 0;
    VkBuffer buffer = VK_NULL_HANDLE;
};

class FrameContext {
public:
    static constexpr VkDeviceSize STAGING_BUFFER_SIZE = 16 * 1024 * 1024; // 16 MB per frame

    FrameContext(VulkanDevice& device, GpuAllocator& allocator,
                 CommandManager& commands, SyncManager& sync, u32 frameIndex);
    ~FrameContext();
    FrameContext(const FrameContext&) = delete;
    FrameContext& operator=(const FrameContext&) = delete;
    FrameContext(FrameContext&&) noexcept;
    FrameContext& operator=(FrameContext&&) noexcept;

    void begin();  // Wait fence, reset command pool, begin cmd buffer, reset staging
    void end();    // End command buffer

    StagingAllocation allocateStaging(VkDeviceSize size, VkDeviceSize alignment = 16);

    VkCommandBuffer getCommandBuffer() const { return cmdBuffer_; }
    u32 getFrameIndex() const { return frameIndex_; }

private:
    VulkanDevice* device_ = nullptr;
    GpuAllocator* allocator_ = nullptr;
    CommandManager* commands_ = nullptr;
    SyncManager* sync_ = nullptr;
    u32 frameIndex_ = 0;

    AllocatedBuffer stagingBuffer_{};
    void* stagingMapped_ = nullptr;
    VkDeviceSize stagingOffset_ = 0;
    VkCommandBuffer cmdBuffer_ = VK_NULL_HANDLE;
};

} // namespace phosphor
