#pragma once

#include "core/types.h"
#include "rhi/vk_common.h"
#include <vector>

namespace phosphor {

/// Accumulates Vulkan pipeline barriers and flushes them in a single
/// vkCmdPipelineBarrier2 call.  Uses VK_KHR_synchronization2 structs.
class BarrierBuilder {
public:
    void addImageBarrier(VkImage                        image,
                         VkImageLayout                  oldLayout,
                         VkImageLayout                  newLayout,
                         VkPipelineStageFlags2          srcStage,
                         VkAccessFlags2                 srcAccess,
                         VkPipelineStageFlags2          dstStage,
                         VkAccessFlags2                 dstAccess,
                         VkImageSubresourceRange        subresourceRange);

    void addBufferBarrier(VkBuffer               buffer,
                          VkPipelineStageFlags2   srcStage,
                          VkAccessFlags2          srcAccess,
                          VkPipelineStageFlags2   dstStage,
                          VkAccessFlags2          dstAccess,
                          VkDeviceSize            offset = 0,
                          VkDeviceSize            size   = VK_WHOLE_SIZE);

    /// Emit all accumulated barriers into a single vkCmdPipelineBarrier2 and
    /// clear internal state.  No-op if no barriers are pending.
    void flush(VkCommandBuffer cmd);

    bool empty() const { return imageBarriers_.empty() && bufferBarriers_.empty(); }

private:
    std::vector<VkImageMemoryBarrier2>  imageBarriers_;
    std::vector<VkBufferMemoryBarrier2> bufferBarriers_;
};

} // namespace phosphor
