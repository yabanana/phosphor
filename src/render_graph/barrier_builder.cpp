#include "render_graph/barrier_builder.h"

namespace phosphor {

void BarrierBuilder::addImageBarrier(VkImage                  image,
                                     VkImageLayout            oldLayout,
                                     VkImageLayout            newLayout,
                                     VkPipelineStageFlags2    srcStage,
                                     VkAccessFlags2           srcAccess,
                                     VkPipelineStageFlags2    dstStage,
                                     VkAccessFlags2           dstAccess,
                                     VkImageSubresourceRange  subresourceRange)
{
    VkImageMemoryBarrier2 barrier{};
    barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcStageMask        = srcStage;
    barrier.srcAccessMask       = srcAccess;
    barrier.dstStageMask        = dstStage;
    barrier.dstAccessMask       = dstAccess;
    barrier.oldLayout           = oldLayout;
    barrier.newLayout           = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image               = image;
    barrier.subresourceRange    = subresourceRange;

    imageBarriers_.push_back(barrier);
}

void BarrierBuilder::addBufferBarrier(VkBuffer              buffer,
                                      VkPipelineStageFlags2 srcStage,
                                      VkAccessFlags2        srcAccess,
                                      VkPipelineStageFlags2 dstStage,
                                      VkAccessFlags2        dstAccess,
                                      VkDeviceSize          offset,
                                      VkDeviceSize          size)
{
    VkBufferMemoryBarrier2 barrier{};
    barrier.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    barrier.srcStageMask        = srcStage;
    barrier.srcAccessMask       = srcAccess;
    barrier.dstStageMask        = dstStage;
    barrier.dstAccessMask       = dstAccess;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer              = buffer;
    barrier.offset              = offset;
    barrier.size                = size;

    bufferBarriers_.push_back(barrier);
}

void BarrierBuilder::flush(VkCommandBuffer cmd)
{
    if (imageBarriers_.empty() && bufferBarriers_.empty()) {
        return;
    }

    VkDependencyInfo depInfo{};
    depInfo.sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    depInfo.imageMemoryBarrierCount  = static_cast<u32>(imageBarriers_.size());
    depInfo.pImageMemoryBarriers     = imageBarriers_.data();
    depInfo.bufferMemoryBarrierCount = static_cast<u32>(bufferBarriers_.size());
    depInfo.pBufferMemoryBarriers    = bufferBarriers_.data();

    vkCmdPipelineBarrier2(cmd, &depInfo);

    imageBarriers_.clear();
    bufferBarriers_.clear();
}

} // namespace phosphor
