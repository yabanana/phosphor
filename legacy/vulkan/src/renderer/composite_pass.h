#pragma once

#include "core/types.h"
#include "rhi/vk_common.h"

namespace phosphor {

class VulkanDevice;
class PipelineManager;

// ---------------------------------------------------------------------------
// CompositePass — final blit of the LDR colour result onto the swapchain
// image using vkCmdBlitImage.
// ---------------------------------------------------------------------------

class CompositePass {
public:
    CompositePass(VulkanDevice& device, PipelineManager& pipelines);

    /// Record a blit from @p srcImage (LDR, TRANSFER_SRC_OPTIMAL layout)
    /// to @p swapchainImage (TRANSFER_DST_OPTIMAL layout).  The caller is
    /// responsible for the layout transitions before and after.
    void record(VkCommandBuffer cmd,
                VkImage srcImage, VkExtent2D srcExtent,
                VkImage swapchainImage, VkImageView swapchainView,
                VkExtent2D swapExtent);

private:
    VulkanDevice&   device_;
    PipelineManager& pipelines_;
};

} // namespace phosphor
