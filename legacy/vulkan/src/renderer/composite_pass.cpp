#include "renderer/composite_pass.h"
#include "rhi/vk_device.h"
#include "rhi/vk_pipeline.h"
#include "diagnostics/debug_utils.h"
#include "core/log.h"

namespace phosphor {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

CompositePass::CompositePass(VulkanDevice& device, PipelineManager& pipelines)
    : device_(device), pipelines_(pipelines) {
}

// ---------------------------------------------------------------------------
// Record the final blit to the swapchain
// ---------------------------------------------------------------------------

void CompositePass::record(VkCommandBuffer cmd,
                           VkImage srcImage, VkExtent2D srcExtent,
                           VkImage swapchainImage, VkImageView /*swapchainView*/,
                           VkExtent2D swapExtent) {
    PHOSPHOR_GPU_LABEL(cmd, "CompositePass");

    // --- Transition source (LDR) image: GENERAL -> TRANSFER_SRC_OPTIMAL ---
    // --- Transition swapchain image: UNDEFINED -> TRANSFER_DST_OPTIMAL ---
    {
        VkImageMemoryBarrier2 barriers[2]{};

        // LDR source -> TRANSFER_SRC
        barriers[0].sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barriers[0].srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barriers[0].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        barriers[0].dstStageMask  = VK_PIPELINE_STAGE_2_BLIT_BIT;
        barriers[0].dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        barriers[0].oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
        barriers[0].newLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barriers[0].image         = srcImage;
        barriers[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        // Swapchain -> TRANSFER_DST
        barriers[1].sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barriers[1].srcStageMask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        barriers[1].srcAccessMask = 0;
        barriers[1].dstStageMask  = VK_PIPELINE_STAGE_2_BLIT_BIT;
        barriers[1].dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barriers[1].oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED; // discard previous content — valid per spec
        barriers[1].newLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barriers[1].image         = swapchainImage;
        barriers[1].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dep.imageMemoryBarrierCount = 2;
        dep.pImageMemoryBarriers    = barriers;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    // --- Blit LDR -> swapchain ---
    VkImageBlit2 blitRegion{VK_STRUCTURE_TYPE_IMAGE_BLIT_2};
    blitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blitRegion.srcSubresource.layerCount = 1;
    blitRegion.srcOffsets[0]             = {0, 0, 0};
    blitRegion.srcOffsets[1]             = {
        static_cast<i32>(srcExtent.width),
        static_cast<i32>(srcExtent.height),
        1
    };

    blitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blitRegion.dstSubresource.layerCount = 1;
    blitRegion.dstOffsets[0]             = {0, 0, 0};
    blitRegion.dstOffsets[1]             = {
        static_cast<i32>(swapExtent.width),
        static_cast<i32>(swapExtent.height),
        1
    };

    VkBlitImageInfo2 blitInfo{VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2};
    blitInfo.srcImage       = srcImage;
    blitInfo.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    blitInfo.dstImage       = swapchainImage;
    blitInfo.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    blitInfo.regionCount    = 1;
    blitInfo.pRegions       = &blitRegion;
    blitInfo.filter         = VK_FILTER_NEAREST;

    vkCmdBlitImage2(cmd, &blitInfo);

    // --- Transition swapchain: TRANSFER_DST_OPTIMAL -> PRESENT_SRC_KHR ---
    {
        VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        barrier.srcStageMask  = VK_PIPELINE_STAGE_2_BLIT_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrier.dstStageMask  = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
        barrier.dstAccessMask = 0;
        barrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // ready for ImGui
        barrier.image         = swapchainImage;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers    = &barrier;
        vkCmdPipelineBarrier2(cmd, &dep);
    }
}

} // namespace phosphor
