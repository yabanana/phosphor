#pragma once

#include "core/types.h"
#include "rhi/vk_common.h"
#include "rhi/vk_allocator.h"

namespace phosphor {

class VulkanDevice;
class GpuAllocator;
class PipelineManager;
class BindlessDescriptorManager;

// ---------------------------------------------------------------------------
// TonemapPass — compute pass that applies ACES tonemapping to an HDR
// input and writes an LDR RGBA8 output.
// ---------------------------------------------------------------------------

class TonemapPass {
public:
    TonemapPass(VulkanDevice& device, GpuAllocator& allocator,
                PipelineManager& pipelines, BindlessDescriptorManager& descriptors,
                VkExtent2D extent);
    ~TonemapPass();

    TonemapPass(const TonemapPass&) = delete;
    TonemapPass& operator=(const TonemapPass&) = delete;

    void recreate(VkExtent2D newExtent);

    /// Run the ACES tonemap compute shader.
    /// @param hdrInput  Image view of the HDR source (RGBA16F, GENERAL layout).
    /// @param exposure  Exposure multiplier applied before the tonemap curve.
    void apply(VkCommandBuffer cmd, VkImageView hdrInput, float exposure);

    [[nodiscard]] AllocatedImage& getLDRImage()  { return ldrImage_; }
    [[nodiscard]] VkImageView     getLDRView() const { return ldrView_; }

private:
    void createResources();
    void cleanupResources();
    void ensurePipelineCreated();
    void updateDescriptorSet(VkImageView hdrInput);

    VulkanDevice&   device_;
    GpuAllocator&   allocator_;
    PipelineManager& pipelines_;
    BindlessDescriptorManager& descriptors_;
    VkExtent2D      extent_;

    AllocatedImage ldrImage_{};  // RGBA8_UNORM
    VkImageView    ldrView_ = VK_NULL_HANDLE;

    VkPipeline            pipeline_    = VK_NULL_HANDLE;
    VkDescriptorSetLayout passLayout_  = VK_NULL_HANDLE;
    VkDescriptorPool      passPool_    = VK_NULL_HANDLE;
    VkDescriptorSet       passSet_     = VK_NULL_HANDLE;
    VkPipelineLayout      passPlLayout_ = VK_NULL_HANDLE;

    bool initialized_ = false;
    bool descriptorsWritten_ = false;
};

} // namespace phosphor
