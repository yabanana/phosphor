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
// AAMode — selects the anti-aliasing technique
// ---------------------------------------------------------------------------

enum class AAMode : u32 {
    None = 0,
    FXAA = 1,
    TAA  = 2
};

// ---------------------------------------------------------------------------
// AAPass — compute-based anti-aliasing (FXAA 3.11 / TAA)
//
// FXAA: luminance-based edge detection + directional filtering
// TAA:  temporal accumulation with neighborhood clamping
//
// Both read from an input image and write to an internal output image.
// TAA also maintains a history buffer for temporal accumulation.
// ---------------------------------------------------------------------------

class AAPass {
public:
    AAPass(VulkanDevice& device, GpuAllocator& allocator,
           PipelineManager& pipelines, BindlessDescriptorManager& descriptors,
           VkExtent2D extent);
    ~AAPass();

    AAPass(const AAPass&) = delete;
    AAPass& operator=(const AAPass&) = delete;

    void recreate(VkExtent2D newExtent);

    /// Apply the selected AA mode to the input image.
    /// @param cmd       Active command buffer
    /// @param input     Image view of the source (expected in GENERAL layout)
    /// @param depthView Depth buffer view (needed by TAA for reprojection)
    /// @param mode      AA technique to apply
    void apply(VkCommandBuffer cmd, VkImageView input, VkImageView depthView, AAMode mode);

    [[nodiscard]] VkImageView     getOutputView()  const { return outputView_; }
    [[nodiscard]] AllocatedImage& getOutputImage()        { return outputImage_; }

    /// Set previous frame VP for TAA reprojection.
    void setTAAMatrices(const float* prevVP, const float* currentInvVP);

private:
    void createResources();
    void cleanupResources();
    void ensurePipelinesCreated();
    void createPassDescriptors();
    void applyFXAA(VkCommandBuffer cmd, VkImageView input);
    void applyTAA(VkCommandBuffer cmd, VkImageView input, VkImageView depthView);

    // --- Device references ---
    VulkanDevice&              device_;
    GpuAllocator&              allocator_;
    PipelineManager&           pipelines_;
    BindlessDescriptorManager& descriptors_;
    VkExtent2D                 extent_;

    // --- Output image (RGBA8 for FXAA, RGBA16F for TAA) ---
    AllocatedImage outputImage_{};
    VkImageView    outputView_ = VK_NULL_HANDLE;

    // --- TAA history buffer (RGBA16F, ping-ponged internally) ---
    AllocatedImage historyBuffer_{};
    VkImageView    historyView_ = VK_NULL_HANDLE;

    // --- TAA UBO ---
    AllocatedBuffer taaUBO_{};

    // --- Pipelines ---
    VkPipeline fxaaPipeline_ = VK_NULL_HANDLE;
    VkPipeline taaPipeline_  = VK_NULL_HANDLE;

    // --- Descriptor infrastructure ---
    VkDescriptorSetLayout fxaaLayout_  = VK_NULL_HANDLE;
    VkDescriptorSetLayout taaLayout_   = VK_NULL_HANDLE;

    VkPipelineLayout fxaaPlLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout taaPlLayout_  = VK_NULL_HANDLE;

    VkDescriptorPool passPool_   = VK_NULL_HANDLE;
    VkDescriptorSet  fxaaSet_    = VK_NULL_HANDLE;
    VkDescriptorSet  taaSet_     = VK_NULL_HANDLE;

    bool initialized_ = false;
};

} // namespace phosphor
