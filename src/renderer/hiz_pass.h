#pragma once

#include "core/types.h"
#include "rhi/vk_common.h"
#include "rhi/vk_allocator.h"

#include <vector>

namespace phosphor {

class VulkanDevice;
class GpuAllocator;
class PipelineManager;
class BindlessDescriptorManager;
class GpuScene;
class Camera;

// ---------------------------------------------------------------------------
// HiZPass — Hierarchical-Z pyramid builder and two-phase GPU occlusion
// culling.
//
// Phase 1: cull instances against LAST frame's Hi-Z (before rendering).
//          Visible instances feed the main mesh-shader pass.
// Phase 2: cull previously-rejected instances against THIS frame's Hi-Z
//          (after the main pass depth is available).
//          Newly visible instances feed a second draw.
//
// The Hi-Z pyramid is an R32_SFLOAT image with a full mip chain, where each
// mip stores the MAX depth of the 2x2 region below it.
// ---------------------------------------------------------------------------

class HiZPass {
public:
    HiZPass(VulkanDevice& device, GpuAllocator& allocator,
            PipelineManager& pipelines, BindlessDescriptorManager& descriptors,
            VkExtent2D extent);
    ~HiZPass();

    HiZPass(const HiZPass&) = delete;
    HiZPass& operator=(const HiZPass&) = delete;

    /// Recreate the Hi-Z pyramid and buffers when the render extent changes.
    void recreate(VkExtent2D newExtent);

    /// Build the Hi-Z pyramid from the current depth buffer.
    /// The depth image must be in SHADER_READ_ONLY_OPTIMAL layout.
    void buildPyramid(VkCommandBuffer cmd, VkImageView depthView);

    /// Phase 1: cull all instances against the PREVIOUS frame's Hi-Z.
    /// Populates the visible-instance buffer and the culled-instance buffer.
    void cullPhase1(VkCommandBuffer cmd, GpuScene& scene, const Camera& camera,
                    u32 frameIndex, float exposure);

    /// Phase 2: cull phase-1 rejects against the CURRENT frame's Hi-Z.
    /// Appends newly-visible instances to the visible-instance buffer.
    void cullPhase2(VkCommandBuffer cmd, GpuScene& scene, const Camera& camera,
                    u32 frameIndex, float exposure);

    /// Reset the atomic counters in the culling buffers (call at frame start).
    void resetCounters(VkCommandBuffer cmd);

    [[nodiscard]] VkImage     getHiZImage() const { return hizImage_.image; }
    [[nodiscard]] VkImageView getHiZView()  const { return hizFullView_; }
    [[nodiscard]] VkExtent2D  getHiZExtent() const { return hizExtent_; }
    [[nodiscard]] u32         getMipLevels() const { return mipCount_; }

    /// Buffer containing visible instance indices after culling.
    /// Layout: [0] = count, [1..N] = instance indices.
    [[nodiscard]] const AllocatedBuffer& getVisibleBuffer() const { return visibleBuffer_; }

    /// Buffer containing instances culled by phase 1 (input for phase 2).
    [[nodiscard]] const AllocatedBuffer& getCulledBuffer() const { return culledBuffer_; }

private:
    void createHiZResources();
    void destroyHiZResources();
    void ensurePipelinesCreated();

    VkImageView createSingleMipView(VkImage image, VkFormat format,
                                    VkImageAspectFlags aspect, u32 mipLevel);

    VulkanDevice&              device_;
    GpuAllocator&              allocator_;
    PipelineManager&           pipelines_;
    BindlessDescriptorManager& descriptors_;

    VkExtent2D extent_{};      // render resolution
    VkExtent2D hizExtent_{};   // power-of-two Hi-Z resolution
    u32        mipCount_ = 0;

    // Hi-Z pyramid image (R32_SFLOAT, full mip chain)
    AllocatedImage hizImage_{};
    VkImageView    hizFullView_ = VK_NULL_HANDLE;   // all mip levels
    std::vector<VkImageView> mipViews_;              // per-mip views for storage writes

    // Culling output buffers
    AllocatedBuffer visibleBuffer_{};   // [count, idx0, idx1, ...]
    AllocatedBuffer culledBuffer_{};    // phase 1 output / phase 2 input
    AllocatedBuffer counterResetBuffer_{}; // staging for counter reset

    // Hi-Z build pass resources
    VkPipeline            hizBuildPipeline_  = VK_NULL_HANDLE;
    VkPipelineLayout      hizBuildPlLayout_  = VK_NULL_HANDLE;
    VkDescriptorSetLayout hizBuildDsLayout_  = VK_NULL_HANDLE;
    VkDescriptorPool      hizBuildPool_      = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> hizBuildSets_;  // one per mip level transition
    VkSampler             nearestSampler_    = VK_NULL_HANDLE;

    // Culling pass resources
    VkPipeline            cullPhase1Pipeline_ = VK_NULL_HANDLE;
    VkPipeline            cullPhase2Pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout      cullPlLayout_       = VK_NULL_HANDLE;
    VkDescriptorSetLayout cullDsLayout_       = VK_NULL_HANDLE;
    VkDescriptorPool      cullPool_           = VK_NULL_HANDLE;
    VkDescriptorSet       cullSet_            = VK_NULL_HANDLE;

    bool initialized_ = false;
};

} // namespace phosphor
