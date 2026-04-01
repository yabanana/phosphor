#pragma once

#include "core/types.h"
#include "rhi/vk_common.h"
#include "rhi/vk_allocator.h"

namespace phosphor {

class VulkanDevice;
class GpuAllocator;
class PipelineManager;
class BindlessDescriptorManager;
class VisibilityBuffer;
class GpuScene;
class Camera;

// ---------------------------------------------------------------------------
// ReSTIRPass — ReSTIR Direct Illumination (DI) pipeline
//
// Four compute stages:
//   1. Candidate generation — sample lights, run WRS per pixel
//   2. Temporal resampling — reproject and merge with previous frame
//   3. Spatial resampling — merge with similar neighbors
//   4. Shade — evaluate BRDF with final reservoir, write RGBA16F
//
// Reservoir buffers are double-buffered for temporal ping-pong.
// ---------------------------------------------------------------------------

class ReSTIRPass {
public:
    ReSTIRPass(VulkanDevice& device, GpuAllocator& allocator,
               PipelineManager& pipelines, BindlessDescriptorManager& descriptors,
               VkExtent2D extent);
    ~ReSTIRPass();

    ReSTIRPass(const ReSTIRPass&) = delete;
    ReSTIRPass& operator=(const ReSTIRPass&) = delete;

    void recreate(VkExtent2D newExtent);

    /// Stage 1: generate initial candidate reservoirs from the light buffer.
    void generateCandidates(VkCommandBuffer cmd, const VisibilityBuffer& visBuf,
                            const GpuScene& scene, const Camera& camera, u32 frameIndex);

    /// Stage 2: temporal resampling — merge with previous frame reservoirs.
    void temporalResample(VkCommandBuffer cmd, const Camera& camera);

    /// Stage 3: spatial resampling — merge with screen-space neighbors.
    void spatialResample(VkCommandBuffer cmd);

    /// Stage 4: final shading with the selected reservoir light.
    void shade(VkCommandBuffer cmd, const GpuScene& scene, const Camera& camera);

    [[nodiscard]] VkImageView getOutputView() const { return outputView_; }
    [[nodiscard]] VkImage     getOutputImage() const { return restirOutput_.image; }

private:
    void createResources();
    void cleanupResources();
    void ensurePipelinesCreated();
    void createPassDescriptors();
    VkPipeline createComputePipeline(const char* spvRelativePath);

    // --- Device references ---
    VulkanDevice&              device_;
    GpuAllocator&              allocator_;
    PipelineManager&           pipelines_;
    BindlessDescriptorManager& descriptors_;
    VkExtent2D                 extent_;

    // --- Reservoir buffers (double-buffered for temporal ping-pong) ---
    // Each entry: selectedLight(u32) + weightSum(f32) + M(u32) + W(f32) = 16 bytes
    AllocatedBuffer reservoirBuffers_[2]{};
    u32 currentReservoir_ = 0;

    // --- Normal G-buffer (RGBA16F, world normal encoded as rgb*0.5+0.5) ---
    AllocatedImage normalImage_{};
    VkImageView    normalView_ = VK_NULL_HANDLE;

    // --- HDR direct lighting output (RGBA16F) ---
    AllocatedImage restirOutput_{};
    VkImageView    outputView_ = VK_NULL_HANDLE;

    // --- Temporal uniform buffer ---
    AllocatedBuffer temporalUBO_{};

    // --- Compute pipelines ---
    VkPipeline candidatesPipeline_ = VK_NULL_HANDLE;
    VkPipeline temporalPipeline_   = VK_NULL_HANDLE;
    VkPipeline spatialPipeline_    = VK_NULL_HANDLE;
    VkPipeline shadePipeline_      = VK_NULL_HANDLE;

    // --- Per-pass descriptor infrastructure ---
    VkDescriptorSetLayout candidatesLayout_  = VK_NULL_HANDLE;
    VkDescriptorSetLayout temporalLayout_    = VK_NULL_HANDLE;
    VkDescriptorSetLayout spatialLayout_     = VK_NULL_HANDLE;
    VkDescriptorSetLayout shadeLayout_       = VK_NULL_HANDLE;

    VkPipelineLayout candidatesPlLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout temporalPlLayout_   = VK_NULL_HANDLE;
    VkPipelineLayout spatialPlLayout_    = VK_NULL_HANDLE;
    VkPipelineLayout shadePlLayout_      = VK_NULL_HANDLE;

    VkDescriptorPool passPool_        = VK_NULL_HANDLE;
    VkDescriptorSet  candidatesSet_   = VK_NULL_HANDLE;
    VkDescriptorSet  temporalSet_     = VK_NULL_HANDLE;
    VkDescriptorSet  spatialSet_      = VK_NULL_HANDLE;
    VkDescriptorSet  shadeSet_        = VK_NULL_HANDLE;

    // Cached push constants state
    float cachedExposure_ = 1.0f;

    bool initialized_ = false;
};

} // namespace phosphor
