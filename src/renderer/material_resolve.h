#pragma once

#include "core/types.h"
#include "rhi/vk_common.h"
#include "rhi/vk_allocator.h"

namespace phosphor {

class VulkanDevice;
class GpuAllocator;
class PipelineManager;
class BindlessDescriptorManager;
class GpuScene;
class VisibilityBuffer;
class Camera;

// ---------------------------------------------------------------------------
// MaterialResolve — compute pass that reads the visibility buffer, fetches
// per-pixel material data, evaluates PBR lighting, and writes to an
// RGBA16F HDR render target.
// ---------------------------------------------------------------------------

class MaterialResolve {
public:
    MaterialResolve(VulkanDevice& device, GpuAllocator& allocator,
                    PipelineManager& pipelines,
                    BindlessDescriptorManager& descriptors,
                    VkExtent2D extent);
    ~MaterialResolve();

    MaterialResolve(const MaterialResolve&) = delete;
    MaterialResolve& operator=(const MaterialResolve&) = delete;

    void recreate(VkExtent2D newExtent);

    /// Dispatch the resolve compute shader.  Reads from the visibility and
    /// depth images and writes into the HDR colour target.
    void resolve(VkCommandBuffer cmd, const VisibilityBuffer& visBuf,
                 GpuScene& scene, const Camera& camera, u32 frameIndex,
                 float exposure);

    [[nodiscard]] VkImage     getHDRImage() const { return hdrColor_.image; }
    [[nodiscard]] VkImageView getHDRView()  const { return hdrView_; }

private:
    void createResources();
    void cleanupResources();
    void ensurePipelineCreated();
    void updateDescriptorSet(const VisibilityBuffer& visBuf);

    VulkanDevice&              device_;
    GpuAllocator&              allocator_;
    PipelineManager&           pipelines_;
    BindlessDescriptorManager& descriptors_;
    VkExtent2D                 extent_;

    AllocatedImage  hdrColor_{};  // RGBA16F
    VkImageView     hdrView_ = VK_NULL_HANDLE;

    VkPipeline              resolvePipeline_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout   passLayout_      = VK_NULL_HANDLE;
    VkDescriptorPool        passPool_        = VK_NULL_HANDLE;
    VkDescriptorSet         passSet_         = VK_NULL_HANDLE;
    VkPipelineLayout        passPlLayout_    = VK_NULL_HANDLE;

    bool initialized_ = false;
};

} // namespace phosphor
