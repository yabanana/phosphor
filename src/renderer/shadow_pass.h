#pragma once

#include "core/types.h"
#include "rhi/vk_common.h"
#include "rhi/vk_allocator.h"

#include <glm/glm.hpp>
#include <array>

namespace phosphor {

class VulkanDevice;
class GpuAllocator;
class PipelineManager;
class BindlessDescriptorManager;
class GpuScene;
class Camera;

// ---------------------------------------------------------------------------
// ShadowPass — Cascaded Shadow Maps (CSM) using mesh shaders.
//
// Renders 4 shadow cascades into a D32_SFLOAT image array (2048x2048 x 4).
// Uses Practical Split Scheme (PSSM) to compute cascade split distances
// and per-cascade light-space view-projection matrices.
//
// The shadow task/mesh shaders cull against the light frustum and output
// depth only (no colour attachments).
// ---------------------------------------------------------------------------

class ShadowPass {
public:
    static constexpr u32 CASCADE_COUNT    = 4;
    static constexpr u32 SHADOW_MAP_SIZE  = 2048;

    ShadowPass(VulkanDevice& device, GpuAllocator& allocator,
               PipelineManager& pipelines, BindlessDescriptorManager& descriptors);
    ~ShadowPass();

    ShadowPass(const ShadowPass&) = delete;
    ShadowPass& operator=(const ShadowPass&) = delete;

    /// Compute cascade split distances and light-space VP matrices.
    /// @param camera       The main camera (defines the view frustum to split).
    /// @param lightDir     Normalized world-space light direction (from light towards scene).
    /// @param sceneRadius  Approximate bounding radius of the scene (for shadow ortho extents).
    void computeCascades(const Camera& camera, glm::vec3 lightDir, float sceneRadius);

    /// Record shadow rendering for all cascades.
    /// @param cmd     Active command buffer.
    /// @param scene   GPU scene data (instances, meshlets, etc.).
    void recordCascades(VkCommandBuffer cmd, GpuScene& scene);

    /// Get the cascade array image view (all 4 layers, for sampling in the lighting pass).
    [[nodiscard]] VkImageView getCascadeArrayView() const { return arrayView_; }

    /// Get the image (for barriers).
    [[nodiscard]] VkImage getShadowImage() const { return shadowMap_.image; }

    /// Get the 4 cascade VP matrices (for shadow sampling in the material resolve).
    [[nodiscard]] const std::array<glm::mat4, CASCADE_COUNT>& getCascadeMatrices() const { return cascadeVP_; }

    /// Get the 4 cascade split distances (in view-space Z).
    [[nodiscard]] const std::array<float, CASCADE_COUNT>& getCascadeSplits() const { return cascadeSplits_; }

    /// Get per-cascade image views (for depth attachment during rendering).
    [[nodiscard]] VkImageView getCascadeView(u32 cascade) const { return cascadeViews_[cascade]; }

private:
    void createResources();
    void destroyResources();
    void ensurePipelineCreated();

    VkImageView createLayerView(VkImage image, VkFormat format,
                                VkImageAspectFlags aspect, u32 layer);

    VulkanDevice&              device_;
    GpuAllocator&              allocator_;
    PipelineManager&           pipelines_;
    BindlessDescriptorManager& descriptors_;

    // Shadow map: D32_SFLOAT, 2048x2048, 4 array layers
    AllocatedImage shadowMap_{};
    VkImageView    arrayView_ = VK_NULL_HANDLE;
    std::array<VkImageView, CASCADE_COUNT> cascadeViews_{};

    // Per-cascade data
    std::array<glm::mat4, CASCADE_COUNT> cascadeVP_{};
    std::array<float, CASCADE_COUNT>     cascadeSplits_{};

    // Pipeline
    VkPipeline shadowPipeline_ = VK_NULL_HANDLE;
    bool       initialized_    = false;

    // Lambda for PSSM split scheme
    static constexpr float SPLIT_LAMBDA = 0.75f;
};

} // namespace phosphor
