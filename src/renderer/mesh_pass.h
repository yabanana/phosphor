#pragma once

#include "core/types.h"
#include "rhi/vk_common.h"

namespace phosphor {

class VulkanDevice;
class PipelineManager;
class BindlessDescriptorManager;
class GpuScene;
class VisibilityBuffer;
class Camera;

// ---------------------------------------------------------------------------
// MeshPass — records the task/mesh-shader visibility-buffer fill pass.
// ---------------------------------------------------------------------------

class MeshPass {
public:
    MeshPass(VulkanDevice& device, PipelineManager& pipelines,
             BindlessDescriptorManager& descriptors);

    /// Record the mesh shading pass into @p cmd.  Writes to the visibility
    /// buffer colour attachment and depth attachment owned by @p visBuf.
    void recordPass(VkCommandBuffer cmd, GpuScene& scene,
                    const VisibilityBuffer& visBuf, const Camera& camera,
                    u32 frameIndex, float exposure);

private:
    void ensureInitialized();

    VulkanDevice&              device_;
    PipelineManager&           pipelines_;
    BindlessDescriptorManager& descriptors_;
    VkPipeline                 meshPipeline_ = VK_NULL_HANDLE;
    bool                       initialized_  = false;
};

} // namespace phosphor
