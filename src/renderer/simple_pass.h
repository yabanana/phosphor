#pragma once

#include "core/types.h"
#include "rhi/vk_common.h"
#include "rhi/vk_allocator.h"

#include <glm/glm.hpp>
#include <vector>

namespace phosphor {

class VulkanDevice;
class GpuAllocator;
class PipelineManager;
class CommandManager;

// ---------------------------------------------------------------------------
// SimplePass -- temporary fallback rendering path.
//
// Draws meshes directly to the swapchain using a traditional vertex + fragment
// pipeline (no mesh shaders, no visibility buffer, no BDA).  The sole purpose
// is to get something visible on screen while the mesh-shader pipeline is
// debugged.
// ---------------------------------------------------------------------------

class SimplePass {
public:
    SimplePass(VulkanDevice& device, GpuAllocator& allocator,
               PipelineManager& pipelines, CommandManager& commands);
    ~SimplePass();

    SimplePass(const SimplePass&)            = delete;
    SimplePass& operator=(const SimplePass&) = delete;
    SimplePass(SimplePass&&)                 = delete;
    SimplePass& operator=(SimplePass&&)      = delete;

    /// Upload vertex/index data for a mesh (positions + normals + indices).
    /// Interleaves into a single VB (vec3 pos + vec3 normal per vertex).
    void uploadMesh(const std::vector<glm::vec3>& positions,
                    const std::vector<glm::vec3>& normals,
                    const std::vector<u32>& indices);

    /// Record a draw into the swapchain image directly.
    /// Handles depth buffer creation, pipeline creation, barriers, and draw.
    void recordPass(VkCommandBuffer cmd,
                    VkImageView swapchainView,
                    VkExtent2D extent,
                    VkFormat swapchainFormat,
                    const glm::mat4& mvp,
                    const glm::vec3& cameraPos);

    /// Destroy and recreate the depth buffer for a new resolution.
    void recreateDepth(VkExtent2D extent);

private:
    struct SimplePushConstants {
        float mvp[16];       // mat4 -- 64 bytes
        float cameraPos[4];  // vec4 -- 16 bytes
        float lightDir[4];   // vec4 -- 16 bytes
    };
    static_assert(sizeof(SimplePushConstants) == 96,
                  "SimplePushConstants must be 96 bytes");

    void ensurePipeline(VkFormat colorFormat, VkFormat depthFormat);
    void ensureDepthBuffer(VkExtent2D extent);

    VulkanDevice&    device_;
    GpuAllocator&    allocator_;
    PipelineManager& pipelines_;
    CommandManager&  commands_;

    VkPipeline       pipeline_       = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;

    AllocatedBuffer  vertexBuffer_{};
    AllocatedBuffer  indexBuffer_{};
    u32              indexCount_ = 0;

    AllocatedImage   depthImage_{};
    VkImageView      depthView_    = VK_NULL_HANDLE;
    VkExtent2D       depthExtent_{};

    bool meshUploaded_    = false;
    bool pipelineCreated_ = false;
};

} // namespace phosphor
