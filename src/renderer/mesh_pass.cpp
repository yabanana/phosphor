#include "renderer/mesh_pass.h"
#include "renderer/visibility_buffer.h"
#include "renderer/push_constants.h"
#include "renderer/gpu_scene.h"
#include "rhi/vk_device.h"
#include "rhi/vk_pipeline.h"
#include "rhi/vk_descriptors.h"
#include "scene/camera.h"
#include "diagnostics/debug_utils.h"
#include "core/log.h"

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <cstring>

namespace phosphor {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

MeshPass::MeshPass(VulkanDevice& device, PipelineManager& pipelines,
                   BindlessDescriptorManager& descriptors)
    : device_(device), pipelines_(pipelines), descriptors_(descriptors) {
}

// ---------------------------------------------------------------------------
// Lazy pipeline creation
// ---------------------------------------------------------------------------

void MeshPass::ensureInitialized() {
    if (initialized_) {
        return;
    }

    MeshPipelineDesc desc{};
    desc.taskShaderPath = "mesh/meshlet.task.spv";
    desc.meshShaderPath = "mesh/meshlet.mesh.spv";
    desc.fragShaderPath = "visibility/visibility.frag.spv";
    desc.colorFormat    = VK_FORMAT_R32_UINT;
    desc.depthFormat    = VK_FORMAT_D32_SFLOAT;
    desc.depthWrite     = true;
    desc.depthCompare   = VK_COMPARE_OP_LESS;
    desc.cullMode       = VK_CULL_MODE_BACK_BIT;
    desc.colorAttachmentCount = 1;

    meshPipeline_ = pipelines_.createMeshPipeline(desc);
    initialized_  = true;
    LOG_INFO("MeshPass pipeline initialized");
}

// ---------------------------------------------------------------------------
// Record the mesh shading / visibility-buffer fill pass
// ---------------------------------------------------------------------------

void MeshPass::recordPass(VkCommandBuffer cmd, GpuScene& scene,
                          const VisibilityBuffer& visBuf, const Camera& camera,
                          u32 frameIndex, float exposure) {
    ensureInitialized();

    PHOSPHOR_GPU_LABEL(cmd, "MeshPass");

    VkExtent2D extent = visBuf.getExtent();

    // --- Transition vis buffer to COLOR_ATTACHMENT_OPTIMAL ---
    {
        VkImageMemoryBarrier2 barriers[2]{};

        // Visibility image: UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL
        barriers[0].sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barriers[0].srcStageMask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        barriers[0].srcAccessMask = 0;
        barriers[0].dstStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        barriers[0].dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        barriers[0].oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
        barriers[0].newLayout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barriers[0].image         = visBuf.getVisImage();
        barriers[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        // Depth image: UNDEFINED -> DEPTH_ATTACHMENT_OPTIMAL
        barriers[1].sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barriers[1].srcStageMask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        barriers[1].srcAccessMask = 0;
        barriers[1].dstStageMask  = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT
                                  | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        barriers[1].dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        barriers[1].oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
        barriers[1].newLayout     = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        barriers[1].image         = visBuf.getDepthImage();
        barriers[1].subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};

        VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dep.imageMemoryBarrierCount = 2;
        dep.pImageMemoryBarriers    = barriers;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    // --- Begin dynamic rendering ---
    VkRenderingAttachmentInfo colorAttach{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    colorAttach.imageView   = visBuf.getVisView();
    colorAttach.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttach.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttach.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttach.clearValue.color = {{0, 0, 0, 0}};  // 0 = "no triangle"

    VkRenderingAttachmentInfo depthAttach{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    depthAttach.imageView   = visBuf.getDepthView();
    depthAttach.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthAttach.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttach.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttach.clearValue.depthStencil = {1.0f, 0};

    VkRenderingInfo renderInfo{VK_STRUCTURE_TYPE_RENDERING_INFO};
    renderInfo.renderArea          = {{0, 0}, extent};
    renderInfo.layerCount          = 1;
    renderInfo.colorAttachmentCount = 1;
    renderInfo.pColorAttachments   = &colorAttach;
    renderInfo.pDepthAttachment    = &depthAttach;

    vkCmdBeginRendering(cmd, &renderInfo);

    // --- Viewport and scissor ---
    VkViewport viewport{};
    viewport.x        = 0.0f;
    viewport.y        = 0.0f;
    viewport.width    = static_cast<float>(extent.width);
    viewport.height   = static_cast<float>(extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{{0, 0}, extent};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // --- Bind pipeline and bindless descriptors ---
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipeline_);
    descriptors_.bindToCommandBuffer(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);

    // --- Upload SceneGlobals to a GPU buffer so shaders can read via BDA ---
    VkDeviceAddress sceneGlobalsAddr = scene.uploadSceneGlobalsBuffer();
    SceneGlobals globals = scene.getSceneGlobals();

    // --- Fill push constants ---
    PushConstants pc{};
    std::memcpy(pc.viewProjection, glm::value_ptr(camera.getViewProjection()),
                sizeof(pc.viewProjection));

    glm::vec3 camPos = camera.getPosition();
    pc.cameraPosition[0] = camPos.x;
    pc.cameraPosition[1] = camPos.y;
    pc.cameraPosition[2] = camPos.z;
    pc.cameraPosition[3] = 0.0f;  // w = time (unused for now)

    pc.sceneGlobalsAddress  = sceneGlobalsAddr;              // BDA to the SceneGlobals GPU buffer
    pc.vertexBufferAddress  = globals.vertexBufferAddress;
    pc.meshletBufferAddress = globals.meshletBufferAddress;
    pc.resolution[0]        = extent.width;
    pc.resolution[1]        = extent.height;
    pc.frameIndex           = frameIndex;
    pc.lightCount           = globals.lightCount;
    pc.exposure             = exposure;
    pc.debugMode            = 0;

    vkCmdPushConstants(cmd, pipelines_.getPipelineLayout(),
                       VK_SHADER_STAGE_ALL, 0, sizeof(PushConstants), &pc);

    // --- Draw all meshlets via mesh shader ---
    // The task shader uses local_size_x = 32, so each workgroup handles 32
    // meshlet candidates.  We need ceil(meshletCount / 32) task workgroups.
    u32 meshletCount = scene.getMeshletTotalCount();
    if (meshletCount > 0) {
        static auto vkCmdDrawMeshTasksEXT_ =
            reinterpret_cast<PFN_vkCmdDrawMeshTasksEXT>(
                vkGetDeviceProcAddr(device_.getDevice(), "vkCmdDrawMeshTasksEXT"));
        u32 taskGroupCount = (meshletCount + 31) / 32;
        vkCmdDrawMeshTasksEXT_(cmd, taskGroupCount, 1, 1);
    }

    vkCmdEndRendering(cmd);
}

} // namespace phosphor
