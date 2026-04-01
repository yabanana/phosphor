#include "renderer/shadow_pass.h"
#include "renderer/push_constants.h"
#include "renderer/gpu_scene.h"
#include "rhi/vk_device.h"
#include "rhi/vk_pipeline.h"
#include "rhi/vk_shader.h"
#include "rhi/vk_descriptors.h"
#include "scene/camera.h"
#include "diagnostics/debug_utils.h"
#include "core/log.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

namespace phosphor {

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------

ShadowPass::ShadowPass(VulkanDevice& device, GpuAllocator& allocator,
                        PipelineManager& pipelines, BindlessDescriptorManager& descriptors)
    : device_(device), allocator_(allocator), pipelines_(pipelines),
      descriptors_(descriptors) {
    createResources();
    LOG_INFO("ShadowPass created (%ux%u, %u cascades)", SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, CASCADE_COUNT);
}

ShadowPass::~ShadowPass() {
    device_.waitIdle();
    destroyResources();

    VkDevice dev = device_.getDevice();
    if (shadowPipeline_) vkDestroyPipeline(dev, shadowPipeline_, nullptr);
}

// ---------------------------------------------------------------------------
// Create a single-layer image view for depth attachment
// ---------------------------------------------------------------------------

VkImageView ShadowPass::createLayerView(VkImage image, VkFormat format,
                                          VkImageAspectFlags aspect, u32 layer) {
    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image    = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format   = format;
    viewInfo.subresourceRange.aspectMask     = aspect;
    viewInfo.subresourceRange.baseMipLevel   = 0;
    viewInfo.subresourceRange.levelCount     = 1;
    viewInfo.subresourceRange.baseArrayLayer = layer;
    viewInfo.subresourceRange.layerCount     = 1;

    VkImageView view = VK_NULL_HANDLE;
    VK_CHECK(vkCreateImageView(device_.getDevice(), &viewInfo, nullptr, &view));
    return view;
}

// ---------------------------------------------------------------------------
// Resource creation
// ---------------------------------------------------------------------------

void ShadowPass::createResources() {
    // --- Shadow map: D32_SFLOAT, 2048x2048, 4 layers ---
    {
        VkImageCreateInfo imgInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        imgInfo.imageType   = VK_IMAGE_TYPE_2D;
        imgInfo.format      = VK_FORMAT_D32_SFLOAT;
        imgInfo.extent      = {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 1};
        imgInfo.mipLevels   = 1;
        imgInfo.arrayLayers = CASCADE_COUNT;
        imgInfo.samples     = VK_SAMPLE_COUNT_1_BIT;
        imgInfo.tiling      = VK_IMAGE_TILING_OPTIMAL;
        imgInfo.usage       = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                            | VK_IMAGE_USAGE_SAMPLED_BIT;
        imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        shadowMap_ = allocator_.createImage(imgInfo, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    }

    // Array view (all 4 layers) for sampling in the lighting pass
    arrayView_ = allocator_.createImageView(
        shadowMap_.image, VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT,
        1, VK_IMAGE_VIEW_TYPE_2D_ARRAY, CASCADE_COUNT);

    // Per-cascade views (single layer each) for depth attachment
    for (u32 i = 0; i < CASCADE_COUNT; ++i) {
        cascadeViews_[i] = createLayerView(
            shadowMap_.image, VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT, i);
    }
}

void ShadowPass::destroyResources() {
    VkDevice dev = device_.getDevice();

    for (u32 i = 0; i < CASCADE_COUNT; ++i) {
        if (cascadeViews_[i] != VK_NULL_HANDLE) {
            vkDestroyImageView(dev, cascadeViews_[i], nullptr);
            cascadeViews_[i] = VK_NULL_HANDLE;
        }
    }

    allocator_.destroyImageView(arrayView_);
    arrayView_ = VK_NULL_HANDLE;

    allocator_.destroyImage(shadowMap_);
}

// ---------------------------------------------------------------------------
// Lazy pipeline creation
// ---------------------------------------------------------------------------

void ShadowPass::ensurePipelineCreated() {
    if (initialized_) {
        return;
    }

    // Build the shadow pipeline manually because PipelineManager::createMeshPipeline
    // always expects a fragment shader.  Shadow passes are depth-only (no frag stage).
    VkDevice dev = device_.getDevice();
    std::string shaderDir = pipelines_.getShaderDir();

    auto taskShader = ShaderModule::loadFromFile(dev, shaderDir + "/mesh/meshlet_shadow.task.spv");
    auto meshShader = ShaderModule::loadFromFile(dev, shaderDir + "/mesh/meshlet_shadow.mesh.spv");

    // Only 2 stages: task + mesh (no fragment shader for depth-only)
    std::array<VkPipelineShaderStageCreateInfo, 2> stages = {
        taskShader.getStageCreateInfo(),
        meshShader.getStageCreateInfo(),
    };

    // Rasterization
    VkPipelineRasterizationStateCreateInfo rasterInfo{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
    rasterInfo.cullMode    = VK_CULL_MODE_NONE;  // render both faces for shadows
    rasterInfo.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterInfo.lineWidth   = 1.0f;

    // No multisampling
    VkPipelineMultisampleStateCreateInfo msInfo{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    msInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // Depth stencil
    VkPipelineDepthStencilStateCreateInfo depthInfo{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depthInfo.depthTestEnable  = VK_TRUE;
    depthInfo.depthWriteEnable = VK_TRUE;
    depthInfo.depthCompareOp   = VK_COMPARE_OP_LESS;

    // No color blend (depth-only)
    VkPipelineColorBlendStateCreateInfo blendInfo{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blendInfo.attachmentCount = 0;
    blendInfo.pAttachments    = nullptr;

    // Dynamic state
    std::array<VkDynamicState, 2> dynStates = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynInfo{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynInfo.dynamicStateCount = static_cast<u32>(dynStates.size());
    dynInfo.pDynamicStates    = dynStates.data();

    // Viewport state (dynamic, count only)
    VkPipelineViewportStateCreateInfo viewportInfo{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewportInfo.viewportCount = 1;
    viewportInfo.scissorCount  = 1;

    // Dynamic rendering: depth-only, no color attachments
    VkPipelineRenderingCreateInfo renderingInfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    renderingInfo.colorAttachmentCount    = 0;
    renderingInfo.pColorAttachmentFormats = nullptr;
    renderingInfo.depthAttachmentFormat   = VK_FORMAT_D32_SFLOAT;

    // Vertex input / input assembly (unused with mesh shaders, but Vulkan requires them)
    VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkGraphicsPipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipelineInfo.pNext               = &renderingInfo;
    pipelineInfo.stageCount          = static_cast<u32>(stages.size());
    pipelineInfo.pStages             = stages.data();
    pipelineInfo.pRasterizationState = &rasterInfo;
    pipelineInfo.pMultisampleState   = &msInfo;
    pipelineInfo.pDepthStencilState  = &depthInfo;
    pipelineInfo.pColorBlendState    = &blendInfo;
    pipelineInfo.pDynamicState       = &dynInfo;
    pipelineInfo.pViewportState      = &viewportInfo;
    pipelineInfo.pVertexInputState   = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.layout              = pipelines_.getPipelineLayout();

    VK_CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &shadowPipeline_));

    initialized_ = true;
    LOG_INFO("ShadowPass pipeline initialized (depth-only, task+mesh)");
}

// ---------------------------------------------------------------------------
// Compute cascade splits and light-space VP matrices (PSSM)
// ---------------------------------------------------------------------------

void ShadowPass::computeCascades(const Camera& camera, glm::vec3 lightDir, float sceneRadius) {
    const float nearClip = camera.getNear();
    const float farClip  = camera.getFar();
    const float fovY     = camera.getFovY();
    const float aspect   = camera.getAspect();

    // --- Compute PSSM split distances ---
    // Practical Split Scheme: blend between logarithmic and uniform splits.
    // C_log(i)  = near * (far/near)^(i/N)
    // C_uni(i)  = near + (far - near) * (i/N)
    // C(i) = lambda * C_log(i) + (1 - lambda) * C_uni(i)

    float splits[CASCADE_COUNT + 1];
    splits[0] = nearClip;

    for (u32 i = 1; i <= CASCADE_COUNT; ++i) {
        float p = static_cast<float>(i) / static_cast<float>(CASCADE_COUNT);

        float logSplit = nearClip * std::pow(farClip / nearClip, p);
        float uniSplit = nearClip + (farClip - nearClip) * p;
        float split    = SPLIT_LAMBDA * logSplit + (1.0f - SPLIT_LAMBDA) * uniSplit;

        splits[i] = split;
    }

    // Store split distances (view-space Z) for use in the lighting shader
    for (u32 i = 0; i < CASCADE_COUNT; ++i) {
        cascadeSplits_[i] = splits[i + 1];
    }

    // --- Compute per-cascade light VP matrices ---
    const glm::mat4 invViewProj = glm::inverse(camera.getViewProjection());
    const glm::vec3 lightDirN   = glm::normalize(lightDir);

    for (u32 cascade = 0; cascade < CASCADE_COUNT; ++cascade) {
        // Compute the 8 corners of this cascade's frustum slice in world space
        float nearSplit = splits[cascade];
        float farSplit  = splits[cascade + 1];

        // Map split distances to NDC Z range [0, 1] (Vulkan)
        // For a perspective projection: z_ndc = (far * z - far * near) / (z * (far - near))
        // But it's simpler to recompute the sub-frustum directly.
        float nearNDC = 0.0f;
        float farNDC  = 1.0f;

        // Build a sub-projection for this cascade's near/far range
        // and extract frustum corners from it.
        //
        // Actually, the most robust approach is to compute frustum corners
        // in NDC and unproject them.  We compute fractional NDC Z for
        // nearSplit and farSplit within the original projection.

        // For Vulkan reversed-Z infinite far plane, the Z values are different.
        // Simplified approach: compute corners from the camera frustum at
        // specific depth values.

        // Frustum corners in NDC (Vulkan Z in [0, 1]):
        // Near plane: z=0 for standard, z=1 for reversed-Z
        // Far plane:  z=1 for standard, z=0 for reversed-Z
        // We compute the clip-space Z for each split distance.

        // Use a perspective projection for the split range
        float tanHalfFovY = std::tan(fovY * 0.5f);
        float nearH = nearSplit * tanHalfFovY;
        float nearW = nearH * aspect;
        float farH  = farSplit * tanHalfFovY;
        float farW  = farH * aspect;

        const glm::mat4& view = camera.getView();
        glm::vec3 camPos   = camera.getPosition();
        glm::vec3 camFront = camera.getFront();
        glm::vec3 camRight = camera.getRight();
        glm::vec3 camUp    = camera.getUp();

        glm::vec3 nearCenter = camPos + camFront * nearSplit;
        glm::vec3 farCenter  = camPos + camFront * farSplit;

        // 8 frustum corners
        std::array<glm::vec3, 8> corners;
        corners[0] = nearCenter - camUp * nearH - camRight * nearW; // near bottom-left
        corners[1] = nearCenter - camUp * nearH + camRight * nearW; // near bottom-right
        corners[2] = nearCenter + camUp * nearH - camRight * nearW; // near top-left
        corners[3] = nearCenter + camUp * nearH + camRight * nearW; // near top-right
        corners[4] = farCenter  - camUp * farH  - camRight * farW;  // far bottom-left
        corners[5] = farCenter  - camUp * farH  + camRight * farW;  // far bottom-right
        corners[6] = farCenter  + camUp * farH  - camRight * farW;  // far top-left
        corners[7] = farCenter  + camUp * farH  + camRight * farW;  // far top-right

        // Compute the centroid of the frustum slice
        glm::vec3 frustumCenter{0.0f};
        for (const auto& c : corners) {
            frustumCenter += c;
        }
        frustumCenter /= 8.0f;

        // Light view matrix: looking from above along lightDir towards the frustum center
        glm::mat4 lightView = glm::lookAt(
            frustumCenter - lightDirN * sceneRadius,
            frustumCenter,
            glm::vec3(0.0f, 1.0f, 0.0f));

        // If the light direction is nearly vertical, use a different up vector
        if (std::abs(glm::dot(lightDirN, glm::vec3(0.0f, 1.0f, 0.0f))) > 0.99f) {
            lightView = glm::lookAt(
                frustumCenter - lightDirN * sceneRadius,
                frustumCenter,
                glm::vec3(0.0f, 0.0f, 1.0f));
        }

        // Transform frustum corners to light view space and find AABB
        float minX =  std::numeric_limits<float>::max();
        float maxX = -std::numeric_limits<float>::max();
        float minY =  std::numeric_limits<float>::max();
        float maxY = -std::numeric_limits<float>::max();
        float minZ =  std::numeric_limits<float>::max();
        float maxZ = -std::numeric_limits<float>::max();

        for (const auto& corner : corners) {
            glm::vec4 lsCorner = lightView * glm::vec4(corner, 1.0f);
            minX = std::min(minX, lsCorner.x);
            maxX = std::max(maxX, lsCorner.x);
            minY = std::min(minY, lsCorner.y);
            maxY = std::max(maxY, lsCorner.y);
            minZ = std::min(minZ, lsCorner.z);
            maxZ = std::max(maxZ, lsCorner.z);
        }

        // Extend the Z range to include objects behind the camera that cast shadows
        float zRange = maxZ - minZ;
        minZ -= zRange * 0.5f;

        // Round the ortho extents to texel boundaries to reduce shadow shimmering
        float worldTexelSize = std::max(maxX - minX, maxY - minY) / static_cast<float>(SHADOW_MAP_SIZE);
        minX = std::floor(minX / worldTexelSize) * worldTexelSize;
        maxX = std::ceil(maxX / worldTexelSize) * worldTexelSize;
        minY = std::floor(minY / worldTexelSize) * worldTexelSize;
        maxY = std::ceil(maxY / worldTexelSize) * worldTexelSize;

        // Orthographic projection for this cascade
        // Vulkan NDC: X [-1,1], Y [-1,1], Z [0,1]
        glm::mat4 lightProj = glm::ortho(minX, maxX, minY, maxY, -maxZ, -minZ);

        // Vulkan clip-space Y is inverted compared to OpenGL
        lightProj[1][1] *= -1.0f;

        cascadeVP_[cascade] = lightProj * lightView;
    }
}

// ---------------------------------------------------------------------------
// Record shadow rendering for all cascades
// ---------------------------------------------------------------------------

void ShadowPass::recordCascades(VkCommandBuffer cmd, GpuScene& scene) {
    ensurePipelineCreated();

    if (shadowPipeline_ == VK_NULL_HANDLE) {
        return;
    }

    PHOSPHOR_GPU_LABEL(cmd, "ShadowPass");

    // --- Transition shadow map to DEPTH_ATTACHMENT_OPTIMAL ---
    {
        VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        barrier.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier.srcStageMask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        barrier.srcAccessMask = 0;
        barrier.dstStageMask  = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT
                              | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        barrier.oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout     = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        barrier.image         = shadowMap_.image;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, CASCADE_COUNT};

        VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers    = &barrier;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    // Load the mesh shader dispatch function pointer
    static auto vkCmdDrawMeshTasksEXT_ =
        reinterpret_cast<PFN_vkCmdDrawMeshTasksEXT>(
            vkGetDeviceProcAddr(device_.getDevice(), "vkCmdDrawMeshTasksEXT"));

    SceneGlobals globals = scene.getSceneGlobals();
    u32 meshletCount = scene.getMeshletTotalCount();

    // --- Render each cascade ---
    for (u32 cascade = 0; cascade < CASCADE_COUNT; ++cascade) {
        // Begin dynamic rendering with this cascade's depth attachment
        VkRenderingAttachmentInfo depthAttach{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        depthAttach.imageView   = cascadeViews_[cascade];
        depthAttach.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depthAttach.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttach.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
        depthAttach.clearValue.depthStencil = {1.0f, 0};

        VkRenderingInfo renderInfo{VK_STRUCTURE_TYPE_RENDERING_INFO};
        renderInfo.renderArea          = {{0, 0}, {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE}};
        renderInfo.layerCount          = 1;
        renderInfo.colorAttachmentCount = 0;         // depth-only
        renderInfo.pColorAttachments   = nullptr;
        renderInfo.pDepthAttachment    = &depthAttach;

        vkCmdBeginRendering(cmd, &renderInfo);

        // Viewport and scissor
        VkViewport viewport{};
        viewport.x        = 0.0f;
        viewport.y        = 0.0f;
        viewport.width    = static_cast<float>(SHADOW_MAP_SIZE);
        viewport.height   = static_cast<float>(SHADOW_MAP_SIZE);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &viewport);

        VkRect2D scissor{{0, 0}, {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE}};
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        // Bind pipeline and bindless descriptors
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipeline_);
        descriptors_.bindToCommandBuffer(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);

        // Fill push constants with the cascade's light VP
        PushConstants pc{};
        std::memcpy(pc.viewProjection, glm::value_ptr(cascadeVP_[cascade]),
                    sizeof(pc.viewProjection));

        // Camera position is unused in shadow shaders, but fill it for consistency
        pc.cameraPosition[0] = 0.0f;
        pc.cameraPosition[1] = 0.0f;
        pc.cameraPosition[2] = 0.0f;
        pc.cameraPosition[3] = 0.0f;

        pc.sceneGlobalsAddress  = globals.vertexBufferAddress;
        pc.vertexBufferAddress  = globals.vertexBufferAddress;
        pc.meshletBufferAddress = globals.meshletBufferAddress;
        pc.resolution[0]        = SHADOW_MAP_SIZE;
        pc.resolution[1]        = SHADOW_MAP_SIZE;
        pc.frameIndex           = 0;
        pc.lightCount           = 0;
        pc.exposure             = 1.0f;
        pc.debugMode            = cascade;  // encode cascade index for debugging

        vkCmdPushConstants(cmd, pipelines_.getPipelineLayout(),
                           VK_SHADER_STAGE_ALL, 0, sizeof(PushConstants), &pc);

        // Dispatch mesh shader draw for all meshlets
        if (meshletCount > 0 && vkCmdDrawMeshTasksEXT_ != nullptr) {
            // Each task shader workgroup processes 32 meshlets
            u32 taskWorkgroups = (meshletCount + 31) / 32;
            vkCmdDrawMeshTasksEXT_(cmd, taskWorkgroups, 1, 1);
        }

        vkCmdEndRendering(cmd);
    }

    // --- Transition shadow map to SHADER_READ_ONLY_OPTIMAL for sampling ---
    {
        VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        barrier.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier.srcStageMask  = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT
                              | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        barrier.dstStageMask  = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
                              | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        barrier.oldLayout     = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        barrier.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.image         = shadowMap_.image;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, CASCADE_COUNT};

        VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers    = &barrier;
        vkCmdPipelineBarrier2(cmd, &dep);
    }
}

} // namespace phosphor
