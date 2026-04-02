#include "renderer/simple_pass.h"
#include "rhi/vk_device.h"
#include "rhi/vk_allocator.h"
#include "rhi/vk_pipeline.h"
#include "rhi/vk_commands.h"
#include "rhi/vk_shader.h"
#include "diagnostics/debug_utils.h"
#include "core/log.h"

#include <glm/gtc/type_ptr.hpp>
#include <cstring>

namespace phosphor {

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------

SimplePass::SimplePass(VulkanDevice& device, GpuAllocator& allocator,
                       PipelineManager& pipelines, CommandManager& commands)
    : device_(device), allocator_(allocator),
      pipelines_(pipelines), commands_(commands) {
    LOG_INFO("SimplePass created (temporary fallback renderer)");
}

SimplePass::~SimplePass() {
    VkDevice dev = device_.getDevice();

    if (depthView_ != VK_NULL_HANDLE) {
        allocator_.destroyImageView(depthView_);
        depthView_ = VK_NULL_HANDLE;
    }
    if (depthImage_.image != VK_NULL_HANDLE) {
        allocator_.destroyImage(depthImage_);
    }
    if (vertexBuffer_.buffer != VK_NULL_HANDLE) {
        allocator_.destroyBuffer(vertexBuffer_);
    }
    if (indexBuffer_.buffer != VK_NULL_HANDLE) {
        allocator_.destroyBuffer(indexBuffer_);
    }
    if (pipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(dev, pipeline_, nullptr);
        pipeline_ = VK_NULL_HANDLE;
    }
    if (pipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(dev, pipelineLayout_, nullptr);
        pipelineLayout_ = VK_NULL_HANDLE;
    }
}

// ---------------------------------------------------------------------------
// Mesh upload -- interleave positions + normals into a single buffer
// ---------------------------------------------------------------------------

void SimplePass::uploadMesh(const std::vector<glm::vec3>& positions,
                            const std::vector<glm::vec3>& normals,
                            const std::vector<u32>& indices) {
    if (positions.size() != normals.size()) {
        LOG_ERROR("SimplePass::uploadMesh: position/normal count mismatch (%zu vs %zu)",
                  positions.size(), normals.size());
        return;
    }

    // Interleave: vec3 position + vec3 normal per vertex = 24 bytes per vertex
    struct InterleavedVertex {
        glm::vec3 position;
        glm::vec3 normal;
    };

    u32 vertexCount = static_cast<u32>(positions.size());
    std::vector<InterleavedVertex> vertices(vertexCount);
    for (u32 i = 0; i < vertexCount; ++i) {
        vertices[i].position = positions[i];
        vertices[i].normal   = normals[i];
    }

    VkDeviceSize vbSize = vertexCount * sizeof(InterleavedVertex);
    VkDeviceSize ibSize = indices.size() * sizeof(u32);

    // Create staging buffers, copy, then transfer to device-local via immediateSubmit
    AllocatedBuffer vbStaging = allocator_.createBuffer(
        vbSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_CPU_ONLY);

    AllocatedBuffer ibStaging = allocator_.createBuffer(
        ibSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_CPU_ONLY);

    // Map and fill staging buffers
    void* vbData = allocator_.mapMemory(vbStaging);
    std::memcpy(vbData, vertices.data(), vbSize);
    allocator_.unmapMemory(vbStaging);

    void* ibData = allocator_.mapMemory(ibStaging);
    std::memcpy(ibData, indices.data(), ibSize);
    allocator_.unmapMemory(ibStaging);

    // Destroy old buffers if re-uploading
    if (vertexBuffer_.buffer != VK_NULL_HANDLE) {
        allocator_.destroyBuffer(vertexBuffer_);
    }
    if (indexBuffer_.buffer != VK_NULL_HANDLE) {
        allocator_.destroyBuffer(indexBuffer_);
    }

    // Create device-local buffers
    vertexBuffer_ = allocator_.createBuffer(
        vbSize,
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);

    indexBuffer_ = allocator_.createBuffer(
        ibSize,
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);

    // Copy via immediate submit
    commands_.immediateSubmit([&](VkCommandBuffer cmd) {
        VkBufferCopy vbCopy{};
        vbCopy.size = vbSize;
        vkCmdCopyBuffer(cmd, vbStaging.buffer, vertexBuffer_.buffer, 1, &vbCopy);

        VkBufferCopy ibCopy{};
        ibCopy.size = ibSize;
        vkCmdCopyBuffer(cmd, ibStaging.buffer, indexBuffer_.buffer, 1, &ibCopy);
    });

    // Clean up staging buffers
    allocator_.destroyBuffer(vbStaging);
    allocator_.destroyBuffer(ibStaging);

    indexCount_ = static_cast<u32>(indices.size());
    meshUploaded_ = true;

    LOG_INFO("SimplePass: uploaded mesh -- %u vertices, %u indices (%zu KB VB, %zu KB IB)",
             vertexCount, indexCount_,
             static_cast<size_t>(vbSize / 1024),
             static_cast<size_t>(ibSize / 1024));
}

// ---------------------------------------------------------------------------
// Depth buffer creation
// ---------------------------------------------------------------------------

void SimplePass::ensureDepthBuffer(VkExtent2D extent) {
    if (depthImage_.image != VK_NULL_HANDLE &&
        depthExtent_.width == extent.width &&
        depthExtent_.height == extent.height) {
        return; // already correct size
    }

    // Destroy old
    if (depthView_ != VK_NULL_HANDLE) {
        allocator_.destroyImageView(depthView_);
        depthView_ = VK_NULL_HANDLE;
    }
    if (depthImage_.image != VK_NULL_HANDLE) {
        allocator_.destroyImage(depthImage_);
    }

    VkImageCreateInfo imgInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imgInfo.imageType   = VK_IMAGE_TYPE_2D;
    imgInfo.format      = VK_FORMAT_D32_SFLOAT;
    imgInfo.extent      = {extent.width, extent.height, 1};
    imgInfo.mipLevels   = 1;
    imgInfo.arrayLayers = 1;
    imgInfo.samples     = VK_SAMPLE_COUNT_1_BIT;
    imgInfo.tiling      = VK_IMAGE_TILING_OPTIMAL;
    imgInfo.usage       = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    depthImage_ = allocator_.createImage(imgInfo, VMA_MEMORY_USAGE_GPU_ONLY);
    depthView_  = allocator_.createImageView(
        depthImage_.image, VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT);
    depthExtent_ = extent;

    LOG_INFO("SimplePass: created depth buffer %ux%u", extent.width, extent.height);
}

void SimplePass::recreateDepth(VkExtent2D extent) {
    // Force recreation by zeroing stored extent
    depthExtent_ = {};
    ensureDepthBuffer(extent);
}

// ---------------------------------------------------------------------------
// Pipeline creation -- traditional vertex + fragment graphics pipeline
// ---------------------------------------------------------------------------

void SimplePass::ensurePipeline(VkFormat colorFormat, VkFormat depthFormat) {
    if (pipelineCreated_) return;

    VkDevice dev = device_.getDevice();

    // --- Pipeline layout: push constants only, no descriptor sets ---
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushRange.offset     = 0;
    pushRange.size       = sizeof(SimplePushConstants);

    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges    = &pushRange;
    layoutInfo.setLayoutCount         = 0;
    layoutInfo.pSetLayouts            = nullptr;

    VK_CHECK(vkCreatePipelineLayout(dev, &layoutInfo, nullptr, &pipelineLayout_));

    // --- Load shaders ---
    std::string shaderDir = pipelines_.getShaderDir();
    auto vertShader = ShaderModule::loadFromFile(dev, shaderDir + "/debug/simple.vert.spv");
    auto fragShader = ShaderModule::loadFromFile(dev, shaderDir + "/debug/simple.frag.spv");

    std::array<VkPipelineShaderStageCreateInfo, 2> stages = {
        vertShader.getStageCreateInfo(),
        fragShader.getStageCreateInfo(),
    };

    // --- Vertex input: binding 0, stride = 24 bytes (vec3 pos + vec3 normal) ---
    VkVertexInputBindingDescription bindingDesc{};
    bindingDesc.binding   = 0;
    bindingDesc.stride    = 24; // sizeof(vec3) + sizeof(vec3)
    bindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    std::array<VkVertexInputAttributeDescription, 2> attrDescs{};
    // location 0: position (vec3 at offset 0)
    attrDescs[0].location = 0;
    attrDescs[0].binding  = 0;
    attrDescs[0].format   = VK_FORMAT_R32G32B32_SFLOAT;
    attrDescs[0].offset   = 0;
    // location 1: normal (vec3 at offset 12)
    attrDescs[1].location = 1;
    attrDescs[1].binding  = 0;
    attrDescs[1].format   = VK_FORMAT_R32G32B32_SFLOAT;
    attrDescs[1].offset   = 12;

    VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vertexInput.vertexBindingDescriptionCount   = 1;
    vertexInput.pVertexBindingDescriptions      = &bindingDesc;
    vertexInput.vertexAttributeDescriptionCount = static_cast<u32>(attrDescs.size());
    vertexInput.pVertexAttributeDescriptions    = attrDescs.data();

    // --- Input assembly ---
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    inputAssembly.topology               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    // --- Rasterization ---
    VkPipelineRasterizationStateCreateInfo rasterInfo{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
    rasterInfo.cullMode    = VK_CULL_MODE_BACK_BIT;
    rasterInfo.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterInfo.lineWidth   = 1.0f;

    // --- Multisampling (no MSAA) ---
    VkPipelineMultisampleStateCreateInfo msInfo{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    msInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // --- Depth stencil ---
    VkPipelineDepthStencilStateCreateInfo depthStencil{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depthStencil.depthTestEnable  = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp   = VK_COMPARE_OP_LESS;

    // --- Color blend (no blending, just write) ---
    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                                   | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo blendInfo{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blendInfo.attachmentCount = 1;
    blendInfo.pAttachments    = &blendAttachment;

    // --- Dynamic state: viewport + scissor ---
    std::array<VkDynamicState, 2> dynStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };
    VkPipelineDynamicStateCreateInfo dynInfo{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynInfo.dynamicStateCount = static_cast<u32>(dynStates.size());
    dynInfo.pDynamicStates    = dynStates.data();

    // --- Viewport state (dynamic, count only) ---
    VkPipelineViewportStateCreateInfo viewportInfo{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewportInfo.viewportCount = 1;
    viewportInfo.scissorCount  = 1;

    // --- Dynamic rendering (Vulkan 1.3, no VkRenderPass) ---
    VkPipelineRenderingCreateInfo renderingInfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    renderingInfo.colorAttachmentCount    = 1;
    renderingInfo.pColorAttachmentFormats = &colorFormat;
    renderingInfo.depthAttachmentFormat   = depthFormat;

    // --- Create the pipeline ---
    VkGraphicsPipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipelineInfo.pNext               = &renderingInfo;
    pipelineInfo.stageCount          = static_cast<u32>(stages.size());
    pipelineInfo.pStages             = stages.data();
    pipelineInfo.pVertexInputState   = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pRasterizationState = &rasterInfo;
    pipelineInfo.pMultisampleState   = &msInfo;
    pipelineInfo.pDepthStencilState  = &depthStencil;
    pipelineInfo.pColorBlendState    = &blendInfo;
    pipelineInfo.pDynamicState       = &dynInfo;
    pipelineInfo.pViewportState      = &viewportInfo;
    pipelineInfo.layout              = pipelineLayout_;

    VK_CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline_));

    pipelineCreated_ = true;
    LOG_INFO("SimplePass: created traditional graphics pipeline (vert+frag)");
}

// ---------------------------------------------------------------------------
// Record the render pass
// ---------------------------------------------------------------------------

void SimplePass::recordPass(VkCommandBuffer cmd,
                            VkImageView swapchainView,
                            VkExtent2D extent,
                            VkFormat swapchainFormat,
                            const glm::mat4& mvp,
                            const glm::vec3& cameraPos) {
    if (!meshUploaded_) {
        LOG_WARN("SimplePass::recordPass called but no mesh uploaded");
        return;
    }

    PHOSPHOR_GPU_LABEL(cmd, "SimplePass");

    // Ensure pipeline and depth buffer are ready
    ensurePipeline(swapchainFormat, VK_FORMAT_D32_SFLOAT);
    ensureDepthBuffer(extent);

    // --- Transition depth image to DEPTH_ATTACHMENT_OPTIMAL ---
    {
        VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        barrier.srcStageMask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        barrier.srcAccessMask = 0;
        barrier.dstStageMask  = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        barrier.oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout     = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        barrier.image         = depthImage_.image;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};

        VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers    = &barrier;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    // --- Begin dynamic rendering ---
    VkRenderingAttachmentInfo colorAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    colorAttachment.imageView   = swapchainView;
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue.color = {{0.02f, 0.02f, 0.03f, 1.0f}}; // near-black

    VkRenderingAttachmentInfo depthAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    depthAttachment.imageView   = depthView_;
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthAttachment.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp     = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.clearValue.depthStencil = {1.0f, 0};

    VkRenderingInfo renderInfo{VK_STRUCTURE_TYPE_RENDERING_INFO};
    renderInfo.renderArea           = {{0, 0}, extent};
    renderInfo.layerCount           = 1;
    renderInfo.colorAttachmentCount = 1;
    renderInfo.pColorAttachments    = &colorAttachment;
    renderInfo.pDepthAttachment     = &depthAttachment;

    vkCmdBeginRendering(cmd, &renderInfo);

    // --- Bind pipeline ---
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);

    // --- Set viewport and scissor ---
    VkViewport viewport{};
    viewport.x        = 0.0f;
    viewport.y        = 0.0f;
    viewport.width    = static_cast<float>(extent.width);
    viewport.height   = static_cast<float>(extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = extent;
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // --- Push constants ---
    glm::vec3 lightDir = glm::normalize(glm::vec3(0.0f, 1.0f, 1.0f));

    SimplePushConstants pc{};
    std::memcpy(pc.mvp, glm::value_ptr(mvp), sizeof(float) * 16);
    pc.cameraPos[0] = cameraPos.x;
    pc.cameraPos[1] = cameraPos.y;
    pc.cameraPos[2] = cameraPos.z;
    pc.cameraPos[3] = 0.0f;
    pc.lightDir[0] = lightDir.x;
    pc.lightDir[1] = lightDir.y;
    pc.lightDir[2] = lightDir.z;
    pc.lightDir[3] = 0.0f;

    vkCmdPushConstants(cmd, pipelineLayout_,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(SimplePushConstants), &pc);

    // --- Bind vertex and index buffers ---
    VkBuffer vbBuffers[] = {vertexBuffer_.buffer};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(cmd, 0, 1, vbBuffers, offsets);
    vkCmdBindIndexBuffer(cmd, indexBuffer_.buffer, 0, VK_INDEX_TYPE_UINT32);

    // --- Draw ---
    vkCmdDrawIndexed(cmd, indexCount_, 1, 0, 0, 0);

    vkCmdEndRendering(cmd);
}

} // namespace phosphor
