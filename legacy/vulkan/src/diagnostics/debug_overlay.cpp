#include "diagnostics/debug_overlay.h"
#include "rhi/vk_device.h"
#include "rhi/vk_pipeline.h"
#include "core/log.h"

#include <array>

namespace phosphor {

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------

DebugOverlay::DebugOverlay(VulkanDevice& device, PipelineManager& pipelines)
    : device_(device)
    , pipelinesRef_(pipelines)
{
}

DebugOverlay::~DebugOverlay() {
    VkDevice dev = device_.getDevice();
    for (u32 i = 0; i < MODE_COUNT; ++i) {
        if (pipelines_[i] != VK_NULL_HANDLE) {
            vkDestroyPipeline(dev, pipelines_[i], nullptr);
            pipelines_[i] = VK_NULL_HANDLE;
        }
    }
}

// ---------------------------------------------------------------------------
// Lazy pipeline creation
// ---------------------------------------------------------------------------

void DebugOverlay::createPipelines() {
    if (initialized_) {
        return;
    }

    VkDevice dev = device_.getDevice();
    std::string shaderDir = pipelinesRef_.getShaderDir();

    // Load shaders
    auto loadShader = [&](const std::string& path) -> VkShaderModule {
        // Read SPIR-V file
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) {
            LOG_ERROR("Failed to open shader: %s", path.c_str());
            return VK_NULL_HANDLE;
        }
        fseek(f, 0, SEEK_END);
        long size = ftell(f);
        fseek(f, 0, SEEK_SET);
        std::vector<u32> code(static_cast<size_t>(size) / sizeof(u32));
        fread(code.data(), 1, static_cast<size_t>(size), f);
        fclose(f);

        VkShaderModuleCreateInfo ci{};
        ci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = static_cast<size_t>(size);
        ci.pCode    = code.data();

        VkShaderModule module = VK_NULL_HANDLE;
        VK_CHECK(vkCreateShaderModule(dev, &ci, nullptr, &module));
        return module;
    };

    VkShaderModule vertModule = loadShader(shaderDir + "/debug/debug_overlay.vert.spv");
    VkShaderModule fragModule = loadShader(shaderDir + "/debug/debug_overlay.frag.spv");

    if (vertModule == VK_NULL_HANDLE || fragModule == VK_NULL_HANDLE) {
        LOG_ERROR("DebugOverlay: failed to load shaders, overlay disabled");
        if (vertModule) vkDestroyShaderModule(dev, vertModule, nullptr);
        if (fragModule) vkDestroyShaderModule(dev, fragModule, nullptr);
        initialized_ = true; // prevent re-attempts
        return;
    }

    // Create one pipeline per mode using specialization constants
    for (u32 mode = 0; mode < MODE_COUNT; ++mode) {
        // Specialization constant for DEBUG_MODE (constant_id = 0)
        VkSpecializationMapEntry specEntry{};
        specEntry.constantID = 0;
        specEntry.offset     = 0;
        specEntry.size       = sizeof(u32);

        VkSpecializationInfo specInfo{};
        specInfo.mapEntryCount = 1;
        specInfo.pMapEntries   = &specEntry;
        specInfo.dataSize      = sizeof(u32);
        specInfo.pData         = &mode;

        std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
        stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vertModule;
        stages[0].pName  = "main";

        stages[1].sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage               = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module              = fragModule;
        stages[1].pName               = "main";
        stages[1].pSpecializationInfo = &specInfo;

        // No vertex input (fullscreen triangle generated in the vertex shader)
        VkPipelineVertexInputStateCreateInfo vertexInput{};
        vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        // Dynamic viewport and scissor
        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount  = 1;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.cullMode    = VK_CULL_MODE_NONE;
        rasterizer.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.lineWidth   = 1.0f;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable  = VK_FALSE;
        depthStencil.depthWriteEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.blendEnable    = VK_FALSE;
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT
                                            | VK_COLOR_COMPONENT_G_BIT
                                            | VK_COLOR_COMPONENT_B_BIT
                                            | VK_COLOR_COMPONENT_A_BIT;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments    = &colorBlendAttachment;

        std::array<VkDynamicState, 2> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<u32>(dynamicStates.size());
        dynamicState.pDynamicStates    = dynamicStates.data();

        // Dynamic rendering format info
        VkFormat colorFormat = VK_FORMAT_B8G8R8A8_UNORM; // swapchain format
        VkPipelineRenderingCreateInfo renderingInfo{};
        renderingInfo.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        renderingInfo.colorAttachmentCount    = 1;
        renderingInfo.pColorAttachmentFormats = &colorFormat;

        VkGraphicsPipelineCreateInfo pipelineCI{};
        pipelineCI.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineCI.pNext               = &renderingInfo;
        pipelineCI.stageCount          = static_cast<u32>(stages.size());
        pipelineCI.pStages             = stages.data();
        pipelineCI.pVertexInputState   = &vertexInput;
        pipelineCI.pInputAssemblyState = &inputAssembly;
        pipelineCI.pViewportState      = &viewportState;
        pipelineCI.pRasterizationState = &rasterizer;
        pipelineCI.pMultisampleState   = &multisampling;
        pipelineCI.pDepthStencilState  = &depthStencil;
        pipelineCI.pColorBlendState    = &colorBlending;
        pipelineCI.pDynamicState       = &dynamicState;
        pipelineCI.layout              = pipelinesRef_.getPipelineLayout();
        pipelineCI.renderPass          = VK_NULL_HANDLE; // dynamic rendering
        pipelineCI.subpass             = 0;

        VK_CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &pipelineCI,
                                           nullptr, &pipelines_[mode]));
    }

    vkDestroyShaderModule(dev, vertModule, nullptr);
    vkDestroyShaderModule(dev, fragModule, nullptr);

    initialized_ = true;
    LOG_INFO("DebugOverlay: %u pipelines created", MODE_COUNT);
}

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------

void DebugOverlay::render(VkCommandBuffer cmd, OverlayMode mode,
                           VkImageView output, VkExtent2D extent) {
    if (mode == OverlayMode::None) {
        return;
    }

    createPipelines();

    // Map the OverlayMode enum to a shader mode index (0-2)
    u32 shaderMode = 0;
    switch (mode) {
        case OverlayMode::Depth:   shaderMode = 1; break;
        case OverlayMode::Normals: shaderMode = 2; break;
        default:                   shaderMode = 0; break; // passthrough for unsupported modes
    }

    if (shaderMode >= MODE_COUNT || pipelines_[shaderMode] == VK_NULL_HANDLE) {
        return;
    }

    // Begin dynamic rendering
    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView   = output;
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp      = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingInfo renderInfo{};
    renderInfo.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderInfo.renderArea.offset    = { 0, 0 };
    renderInfo.renderArea.extent    = extent;
    renderInfo.layerCount           = 1;
    renderInfo.colorAttachmentCount = 1;
    renderInfo.pColorAttachments    = &colorAttachment;

    vkCmdBeginRendering(cmd, &renderInfo);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines_[shaderMode]);

    // Set dynamic viewport and scissor
    VkViewport viewport{};
    viewport.x        = 0.0f;
    viewport.y        = 0.0f;
    viewport.width    = static_cast<float>(extent.width);
    viewport.height   = static_cast<float>(extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = { 0, 0 };
    scissor.extent = extent;
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // Draw fullscreen triangle (3 vertices, 1 instance, no vertex buffer)
    vkCmdDraw(cmd, 3, 1, 0, 0);

    vkCmdEndRendering(cmd);
}

} // namespace phosphor
