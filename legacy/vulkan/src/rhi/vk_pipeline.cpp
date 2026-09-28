#include "rhi/vk_pipeline.h"
#include "rhi/vk_device.h"
#include "rhi/vk_shader.h"
#include "rhi/vk_descriptors.h"
#include <fstream>
#include <filesystem>
#include <unistd.h>

namespace phosphor {

static std::string getExecutableDir() {
    char buf[4096];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len <= 0) return ".";
    buf[len] = '\0';
    std::string path(buf);
    auto pos = path.rfind('/');
    return (pos != std::string::npos) ? path.substr(0, pos) : ".";
}

PipelineManager::PipelineManager(VulkanDevice& device, BindlessDescriptorManager& descriptors)
    : device_(device), descriptors_(descriptors) {
    shaderDir_ = getExecutableDir() + "/shaders";
    loadCacheFromDisk();
    LOG_INFO("Pipeline manager initialized, shader dir: %s", shaderDir_.c_str());
}

PipelineManager::~PipelineManager() {
    VkDevice dev = device_.getDevice();
    saveCacheToDisk();
    for (auto p : ownedPipelines_) {
        vkDestroyPipeline(dev, p, nullptr);
    }
    if (pipelineCache_) vkDestroyPipelineCache(dev, pipelineCache_, nullptr);
}

void PipelineManager::loadCacheFromDisk() {
    VkPipelineCacheCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};

    std::string cachePath = getExecutableDir() + "/phosphor_pipeline_cache.bin";
    std::ifstream file(cachePath, std::ios::binary | std::ios::ate);
    std::vector<char> data;

    if (file.is_open()) {
        auto size = file.tellg();
        if (size > 0) {
            data.resize(static_cast<size_t>(size));
            file.seekg(0);
            file.read(data.data(), size);
            info.initialDataSize = data.size();
            info.pInitialData = data.data();
            LOG_INFO("Loaded pipeline cache: %zu bytes", data.size());
        }
    }

    VK_CHECK(vkCreatePipelineCache(device_.getDevice(), &info, nullptr, &pipelineCache_));
}

void PipelineManager::saveCacheToDisk() {
    if (!pipelineCache_) return;

    size_t size = 0;
    VK_CHECK(vkGetPipelineCacheData(device_.getDevice(), pipelineCache_, &size, nullptr));
    if (size == 0) return;

    std::vector<char> data(size);
    VK_CHECK(vkGetPipelineCacheData(device_.getDevice(), pipelineCache_, &size, data.data()));

    std::string cachePath = getExecutableDir() + "/phosphor_pipeline_cache.bin";
    std::ofstream file(cachePath, std::ios::binary);
    if (file.is_open()) {
        file.write(data.data(), static_cast<std::streamsize>(size));
        LOG_INFO("Saved pipeline cache: %zu bytes", size);
    }
}

std::string PipelineManager::resolveShaderPath(const std::string& relativePath) const {
    return shaderDir_ + "/" + relativePath;
}

std::string PipelineManager::getShaderDir() const { return shaderDir_; }

VkPipelineLayout PipelineManager::getPipelineLayout() const {
    return descriptors_.getBindlessPipelineLayout();
}

VkPipeline PipelineManager::createMeshPipeline(const MeshPipelineDesc& desc) {
    VkDevice dev = device_.getDevice();

    auto taskShader = ShaderModule::loadFromFile(dev, resolveShaderPath(desc.taskShaderPath));
    auto meshShader = ShaderModule::loadFromFile(dev, resolveShaderPath(desc.meshShaderPath));
    auto fragShader = ShaderModule::loadFromFile(dev, resolveShaderPath(desc.fragShaderPath));

    std::array<VkPipelineShaderStageCreateInfo, 3> stages = {
        taskShader.getStageCreateInfo(),
        meshShader.getStageCreateInfo(),
        fragShader.getStageCreateInfo(),
    };

    // Rasterization
    VkPipelineRasterizationStateCreateInfo rasterInfo{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
    rasterInfo.cullMode = desc.cullMode;
    rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterInfo.lineWidth = 1.0f;

    // Multisampling (no MSAA)
    VkPipelineMultisampleStateCreateInfo msInfo{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    msInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // Depth stencil
    VkPipelineDepthStencilStateCreateInfo depthInfo{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depthInfo.depthTestEnable = VK_TRUE;
    depthInfo.depthWriteEnable = desc.depthWrite ? VK_TRUE : VK_FALSE;
    depthInfo.depthCompareOp = desc.depthCompare;

    // Color blend (no blending for vis buffer)
    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                                   | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    std::vector<VkPipelineColorBlendAttachmentState> blendAttachments(desc.colorAttachmentCount, blendAttachment);

    VkPipelineColorBlendStateCreateInfo blendInfo{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blendInfo.attachmentCount = desc.colorAttachmentCount;
    blendInfo.pAttachments = blendAttachments.data();

    // Dynamic state: viewport + scissor
    std::array<VkDynamicState, 2> dynStates = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynInfo{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynInfo.dynamicStateCount = static_cast<u32>(dynStates.size());
    dynInfo.pDynamicStates = dynStates.data();

    // Viewport state (dynamic, count only)
    VkPipelineViewportStateCreateInfo viewportInfo{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewportInfo.viewportCount = 1;
    viewportInfo.scissorCount = 1;

    // Dynamic rendering: no VkRenderPass
    VkFormat colorFormat = desc.colorFormat;
    VkPipelineRenderingCreateInfo renderingInfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    renderingInfo.colorAttachmentCount = desc.colorAttachmentCount;
    renderingInfo.pColorAttachmentFormats = &colorFormat;
    renderingInfo.depthAttachmentFormat = desc.depthFormat;

    VkGraphicsPipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipelineInfo.pNext = &renderingInfo;
    pipelineInfo.stageCount = static_cast<u32>(stages.size());
    pipelineInfo.pStages = stages.data();
    pipelineInfo.pRasterizationState = &rasterInfo;
    pipelineInfo.pMultisampleState = &msInfo;
    pipelineInfo.pDepthStencilState = &depthInfo;
    pipelineInfo.pColorBlendState = &blendInfo;
    pipelineInfo.pDynamicState = &dynInfo;
    pipelineInfo.pViewportState = &viewportInfo;
    pipelineInfo.layout = getPipelineLayout();

    // No vertex input for mesh shaders
    VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    pipelineInfo.pVertexInputState = &vertexInput;
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    pipelineInfo.pInputAssemblyState = &inputAssembly;

    VkPipeline pipeline = VK_NULL_HANDLE;
    VK_CHECK(vkCreateGraphicsPipelines(dev, pipelineCache_, 1, &pipelineInfo, nullptr, &pipeline));
    ownedPipelines_.push_back(pipeline);

    LOG_INFO("Created mesh pipeline: %s + %s + %s",
             desc.taskShaderPath.c_str(), desc.meshShaderPath.c_str(), desc.fragShaderPath.c_str());
    return pipeline;
}

VkPipeline PipelineManager::createComputePipeline(const ComputePipelineDesc& desc) {
    VkDevice dev = device_.getDevice();

    auto shader = ShaderModule::loadFromFile(dev, resolveShaderPath(desc.shaderPath));

    VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.stage = shader.getStageCreateInfo();
    info.layout = getPipelineLayout();

    VkPipeline pipeline = VK_NULL_HANDLE;
    VK_CHECK(vkCreateComputePipelines(dev, pipelineCache_, 1, &info, nullptr, &pipeline));
    ownedPipelines_.push_back(pipeline);

    LOG_INFO("Created compute pipeline: %s", desc.shaderPath.c_str());
    return pipeline;
}

VkPipeline PipelineManager::createRayTracingPipeline(const RayTracingPipelineDesc& desc) {
    VkDevice dev = device_.getDevice();

    auto rgenShader = ShaderModule::loadFromFile(dev, resolveShaderPath(desc.rgenPath));
    auto rmissShader = ShaderModule::loadFromFile(dev, resolveShaderPath(desc.rmissPath));
    auto rchitShader = ShaderModule::loadFromFile(dev, resolveShaderPath(desc.rchitPath));

    std::array<VkPipelineShaderStageCreateInfo, 3> stages = {
        rgenShader.getStageCreateInfo(),
        rmissShader.getStageCreateInfo(),
        rchitShader.getStageCreateInfo(),
    };

    // Shader groups
    std::array<VkRayTracingShaderGroupCreateInfoKHR, 3> groups{};

    // Raygen group
    groups[0].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[0].generalShader = 0;
    groups[0].closestHitShader = VK_SHADER_UNUSED_KHR;
    groups[0].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[0].intersectionShader = VK_SHADER_UNUSED_KHR;

    // Miss group
    groups[1].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[1].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[1].generalShader = 1;
    groups[1].closestHitShader = VK_SHADER_UNUSED_KHR;
    groups[1].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[1].intersectionShader = VK_SHADER_UNUSED_KHR;

    // Closest hit group
    groups[2].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[2].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    groups[2].generalShader = VK_SHADER_UNUSED_KHR;
    groups[2].closestHitShader = 2;
    groups[2].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[2].intersectionShader = VK_SHADER_UNUSED_KHR;

    VkRayTracingPipelineCreateInfoKHR info{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
    info.stageCount = static_cast<u32>(stages.size());
    info.pStages = stages.data();
    info.groupCount = static_cast<u32>(groups.size());
    info.pGroups = groups.data();
    info.maxPipelineRayRecursionDepth = desc.maxRecursionDepth;
    info.layout = getPipelineLayout();

    auto vkCreateRayTracingPipelinesKHR_ = reinterpret_cast<PFN_vkCreateRayTracingPipelinesKHR>(
        vkGetDeviceProcAddr(dev, "vkCreateRayTracingPipelinesKHR"));

    VkPipeline pipeline = VK_NULL_HANDLE;
    VK_CHECK(vkCreateRayTracingPipelinesKHR_(dev, VK_NULL_HANDLE, pipelineCache_, 1, &info, nullptr, &pipeline));
    ownedPipelines_.push_back(pipeline);

    LOG_INFO("Created RT pipeline: %s + %s + %s",
             desc.rgenPath.c_str(), desc.rmissPath.c_str(), desc.rchitPath.c_str());
    return pipeline;
}

} // namespace phosphor
