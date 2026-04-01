#pragma once

#include "rhi/vk_common.h"
#include "core/types.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace phosphor {

class VulkanDevice;
class ShaderModule;
class BindlessDescriptorManager;

struct MeshPipelineDesc {
    std::string taskShaderPath;
    std::string meshShaderPath;
    std::string fragShaderPath;
    VkFormat colorFormat = VK_FORMAT_R32_UINT;       // visibility buffer
    VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;
    bool depthWrite = true;
    VkCompareOp depthCompare = VK_COMPARE_OP_LESS;
    VkCullModeFlags cullMode = VK_CULL_MODE_BACK_BIT;
    u32 colorAttachmentCount = 1;
};

struct ComputePipelineDesc {
    std::string shaderPath;
};

struct RayTracingPipelineDesc {
    std::string rgenPath;
    std::string rmissPath;
    std::string rchitPath;
    u32 maxRecursionDepth = 1;
};

class PipelineManager {
public:
    PipelineManager(VulkanDevice& device, BindlessDescriptorManager& descriptors);
    ~PipelineManager();
    PipelineManager(const PipelineManager&) = delete;
    PipelineManager& operator=(const PipelineManager&) = delete;

    VkPipeline createMeshPipeline(const MeshPipelineDesc& desc);
    VkPipeline createComputePipeline(const ComputePipelineDesc& desc);
    VkPipeline createRayTracingPipeline(const RayTracingPipelineDesc& desc);

    VkPipelineLayout getPipelineLayout() const;
    std::string getShaderDir() const;

    void saveCacheToDisk();

private:
    void loadCacheFromDisk();
    std::string resolveShaderPath(const std::string& relativePath) const;

    VulkanDevice& device_;
    BindlessDescriptorManager& descriptors_;
    VkPipelineCache pipelineCache_ = VK_NULL_HANDLE;
    std::string shaderDir_;
    std::vector<VkPipeline> ownedPipelines_;
};

} // namespace phosphor
