#pragma once

#include "rhi/vk_common.h"
#include <string>

namespace phosphor {

class ShaderModule {
public:
    static ShaderModule loadFromFile(VkDevice device, const std::string& spvPath);

    ShaderModule() = default;
    ~ShaderModule();

    ShaderModule(ShaderModule&& o) noexcept;
    ShaderModule& operator=(ShaderModule&& o) noexcept;

    ShaderModule(const ShaderModule&)            = delete;
    ShaderModule& operator=(const ShaderModule&) = delete;

    VkShaderModule                  getModule() const;
    VkShaderStageFlagBits           getStage() const;
    VkPipelineShaderStageCreateInfo getStageCreateInfo() const;

private:
    ShaderModule(VkDevice device, VkShaderModule module, VkShaderStageFlagBits stage);
    static VkShaderStageFlagBits inferStage(const std::string& path);

    VkDevice              device_ = VK_NULL_HANDLE;
    VkShaderModule        module_ = VK_NULL_HANDLE;
    VkShaderStageFlagBits stage_  = VK_SHADER_STAGE_ALL;
};

} // namespace phosphor
