#include "rhi/vk_shader.h"
#include <fstream>
#include <vector>
#include <filesystem>
#include <unistd.h>   // readlink

namespace phosphor {

// ---------- helpers -----------------------------------------------------

/// Resolve a path relative to the directory that contains the running
/// executable (useful for finding shaders next to the binary).
static std::string resolveRelativeToExe(const std::string& relPath) {
    namespace fs = std::filesystem;

    char buf[4096];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len <= 0) {
        // Fall back to relPath as-is if readlink fails.
        return relPath;
    }
    buf[len] = '\0';

    fs::path exeDir = fs::path(buf).parent_path();
    fs::path candidate = exeDir / relPath;

    if (fs::exists(candidate)) {
        return candidate.string();
    }

    // If not found relative to exe, return the original path unmodified
    // so the caller gets a clear "file not found" error.
    return relPath;
}

static std::vector<char> readBinaryFile(const std::string& path) {
    std::string resolved = path;

    // If the path is not absolute and does not exist at cwd, try
    // resolving relative to the executable directory.
    if (!std::filesystem::path(path).is_absolute() &&
        !std::filesystem::exists(path)) {
        resolved = resolveRelativeToExe(path);
    }

    std::ifstream file(resolved, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR("ShaderModule: failed to open '%s'", resolved.c_str());
        std::abort();
    }

    auto fileSize = static_cast<std::streamsize>(file.tellg());
    std::vector<char> buffer(static_cast<size_t>(fileSize));
    file.seekg(0);
    file.read(buffer.data(), fileSize);

    return buffer;
}

// ---------- stage inference ----------------------------------------------

VkShaderStageFlagBits ShaderModule::inferStage(const std::string& path) {
    // Match on the second-to-last extension before .spv.
    // e.g.  "shader.vert.spv" -> ".vert"
    //       "shader.frag.spv" -> ".frag"
    auto dotSpv = path.rfind(".spv");
    if (dotSpv == std::string::npos) {
        LOG_ERROR("ShaderModule: path '%s' does not end in .spv", path.c_str());
        std::abort();
    }

    // Find the dot before ".spv".
    auto stageDot = path.rfind('.', dotSpv - 1);
    if (stageDot == std::string::npos) {
        LOG_ERROR("ShaderModule: cannot infer stage from '%s'", path.c_str());
        std::abort();
    }

    std::string ext = path.substr(stageDot, dotSpv - stageDot);

    if (ext == ".vert")  return VK_SHADER_STAGE_VERTEX_BIT;
    if (ext == ".frag")  return VK_SHADER_STAGE_FRAGMENT_BIT;
    if (ext == ".comp")  return VK_SHADER_STAGE_COMPUTE_BIT;
    if (ext == ".geom")  return VK_SHADER_STAGE_GEOMETRY_BIT;
    if (ext == ".tesc")  return VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
    if (ext == ".tese")  return VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
    if (ext == ".task")  return VK_SHADER_STAGE_TASK_BIT_EXT;
    if (ext == ".mesh")  return VK_SHADER_STAGE_MESH_BIT_EXT;
    if (ext == ".rgen")  return VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    if (ext == ".rmiss") return VK_SHADER_STAGE_MISS_BIT_KHR;
    if (ext == ".rchit") return VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    if (ext == ".rahit") return VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
    if (ext == ".rint")  return VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
    if (ext == ".rcall") return VK_SHADER_STAGE_CALLABLE_BIT_KHR;

    LOG_ERROR("ShaderModule: unknown stage extension '%s' in '%s'",
              ext.c_str(), path.c_str());
    std::abort();
}

// ---------- construction / destruction -----------------------------------

ShaderModule::ShaderModule(VkDevice device, VkShaderModule module,
                           VkShaderStageFlagBits stage)
    : device_(device), module_(module), stage_(stage) {}

ShaderModule ShaderModule::loadFromFile(VkDevice device,
                                        const std::string& spvPath) {
    std::vector<char> code = readBinaryFile(spvPath);

    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode    = reinterpret_cast<const u32*>(code.data());

    VkShaderModule module = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(device, &createInfo, nullptr, &module));

    VkShaderStageFlagBits stage = inferStage(spvPath);

    LOG_DEBUG("ShaderModule: loaded '%s' (stage 0x%x, %zu bytes)",
              spvPath.c_str(), static_cast<unsigned>(stage), code.size());

    return ShaderModule(device, module, stage);
}

ShaderModule::~ShaderModule() {
    if (module_ != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device_, module_, nullptr);
    }
}

ShaderModule::ShaderModule(ShaderModule&& o) noexcept
    : device_(o.device_),
      module_(std::exchange(o.module_, VK_NULL_HANDLE)),
      stage_(o.stage_) {}

ShaderModule& ShaderModule::operator=(ShaderModule&& o) noexcept {
    if (this != &o) {
        if (module_ != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device_, module_, nullptr);
        }
        device_ = o.device_;
        module_ = std::exchange(o.module_, VK_NULL_HANDLE);
        stage_  = o.stage_;
    }
    return *this;
}

// ---------- accessors ----------------------------------------------------

VkShaderModule ShaderModule::getModule() const {
    return module_;
}

VkShaderStageFlagBits ShaderModule::getStage() const {
    return stage_;
}

VkPipelineShaderStageCreateInfo ShaderModule::getStageCreateInfo() const {
    VkPipelineShaderStageCreateInfo info{};
    info.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage  = stage_;
    info.module = module_;
    info.pName  = "main";
    return info;
}

} // namespace phosphor
