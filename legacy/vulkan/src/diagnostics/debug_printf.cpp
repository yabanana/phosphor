#include "diagnostics/debug_printf.h"
#include "core/log.h"

#include <cstring>
#include <vector>

namespace phosphor {

// ---------------------------------------------------------------------------
// isSupported -- checks for VK_KHR_shader_non_semantic_info
// ---------------------------------------------------------------------------

bool DebugPrintfConfig::isSupported(VkPhysicalDevice physicalDevice) {
    u32 extCount = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extCount, nullptr);
    if (extCount == 0) {
        return false;
    }

    std::vector<VkExtensionProperties> extensions(extCount);
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extCount, extensions.data());

    for (const auto& ext : extensions) {
        if (std::strcmp(ext.extensionName, VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME) == 0) {
            LOG_DEBUG("Debug printf supported (VK_KHR_shader_non_semantic_info found)");
            return true;
        }
    }

    LOG_DEBUG("Debug printf not supported (VK_KHR_shader_non_semantic_info not found)");
    return false;
}

// ---------------------------------------------------------------------------
// getFeature -- return the validation feature enable enum
// ---------------------------------------------------------------------------

VkValidationFeatureEnableEXT DebugPrintfConfig::getFeature() {
    return VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT;
}

} // namespace phosphor
