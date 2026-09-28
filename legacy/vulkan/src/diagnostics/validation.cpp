#include "diagnostics/validation.h"
#include "core/types.h"
#include "core/log.h"
#include <cstring>
#include <vector>

namespace phosphor {

std::vector<const char*> ValidationConfig::getRequiredLayers() {
    return {"VK_LAYER_KHRONOS_validation"};
}

bool ValidationConfig::isLayerAvailable(const char* layerName) {
    u32 count = 0;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    std::vector<VkLayerProperties> layers(count);
    vkEnumerateInstanceLayerProperties(&count, layers.data());

    for (auto& layer : layers) {
        if (strcmp(layer.layerName, layerName) == 0) return true;
    }
    return false;
}

VkValidationFeaturesEXT ValidationConfig::buildValidationFeatures(
    const ValidationOptions& opts,
    std::vector<VkValidationFeatureEnableEXT>& enablesOut) {

    enablesOut.clear();

    if (opts.syncValidation) {
        enablesOut.push_back(VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT);
        LOG_INFO("Validation: synchronization validation enabled");
    }

    if (opts.debugPrintf) {
        enablesOut.push_back(VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT);
        LOG_INFO("Validation: debug printf enabled");
    } else if (opts.gpuAssisted) {
        enablesOut.push_back(VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT);
        enablesOut.push_back(VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_RESERVE_BINDING_SLOT_EXT);
        LOG_INFO("Validation: GPU-assisted validation enabled");
    }

    if (opts.bestPractices) {
        enablesOut.push_back(VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT);
        LOG_INFO("Validation: best practices enabled");
    }

    VkValidationFeaturesEXT features{VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT};
    features.enabledValidationFeatureCount = static_cast<u32>(enablesOut.size());
    features.pEnabledValidationFeatures = enablesOut.data();
    return features;
}

} // namespace phosphor
