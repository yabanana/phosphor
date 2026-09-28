#pragma once

#include "rhi/vk_common.h"
#include <vector>

namespace phosphor {

struct ValidationOptions {
    bool syncValidation = true;
    bool gpuAssisted = false;
    bool bestPractices = true;
    bool debugPrintf = false; // mutually exclusive with gpuAssisted
};

class ValidationConfig {
public:
    static std::vector<const char*> getRequiredLayers();
    static bool isLayerAvailable(const char* layerName);

    // Call before instance creation to build features struct
    // The returned struct's pNext chain must be kept alive until vkCreateInstance returns
    static VkValidationFeaturesEXT buildValidationFeatures(
        const ValidationOptions& opts,
        std::vector<VkValidationFeatureEnableEXT>& enablesOut);
};

} // namespace phosphor
