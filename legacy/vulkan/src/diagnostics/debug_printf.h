#pragma once

#include <vulkan/vulkan.h>
#include "core/types.h"

namespace phosphor {

// ---------------------------------------------------------------------------
// DebugPrintfConfig -- helpers for Vulkan debug printf.
//
// VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT allows shaders to use
// debugPrintfEXT(...) calls whose output appears in the validation-layer
// message callback.  This is invaluable for inspecting per-pixel / per-thread
// values without any CPU-side infrastructure.
//
// Note: debug printf and GPU-assisted validation are mutually exclusive in
// the validation layers.  The engine's ValidationConfig takes care of this;
// these helpers just check support and provide the enum value.
// ---------------------------------------------------------------------------

class DebugPrintfConfig {
public:
    /// Returns true if the physical device supports debug printf
    /// (VK_KHR_shader_non_semantic_info is required).
    static bool isSupported(VkPhysicalDevice physicalDevice);

    /// Returns the validation-feature-enable enum needed to activate debug
    /// printf when creating the VkInstance.
    static VkValidationFeatureEnableEXT getFeature();
};

} // namespace phosphor
