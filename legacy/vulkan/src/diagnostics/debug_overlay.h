#pragma once

#include "core/types.h"
#include "rhi/vk_common.h"
#include <vulkan/vulkan.h>

namespace phosphor {

class VulkanDevice;
class PipelineManager;

// ---------------------------------------------------------------------------
// OverlayMode -- debug visualization overlays rendered as a full-screen
// pass that remaps the visibility/depth/normal buffers into a color image
// for on-screen display.
// ---------------------------------------------------------------------------

enum class OverlayMode : u32 {
    None         = 0,
    Wireframe    = 1,
    Meshlets     = 2,  // color per meshlet
    Normals      = 3,
    UVs          = 4,
    Depth        = 5,
    MotionVec    = 6,
    Overdraw     = 7,
    LightHeat    = 8,  // lights per pixel heatmap
    COUNT
};

inline const char* overlayModeName(OverlayMode mode) {
    switch (mode) {
        case OverlayMode::None:      return "None";
        case OverlayMode::Wireframe: return "Wireframe";
        case OverlayMode::Meshlets:  return "Meshlets";
        case OverlayMode::Normals:   return "Normals";
        case OverlayMode::UVs:       return "UVs";
        case OverlayMode::Depth:     return "Depth";
        case OverlayMode::MotionVec: return "Motion Vectors";
        case OverlayMode::Overdraw:  return "Overdraw";
        case OverlayMode::LightHeat: return "Light Heatmap";
        default:                     return "Unknown";
    }
}

// ---------------------------------------------------------------------------
// DebugOverlay -- renders a full-screen debug visualization pass.
//
// Uses the existing debug_overlay.vert.glsl and debug_overlay.frag.glsl
// shaders.  Each mode uses a specialization constant in the fragment shader
// to select the visualization.  Pipelines are created lazily.
// ---------------------------------------------------------------------------

class DebugOverlay {
public:
    DebugOverlay(VulkanDevice& device, PipelineManager& pipelines);
    ~DebugOverlay();

    DebugOverlay(const DebugOverlay&) = delete;
    DebugOverlay& operator=(const DebugOverlay&) = delete;

    /// Render the debug overlay into the current render pass.
    /// @param cmd    Active command buffer (must be inside a render pass)
    /// @param mode   Which overlay to draw (None = passthrough)
    /// @param output Target color image view for dynamic rendering
    /// @param extent Output resolution
    void render(VkCommandBuffer cmd, OverlayMode mode,
                VkImageView output, VkExtent2D extent);

private:
    void createPipelines();

    static constexpr u32 MODE_COUNT = 3; // passthrough, depth, normals (shader supports 0-2)

    VkPipeline       pipelines_[MODE_COUNT] = {};
    VulkanDevice&    device_;
    PipelineManager& pipelinesRef_;
    bool             initialized_ = false;
};

} // namespace phosphor
