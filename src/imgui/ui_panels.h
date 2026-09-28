#pragma once

#include "core/types.h"

namespace phosphor {

class FrameStats;

/// Renderer settings exposed in the debug UI.
struct RenderSettings {
    int   debugMode = 0;     // 0 = lit, 1 = normals, 2 = base color
    float exposure  = 1.0f;
    bool  vsync     = true;
};

/// Read-only information shown in the stats panel.
struct RendererInfo {
    const char* gpuName     = "";
    bool        apple9      = false;
    u32         width       = 0;
    u32         height      = 0;
    u32         instances   = 0;
    u32         drawBatches = 0;
    u32         triangles   = 0;
    u32         meshlets    = 0;
    u32         textures    = 0;
    float       gpuMs       = 0.0f;
};

// ---------------------------------------------------------------------------
// UIPanels -- stateless helpers that draw the ImGui diagnostic windows.
// ---------------------------------------------------------------------------

class UIPanels {
public:
    /// Combo box listing the test benches. Sets `changed` if the user picked
    /// a different bench.
    static void drawTestBenchSelector(int& currentBench, bool& changed);

    /// FPS / CPU / GPU timings and frame-time graphs.
    static void drawPerformancePanel(const FrameStats& stats, const RendererInfo& info);

    /// Debug view selection and renderer tweaks.
    static void drawRenderPanel(RenderSettings& settings);
};

} // namespace phosphor
