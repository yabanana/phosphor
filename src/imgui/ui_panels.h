#pragma once

#include "core/types.h"
#include "diagnostics/debug_overlay.h"
#include "renderer/aa_pass.h"

namespace phosphor {

class FrameStats;
class GpuProfiler;
class ResourceTracker;
class RenderDocCapture;

// ---------------------------------------------------------------------------
// UIPanels -- static helper class that draws all ImGui diagnostic panels.
// Each method is a self-contained ImGui window.  Callers choose which
// panels to show each frame.
// ---------------------------------------------------------------------------

class UIPanels {
public:
    /// Combo box listing the available test benches.
    /// Sets @p changed to true if the user picked a different bench.
    static void drawTestBenchSelector(int& currentBench, bool& changed);

    /// FPS / CPU ms / GPU ms graphs and per-pass GPU timings.
    static void drawPerformancePanel(const FrameStats& stats, const GpuProfiler& profiler);

    /// Live GPU resource allocation counts and memory usage.
    static void drawResourcePanel(const ResourceTracker& tracker);

    /// Debug visualization overlay selector and renderer tweaks.
    static void drawDebugPanel(OverlayMode& overlayMode, AAMode& aaMode,
                               float& exposure, bool& ddgiEnabled,
                               bool& restirEnabled);

    /// RenderDoc capture controls.
    static void drawRenderDocPanel(RenderDocCapture& renderdoc);
};

} // namespace phosphor
