#pragma once

#include "core/launch_options.h"
#include "core/types.h"
#include "pipeline/pipeline_registry.h"
#include "platform/metal/metal_context.h"
#include "rendergraph/render_graph.h"

namespace phosphor {

class PipelineCache;
class SceneRenderer;

// ---------------------------------------------------------------------------
// DebugOverlays (F4.7) -- heatmaps composited over the frame, as render
// graph passes added only while an overlay is active (the overlay mode is
// part of the engine's graph key): with OverlayMode::None nothing is added
// and the frame is identical to a build without overlays.
//
//   Overdraw   rasterised fragments per pixel: the scene geometry drawn
//              again with additive blending and no depth test.  Declared
//              approximation: the TBDR hidden-surface removal of the real
//              forward pass shades fewer fragments than this counts.
//   LightCount lights whose attenuation is > 0 at the visible surface (own
//              depth pass, memoryless), i.e. what the forward loop evaluates
//              with a non-zero contribution.
//   TileCost   per 32x32 tile: sum over its pixels of overdraw x (1 +
//              lights).  Declared approximation of shading cost: Apple GPUs
//              expose no per-tile time.
//   Timings    no GPU pass (the engine draws the pass timings with ImGui).
//
// Passes per mode (all transient, only while the mode is active):
//   Overdraw    "Overlay overdraw" (R16Float, programmable blending) ->
//               "Overlay composite"
//   LightCount  "Overlay light count" (R16Float, cleared to -1 = no geometry,
//               own memoryless depth) -> "Overlay composite"
//   TileCost    both passes above -> "Overlay tile cost" (compute, one 32x32
//               threadgroup per tile, R32Float grid) -> "Overlay composite"
// The composite is a fullscreen triangle alpha-blended (0.7) over the colour;
// pixels without data (no fragment drawn, no geometry) stay untouched.  A
// legend strip (the palette from 0 to legend().maxValue) sits bottom-left.
//
// Each heatmap is composited over `color` (before the ImGui overlay) with a
// fixed, documented scale; legend() gives the scale for the UI.  Pipelines
// come from the pipeline cache; the scene geometry is drawn through
// SceneRenderer::encodeOverlay() with the renderer's bindings.
// ---------------------------------------------------------------------------

class DebugOverlays {
public:
    DebugOverlays(MetalContext& context, PipelineCache& pipelines);
    ~DebugOverlays();

    DebugOverlays(const DebugOverlays&) = delete;
    DebugOverlays& operator=(const DebugOverlays&) = delete;

    /// Add the passes of `mode` after the scene pass; returns the new version
    /// of `color`.  Called by Engine::buildFrameGraph (graph compile only).
    rg::TextureRef addToGraph(rg::RenderGraph& graph, rg::TextureRef color, u32 width, u32 height, OverlayMode mode,
                              const SceneRenderer& scene);

    /// Per frame, before the graph executes: overlay constants (scale, light
    /// count) into the frame upload ring.
    /// (`lightCount` is unused: the light loop reads it from FrameConstants.)
    void prepareFrame(OverlayMode mode, u32 lightCount, u32 width, u32 height);

    struct Legend {
        const char* quantity = "";  // e.g. "fragments per pixel"
        float       maxValue = 0.0f; // value mapped to the hottest colour
        const char* note     = "";  // the declared approximation
        bool        logScale = false; // false: linear 0..maxValue
    };
    [[nodiscard]] static Legend legend(OverlayMode mode);

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

} // namespace phosphor
