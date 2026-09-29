#pragma once

// ---------------------------------------------------------------------------
// F4.7: math of the debug overlays, compiled by the host AND by the Metal
// compiler (shaders/overlay.metal includes it): the heatmap palette, the value
// scales and the tile constants exist once, so the CPU reference and the
// shader cannot drift apart.  Scalars only, like gpu_types.h.
//
// The scales are fixed and declared here (the UI legend prints them):
//   Overdraw   linear, 0 .. OVERDRAW_MAX fragments per pixel.
//   LightCount logarithmic, 0 .. LIGHTS_MAX lights: Many Lights has up to 1024
//              lights of which a few tens reach a given surface point.
//   TileCost   logarithmic, 0 .. TILECOST_MAX: the average over the tile's
//              pixels of overdraw x (1 + lights).
// Values above the maximum saturate at the hottest colour.
// ---------------------------------------------------------------------------

#include "renderer/gpu_types.h"

#ifdef __METAL_VERSION__
#define PHOSPHOR_OVERLAY_LOG2(x) metal::log2(x)
#else
#include <cmath>
#define PHOSPHOR_OVERLAY_LOG2(x) std::log2(x)
#include <vector>
#endif

namespace phosphor::overlay {

// Same numbering as OverlayMode (core/launch_options.h).
PHOSPHOR_GPU_CONSTANT u32 KIND_OVERDRAW  = 1;
PHOSPHOR_GPU_CONSTANT u32 KIND_LIGHTS    = 2;
PHOSPHOR_GPU_CONSTANT u32 KIND_TILE_COST = 3;

/// Side of a tile of the cost heatmap, in pixels (a 32x32 threadgroup).
PHOSPHOR_GPU_CONSTANT u32 TILE_SIZE = 32;

PHOSPHOR_GPU_CONSTANT float OVERDRAW_MAX = 8.0f;
PHOSPHOR_GPU_CONSTANT float LIGHTS_MAX   = 64.0f;
PHOSPHOR_GPU_CONSTANT float TILECOST_MAX = 128.0f;

/// Opacity of the heatmap over the scene (the legend strip is opaque).
PHOSPHOR_GPU_CONSTANT float OVERLAY_ALPHA = 0.7f;

/// Legend strip: pixels, from the bottom-left corner of the target.
PHOSPHOR_GPU_CONSTANT u32 LEGEND_X      = 24;
PHOSPHOR_GPU_CONSTANT u32 LEGEND_MARGIN = 24; // from the bottom edge
PHOSPHOR_GPU_CONSTANT u32 LEGEND_WIDTH  = 320;
PHOSPHOR_GPU_CONSTANT u32 LEGEND_HEIGHT = 16;

/// Heat palette: 6 stops, sRGB, blue -> cyan -> green -> yellow -> orange ->
/// hot pink; evenly spaced over t in [0, 1], linearly interpolated.
PHOSPHOR_GPU_CONSTANT u32 HEAT_STOP_COUNT = 6;
PHOSPHOR_GPU_CONSTANT float HEAT_STOPS[HEAT_STOP_COUNT * 3] = {
    0.05f, 0.05f, 0.45f, //
    0.00f, 0.45f, 0.90f, //
    0.00f, 0.80f, 0.45f, //
    0.95f, 0.90f, 0.10f, //
    0.95f, 0.35f, 0.05f, //
    1.00f, 0.85f, 0.95f, //
};

/// Channel c (0 r, 1 g, 2 b) of the heat palette at t in [0, 1] (clamped).
inline float heatChannel(float t, u32 c) {
    t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
    const float x = t * static_cast<float>(HEAT_STOP_COUNT - 1);
    u32 i = static_cast<u32>(x);
    if (i > HEAT_STOP_COUNT - 2) i = HEAT_STOP_COUNT - 2;
    const float f = x - static_cast<float>(i);
    return HEAT_STOPS[i * 3 + c] * (1.0f - f) + HEAT_STOPS[(i + 1) * 3 + c] * f;
}

/// Maximum of the scale of `kind`.
inline float scaleMax(u32 kind) {
    return kind == KIND_OVERDRAW ? OVERDRAW_MAX : (kind == KIND_LIGHTS ? LIGHTS_MAX : TILECOST_MAX);
}

/// Position of `value` on the scale of `kind`, in [0, 1].
inline float normalize(u32 kind, float value) {
    const float maxValue = scaleMax(kind);
    float t = kind == KIND_OVERDRAW ? value / maxValue
                                    : PHOSPHOR_OVERLAY_LOG2(1.0f + (value < 0.0f ? 0.0f : value)) /
                                          PHOSPHOR_OVERLAY_LOG2(1.0f + maxValue);
    return t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
}

/// True if the pixel carries data: overdraw and tile cost are 0 where nothing
/// was drawn (the scene shows through); the light count is cleared to -1
/// where no geometry is visible, so that 0 lights is a real (cold) value.
inline bool hasData(u32 kind, float value) {
    return kind == KIND_LIGHTS ? value >= 0.0f : value > 0.0f;
}

#ifndef __METAL_VERSION__

struct Rgb {
    float r = 0.0f, g = 0.0f, b = 0.0f;
    bool operator==(const Rgb&) const = default;
};

/// The heat palette at t (sRGB, before the shader's transfer to linear).
inline Rgb heatColor(float t) { return {heatChannel(t, 0), heatChannel(t, 1), heatChannel(t, 2)}; }

/// Tiles per axis for a target of `size` pixels.
inline u32 tileCount(u32 size) { return (size + TILE_SIZE - 1) / TILE_SIZE; }

/// CPU reference of the tile-cost kernel (overlay_tilecost): per tile, the
/// average over its in-bounds pixels of overdraw x (1 + max(lights, 0)).
/// `overdraw` and `lights` are row-major width x height.  Returns
/// tileCount(width) x tileCount(height) values, row-major.
std::vector<float> tileCostReference(const float* overdraw, const float* lights, u32 width, u32 height);

#endif

} // namespace phosphor::overlay
