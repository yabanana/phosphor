#pragma once

#include "core/types.h"

namespace phosphor {

// ---------------------------------------------------------------------------
// Push constants layout — 128 bytes, shared across all render passes.
// Must match the GLSL PushConstants declaration in common/types.glsl.
// ---------------------------------------------------------------------------

struct PushConstants {
    float viewProjection[16];     // mat4 — 64 bytes
    float cameraPosition[4];      // vec4 (w = time) — 16 bytes
    u64   sceneGlobalsAddress;    // BDA — 8 bytes
    u64   vertexBufferAddress;    // BDA — 8 bytes
    u64   meshletBufferAddress;   // BDA — 8 bytes
    u32   resolution[2];          // uvec2 — 8 bytes
    u32   frameIndex;             // 4 bytes
    u32   lightCount;             // 4 bytes
    float exposure;               // 4 bytes
    u32   debugMode;              // 4 bytes
};
static_assert(sizeof(PushConstants) == 128, "Push constants must be exactly 128 bytes");

} // namespace phosphor
