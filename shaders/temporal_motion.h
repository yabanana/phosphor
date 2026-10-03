#pragma once
#include "renderer/temporal_layout.h"

static float4x4 temporalMatrix(constant float *m) {
    return float4x4(float4(m[0], m[1], m[2], m[3]), float4(m[4], m[5], m[6], m[7]), float4(m[8], m[9], m[10], m[11]),
                    float4(m[12], m[13], m[14], m[15]));
}
static float4x4 temporalMatrix(const device float *m) {
    return float4x4(float4(m[0], m[1], m[2], m[3]), float4(m[4], m[5], m[6], m[7]), float4(m[8], m[9], m[10], m[11]),
                    float4(m[12], m[13], m[14], m[15]));
}
// Current -> previous, in input pixels, +Y down. Never includes projection
// jitter (MetalFX's default convention). Invalid history has zero motion.
static float2 temporalMotion(float4 current, float4 previous, constant GPUTemporalParams &p, bool valid) {
    if (!valid || current.w <= 1e-6f || previous.w <= 1e-6f)
        return float2(0);
    float2 motion =
        (previous.xy / previous.w - current.xy / current.w) * float2(p.renderSize[0] * 0.5f, -p.renderSize[1] * 0.5f);
    if (p.debugFlags & 1u)
        motion.x += 4.0f; // explicit negative control
    return all(isfinite(motion)) ? clamp(motion, float2(-16384), float2(16384)) : float2(0);
}
