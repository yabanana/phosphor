#version 460

#extension GL_EXT_buffer_reference                     : require
#extension GL_EXT_buffer_reference2                    : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_scalar_block_layout                  : require

#include "types.glsl"

// ---------------------------------------------------------------------------
// Tonemapping compute shader
//
// Reads HDR input, applies exposure and ACES fitted tonemapping, writes LDR.
// Dispatch: ceil(width/8) x ceil(height/8) x 1
// ---------------------------------------------------------------------------

layout(local_size_x = 8, local_size_y = 8) in;

layout(set = 1, binding = 0, rgba16f) uniform readonly  image2D hdrInput;
layout(set = 1, binding = 1, rgba8)   uniform writeonly image2D ldrOutput;

// ---------------------------------------------------------------------------
// ACES fitted tonemapping — Stephen Hill's approximation
// Reference: https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
//
// sRGB working space approximation (no full RRT+ODT transform matrices).
// ---------------------------------------------------------------------------

// Input transform: sRGB -> approximate ACEScg
vec3 sRGBtoACES(vec3 color) {
    // Approximate sRGB to ACES (ACEScg) matrix
    const mat3 m = mat3(
        0.59719, 0.07600, 0.02840,
        0.35458, 0.90834, 0.13383,
        0.04823, 0.01566, 0.83777
    );
    return m * color;
}

// Output transform: ACEScg -> sRGB
vec3 ACEStoSRGB(vec3 color) {
    const mat3 m = mat3(
         1.60475, -0.10208, -0.00327,
        -0.53108,  1.10813, -0.07276,
        -0.07367, -0.00605,  1.07602
    );
    return m * color;
}

// RRT + ODT fit (Stephen Hill)
vec3 rrtAndOdtFit(vec3 v) {
    vec3 a = v * (v + 0.0245786) - 0.000090537;
    vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return a / b;
}

vec3 acesFitted(vec3 color) {
    color = sRGBtoACES(color);
    color = rrtAndOdtFit(color);
    color = ACEStoSRGB(color);
    return clamp(color, 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// Linear to sRGB gamma curve
// ---------------------------------------------------------------------------

vec3 linearToSRGB(vec3 color) {
    vec3 lo = color * 12.92;
    vec3 hi = 1.055 * pow(color, vec3(1.0 / 2.4)) - 0.055;
    return mix(lo, hi, step(vec3(0.0031308), color));
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

void main() {
    ivec2 pixelCoord = ivec2(gl_GlobalInvocationID.xy);

    // Bounds check
    if (any(greaterThanEqual(uvec2(pixelCoord), pc.resolution))) {
        return;
    }

    vec4 hdr = imageLoad(hdrInput, pixelCoord);

    // Apply exposure
    vec3 color = hdr.rgb * pc.exposure;

    // ACES fitted tonemapping
    color = acesFitted(color);

    // Linear -> sRGB gamma
    color = linearToSRGB(color);

    imageStore(ldrOutput, pixelCoord, vec4(color, 1.0));
}
