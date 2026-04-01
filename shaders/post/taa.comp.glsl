#version 460

#extension GL_EXT_buffer_reference                     : require
#extension GL_EXT_buffer_reference2                    : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_scalar_block_layout                  : require

#include "types.glsl"
#include "math.glsl"

// ---------------------------------------------------------------------------
// Temporal Anti-Aliasing (TAA) with neighborhood clamping
//
// Reads current color and depth, reprojects to the previous frame using
// motion vectors derived from current/previous VP matrices, samples
// history with bilinear filtering, clamps history to the 3x3 current
// neighborhood AABB, and blends with exponential moving average.
//
// Dispatch: ceil(width/8) x ceil(height/8) x 1
// ---------------------------------------------------------------------------

layout(local_size_x = 8, local_size_y = 8) in;

// Per-pass descriptor set (set = 1)
layout(set = 1, binding = 0, rgba16f) uniform readonly  image2D  currentColor;
layout(set = 1, binding = 1)          uniform sampler2D  depthBuffer;
layout(set = 1, binding = 2)          uniform sampler2D  historyBuffer;
layout(set = 1, binding = 3, rgba16f) uniform writeonly  image2D  outputImage;
layout(set = 1, binding = 4, rgba16f) uniform writeonly  image2D  historyOutput;

layout(set = 1, binding = 5, std140) uniform TAAData {
    mat4 prevViewProjection;
    mat4 currentInvViewProjection;
    float blendFactor;   // typically 0.05
    float pad0;
    float pad1;
    float pad2;
};

// ---------------------------------------------------------------------------
// Safe image load with clamping
// ---------------------------------------------------------------------------

vec3 loadCurrentClamped(ivec2 coord) {
    coord = clamp(coord, ivec2(0), ivec2(pc.resolution) - 1);
    return imageLoad(currentColor, coord).rgb;
}

// ---------------------------------------------------------------------------
// Luminance for Tonemapping weight
// ---------------------------------------------------------------------------

float luminance(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

// ---------------------------------------------------------------------------
// YCoCg color space for better clamping
// ---------------------------------------------------------------------------

vec3 rgbToYCoCg(vec3 rgb) {
    return vec3(
         0.25 * rgb.r + 0.50 * rgb.g + 0.25 * rgb.b,
         0.50 * rgb.r                 - 0.50 * rgb.b,
        -0.25 * rgb.r + 0.50 * rgb.g - 0.25 * rgb.b
    );
}

vec3 yCoCgToRgb(vec3 ycocg) {
    float Y  = ycocg.x;
    float Co = ycocg.y;
    float Cg = ycocg.z;
    return vec3(
        Y + Co - Cg,
        Y + Cg,
        Y - Co - Cg
    );
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

void main() {
    ivec2 pixelCoord = ivec2(gl_GlobalInvocationID.xy);

    if (any(greaterThanEqual(uvec2(pixelCoord), pc.resolution))) {
        return;
    }

    vec2 texelSize = 1.0 / vec2(pc.resolution);
    vec2 uv = (vec2(pixelCoord) + 0.5) * texelSize;

    // --- Read current color ---
    vec3 currentRgb = imageLoad(currentColor, pixelCoord).rgb;

    // --- Reconstruct world position from depth ---
    float depth = texture(depthBuffer, uv).r;

    // For background pixels, just pass through current color
    if (depth <= 0.0) {
        imageStore(outputImage, pixelCoord, vec4(currentRgb, 1.0));
        imageStore(historyOutput, pixelCoord, vec4(currentRgb, 1.0));
        return;
    }

    vec3 worldPos = reconstructWorldPos(uv, depth, currentInvViewProjection);

    // --- Compute motion vector via reprojection ---
    vec4 prevClip = prevViewProjection * vec4(worldPos, 1.0);
    vec2 prevNdc  = prevClip.xy / prevClip.w;
    vec2 prevUv   = prevNdc * 0.5 + 0.5;

    // --- Sample history (bilinear via sampler) ---
    vec3 historyRgb = texture(historyBuffer, prevUv).rgb;

    // --- 3x3 neighborhood AABB for clamping (in YCoCg space) ---
    vec3 minColor = vec3( 1e10);
    vec3 maxColor = vec3(-1e10);

    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec3 neighbor = loadCurrentClamped(pixelCoord + ivec2(x, y));
            vec3 ycocg = rgbToYCoCg(neighbor);
            minColor = min(minColor, ycocg);
            maxColor = max(maxColor, ycocg);
        }
    }

    // Slightly expand the AABB to reduce flickering
    vec3 aabbCenter = (minColor + maxColor) * 0.5;
    vec3 aabbExtent = (maxColor - minColor) * 0.5;
    aabbExtent = max(aabbExtent, vec3(0.001));
    // Expand by 10% for stability
    minColor = aabbCenter - aabbExtent * 1.1;
    maxColor = aabbCenter + aabbExtent * 1.1;

    // Clamp history in YCoCg space
    vec3 historyYCoCg = rgbToYCoCg(historyRgb);
    historyYCoCg = clamp(historyYCoCg, minColor, maxColor);
    historyRgb = yCoCgToRgb(historyYCoCg);

    // Ensure non-negative after YCoCg roundtrip
    historyRgb = max(historyRgb, vec3(0.0));

    // --- Confidence / rejection ---
    // If reprojected UV is out of bounds, reject history entirely
    float historyWeight = 1.0;
    if (any(lessThan(prevUv, vec2(0.0))) || any(greaterThan(prevUv, vec2(1.0)))) {
        historyWeight = 0.0;
    }

    // --- Blend ---
    // Lower blend factor = more history (smoother but more ghosting)
    float alpha = mix(blendFactor, 1.0, 1.0 - historyWeight);

    // Tonemap before blending (reduces aliasing on bright edges)
    float weightCurrent = 1.0 / (1.0 + luminance(currentRgb));
    float weightHistory = 1.0 / (1.0 + luminance(historyRgb));

    vec3 result = (currentRgb * weightCurrent * alpha +
                   historyRgb * weightHistory * (1.0 - alpha));
    result /= (weightCurrent * alpha + weightHistory * (1.0 - alpha));

    // Ensure non-negative
    result = max(result, vec3(0.0));

    imageStore(outputImage, pixelCoord, vec4(result, 1.0));
    imageStore(historyOutput, pixelCoord, vec4(result, 1.0));
}
