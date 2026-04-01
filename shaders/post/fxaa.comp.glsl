#version 460

#extension GL_EXT_buffer_reference                     : require
#extension GL_EXT_buffer_reference2                    : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_scalar_block_layout                  : require

#include "types.glsl"

// ---------------------------------------------------------------------------
// FXAA 3.11 Quality — Compute shader implementation
//
// Reads an LDR input image, performs luminance-based edge detection and
// directional sub-pixel anti-aliasing, then writes filtered output.
//
// Reference: Timothy Lottes, "FXAA 3.11 in 15 Slides"
//
// Dispatch: ceil(width/8) x ceil(height/8) x 1
// ---------------------------------------------------------------------------

layout(local_size_x = 8, local_size_y = 8) in;

// Per-pass descriptor set (set = 1)
layout(set = 1, binding = 0, rgba8) uniform readonly  image2D inputImage;
layout(set = 1, binding = 1, rgba8) uniform writeonly image2D outputImage;

// ---------------------------------------------------------------------------
// FXAA tuning parameters
// ---------------------------------------------------------------------------

const float FXAA_EDGE_THRESHOLD     = 0.0833;  // 1/12 — minimum local contrast for edge detection
const float FXAA_EDGE_THRESHOLD_MIN = 0.0625;  // avoid processing very dark areas
const float FXAA_SUBPIX_QUALITY     = 0.75;    // sub-pixel aliasing removal strength
const int   FXAA_SEARCH_STEPS       = 10;      // edge search iterations
const float FXAA_SEARCH_ACCEL       = 1.0;     // search acceleration (step multiplier)

// ---------------------------------------------------------------------------
// Luma from color (green channel approximation, perceptual weight)
// ---------------------------------------------------------------------------

float luma(vec3 color) {
    return dot(color, vec3(0.299, 0.587, 0.114));
}

// ---------------------------------------------------------------------------
// Safe image load with clamping
// ---------------------------------------------------------------------------

vec3 loadClamped(ivec2 coord) {
    coord = clamp(coord, ivec2(0), ivec2(pc.resolution) - 1);
    return imageLoad(inputImage, coord).rgb;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

void main() {
    ivec2 pixelCoord = ivec2(gl_GlobalInvocationID.xy);

    if (any(greaterThanEqual(uvec2(pixelCoord), pc.resolution))) {
        return;
    }

    // Load 3x3 neighborhood luma
    vec3 rgbM  = loadClamped(pixelCoord);
    vec3 rgbN  = loadClamped(pixelCoord + ivec2( 0, -1));
    vec3 rgbS  = loadClamped(pixelCoord + ivec2( 0,  1));
    vec3 rgbE  = loadClamped(pixelCoord + ivec2( 1,  0));
    vec3 rgbW  = loadClamped(pixelCoord + ivec2(-1,  0));

    float lumaM = luma(rgbM);
    float lumaN = luma(rgbN);
    float lumaS = luma(rgbS);
    float lumaE = luma(rgbE);
    float lumaW = luma(rgbW);

    // Determine local contrast range
    float lumaMin = min(lumaM, min(min(lumaN, lumaS), min(lumaE, lumaW)));
    float lumaMax = max(lumaM, max(max(lumaN, lumaS), max(lumaE, lumaW)));
    float lumaRange = lumaMax - lumaMin;

    // If contrast is below threshold, skip AA
    if (lumaRange < max(FXAA_EDGE_THRESHOLD_MIN, lumaMax * FXAA_EDGE_THRESHOLD)) {
        imageStore(outputImage, pixelCoord, vec4(rgbM, 1.0));
        return;
    }

    // Load diagonal neighbors for sub-pixel aliasing detection
    vec3 rgbNW = loadClamped(pixelCoord + ivec2(-1, -1));
    vec3 rgbNE = loadClamped(pixelCoord + ivec2( 1, -1));
    vec3 rgbSW = loadClamped(pixelCoord + ivec2(-1,  1));
    vec3 rgbSE = loadClamped(pixelCoord + ivec2( 1,  1));

    float lumaNW = luma(rgbNW);
    float lumaNE = luma(rgbNE);
    float lumaSW = luma(rgbSW);
    float lumaSE = luma(rgbSE);

    // Sub-pixel aliasing: compare center luma to neighborhood average
    float lumaAvg = (lumaN + lumaS + lumaE + lumaW) * 0.25;
    float subpixRatio = clamp(abs(lumaAvg - lumaM) / max(lumaRange, 0.001), 0.0, 1.0);
    float subpixBlend = smoothstep(0.0, 1.0, subpixRatio);
    subpixBlend = subpixBlend * subpixBlend * FXAA_SUBPIX_QUALITY;

    // Determine edge direction: horizontal or vertical
    float edgeH = abs(lumaN + lumaS - 2.0 * lumaM) * 2.0
                + abs(lumaNE + lumaSE - 2.0 * lumaE)
                + abs(lumaNW + lumaSW - 2.0 * lumaW);

    float edgeV = abs(lumaE + lumaW - 2.0 * lumaM) * 2.0
                + abs(lumaNE + lumaNW - 2.0 * lumaN)
                + abs(lumaSE + lumaSW - 2.0 * lumaS);

    bool isHorizontal = (edgeH >= edgeV);

    // Select edge endpoints
    float lumaPos = isHorizontal ? lumaS : lumaE;
    float lumaNeg = isHorizontal ? lumaN : lumaW;

    float gradientPos = abs(lumaPos - lumaM);
    float gradientNeg = abs(lumaNeg - lumaM);

    // Choose the side with the steeper gradient
    bool positiveSide = (gradientPos >= gradientNeg);
    float gradient = positiveSide ? gradientPos : gradientNeg;
    float lumaEdge = positiveSide ? lumaPos : lumaNeg;

    // Step direction in pixels (perpendicular to edge)
    vec2 stepDir = isHorizontal ? vec2(0.0, 1.0) : vec2(1.0, 0.0);
    if (!positiveSide) stepDir = -stepDir;

    // Start at the edge center (half pixel offset)
    vec2 edgeCenter = vec2(pixelCoord) + 0.5 + stepDir * 0.5;
    float lumaEdgeAvg = 0.5 * (lumaM + lumaEdge);
    float gradientScaled = gradient * 0.25;

    // Search along the edge in both directions
    vec2 searchDir = isHorizontal ? vec2(1.0, 0.0) : vec2(0.0, 1.0);

    vec2 posEnd = edgeCenter;
    vec2 negEnd = edgeCenter;
    bool donePos = false;
    bool doneNeg = false;

    for (int i = 0; i < FXAA_SEARCH_STEPS; i++) {
        float step = (1.0 + float(i) * FXAA_SEARCH_ACCEL);

        if (!donePos) {
            posEnd += searchDir * step;
            ivec2 posCoord = clamp(ivec2(posEnd), ivec2(0), ivec2(pc.resolution) - 1);
            float lumaSample = luma(imageLoad(inputImage, posCoord).rgb);
            donePos = abs(lumaSample - lumaEdgeAvg) >= gradientScaled;
        }

        if (!doneNeg) {
            negEnd -= searchDir * step;
            ivec2 negCoord = clamp(ivec2(negEnd), ivec2(0), ivec2(pc.resolution) - 1);
            float lumaSample = luma(imageLoad(inputImage, negCoord).rgb);
            doneNeg = abs(lumaSample - lumaEdgeAvg) >= gradientScaled;
        }

        if (donePos && doneNeg) break;
    }

    // Compute distances to edge endpoints
    float distPos = isHorizontal
        ? (posEnd.x - (float(pixelCoord.x) + 0.5))
        : (posEnd.y - (float(pixelCoord.y) + 0.5));
    float distNeg = isHorizontal
        ? ((float(pixelCoord.x) + 0.5) - negEnd.x)
        : ((float(pixelCoord.y) + 0.5) - negEnd.y);

    float distMin = min(distPos, distNeg);
    float edgeLength = distPos + distNeg;

    // Pixel blend factor: based on distance from the nearest edge endpoint
    float pixelBlend = 0.5 - distMin / max(edgeLength, 0.001);
    pixelBlend = max(0.0, pixelBlend);

    // Take the maximum of sub-pixel and edge blend factors
    float finalBlend = max(subpixBlend, pixelBlend);

    // Blend along the step direction (perpendicular to edge)
    vec2 sampleOffset = stepDir * finalBlend;
    ivec2 blendCoord = clamp(
        ivec2(vec2(pixelCoord) + 0.5 + sampleOffset),
        ivec2(0),
        ivec2(pc.resolution) - 1);

    vec3 result = imageLoad(inputImage, blendCoord).rgb;

    imageStore(outputImage, pixelCoord, vec4(result, 1.0));
}
