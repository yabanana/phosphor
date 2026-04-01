#version 460

#extension GL_EXT_buffer_reference                     : require
#extension GL_EXT_buffer_reference2                    : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_scalar_block_layout                  : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

#include "types.glsl"
#include "math.glsl"
#include "brdf.glsl"
#include "packing.glsl"

// ---------------------------------------------------------------------------
// ReSTIR DI — Initial candidate generation
//
// Per pixel: reconstruct world position from visibility buffer + depth,
// sample N=32 candidate lights using power-proportional PDF, run Weighted
// Reservoir Sampling (WRS) to select a single light with weight proportional
// to the ratio of target PDF (unshadowed radiance) to source PDF (power).
//
// Dispatch: ceil(width/8) x ceil(height/8) x 1
// ---------------------------------------------------------------------------

layout(local_size_x = 8, local_size_y = 8) in;

// Per-pass descriptor set (set = 1)
layout(set = 1, binding = 0, r32ui)  uniform readonly  uimage2D visibilityBuffer;
layout(set = 1, binding = 1)         uniform sampler2D  depthBuffer;
layout(set = 1, binding = 2, rgba16f) uniform readonly  image2D  normalImage;

// Reservoir storage buffer: one Reservoir per pixel
struct Reservoir {
    uint  selectedLight;   // index into light buffer
    float weightSum;       // sum of weights seen so far
    uint  M;               // number of candidates seen
    float W;               // final weight: (1/M) * (weightSum / targetPdf(selected))
};

layout(set = 1, binding = 3, std430) writeonly buffer ReservoirOut {
    Reservoir reservoirsOut[];
};

// ---------------------------------------------------------------------------
// PCG hash for random number generation
// ---------------------------------------------------------------------------

uint pcgHash(uint val) {
    uint state = val * 747796405u + 2891336453u;
    uint word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Generate a uniform float in [0,1) from a seed, and advance the seed
float randomFloat(inout uint seed) {
    seed = pcgHash(seed);
    return float(seed) / 4294967296.0;
}

// ---------------------------------------------------------------------------
// Light power estimate (for source PDF — proportional to emitted power)
// ---------------------------------------------------------------------------

float lightPower(GPULight light) {
    float luminance = dot(light.color, vec3(0.2126, 0.7152, 0.0722));
    return luminance * light.intensity;
}

// ---------------------------------------------------------------------------
// Compute unshadowed radiance contribution of a light at a surface point
// (target PDF for ReSTIR)
// ---------------------------------------------------------------------------

float targetPdf(GPULight light, vec3 worldPos, vec3 N) {
    vec3 L;
    float attenuation = 1.0;

    if (light.type == 0) {
        // Directional
        L = normalize(-light.direction);
    } else {
        // Point or spot
        vec3 toLight = light.position - worldPos;
        float dist = length(toLight);
        L = toLight / max(dist, EPSILON);

        float distSq = dist * dist;
        attenuation = 1.0 / max(distSq, 0.0001);

        if (light.range > 0.0) {
            float rangeSq = light.range * light.range;
            float factor = distSq / rangeSq;
            float smooth_ = clamp(1.0 - factor * factor, 0.0, 1.0);
            attenuation *= smooth_ * smooth_;
        }

        if (light.type == 2) {
            // Spot cone
            float cosAngle = dot(normalize(light.direction), -L);
            float spotFactor = clamp(
                (cosAngle - light.outerCone) / max(light.innerCone - light.outerCone, EPSILON),
                0.0, 1.0);
            attenuation *= spotFactor * spotFactor;
        }
    }

    float NdotL = max(dot(N, L), 0.0);
    float luminance = dot(light.color, vec3(0.2126, 0.7152, 0.0722));
    return luminance * light.intensity * attenuation * NdotL;
}

// ---------------------------------------------------------------------------
// Select a light index proportional to power using a random float
// This is a simple linear scan; fine for small light counts.
// ---------------------------------------------------------------------------

uint selectLightByPower(uint64_t lightAddr, uint lightCount, float xi, out float pdf) {
    // First pass: compute total power
    float totalPower = 0.0;
    for (uint i = 0; i < lightCount; i++) {
        GPULight light = loadLight(lightAddr, i);
        totalPower += max(lightPower(light), EPSILON);
    }

    // Second pass: pick light using CDF
    float threshold = xi * totalPower;
    float cumulative = 0.0;
    uint selected = 0;

    for (uint i = 0; i < lightCount; i++) {
        GPULight light = loadLight(lightAddr, i);
        float p = max(lightPower(light), EPSILON);
        cumulative += p;
        if (cumulative >= threshold) {
            selected = i;
            pdf = p / totalPower;
            return selected;
        }
    }

    // Fallback: last light
    pdf = max(lightPower(loadLight(lightAddr, lightCount - 1)), EPSILON) / totalPower;
    return lightCount - 1;
}

// ---------------------------------------------------------------------------
// Weighted Reservoir Sampling: update reservoir with a new sample
// ---------------------------------------------------------------------------

void updateReservoir(inout Reservoir r, uint lightIdx, float weight, float xi) {
    r.weightSum += weight;
    r.M += 1;
    if (xi * r.weightSum < weight) {
        r.selectedLight = lightIdx;
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

void main() {
    ivec2 pixelCoord = ivec2(gl_GlobalInvocationID.xy);

    if (any(greaterThanEqual(uvec2(pixelCoord), pc.resolution))) {
        return;
    }

    uint pixelIndex = pixelCoord.y * pc.resolution.x + pixelCoord.x;

    // Read visibility buffer
    uint visEncoded = imageLoad(visibilityBuffer, pixelCoord).r;

    // Initialize empty reservoir
    Reservoir reservoir;
    reservoir.selectedLight = 0xFFFFFFFF;
    reservoir.weightSum     = 0.0;
    reservoir.M             = 0;
    reservoir.W             = 0.0;

    // Background pixel: write empty reservoir
    if (visEncoded == VISIBILITY_CLEAR || pc.lightCount == 0) {
        reservoirsOut[pixelIndex] = reservoir;
        return;
    }

    // Reconstruct world position from depth
    vec2 uv = (vec2(pixelCoord) + 0.5) / vec2(pc.resolution);
    float depth = texture(depthBuffer, uv).r;

    mat4 invVP = inverse(pc.viewProjection);
    vec3 worldPos = reconstructWorldPos(uv, depth, invVP);

    // Read normal from normal image (stored as RGB = world normal)
    vec3 N = normalize(imageLoad(normalImage, pixelCoord).rgb * 2.0 - 1.0);

    // Scene globals for light access
    SceneGlobalsBuffer globals = SceneGlobalsBuffer(pc.sceneGlobalsAddress);

    // RNG seed: pixel + frame
    uint seed = pcgHash(pixelIndex ^ pcgHash(pc.frameIndex * 1099087573u));

    // Sample N=32 candidate lights
    const uint NUM_CANDIDATES = 32;
    uint candidateCount = min(NUM_CANDIDATES, pc.lightCount);

    for (uint i = 0; i < candidateCount; i++) {
        float xi = randomFloat(seed);

        // Select light using power-proportional PDF
        float sourcePdf;
        uint lightIdx = selectLightByPower(globals.lightAddr, pc.lightCount, xi, sourcePdf);

        // Compute target PDF (unshadowed radiance)
        GPULight light = loadLight(globals.lightAddr, lightIdx);
        float pHat = targetPdf(light, worldPos, N);

        // WRS weight = targetPdf / sourcePdf
        float weight = pHat / max(sourcePdf, EPSILON);

        float xi2 = randomFloat(seed);
        updateReservoir(reservoir, lightIdx, weight, xi2);
    }

    // Finalize reservoir weight
    if (reservoir.M > 0 && reservoir.selectedLight != 0xFFFFFFFF) {
        GPULight selected = loadLight(globals.lightAddr, reservoir.selectedLight);
        float pHat = targetPdf(selected, worldPos, N);
        reservoir.W = (pHat > 0.0)
            ? (1.0 / max(float(reservoir.M), 1.0)) * (reservoir.weightSum / pHat)
            : 0.0;
    }

    reservoirsOut[pixelIndex] = reservoir;
}
