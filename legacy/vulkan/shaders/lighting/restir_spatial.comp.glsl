#version 460

#extension GL_EXT_buffer_reference                     : require
#extension GL_EXT_buffer_reference2                    : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_scalar_block_layout                  : require

#include "types.glsl"
#include "math.glsl"

// ---------------------------------------------------------------------------
// ReSTIR DI — Spatial resampling
//
// Per pixel: select 5 random spatial neighbors in a disk pattern (radius
// ~30 pixels), check geometry similarity (normal dot > 0.9, depth ratio
// < 1.1), and combine accepted neighbor reservoirs via WRS with MIS weights.
//
// Dispatch: ceil(width/8) x ceil(height/8) x 1
// ---------------------------------------------------------------------------

layout(local_size_x = 8, local_size_y = 8) in;

struct Reservoir {
    uint  selectedLight;
    float weightSum;
    uint  M;
    float W;
};

// Per-pass descriptor set (set = 1)
layout(set = 1, binding = 0)          uniform sampler2D  depthBuffer;
layout(set = 1, binding = 1, rgba16f) uniform readonly   image2D  normalImage;

// Input reservoirs (from temporal pass)
layout(set = 1, binding = 2, std430) readonly buffer ReservoirIn {
    Reservoir inputReservoirs[];
};

// Output reservoirs
layout(set = 1, binding = 3, std430) writeonly buffer ReservoirOut {
    Reservoir outputReservoirs[];
};

layout(set = 1, binding = 4, std140) uniform SpatialData {
    mat4 invViewProjection;
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const uint  NUM_SPATIAL_NEIGHBORS = 5;
const float SPATIAL_RADIUS        = 30.0;
const float NORMAL_THRESHOLD      = 0.9;  // dot product
const float DEPTH_THRESHOLD       = 1.1;  // ratio

// ---------------------------------------------------------------------------
// PCG hash
// ---------------------------------------------------------------------------

uint pcgHash(uint val) {
    uint state = val * 747796405u + 2891336453u;
    uint word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float randomFloat(inout uint seed) {
    seed = pcgHash(seed);
    return float(seed) / 4294967296.0;
}

// ---------------------------------------------------------------------------
// Target PDF (must match other passes)
// ---------------------------------------------------------------------------

float targetPdf(GPULight light, vec3 worldPos, vec3 N) {
    vec3 L;
    float attenuation = 1.0;

    if (light.type == 0) {
        L = normalize(-light.direction);
    } else {
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
// Main
// ---------------------------------------------------------------------------

void main() {
    ivec2 pixelCoord = ivec2(gl_GlobalInvocationID.xy);

    if (any(greaterThanEqual(uvec2(pixelCoord), pc.resolution))) {
        return;
    }

    uint pixelIndex = pixelCoord.y * pc.resolution.x + pixelCoord.x;

    // Read center pixel data
    Reservoir center = inputReservoirs[pixelIndex];

    if (center.selectedLight == 0xFFFFFFFF) {
        outputReservoirs[pixelIndex] = center;
        return;
    }

    // Reconstruct center world position and normal
    vec2 centerUv = (vec2(pixelCoord) + 0.5) / vec2(pc.resolution);
    float centerDepth = texture(depthBuffer, centerUv).r;
    vec3 centerWorldPos = reconstructWorldPos(centerUv, centerDepth, invViewProjection);
    vec3 centerN = normalize(imageLoad(normalImage, pixelCoord).rgb * 2.0 - 1.0);

    // Scene globals
    SceneGlobalsBuffer globals = SceneGlobalsBuffer(pc.sceneGlobalsAddress);

    // RNG seed
    uint seed = pcgHash(pixelIndex ^ pcgHash(pc.frameIndex * 104729u + 3u));

    // Initialize combined reservoir from center
    Reservoir combined;
    combined.selectedLight = center.selectedLight;
    combined.weightSum     = center.weightSum;
    combined.M             = center.M;
    combined.W             = center.W;

    // Sample spatial neighbors
    for (uint i = 0; i < NUM_SPATIAL_NEIGHBORS; i++) {
        // Random point in disk
        float angle = randomFloat(seed) * TWO_PI;
        float radius = sqrt(randomFloat(seed)) * SPATIAL_RADIUS;
        ivec2 offset = ivec2(cos(angle) * radius, sin(angle) * radius);
        ivec2 neighborCoord = pixelCoord + offset;

        // Bounds check
        if (any(lessThan(neighborCoord, ivec2(0))) ||
            any(greaterThanEqual(neighborCoord, ivec2(pc.resolution)))) {
            continue;
        }

        uint neighborIndex = neighborCoord.y * pc.resolution.x + neighborCoord.x;

        // Geometry similarity checks
        vec2 neighborUv = (vec2(neighborCoord) + 0.5) / vec2(pc.resolution);
        float neighborDepth = texture(depthBuffer, neighborUv).r;

        // Depth similarity: ratio of linear depths
        float depthRatio = (centerDepth > EPSILON && neighborDepth > EPSILON)
            ? max(centerDepth / neighborDepth, neighborDepth / centerDepth)
            : 100.0;

        if (depthRatio > DEPTH_THRESHOLD) {
            continue;
        }

        // Normal similarity
        vec3 neighborN = normalize(imageLoad(normalImage, neighborCoord).rgb * 2.0 - 1.0);
        if (dot(centerN, neighborN) < NORMAL_THRESHOLD) {
            continue;
        }

        // Read neighbor reservoir
        Reservoir neighbor = inputReservoirs[neighborIndex];
        if (neighbor.selectedLight == 0xFFFFFFFF || neighbor.M == 0) {
            continue;
        }

        // Evaluate neighbor's selected light at center pixel surface
        GPULight neighborLight = loadLight(globals.lightAddr, neighbor.selectedLight);
        float pHatAtCenter = targetPdf(neighborLight, centerWorldPos, centerN);

        // MIS weight: proportional to targetPdf * W * M
        float neighborWeight = pHatAtCenter * neighbor.W * float(neighbor.M);

        // WRS merge
        combined.weightSum += neighborWeight;
        combined.M += neighbor.M;

        float xi = randomFloat(seed);
        if (xi * combined.weightSum < neighborWeight) {
            combined.selectedLight = neighbor.selectedLight;
        }
    }

    // Finalize combined W
    if (combined.M > 0 && combined.selectedLight != 0xFFFFFFFF) {
        GPULight selected = loadLight(globals.lightAddr, combined.selectedLight);
        float pHat = targetPdf(selected, centerWorldPos, centerN);
        combined.W = (pHat > 0.0)
            ? (1.0 / max(float(combined.M), 1.0)) * (combined.weightSum / pHat)
            : 0.0;
    }

    outputReservoirs[pixelIndex] = combined;
}
