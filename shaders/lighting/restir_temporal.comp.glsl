#version 460

#extension GL_EXT_buffer_reference                     : require
#extension GL_EXT_buffer_reference2                    : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_scalar_block_layout                  : require

#include "types.glsl"
#include "math.glsl"

// ---------------------------------------------------------------------------
// ReSTIR DI — Temporal resampling
//
// Per pixel: reproject to previous frame UV using previous VP matrix, read
// previous frame's reservoir at reprojected position, combine current and
// previous reservoirs via WRS, and clamp previous M to 20x current for
// stability.
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
layout(set = 1, binding = 0)         uniform sampler2D  depthBuffer;
layout(set = 1, binding = 1, rgba16f) uniform readonly  image2D  normalImage;

// Current reservoir (input from candidate generation)
layout(set = 1, binding = 2, std430) readonly buffer ReservoirCurrent {
    Reservoir currentReservoirs[];
};

// Previous frame reservoir (read)
layout(set = 1, binding = 3, std430) readonly buffer ReservoirPrev {
    Reservoir prevReservoirs[];
};

// Output reservoir (temporal combined)
layout(set = 1, binding = 4, std430) writeonly buffer ReservoirOut {
    Reservoir outReservoirs[];
};

// Previous frame VP matrix passed via a small UBO
layout(set = 1, binding = 5, std140) uniform TemporalData {
    mat4 prevViewProjection;
    mat4 currentInvViewProjection;
};

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
// Target PDF (must match candidate generation)
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

    // Read current reservoir
    Reservoir current = currentReservoirs[pixelIndex];

    // If current reservoir is empty, pass through
    if (current.selectedLight == 0xFFFFFFFF) {
        outReservoirs[pixelIndex] = current;
        return;
    }

    // Reconstruct world position
    vec2 uv = (vec2(pixelCoord) + 0.5) / vec2(pc.resolution);
    float depth = texture(depthBuffer, uv).r;
    vec3 worldPos = reconstructWorldPos(uv, depth, currentInvViewProjection);

    // Normal
    vec3 N = normalize(imageLoad(normalImage, pixelCoord).rgb * 2.0 - 1.0);

    // Reproject to previous frame
    vec4 prevClip = prevViewProjection * vec4(worldPos, 1.0);
    vec2 prevNdc  = prevClip.xy / prevClip.w;
    vec2 prevUv   = prevNdc * 0.5 + 0.5;

    // RNG
    uint seed = pcgHash(pixelIndex ^ pcgHash(pc.frameIndex * 7919u + 1u));

    // Start combined reservoir from current
    Reservoir combined;
    combined.selectedLight = current.selectedLight;
    combined.weightSum     = current.weightSum;
    combined.M             = current.M;
    combined.W             = current.W;

    // Check if reprojection lands within the image
    if (all(greaterThanEqual(prevUv, vec2(0.0))) && all(lessThan(prevUv, vec2(1.0)))) {
        ivec2 prevPixel = ivec2(prevUv * vec2(pc.resolution));
        uint prevIndex  = prevPixel.y * pc.resolution.x + prevPixel.x;

        Reservoir prev = prevReservoirs[prevIndex];

        if (prev.selectedLight != 0xFFFFFFFF && prev.M > 0) {
            // Clamp previous M to 20x current M for stability
            uint maxPrevM = max(current.M * 20, 1u);
            if (prev.M > maxPrevM) {
                float scale = float(maxPrevM) / float(prev.M);
                prev.weightSum *= scale;
                prev.M = maxPrevM;
            }

            // Load previous light and compute its target PDF at current surface
            SceneGlobalsBuffer globals = SceneGlobalsBuffer(pc.sceneGlobalsAddress);
            GPULight prevLight = loadLight(globals.lightAddr, prev.selectedLight);
            float prevTargetPdf = targetPdf(prevLight, worldPos, N);

            // WRS: merge previous into combined
            float prevWeight = prevTargetPdf * prev.W * float(prev.M);
            combined.weightSum += prevWeight;
            combined.M += prev.M;

            float xi = randomFloat(seed);
            if (xi * combined.weightSum < prevWeight) {
                combined.selectedLight = prev.selectedLight;
            }

            // Recompute final W
            SceneGlobalsBuffer g2 = SceneGlobalsBuffer(pc.sceneGlobalsAddress);
            GPULight selected = loadLight(g2.lightAddr, combined.selectedLight);
            float pHat = targetPdf(selected, worldPos, N);
            combined.W = (pHat > 0.0)
                ? (1.0 / max(float(combined.M), 1.0)) * (combined.weightSum / pHat)
                : 0.0;
        }
    }

    outReservoirs[pixelIndex] = combined;
}
