#version 460
#extension GL_EXT_scalar_block_layout : require

// ---------------------------------------------------------------------------
// DDGI Probe Visibility Update — Compute Shader
//
// Same structure as irradiance update, but accumulates hit distance and
// distance-squared for Chebyshev visibility testing.
//
// Each workgroup processes one probe.  Each thread handles one texel of
// the 16x16 octahedral visibility map.
//
// Dispatch: (1, totalProbes, 1)
// Workgroup: (16, 16, 1)
// ---------------------------------------------------------------------------

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// -- DDGI uniforms ----------------------------------------------------------

struct DDGIUniforms {
    ivec3  probeGridDims;
    float  probeSpacing;
    vec3   probeGridOrigin;
    float  maxRayDistance;
    uint   raysPerProbe;
    float  hysteresis;
    float  irradianceGamma;
    float  pad;
};

layout(set = 1, binding = 0, scalar) readonly buffer DDGIUniformsBuffer {
    DDGIUniforms ddgi;
};

// -- Ray data input ---------------------------------------------------------

struct RayData {
    vec3  radiance;
    float hitDistance;
};

layout(set = 1, binding = 1, scalar) readonly buffer RayDataBuffer {
    RayData rayData[];
};

// -- Visibility atlas (RG16F: R = mean distance, G = mean distance^2) ------

layout(set = 1, binding = 5, rg16f) uniform image2D visibilityAtlas;

// ---------------------------------------------------------------------------
// Spherical Fibonacci — must match rgen and irradiance shaders
// ---------------------------------------------------------------------------

const float GOLDEN_RATIO = 1.6180339887498948482;
const float PI = 3.14159265358979323846;

vec3 sphericalFibonacci(uint index, uint sampleCount) {
    float i  = float(index) + 0.5;
    float n  = float(sampleCount);

    float cosTheta = 1.0 - 2.0 * i / n;
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    float phi      = 2.0 * PI * fract(i / GOLDEN_RATIO);

    return vec3(sinTheta * cos(phi),
                sinTheta * sin(phi),
                cosTheta);
}

// ---------------------------------------------------------------------------
// Octahedral decode
// ---------------------------------------------------------------------------

vec3 octDecode(vec2 e) {
    e = e * 2.0 - 1.0;
    vec3 n = vec3(e.xy, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0) {
        n.xy = (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0,
                                          n.y >= 0.0 ? 1.0 : -1.0);
    }
    return normalize(n);
}

// ---------------------------------------------------------------------------
// Atlas tiling helpers
// ---------------------------------------------------------------------------

const int VISIBILITY_PROBE_SIZE = 16;
const int VISIBILITY_WITH_BORDER = VISIBILITY_PROBE_SIZE + 2;

ivec2 probeAtlasOffset(uint probeIndex) {
    ivec3 dims = ddgi.probeGridDims;
    int ix = int(probeIndex) % dims.x;
    int iy = (int(probeIndex) / dims.x) % dims.y;
    int iz = int(probeIndex) / (dims.x * dims.y);

    int col = ix + iz * dims.x;
    int row = iy;

    return ivec2(col * VISIBILITY_WITH_BORDER + 1,
                 row * VISIBILITY_WITH_BORDER + 1);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

void main() {
    uint probeIndex = gl_WorkGroupID.y;
    uint texelX     = gl_LocalInvocationID.x;
    uint texelY     = gl_LocalInvocationID.y;

    uint totalProbes = uint(ddgi.probeGridDims.x) *
                       uint(ddgi.probeGridDims.y) *
                       uint(ddgi.probeGridDims.z);
    if (probeIndex >= totalProbes) return;

    // Texel direction
    vec2 texelUV  = (vec2(texelX, texelY) + 0.5) / float(VISIBILITY_PROBE_SIZE);
    vec3 texelDir = octDecode(texelUV);

    // Accumulate distance and distance^2 weighted by cosine
    float meanDist   = 0.0;
    float meanDist2  = 0.0;
    float totalWeight = 0.0;

    uint rayBase = probeIndex * ddgi.raysPerProbe;

    for (uint r = 0; r < ddgi.raysPerProbe; ++r) {
        vec3 rayDir = sphericalFibonacci(r, ddgi.raysPerProbe);
        RayData rd  = rayData[rayBase + r];

        float weight = max(0.0, dot(rayDir, texelDir));
        if (weight <= 0.0) continue;

        // For misses, use maxRayDistance
        float dist = rd.hitDistance;
        if (dist < 0.0) {
            dist = ddgi.maxRayDistance;
        }

        meanDist   += dist * weight;
        meanDist2  += dist * dist * weight;
        totalWeight += weight;
    }

    if (totalWeight > 0.0) {
        meanDist  /= totalWeight;
        meanDist2 /= totalWeight;
    } else {
        meanDist  = ddgi.maxRayDistance;
        meanDist2 = ddgi.maxRayDistance * ddgi.maxRayDistance;
    }

    // Blend with previous frame (temporal hysteresis)
    ivec2 atlasOffset = probeAtlasOffset(probeIndex);
    ivec2 atlasCoord  = atlasOffset + ivec2(texelX, texelY);

    vec2 previous = imageLoad(visibilityAtlas, atlasCoord).rg;
    vec2 current  = vec2(meanDist, meanDist2);
    vec2 blended  = mix(current, previous, ddgi.hysteresis);

    imageStore(visibilityAtlas, atlasCoord, vec4(blended, 0.0, 1.0));

    // -- Update border texels for bilinear filtering --
    ivec2 borderBase = atlasOffset - ivec2(1);

    if (texelX == 0) {
        imageStore(visibilityAtlas, borderBase + ivec2(0, int(texelY) + 1),
                   vec4(blended, 0.0, 1.0));
    }
    if (texelX == VISIBILITY_PROBE_SIZE - 1) {
        imageStore(visibilityAtlas,
                   borderBase + ivec2(VISIBILITY_WITH_BORDER - 1, int(texelY) + 1),
                   vec4(blended, 0.0, 1.0));
    }
    if (texelY == 0) {
        imageStore(visibilityAtlas, borderBase + ivec2(int(texelX) + 1, 0),
                   vec4(blended, 0.0, 1.0));
    }
    if (texelY == VISIBILITY_PROBE_SIZE - 1) {
        imageStore(visibilityAtlas,
                   borderBase + ivec2(int(texelX) + 1, VISIBILITY_WITH_BORDER - 1),
                   vec4(blended, 0.0, 1.0));
    }

    // Corners
    if (texelX == 0 && texelY == 0) {
        imageStore(visibilityAtlas, borderBase, vec4(blended, 0.0, 1.0));
    }
    if (texelX == VISIBILITY_PROBE_SIZE - 1 && texelY == 0) {
        imageStore(visibilityAtlas,
                   borderBase + ivec2(VISIBILITY_WITH_BORDER - 1, 0),
                   vec4(blended, 0.0, 1.0));
    }
    if (texelX == 0 && texelY == VISIBILITY_PROBE_SIZE - 1) {
        imageStore(visibilityAtlas,
                   borderBase + ivec2(0, VISIBILITY_WITH_BORDER - 1),
                   vec4(blended, 0.0, 1.0));
    }
    if (texelX == VISIBILITY_PROBE_SIZE - 1 && texelY == VISIBILITY_PROBE_SIZE - 1) {
        imageStore(visibilityAtlas,
                   borderBase + ivec2(VISIBILITY_WITH_BORDER - 1, VISIBILITY_WITH_BORDER - 1),
                   vec4(blended, 0.0, 1.0));
    }
}
