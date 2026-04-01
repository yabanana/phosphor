#version 460
#extension GL_EXT_scalar_block_layout : require

// ---------------------------------------------------------------------------
// DDGI Probe Irradiance Update — Compute Shader
//
// Each workgroup processes one probe.  Each thread within the workgroup
// handles one texel of the 8x8 octahedral irradiance map for that probe.
//
// For every texel, we:
//   1. Compute the world-space direction from the octahedral UV.
//   2. Accumulate irradiance from all rays weighted by max(0, dot(rayDir, texelDir)).
//   3. Blend with the previous frame using exponential hysteresis.
//   4. Write to the irradiance atlas.
//
// Dispatch: (1, totalProbes, 1) — one workgroup per probe.
// Workgroup: (8, 8, 1) — one thread per texel.
// ---------------------------------------------------------------------------

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

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

// -- Irradiance atlas (R11G11B10F, 8x8 per probe, tiled in a 2D texture) ---

layout(set = 1, binding = 4, r11f_g11f_b10f) uniform image2D irradianceAtlas;

// ---------------------------------------------------------------------------
// Spherical Fibonacci — must match the rgen shader exactly
// ---------------------------------------------------------------------------

const float GOLDEN_RATIO = 1.6180339887498948482;
const float PI = 3.14159265358979323846;

vec3 sphericalFibonacci(uint index, uint sampleCount) {
    float i  = float(index) + 0.5;
    float n  = float(sampleCount);

    float cosTheta = 1.0 - 2.0 * i / n;
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));

    float phi = 2.0 * PI * fract(i / GOLDEN_RATIO);

    return vec3(sinTheta * cos(phi),
                sinTheta * sin(phi),
                cosTheta);
}

// ---------------------------------------------------------------------------
// Octahedral mapping: UV in [0,1]^2 -> unit direction on sphere
// ---------------------------------------------------------------------------

vec3 octDecode(vec2 e) {
    // Map [0,1] -> [-1,1]
    e = e * 2.0 - 1.0;

    vec3 n = vec3(e.xy, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0) {
        n.xy = (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0,
                                          n.y >= 0.0 ? 1.0 : -1.0);
    }
    return normalize(n);
}

// ---------------------------------------------------------------------------
// Atlas coordinate helper — converts (texelX, texelY, probeIndex) to the
// 2D position in the tiled atlas.
//
// Layout: probes are tiled in rows of probeGridDims.x * probeGridDims.z,
// with probeGridDims.y rows.  Each probe occupies IRRADIANCE_PROBE_SIZE
// texels plus a 1-texel border on each side.
// ---------------------------------------------------------------------------

const int IRRADIANCE_PROBE_SIZE = 8;
const int IRRADIANCE_WITH_BORDER = IRRADIANCE_PROBE_SIZE + 2; // 1-texel border

ivec2 probeAtlasOffset(uint probeIndex) {
    ivec3 dims = ddgi.probeGridDims;
    int ix = int(probeIndex) % dims.x;
    int iy = (int(probeIndex) / dims.x) % dims.y;
    int iz = int(probeIndex) / (dims.x * dims.y);

    // Tile probes in a 2D grid: columns = ix + iz * dims.x, rows = iy
    int col = ix + iz * dims.x;
    int row = iy;

    return ivec2(col * IRRADIANCE_WITH_BORDER + 1,   // +1 for border
                 row * IRRADIANCE_WITH_BORDER + 1);
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

    // Texel centre UV in [0,1]^2 within the octahedral map
    vec2 texelUV = (vec2(texelX, texelY) + 0.5) / float(IRRADIANCE_PROBE_SIZE);
    vec3 texelDir = octDecode(texelUV);

    // Accumulate irradiance from all rays for this probe
    vec3  irradiance  = vec3(0.0);
    float totalWeight = 0.0;

    uint rayBase = probeIndex * ddgi.raysPerProbe;

    for (uint r = 0; r < ddgi.raysPerProbe; ++r) {
        vec3  rayDir = sphericalFibonacci(r, ddgi.raysPerProbe);
        RayData rd   = rayData[rayBase + r];

        // Cosine weight: how aligned is this ray with the texel direction
        float weight = max(0.0, dot(rayDir, texelDir));
        if (weight <= 0.0) continue;

        // Ignore back-face hits (very short distances can indicate probe
        // embedded in geometry — use a small threshold).
        if (rd.hitDistance >= 0.0 && rd.hitDistance < 0.001) continue;

        irradiance  += rd.radiance * weight;
        totalWeight += weight;
    }

    if (totalWeight > 0.0) {
        irradiance /= totalWeight;
    }

    // Apply perceptual encoding (gamma) for better precision in low-light
    if (ddgi.irradianceGamma > 0.0) {
        irradiance = pow(max(irradiance, vec3(0.0)), vec3(1.0 / ddgi.irradianceGamma));
    }

    // Blend with previous frame (temporal hysteresis)
    ivec2 atlasOffset = probeAtlasOffset(probeIndex);
    ivec2 atlasCoord  = atlasOffset + ivec2(texelX, texelY);

    vec3 previous = imageLoad(irradianceAtlas, atlasCoord).rgb;
    vec3 blended  = mix(irradiance, previous, ddgi.hysteresis);

    imageStore(irradianceAtlas, atlasCoord, vec4(blended, 1.0));

    // -- Update border texels for bilinear filtering --
    // Border texels mirror the interior edge texels.  Each thread on the
    // edge writes its corresponding border pixel.

    ivec2 borderBase = atlasOffset - ivec2(1); // top-left corner including border

    // Left column border
    if (texelX == 0) {
        ivec2 mirror = atlasOffset + ivec2(IRRADIANCE_PROBE_SIZE - 1, int(texelY));
        vec3 val = imageLoad(irradianceAtlas, mirror).rgb;
        // Use blended value for consistency; however the mirror source may
        // not have been written yet in this dispatch (same workgroup though),
        // so we write our own blended value mirrored.
        imageStore(irradianceAtlas, borderBase + ivec2(0, int(texelY) + 1), vec4(blended, 1.0));
    }
    // Right column border
    if (texelX == IRRADIANCE_PROBE_SIZE - 1) {
        imageStore(irradianceAtlas, borderBase + ivec2(IRRADIANCE_WITH_BORDER - 1, int(texelY) + 1),
                   vec4(blended, 1.0));
    }
    // Top row border
    if (texelY == 0) {
        imageStore(irradianceAtlas, borderBase + ivec2(int(texelX) + 1, 0),
                   vec4(blended, 1.0));
    }
    // Bottom row border
    if (texelY == IRRADIANCE_PROBE_SIZE - 1) {
        imageStore(irradianceAtlas, borderBase + ivec2(int(texelX) + 1, IRRADIANCE_WITH_BORDER - 1),
                   vec4(blended, 1.0));
    }

    // Corner texels (only 4 threads handle these)
    if (texelX == 0 && texelY == 0) {
        imageStore(irradianceAtlas, borderBase, vec4(blended, 1.0));
    }
    if (texelX == IRRADIANCE_PROBE_SIZE - 1 && texelY == 0) {
        imageStore(irradianceAtlas, borderBase + ivec2(IRRADIANCE_WITH_BORDER - 1, 0),
                   vec4(blended, 1.0));
    }
    if (texelX == 0 && texelY == IRRADIANCE_PROBE_SIZE - 1) {
        imageStore(irradianceAtlas, borderBase + ivec2(0, IRRADIANCE_WITH_BORDER - 1),
                   vec4(blended, 1.0));
    }
    if (texelX == IRRADIANCE_PROBE_SIZE - 1 && texelY == IRRADIANCE_PROBE_SIZE - 1) {
        imageStore(irradianceAtlas,
                   borderBase + ivec2(IRRADIANCE_WITH_BORDER - 1, IRRADIANCE_WITH_BORDER - 1),
                   vec4(blended, 1.0));
    }
}
