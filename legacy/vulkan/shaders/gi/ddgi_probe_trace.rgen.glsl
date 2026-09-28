#version 460
#extension GL_EXT_ray_tracing                              : require
#extension GL_EXT_buffer_reference                         : require
#extension GL_EXT_buffer_reference2                        : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64   : require
#extension GL_EXT_scalar_block_layout                      : require

// ---------------------------------------------------------------------------
// DDGI Probe Ray Generation Shader
//
// gl_LaunchIDEXT.x  = ray index   [0 .. raysPerProbe-1]
// gl_LaunchIDEXT.y  = probe index [0 .. totalProbes-1]
//
// Each invocation traces one ray from a probe origin in a direction computed
// via spherical Fibonacci sampling, and writes the result (radiance + hit
// distance) into a storage buffer for the update passes to consume.
// ---------------------------------------------------------------------------

// -- DDGI uniforms ----------------------------------------------------------

struct DDGIUniforms {
    ivec3  probeGridDims;      // e.g. 8, 4, 8
    float  probeSpacing;
    vec3   probeGridOrigin;
    float  maxRayDistance;
    uint   raysPerProbe;       // 256
    float  hysteresis;         // 0.97
    float  irradianceGamma;
    float  pad;
};

layout(set = 1, binding = 0, scalar) readonly buffer DDGIUniformsBuffer {
    DDGIUniforms ddgi;
};

// -- Ray data output --------------------------------------------------------

struct RayData {
    vec3  radiance;
    float hitDistance;   // negative => miss (sky)
};

layout(set = 1, binding = 1, scalar) writeonly buffer RayDataBuffer {
    RayData rayData[];
};

// -- Acceleration structure --------------------------------------------------

layout(set = 1, binding = 2) uniform accelerationStructureEXT topLevelAS;

// -- Scene globals (BDA root pointer) ----------------------------------------

layout(set = 1, binding = 3, scalar) readonly buffer SceneGlobalsRef {
    uint64_t vertexAddr;
    uint64_t meshletAddr;
    uint64_t meshletVertexAddr;
    uint64_t meshletTriangleAddr;
    uint64_t instanceAddr;
    uint64_t materialAddr;
    uint64_t meshInfoAddr;
    uint64_t lightAddr;
    uint64_t meshletBoundsAddr;
    uint     instanceCount;
    uint     lightCount;
    uint     meshletTotalCount;
    uint     scenePad;
} sceneGlobals;

// -- Ray payload (shared with miss/closest-hit) ------------------------------

layout(location = 0) rayPayloadEXT vec4 payload; // xyz = radiance, w = hitDistance

// ---------------------------------------------------------------------------
// Spherical Fibonacci sampling — produces a quasi-uniform distribution of
// directions on the unit sphere.  The golden-ratio offset provides good
// stratification across probes when combined with per-frame random rotation.
// ---------------------------------------------------------------------------

const float GOLDEN_RATIO = 1.6180339887498948482;

vec3 sphericalFibonacci(uint index, uint sampleCount) {
    float i  = float(index) + 0.5;
    float n  = float(sampleCount);

    // Cosine of polar angle: uniform in [-1, 1]
    float cosTheta = 1.0 - 2.0 * i / n;
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));

    // Azimuthal angle based on golden ratio
    float phi = 2.0 * 3.14159265358979323846 * fract(i / GOLDEN_RATIO);

    return vec3(sinTheta * cos(phi),
                sinTheta * sin(phi),
                cosTheta);
}

// ---------------------------------------------------------------------------
// Probe grid helpers
// ---------------------------------------------------------------------------

vec3 probeWorldPosition(uint probeIndex) {
    ivec3 dims = ddgi.probeGridDims;
    int ix = int(probeIndex) % dims.x;
    int iy = (int(probeIndex) / dims.x) % dims.y;
    int iz = int(probeIndex) / (dims.x * dims.y);

    return ddgi.probeGridOrigin + vec3(float(ix), float(iy), float(iz)) * ddgi.probeSpacing;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

void main() {
    uint rayIndex   = gl_LaunchIDEXT.x;
    uint probeIndex = gl_LaunchIDEXT.y;

    vec3 origin    = probeWorldPosition(probeIndex);
    vec3 direction = sphericalFibonacci(rayIndex, ddgi.raysPerProbe);

    // Initialise payload to zero (miss shader will fill sky colour)
    payload = vec4(0.0, 0.0, 0.0, -1.0);

    uint  rayFlags  = gl_RayFlagsOpaqueEXT;
    float tMin      = 0.01;  // small offset to avoid self-intersection
    float tMax      = ddgi.maxRayDistance;

    traceRayEXT(
        topLevelAS,
        rayFlags,
        0xFF,            // cull mask — hit everything
        0,               // SBT record offset (hit group 0)
        0,               // SBT record stride
        0,               // miss shader index 0
        origin,
        tMin,
        direction,
        tMax,
        0                // payload location 0
    );

    // Write result
    uint outputIndex = probeIndex * ddgi.raysPerProbe + rayIndex;
    rayData[outputIndex].radiance    = payload.xyz;
    rayData[outputIndex].hitDistance  = payload.w;
}
