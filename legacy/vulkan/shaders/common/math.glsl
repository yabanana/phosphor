#ifndef MATH_GLSL
#define MATH_GLSL

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const float PI         = 3.14159265358979323846;
const float TWO_PI     = 6.28318530717958647692;
const float HALF_PI    = 1.57079632679489661923;
const float INV_PI     = 0.31830988618379067154;
const float EPSILON    = 1e-6;

// ---------------------------------------------------------------------------
// Depth utilities
// ---------------------------------------------------------------------------

// Linearize a Vulkan reversed-Z depth value to view-space distance.
// In reversed-Z: near maps to 1.0, far maps to 0.0.
// Projection matrix has: z_ndc = near / z_view  (for reversed-Z infinite far).
float linearizeDepth(float z, float near, float far) {
    // Standard Vulkan perspective with reversed-Z:
    // z_ndc = near * far / (far - z_view * (far - near))
    // Solving for z_view:
    return near * far / (far - z * (far - near));
}

// Linearize depth for reversed-Z infinite far plane (far = infinity).
// z_ndc = near / z_view  =>  z_view = near / z_ndc
float linearizeDepthInf(float z, float near) {
    return near / max(z, EPSILON);
}

// ---------------------------------------------------------------------------
// World position reconstruction from depth buffer
// ---------------------------------------------------------------------------

// Reconstruct world-space position from UV [0,1], depth [0,1], and inverse VP.
vec3 reconstructWorldPos(vec2 uv, float depth, mat4 invVP) {
    // UV to NDC: x,y in [-1,1], z stays in [0,1] for Vulkan
    vec4 clipPos = vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec4 worldPos = invVP * clipPos;
    return worldPos.xyz / worldPos.w;
}

// ---------------------------------------------------------------------------
// Octahedral normal encoding/decoding
// Maps a unit normal (S2) to [-1,1]^2 and back, losslessly for FP16 storage.
// Reference: "Survey of Efficient Representations for Independent Unit Vectors"
//            (Cigolle, Donow, Evangelakos, Mara, McGuire, Meyer — JCGT 2014)
// ---------------------------------------------------------------------------

// Encode a unit normal to octahedral representation in [-1,1]^2
vec2 octEncode(vec3 n) {
    // Project onto L1-norm unit octahedron
    vec3 a = n / (abs(n.x) + abs(n.y) + abs(n.z));
    // Wrap the lower hemisphere
    if (a.z < 0.0) {
        a.xy = (1.0 - abs(a.yx)) * vec2(a.x >= 0.0 ? 1.0 : -1.0,
                                          a.y >= 0.0 ? 1.0 : -1.0);
    }
    return a.xy;
}

// Decode an octahedral-encoded vec2 back to a unit normal
vec3 octDecode(vec2 e) {
    vec3 n = vec3(e.xy, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0) {
        n.xy = (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0,
                                          n.y >= 0.0 ? 1.0 : -1.0);
    }
    return normalize(n);
}

// ---------------------------------------------------------------------------
// Octahedral encoding packed into a uint (SNORM 16-bit per component)
// ---------------------------------------------------------------------------

uint octEncodeSnorm16(vec3 n) {
    vec2 e = octEncode(n);
    // Map [-1,1] to [0,65535]
    uvec2 packed = uvec2(round(clamp(e * 0.5 + 0.5, 0.0, 1.0) * 65535.0));
    return packed.x | (packed.y << 16);
}

vec3 octDecodeSnorm16(uint packed) {
    vec2 e;
    e.x = float(packed & 0xFFFF) / 65535.0 * 2.0 - 1.0;
    e.y = float(packed >> 16)    / 65535.0 * 2.0 - 1.0;
    return octDecode(e);
}

// ---------------------------------------------------------------------------
// Screen-space barycentric coordinate computation
//
// Given three screen-space triangle vertices (after perspective divide) and
// a target pixel position, compute barycentric weights (u, v, w) such that
// P = u*v0 + v*v1 + w*v2 with u + v + w = 1.
// ---------------------------------------------------------------------------

vec3 computeBarycentrics(vec2 p, vec2 a, vec2 b, vec2 c) {
    vec2 v0 = b - a;
    vec2 v1 = c - a;
    vec2 v2 = p - a;

    float d00 = dot(v0, v0);
    float d01 = dot(v0, v1);
    float d11 = dot(v1, v1);
    float d20 = dot(v2, v0);
    float d21 = dot(v2, v1);

    float invDenom = 1.0 / (d00 * d11 - d01 * d01);
    float v = (d11 * d20 - d01 * d21) * invDenom;
    float w = (d00 * d21 - d01 * d20) * invDenom;
    float u = 1.0 - v - w;

    return vec3(u, v, w);
}

#endif // MATH_GLSL
