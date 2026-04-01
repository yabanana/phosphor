#version 460

#extension GL_EXT_buffer_reference                     : require
#extension GL_EXT_buffer_reference2                    : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_scalar_block_layout                  : require
#extension GL_EXT_nonuniform_qualifier                 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

#include "types.glsl"
#include "bindless.glsl"
#include "math.glsl"
#include "packing.glsl"

// ---------------------------------------------------------------------------
// Debug overlay fragment shader
//
// Visualizes different buffers based on a specialization constant.
//   Mode 0: passthrough (final rendered color)
//   Mode 1: depth visualization (linearized, grayscale)
//   Mode 2: world-space normals (reconstructed from visibility buffer)
// ---------------------------------------------------------------------------

layout(constant_id = 0) const uint DEBUG_MODE = 0;

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

// Debug input images
layout(set = 1, binding = 0) uniform sampler2D finalColor;
layout(set = 1, binding = 1) uniform sampler2D depthBuffer;
layout(set = 1, binding = 2, r32ui) uniform readonly uimage2D visibilityBuffer;

// ---------------------------------------------------------------------------
// Mode 0: Passthrough
// ---------------------------------------------------------------------------

vec4 debugPassthrough() {
    return texture(finalColor, inUV);
}

// ---------------------------------------------------------------------------
// Mode 1: Depth visualization
// ---------------------------------------------------------------------------

vec4 debugDepth() {
    float depth = texture(depthBuffer, inUV).r;

    // Linearize with reasonable near/far for visualization
    // These could come from push constants in a real implementation
    const float near = 0.1;
    const float far  = 1000.0;
    float linear = linearizeDepth(depth, near, far);

    // Remap to visible range and apply pow for better contrast
    float normalized = clamp(linear / far, 0.0, 1.0);
    float vis = pow(1.0 - normalized, 2.0); // invert + gamma for contrast

    return vec4(vec3(vis), 1.0);
}

// ---------------------------------------------------------------------------
// Mode 2: Normal visualization
// ---------------------------------------------------------------------------

vec4 debugNormals() {
    ivec2 pixelCoord = ivec2(inUV * vec2(pc.resolution));
    uint visEncoded = imageLoad(visibilityBuffer, pixelCoord).r;

    if (visEncoded == VISIBILITY_CLEAR) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }

    uvec2 vis = decodeVisibility(visEncoded);
    uint instanceIdx = vis.x;
    uint triangleIdx = vis.y;

    SceneGlobalsBuffer globals = SceneGlobalsBuffer(pc.sceneGlobalsAddress);
    GPUInstance instance = loadInstance(globals.instanceAddr, instanceIdx);

    // Find the triangle's meshlet and vertices
    MeshInfoBuffer meshInfo = MeshInfoBuffer(globals.meshInfoAddr + uint64_t(instance.meshIndex) * 32);

    uint remainingTriangles = triangleIdx;
    uint foundMeshletIdx = 0;
    uint localTriIdx = 0;

    for (uint m = 0; m < meshInfo.meshletCount; m++) {
        uint globalMeshletIdx = meshInfo.meshletOffset + m;
        Meshlet meshlet = loadMeshlet(globals.meshletAddr, globalMeshletIdx);

        if (remainingTriangles < meshlet.triangleCount) {
            foundMeshletIdx = globalMeshletIdx;
            localTriIdx = remainingTriangles;
            break;
        }
        remainingTriangles -= meshlet.triangleCount;
    }

    Meshlet meshlet = loadMeshlet(globals.meshletAddr, foundMeshletIdx);

    MeshletTriangleBuffer triBuffer = MeshletTriangleBuffer(globals.meshletTriangleAddr);
    uint triBase = meshlet.triangleOffset + localTriIdx * 3;
    uint localIdx0 = uint(triBuffer.indices[triBase + 0]);
    uint localIdx1 = uint(triBuffer.indices[triBase + 1]);
    uint localIdx2 = uint(triBuffer.indices[triBase + 2]);

    MeshletVertexBuffer meshletVerts = MeshletVertexBuffer(globals.meshletVertexAddr);
    uint globalIdx0 = meshletVerts.indices[meshlet.vertexOffset + localIdx0];
    uint globalIdx1 = meshletVerts.indices[meshlet.vertexOffset + localIdx1];
    uint globalIdx2 = meshletVerts.indices[meshlet.vertexOffset + localIdx2];

    GPUVertex v0 = loadVertex(globals.vertexAddr, globalIdx0);
    GPUVertex v1 = loadVertex(globals.vertexAddr, globalIdx1);
    GPUVertex v2 = loadVertex(globals.vertexAddr, globalIdx2);

    // Compute screen-space barycentrics
    mat4 mvp = pc.viewProjection * instance.modelMatrix;
    vec2 resolution = vec2(pc.resolution);

    vec4 c0 = mvp * vec4(v0.px, v0.py, v0.pz, 1.0);
    vec4 c1 = mvp * vec4(v1.px, v1.py, v1.pz, 1.0);
    vec4 c2 = mvp * vec4(v2.px, v2.py, v2.pz, 1.0);

    vec2 sp0 = (c0.xy / c0.w * 0.5 + 0.5) * resolution;
    vec2 sp1 = (c1.xy / c1.w * 0.5 + 0.5) * resolution;
    vec2 sp2 = (c2.xy / c2.w * 0.5 + 0.5) * resolution;

    vec3 bary = computeBarycentrics(vec2(pixelCoord) + 0.5, sp0, sp1, sp2);

    // Interpolate normal
    vec3 n0 = vec3(v0.nx, v0.ny, v0.nz);
    vec3 n1 = vec3(v1.nx, v1.ny, v1.nz);
    vec3 n2 = vec3(v2.nx, v2.ny, v2.nz);
    vec3 localNormal = normalize(n0 * bary.x + n1 * bary.y + n2 * bary.z);

    mat3 normalMatrix = transpose(inverse(mat3(instance.modelMatrix)));
    vec3 worldNormal = normalize(normalMatrix * localNormal);

    // Map [-1,1] normal to [0,1] for visualization
    return vec4(worldNormal * 0.5 + 0.5, 1.0);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

void main() {
    switch (DEBUG_MODE) {
        case 0:  outColor = debugPassthrough(); break;
        case 1:  outColor = debugDepth();       break;
        case 2:  outColor = debugNormals();     break;
        default: outColor = debugPassthrough(); break;
    }
}
