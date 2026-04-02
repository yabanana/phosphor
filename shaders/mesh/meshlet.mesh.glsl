#version 460

#extension GL_EXT_mesh_shader                          : require
#extension GL_EXT_buffer_reference                     : require
#extension GL_EXT_buffer_reference2                    : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_scalar_block_layout                  : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

#include "types.glsl"

// ---------------------------------------------------------------------------
// Mesh shader — meshlet rasterization
//
// Each workgroup processes one meshlet (up to 64 vertices, 124 triangles).
// Vertices are fetched via BDA, transformed to clip space, and output.
// Triangle indices are packed as uint8_t triplets in the meshlet triangle buffer.
// ---------------------------------------------------------------------------

layout(local_size_x = 32) in;
layout(triangles, max_vertices = 64, max_primitives = 124) out;

// Payload from task shader
struct TaskPayload {
    uint meshletIndices[32];
    uint instanceIndices[32];
    uint count;
};

taskPayloadSharedEXT TaskPayload payload;

// Per-vertex outputs (interpolated)
layout(location = 0) out vec2 outUV[];
layout(location = 1) out vec3 outWorldPos[];
layout(location = 2) out vec3 outWorldNormal[];
layout(location = 3) out vec4 outWorldTangent[];

// Per-primitive outputs (flat, not interpolated)
layout(location = 4) perprimitiveEXT flat out uint outInstanceID[];

void main() {
    uint tid = gl_LocalInvocationIndex;
    uint payloadIdx = gl_WorkGroupID.x;

    // DEBUG: emit a hardcoded triangle — no BDA access at all
    SetMeshOutputsEXT(3, 1);

    if (tid < 3) {
        vec2 positions[3] = vec2[](
            vec2(-0.5, -0.5),
            vec2( 0.5, -0.5),
            vec2( 0.0,  0.5)
        );
        gl_MeshVerticesEXT[tid].gl_Position = vec4(positions[tid], 0.0, 1.0);
        outUV[tid] = vec2(0.0);
        outWorldPos[tid] = vec3(0.0);
        outWorldNormal[tid] = vec3(0.0, 0.0, 1.0);
        outWorldTangent[tid] = vec4(1.0, 0.0, 0.0, 1.0);
    }
    if (tid == 0) {
        gl_PrimitiveTriangleIndicesEXT[0] = uvec3(0, 1, 2);
        outInstanceID[0] = 0;
    }
}
