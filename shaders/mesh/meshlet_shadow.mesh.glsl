#version 460

#extension GL_EXT_mesh_shader                          : require
#extension GL_EXT_buffer_reference                     : require
#extension GL_EXT_buffer_reference2                    : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_scalar_block_layout                  : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

#include "types.glsl"

// ---------------------------------------------------------------------------
// Shadow mesh shader — depth-only meshlet rasterization.
//
// Transforms vertices by the light's VP matrix (in pc.viewProjection) and
// outputs position only.  No per-vertex attributes are needed since we only
// write to the depth buffer.
//
// Each workgroup processes one meshlet (up to 64 vertices, 124 triangles).
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

// No per-vertex or per-primitive outputs needed for shadow pass
// (depth-only rendering — gl_Position is sufficient)

void main() {
    uint tid = gl_LocalInvocationIndex;
    uint payloadIdx = gl_WorkGroupID.x;

    if (payloadIdx >= payload.count) {
        SetMeshOutputsEXT(0, 0);
        return;
    }

    uint meshletIdx  = payload.meshletIndices[payloadIdx];
    uint instanceIdx = payload.instanceIndices[payloadIdx];

    // Load scene globals
    SceneGlobalsBuffer globals = SceneGlobalsBuffer(pc.sceneGlobalsAddress);

    // Load meshlet descriptor
    Meshlet meshlet = loadMeshlet(globals.meshletAddr, meshletIdx);

    // Load instance transform
    GPUInstance instance = loadInstance(globals.instanceAddr, instanceIdx);

    // Light VP * model matrix
    mat4 lightMVP = pc.viewProjection * instance.modelMatrix;

    // Declare output counts
    SetMeshOutputsEXT(meshlet.vertexCount, meshlet.triangleCount);

    // -----------------------------------------------------------------------
    // Emit vertices: position only (depth-only pass)
    // -----------------------------------------------------------------------
    MeshletVertexBuffer meshletVerts = MeshletVertexBuffer(globals.meshletVertexAddr);

    for (uint i = tid; i < meshlet.vertexCount; i += 32) {
        uint globalVertexIdx = meshletVerts.indices[meshlet.vertexOffset + i];

        // Load vertex position via BDA
        GPUVertex vtx = loadVertex(globals.vertexAddr, globalVertexIdx);
        vec3 localPos = vec3(vtx.px, vtx.py, vtx.pz);

        gl_MeshVerticesEXT[i].gl_Position = lightMVP * vec4(localPos, 1.0);
    }

    // -----------------------------------------------------------------------
    // Emit triangle indices
    // -----------------------------------------------------------------------
    MeshletTriangleBuffer meshletTris = MeshletTriangleBuffer(globals.meshletTriangleAddr);

    for (uint i = tid; i < meshlet.triangleCount; i += 32) {
        uint triBase = meshlet.triangleOffset + i * 3;
        uint idx0 = uint(meshletTris.indices[triBase + 0]);
        uint idx1 = uint(meshletTris.indices[triBase + 1]);
        uint idx2 = uint(meshletTris.indices[triBase + 2]);

        gl_PrimitiveTriangleIndicesEXT[i] = uvec3(idx0, idx1, idx2);
    }
}
