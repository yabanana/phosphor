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

    // Safety: don't process beyond the compacted count
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

    mat4 mvp = pc.viewProjection * instance.modelMatrix;
    mat3 normalMatrix = transpose(inverse(mat3(instance.modelMatrix)));

    // Declare output counts
    SetMeshOutputsEXT(meshlet.vertexCount, meshlet.triangleCount);

    // -----------------------------------------------------------------------
    // Emit vertices: each thread processes ceil(vertexCount / 32) vertices
    // -----------------------------------------------------------------------
    MeshletVertexBuffer meshletVerts = MeshletVertexBuffer(globals.meshletVertexAddr);

    for (uint i = tid; i < meshlet.vertexCount; i += 32) {
        // Meshlet vertex buffer stores global vertex indices
        uint globalVertexIdx = meshletVerts.indices[meshlet.vertexOffset + i];

        // Load vertex data via BDA
        GPUVertex vtx = loadVertex(globals.vertexAddr, globalVertexIdx);

        vec3 localPos = vec3(vtx.px, vtx.py, vtx.pz);
        vec4 worldPos = instance.modelMatrix * vec4(localPos, 1.0);
        vec3 worldNormal = normalize(normalMatrix * vec3(vtx.nx, vtx.ny, vtx.nz));
        vec3 worldTangent = normalize(normalMatrix * vec3(vtx.tx, vtx.ty, vtx.tz));

        gl_MeshVerticesEXT[i].gl_Position = mvp * vec4(localPos, 1.0);

        outUV[i]           = vec2(vtx.u, vtx.v);
        outWorldPos[i]     = worldPos.xyz;
        outWorldNormal[i]  = worldNormal;
        outWorldTangent[i] = vec4(worldTangent, vtx.tw);
    }

    // -----------------------------------------------------------------------
    // Emit triangle indices: each thread processes ceil(triangleCount / 32)
    // Triangle indices are stored as uint8_t triplets in the meshlet triangle buffer.
    // -----------------------------------------------------------------------
    MeshletTriangleBuffer meshletTris = MeshletTriangleBuffer(globals.meshletTriangleAddr);

    for (uint i = tid; i < meshlet.triangleCount; i += 32) {
        uint triBase = meshlet.triangleOffset + i * 3;
        uint idx0 = uint(meshletTris.indices[triBase + 0]);
        uint idx1 = uint(meshletTris.indices[triBase + 1]);
        uint idx2 = uint(meshletTris.indices[triBase + 2]);

        gl_PrimitiveTriangleIndicesEXT[i] = uvec3(idx0, idx1, idx2);

        // Per-primitive: store instance ID for the visibility buffer
        outInstanceID[i] = instanceIdx;
    }
}
