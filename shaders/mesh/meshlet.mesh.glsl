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

    // Bail if this workgroup is beyond the payload count
    if (payloadIdx >= payload.count) {
        SetMeshOutputsEXT(0, 0);
        return;
    }

    uint meshletIdx  = payload.meshletIndices[payloadIdx];
    uint instanceIdx = payload.instanceIndices[payloadIdx];

    // Load scene globals
    SceneGlobalsBuffer globals = SceneGlobalsBuffer(pc.sceneGlobalsAddress);

    // Load instance and meshlet
    GPUInstance instance = loadInstance(globals.instanceAddr, instanceIdx);
    Meshlet meshlet = loadMeshlet(globals.meshletAddr, meshletIdx);

    uint vertexCount   = meshlet.vertexCount;
    uint triangleCount = meshlet.triangleCount;

    SetMeshOutputsEXT(vertexCount, triangleCount);

    // --- Emit vertices (loop: each invocation handles multiple vertices) ---
    mat4 model = instance.modelMatrix;
    mat4 mvp   = pc.viewProjection * model;
    mat3 normalMatrix = transpose(inverse(mat3(model)));

    MeshletVertexBuffer meshletVerts = MeshletVertexBuffer(globals.meshletVertexAddr);

    for (uint v = tid; v < vertexCount; v += 32) {
        uint globalVertexIdx = meshletVerts.indices[meshlet.vertexOffset + v];
        GPUVertex vtx = loadVertex(globals.vertexAddr, globalVertexIdx);

        vec4 localPos = vec4(vtx.px, vtx.py, vtx.pz, 1.0);
        gl_MeshVerticesEXT[v].gl_Position = mvp * localPos;

        outWorldPos[v]     = (model * localPos).xyz;
        outWorldNormal[v]  = normalize(normalMatrix * vec3(vtx.nx, vtx.ny, vtx.nz));
        outWorldTangent[v] = vec4(normalize(normalMatrix * vec3(vtx.tx, vtx.ty, vtx.tz)), vtx.tw);
        outUV[v]           = vec2(vtx.u, vtx.v);
    }

    // --- Emit triangle indices + per-primitive instanceID ---
    MeshletTriangleBuffer triBuf = MeshletTriangleBuffer(globals.meshletTriangleAddr);

    for (uint t = tid; t < triangleCount; t += 32) {
        uint triBase = meshlet.triangleOffset + t * 3;
        uint i0 = uint(triBuf.indices[triBase + 0]);
        uint i1 = uint(triBuf.indices[triBase + 1]);
        uint i2 = uint(triBuf.indices[triBase + 2]);
        gl_PrimitiveTriangleIndicesEXT[t] = uvec3(i0, i1, i2);
        outInstanceID[t] = instanceIdx;
    }
}
