#version 460

#extension GL_EXT_mesh_shader                          : require
#extension GL_EXT_buffer_reference                     : require
#extension GL_EXT_buffer_reference2                    : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_scalar_block_layout                  : require
#extension GL_KHR_shader_subgroup_ballot               : enable

#include "types.glsl"

// ---------------------------------------------------------------------------
// Shadow task shader — meshlet culling against light-space frustum.
//
// Identical structure to the main task shader (meshlet.task.glsl) but culls
// against the light's view-projection matrix stored in pc.viewProjection.
// No backface cone culling is performed for shadow maps (we want to render
// both front and back faces to avoid light leaking through thin geometry).
//
// Dispatch: one workgroup per 32 meshlet candidates.
// ---------------------------------------------------------------------------

layout(local_size_x = 32) in;

// Payload shared between task and mesh shader
struct TaskPayload {
    uint meshletIndices[32];
    uint instanceIndices[32];
    uint count;
};

taskPayloadSharedEXT TaskPayload payload;

// ---------------------------------------------------------------------------
// Frustum planes extracted from the viewProjection matrix (Gribb-Hartmann)
// ---------------------------------------------------------------------------

void extractFrustumPlanes(mat4 vp, out vec4 planes[6]) {
    // Left
    planes[0] = vec4(vp[0][3] + vp[0][0],
                      vp[1][3] + vp[1][0],
                      vp[2][3] + vp[2][0],
                      vp[3][3] + vp[3][0]);
    // Right
    planes[1] = vec4(vp[0][3] - vp[0][0],
                      vp[1][3] - vp[1][0],
                      vp[2][3] - vp[2][0],
                      vp[3][3] - vp[3][0]);
    // Bottom
    planes[2] = vec4(vp[0][3] + vp[0][1],
                      vp[1][3] + vp[1][1],
                      vp[2][3] + vp[2][1],
                      vp[3][3] + vp[3][1]);
    // Top
    planes[3] = vec4(vp[0][3] - vp[0][1],
                      vp[1][3] - vp[1][1],
                      vp[2][3] - vp[2][1],
                      vp[3][3] - vp[3][1]);
    // Near (Vulkan: [0,1] depth range)
    planes[4] = vec4(vp[0][2],
                      vp[1][2],
                      vp[2][2],
                      vp[3][2]);
    // Far
    planes[5] = vec4(vp[0][3] - vp[0][2],
                      vp[1][3] - vp[1][2],
                      vp[2][3] - vp[2][2],
                      vp[3][3] - vp[3][2]);

    // Normalize
    for (int i = 0; i < 6; i++) {
        float len = length(planes[i].xyz);
        planes[i] /= len;
    }
}

// ---------------------------------------------------------------------------
// Frustum cull a bounding sphere (world space) against 6 planes.
// Returns true if the sphere is OUTSIDE the frustum (i.e., culled).
// ---------------------------------------------------------------------------

bool frustumCullSphere(vec4 planes[6], vec3 center, float radius) {
    for (int i = 0; i < 6; i++) {
        if (dot(planes[i].xyz, center) + planes[i].w + radius < 0.0) {
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

void main() {
    uint tid = gl_LocalInvocationIndex;
    uint gid = gl_WorkGroupID.x * 32 + tid;

    SceneGlobalsBuffer globals = SceneGlobalsBuffer(pc.sceneGlobalsAddress);

    bool valid = gid < globals.meshletTotalCount;

    bool visible = false;
    uint meshletIdx = 0;
    uint instanceIdx = 0;

    if (valid) {
        // Walk instances to find which instance owns this global meshlet index
        uint remaining = gid;
        uint instCount = globals.instanceCount;

        for (uint i = 0; i < instCount; i++) {
            InstanceBuffer inst = InstanceBuffer(globals.instanceAddr + uint64_t(i) * 80);
            MeshInfoBuffer mesh = MeshInfoBuffer(globals.meshInfoAddr + uint64_t(inst.meshIndex) * 32);
            uint meshletCount = mesh.meshletCount;

            if (remaining < meshletCount) {
                instanceIdx = i;
                meshletIdx = mesh.meshletOffset + remaining;
                break;
            }
            remaining -= meshletCount;
        }

        // Load instance transform
        GPUInstance instance = loadInstance(globals.instanceAddr, instanceIdx);

        // Load meshlet bounds
        MeshletBounds bounds = loadMeshletBounds(globals.meshletBoundsAddr, meshletIdx);

        // Transform bounding sphere to world space
        vec3 worldCenter = (instance.modelMatrix * vec4(bounds.center, 1.0)).xyz;
        float scaleX = length(instance.modelMatrix[0].xyz);
        float scaleY = length(instance.modelMatrix[1].xyz);
        float scaleZ = length(instance.modelMatrix[2].xyz);
        float maxScale = max(scaleX, max(scaleY, scaleZ));
        float worldRadius = bounds.radius * maxScale;

        // Extract frustum planes from the LIGHT's VP (stored in pc.viewProjection)
        vec4 frustumPlanes[6];
        extractFrustumPlanes(pc.viewProjection, frustumPlanes);

        // Frustum test only — no backface cone culling for shadows
        bool culledByFrustum = frustumCullSphere(frustumPlanes, worldCenter, worldRadius);

        visible = !culledByFrustum;
    }

    // Compact surviving meshlets using subgroup ballot
    uvec4 ballot = subgroupBallot(visible);
    uint survivorCount = subgroupBallotBitCount(ballot);
    uint localIndex = subgroupBallotExclusiveBitCount(ballot);

    if (tid == 0) {
        payload.count = survivorCount;
    }

    barrier();

    if (visible) {
        payload.meshletIndices[localIndex]  = meshletIdx;
        payload.instanceIndices[localIndex] = instanceIdx;
    }

    barrier();

    EmitMeshTasksEXT(payload.count, 1, 1);
}
