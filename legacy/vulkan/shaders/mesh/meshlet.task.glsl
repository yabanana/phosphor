#version 460

#extension GL_EXT_mesh_shader                          : require
#extension GL_EXT_buffer_reference                     : require
#extension GL_EXT_buffer_reference2                    : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_scalar_block_layout                  : require
#extension GL_KHR_shader_subgroup_ballot               : enable

#include "types.glsl"

// ---------------------------------------------------------------------------
// Task shader — meshlet culling (frustum + backface cone)
//
// Dispatch: one workgroup per meshlet candidate.
// Each invocation evaluates one meshlet. Surviving meshlets emit mesh workgroups.
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
// Frustum cull a bounding sphere (world space) against 6 planes
// Returns true if the sphere is OUTSIDE the frustum (i.e., culled).
// ---------------------------------------------------------------------------

bool frustumCullSphere(vec4 planes[6], vec3 center, float radius) {
    for (int i = 0; i < 6; i++) {
        if (dot(planes[i].xyz, center) + planes[i].w + radius < 0.0) {
            return true; // fully outside this plane
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Backface cone culling
// Returns true if the meshlet is entirely backfacing from the camera (culled).
// ---------------------------------------------------------------------------

bool coneCull(vec3 coneApex, vec3 coneAxis, float coneCutoff, vec3 cameraPos) {
    // If coneCutoff >= 1.0, the cone is degenerate — don't cull
    if (coneCutoff >= 1.0) {
        return false;
    }
    vec3 apexToCamera = normalize(cameraPos - coneApex);
    return dot(apexToCamera, coneAxis) < coneCutoff;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

void main() {
    uint tid = gl_LocalInvocationIndex;
    uint gid = gl_WorkGroupID.x * 32 + tid;

    SceneGlobalsBuffer globals = SceneGlobalsBuffer(pc.sceneGlobalsAddress);

    // Bounds check: each workgroup processes 32 meshlets
    bool valid = gid < globals.meshletTotalCount;

    bool visible = false;
    uint meshletIdx = 0;
    uint instanceIdx = 0;

    if (valid) {
        // The dispatch is organized so that meshlets are laid out per-instance:
        // For each instance, its meshlets are contiguous.
        // We need to find which instance this global meshlet index belongs to.
        // Walk through instances to find the right one.
        // For now, we use a simple scan (GPU-efficient alternative: prefix sum buffer).
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

        // Transform bounding sphere center to world space
        vec3 worldCenter = (instance.modelMatrix * vec4(bounds.center, 1.0)).xyz;
        // Approximate world-space radius by taking max scale axis
        float scaleX = length(instance.modelMatrix[0].xyz);
        float scaleY = length(instance.modelMatrix[1].xyz);
        float scaleZ = length(instance.modelMatrix[2].xyz);
        float maxScale = max(scaleX, max(scaleY, scaleZ));
        float worldRadius = bounds.radius * maxScale;

        // Transform cone apex and axis to world space
        vec3 worldConeApex = (instance.modelMatrix * vec4(bounds.coneApex, 1.0)).xyz;
        mat3 normalMatrix = mat3(instance.modelMatrix); // for cone axis rotation
        vec3 worldConeAxis = normalize(normalMatrix * bounds.coneAxis);

        // Extract frustum planes from the combined MVP
        vec4 frustumPlanes[6];
        extractFrustumPlanes(pc.viewProjection, frustumPlanes);

        // Frustum test
        bool culledByFrustum = frustumCullSphere(frustumPlanes, worldCenter, worldRadius);

        // Backface cone test
        bool culledByCone = coneCull(worldConeApex, worldConeAxis, bounds.coneCutoff, pc.cameraPosition.xyz);

        visible = !culledByFrustum; // cone cull disabled — may be too aggressive
    }

    // Compact surviving meshlets using subgroup ballot
    uvec4 ballot = subgroupBallot(visible);
    uint survivorCount = subgroupBallotBitCount(ballot);
    uint localIndex = subgroupBallotExclusiveBitCount(ballot);

    // First invocation writes the total count
    if (tid == 0) {
        payload.count = survivorCount;
    }

    barrier();

    if (visible) {
        payload.meshletIndices[localIndex]  = meshletIdx;
        payload.instanceIndices[localIndex] = instanceIdx;
    }

    barrier();

    // Emit mesh shader workgroups for surviving meshlets
    EmitMeshTasksEXT(payload.count, 1, 1);
}
