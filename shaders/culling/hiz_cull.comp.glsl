#version 460

#extension GL_EXT_buffer_reference                      : require
#extension GL_EXT_buffer_reference2                     : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_scalar_block_layout                   : require

#include "types.glsl"

// ---------------------------------------------------------------------------
// Hi-Z occlusion culling — per-instance bounding-sphere test against the
// Hi-Z depth pyramid.
//
// Two-phase usage:
//   Phase 1: test against LAST frame's Hi-Z (before any rendering).
//            Visible instances are written to the output buffer for the
//            main mesh-shader pass.
//   Phase 2: test previously-culled instances against THIS frame's Hi-Z
//            (after the main pass has produced a fresh depth buffer).
//            Newly visible instances are appended for a second draw.
//
// The phase is selected via a specialization constant.
//
// Dispatch: ceil(instanceCount / 64) x 1 x 1
// ---------------------------------------------------------------------------

layout(local_size_x = 64) in;

// Specialization constant: 0 = phase 1 (last frame HiZ), 1 = phase 2 (current frame HiZ)
layout(constant_id = 0) const uint CULL_PHASE = 0;

// Hi-Z pyramid (sampled with nearest filtering)
layout(set = 1, binding = 0) uniform sampler2D hizPyramid;

// Output buffer: visible instance indices.
// Layout: [0] = atomic counter, [1..N] = instance indices
layout(set = 1, binding = 1, scalar) buffer VisibleInstances {
    uint count;
    uint indices[];
} visibleOut;

// Phase 2 input: instances that were culled in phase 1
// Layout: [0] = count, [1..N] = culled instance indices
layout(set = 1, binding = 2, scalar) buffer CulledInstances {
    uint count;
    uint indices[];
} culledIn;

// Phase 1 output: instances culled by phase 1 (input for phase 2)
layout(set = 1, binding = 3, scalar) buffer CulledOutput {
    uint count;
    uint indices[];
} culledOut;

// Use main push constants for viewProjection and scene globals
// (the 128-byte layout from types.glsl is already declared)

// ---------------------------------------------------------------------------
// Project bounding sphere to screen-space AABB and test against Hi-Z
// ---------------------------------------------------------------------------

bool isOccluded(vec3 worldCenter, float worldRadius) {
    // Project sphere center to clip space
    vec4 clipCenter = pc.viewProjection * vec4(worldCenter, 1.0);

    // Behind the near plane: not occluded (always visible)
    if (clipCenter.w <= 0.0) {
        return false;
    }

    // NDC coordinates of the sphere center
    vec3 ndc = clipCenter.xyz / clipCenter.w;

    // Compute screen-space radius.
    // Approximate: project a point offset by radius in the plane perpendicular to view.
    // Use the projected radius in X: r_screen = radius / (distance * tan(fov/2))
    // Simplified: project (center + right * radius) and measure difference.
    float projRadius = worldRadius / clipCenter.w;

    // Screen-space AABB of the projected sphere
    vec2 aabbMin = ndc.xy - vec2(projRadius);
    vec2 aabbMax = ndc.xy + vec2(projRadius);

    // Frustum test: if the AABB is entirely outside [-1, 1] NDC, it's outside the frustum
    if (aabbMax.x < -1.0 || aabbMin.x > 1.0 ||
        aabbMax.y < -1.0 || aabbMin.y > 1.0) {
        return false; // Outside frustum — not occluded, but will be frustum-culled elsewhere
    }

    // Clamp to screen bounds (NDC [-1, 1] -> UV [0, 1])
    vec2 uvMin = clamp(aabbMin * 0.5 + 0.5, 0.0, 1.0);
    vec2 uvMax = clamp(aabbMax * 0.5 + 0.5, 0.0, 1.0);

    // Choose mip level based on projected size.
    // The mip level where one texel covers the entire AABB is:
    //   mip = ceil(log2(max(aabb_width_pixels, aabb_height_pixels)))
    vec2 hizSize = vec2(textureSize(hizPyramid, 0));
    vec2 aabbPixels = (uvMax - uvMin) * hizSize;
    float maxPixelSize = max(aabbPixels.x, aabbPixels.y);
    float mipLevel = ceil(log2(max(maxPixelSize, 1.0)));

    // Clamp to maximum available mip level
    int maxMip = textureQueryLevels(hizPyramid) - 1;
    int mip = clamp(int(mipLevel), 0, maxMip);

    // Sample the Hi-Z at the 4 corners of the AABB at the chosen mip level
    float d0 = textureLod(hizPyramid, uvMin, float(mip)).r;
    float d1 = textureLod(hizPyramid, vec2(uvMax.x, uvMin.y), float(mip)).r;
    float d2 = textureLod(hizPyramid, vec2(uvMin.x, uvMax.y), float(mip)).r;
    float d3 = textureLod(hizPyramid, uvMax, float(mip)).r;

    // For standard Z (near=0, far=1): the farthest depth in the region
    float hizDepth = max(max(d0, d1), max(d2, d3));

    // The closest depth of the bounding sphere (its near surface)
    // ndc.z is already in [0, 1] range for Vulkan
    float sphereNearDepth = ndc.z - projRadius;

    // Occluded if the sphere's near surface is behind the Hi-Z depth
    return sphereNearDepth > hizDepth;
}

void main() {
    uint tid = gl_GlobalInvocationID.x;

    SceneGlobalsBuffer globals = SceneGlobalsBuffer(pc.sceneGlobalsAddress);
    uint instanceCount = globals.instanceCount;

    uint instanceIdx;

    if (CULL_PHASE == 0) {
        // Phase 1: test all instances
        if (tid >= instanceCount) {
            return;
        }
        instanceIdx = tid;
    } else {
        // Phase 2: test only instances culled by phase 1
        if (tid >= culledIn.count) {
            return;
        }
        instanceIdx = culledIn.indices[tid];
    }

    // Load instance and mesh info
    GPUInstance instance = loadInstance(globals.instanceAddr, instanceIdx);
    MeshInfoBuffer meshInfo = MeshInfoBuffer(globals.meshInfoAddr + uint64_t(instance.meshIndex) * 32);

    // Transform bounding sphere to world space
    vec4 bs = meshInfo.boundingSphere;
    vec3 localCenter = bs.xyz;
    float localRadius = bs.w;

    vec3 worldCenter = (instance.modelMatrix * vec4(localCenter, 1.0)).xyz;

    // Approximate world-space radius
    float scaleX = length(instance.modelMatrix[0].xyz);
    float scaleY = length(instance.modelMatrix[1].xyz);
    float scaleZ = length(instance.modelMatrix[2].xyz);
    float maxScale = max(scaleX, max(scaleY, scaleZ));
    float worldRadius = localRadius * maxScale;

    // Test against Hi-Z pyramid
    bool occluded = isOccluded(worldCenter, worldRadius);

    if (!occluded) {
        // Visible: append to output
        uint idx = atomicAdd(visibleOut.count, 1);
        visibleOut.indices[idx] = instanceIdx;
    } else if (CULL_PHASE == 0) {
        // Phase 1 only: save culled instances for phase 2
        uint idx = atomicAdd(culledOut.count, 1);
        culledOut.indices[idx] = instanceIdx;
    }
    // Phase 2 culled instances are simply discarded
}
