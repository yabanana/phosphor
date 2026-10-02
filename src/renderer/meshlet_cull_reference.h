#pragma once

// ---------------------------------------------------------------------------
// F6.2/F6.4 -- portable references of the meshlet kernels (shaders/meshlet.
// metal, shaders/hiz.metal).  They define the expected output exactly; the
// unit tests check them against hand-made and brute-force cases, and the F6
// spike / self-check compare the real kernels with them.  The decision
// functions call the shared math of renderer/meshlet_cull_math.h, so the CPU
// and the GPU run the same code.
//
// Hi-Z pyramid: reverse-Z (0 = far / background, 1 = near), every texel is
// the MINIMUM depth of the pixels it covers; pixels outside the image count
// as 1.0 (the neutral value of the min).
// ---------------------------------------------------------------------------

#include "core/types.h"
#include "renderer/gpu_types.h"
#include "renderer/meshlet_cull_math.h"
#include "renderer/scene_store.h"

#include <span>
#include <vector>

namespace phosphor {

/// CPU depth pyramid with the layout of GPUMeshletCullParams (power-of-two
/// level 0 >= ceil(viewport / 2), levels down to 1x1).
struct HiZPyramid {
    u32 width0  = 0;
    u32 height0 = 0;
    u32 levels  = 0;
    std::vector<std::vector<float>> level; // level[L], row-major, hizLevelSize(w0, L) x hizLevelSize(h0, L)

    float at(u32 lvl, u32 x, u32 y) const {
        return level[lvl][static_cast<size_t>(y) * hizLevelSize(width0, lvl) + x];
    }
};

/// Exact reference of the reduction (min is exact in float).  `depth` is
/// row-major width x height.
HiZPyramid buildHiZReference(const float* depth, u32 width, u32 height);

/// Minimum over the texels [x0..x1] x [y0..y1] of footprint.level.
float hizMinOverFootprint(const HiZPyramid& pyramid, const HiZFootprint& footprint);

/// Relative margin of the occlusion test: positive = occluded (see
/// hizOccluded).  Checkers accept a flip between the reference and the GPU
/// inside a band around 0.
float hizOcclusionMargin(float nearestDepth, float minTexel);

/// One candidate: its instance, its meshlet bounds and its cull class
/// (CullClass: 0 Back, 1 BackMirrored, 2 None).
struct MeshletDecisionInput {
    const GPUInstance*      instance;
    const GPUMeshletBounds* bounds;
    u32                     cullClass;
};

/// Phase A decision, in this order: frustum, cone (not for class None),
/// size (approximate), history Hi-Z.  Returns MESHLET_DECISION_FRUSTUM /
/// _CONE / _SIZE / _HISTORY / _DRAWN_A.  `history` may be null (treated as
/// not valid).
u32 meshletDecisionPhaseA(const GPUMeshletCullParams& p, const MeshletDecisionInput& in, const HiZPyramid* history);

/// Phase B decision against the current pyramid: _OCCLUDED or _DRAWN_B.
u32 meshletDecisionPhaseB(const GPUMeshletCullParams& p, const MeshletDecisionInput& in, const HiZPyramid& current);

/// Candidate list (3 contiguous class regions) with the phase's draw ranges.
struct MeshletListRef {
    std::vector<GPUMeshletCandidate> list;
    GPUMeshletDrawRange              ranges[3] = {};
    u32                              total     = 0;
};

/// Every meshlet of every visible, valid slot < slotCount.  Class: double
/// sided material -> 2, else mirrored -> 1, else 0.  Regions in class order,
/// slots in increasing order inside a region, meshlets in order.
/// `sceneFlags` = F5 cull flags per slot (bit 0 = visible).
MeshletListRef buildCandidatesReference(std::span<const GPUInstance> instances, std::span<const GPUMeshInfo> meshes,
                                        std::span<const GPUMaterial> materials, std::span<const u32> sceneFlags,
                                        u32 slotCount);

/// Phase-B list: stable compaction of the candidates with bFlags[i] != 0,
/// class regions kept.
MeshletListRef buildBListReference(const MeshletListRef& candidates, std::span<const u32> bFlags);

/// Upper bound of any frame's candidates: sum of capacity x meshlets of the
/// bucket's mesh.
u64 meshletCandidateCapacity(std::span<const SceneBucket> buckets, std::span<const GPUMeshInfo> meshes);

} // namespace phosphor
