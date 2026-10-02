#pragma once

#include "core/types.h"
#include "renderer/gpu_types.h"
#include "renderer/meshlet_cull_reference.h"

#include <span>
#include <string>

namespace phosphor {

// ---------------------------------------------------------------------------
// F6.7 --debug-meshlets: compare what the GPU did in one completed frame with
// the CPU references (renderer/meshlet_cull_reference.h).  Portable: the
// caller (platform/metal/meshlet_check.cpp) reads the buffers back.
//
// Checks, each named in the result:
//   overflow    the overflow word and gate are 0 (a capacity that is an
//               upper bound can never overflow)
//   candidates  the candidate list, class ranges and phase-A indirect
//               arguments equal buildCandidatesReference exactly
//   decisionsA  every candidate's recorded phase-A decision equals
//               meshletDecisionPhaseA against the history snapshot, except
//               where the decision flips inside the rounding band (bounds
//               radius and cone cutoff +-1e-4, occlusion margin < 1e-4)
//   bflags      (two-phase) B flag == (decision A == HISTORY)
//   blist       the B list, ranges and arguments equal buildBListReference
//               of the GPU flags
//   decisionsB  every B entry's recorded decision equals meshletDecisionPhaseB
//               against the CURRENT pyramid read back (band as above)
//   lost        no surface lost: every candidate the GPU rejected as occluded
//               in phase B is also occluded against the CPU pyramid of the
//               FINAL depth (current <= final pointwise in reverse-Z, so a
//               correct rejection always passes; a too-near pyramid fails)
//   pyramids    the new history equals buildHiZReference(final depth) bit for
//               bit; current <= new history texel by texel; every level of
//               the current pyramid and of the history snapshot is the min of
//               its children
//   counters    candidates, drawnA + frustum + cone + history (+ size) ==
//               candidates, testedB == history rejects == B total, drawnB +
//               occludedB == testedB, and the counts equal the decisions
// ---------------------------------------------------------------------------

struct MeshletCheckInput {
    GPUMeshletCullParams params{}; // the frame's (copy written by the host)
    bool twoPhase = false;
    u64  capacity = 0;
    u32  slotCount = 0;
    std::span<const GPUInstance>      instances;   // slot space (GPU readback)
    std::span<const GPUMaterial>      materials;   // GPU readback
    std::span<const u32>              sceneFlags;  // F5 cull flags (GPU readback)
    std::span<const GPUMeshInfo>      meshes;      // CPU (GpuScene)
    std::span<const GPUMeshletBounds> bounds;      // CPU (GpuScene)
    std::span<const GPUMeshlet>       meshlets;    // CPU (GpuScene)
    std::span<const GPUMeshletCandidate> candidates; // capacity
    std::span<const GPUMeshletCandidate> bList;      // capacity
    std::span<const u32> bFlags;                     // capacity
    std::span<const u32> decisions;                  // 2 * capacity
    const GPUMeshletDrawRange* ranges = nullptr;     // MESHLET_DRAWS
    const u32* args = nullptr;                       // MESHLET_DRAWS * 3
    GPUMeshletCounters counters{};
    u32 gate = 0;
    // Pyramids read back level after level (tight rows) and the final depth.
    std::span<const float> history;  // phase A's
    std::span<const float> current;  // phase B's (two-phase)
    std::span<const float> next;     // Hi-Z final's (two-phase)
    std::span<const float> depth;    // final depth, width x height
    u32 width = 0, height = 0;
};

struct MeshletCheckResult {
    bool        pass = true;
    std::string failures; // "name: detail; " per failed check
    u32 candidates = 0, ambiguousA = 0, ambiguousB = 0, occludedB = 0;
};

[[nodiscard]] MeshletCheckResult checkMeshletFrame(const MeshletCheckInput& in);
[[nodiscard]] std::string formatMeshletCheck(const MeshletCheckResult& r);

/// A pyramid of the given level-0 size from a tight level-after-level dump.
[[nodiscard]] HiZPyramid pyramidFromReadback(std::span<const float> data, u32 width0, u32 height0, u32 levels);

} // namespace phosphor
