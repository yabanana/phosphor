#pragma once

#include "core/types.h"
#include "renderer/gpu_types.h"

#include <span>
#include <string>

namespace phosphor {

class ECS;
class GpuScene;
class SceneStore;

// ---------------------------------------------------------------------------
// F5 self-check (--debug-gpu-scene): compare what the GPU holds after a frame
// with the CPU mirror and the CPU references.  Portable; the Metal side only
// reads the buffers back (platform/metal/gpu_scene_check).
//
//   instances  every non-matrix field == SceneStore mirror, every model
//              matrix == referenceWorlds() bit for bit (CPU-owned slots: the
//              mirror; motion and hierarchy slots: the GPU's product, exact
//              because both sides compute without contraction, spike S6)
//   materials  == mirror byte for byte
//   visible    (gpu-driven on) the GPU's per-slot visibility (from the
//              prefix) == cullReference() on the read-back instances, except
//              slots whose margin to a decision boundary is inside the
//              declared band (counted, allowed: sqrt may differ by an ulp);
//              the visible list == the slots of the flags in slot order
//   arguments  draw arguments == drawArgsReference(GPU prefix)
//   counters   tested / visible / draw commands consistent with the above
//   ecs        SceneStore::verifyAgainstEcs() (a change the ECS missed)
// ---------------------------------------------------------------------------

struct SceneReadback {
    std::span<const GPUInstance> instances; // slot capacity
    std::span<const GPUMaterial> materials;
    std::span<const u32> prefix;            // on: slots + 1
    std::span<const u32> visible;           // on: slots
    std::span<const u32> drawArgs;          // on: 2 per command
    GPUSceneCounters     counters{};
};

struct SceneCheckResult {
    bool        pass = true;
    u32         slots           = 0;
    u32         instanceErrors  = 0;
    u32         matrixErrors    = 0;
    u32         materialErrors  = 0;
    u32         visibleErrors   = 0; // outside the band
    u32         bandDifferences = 0; // inside the band (allowed)
    u32         listErrors      = 0;
    u32         argErrors       = 0;
    u32         counterErrors   = 0;
    std::string ecs;                 // empty = mirror equals the ECS
    std::string first;               // first failure, for the log
};

[[nodiscard]] SceneCheckResult compareScene(const SceneStore& store, const GpuScene& scene, const ECS& ecs,
                                            const GPUCullParams& cull, const float* motionSinCos,
                                            const SceneReadback& gpu, bool gpuDriven);

/// "instances ok | ... | PASS" (one line).
[[nodiscard]] std::string formatSceneCheck(const SceneCheckResult& r);

} // namespace phosphor
