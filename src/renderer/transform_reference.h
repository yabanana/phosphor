#pragma once

#include "core/types.h"
#include "renderer/gpu_types.h"
#include "renderer/scene_store.h"

#include <span>
#include <vector>

namespace phosphor {

// ---------------------------------------------------------------------------
// F5.2 -- CPU reference of what shaders/transforms.metal computes (portable).
// The GPU results must match it bit for bit (transform_math.h uses the same
// operations in the same order on both sides).
// ---------------------------------------------------------------------------

/// The arrays of the transform pass, in slot space (what the GPU buffers hold).
struct HierarchyView {
    std::span<const GPUInstance>      instances;     // slotCapacity; modelMatrix of non-moving roots
    std::span<const GPUTransformNode> nodes;         // slotCapacity; local of children
    std::span<const GPUMotion>        motions;       // slotCapacity; read for motionSlots only
    std::span<const u32>              motionSlots;   // roots driven by scene_motion
    std::span<const u32>              childOffsets;  // slotCapacity + 1 (CSR)
    std::span<const u32>              childSlots;
};

/// world[slot * 16 ..] for every slot: motion slots get motionWorld (sin/cos
/// of their class from `sinCos`, SCENE_MOTION_CLASSES * 2 floats), children
/// get world(parent) * local (processed parent first), everything else keeps
/// instances[slot].modelMatrix.  A slot is a child iff the CSR lists it.
void referenceWorlds(const HierarchyView& view, const float* sinCos, std::vector<float>& world);

/// The same for a SceneStore (thin wrapper over the view of its mirror; inline
/// so this header does not force the store's translation unit into a link).
inline void referenceWorlds(const SceneStore& store, const float* sinCos, std::vector<float>& world) {
    HierarchyView v;
    v.instances    = store.instances();
    v.nodes        = store.nodes();
    v.motions      = store.motions();
    v.motionSlots  = store.motionSlots();
    v.childOffsets = store.childOffsets();
    v.childSlots   = store.childSlots();
    referenceWorlds(v, sinCos, world);
}

} // namespace phosphor
