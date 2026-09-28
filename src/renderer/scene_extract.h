#pragma once

#include "core/types.h"
#include "renderer/gpu_types.h"

#include <vector>

namespace phosphor {

class ECS;
class GpuScene;
struct MaterialComponent;

/// Culling class of a batch (see DrawBatch::cull).
enum class CullClass : u8 {
    Back,         // cull back faces, front faces counter-clockwise
    BackMirrored, // cull back faces, front faces clockwise (mirrored instances)
    None          // double-sided material: no culling
};

/// A run of instances that share one mesh and one culling class and can be
/// drawn with a single instanced draw call.
struct DrawBatch {
    u32 meshIndex;
    u32 firstInstance;
    u32 instanceCount;
    CullClass cull = CullClass::Back; // all instances of a batch share it
};

/// Everything the renderer needs from the ECS for one frame.
struct FrameScene {
    std::vector<GPUInstance> instances; // sorted by (meshIndex, cull class)
    std::vector<GPUMaterial> materials; // library materials, then per-entity ones
    std::vector<GPULight>    lights;
    std::vector<DrawBatch>   batches;
};

/// Convert an ECS material component to its GPU layout.
GPUMaterial toGPUMaterial(const MaterialComponent& material);

/// Gather visible instances, materials and lights from the ECS.
///
/// Material resolution: an entity with a MaterialComponent gets its own GPU
/// material appended after the scene's material library; otherwise
/// MeshInstanceComponent::materialIndex indexes the library directly.
/// Instances referencing unknown meshes are skipped.
void extractFrameScene(ECS& ecs, const GpuScene& scene, FrameScene& out);

} // namespace phosphor
