#pragma once

#include "core/types.h"
#include "renderer/gpu_types.h"

#include <vector>

namespace phosphor {

class ECS;
class GpuScene;
struct MaterialComponent;

/// A run of instances that share one mesh and can be drawn with a single
/// instanced draw call.
struct DrawBatch {
    u32 meshIndex;
    u32 firstInstance;
    u32 instanceCount;
};

/// Everything the renderer needs from the ECS for one frame.
struct FrameScene {
    std::vector<GPUInstance> instances; // sorted by meshIndex
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
