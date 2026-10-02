#pragma once

#include "core/launch_options.h"
#include "renderer/scene_check.h"

#include <Metal/Metal.hpp>

namespace phosphor {

class ECS;
class GpuScene;
class MetalContext;
class SceneRenderer;
class SceneStore;

// ---------------------------------------------------------------------------
// --debug-gpu-scene: read the GPU scene back after a completed frame (the
// caller waited for the GPU) and compare it (renderer/scene_check.h).  The
// read-back buffers are shared, MemoryCategory::Other, allocated on the
// first check and grown with the scene (debug path: not an O7 run).
// ---------------------------------------------------------------------------

class GpuSceneChecker {
public:
    explicit GpuSceneChecker(MetalContext& context) : context_(context) {}
    ~GpuSceneChecker();
    GpuSceneChecker(const GpuSceneChecker&) = delete;
    GpuSceneChecker& operator=(const GpuSceneChecker&) = delete;

    /// `slot`: the frame slot of the completed frame to check.
    SceneCheckResult check(const SceneRenderer& renderer, const SceneStore& store, const GpuScene& scene,
                           const ECS& ecs, const GPUCullParams& cull, const float* motionSinCos, u32 slot,
                           GpuDrivenMode mode, bool drawGateOpen = true);

private:
    MTL::Buffer* readback(MTL::Buffer*& cache, u64 size, const char* label);

    MetalContext& context_;
    MTL::Buffer*  instances_ = nullptr;
    MTL::Buffer*  materials_ = nullptr;
    MTL::Buffer*  prefix_    = nullptr;
    MTL::Buffer*  visible_   = nullptr;
};

} // namespace phosphor
