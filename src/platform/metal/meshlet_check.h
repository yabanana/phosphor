#pragma once

#include "renderer/meshlet_check.h"

#include <Metal/Metal.hpp>

namespace phosphor {

class GpuScene;
class MeshRenderer;
class MetalContext;
class SceneRenderer;
class SceneStore;

// ---------------------------------------------------------------------------
// --debug-meshlets (F6.7): read a completed check frame back (the caller
// waited for the GPU; the frame recorded its decisions and snapshotted the
// pyramids, MeshRenderer::requestCheckReadback) and compare it with the CPU
// references (renderer/meshlet_check.h).  Private scene buffers are copied
// into shared read-back buffers (MemoryCategory::Other, debug path: not an
// O7 run).
// ---------------------------------------------------------------------------

class MeshletChecker {
public:
    explicit MeshletChecker(MetalContext& context) : context_(context) {}
    ~MeshletChecker();
    MeshletChecker(const MeshletChecker&) = delete;
    MeshletChecker& operator=(const MeshletChecker&) = delete;

    MeshletCheckResult check(const MeshRenderer& mesh, const SceneRenderer& scene, const SceneStore& store,
                             const GpuScene& gpuScene, u32 slot);

private:
    MTL::Buffer* readback(MTL::Buffer*& cache, u64 size, const char* label);

    MetalContext& context_;
    MTL::Buffer*  instances_ = nullptr;
    MTL::Buffer*  materials_ = nullptr;
    MTL::Buffer*  flags_     = nullptr;
};

} // namespace phosphor
