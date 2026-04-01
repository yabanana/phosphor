#pragma once

#include "testbench/testbench.h"
#include "scene/components.h"
#include <vector>
#include <string>

namespace phosphor {

// ---------------------------------------------------------------------------
// SceneViewer -- loads a glTF/GLB scene from disk, with a fallback to
// procedural geometry (cubes + spheres) if no asset file is found.
// ---------------------------------------------------------------------------

class SceneViewer final : public TestBench {
public:
    void setup(ECS& ecs, GpuScene& gpuScene, TextureManager& textures) override;
    void update(float dt, ECS& ecs) override;
    void teardown(ECS& ecs, GpuScene& gpuScene) override;

    [[nodiscard]] const char* getName() const override { return "Scene Viewer (glTF)"; }
    [[nodiscard]] CameraSetup getDefaultCamera() const override;

private:
    void setupProceduralFallback(ECS& ecs, GpuScene& gpuScene, TextureManager& textures);

    std::vector<EntityID> entities_;
    bool loadedGltf_ = false;
};

} // namespace phosphor
