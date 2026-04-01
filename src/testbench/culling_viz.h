#pragma once

#include "testbench/testbench.h"
#include "scene/components.h"
#include <vector>

namespace phosphor {

// ---------------------------------------------------------------------------
// CullingViz -- 10K instances arranged in a city-grid layout for
// visualizing the two-phase HiZ occlusion culling pipeline.
// Best used with the "Meshlets" or "Overdraw" debug overlay enabled.
// ---------------------------------------------------------------------------

class CullingViz final : public TestBench {
public:
    void setup(ECS& ecs, GpuScene& gpuScene, TextureManager& textures) override;
    void update(float dt, ECS& ecs) override;
    void teardown(ECS& ecs, GpuScene& gpuScene) override;

    [[nodiscard]] const char* getName() const override { return "Culling Visualization"; }
    [[nodiscard]] CameraSetup getDefaultCamera() const override;

private:
    static constexpr u32 GRID_DIM   = 100;  // 100x100 = 10,000 buildings
    static constexpr float STREET_WIDTH = 3.0f;
    static constexpr float BLOCK_SIZE   = 5.0f;

    std::vector<EntityID> entities_;
};

} // namespace phosphor
