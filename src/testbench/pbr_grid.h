#pragma once

#include "testbench/testbench.h"
#include "scene/components.h"
#include <vector>

namespace phosphor {

// ---------------------------------------------------------------------------
// PBRGrid -- 10x10 grid of spheres varying metallic (rows) and roughness
// (columns) from 0 to 1. Four point lights for even illumination.
// ---------------------------------------------------------------------------

class PBRGrid final : public TestBench {
public:
    void setup(ECS& ecs, GpuScene& gpuScene, TextureManager& textures) override;
    void update(float dt, ECS& ecs) override;
    void teardown(ECS& ecs, GpuScene& gpuScene) override;

    [[nodiscard]] const char* getName() const override { return "PBR Material Grid"; }
    [[nodiscard]] CameraSetup getDefaultCamera() const override;

private:
    static constexpr u32 GRID_SIZE = 10;

    std::vector<EntityID> entities_;
    std::vector<EntityID> sphereEntities_; // subset for rotation
    float rotationAngle_ = 0.0f;
};

} // namespace phosphor
