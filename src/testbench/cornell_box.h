#pragma once

#include "testbench/testbench.h"
#include "scene/components.h"
#include <vector>

namespace phosphor {

// ---------------------------------------------------------------------------
// CornellBox -- classic Cornell box for testing global illumination.
// Five walls (floor, ceiling, back, left=red, right=green), two cubes
// inside, and a ceiling-mounted area light approximated as a point light.
// ---------------------------------------------------------------------------

class CornellBox final : public TestBench {
public:
    void setup(ECS& ecs, GpuScene& gpuScene, TextureManager& textures) override;
    void update(float dt, ECS& ecs) override;
    void teardown(ECS& ecs, GpuScene& gpuScene) override;

    [[nodiscard]] const char* getName() const override { return "Cornell Box (GI)"; }
    [[nodiscard]] CameraSetup getDefaultCamera() const override;

private:
    std::vector<EntityID> entities_;
    EntityID tallBoxEntity_ = INVALID_ENTITY;
    float rotationAngle_ = 0.0f;
};

} // namespace phosphor
