#pragma once

#include "testbench/testbench.h"
#include "scene/components.h"
#include <vector>

namespace phosphor {

// ---------------------------------------------------------------------------
// TorusDemo -- rotating PBR torus with a ground plane and directional light.
// Showcases mesh shading pipeline, material resolve, and tonemapping.
// ---------------------------------------------------------------------------

class TorusDemo final : public TestBench {
public:
    void setup(ECS& ecs, GpuScene& gpuScene, TextureManager& textures) override;
    void update(float dt, ECS& ecs) override;
    void teardown(ECS& ecs, GpuScene& gpuScene) override;

    [[nodiscard]] const char* getName() const override { return "Torus Demo"; }
    [[nodiscard]] CameraSetup getDefaultCamera() const override;

private:
    std::vector<EntityID> entities_;
    EntityID torusEntity_ = INVALID_ENTITY;
    float    rotationAngle_ = 0.0f;
};

} // namespace phosphor
