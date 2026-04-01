#pragma once

#include "testbench/testbench.h"
#include "scene/components.h"
#include <vector>

namespace phosphor {

// ---------------------------------------------------------------------------
// ManyLights -- 1024 point lights with random colors circling inside a
// large room.  Designed to stress the ReSTIR direct illumination pipeline.
// ---------------------------------------------------------------------------

class ManyLights final : public TestBench {
public:
    void setup(ECS& ecs, GpuScene& gpuScene, TextureManager& textures) override;
    void update(float dt, ECS& ecs) override;
    void teardown(ECS& ecs, GpuScene& gpuScene) override;

    [[nodiscard]] const char* getName() const override { return "Many Lights (1024)"; }
    [[nodiscard]] CameraSetup getDefaultCamera() const override;

private:
    static constexpr u32 LIGHT_COUNT = 1024;
    static constexpr float ROOM_SIZE = 30.0f;
    static constexpr float ROOM_HEIGHT = 8.0f;

    struct LightAnim {
        EntityID entity;
        glm::vec3 center;       // orbit center
        float     orbitRadius;
        float     orbitSpeed;
        float     phase;        // initial angle offset
        float     yOffset;
    };

    std::vector<EntityID>  entities_;
    std::vector<LightAnim> lightAnims_;
    float time_ = 0.0f;
};

} // namespace phosphor
