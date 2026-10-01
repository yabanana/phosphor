#pragma once

#include "testbench/testbench.h"
#include "scene/components.h"
#include <vector>

namespace phosphor {

// ---------------------------------------------------------------------------
// StressTest -- 100K instances of a low-poly sphere scattered in a large
// volume with random materials. Designed to stress the mesh shading and
// GPU culling pipelines.
// ---------------------------------------------------------------------------

class StressTest final : public TestBench {
public:
    void setup(ECS& ecs, GpuScene& gpuScene, TextureManager& textures) override;
    void update(float dt, ECS& ecs) override;
    void teardown(ECS& ecs, GpuScene& gpuScene) override;

    [[nodiscard]] const char* getName() const override { return "Stress Test (100K)"; }
    [[nodiscard]] CameraSetup getDefaultCamera() const override;

private:
    static constexpr u32 INSTANCE_COUNT  = 100000;
    static constexpr u32 MATERIAL_COUNT  = 64;
    static constexpr float VOLUME_EXTENT = 200.0f;

    std::vector<EntityID> entities_;
};

} // namespace phosphor
