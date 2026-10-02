#pragma once

#include "testbench/testbench.h"
#include "scene/components.h"
#include <vector>

namespace phosphor {

// ---------------------------------------------------------------------------
// CullingViz -- 10K instances arranged in a city-grid layout for
// visualizing the two-phase HiZ occlusion culling pipeline.
// Best used with the "Meshlets" or "Overdraw" debug overlay enabled.
//
// F6 --culling-script (the frozen preset of the F6 gate, deterministic with
// --fixed-timestep): the buildings are a subdivided box (12 x 12 quads per
// face, 1728 triangles, ~14 meshlets) and a 20 s loop scripts
//   [0, 6)   street-level fly-through towards a 120 x 40 wall at z = 30 that
//            disappears at 3 s (occluder removed: phase B must recover what
//            the history hid) and returns at 10 s, gone again at 13 s
//   6 s      camera cut to another district, street level until 12 s
//   [12, 16) cut, then a rising wide view over the city
//   [16, 20) cut, street-level 360 degree pan (history mispredictions)
// plus a fast sphere crossing the view at 150 units/s, and every 0.25 s 20
// buildings destroyed and 20 created elsewhere (recycled entities, slot
// reuse: nothing inherits visibility).
// ---------------------------------------------------------------------------

class CullingViz final : public TestBench {
public:
    explicit CullingViz(bool script = false) : script_(script) {}
    void setup(ECS& ecs, GpuScene& gpuScene, TextureManager& textures) override;
    void update(float dt, ECS& ecs) override;
    void teardown(ECS& ecs, GpuScene& gpuScene) override;

    [[nodiscard]] const char* getName() const override { return "Culling Visualization"; }
    [[nodiscard]] CameraSetup getDefaultCamera() const override;
    bool scriptedCamera(double t, glm::vec3& position, glm::vec3& target, bool& cut) const override;

private:
    static constexpr u32 GRID_DIM   = 100;  // 100x100 = 10,000 buildings
    static constexpr float STREET_WIDTH = 3.0f;
    static constexpr float BLOCK_SIZE   = 5.0f;

    EntityID addBuilding(ECS& ecs, u32 mesh, glm::vec3 position, float height, float color, float roughness);
    EntityID addWall(ECS& ecs);
    void     setupScript(ECS& ecs, GpuScene& gpuScene, TextureManager& textures);
    void     updateScript(float dt, ECS& ecs);

    std::vector<EntityID> entities_;
    bool script_ = false;
    // --culling-script state
    std::vector<EntityID> buildings_;
    EntityID wall_   = INVALID_ENTITY;
    EntityID sphere_ = INVALID_ENTITY;
    u32      buildingMesh_ = 0;
    u32      cubeMesh_     = 0;
    u32      sphereMesh_   = 0;
    u32      wallMaterial_ = 0;
    double   time_     = 0.0;
    double   nextChurn_ = 0.25;
    u64      churnRng_ = 0x9E3779B97F4A7C15ull;
    mutable int lastSegment_ = -1;
};

} // namespace phosphor
