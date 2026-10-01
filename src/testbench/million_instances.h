#pragma once

#include "testbench/testbench.h"
#include "scene/components.h"

#include <random>
#include <vector>

namespace phosphor {

// ---------------------------------------------------------------------------
// MillionInstances (F5.6) -- bench 8, "1M Instances (dynamic)": the scene the
// GPU-driven pipeline is designed for.
//
//   * `instances` (default 1,000,000) of `meshes` (default 8, up to 1024)
//     distinct low-poly meshes (8-48 triangles) scattered in a slab around the
//     default camera, so that a realistic fraction is inside the frustum and
//     the visible triangle count stays far below the ~20 M of the partial
//     render knee (B-12).
//   * ~90 % of the plain roots carry a MotionComponent (evaluated on the GPU:
//     no CPU work per frame); ~10 % of the instances are satellites, children
//     (HierarchyComponent) of moving roots or of other satellites, down to
//     depth 3.
//   * ~5 % of the instances are mirrored (one negative scale component), the
//     materials are a shared library (gpuScene.addMaterial, ~5 % double
//     sided) plus ~1 % of the entities with their own MaterialComponent.
//   * update(): `dynamicCpuPercent` % of the instances get a CPU transform
//     change per frame (a rotating window over the instances, through
//     getComponent<TransformComponent>, which marks them changed), and
//     `churn` leaf instances are destroyed and as many created.
//
// Everything is deterministic: fixed RNG seeds, frame-indexed windows.
// ---------------------------------------------------------------------------

class MillionInstances final : public TestBench {
public:
    static constexpr u32   DEFAULT_INSTANCES = 1000000;
    static constexpr u32   DEFAULT_MESHES    = 8;
    static constexpr u32   MAX_MESHES        = 1024;
    static constexpr u32   MATERIAL_COUNT    = 256;
    static constexpr u32   MAX_SATELLITE_DEPTH = 3;
    static constexpr float DEFAULT_DYNAMIC_PERCENT = 1.0f;

    explicit MillionInstances(const TestBenchParams& params = {});

    void setup(ECS& ecs, GpuScene& gpuScene, TextureManager& textures) override;
    void update(float dt, ECS& ecs) override;
    void teardown(ECS& ecs, GpuScene& gpuScene) override;

    [[nodiscard]] const char* getName() const override { return "1M Instances (dynamic)"; }
    [[nodiscard]] CameraSetup getDefaultCamera() const override;

    // Resolved parameters (defaults filled in, limits applied).
    [[nodiscard]] u32   instances() const { return instances_; }
    [[nodiscard]] u32   meshes() const { return meshes_; }
    [[nodiscard]] float dynamicCpuPercent() const { return dynamicPercent_; }
    [[nodiscard]] u32   churn() const { return churn_; }
    /// Half extent of the scene slab in x and z.
    [[nodiscard]] float extent() const { return extent_; }
    /// Entities of the instances currently alive (roots and satellites).
    [[nodiscard]] u32   liveInstances() const { return static_cast<u32>(records_.size()); }

private:
    struct Record {
        EntityID entity = INVALID_ENTITY;
    };

    EntityID createInstance(ECS& ecs, std::mt19937& rng, TransformComponent xf, const MotionComponent* motion,
                            EntityID parent);
    EntityID createLeafRoot(ECS& ecs, std::mt19937& rng);
    void     mirrorSome(std::mt19937& rng, TransformComponent& xf) const;

    u32   instances_       = DEFAULT_INSTANCES;
    u32   meshes_          = DEFAULT_MESHES;
    float dynamicPercent_  = DEFAULT_DYNAMIC_PERCENT;
    u32   churn_           = 0;
    float extent_          = 300.0f;

    u32 meshBase_ = 0;                // handle of the first mesh
    u32 whiteTex_ = ~0u, normalTex_ = ~0u, mrTex_ = ~0u; // default textures for own materials
    u32 libraryBase_ = 0;             // materialIndex of library material 0

    std::vector<Record>   records_;   // every live instance (roots + satellites)
    std::vector<u32>      leaves_;    // indices into records_: churnable leaf roots
    std::vector<EntityID> lights_;
    u64  frame_ = 0;
};

} // namespace phosphor
