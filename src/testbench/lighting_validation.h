#pragma once

#include "testbench/testbench.h"
#include <glm/gtc/quaternion.hpp>
#include <span>
#include <string_view>
#include <vector>

namespace phosphor {

// F10/F12 analytic scene corpus. SOURCE ONLY / NON VERIFIED: these are test
// fixtures, not GPU references or acceptance evidence. World units = metres,
// linear material/emission values; no external assets. The alpha-mip fixture
// uploads a deterministic procedural texture with an analytically distinct mip1.
// Root registers this subclass for --lighting-scene SCENARIO with bench6.
class LightingValidation final : public TestBench {
public:
    explicit LightingValidation(std::string scenario);
    [[nodiscard]] static bool validScenario(std::string_view scenario);
    [[nodiscard]] static std::span<const std::string_view> scenarios();

    void setup(ECS&, GpuScene&, TextureManager&) override;
    void update(float dt, ECS&) override;
    void teardown(ECS&, GpuScene&) override;
    [[nodiscard]] const char* getName() const override { return name_.c_str(); }
    [[nodiscard]] const char* assetSource() const override { return "procedural-lighting-validation-v1"; }
    [[nodiscard]] CameraSetup getDefaultCamera() const override;
    bool scriptedCamera(double time, glm::vec3& position, glm::vec3& target, bool& cut) const override;

    [[nodiscard]] std::string_view scenario() const { return scenario_; }
    [[nodiscard]] double elapsedSeconds() const { return time_; }
    [[nodiscard]] std::span<const EntityID> entities() const { return entities_; }
    [[nodiscard]] EntityID emissivePanel() const { return panel_; }
    [[nodiscard]] EntityID sun() const { return sun_; }
    [[nodiscard]] EntityID movingCaster() const { return mover_; }
    [[nodiscard]] EntityID thinWall() const { return thinWall_; }
    [[nodiscard]] EntityID embeddedSolid() const { return embeddedSolid_; }
    [[nodiscard]] EntityID alphaReceiver() const { return alphaReceiver_; }

    // The root runner MUST explicitly select/inspect a probe at this anchor;
    // an automatically fitted DDGI grid is not guaranteed to contain it.
    [[nodiscard]] glm::vec3 embeddedProbeAnchor() const { return {0.9f, 1.0f, -0.8f}; }
    // update() runs before render frame0. Only positive-dt updates count.
    static constexpr u32 EmissiveStepFrame = 256;
    static constexpr float ThinWallThickness = 0.01f;
    static constexpr float NominalSunAngularRadius = 0.00465f;
    static constexpr float SceneUnitsInMetres = 1.0f;
    static constexpr u32 AlphaTextureSide = 256;
    static constexpr float AlphaMipCutoff = 0.75f;

private:
    EntityID mesh(ECS&, u32 handle, glm::vec3 position, glm::vec3 fullSize,
                  glm::quat rotation, glm::vec3 color, glm::vec3 emission = glm::vec3(0.0f), bool isStatic = true);
    EntityID directional(ECS&, glm::vec3 towardLight, float intensity);
    void room(ECS&);
    void exterior(ECS&);
    void move(ECS&, EntityID entity, glm::vec3 position, glm::quat rotation);

    std::string scenario_, name_;
    std::vector<EntityID> entities_, cacheCasters_;
    std::vector<glm::vec3> cacheOrigins_;
    u32 plane_ = ~0u, cube_ = ~0u;
    EntityID panel_ = INVALID_ENTITY, sun_ = INVALID_ENTITY, mover_ = INVALID_ENTITY;
    EntityID alphaReceiver_ = INVALID_ENTITY;
    EntityID thinWall_ = INVALID_ENTITY, embeddedSolid_ = INVALID_ENTITY;
    double time_ = 0;
    mutable u32 cameraSegment_ = ~0u;
    u32 materialStep_ = 0, positiveStepUpdates_ = 0;
};

} // namespace phosphor
