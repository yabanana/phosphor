#pragma once

#include "testbench/testbench.h"
#include "testbench/transmission_validation.h"
#include <glm/glm.hpp>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace phosphor {

// F15.3 (tickets 14 and 15) transmissive-shadow validation bench (bench 2 with --material-scenario NAME, the names start with
// "transmissive-shadow-").  A horizontal diffuse floor (the RECEIVER) under horizontal planar casters (thin-walled glass panes,
// thick slabs with volume attenuation, flat shallow water, an opaque comparison caster, and curved glass spheres for the
// qualitative case) lit by ONE light: the sun within 30 degrees of the casters' normal (the vertical) or one local point light.
// The environment is 0 and the camera looks at the FLOOR shadows from above and in front: it never sees the floor through the
// glass (the shadow of every caster lies in front of it on the screen), so the gated mask is the shadowed floor.  Ground truth:
// tools/f15_shadow_reference.py (Mitsuba 3.9.1 rays from the receiver through the casters; the receiver irradiance through a
// parallel slab is the straight-line transmittance up to a lateral shift).  Mirrored by tools/f15_check.py SCENARIOS
// (tests/test_transmissive_shadow_validation.cpp compares the names) and by tools/f15_shadow_fixtures.py.
//
// SOURCE ONLY / NOT VERIFIED on a device until tools/f15_check.py has captured the scenarios.
struct ShadowCaster {
    GlassObject glass;      // geometry and material factors (Pane: horizontal quad facing +Y; Cube: a slab; Sphere)
    bool opaque = false;    // an ordinary opaque material (base colour only, no extension): the comparison caster
    bool sss = false;       // F15.1 (ticket 14 round 3): a volumetric subsurface sphere (rt-walk receiver); implies opaque base
    bool mirror = false;    // F15.1 (ticket 14 round 3): an opaque polished metal slab (the reflection-secondary-lighting receiver); implies opaque
};

struct ShadowPointLight {
    glm::vec3 position{0.0f};
    float intensity = 0.0f;  // W/sr (irradiance facing the light at distance d: intensity / d^2)
    float range = 100.0f;
};

struct ShadowScenarioSpec {
    std::string_view name;
    std::vector<ShadowCaster> casters;           // spec order
    bool sun = false;                            // one directional light (disk of angular radius NominalSunAngularRadius)
    float sunIrradiance = 0.0f;                  // W/m^2 on a surface facing the sun
    glm::vec3 towardSun{0.0f, 1.0f, 0.0f};       // unit vector from the surface towards the sun
    std::vector<ShadowPointLight> pointLights;
    glm::vec3 floorBase{0.8f};
    float floorRoughness = 1.0f;
    glm::vec3 cameraTarget{0.0f, 0.0f, 0.3f};
    float cameraDistance = 6.2f;
    float cameraElevationDegrees = 38.0f;        // above the floor plane, looking down at the target from +Z
};

class TransmissiveShadowValidation final : public TestBench {
public:
    explicit TransmissiveShadowValidation(std::string scenario);
    [[nodiscard]] static bool validScenario(std::string_view scenario);
    [[nodiscard]] static std::span<const std::string_view> scenarios();
    // F15.1 (ticket 14, round 3): the INDIRECT-light fixtures (run by tools/f15_gi_glass_shadow_check.py, not by tools/f15_check.py):
    // a huge thin green pane 60 m above the scene (outside the probe grid, inside the 100 m ray range) shadows the sun for everything
    // below it, so every radiance of the scene is the pane-free radiance times the pane's transmittance.  `-gi-*`: an opaque cube whose
    // +Z face the sun never reaches (lit by GI only); `-refl-*`: a polished metal wall reflecting the sunlit floor (secondary lighting of
    // a reflection hit).  `-open` = the same scene without the pane (the reference of the ratio).  Not in scenarios(): the five
    // planar-shadow keys keep their tests; validScenario() and spec() know both lists.
    [[nodiscard]] static std::span<const std::string_view> indirectScenarios();  // + `-sss-*`: a subsurface sphere under the sun, sun from +Z
    // Declared constants of a scenario (throws std::invalid_argument when unknown).
    [[nodiscard]] static const ShadowScenarioSpec& spec(std::string_view scenario);

    void setup(ECS&, GpuScene&, TextureManager&) override;
    void update(float, ECS&) override {}
    void teardown(ECS&, GpuScene&) override;
    [[nodiscard]] const char* getName() const override { return name_.c_str(); }
    [[nodiscard]] const char* assetSource() const override { return "procedural-transmissive-shadow-validation-v1"; }
    [[nodiscard]] CameraSetup getDefaultCamera() const override;

    [[nodiscard]] std::span<const EntityID> casters() const { return casters_; }  // spec order
    [[nodiscard]] EntityID floor() const { return floor_; }
    [[nodiscard]] EntityID sun() const { return sun_; }
    [[nodiscard]] std::span<const EntityID> pointLights() const { return pointLights_; }

    static constexpr float FloorSize = 16.0f;            // the floor is a square of this side (instance scale of the unit plane)
    static constexpr float NominalSunAngularRadius = 0.00465f;
    static constexpr float CasterHeight = 0.8f;          // centre height of the sun and point casters above the floor
    static constexpr float SlabHalfExtent = 0.5f;        // the cube mesh half extent (instance scale sizes it)

private:
    std::string scenario_, name_;
    std::vector<EntityID> casters_, pointLights_;
    EntityID floor_ = INVALID_ENTITY, sun_ = INVALID_ENTITY;
};

} // namespace phosphor
