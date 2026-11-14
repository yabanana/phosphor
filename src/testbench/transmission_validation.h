#pragma once

#include "testbench/testbench.h"
#include <glm/glm.hpp>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace phosphor {

// F15.3 (ticket 13) transmission/volume validation bench (bench 2 with --material-scenario NAME, the names start with
// "transmission-").  Glass objects (spheres, cubes/slabs, panes) in front of an emissive backdrop under a constant
// environment, frozen camera and exact factors (no textures), so that a linear capture can be compared with the
// Mitsuba path-traced reference of tools/f15_transmission_reference.py.  Gating policy (decision 11): nothing seen
// through the glass is lit through curved glass: the backdrop is purely EMISSIVE (a checkerboard of emissive cells or
// emissive panels) in front of nothing diffuse, so every ray leaving the glass ends at the environment or at an
// emitter.  Mirrored by tools/f15_check.py SCENARIOS (tests/test_transmission_validation.cpp compares the names).
//
// SOURCE ONLY / NOT VERIFIED on a device until tools/f15_check.py has captured the scenarios.
struct GlassObject {
    enum class Shape { Sphere, Cube, Pane };
    Shape     shape = Shape::Sphere;
    glm::vec3 center{0.0f};
    glm::vec3 scale{1.0f};       // instance scale (a slab is a scaled cube; spheres and panes keep 1)
    float     yawDegrees = 0.0f; // rotation: yaw about Y, then pitch about X (degrees); a pane faces +Z before it
    float     pitchDegrees = 0.0f;
    glm::vec3 baseColor{1.0f};
    float     roughness = 0.0f;
    float     transmission = 1.0f;
    float     ior = 1.5f;
    float     thickness = 0.0f;  // KHR_materials_volume thicknessFactor, mesh-local units: 0 = thin-walled, > 0 = volume
    glm::vec3 attenuationColor{1.0f};
    float     attenuationDistance = std::numeric_limits<float>::infinity();
    bool      doubleSided = false;
    // F15.3 (ticket 19): KHR_materials_iridescence (factor 0 = no block; the film ior; thickness in nm, the maximum) and the thin
    // subsurface block (diffuse transmission weight 0 = no block; colour) on the glass material: the declared --rt off model composes
    // them with the transmission (tools/f15_transmission_reference.py declared_pixel).
    float     iridescenceFactor = 0.0f;
    float     iridescenceIor = 1.3f;
    float     iridescenceThickness = 400.0f;
    float     diffuseTransmission = 0.0f;
    glm::vec3 diffuseTransmissionColor{1.0f};
};

struct TransmissionScenarioSpec {
    enum class Backdrop { None, Checker, Panels };
    std::string_view name;
    std::vector<GlassObject> glass;
    Backdrop backdrop = Backdrop::Checker;
    float environment = 0.5f;   // constant isotropic radiance of the declared environment
    float emission = 2.0f;      // radiance of the emissive backdrop cells / panels
    glm::vec3 cameraTarget{0.0f, 1.0f, 0.0f};
    float cameraDistance = 5.5f; // along +Z from the target
    // Panels backdrop only: (centre x, centre y, half width, half height) of each emissive panel at z = PanelZ.
    std::vector<glm::vec4> panels;
};

class TransmissionValidation final : public TestBench {
public:
    // `motion` (--transmission-motion, F15.1 ticket 13 temporal clip): a scripted camera orbit and an extra emissive panel that
    // slides behind the glass; the static fixture is unchanged without it (tools/f15_clip_check.py transmission clips).
    explicit TransmissionValidation(std::string scenario, bool motion = false);
    [[nodiscard]] static bool validScenario(std::string_view scenario);
    [[nodiscard]] static std::span<const std::string_view> scenarios();
    // Declared constants of a scenario (throws std::invalid_argument when unknown).
    [[nodiscard]] static const TransmissionScenarioSpec& spec(std::string_view scenario);
    // Radiance of the scene's declared isotropic environment (0 for an unknown scenario).
    [[nodiscard]] static float environmentRadiance(std::string_view scenario);

    void setup(ECS&, GpuScene&, TextureManager&) override;
    void update(float dt, ECS& ecs) override;
    void teardown(ECS&, GpuScene&) override;
    [[nodiscard]] const char* getName() const override { return name_.c_str(); }
    [[nodiscard]] const char* assetSource() const override { return "procedural-transmission-validation-v1"; }
    [[nodiscard]] CameraSetup getDefaultCamera() const override;
    bool scriptedCamera(double t, glm::vec3& position, glm::vec3& target, bool& cut) const override;
    // The motion script, a pure function of the scene time (unit tested): camera yaw about the target and the panel centre.
    [[nodiscard]] static float orbitYawRadians(double t);
    [[nodiscard]] static glm::vec3 movingPanelCenter(double t);
    [[nodiscard]] bool motion() const { return motion_; }
    [[nodiscard]] EntityID movingPanel() const { return movingPanel_; } // INVALID_ENTITY without motion

    [[nodiscard]] std::string_view scenario() const { return scenario_; }
    [[nodiscard]] std::span<const EntityID> glass() const { return glass_; }       // spec order
    [[nodiscard]] std::span<const EntityID> emitters() const { return emitters_; } // backdrop cells / panels

    static constexpr float SphereRadius = 0.4f;
    static constexpr u32 SphereSlices = 64, SphereStacks = 32;
    static constexpr float CubeHalfExtent = 0.5f;     // the mesh of cubes and slabs (instance scale sizes them)
    static constexpr float CellSize = 0.4f;           // checkerboard cell edge
    static constexpr float BackdropZ = -3.0f;         // checkerboard plane
    static constexpr float PanelZ = -0.6f;            // emissive panels of the "Panels" backdrop
    static constexpr int CheckerColumns = 24, CheckerRows = 16;  // x in [-4.8, 4.8], y in [-1.2, 5.2]
    static constexpr float CheckerOriginX = -4.8f, CheckerOriginY = -1.2f;
    static constexpr float PaneHalfExtent = 0.45f;    // a pane is a 0.9 x 0.9 square
    // Motion script (--transmission-motion): yaw = 12 degrees * sin(1.0 t) about the target at the default distance (a horizontal arc,
    // at most 0.2 degrees per frame at 60 fps); the panel (1.0 x 1.0, radiance 1.5 x the backdrop emission) slides along x in
    // +-1.8 m at 1.3 rad/s at height 2.2 m, z = -1.2 m: behind every glass object and in front of the checkerboard.
    static constexpr float OrbitAmplitudeDegrees = 12.0f, OrbitRate = 1.0f;
    static constexpr float PanelAmplitude = 1.8f, PanelRate = 1.3f, PanelY = 2.2f, PanelMotionZ = -1.2f, PanelSize = 1.0f, PanelEmissionScale = 1.5f;

private:
    std::string scenario_, name_;
    std::vector<EntityID> glass_, emitters_;
    bool motion_ = false;
    EntityID movingPanel_ = INVALID_ENTITY;
    double time_ = 0.0;
};

} // namespace phosphor
