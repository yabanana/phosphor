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
    explicit TransmissionValidation(std::string scenario);
    [[nodiscard]] static bool validScenario(std::string_view scenario);
    [[nodiscard]] static std::span<const std::string_view> scenarios();
    // Declared constants of a scenario (throws std::invalid_argument when unknown).
    [[nodiscard]] static const TransmissionScenarioSpec& spec(std::string_view scenario);
    // Radiance of the scene's declared isotropic environment (0 for an unknown scenario).
    [[nodiscard]] static float environmentRadiance(std::string_view scenario);

    void setup(ECS&, GpuScene&, TextureManager&) override;
    void update(float, ECS&) override {}
    void teardown(ECS&, GpuScene&) override;
    [[nodiscard]] const char* getName() const override { return name_.c_str(); }
    [[nodiscard]] const char* assetSource() const override { return "procedural-transmission-validation-v1"; }
    [[nodiscard]] CameraSetup getDefaultCamera() const override;

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

private:
    std::string scenario_, name_;
    std::vector<EntityID> glass_, emitters_;
};

} // namespace phosphor
