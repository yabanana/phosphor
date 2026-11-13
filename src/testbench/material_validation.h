#pragma once

#include "testbench/testbench.h"
#include <glm/glm.hpp>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace phosphor {

// F15.3 material-validation bench (bench 2 with --material-scenario NAME).
// Texture-free spheres (metallic x roughness grid) with a frozen camera, light
// rig and environment per scenario, so a linear capture is radiance that an
// independent glTF Appendix B reference (tools/f15_reference.py) can predict.
// Materials use exact factors: no texture indices, so the shader fallbacks
// (white, flat normal, MR = 1) leave every factor untouched.
//
// SOURCE ONLY / NOT VERIFIED on a device until the orchestrator runs the
// captures; see tools/f15_check.py for the exact command lines.
struct MaterialScenarioSpec {
    std::string_view name;
    std::vector<float> roughness; // one grid column each (left to right)
    std::vector<float> metallic;  // one grid row each (bottom to top)
    glm::vec3 baseColor{1.0f};    // linear albedo
    bool sun = false;             // one directional light (disk of angular radius NominalSunAngularRadius)
    float sunIrradiance = 0.0f;   // W/m^2 on a surface facing the sun (colour is white)
    glm::vec3 towardSun{0, 1, 0}; // unit vector from the surface towards the sun
    float environment = 0.0f;     // constant isotropic radiance of the declared environment
    // F15.1 (ticket 05) specular grids: KHR_materials_ior per column and KHR_materials_specular per row
    // (all three empty = no extension, the base scenarios).  Rows keep `metallic`, columns `roughness`.
    std::vector<float>     ior;            // per column; 0 is the permanent "Fresnel = 1" mode
    std::vector<float>     specularFactor; // per row, [0, 1]
    std::vector<glm::vec3> specularColor;  // per row, may exceed 1
    [[nodiscard]] bool hasSpecularGrid() const { return !ior.empty(); }
};

class MaterialValidation final : public TestBench {
public:
    // `toggleEvery` (F15.1, specular grids only): every N updates the extension of the two probe spheres
    // is edited (a parameter change on one, a feature toggle on the other: a counted record rewrite,
    // never a reallocation); 0 = static scene.
    explicit MaterialValidation(std::string scenario, u32 toggleEvery = 0);
    [[nodiscard]] static bool validScenario(std::string_view scenario);
    [[nodiscard]] static std::span<const std::string_view> scenarios();
    // Declared constants of a scenario (throws std::invalid_argument when unknown).
    [[nodiscard]] static const MaterialScenarioSpec& spec(std::string_view scenario);
    // Radiance of the scene's declared isotropic environment (0 for an unknown
    // or empty scenario); the engine reads it for every light path.
    [[nodiscard]] static float environmentRadiance(std::string_view scenario);

    void setup(ECS&, GpuScene&, TextureManager&) override;
    void update(float, ECS&) override;
    void teardown(ECS&, GpuScene&) override;
    [[nodiscard]] const char* getName() const override { return name_.c_str(); }
    [[nodiscard]] const char* assetSource() const override { return "procedural-material-validation-v2"; }
    [[nodiscard]] CameraSetup getDefaultCamera() const override;

    [[nodiscard]] std::string_view scenario() const { return scenario_; }
    [[nodiscard]] std::span<const EntityID> spheres() const { return spheres_; } // row-major: row * columns + column
    [[nodiscard]] EntityID sun() const { return sun_; }

    // Probe spheres of the toggle test (row-major index): the neutral sphere whose feature bit flips
    // and the sphere whose specularFactor alternates.
    [[nodiscard]] static u32 toggleFeatureSphere() { return 0 * 5 + 3; }
    [[nodiscard]] static u32 toggleParameterSphere() { return 1 * 5 + 2; }

    static constexpr float SphereRadius = 0.4f;
    static constexpr float Spacing = 1.0f;
    static constexpr u32 SphereSlices = 64, SphereStacks = 32;
    static constexpr float NominalSunAngularRadius = 0.00465f; // GiEnvironment default
    // World centre of the sphere at (column, row); row 0 sits on y = 0.
    [[nodiscard]] static glm::vec3 sphereCenter(const MaterialScenarioSpec&, u32 column, u32 row);

private:
    std::string scenario_, name_;
    u32 toggleEvery_ = 0, updates_ = 0;
    std::vector<EntityID> spheres_;
    EntityID sun_ = INVALID_ENTITY;
};

} // namespace phosphor
