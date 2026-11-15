#pragma once

#include "testbench/testbench.h"
#include <glm/glm.hpp>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace phosphor {

// F15 closure (tickets 13 and 15) glass-RECEIVER fixtures (bench 2 with --material-scenario NAME, the names start with "glass-").
// Two questions the floor-receiver benches cannot ask, both about a pane that is itself lit by a delta light:
//   * glass-receiver-stack / glass-receiver-single: a thin rough pane B in the shadow of a clear pane A (the sun on the camera side of both):
//     the sun reflection on B must be attenuated by A once (the CSM transmittance map sums every surface in a texel; B's own surface must
//     not attenuate B again).  `single` is the same scene without A: the unshadowed baseline (ratio = the transmittance of A).
//   * glass-btdf-sun / glass-btdf-point: vertical rough thin-walled panes (roughness {0.1, 0.3, 0.6} x white / tinted) lit by ONE delta light
//     BEHIND them: the camera sees only the transmitted lobe (the glTF thin-walled specular_btdf of the light mirrored to the front side,
//     (1 - F) t (1 - metallic) tint f_t |N.L| x radiance), checked per pixel against tools/f15_glass_receiver_check.py.
// Environment 0, no floor, no emissive geometry: nothing but the delta light lights the panes.  Mirrored by that tool's constants
// (tests/test_glass_receiver_validation.cpp pins the geometry).
struct GlassReceiverPane {
    glm::vec3 center{0.0f};
    glm::vec2 size{1.0f};      // extent along the pane's two axes (a unit plane scaled)
    bool horizontal = false;   // false: vertical, facing +Z; true: facing +Y
    float roughness = 0.0f;
    glm::vec3 baseColor{1.0f};
    float transmission = 1.0f;
    float ior = 1.5f;
};

struct GlassReceiverSpec {
    std::string_view name;
    std::vector<GlassReceiverPane> panes;
    bool sun = false;
    float sunIrradiance = 0.0f;                  // W/m^2 on a surface facing the sun
    glm::vec3 towardSun{0.0f, 1.0f, 0.0f};       // unit vector from the surface towards the sun
    bool point = false;
    glm::vec3 pointPosition{0.0f};
    float pointIntensity = 0.0f;                 // W/sr
    glm::vec3 cameraTarget{0.0f};
    float cameraDistance = 5.0f;
    float cameraElevationDegrees = 0.0f;         // above the horizontal, looking down at the target from +Z
};

class GlassReceiverValidation final : public TestBench {
public:
    explicit GlassReceiverValidation(std::string scenario);
    [[nodiscard]] static bool validScenario(std::string_view scenario);
    [[nodiscard]] static std::span<const std::string_view> scenarios();
    [[nodiscard]] static const GlassReceiverSpec& spec(std::string_view scenario); // throws std::invalid_argument when unknown

    void setup(ECS&, GpuScene&, TextureManager&) override;
    void update(float, ECS&) override {}
    void teardown(ECS&, GpuScene&) override;
    [[nodiscard]] const char* getName() const override { return name_.c_str(); }
    [[nodiscard]] const char* assetSource() const override { return "procedural-glass-receiver-validation-v1"; }
    [[nodiscard]] CameraSetup getDefaultCamera() const override;

    [[nodiscard]] std::span<const EntityID> panes() const { return panes_; } // spec order
    [[nodiscard]] EntityID light() const { return light_; }
    [[nodiscard]] EntityID anchor() const { return anchor_; } // a small black opaque cube that gives the scene a volume (the static reflection probe capture needs a non-degenerate bound)

    static constexpr float NominalSunAngularRadius = 0.00465f;
    static constexpr glm::vec3 AnchorCenter{3.2f, 0.05f, -1.0f}; // outside every pane footprint, shadow and view region of the gated pixels
    static constexpr float AnchorHalfExtent = 0.05f;

private:
    std::string scenario_, name_;
    std::vector<EntityID> panes_;
    EntityID light_ = INVALID_ENTITY, anchor_ = INVALID_ENTITY;
};

} // namespace phosphor
