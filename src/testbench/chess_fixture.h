#pragma once

#include "testbench/testbench.h"
#include <glm/glm.hpp>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace phosphor {

// F15.1/F15.3 (ticket 20) chess fixture: the roadmap's Track Studio case, rebuilt from public assets.  Bench 2 with
// `--material-scenario chess-<rig>[/key=value...]`, `--scene <ABeautifulGame.gltf>` (the pinned Khronos asset, loaded UNCHANGED:
// the pawn tops keep transmission 1, the tint in baseColorFactor and KHR_materials_volume thickness 0.22 with infinite attenuation)
// and `--material-overrides <file>` (fixtures/chess/chess-subsurface.overrides.json: the MaterialX Standard Surface subsurface values of
// the four kings and queens; texture paths in THAT file are relative to the glTF's directory, see fixtures/chess/README.md).
//
// Three declared rigs, each with a fixed camera and no scripted motion (deterministic):
//   chess-backlight   the black pawn row seen from the white side, an emissive checkerboard (the back light) standing just behind
//                     the row, environment 0.2, no sun; the other pieces are hidden (declared, they would block the line of sight).
//                     Options: transmission=F (factor of the pawn-top materials, default the asset's 1), tint=asset|white,
//                     panel=on|off (off: control, nothing but the environment behind the glass).
//   chess-sidelight   King_Black and Queen_Black (view=black, default) or King_White and Queen_White (view=white) seen from behind
//                     their own end of the board under one grazing sun, environment 0, shadows must be off (the pieces would shade each
//                     other).  Options: weight=F (subsurface weight of the four override materials, default the file's 1), scale=F
//                     (multiplier of the subsurface scale, default 1), sss=on|off (off: the override file is not applied: the asset's
//                     plain metal-rough pieces), view=black|white.
//   chess-shadow      the pawns and the board under one sun from behind the black pawn row; the camera looks down at the empty rank in
//                     front of the row where the pawn-top shadows fall.  Options: tops=on|opaque|hidden (the pawn-top materials as
//                     authored, transmission forced to 0, or the pawn tops invisible), tint=asset|white.
// Common option mask=1: every material is replaced by black (no textures, specular off) except the target materials of the rig, which
// are white diffuse; the only light is a sun of pi W/m^2 along the view direction and the environment is 0: a target pixel reads
// cos(N, V), everything else is black, so the capture is the rig's mask of the same geometry and camera.
// Mirrored by tools/f15_chess_check.py (tests/test_chess_fixture.cpp compares the names and the camera constants).
//
// SOURCE ONLY / NOT VERIFIED on a device until tools/f15_chess_check.py has captured the rigs.
struct ChessOptions {
    enum class Rig { Backlight, Sidelight, Shadow };
    enum class Tops { AsAuthored, Opaque, Hidden };
    Rig rig = Rig::Backlight;
    bool mask = false;
    bool emit = false;            // mask=1 only: the targets are pure emissive white (tens of thousands of emissive triangles, F15 GPU-safety regression)
    float transmission = -1.0f;   // < 0: the asset's factor
    bool whiteTint = false;       // base colour factor of the pawn tops forced to white
    bool panel = true;            // backlight: the emissive backdrop
    float weight = -1.0f;         // < 0: the override file's weight
    float scale = 1.0f;           // multiplier of subsurface.radius
    bool sss = true;              // apply the override file
    bool viewWhite = false;       // sidelight: the white pieces
    Tops tops = Tops::AsAuthored; // shadow
    bool pawnsOnly = false;       // hide kings, queens, castles, knights and bishops (declared by the rig)
};

class ChessFixture final : public TestBench {
public:
    // `scenePath` and `overridesPath` come from --scene and --material-overrides (setup throws without a loadable asset).
    ChessFixture(std::string scenario, std::string scenePath, std::string overridesPath);
    [[nodiscard]] static bool validScenario(std::string_view scenario);
    // Parse `chess-<rig>[/key=value...]`; false with `error` set for an unknown rig, key or value.
    [[nodiscard]] static bool parse(std::string_view scenario, ChessOptions& out, std::string& error);
    [[nodiscard]] static std::span<const std::string_view> rigs();
    // Radiance of the rig's declared constant environment (0 for an unknown scenario or a mask).
    [[nodiscard]] static float environmentRadiance(std::string_view scenario);

    void setup(ECS&, GpuScene&, TextureManager&) override;
    void update(float, ECS&) override {}
    void teardown(ECS&, GpuScene&) override;
    [[nodiscard]] const char* getName() const override { return name_.c_str(); }
    [[nodiscard]] const char* assetSource() const override { return scenePath_.c_str(); }
    [[nodiscard]] CameraSetup getDefaultCamera() const override;

    [[nodiscard]] const ChessOptions& options() const { return options_; }
    // Names of the materials that carry the rig's measured surface (the mask targets).
    [[nodiscard]] static std::vector<std::string> targetMaterials(const ChessOptions& options);
    // Declared constants (mirrored in tools/f15_chess_check.py).
    [[nodiscard]] static CameraSetup camera(const ChessOptions& options);
    [[nodiscard]] static glm::vec3 towardSun(const ChessOptions& options); // zero for the backlight rig
    static constexpr float BacklightEnvironment = 0.2f;
    static constexpr float PanelEmission = 3.0f;       // radiance of the bright cells
    static constexpr float PanelZ = 0.182f;            // the panel plane, between the black pawn row (z <= 0.1774) and the back rank (z >= 0.188)
    static constexpr float PanelHalfWidth = 0.32f, PanelHeight = 0.14f, CellSize = 0.006f;
    static constexpr float SunIrradiance = 3.0f;

private:
    std::string scenario_, name_, scenePath_, overridesPath_;
    ChessOptions options_;
    std::vector<EntityID> entities_;
};

} // namespace phosphor
