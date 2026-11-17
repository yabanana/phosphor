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
// F15.1 (ticket 07): a white point light of the declared model (inverse square times the engine's range
// window; intensity in W/sr, so the irradiance at distance d facing the light is intensity / d^2).
struct MaterialPointLight {
    glm::vec3 position{0.0f};
    float intensity = 0.0f;
    float range = 100.0f;
};

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
    // F15.1 (ticket 07): local point lights (DI lights with --lighting restir|clustered, resolve lights with legacy).
    std::vector<MaterialPointLight> pointLights;
    // F15.3 (ticket 09) clearcoat grids: KHR_materials_clearcoat factor per row, coat roughness per column
    // (both empty = no coat).  `baseIor` (optional, per row) puts a KHR_materials_ior block under the coat of
    // that row, so the "coat F0 from the material ior" control has a sphere that discriminates it.  A row with
    // factor 0 and no ior carries no extension (the neutral material).
    std::vector<float> clearcoatFactor;
    std::vector<float> clearcoatRoughness;
    std::vector<float> baseIor;
    // F15.1 (ticket 08) clip fixtures (tools/f15_clip_check.py): --material-toggle-every edits the two probe spheres
    // STRONGLY (feature sphere: ior 1.5 and 2.4; parameter sphere: specularColor 1 and 25, f0 clamped to 1), so that a history kept across the edit is visible at the F13.6 recovery threshold.  The default edit is the
    // small one of ticket 05 (ior 1.33 / weight 0.6).
    bool strongEdit = false;
    // F15.3 (ticket 10) sheen grids: KHR_materials_sheen colour per row (a zero colour carries no extension, the neutral
    // material) and sheen roughness per column; `roughness` then holds the (constant) base roughness of each column.
    std::vector<glm::vec3> sheenColor;
    std::vector<float>     sheenRoughness;
    // F15.3 (ticket 11) anisotropy grids: KHR_materials_anisotropy strength per row (a zero strength carries no extension,
    // the neutral material) and rotation (radians, counter-clockwise from the tangent) per column; `roughness` then holds the
    // (constant) base roughness of each column.  The spheres are COMBED spheres (ProceduralMeshes::generateCombedSphere,
    // comb direction +X, tangent handedness -1 like the engine's UV sphere): a uniform lobe direction facing the camera.
    // Mirrored carrier fixture (both empty elsewhere): per column the instance is mirrored (scale x = -1, INSTANCE_FLAG_MIRRORED)
    // and the handedness w (+1 or -1) of the mesh tangents.
    std::vector<float> anisotropyStrength;
    std::vector<float> anisotropyRotation;
    std::vector<bool>  mirroredColumn;
    std::vector<float> tangentHandedness;
    // F15.3 (ticket 12) iridescence grids: KHR_materials_iridescence factor and film ior per row, thickness (nm, the maximum:
    // there is no thickness texture) per column; `roughness` holds the (constant) base roughness of each column.  Every sphere
    // carries the block (thickness 0 included: the extension is present, the film is disabled by the specification's rule).
    std::vector<float> iridescenceFactor;
    std::vector<float> iridescenceIor;
    std::vector<float> iridescenceThickness;
    // F15.3 (ticket 16) thin subsurface (diffuse transmission) grids: the subsurface weight per row (0 carries no extension, the
    // neutral material) and the transmission colour per column (thin mode: SubsurfaceBlock mode Thin, weight, colour; radius,
    // radiusScale and anisotropy keep their defaults and are ignored by the thin lobe).  `quadCarriers`: the carriers are
    // single-sided quads in the z = 0 plane facing the camera (side SphereRadius * 2) instead of spheres, so that a light behind
    // them is not occluded by their own far side (the DI visibility rays of the point-light variants).
    std::vector<float>     subsurfaceWeight;
    std::vector<glm::vec3> subsurfaceColor;
    bool                   quadCarriers = false;
    float                  carrierYawDegrees  = 0.0f; // quadCarriers: yaw about the vertical (after facing the camera), ticket 19 panels
    float                  carrierRollDegrees = 0.0f; // quadCarriers: roll about the quad normal before the yaw
    // F15.3 (ticket 17) volumetric subsurface grids (SubsurfaceMode::Volumetric, OpenPBR semantics of ADR-0005): the subsurface
    // weight and radius (metres) per COLUMN (weight 0 carries no extension, the neutral material), the multiple-scattering
    // colour, radiusScale (the per-channel mean free path is radius * radiusScale) and the Henyey-Greenstein anisotropy per ROW.  A
    // volumetric block needs a volume block (validate()): every piece carries one of thickness = the sphere diameter and neutral
    // attenuation (it marks the object as a solid, nothing else).  `cameraDistance` > 0 overrides the camera rule below.
    std::vector<float>     volumetricWeight;
    std::vector<float>     volumetricRadius;
    std::vector<glm::vec3> volumetricColor;
    std::vector<glm::vec3> volumetricRadiusScale;
    std::vector<float>     volumetricAnisotropy;
    float                  cameraDistance = 0.0f;
    // F15.3 (ticket 19) combination grids: `combo` holds one cell per sphere (row-major, rows x columns like every grid; the rows keep
    // `metallic`, the columns `roughness`, the base colour is `baseColor`) with the extension blocks of that sphere.  A block is present
    // when its factor/colour/weight is non-zero (specular: `hasSpecular`), so a cell may carry any subset of the lobes; the neutral
    // cell carries no extension at all.  `combedCarriers` selects the combed spheres (needed by every cell with anisotropy).
    struct ComboCell {
        bool      hasSpecular    = false;
        float     ior            = 1.5f;
        float     specularFactor = 1.0f;
        glm::vec3 specularColor{1.0f};
        float     coatFactor     = 0.0f;
        float     coatRoughness  = 0.0f;
        glm::vec3 sheenColor{0.0f};
        float     sheenRoughness = 0.3f;
        float     anisoStrength  = 0.0f;
        float     anisoRotation  = 0.0f;
        float     iridFactor     = 0.0f;
        float     iridIor        = 1.3f;
        float     iridThickness  = 400.0f; // nm, the maximum (no thickness texture)
        float     thinWeight     = 0.0f;
        glm::vec3 thinColor{1.0f};
    };
    std::vector<ComboCell> combo;
    bool                   combedCarriers = false;
    // F15.1 (ticket 19, tools/f15_thin_gi_back_check.py): a diffuse WALL 0.6 m behind the carriers (albedo `giWallAlbedo`, facing the
    // camera, 8 m square, lit by the sun of the scenario) and a tiny anchor instance off-screen in front of them, so that the DDGI
    // volume covers both sides of the thin carriers while only the BACK side sees the lit wall (nothing in front of the quads reflects).
    bool                   giBackWall = false;
    float                  giWallAlbedo = 0.0f;
    // F15 closure (ticket 06, tools/f15_lobe_guide_coplanar_check.py): when non-empty the carrier is ONE mesh (one draw) of
    // layerTiltDegrees.size() exactly coplanar, fully overlapping quads (identical positions: every fragment of the stack has the same
    // depth), the vertex normals of layer i tilted by layerTiltDegrees[i] degrees about +X, so the layer a fragment belongs to shows in its
    // shading normal (colour attachments and the coat normal of the lobe guide).
    std::vector<float>     layerTiltDegrees;
    // F15.3 (ticket 22, texture closure): TEXTURED lobe fixtures.  `textureFixture` selects the texture set and the lobe of the
    // grid (rows keep `metallic`, columns `roughness` = the base roughness); `textureCells` holds one cell per sphere (row-major).
    // `use` is a bit set naming which texture slots of the lobe the cell binds (bit 0, 1, 2 = the first, second, third slot of the
    // lobe in tools/f15_reference.py "KHR texture fixtures"), a..d and `color` are the factor parameters of that sphere (their
    // meaning per lobe is documented at the cells in addTextureScenarios).  The textures are small (32 x 16), smooth and generated
    // deterministically in setup(); the reference reads the exported texels, never this generator.
    enum class TextureFixture : u8 { None, Coat, Anisotropy, Iridescence, Sheen, Thin };
    struct TextureCell {
        u32       use = 0;
        float     a = 0.0f, b = 0.0f, c = 0.0f, d = 0.0f;
        glm::vec3 color{1.0f};
    };
    TextureFixture           textureFixture = TextureFixture::None;
    std::vector<TextureCell> textureCells;
    [[nodiscard]] bool hasTextureFixture() const { return textureFixture != TextureFixture::None; }
    [[nodiscard]] bool hasComboGrid() const { return !combo.empty(); }
    [[nodiscard]] bool hasVolumetricGrid() const { return !volumetricWeight.empty(); }
    [[nodiscard]] bool hasSubsurfaceGrid() const { return !subsurfaceWeight.empty(); }
    [[nodiscard]] bool hasIridescenceGrid() const { return !iridescenceFactor.empty(); }
    [[nodiscard]] bool hasSpecularGrid() const { return !ior.empty(); }
    [[nodiscard]] bool hasAnisotropyGrid() const { return !anisotropyStrength.empty(); }
    [[nodiscard]] bool hasClearcoatGrid() const { return !clearcoatFactor.empty(); }
    [[nodiscard]] bool hasSheenGrid() const { return !sheenColor.empty(); }
};

class MaterialValidation final : public TestBench {
public:
    // `toggleEvery` (F15.1, specular grids only): every N updates the extension of the two probe spheres
    // is edited (a parameter change on one, a feature toggle on the other: a counted record rewrite,
    // never a reallocation); 0 = static scene.  The edit of event k is visible in the capture of frame
    // k * N - 1 (the update of frame f runs before frame f is rendered; frames count from 0).
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
    /// F15.1 (ticket 17/18, walk clip): with `motion` the camera strafes +-1.5 m sideways looking at the scene target (--material-motion), never a camera cut.
    bool scriptedCamera(double t, glm::vec3& position, glm::vec3& target, bool& cut) const override;
    void setMotion(bool motion) { motion_ = motion; }

    [[nodiscard]] std::string_view scenario() const { return scenario_; }
    [[nodiscard]] std::span<const EntityID> spheres() const { return spheres_; } // row-major: row * columns + column
    [[nodiscard]] EntityID sun() const { return sun_; }
    [[nodiscard]] std::span<const EntityID> pointLights() const { return pointLights_; }

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
    bool motion_ = false;
    u32 toggleEvery_ = 0, updates_ = 0;
    std::vector<EntityID> spheres_;
    EntityID sun_ = INVALID_ENTITY;
    std::vector<EntityID> pointLights_;
    std::vector<EntityID> extras_; // giBackWall: the wall and the anchor
};

} // namespace phosphor
