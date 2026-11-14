#include <doctest/doctest.h>
#include "testbench/chess_fixture.h"
#include "testbench/material_validation.h"
#include "renderer/gpu_scene.h"
#include "renderer/material_description.h"
#include "renderer/material_description_io.h"
#include "scene/components.h"
#include "scene/ecs.h"
#include "null_texture_manager.h"
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
using namespace phosphor;

namespace {
std::string readFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream text;
    text << in.rdbuf();
    return text.str();
}
std::string overridesFile() { return std::string(PHOSPHOR_SOURCE_DIR) + "/fixtures/chess/chess-subsurface.overrides.json"; }

// The pinned Khronos asset: PHOSPHOR_CHESS_GLTF, else <repo>/build/khronos-corpus/ABeautifulGame/glTF/ABeautifulGame.gltf.  Tests that
// need it skip cleanly when it is absent (the corpus is downloaded by tools/fetch_khronos_corpus.py, never committed).
std::string chessAsset() {
    if (const char* env = std::getenv("PHOSPHOR_CHESS_GLTF")) return std::filesystem::exists(env) ? env : "";
    const std::string path = std::string(PHOSPHOR_SOURCE_DIR) + "/build/khronos-corpus/ABeautifulGame/glTF/ABeautifulGame.gltf";
    return std::filesystem::exists(path) ? path : "";
}

// The 60 degree vertical field of view, 16:9: normalised device coordinates of a world point (depth > 0 in front of the camera).
struct Projection {
    float x, y, depth;
};
Projection project(const CameraSetup& camera, glm::vec3 p) {
    const glm::vec3 forward = glm::normalize(camera.target - camera.position);
    const glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0, 1, 0)));
    const glm::vec3 up = glm::cross(right, forward);
    const glm::vec3 d = p - camera.position;
    const float depth = glm::dot(d, forward), tanHalf = 0.57735027f;
    return {glm::dot(d, right) / (depth * tanHalf * 16.0f / 9.0f), glm::dot(d, up) / (depth * tanHalf), depth};
}
constexpr float kBlackTopX[8] = {-0.2192f, -0.15675f, -0.0938f, -0.0313f, 0.03125f, 0.0938f, 0.1561f, 0.21875f};
constexpr float kTopY = 0.0704f, kTopZ = 0.1564f;
} // namespace

TEST_CASE("F15.1 ticket 20: the chess scenarios parse, reject unknown keys and declare their environment") {
    CHECK(ChessFixture::rigs().size() == 3);
    for (auto name : ChessFixture::rigs()) {
        CHECK(ChessFixture::validScenario(name));
        CHECK_FALSE(MaterialValidation::validScenario(name));  // a separate bench class
    }
    ChessOptions o;
    std::string error;
    REQUIRE(ChessFixture::parse("chess-backlight/transmission=0.25/tint=white/panel=off/mask=1", o, error));
    CHECK(o.rig == ChessOptions::Rig::Backlight);
    CHECK(o.transmission == doctest::Approx(0.25f));
    CHECK(o.whiteTint);
    CHECK_FALSE(o.panel);
    CHECK(o.mask);
    CHECK(o.pawnsOnly);
    REQUIRE(ChessFixture::parse("chess-sidelight/weight=0.5/scale=2/view=white/sss=off", o, error));
    CHECK(o.weight == doctest::Approx(0.5f));
    CHECK(o.scale == doctest::Approx(2.0f));
    CHECK(o.viewWhite);
    CHECK_FALSE(o.sss);
    CHECK_FALSE(o.pawnsOnly);
    REQUIRE(ChessFixture::parse("chess-shadow/tops=opaque", o, error));
    CHECK(o.tops == ChessOptions::Tops::Opaque);
    CHECK(ChessFixture::parse("chess-shadow/tops=hidden/tint=white", o, error));
    for (const char* bad : {"chess", "chess-ghost", "chess-backlight/tops=on", "chess-sidelight/transmission=1", "chess-backlight/transmission=1.5",
                            "chess-backlight/transmission=x", "chess-sidelight/scale=0", "chess-sidelight/weight=-1", "chess-shadow/panel=off",
                            "chess-backlight/bogus=1", "chess-backlight/panel", "chess-sidelight/view=grey", "chess-backlight/mask=2"}) {
        CHECK_FALSE(ChessFixture::validScenario(bad));
        CHECK_THROWS(ChessFixture(bad, "x", ""));
    }
    CHECK(ChessFixture::environmentRadiance("chess-backlight") == doctest::Approx(0.2f));
    CHECK(ChessFixture::environmentRadiance("chess-backlight/mask=1") == 0.0f);
    CHECK(ChessFixture::environmentRadiance("chess-sidelight") == 0.0f);
    CHECK(ChessFixture::environmentRadiance("chess-shadow") == 0.0f);
    CHECK(ChessFixture::environmentRadiance("unknown") == 0.0f);
    CHECK(MaterialValidation::environmentRadiance("chess-backlight") == doctest::Approx(0.2f));  // the engine reads it there
}

TEST_CASE("F15.1 ticket 20: the override file carries the MaterialX subsurface values of the four kings and queens") {
    std::vector<NamedMaterial> materials;
    for (const char* name : {"King_Black", "King_White", "Queen_Black", "Queen_White", "Pawn_Top_Black"}) materials.push_back({name, {}});
    std::string error;
    std::vector<size_t> changed;
    REQUIRE(applyMaterialOverrides(readFile(overridesFile()), materials, error, &changed));
    CHECK(error.empty());
    CHECK(changed == std::vector<size_t>{0, 1, 2, 3});  // the pawn top is not touched: it is loaded unchanged from the asset
    const char* texture[4] = {"King_black_base_color.jpg", "King_white_base_color.jpg", "Queen_black_base_color.jpg", "Queen_white_base_color.jpg"};
    for (size_t i = 0; i < 4; ++i) {
        const MaterialDescription& d = materials[i].description;
        CHECK(validate(d).empty());
        REQUIRE(d.subsurface.has_value());
        const SubsurfaceBlock& s = *d.subsurface;
        CHECK(s.mode == SubsurfaceMode::Volumetric);  // MaterialX thin_walled false
        CHECK(s.weight == 1.0f);                      // the scattering map is not in the pinned subset: constant (see the conversion note)
        CHECK_FALSE(s.weightTexture.present());
        CHECK(s.radius == doctest::Approx(i == 3 ? 0.001f : 0.003f));  // subsurface_scale: 0.003 m, Queen_White 0.001 m
        for (int c = 0; c < 3; ++c) {
            CHECK(s.radiusScale[c] == 1.0f);
            CHECK(s.color[c] == 1.0f);
            CHECK(meanFreePath(s)[c] == doctest::Approx(s.radius));  // times the texel of radiusScaleTexture
        }
        CHECK(s.anisotropy == 0.0f);
        // subsurface_color and subsurface_radius are both the base colour image.
        CHECK(s.colorTexture.path == texture[i]);
        CHECK(s.radiusScaleTexture.path == texture[i]);
        CHECK(s.colorTexture.colorSpace == TextureColorSpace::SRGB);
        CHECK(s.radiusScaleTexture.colorSpace == TextureColorSpace::SRGB);
        REQUIRE(d.volume.has_value());  // only the solid marker the validation demands: neutral attenuation
        CHECK(d.volume->thickness > 0.0f);
        CHECK(d.volume->attenuationDistance == DESCRIPTION_INFINITY);
        CHECK_FALSE(d.transmission.has_value());
    }
}

TEST_CASE("F15.1 ticket 20: the rigs' fixed cameras see their subjects (mirrored constants of tools/f15_chess_check.py)") {
    ChessOptions o;
    std::string error;
    REQUIRE(ChessFixture::parse("chess-backlight", o, error));
    const CameraSetup back = ChessFixture::camera(o);
    for (float x : kBlackTopX) {
        const float z = x == -0.0313f ? 0.0936f : kTopZ;  // Pawn_Top_B4 has advanced two squares
        const Projection p = project(back, {x, kTopY, z});
        CHECK(p.depth > 0.3f);
        CHECK(std::abs(p.x) < 0.8f);
        CHECK(std::abs(p.y) < 0.8f);
        // The straight line through the top reaches the emissive panel (z = PanelZ) inside it, above the board.
        const glm::vec3 d = glm::vec3(x, kTopY, z) - back.position;
        const float t = (ChessFixture::PanelZ - back.position.z) / d.z;
        const glm::vec3 hit = back.position + t * d;
        CHECK(std::abs(hit.x) < ChessFixture::PanelHalfWidth);
        CHECK(hit.y > 0.01f);
        CHECK(hit.y < ChessFixture::PanelHeight);
    }
    // The line of sight to the tops clears the white pawn row (z about -0.156, up to y = 0.0814).
    const float tWhite = (-0.156f - back.position.z) / (kTopZ - back.position.z);
    CHECK(back.position.y + tWhite * (kTopY - back.position.y) > 0.0814f + 0.01f);
    // The panel sits between the black pawn row (z <= 0.1774) and the back rank (z >= 0.188).
    CHECK(ChessFixture::PanelZ > 0.1774f);
    CHECK(ChessFixture::PanelZ < 0.188f);
    CHECK(glm::dot(ChessFixture::towardSun(o), ChessFixture::towardSun(o)) == 0.0f);

    REQUIRE(ChessFixture::parse("chess-sidelight", o, error));
    for (bool white : {false, true}) {
        o.viewWhite = white;
        const CameraSetup cam = ChessFixture::camera(o);
        const float z = white ? -1.0f : 1.0f;
        for (float x : {-0.031f, 0.031f}) {  // King and Queen centres
            const Projection p = project(cam, {x, 0.095f, 0.22f * z});
            CHECK(p.depth > 0.2f);
            CHECK(std::abs(p.x) < 0.5f);
            CHECK(std::abs(p.y) < 0.8f);
        }
        const glm::vec3 sun = ChessFixture::towardSun(o);
        CHECK(glm::length(sun) == doctest::Approx(1.0f));
        CHECK(sun.y > 0.0f);
        // The sun is lateral: its component along the view direction is small, its x component dominant.
        const glm::vec3 forward = glm::normalize(cam.target - cam.position);
        CHECK(std::abs(glm::dot(sun, forward)) < 0.4f);
        CHECK(std::abs(sun.x) > 0.8f);
    }
    REQUIRE(ChessFixture::parse("chess-shadow", o, error));
    const CameraSetup shadow = ChessFixture::camera(o);
    const glm::vec3 sun = ChessFixture::towardSun(o);
    CHECK(glm::length(sun) == doctest::Approx(1.0f));
    // The shadow of a black pawn top (y = 0.0704, z = 0.1564) on the board plane (y = 0.0185) falls in front of the row, inside the view.
    const float shift = (kTopY - 0.0185f) * sun.z / sun.y;
    const Projection p = project(shadow, {0.0313f, 0.0185f, kTopZ - shift});
    CHECK(shift > 0.05f);
    CHECK(kTopZ - shift > 0.0f);
    CHECK(std::abs(p.x) < 0.5f);
    CHECK(std::abs(p.y) < 0.8f);
}

TEST_CASE("F15.1 ticket 20: the chess rigs on the pinned ABeautifulGame (skipped without the corpus)") {
    const std::string asset = chessAsset();
    if (asset.empty()) {
        MESSAGE("skipped: ABeautifulGame is not available (tools/fetch_khronos_corpus.py or PHOSPHOR_CHESS_GLTF)");
        return;
    }
    auto build = [&](const std::string& scenario, ECS& ecs, GpuScene& scene, test::NullTextureManager& textures) {
        ChessFixture f(scenario, asset, overridesFile());
        f.setup(ecs, scene, textures);
    };
    auto materialIndex = [](GpuScene& scene, const std::string& name) {
        const auto found = scene.findMaterial(name);
        REQUIRE(found.has_value());
        return *found;
    };
    SUBCASE("the pawn tops are the asset's, unchanged") {
        ECS ecs;
        GpuScene scene;
        test::NullTextureManager textures;
        build("chess-backlight", ecs, scene, textures);
        for (const char* name : {"Pawn_Top_Black", "Pawn_Top_White"}) {
            const MaterialDescription& d = scene.materialDescriptions()[materialIndex(scene, name)];
            REQUIRE(d.transmission.has_value());
            CHECK(d.transmission->factor == 1.0f);
            REQUIRE(d.volume.has_value());
            CHECK(d.volume->thickness == doctest::Approx(0.22f));
            CHECK(d.volume->attenuationDistance == DESCRIPTION_INFINITY);
        }
        const auto& black = scene.materialDescriptions()[materialIndex(scene, "Pawn_Top_Black")].base.baseColor;
        CHECK(black[0] == doctest::Approx(0.2995f));
        CHECK(black[1] == doctest::Approx(0.5f));
        CHECK(black[2] == doctest::Approx(0.45f));
        const auto& white = scene.materialDescriptions()[materialIndex(scene, "Pawn_Top_White")].base.baseColor;
        CHECK(white[2] == doctest::Approx(0.828f));
    }
    SUBCASE("options edit only what they declare") {
        ECS ecs;
        GpuScene scene;
        test::NullTextureManager textures;
        build("chess-backlight/transmission=0.25/tint=white", ecs, scene, textures);
        const MaterialDescription& d = scene.materialDescriptions()[materialIndex(scene, "Pawn_Top_Black")];
        CHECK(d.transmission->factor == doctest::Approx(0.25f));
        CHECK(d.base.baseColor[0] == 1.0f);
        CHECK(d.volume->thickness == doctest::Approx(0.22f));
    }
    SUBCASE("the sidelight rig binds the four subsurface materials; sss=off leaves the asset's pieces plain") {
        {
            ECS ecs;
            GpuScene scene;
            test::NullTextureManager textures;
            build("chess-sidelight/weight=0.5/scale=2", ecs, scene, textures);
            for (const char* name : {"King_Black", "King_White", "Queen_Black", "Queen_White"}) {
                const MaterialDescription& d = scene.materialDescriptions()[materialIndex(scene, name)];
                REQUIRE(d.subsurface.has_value());
                CHECK(d.subsurface->weight == doctest::Approx(0.5f));
                const bool queenWhite = std::string(name) == "Queen_White";
                CHECK(d.subsurface->radius == doctest::Approx(queenWhite ? 0.002f : 0.006f));
                CHECK(d.subsurface->colorTexture.index != INVALID_TEXTURE_INDEX);
                CHECK(d.subsurface->radiusScaleTexture.index == d.subsurface->colorTexture.index);  // the same image
                // = the base colour texture: the override path resolves its own upload of the same JPEG (the texture manager does not
                // share indices between the glTF image and a path), so the decoded texels must be identical.
                REQUIRE(d.subsurface->colorTexture.index < textures.uploads.size());
                REQUIRE(d.base.baseColorTexture.index < textures.uploads.size());
                const auto& a = textures.uploads[d.subsurface->colorTexture.index];
                const auto& b = textures.uploads[d.base.baseColorTexture.index];
                CHECK(a.width == b.width);
                CHECK(a.height == b.height);
                CHECK(a.sRGB == b.sRGB);
                CHECK(a.rgba == b.rgba);
                CHECK((featureMask(d) & featureBit(MaterialFeature::Subsurface)) != 0u);
            }
            const MaterialDescription& pawn = scene.materialDescriptions()[materialIndex(scene, "Pawn_Top_Black")];
            CHECK(pawn.transmission->factor == 1.0f);  // untouched
        }
        ECS ecs;
        GpuScene scene;
        test::NullTextureManager textures;
        build("chess-sidelight/sss=off", ecs, scene, textures);
        CHECK_FALSE(scene.materialDescriptions()[materialIndex(scene, "King_Black")].subsurface.has_value());
    }
    SUBCASE("mask mode: every material is black, the targets white diffuse, one sun along the view direction, specular off") {
        ECS ecs;
        GpuScene scene;
        test::NullTextureManager textures;
        build("chess-backlight/mask=1", ecs, scene, textures);
        for (size_t i = 0; i < scene.materialNames().size(); ++i) {
            const MaterialDescription& d = scene.materialDescriptions()[i];
            const bool target = scene.materialNames()[i] == "Pawn_Top_Black";
            CHECK(d.base.emissive[0] == 0.0f);
            CHECK(d.base.baseColor[0] == (target ? 1.0f : 0.0f));
            CHECK(d.specular->specularFactor == 0.0f);
            CHECK(featureMask(d) == featureBit(MaterialFeature::SpecularIor));  // nothing but the specular-off block
            CHECK_FALSE(d.transmission.has_value());
            CHECK_FALSE(d.subsurface.has_value());
        }
        size_t lights = 0;
        for (EntityID e = 0; e < ecs.entityCount(); ++e) {
            if (!std::as_const(ecs).hasComponent<LightComponent>(e)) continue;
            ++lights;
            const auto& light = std::as_const(ecs).getComponent<LightComponent>(e);
            CHECK(light.intensity == doctest::Approx(3.14159265f));
            const CameraSetup view = ChessFixture::camera(ChessOptions{});
            CHECK(view.position.z < 0.0f);  // the backlight camera: the sun (pi along the view) is built from it
        }
        CHECK(lights == 1);
    }
    SUBCASE("the backlight rig hides the other pieces and builds the panel; tops=hidden hides the pawn tops") {
        ECS ecs;
        GpuScene scene;
        test::NullTextureManager textures;
        build("chess-backlight", ecs, scene, textures);
        size_t visible = 0, hiddenTotal = 0, emissive = 0;
        for (EntityID e = 0; e < ecs.entityCount(); ++e) {
            if (!std::as_const(ecs).hasComponent<MeshInstanceComponent>(e)) continue;
            const auto& instance = std::as_const(ecs).getComponent<MeshInstanceComponent>(e);
            (instance.isVisible() ? visible : hiddenTotal)++;
            if (instance.isVisible() != instance.castsShadows()) FAIL("visible and shadow roles must be equal");
            if (std::as_const(ecs).hasComponent<MaterialComponent>(e) && std::as_const(ecs).getComponent<MaterialComponent>(e).emissiveFactor.x > 0.0f) ++emissive;
        }
        CHECK(hiddenTotal == 16);   // 2 kings, 2 queens, 4 castles, 4 knights, 4 bishops
        CHECK(visible > 16 + 1 + 100);  // 16 pawn bodies + 16 tops + board + the panel
        CHECK(emissive > 100);
    }
}

TEST_CASE("F15.1 ticket 20: the runner and the bench agree on the rig names and camera constants") {
    const std::string source = readFile(std::string(PHOSPHOR_SOURCE_DIR) + "/tools/f15_chess_check.py");
    REQUIRE_FALSE(source.empty());
    for (auto name : ChessFixture::rigs()) CHECK(source.find(std::string("\"") + std::string(name)) != std::string::npos);
    ChessOptions o;
    std::string error;
    REQUIRE(ChessFixture::parse("chess-backlight", o, error));
    const CameraSetup c = ChessFixture::camera(o);
    // CAMERA_BACKLIGHT = (position, target) in the runner (the mask centroid check needs the same projection).
    std::smatch m;
    const std::regex pattern(R"(CAMERA_BACKLIGHT = \(\(([-0-9.]+), ([-0-9.]+), ([-0-9.]+)\), \(([-0-9.]+), ([-0-9.]+), ([-0-9.]+)\)\))");
    REQUIRE(std::regex_search(source, m, pattern));
    CHECK(std::stof(m[1]) == doctest::Approx(c.position.x));
    CHECK(std::stof(m[2]) == doctest::Approx(c.position.y));
    CHECK(std::stof(m[3]) == doctest::Approx(c.position.z));
    CHECK(std::stof(m[4]) == doctest::Approx(c.target.x));
    CHECK(std::stof(m[5]) == doctest::Approx(c.target.y));
    CHECK(std::stof(m[6]) == doctest::Approx(c.target.z));
}
