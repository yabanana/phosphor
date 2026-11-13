#include <doctest/doctest.h>
#include "testbench/material_validation.h"
#include "testbench/transmission_validation.h"
#include "renderer/gpu_scene.h"
#include "renderer/material_description.h"
#include "scene/components.h"
#include "scene/ecs.h"
#include "null_texture_manager.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <regex>
#include <set>
#include <sstream>
#include <utility>
using namespace phosphor;

namespace {
std::vector<std::string> runnerKeys() {
    std::ifstream in(std::string(PHOSPHOR_SOURCE_DIR) + "/tools/f15_check.py");
    std::stringstream text;
    text << in.rdbuf();
    const std::string source = text.str();
    const size_t start = source.find("SCENARIOS = {"), end = source.find("\nGPU_PROCESS", start);
    std::vector<std::string> keys;
    if (start == std::string::npos || end == std::string::npos) return keys;
    const std::string block = source.substr(start, end - start);
    const std::regex key(R"re(\n    "(transmission-[a-z0-9-]+(@[a-z]+)?)": \{)re");
    for (std::sregex_iterator it(block.begin(), block.end(), key), last; it != last; ++it) keys.push_back((*it)[1]);
    return keys;
}
} // namespace

TEST_CASE("F15.3 ticket 13: the transmission scenarios are exactly the eight declared names") {
    const auto names = TransmissionValidation::scenarios();
    CHECK(names.size() == 8);
    for (auto name : {"transmission-roughness-ior", "transmission-attenuation", "transmission-factor-tint", "transmission-thin-volume",
                      "transmission-single-sided", "transmission-glass-behind-glass", "transmission-emissive-panel", "transmission-rtoff"}) {
        CHECK(TransmissionValidation::validScenario(name));
        CHECK_FALSE(MaterialValidation::validScenario(name)); // a separate bench class, not a grid scenario
    }
    CHECK_FALSE(TransmissionValidation::validScenario("base-grid-sun"));
    CHECK_THROWS(TransmissionValidation{"unknown"});
    // The declared environment is reachable through the shared --material-scenario helper.
    for (auto name : names) CHECK(MaterialValidation::environmentRadiance(name) == 0.5f);
}

TEST_CASE("F15.3 ticket 13: the bench and the runner (tools/f15_check.py) list the same transmission scenarios") {
    const auto keys = runnerKeys();
    REQUIRE_FALSE(keys.empty());
    std::set<std::string> runner;
    for (const auto& key : keys) runner.insert(key.substr(0, key.find('@')));
    std::set<std::string> bench;
    for (auto name : TransmissionValidation::scenarios()) bench.insert(std::string(name));
    CHECK(runner == bench);
}

TEST_CASE("F15.3 ticket 13: glass objects carry exact factors, the transmission/volume/ior blocks and no textures") {
    for (auto name : TransmissionValidation::scenarios()) {
        ECS ecs;
        GpuScene scene;
        test::NullTextureManager textures;
        TransmissionValidation f{std::string(name)};
        f.setup(ecs, scene, textures);
        const auto& s = TransmissionValidation::spec(name);
        REQUIRE(f.glass().size() == s.glass.size());
        CHECK_FALSE(s.glass.empty());
        for (size_t k = 0; k < s.glass.size(); ++k) {
            const GlassObject& g = s.glass[k];
            const auto& m = std::as_const(ecs).getComponent<MaterialComponent>(f.glass()[k]);
            REQUIRE(m.extension);
            REQUIRE(m.extension->transmission);
            REQUIRE(m.extension->volume);
            REQUIRE(m.extension->specular);
            CHECK(m.extension->transmission->factor == g.transmission);
            CHECK(m.extension->specular->ior == g.ior);
            CHECK(m.extension->volume->thickness == g.thickness);
            CHECK(m.extension->volume->attenuationDistance == g.attenuationDistance);
            for (int c = 0; c < 3; ++c) CHECK(m.extension->volume->attenuationColor[c] == g.attenuationColor[c]);
            // transmission > 0 is the Transmission bit, thickness > 0 the Volume bit (thin-walled = no Volume bit).
            const u32 mask = featureMask(*m.extension);
            CHECK(((mask & featureBit(MaterialFeature::Transmission)) != 0) == (g.transmission > 0.0f));
            CHECK(((mask & featureBit(MaterialFeature::Volume)) != 0) == (g.thickness > 0.0f));
            CHECK(m.baseColorFactor == glm::vec4(g.baseColor, 1.0f));
            CHECK(m.metallicFactor == 0.0f);
            CHECK(m.roughnessFactor == g.roughness);
            CHECK(m.doubleSided == g.doubleSided);
            CHECK(m.baseColorTexIndex == INVALID_TEXTURE_INDEX);
            CHECK(m.normalTexIndex == INVALID_TEXTURE_INDEX);
            CHECK(m.metallicRoughnessTexIndex == INVALID_TEXTURE_INDEX);
            CHECK(m.emissiveFactor == glm::vec3(0.0f));
            // Spec ranges of KHR_materials_transmission/volume/ior.
            CHECK(g.transmission >= 0.0f);
            CHECK(g.transmission <= 1.0f);
            CHECK(g.ior >= 1.0f);
            for (int c = 0; c < 3; ++c) {
                CHECK(g.attenuationColor[c] > 0.0f); // a zero channel would be an infinite absorption coefficient
                CHECK(g.attenuationColor[c] <= 1.0f);
            }
        }
        f.teardown(ecs, scene);
        CHECK(f.glass().empty());
    }
}

TEST_CASE("F15.3 ticket 13: nothing diffuse sits behind the glass (gating policy: emitters and the environment only)") {
    for (auto name : TransmissionValidation::scenarios()) {
        ECS ecs;
        GpuScene scene;
        test::NullTextureManager textures;
        TransmissionValidation f{std::string(name)};
        f.setup(ecs, scene, textures);
        const auto& s = TransmissionValidation::spec(name);
        size_t expected = 0;
        if (s.backdrop == TransmissionScenarioSpec::Backdrop::Checker)
            expected = size_t(TransmissionValidation::CheckerColumns * TransmissionValidation::CheckerRows / 2);
        if (s.backdrop == TransmissionScenarioSpec::Backdrop::Panels) expected = s.panels.size();
        CHECK(f.emitters().size() == expected);
        for (EntityID e : f.emitters()) {
            const auto& m = std::as_const(ecs).getComponent<MaterialComponent>(e);
            CHECK(m.baseColorFactor == glm::vec4(0.0f, 0.0f, 0.0f, 1.0f)); // no diffuse base to light through the glass
            CHECK(m.emissiveFactor == glm::vec3(s.emission));
            CHECK_FALSE(m.extension);
            CHECK(m.emissiveTexIndex == INVALID_TEXTURE_INDEX);
            // The emitters sit behind every glass object (the camera looks down -Z).
            const auto& t = std::as_const(ecs).getComponent<TransformComponent>(e);
            for (const GlassObject& g : s.glass) CHECK(t.position.z < g.center.z - 0.3f);
        }
        if (s.backdrop == TransmissionScenarioSpec::Backdrop::None) CHECK(f.emitters().empty());
        CHECK(s.environment == 0.5f);
        f.teardown(ecs, scene);
    }
}

TEST_CASE("F15.3 ticket 13: declared fixture numbers are pinned (tools/test_f15_check.py and the reference tests mirror them)") {
    const auto& grid = TransmissionValidation::spec("transmission-roughness-ior");
    REQUIRE(grid.glass.size() == 12);
    std::set<float> roughness, ior;
    for (const auto& g : grid.glass) {
        roughness.insert(g.roughness);
        ior.insert(g.ior);
        CHECK(g.thickness > 0.0f);
        CHECK(g.transmission == 1.0f);
        CHECK(g.baseColor == glm::vec3(1.0f));
        CHECK(std::isinf(g.attenuationDistance)); // no absorption: a pure refracting volume
    }
    CHECK(roughness == std::set<float>{0.0f, 0.1f, 0.3f, 0.6f});
    CHECK(ior == std::set<float>{1.33f, 1.5f, 2.0f});
    const auto& atten = TransmissionValidation::spec("transmission-attenuation");
    CHECK(atten.glass.size() == 6);
    std::set<float> distances;
    for (const auto& g : atten.glass) distances.insert(g.attenuationDistance);
    CHECK(distances == std::set<float>{0.6f, 1.2f, 3.0f});
    // Factor x tint: factors 1 and 0.5 on panes (thin) and slabs (volume), a tinted base on the panes only.
    const auto& factor = TransmissionValidation::spec("transmission-factor-tint");
    CHECK(factor.glass.size() == 8);
    int tinted = 0, half = 0;
    for (const auto& g : factor.glass) {
        if (g.shape == GlassObject::Shape::Pane) {
            CHECK(g.thickness == 0.0f); // thin-walled, one crossing
            tinted += g.baseColor != glm::vec3(1.0f);
        } else {
            CHECK(g.shape == GlassObject::Shape::Cube);
            CHECK(g.thickness > 0.0f);
            CHECK(g.baseColor == glm::vec3(1.0f)); // the tint of a two-crossing volume is ambiguous: not gated
        }
        half += g.transmission == 0.5f;
    }
    CHECK(tinted == 2);
    CHECK(half == 4);
    // Single-sided closed meshes: no double-sided material, a closed volume each.
    for (const auto& g : TransmissionValidation::spec("transmission-single-sided").glass) {
        CHECK_FALSE(g.doubleSided);
        CHECK(g.thickness > 0.0f);
        CHECK(g.shape != GlassObject::Shape::Pane);
    }
    // Thin vs volume: exactly one thin-walled sphere.
    int thin = 0;
    for (const auto& g : TransmissionValidation::spec("transmission-thin-volume").glass) thin += g.thickness == 0.0f;
    CHECK(thin == 1);
    // Glass behind glass: some pair of objects overlaps on the screen with one behind the other.
    const auto& gbg = TransmissionValidation::spec("transmission-glass-behind-glass");
    bool overlap = false;
    for (size_t a = 0; a < gbg.glass.size(); ++a)
        for (size_t b = 0; b < gbg.glass.size(); ++b)
            if (a != b && gbg.glass[a].center.z > gbg.glass[b].center.z + 0.5f &&
                std::abs(gbg.glass[a].center.x - gbg.glass[b].center.x) < 1.0f && std::abs(gbg.glass[a].center.y - gbg.glass[b].center.y) < 1.0f)
                overlap = true;
    CHECK(overlap);
    // Emissive panels: smaller than their pane so both the panel and the environment are seen through the glass.
    const auto& panel = TransmissionValidation::spec("transmission-emissive-panel");
    REQUIRE(panel.panels.size() == panel.glass.size());
    for (size_t k = 0; k < panel.glass.size(); ++k) {
        CHECK(panel.glass[k].thickness == 0.0f);
        // On the screen: the panel (half width .z, farther from the camera) is narrower than the pane's half width.
        const float paneDepth = panel.cameraDistance - panel.glass[k].center.z, panelDepth = panel.cameraDistance - TransmissionValidation::PanelZ;
        CHECK(panel.panels[k].z * paneDepth / panelDepth < 0.5f * panel.glass[k].scale.x);
        CHECK(panel.panels[k].x == panel.glass[k].center.x);
    }
    CHECK(panel.emission == 3.0f);
    // --rt off: no backdrop, rows factor 1 and 0.5.
    const auto& off = TransmissionValidation::spec("transmission-rtoff");
    CHECK(off.backdrop == TransmissionScenarioSpec::Backdrop::None);
    CHECK(off.glass.size() == 8);
}

TEST_CASE("F15.3 ticket 13: the camera frames every glass object and the backdrop covers what is seen through it") {
    for (auto name : TransmissionValidation::scenarios()) {
        const auto& s = TransmissionValidation::spec(name);
        TransmissionValidation f{std::string(name)};
        const auto camera = f.getDefaultCamera();
        CHECK(camera.position == glm::vec3(s.cameraTarget.x, s.cameraTarget.y, s.cameraTarget.z + s.cameraDistance));
        CHECK_FALSE(camera.orbit);
        const float halfV = std::tan(glm::radians(30.0f)), halfH = halfV * 16.0f / 9.0f;
        for (const auto& g : s.glass) {
            const float depth = camera.position.z - g.center.z;
            const float extent = 0.5f * std::max({g.scale.x, g.scale.y, g.scale.z}) * 1.8f; // cube/pane diagonal bound
            const float radius = g.shape == GlassObject::Shape::Sphere ? TransmissionValidation::SphereRadius : extent;
            CHECK(std::abs(g.center.x - camera.target.x) + radius < halfH * depth);
            CHECK(std::abs(g.center.y - camera.target.y) + radius < halfV * depth);
        }
        // The checkerboard spans the screen region behind the glass at its plane.
        if (s.backdrop == TransmissionScenarioSpec::Backdrop::Checker) {
            const float x0 = TransmissionValidation::CheckerOriginX, x1 = x0 + TransmissionValidation::CheckerColumns * TransmissionValidation::CellSize;
            const float y0 = TransmissionValidation::CheckerOriginY, y1 = y0 + TransmissionValidation::CheckerRows * TransmissionValidation::CellSize;
            for (const auto& g : s.glass) {
                CHECK(g.center.x > x0);
                CHECK(g.center.x < x1);
                CHECK(g.center.y - 0.5f > y0);
                CHECK(g.center.y + 0.5f < y1);
            }
        }
    }
}
