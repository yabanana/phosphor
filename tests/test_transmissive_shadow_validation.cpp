#include <doctest/doctest.h>
#include "testbench/material_validation.h"
#include "testbench/transmission_validation.h"
#include "testbench/transmissive_shadow_validation.h"
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
    const std::regex key(R"re(\n    "(transmissive-shadow-[a-z0-9-]+(@[a-z]+)?)": \{)re");
    for (std::sregex_iterator it(block.begin(), block.end(), key), last; it != last; ++it) keys.push_back((*it)[1]);
    return keys;
}
constexpr float kHalfV = 0.57735027f; // tan(30 degrees): the 60 degree vertical field of view
} // namespace

TEST_CASE("F15.3 tickets 14/15: the transmissive-shadow scenarios are exactly the five declared names") {
    const auto names = TransmissiveShadowValidation::scenarios();
    CHECK(names.size() == 5);
    for (auto name : {"transmissive-shadow-pane-sun", "transmissive-shadow-water-sun", "transmissive-shadow-volume-pane",
                      "transmissive-shadow-point", "transmissive-shadow-curved"}) {
        CHECK(TransmissiveShadowValidation::validScenario(name));
        CHECK_FALSE(MaterialValidation::validScenario(name));      // separate bench classes
        CHECK_FALSE(TransmissionValidation::validScenario(name));
        CHECK(MaterialValidation::environmentRadiance(name) == 0.0f); // the environment is 0: only the light reaches the floor
    }
    CHECK_FALSE(TransmissiveShadowValidation::validScenario("transmission-attenuation"));
    CHECK_FALSE(TransmissiveShadowValidation::validScenario("base-grid-sun"));
    CHECK_THROWS(TransmissiveShadowValidation{"unknown"});
}

TEST_CASE("F15.3 tickets 14/15: the bench and the runner (tools/f15_check.py) list the same scenarios; @rt and @csm exist per sun key") {
    const auto keys = runnerKeys();
    REQUIRE_FALSE(keys.empty());
    std::set<std::string> all(keys.begin(), keys.end()), runner;
    for (const auto& key : keys) runner.insert(key.substr(0, key.find('@')));
    std::set<std::string> bench;
    for (auto name : TransmissiveShadowValidation::scenarios()) bench.insert(std::string(name));
    CHECK(runner == bench);
    for (auto name : {"transmissive-shadow-pane-sun", "transmissive-shadow-water-sun", "transmissive-shadow-volume-pane"}) {
        CHECK(all.count(std::string(name) + "@rt") == 1);
        CHECK(all.count(std::string(name) + "@csm") == 1);
    }
    CHECK(all.count("transmissive-shadow-point@restir") == 1);
}

TEST_CASE("F15.3 tickets 14/15: casters carry exact factors; the floor is an opaque diffuse receiver; one light per scenario") {
    for (auto name : TransmissiveShadowValidation::scenarios()) {
        ECS ecs;
        GpuScene scene;
        test::NullTextureManager textures;
        TransmissiveShadowValidation f{std::string(name)};
        f.setup(ecs, scene, textures);
        const auto& s = TransmissiveShadowValidation::spec(name);
        REQUIRE(f.casters().size() == s.casters.size());
        CHECK_FALSE(s.casters.empty());
        // The floor: a large opaque diffuse plane, no extension, nothing emissive.
        const auto& floorMaterial = std::as_const(ecs).getComponent<MaterialComponent>(f.floor());
        CHECK_FALSE(floorMaterial.extension);
        CHECK(floorMaterial.baseColorFactor == glm::vec4(s.floorBase, 1.0f));
        CHECK(floorMaterial.metallicFactor == 0.0f);
        CHECK(floorMaterial.roughnessFactor == s.floorRoughness);
        CHECK(floorMaterial.emissiveFactor == glm::vec3(0.0f));
        const auto& floorTransform = std::as_const(ecs).getComponent<TransformComponent>(f.floor());
        CHECK(floorTransform.position == glm::vec3(0.0f));
        CHECK(floorTransform.scale.x >= 5.0f); // the reference tool recognises the receiver as the large plane
        CHECK(floorTransform.scale.z >= 5.0f);
        // Exactly one light: the sun or one point light.
        CHECK((s.sun ? 1 : 0) + int(s.pointLights.size()) == 1);
        CHECK((f.sun() != INVALID_ENTITY) == s.sun);
        CHECK(f.pointLights().size() == s.pointLights.size());
        for (size_t k = 0; k < s.casters.size(); ++k) {
            const GlassObject& g = s.casters[k].glass;
            const auto& m = std::as_const(ecs).getComponent<MaterialComponent>(f.casters()[k]);
            const auto& instance = std::as_const(ecs).getComponent<MeshInstanceComponent>(f.casters()[k]);
            CHECK(instance.castsShadows());
            CHECK(instance.isVisible());
            CHECK(m.baseColorTexIndex == INVALID_TEXTURE_INDEX);
            CHECK(m.normalTexIndex == INVALID_TEXTURE_INDEX);
            CHECK(m.metallicRoughnessTexIndex == INVALID_TEXTURE_INDEX);
            CHECK(m.emissiveFactor == glm::vec3(0.0f));
            if (s.casters[k].opaque) {
                CHECK_FALSE(m.extension); // today's engine opaque shadow: the comparison caster stays opaque after tickets 14/15
                continue;
            }
            REQUIRE(m.extension);
            REQUIRE(m.extension->transmission);
            REQUIRE(m.extension->volume);
            REQUIRE(m.extension->specular);
            CHECK(m.extension->transmission->factor == 1.0f);
            CHECK(m.extension->specular->ior == g.ior);
            CHECK(m.extension->volume->thickness == g.thickness);
            CHECK(m.extension->volume->attenuationDistance == g.attenuationDistance);
            for (int c = 0; c < 3; ++c) CHECK(m.extension->volume->attenuationColor[c] == g.attenuationColor[c]);
            const u32 mask = featureMask(*m.extension);
            CHECK((mask & featureBit(MaterialFeature::Transmission)) != 0);
            CHECK(((mask & featureBit(MaterialFeature::Volume)) != 0) == (g.thickness > 0.0f));
            CHECK(m.baseColorFactor == glm::vec4(g.baseColor, 1.0f));
            CHECK(m.metallicFactor == 0.0f);
            CHECK(m.roughnessFactor == 0.0f);
            CHECK(m.doubleSided == g.doubleSided);
            for (int c = 0; c < 3; ++c) {
                CHECK(g.attenuationColor[c] > 0.0f);
                CHECK(g.attenuationColor[c] <= 1.0f);
            }
        }
        f.teardown(ecs, scene);
        CHECK(f.casters().empty());
        CHECK(f.floor() == INVALID_ENTITY);
    }
}

TEST_CASE("F15.3 tickets 14/15: the planar fixtures sit inside the declared gating regime (sun within 30 degrees of the normal)") {
    for (auto name : TransmissiveShadowValidation::scenarios()) {
        const auto& s = TransmissiveShadowValidation::spec(name);
        if (s.sun) {
            CHECK(std::abs(glm::length(s.towardSun) - 1.0f) < 1e-5f);
            // The casters' normal is +Y: the sun is within 30 degrees of it (the curved key is report only but keeps the rule).
            CHECK(s.towardSun.y >= std::cos(glm::radians(30.0f)) - 1e-5f);
            CHECK(s.towardSun.z < 0.0f); // behind the casters seen from the camera: the shadows fall towards the camera
            CHECK(s.sunIrradiance == 3.0f);
        }
        for (const auto& light : s.pointLights) {
            // The light-to-receiver ray through the caster centre is within 30 degrees of the vertical (the caster normal); the
            // outermost footprint edge is at most 31 degrees (the gated region excludes the envelope band near that edge).
            for (const auto& c : s.casters) {
                const float lateral = std::abs(c.glass.center.x - light.position.x);
                const float magnification = light.position.y / (light.position.y - c.glass.center.y);
                CHECK(std::atan2(lateral * magnification, light.position.y) < glm::radians(30.0f));
                CHECK(std::atan2((lateral + 0.5f * c.glass.scale.x) * magnification, light.position.y) < glm::radians(31.0f));
            }
            CHECK(light.intensity == 60.0f);
            CHECK(light.range == 100.0f);
        }
    }
}

TEST_CASE("F15.3 tickets 14/15: casters are planar slabs/panes except the report-only curved key; opaque comparison where declared") {
    for (auto name : TransmissiveShadowValidation::scenarios()) {
        const auto& s = TransmissiveShadowValidation::spec(name);
        const bool curved = name == std::string_view("transmissive-shadow-curved");
        int opaque = 0;
        for (const auto& c : s.casters) {
            opaque += c.opaque;
            if (curved) {
                CHECK(c.glass.shape == GlassObject::Shape::Sphere);
                CHECK(c.glass.thickness > 0.0f);
            } else {
                CHECK(c.glass.shape != GlassObject::Shape::Sphere);
                // Horizontal: the shapes are unrotated, their normal is +Y.
                CHECK(c.glass.yawDegrees == 0.0f);
                CHECK(c.glass.pitchDegrees == 0.0f);
            }
            if (c.glass.shape == GlassObject::Shape::Pane) {
                CHECK(c.glass.thickness == 0.0f); // thin-walled: ONE Fresnel surface
                CHECK(c.glass.doubleSided);
            }
            if (c.glass.shape == GlassObject::Shape::Cube) {
                CHECK(c.glass.scale.y <= 0.12f);  // a slab: thin along the vertical
                CHECK_FALSE(c.glass.doubleSided); // single-sided closed mesh: the shadow ray leaves through a back face
            }
        }
        CHECK(opaque == (curved || std::string_view(name) == "transmissive-shadow-volume-pane" ? 0 : 1));
    }
}

TEST_CASE("F15.3 tickets 14/15: declared fixture numbers are pinned (tools/f15_shadow_fixtures.py and the tests mirror them)") {
    const auto& pane = TransmissiveShadowValidation::spec("transmissive-shadow-pane-sun");
    REQUIRE(pane.casters.size() == 4);
    CHECK(pane.casters[0].glass.baseColor == glm::vec3(1.0f));
    CHECK(pane.casters[1].glass.baseColor == glm::vec3(0.4f, 0.9f, 0.45f)); // the green tint
    CHECK(pane.casters[2].glass.thickness > 0.0f);                          // thick clear: two Fresnel surfaces, no absorption
    CHECK(std::isinf(pane.casters[2].glass.attenuationDistance));
    CHECK(pane.casters[3].opaque);
    CHECK(pane.casters[0].glass.center.x == -2.1f);
    CHECK(pane.casters[3].glass.center.x == 2.1f);
    CHECK(pane.casters[0].glass.center.y == TransmissiveShadowValidation::CasterHeight);
    CHECK(pane.casters[0].glass.scale == glm::vec3(0.9f, 1.0f, 0.9f));
    CHECK(pane.casters[2].glass.scale == glm::vec3(0.9f, 0.06f, 0.9f));
    CHECK(pane.towardSun.y == doctest::Approx(std::cos(glm::radians(25.0f))));
    const auto& water = TransmissiveShadowValidation::spec("transmissive-shadow-water-sun");
    REQUIRE(water.casters.size() == 3);
    for (int k = 0; k < 2; ++k) CHECK(water.casters[k].glass.ior == 1.33f);
    CHECK(std::isinf(water.casters[0].glass.attenuationDistance)); // clear water
    CHECK(water.casters[1].glass.attenuationDistance == 0.3f);     // murky water
    CHECK(water.casters[1].glass.attenuationColor == glm::vec3(0.45f, 0.75f, 0.9f));
    CHECK(water.casters[2].opaque);
    CHECK(water.towardSun.y == doctest::Approx(std::cos(glm::radians(20.0f))));
    const auto& volume = TransmissiveShadowValidation::spec("transmissive-shadow-volume-pane");
    REQUIRE(volume.casters.size() == 3);
    std::set<float> distances;
    for (const auto& c : volume.casters) {
        distances.insert(c.glass.attenuationDistance);
        CHECK(c.glass.scale.y == 0.1f);
        CHECK(c.glass.ior == 1.5f);
    }
    CHECK(distances == std::set<float>{0.2f, 0.15f});
    const auto& point = TransmissiveShadowValidation::spec("transmissive-shadow-point");
    REQUIRE(point.casters.size() == 4);
    REQUIRE(point.pointLights.size() == 1);
    CHECK(point.pointLights[0].position == glm::vec3(0.0f, 4.5f, -0.3f));
    for (const auto& c : point.casters) CHECK(c.glass.scale.x == 0.7f);
    CHECK_FALSE(point.sun);
    const auto& curved = TransmissiveShadowValidation::spec("transmissive-shadow-curved");
    REQUIRE(curved.casters.size() == 2);
    CHECK(curved.casters[1].glass.attenuationDistance == 0.6f);
}

TEST_CASE("F15.3 tickets 14/15: the camera sees the floor shadows and never the floor through a caster") {
    for (auto name : TransmissiveShadowValidation::scenarios()) {
        const auto& s = TransmissiveShadowValidation::spec(name);
        TransmissiveShadowValidation f{std::string(name)};
        const auto camera = f.getDefaultCamera();
        CHECK_FALSE(camera.orbit);
        CHECK(glm::length(camera.position - camera.target) == doctest::Approx(s.cameraDistance));
        CHECK(camera.position.y > camera.target.y);
        const glm::vec3 forward = glm::normalize(camera.target - camera.position);
        const glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0, 1, 0)));
        const glm::vec3 up = glm::cross(right, forward);
        auto onScreen = [&](glm::vec3 p) {
            const glm::vec3 d = p - camera.position;
            const float depth = glm::dot(d, forward);
            const float x = glm::dot(d, right) / (depth * kHalfV * 16.0f / 9.0f), y = glm::dot(d, up) / (depth * kHalfV);
            return depth > 0.0f && std::abs(x) < 0.97f && std::abs(y) < 0.97f;
        };
        for (const auto& c : s.casters) {
            const glm::vec3 centre = c.glass.center;
            glm::vec3 shadowCentre; // where the straight shadow ray from the caster centre meets the floor
            if (s.sun) shadowCentre = centre - s.towardSun * (centre.y / s.towardSun.y);
            else {
                const glm::vec3 light = s.pointLights[0].position, direction = centre - light;
                shadowCentre = light + direction * (light.y / -direction.y);
            }
            CHECK(onScreen(shadowCentre));
            CHECK(onScreen(centre));
            // The view ray from the camera to the shadow centre must cross the caster's height outside the caster's footprint.
            const float along = (camera.position.y - centre.y) / (camera.position.y - shadowCentre.y);
            const glm::vec3 crossing = camera.position + along * (shadowCentre - camera.position);
            const float half = c.glass.shape == GlassObject::Shape::Sphere ? 0.4f : 0.5f * c.glass.scale.x;
            CHECK((std::abs(crossing.x - centre.x) > half + 0.1f || std::abs(crossing.z - centre.z) > half + 0.1f));
        }
    }
}

TEST_CASE("F15.1 ticket 14 round 3: the indirect-light fixtures (tools/f15_gi_glass_shadow_check.py) are valid, outside the planar list, and geometrically closed") {
    const auto names = TransmissiveShadowValidation::indirectScenarios();
    REQUIRE(names.size() == 6);
    std::string runner;
    {
        std::ifstream in(std::string(PHOSPHOR_SOURCE_DIR) + "/tools/f15_gi_glass_shadow_check.py");
        std::stringstream text;
        text << in.rdbuf();
        runner = text.str();
    }
    REQUIRE_FALSE(runner.empty());
    for (auto name : names) {
        CHECK(TransmissiveShadowValidation::validScenario(name));
        for (auto planar : TransmissiveShadowValidation::scenarios()) CHECK(name != planar);
        const auto& s = TransmissiveShadowValidation::spec(name);
        CHECK(s.sun);
        CHECK(s.towardSun.y == doctest::Approx(std::cos(glm::radians(30.0f))));
        const bool pane = std::string_view(name).ends_with("-pane");
        if (std::string_view(name).find("-sss-") != std::string_view::npos) {
            // The subsurface receiver (rt-walk): sun from +Z (the lit side faces the camera), the pane mirrored to -Z, a volumetric sphere.
            CHECK(s.towardSun.z > 0.0f);
            REQUIRE(s.casters.size() == (pane ? 2u : 1u));
            CHECK(s.casters.back().sss);
            CHECK(s.casters.back().opaque);
            if (pane) {
                const auto& g = s.casters[0].glass;
                const glm::vec3 crossing = glm::vec3(0, 0.4f, 0) + s.towardSun * ((g.center.y - 0.4f) / s.towardSun.y);
                CHECK(std::abs(crossing.z - g.center.z) < 1.0f);
                CHECK(g.center.y < 90.0f);
            }
            CHECK(runner.find("\"sss\": ((0.0, 0.4, 0.0), 2.0, 25.0)") != std::string::npos);
            CHECK(s.cameraTarget == glm::vec3(0.0f, 0.4f, 0.0f));
            CHECK(s.cameraDistance == 2.0f);
            CHECK(s.cameraElevationDegrees == 25.0f);
            continue;
        }
        const bool refl = std::string_view(name).find("-refl-") != std::string_view::npos;
        REQUIRE(s.casters.size() == (pane ? 2u : 1u));
        if (pane) {
            const auto& g = s.casters[0].glass;
            CHECK(g.shape == GlassObject::Shape::Pane);
            CHECK(g.thickness == 0.0f);
            CHECK(g.baseColor == glm::vec3(0.4f, 0.9f, 0.45f));
            // Inside the 100 m ray range of the sun shadow ray of a secondary hit, above the 8 m of the probe grid, and every sun ray from
            // the scene (|x|, |z| <= 8 m, y <= 2 m) crosses it: the crossing point lies inside the quad.
            CHECK(g.center.y > 10.0f);
            CHECK(g.center.y < 90.0f);
            for (const glm::vec3 p : {glm::vec3(-8, 0, -8), glm::vec3(8, 0, 8), glm::vec3(-8, 2, 8), glm::vec3(8, 2, -8)}) {
                const glm::vec3 crossing = p + s.towardSun * ((g.center.y - p.y) / s.towardSun.y);
                CHECK(std::abs(crossing.x - g.center.x) < 0.5f * g.scale.x);
                CHECK(std::abs(crossing.z - g.center.z) < 0.5f * g.scale.z);
                CHECK(glm::length(crossing - p) < 100.0f);
            }
        }
        const auto& receiver = s.casters.back();
        CHECK(receiver.opaque);
        CHECK(receiver.mirror == refl);
        // The runner mirrors the camera of the spec.
        const auto camera = TransmissiveShadowValidation(std::string(name)).getDefaultCamera();
        CHECK(camera.target == s.cameraTarget);
        CHECK(runner.find(refl ? "\"refl\": ((0.0, 0.5, 1.0), 5.0, 25.0)" : "\"gi\": ((0.0, 0.4, 0.3), 4.0, 30.0)") != std::string::npos);
        CHECK(s.cameraTarget == (refl ? glm::vec3(0.0f, 0.5f, 1.0f) : glm::vec3(0.0f, 0.4f, 0.3f)));
        CHECK(s.cameraDistance == (refl ? 5.0f : 4.0f));
        CHECK(s.cameraElevationDegrees == (refl ? 25.0f : 30.0f));
    }
}
