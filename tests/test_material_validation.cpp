#include <doctest/doctest.h>
#include "testbench/material_validation.h"
#include "renderer/gpu_scene.h"
#include "renderer/scene_extract.h"
#include "scene/components.h"
#include "scene/ecs.h"
#include "null_texture_manager.h"
#include <cmath>
#include <utility>
using namespace phosphor;

TEST_CASE("F15.3 material scenarios are exactly the four declared names") {
    CHECK(MaterialValidation::scenarios().size() == 4);
    for (auto name : {"base-grid-sun", "base-grid-env", "furnace-base", "roughness-extremes"})
        CHECK(MaterialValidation::validScenario(name));
    CHECK_FALSE(MaterialValidation::validScenario("unknown"));
    CHECK_THROWS(MaterialValidation{"unknown"});
}

TEST_CASE("F15.3 declared light rigs: sun-only, environment-only, furnace, extremes") {
    const auto& sun = MaterialValidation::spec("base-grid-sun");
    CHECK(sun.sun);
    CHECK(sun.environment == 0.0f);
    const auto& env = MaterialValidation::spec("base-grid-env");
    CHECK_FALSE(env.sun);
    CHECK(env.environment > 0.0f);
    const auto& furnace = MaterialValidation::spec("furnace-base");
    CHECK_FALSE(furnace.sun);
    CHECK(furnace.environment == 1.0f); // the furnace radiance is exactly 1
    CHECK(furnace.baseColor == glm::vec3(1.0f));
    const auto& extremes = MaterialValidation::spec("roughness-extremes");
    CHECK(extremes.sun);
    CHECK(extremes.environment > 0.0f);
    CHECK(extremes.roughness == std::vector<float>{0.0f, 1.0f});
    CHECK(extremes.metallic == std::vector<float>{0.0f, 0.5f, 1.0f});
    CHECK(MaterialValidation::environmentRadiance("furnace-base") == 1.0f);
    CHECK(MaterialValidation::environmentRadiance("") == 0.0f);
    CHECK(glm::length(sun.towardSun) == doctest::Approx(1.0f));
}

TEST_CASE("F15.3 grid spheres carry exact factors and no textures") {
    for (auto name : MaterialValidation::scenarios()) {
        ECS ecs;
        GpuScene scene;
        test::NullTextureManager textures;
        MaterialValidation f{std::string(name)};
        f.setup(ecs, scene, textures);
        const auto& s = MaterialValidation::spec(name);
        REQUIRE(f.spheres().size() == s.roughness.size() * s.metallic.size());
        CHECK(scene.getMeshCount() == 1);
        for (u32 row = 0; row < s.metallic.size(); ++row)
            for (u32 col = 0; col < s.roughness.size(); ++col) {
                const EntityID e = f.spheres()[row * s.roughness.size() + col];
                const auto& m = std::as_const(ecs).getComponent<MaterialComponent>(e);
                CHECK(m.metallicFactor == s.metallic[row]);
                CHECK(m.roughnessFactor == s.roughness[col]);
                CHECK(m.baseColorFactor == glm::vec4(s.baseColor, 1.0f));
                CHECK(m.baseColorTexIndex == INVALID_TEXTURE_INDEX);
                CHECK(m.normalTexIndex == INVALID_TEXTURE_INDEX);
                CHECK(m.metallicRoughnessTexIndex == INVALID_TEXTURE_INDEX);
                CHECK(m.occlusionTexIndex == INVALID_TEXTURE_INDEX);
                CHECK(m.emissiveFactor == glm::vec3(0.0f));
                const auto& t = std::as_const(ecs).getComponent<TransformComponent>(e);
                const glm::vec3 centre = MaterialValidation::sphereCenter(s, col, row);
                CHECK(t.position == centre);
            }
        FrameScene frame;
        extractFrameScene(ecs, scene, frame);
        CHECK(frame.instances.size() == f.spheres().size());
        CHECK(frame.lights.size() == (s.sun ? 1u : 0u));
        if (s.sun) {
            // Directional light: the engine's toward-sun vector is the negated travel direction.
            const auto& l = frame.lights.front();
            const glm::vec3 travel(l.direction[0], l.direction[1], l.direction[2]);
            CHECK(glm::dot(-glm::normalize(travel), s.towardSun) == doctest::Approx(1.0f).epsilon(1e-5));
            CHECK(l.intensity == s.sunIrradiance);
        }
        f.teardown(ecs, scene);
        CHECK(f.spheres().empty());
    }
}

TEST_CASE("F15.3 spheres are separated and the camera frames the whole grid") {
    for (auto name : MaterialValidation::scenarios()) {
        const auto& s = MaterialValidation::spec(name);
        MaterialValidation f{std::string(name)};
        const auto camera = f.getDefaultCamera();
        CHECK(camera.position.x == 0.0f);
        CHECK_FALSE(camera.orbit);
        const float halfV = std::tan(glm::radians(30.0f)), halfH = halfV * 16.0f / 9.0f;
        const float dist = camera.position.z; // grid plane is z = 0
        for (u32 row = 0; row < s.metallic.size(); ++row)
            for (u32 col = 0; col < s.roughness.size(); ++col) {
                const glm::vec3 c = MaterialValidation::sphereCenter(s, col, row);
                CHECK(std::abs(c.x) + MaterialValidation::SphereRadius < halfH * dist);
                CHECK(std::abs(c.y - camera.position.y) + MaterialValidation::SphereRadius < halfV * dist);
            }
        CHECK(MaterialValidation::Spacing > 2 * MaterialValidation::SphereRadius);
    }
}

TEST_CASE("F15.3 declared scenario numbers are pinned (tools/test_f15_check.py mirrors them)") {
    const auto& sun = MaterialValidation::spec("base-grid-sun");
    CHECK(sun.roughness == std::vector<float>{0.1f, 0.2f, 0.35f, 0.5f, 0.7f, 0.85f, 1.0f});
    CHECK(sun.metallic == std::vector<float>{0.0f, 0.25f, 0.5f, 0.75f, 1.0f});
    CHECK(sun.baseColor == glm::vec3(0.8f, 0.35f, 0.15f));
    CHECK(sun.sunIrradiance == 3.0f);
    const glm::vec3 expected = glm::normalize(glm::vec3(-0.8f, 0.35f, -0.3f));
    CHECK(glm::length(sun.towardSun - expected) < 1e-6f);
    CHECK(MaterialValidation::spec("base-grid-env").environment == 0.5f);
    CHECK(MaterialValidation::spec("roughness-extremes").environment == 0.3f);
    CHECK(MaterialValidation::spec("roughness-extremes").sunIrradiance == 3.0f);
    MaterialValidation fixture{"base-grid-sun"};
    CHECK(fixture.getDefaultCamera().position == glm::vec3(0.0f, 2.0f, 5.5f));
    CHECK(MaterialValidation{"roughness-extremes"}.getDefaultCamera().position == glm::vec3(0.0f, 1.0f, 4.0f));
}
