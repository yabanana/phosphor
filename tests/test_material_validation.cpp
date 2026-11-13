#include <doctest/doctest.h>
#include "testbench/material_validation.h"
#include "renderer/gpu_scene.h"
#include "renderer/scene_store.h"
#include "renderer/scene_extract.h"
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

TEST_CASE("F15.3 material scenarios are exactly the nine declared names") {
    CHECK(MaterialValidation::scenarios().size() == 9);
    for (auto name : {"base-grid-sun", "base-grid-env", "base-grid-env-resolve", "furnace-base", "furnace-resolve",
                      "roughness-extremes", "roughness-extremes-sun", "specular-grid-sun", "specular-grid-env-resolve"})
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

TEST_CASE("F15.3 the deterministic furnace declares the same scene as the transport furnace") {
    const auto& transport = MaterialValidation::spec("furnace-base");
    const auto& resolve = MaterialValidation::spec("furnace-resolve");
    CHECK(resolve.roughness == transport.roughness);
    CHECK(resolve.metallic == transport.metallic);
    CHECK(resolve.baseColor == transport.baseColor);
    CHECK_FALSE(resolve.sun);
    CHECK(MaterialValidation::environmentRadiance("furnace-resolve") == 1.0f);
}

TEST_CASE("F15.3 the new deterministic scenarios declare the scenes the review asked for") {
    // base-grid-env-resolve: the base-grid-env scene, resolved deterministically (no transport).
    const auto& env = MaterialValidation::spec("base-grid-env");
    const auto& envResolve = MaterialValidation::spec("base-grid-env-resolve");
    CHECK(envResolve.roughness == env.roughness);
    CHECK(envResolve.metallic == env.metallic);
    CHECK(envResolve.baseColor == env.baseColor);
    CHECK_FALSE(envResolve.sun);
    CHECK(envResolve.environment == 0.5f);
    CHECK(MaterialValidation::environmentRadiance("base-grid-env-resolve") == 0.5f);
    // roughness-extremes-sun: the extremes lit by the sun only.
    const auto& extremes = MaterialValidation::spec("roughness-extremes");
    const auto& sunOnly = MaterialValidation::spec("roughness-extremes-sun");
    CHECK(sunOnly.roughness == extremes.roughness);
    CHECK(sunOnly.metallic == extremes.metallic);
    CHECK(sunOnly.baseColor == extremes.baseColor);
    CHECK(sunOnly.sun);
    CHECK(sunOnly.sunIrradiance == extremes.sunIrradiance);
    CHECK(sunOnly.towardSun == extremes.towardSun);
    CHECK(sunOnly.environment == 0.0f);
    CHECK(MaterialValidation::environmentRadiance("roughness-extremes-sun") == 0.0f);
    CHECK(MaterialValidation{"roughness-extremes-sun"}.getDefaultCamera().position == glm::vec3(0.0f, 1.0f, 4.0f));
}

TEST_CASE("F15.3 the bench and the runner (tools/f15_check.py) list the same scenarios") {
    std::ifstream in(std::string(PHOSPHOR_SOURCE_DIR) + "/tools/f15_check.py");
    REQUIRE(in.good());
    std::stringstream text;
    text << in.rdbuf();
    const std::string source = text.str();
    const size_t start = source.find("SCENARIOS = {");
    REQUIRE(start != std::string::npos);
    const size_t end = source.find("\nGPU_PROCESS", start);
    REQUIRE(end != std::string::npos);
    const std::string block = source.substr(start, end - start);
    std::set<std::string> runner;
    const std::regex key(R"re(\n    "([a-z0-9-]+)": \{)re");
    for (std::sregex_iterator it(block.begin(), block.end(), key), last; it != last; ++it) runner.insert((*it)[1]);
    std::set<std::string> bench;
    for (auto name : MaterialValidation::scenarios()) bench.insert(std::string(name));
    CHECK(runner == bench);
}

TEST_CASE("F15.1 specular grids declare ior x specularFactor x specularColor, metallic 0 rows and one metallic 1 row") {
    for (auto name : {"specular-grid-sun", "specular-grid-env-resolve"}) {
        const auto& s = MaterialValidation::spec(name);
        REQUIRE(s.hasSpecularGrid());
        REQUIRE(s.ior.size() == s.roughness.size());
        REQUIRE(s.specularFactor.size() == s.metallic.size());
        REQUIRE(s.specularColor.size() == s.metallic.size());
        // ior = 0 (special mode) and 1.0, 1.33, 1.5, 2.4 are all declared.
        for (float ior : {0.0f, 1.0f, 1.33f, 1.5f, 2.4f}) CHECK(std::count(s.ior.begin(), s.ior.end(), ior) == 1);
        // Weight and colour vary; the colour reaches 2.0; both are valid KHR ranges.
        float maxColor = 0.0f;
        for (size_t r = 0; r < s.metallic.size(); ++r) {
            CHECK(s.specularFactor[r] >= 0.0f);
            CHECK(s.specularFactor[r] <= 1.0f);
            for (int c = 0; c < 3; ++c) maxColor = std::max(maxColor, s.specularColor[r][c]);
        }
        CHECK(maxColor == 2.0f);
        CHECK(std::count(s.specularFactor.begin(), s.specularFactor.end(), 1.0f) >= 1);
        CHECK(std::count(s.specularFactor.begin(), s.specularFactor.end(), 0.5f) >= 1);
        // Metallic 0 rows plus exactly one metallic 1 row (on which the extension must change nothing).
        CHECK(std::count(s.metallic.begin(), s.metallic.end(), 1.0f) == 1);
        CHECK(std::count(s.metallic.begin(), s.metallic.end(), 0.0f) == long(s.metallic.size()) - 1);
    }
    const auto& sun = MaterialValidation::spec("specular-grid-sun");
    CHECK(sun.sun);
    CHECK(sun.environment == 0.0f);
    CHECK(sun.sunIrradiance == 3.0f);
    const auto& env = MaterialValidation::spec("specular-grid-env-resolve");
    CHECK_FALSE(env.sun);
    CHECK(env.environment == 0.5f);
    CHECK(MaterialValidation::environmentRadiance("specular-grid-env-resolve") == 0.5f);
    CHECK(MaterialValidation::environmentRadiance("specular-grid-sun") == 0.0f);
    // The base scenarios carry no extension.
    for (auto name : {"base-grid-sun", "furnace-base", "roughness-extremes"})
        CHECK_FALSE(MaterialValidation::spec(name).hasSpecularGrid());
}

TEST_CASE("F15.1 specular grid spheres carry the extension of their row and column, nothing else changes") {
    ECS ecs;
    GpuScene scene;
    test::NullTextureManager textures;
    MaterialValidation f{"specular-grid-sun"};
    f.setup(ecs, scene, textures);
    const auto& s = MaterialValidation::spec("specular-grid-sun");
    for (u32 row = 0; row < s.metallic.size(); ++row)
        for (u32 col = 0; col < s.roughness.size(); ++col) {
            const EntityID e = f.spheres()[row * s.roughness.size() + col];
            const auto& m = std::as_const(ecs).getComponent<MaterialComponent>(e);
            REQUIRE(m.extension);
            REQUIRE(m.extension->specular);
            CHECK(m.extension->specular->ior == s.ior[col]);
            CHECK(m.extension->specular->specularFactor == s.specularFactor[row]);
            CHECK(m.extension->specular->specularColor[0] == s.specularColor[row].x);
            CHECK(m.extension->specular->specularColor[2] == s.specularColor[row].z);
            CHECK(m.baseColorTexIndex == INVALID_TEXTURE_INDEX);
            CHECK(m.metallicFactor == s.metallic[row]);
            CHECK(m.roughnessFactor == s.roughness[col]);
            // Neutral values (ior 1.5, weight 1, colour 1) are the base material: the bit stays clear.
            const bool neutral = s.ior[col] == 1.5f && s.specularFactor[row] == 1.0f && s.specularColor[row] == glm::vec3(1.0f);
            CHECK((featureMask(*m.extension) != 0) == !neutral);
        }
    f.teardown(ecs, scene);
}

TEST_CASE("F15.1 --material-toggle-every edits one sphere's weight and flips another's feature bit without reallocating") {
    CHECK_THROWS(MaterialValidation{"base-grid-sun", 4});
    ECS ecs;
    GpuScene scene;
    test::NullTextureManager textures;
    MaterialValidation f{"specular-grid-sun", 4};
    f.setup(ecs, scene, textures);
    SceneStore store;
    store.sync(ecs, scene);
    ecs.endFrame();
    const u32 groups = store.extensionGroups(), capacity = store.extensionGroupCapacity();
    const u64 structure = store.structureVersion();
    const EntityID featureSphere = f.spheres()[MaterialValidation::toggleFeatureSphere()];
    const EntityID parameterSphere = f.spheres()[MaterialValidation::toggleParameterSphere()];
    CHECK((store.materials()[store.materialIndexOf(featureSphere)].flags >> MATERIAL_FEATURE_SHIFT) == 0); // neutral
    CHECK(store.materials()[store.materialIndexOf(featureSphere)].extensionIndex == MATERIAL_EXTENSION_INVALID);
    const u32 factorWord = 1; // GPUMaterialExt0::specularFactor
    const float baseWeight = [&] {
        float v;
        const auto& rec = store.extensions()[store.materials()[store.materialIndexOf(parameterSphere)].extensionIndex];
        std::memcpy(&v, &rec.w[factorWord], 4);
        return v;
    }();
    CHECK(baseWeight == 0.5f);

    for (u32 frame = 1; frame <= 3; ++frame) { // updates 1..3: nothing yet
        f.update(0.0f, ecs);
        store.sync(ecs, scene);
        CHECK_FALSE(store.materialsChanged());
        CHECK(store.verifyAgainstEcs(ecs, scene).empty());
        ecs.endFrame();
    }
    f.update(0.0f, ecs); // event 1: feature on, weight 0.6
    store.sync(ecs, scene);
    CHECK(store.materialsChanged());
    CHECK((store.materials()[store.materialIndexOf(featureSphere)].flags >> MATERIAL_FEATURE_SHIFT) ==
          MATERIAL_FEATURE_SPECULAR_IOR);
    const u32 group = store.materials()[store.materialIndexOf(featureSphere)].extensionIndex;
    CHECK(group != MATERIAL_EXTENSION_INVALID);
    CHECK(store.extensionGroupCapacity() == capacity); // a free group of the slack: no reallocation
    CHECK(store.extensionGroups() == groups + 1);
    CHECK(store.structureVersion() == structure);
    CHECK(store.verifyAgainstEcs(ecs, scene).empty());
    ecs.endFrame();
    for (u32 i = 0; i < 4; ++i) { // event 2: the feature goes off again, the group stays
        f.update(0.0f, ecs);
        store.sync(ecs, scene);
        ecs.endFrame();
    }
    CHECK((store.materials()[store.materialIndexOf(featureSphere)].flags >> MATERIAL_FEATURE_SHIFT) == 0);
    CHECK(store.materials()[store.materialIndexOf(featureSphere)].extensionIndex == group);
    CHECK(store.structureVersion() == structure);
    CHECK(store.verifyAgainstEcs(ecs, scene).empty());
    CHECK(store.featureUnion() == MATERIAL_FEATURE_SPECULAR_IOR);
}
