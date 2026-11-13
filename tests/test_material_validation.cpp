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
    CHECK(MaterialValidation::scenarios().size() == 17);
    for (auto name : {"base-grid-sun", "base-grid-env", "base-grid-env-resolve", "furnace-base", "furnace-resolve",
                      "roughness-extremes", "roughness-extremes-sun", "specular-grid-sun", "specular-grid-env-resolve", "specular-grid-point",
                      "specular-grid-env", "furnace-specular", "clearcoat-grid-sun", "clearcoat-grid-env-resolve",
                      "clearcoat-grid-env", "clearcoat-grid-point", "furnace-clearcoat"})
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
        CHECK(frame.lights.size() == (s.sun ? 1u : 0u) + s.pointLights.size());
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
    // Runner keys are bench scenario names, optionally with an "@lighting" variant suffix.
    const std::regex key(R"re(\n    "([a-z0-9-]+)(@[a-z]+)?": \{)re");
    for (std::sregex_iterator it(block.begin(), block.end(), key), last; it != last; ++it) runner.insert((*it)[1]);
    std::set<std::string> bench;
    for (auto name : MaterialValidation::scenarios()) bench.insert(std::string(name));
    CHECK(runner == bench);
}

TEST_CASE("F15.1 specular grids declare ior x specularFactor x specularColor, metallic 0 rows and one metallic 1 row") {
    for (auto name : {"specular-grid-sun", "specular-grid-env-resolve", "specular-grid-env", "furnace-specular"}) {
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
    // F15.1 (ticket 08): the transport twin of the resolve grid, and the extended furnace (albedo 1, environment 1).
    const auto& transport = MaterialValidation::spec("specular-grid-env");
    CHECK_FALSE(transport.sun);
    CHECK(transport.environment == 0.5f);
    CHECK(transport.baseColor == env.baseColor);
    CHECK(transport.ior == env.ior);
    CHECK(transport.specularFactor == env.specularFactor);
    CHECK(transport.specularColor == env.specularColor);
    const auto& furnace = MaterialValidation::spec("furnace-specular");
    CHECK_FALSE(furnace.sun);
    CHECK(furnace.environment == 1.0f);
    CHECK(furnace.baseColor == glm::vec3(1.0f));
    CHECK(furnace.ior == env.ior);
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

TEST_CASE("F15.1 specular-grid-point: the specular grid lit by one point light from behind the grid") {
    const auto& sun = MaterialValidation::spec("specular-grid-sun");
    const auto& point = MaterialValidation::spec("specular-grid-point");
    CHECK(point.roughness == sun.roughness);
    CHECK(point.metallic == sun.metallic);
    CHECK(point.ior == sun.ior);
    CHECK(point.specularFactor == sun.specularFactor);
    CHECK(point.specularColor == sun.specularColor);
    CHECK_FALSE(point.sun);
    CHECK(point.environment == 0.0f);
    REQUIRE(point.pointLights.size() == 1);
    const MaterialPointLight light = point.pointLights[0];
    CHECK(light.intensity == 250.0f);
    CHECK(light.range == 100.0f);
    const glm::vec3 camera = MaterialValidation{"specular-grid-point"}.getDefaultCamera().position;
    // Side-lit from behind at the grid centre: V.L about -0.5, the light behind the grid plane.
    const glm::vec3 centre(0.0f, 2.0f, 0.0f);
    const float vl = glm::dot(glm::normalize(camera - centre), glm::normalize(light.position - centre));
    CHECK(vl < -0.4f);
    CHECK(vl > -0.6f);
    CHECK(light.position.z < -MaterialValidation::SphereRadius);
    CHECK(glm::length(light.position - centre) == doctest::Approx(9.0f));
    // The engine's range window stays within 5e-4 of 1 for every sphere (the reference declares it exactly).
    for (u32 row = 0; row < point.metallic.size(); ++row)
        for (u32 col = 0; col < point.roughness.size(); ++col) {
            const float d = glm::length(light.position - MaterialValidation::sphereCenter(point, col, row));
            const float r = d / light.range, window = (1.0f - r * r * r * r) * (1.0f - r * r * r * r);
            CHECK(window > 0.9995f);
        }
}

TEST_CASE("F15.3 clearcoat grids declare coat factor x coat roughness, an ior row under a full coat and a metal row") {
    for (auto name : {"clearcoat-grid-sun", "clearcoat-grid-env-resolve", "clearcoat-grid-env", "clearcoat-grid-point",
                      "furnace-clearcoat"}) {
        const auto& s = MaterialValidation::spec(name);
        REQUIRE(s.hasClearcoatGrid());
        CHECK_FALSE(s.hasSpecularGrid());
        REQUIRE(s.clearcoatFactor.size() == s.metallic.size());
        REQUIRE(s.baseIor.size() == s.metallic.size());
        REQUIRE(s.clearcoatRoughness.size() == s.roughness.size());
        CHECK(s.clearcoatFactor == std::vector<float>{0.0f, 0.25f, 0.5f, 1.0f, 1.0f, 1.0f});
        CHECK(s.clearcoatRoughness == std::vector<float>{0.0f, 0.05f, 0.15f, 0.4f, 1.0f});
        CHECK(s.baseIor == std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f, 2.4f, 0.0f});
        CHECK(s.metallic == std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f});
        CHECK(s.roughness == std::vector<float>(5, 0.6f)); // a rough base under every coat
        // alpha = roughness^2 < 0.05 for coat roughness 0, 0.05, 0.15: the narrow-lobe disk quadrature.
        CHECK(s.clearcoatRoughness[2] * s.clearcoatRoughness[2] < 0.05f);
        CHECK(s.clearcoatRoughness[3] * s.clearcoatRoughness[3] > 0.05f);
    }
    const auto& sun = MaterialValidation::spec("clearcoat-grid-sun");
    CHECK(sun.sun);
    CHECK(sun.environment == 0.0f);
    CHECK(sun.sunIrradiance == 3.0f);
    CHECK(sun.baseColor == glm::vec3(0.8f, 0.35f, 0.15f));
    const auto& resolve = MaterialValidation::spec("clearcoat-grid-env-resolve");
    const auto& transport = MaterialValidation::spec("clearcoat-grid-env");
    CHECK_FALSE(resolve.sun);
    CHECK(resolve.environment == 0.5f);
    CHECK(MaterialValidation::environmentRadiance("clearcoat-grid-env-resolve") == 0.5f);
    CHECK(transport.environment == 0.5f);
    CHECK(transport.baseColor == resolve.baseColor);
    CHECK(transport.clearcoatFactor == resolve.clearcoatFactor);
    const auto& furnace = MaterialValidation::spec("furnace-clearcoat");
    CHECK_FALSE(furnace.sun);
    CHECK(furnace.environment == 1.0f);
    CHECK(furnace.baseColor == glm::vec3(1.0f));
    // The other scenarios carry no coat.
    for (auto name : {"base-grid-sun", "specular-grid-sun", "furnace-specular"})
        CHECK_FALSE(MaterialValidation::spec(name).hasClearcoatGrid());
}

TEST_CASE("F15.3 clearcoat grid spheres carry the coat of their row and column; neutral rows carry no feature") {
    ECS ecs;
    GpuScene scene;
    test::NullTextureManager textures;
    MaterialValidation f{"clearcoat-grid-sun"};
    f.setup(ecs, scene, textures);
    const auto& s = MaterialValidation::spec("clearcoat-grid-sun");
    for (u32 row = 0; row < s.metallic.size(); ++row)
        for (u32 col = 0; col < s.roughness.size(); ++col) {
            const EntityID e = f.spheres()[row * s.roughness.size() + col];
            const auto& m = std::as_const(ecs).getComponent<MaterialComponent>(e);
            CHECK(m.baseColorTexIndex == INVALID_TEXTURE_INDEX);
            CHECK(m.metallicFactor == s.metallic[row]);
            CHECK(m.roughnessFactor == s.roughness[col]);
            if (row == 0) { // coat 0 and no ior block: the neutral material, no extension at all
                CHECK((!m.extension || featureMask(*m.extension) == 0));
                continue;
            }
            REQUIRE(m.extension);
            REQUIRE(m.extension->clearcoat);
            CHECK(m.extension->clearcoat->factor == s.clearcoatFactor[row]);
            CHECK(m.extension->clearcoat->roughness == s.clearcoatRoughness[col]);
            CHECK(m.extension->clearcoat->normalScale == 1.0f);
            CHECK_FALSE(m.extension->clearcoat->factorTexture.present());
            CHECK_FALSE(m.extension->clearcoat->roughnessTexture.present());
            CHECK_FALSE(m.extension->clearcoat->normalTexture.present());
            CHECK((featureMask(*m.extension) & MATERIAL_FEATURE_CLEARCOAT) != 0);
            if (row == 4) { // the ior 2.4 row: both extensions are active
                REQUIRE(m.extension->specular);
                CHECK(m.extension->specular->ior == 2.4f);
                CHECK(featureMask(*m.extension) == (MATERIAL_FEATURE_CLEARCOAT | MATERIAL_FEATURE_SPECULAR_IOR));
            } else {
                CHECK_FALSE(m.extension->specular);
                CHECK(featureMask(*m.extension) == MATERIAL_FEATURE_CLEARCOAT);
            }
        }
    SceneStore store;
    store.sync(ecs, scene);
    CHECK(store.verifyAgainstEcs(ecs, scene).empty());
    CHECK(store.featureUnion() == (MATERIAL_FEATURE_CLEARCOAT | MATERIAL_FEATURE_SPECULAR_IOR));
    // The record words the reference reads (GPUMaterialExt0 word offsets 7, 8, 9: factor, roughness, normalScale).
    const u32 coatRow = 3, coatCol = 2;
    const EntityID e = f.spheres()[coatRow * s.roughness.size() + coatCol];
    const auto& rec = store.extensions()[store.materials()[store.materialIndexOf(e)].extensionIndex];
    float factor, roughness, normalScale;
    std::memcpy(&factor, &rec.w[7], 4);
    std::memcpy(&roughness, &rec.w[8], 4);
    std::memcpy(&normalScale, &rec.w[9], 4);
    CHECK(factor == 1.0f);
    CHECK(roughness == 0.15f);
    CHECK(normalScale == 1.0f);
    CHECK(rec.w[10] == INVALID_TEXTURE_INDEX);
    CHECK(rec.w[11] == INVALID_TEXTURE_INDEX);
    CHECK(rec.w[12] == INVALID_TEXTURE_INDEX);
    f.teardown(ecs, scene);
}

TEST_CASE("F15.3 clearcoat-grid-point: the coat grid lit by one point light from behind the grid") {
    const auto& sun = MaterialValidation::spec("clearcoat-grid-sun");
    const auto& point = MaterialValidation::spec("clearcoat-grid-point");
    CHECK(point.clearcoatFactor == sun.clearcoatFactor);
    CHECK(point.clearcoatRoughness == sun.clearcoatRoughness);
    CHECK(point.baseIor == sun.baseIor);
    CHECK_FALSE(point.sun);
    CHECK(point.environment == 0.0f);
    REQUIRE(point.pointLights.size() == 1);
    const MaterialPointLight light = point.pointLights[0];
    CHECK(light.intensity == 250.0f);
    CHECK(light.range == 100.0f);
    const glm::vec3 camera = MaterialValidation{"clearcoat-grid-point"}.getDefaultCamera().position;
    const glm::vec3 centre(0.0f, 2.5f, 0.0f); // six rows: the grid centre is at y = 2.5
    CHECK(camera.y == doctest::Approx(centre.y));
    const float vl = glm::dot(glm::normalize(camera - centre), glm::normalize(light.position - centre));
    CHECK(vl < -0.4f);
    CHECK(vl > -0.6f);
    CHECK(light.position.z < -MaterialValidation::SphereRadius);
    CHECK(glm::length(light.position - centre) == doctest::Approx(9.0f));
    for (u32 row = 0; row < point.metallic.size(); ++row)
        for (u32 col = 0; col < point.roughness.size(); ++col) {
            const float d = glm::length(light.position - MaterialValidation::sphereCenter(point, col, row));
            const float r = d / light.range, window = (1.0f - r * r * r * r) * (1.0f - r * r * r * r);
            CHECK(window > 0.9995f);
        }
    // The coat highlight (N = H of the side light) falls on the visible front of the spheres: a coat reflection to see.
    const glm::vec3 h = glm::normalize(glm::normalize(camera - centre) + glm::normalize(light.position - centre));
    CHECK(h.z > 0.3f);
}
