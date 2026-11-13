#include <doctest/doctest.h>
#include "testbench/material_validation.h"
#include "testbench/transmission_validation.h"
#include "renderer/gpu_scene.h"
#include "renderer/scene_store.h"
#include "renderer/scene_extract.h"
#include "scene/components.h"
#include "scene/ecs.h"
#include "null_texture_manager.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <regex>
#include <set>
#include <sstream>
#include <utility>
#include <glm/gtc/quaternion.hpp>
using namespace phosphor;

TEST_CASE("F15.3 material scenarios are exactly the nine declared names") {
    CHECK(MaterialValidation::scenarios().size() == 40);
    for (auto name : {"base-grid-sun", "base-grid-env", "base-grid-env-resolve", "furnace-base", "furnace-resolve",
                      "roughness-extremes", "roughness-extremes-sun", "specular-grid-sun", "specular-grid-env-resolve", "specular-grid-point",
                      "specular-grid-env", "furnace-specular", "clearcoat-grid-sun", "clearcoat-grid-env-resolve",
                      "clearcoat-grid-env", "clearcoat-grid-point", "furnace-clearcoat",
                      "specular-grid-edit", "specular-grid-rough", "sheen-grid-sun", "sheen-grid-env-resolve",
                      "sheen-grid-env", "sheen-grid-point", "furnace-sheen", "aniso-grid-sun", "aniso-grid-env-resolve",
                      "aniso-grid-env", "aniso-grid-point", "furnace-aniso", "aniso-mirrored-sun", "iridescence-grid-sun",
                      "iridescence-grid-env-resolve", "iridescence-grid-env", "iridescence-grid-point", "furnace-iridescence",
                      "subsurface-thin-sun", "subsurface-thin-env-resolve", "subsurface-thin-env", "subsurface-thin-point",
                      "furnace-thin-subsurface"})
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
        // One mesh, two for the mirrored carrier fixture (tangent handedness -1 and +1).
        CHECK(scene.getMeshCount() == (std::find(s.tangentHandedness.begin(), s.tangentHandedness.end(), 1.0f) !=
                                               s.tangentHandedness.end() ? 2u : 1u));
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
    // The two clip fixtures of F15.1 ticket 08 belong to tools/f15_clip_check.py, not to the reference runner.
    for (auto name : MaterialValidation::scenarios())
        if (std::string_view(name) != "specular-grid-edit" && std::string_view(name) != "specular-grid-rough") bench.insert(std::string(name));
    // F15.3 ticket 13: the transmission scenarios are a separate bench class on the same --material-scenario option.
    for (auto name : TransmissionValidation::scenarios()) bench.insert(std::string(name));
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

TEST_CASE("F15.1 the clip fixture's toggle is the strong edit: ior 2.4 on the feature sphere, specularColor 25 on the parameter sphere") {
    CHECK(MaterialValidation::spec("specular-grid-edit").strongEdit);
    CHECK_FALSE(MaterialValidation::spec("specular-grid-env").strongEdit);
    ECS ecs;
    GpuScene scene;
    test::NullTextureManager textures;
    MaterialValidation f{"specular-grid-edit", 2};
    f.setup(ecs, scene, textures);
    const EntityID featureSphere = f.spheres()[MaterialValidation::toggleFeatureSphere()];
    const EntityID parameterSphere = f.spheres()[MaterialValidation::toggleParameterSphere()];
    auto block = [&](EntityID e) { return *ecs.getComponent<MaterialComponent>(e).extension->specular; };
    CHECK(block(featureSphere).ior == 1.5f);
    CHECK(block(parameterSphere).specularColor == std::array<float, 3>{1.0f, 1.0f, 1.0f});
    f.update(0.0f, ecs); // update 1: nothing
    CHECK(block(featureSphere).ior == 1.5f);
    f.update(0.0f, ecs); // update 2 = event 1
    CHECK(block(featureSphere).ior == 2.4f);
    CHECK(block(parameterSphere).specularColor == std::array<float, 3>{25.0f, 25.0f, 25.0f});
    CHECK(block(parameterSphere).specularFactor == 0.5f); // the weight is not touched by the strong edit
    f.update(0.0f, ecs);
    f.update(0.0f, ecs); // event 2: back
    CHECK(block(featureSphere).ior == 1.5f);
    CHECK(block(parameterSphere).specularColor == std::array<float, 3>{1.0f, 1.0f, 1.0f});
    // the rough adaptive fixture: reuse domain of visibility_adaptive (roughness >= 0.75, metallic <= 0.05), every extension of the grid
    const auto& rough = MaterialValidation::spec("specular-grid-rough");
    for (float r : rough.roughness) CHECK(r >= 0.75f);
    for (float m : rough.metallic) CHECK(m <= 0.05f);
    CHECK(rough.sun);
    CHECK(rough.ior == MaterialValidation::spec("specular-grid-env").ior);
    f.teardown(ecs, scene);
}

TEST_CASE("F15.3 sheen grids declare sheen colour x sheen roughness over a dark rough base (tools/test_f15_check.py mirrors them)") {
    for (auto name : {"sheen-grid-sun", "sheen-grid-env-resolve", "sheen-grid-env", "sheen-grid-point", "furnace-sheen"}) {
        const auto& s = MaterialValidation::spec(name);
        REQUIRE(s.hasSheenGrid());
        CHECK_FALSE(s.hasSpecularGrid());
        CHECK_FALSE(s.hasClearcoatGrid());
        REQUIRE(s.sheenColor.size() == s.metallic.size());
        REQUIRE(s.sheenRoughness.size() == s.roughness.size());
        CHECK(s.sheenColor == std::vector<glm::vec3>{glm::vec3(0.0f), glm::vec3(1.0f), glm::vec3(1.0f, 0.5f, 0.25f),
                                                     glm::vec3(0.25f, 0.5f, 1.0f)});
        CHECK(s.sheenRoughness == std::vector<float>{0.1f, 0.3f, 0.5f, 0.75f, 1.0f});
        CHECK(s.metallic == std::vector<float>(4, 0.0f));
        CHECK(s.roughness == std::vector<float>(5, 0.7f)); // the base roughness, constant over the columns
        CHECK(s.sheenRoughness.front() >= 0.07f);          // above the declared lower bound of the Charlie lobe
        for (float r : s.sheenRoughness) CHECK(r <= 1.0f);
    }
    const auto& sun = MaterialValidation::spec("sheen-grid-sun");
    CHECK(sun.sun);
    CHECK(sun.environment == 0.0f);
    CHECK(sun.sunIrradiance == 3.0f);
    CHECK(sun.baseColor == glm::vec3(0.1f)); // dark base: the sheen is the visible feature
    const glm::vec3 expected = glm::normalize(glm::vec3(-0.8f, 0.35f, -0.3f));
    CHECK(glm::length(sun.towardSun - expected) < 1e-6f);
    const auto& resolve = MaterialValidation::spec("sheen-grid-env-resolve");
    const auto& transport = MaterialValidation::spec("sheen-grid-env");
    CHECK_FALSE(resolve.sun);
    CHECK(resolve.environment == 0.5f);
    CHECK(MaterialValidation::environmentRadiance("sheen-grid-env-resolve") == 0.5f);
    CHECK(transport.environment == 0.5f);
    CHECK(transport.baseColor == resolve.baseColor);
    CHECK(transport.sheenColor == resolve.sheenColor);
    const auto& furnace = MaterialValidation::spec("furnace-sheen");
    CHECK_FALSE(furnace.sun);
    CHECK(furnace.environment == 1.0f);
    CHECK(furnace.baseColor == glm::vec3(1.0f));
    for (auto name : {"base-grid-sun", "specular-grid-sun", "clearcoat-grid-sun", "furnace-clearcoat"})
        CHECK_FALSE(MaterialValidation::spec(name).hasSheenGrid());
    CHECK(MaterialValidation{"sheen-grid-sun"}.getDefaultCamera().position == glm::vec3(0.0f, 1.5f, 5.5f));
}

TEST_CASE("F15.3 sheen grid spheres carry the sheen of their row and column; the zero-colour row carries no feature") {
    ECS ecs;
    GpuScene scene;
    test::NullTextureManager textures;
    MaterialValidation f{"sheen-grid-sun"};
    f.setup(ecs, scene, textures);
    const auto& s = MaterialValidation::spec("sheen-grid-sun");
    for (u32 row = 0; row < s.metallic.size(); ++row)
        for (u32 col = 0; col < s.roughness.size(); ++col) {
            const EntityID e = f.spheres()[row * s.roughness.size() + col];
            const auto& m = std::as_const(ecs).getComponent<MaterialComponent>(e);
            CHECK(m.baseColorTexIndex == INVALID_TEXTURE_INDEX);
            CHECK(m.metallicFactor == s.metallic[row]);
            CHECK(m.roughnessFactor == s.roughness[col]);
            if (row == 0) { // sheen colour 0: the layer is disabled, the neutral material, no extension at all
                CHECK((!m.extension || featureMask(*m.extension) == 0));
                continue;
            }
            REQUIRE(m.extension);
            REQUIRE(m.extension->sheen);
            CHECK(m.extension->sheen->color == std::array<float, 3>{s.sheenColor[row].x, s.sheenColor[row].y, s.sheenColor[row].z});
            CHECK(m.extension->sheen->roughness == s.sheenRoughness[col]);
            CHECK_FALSE(m.extension->sheen->colorTexture.present());
            CHECK_FALSE(m.extension->sheen->roughnessTexture.present());
            CHECK_FALSE(m.extension->specular);
            CHECK_FALSE(m.extension->clearcoat);
            CHECK(featureMask(*m.extension) == MATERIAL_FEATURE_SHEEN);
        }
    SceneStore store;
    store.sync(ecs, scene);
    CHECK(store.verifyAgainstEcs(ecs, scene).empty());
    CHECK(store.featureUnion() == MATERIAL_FEATURE_SHEEN);
    // The record words the reference reads (GPUMaterialExt0 word offsets 13-15: sheenColor, 16: sheenRoughness,
    // 17 and 18: the two texture words).
    const u32 row = 2, col = 2;
    const EntityID e = f.spheres()[row * s.roughness.size() + col];
    const auto& rec = store.extensions()[store.materials()[store.materialIndexOf(e)].extensionIndex];
    float color[3], roughness;
    std::memcpy(color, &rec.w[13], 12);
    std::memcpy(&roughness, &rec.w[16], 4);
    CHECK(color[0] == 1.0f);
    CHECK(color[1] == 0.5f);
    CHECK(color[2] == 0.25f);
    CHECK(roughness == 0.5f);
    CHECK(rec.w[17] == INVALID_TEXTURE_INDEX);
    CHECK(rec.w[18] == INVALID_TEXTURE_INDEX);
    f.teardown(ecs, scene);
}

TEST_CASE("F15.3 sheen-grid-point: the sheen grid lit by one point light from behind the grid") {
    const auto& sun = MaterialValidation::spec("sheen-grid-sun");
    const auto& point = MaterialValidation::spec("sheen-grid-point");
    CHECK(point.sheenColor == sun.sheenColor);
    CHECK(point.sheenRoughness == sun.sheenRoughness);
    CHECK(point.baseColor == sun.baseColor);
    CHECK_FALSE(point.sun);
    CHECK(point.environment == 0.0f);
    REQUIRE(point.pointLights.size() == 1);
    const MaterialPointLight light = point.pointLights[0];
    CHECK(light.intensity == 250.0f);
    CHECK(light.range == 100.0f);
    const glm::vec3 camera = MaterialValidation{"sheen-grid-point"}.getDefaultCamera().position;
    const glm::vec3 centre(0.0f, 1.5f, 0.0f); // four rows: the grid centre is at y = 1.5
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
}

TEST_CASE("F15.3 anisotropy grids declare strength x rotation over a copper base (tools/test_f15_check.py mirrors them)") {
    for (auto name : {"aniso-grid-sun", "aniso-grid-env-resolve", "aniso-grid-env", "aniso-grid-point", "furnace-aniso"}) {
        const auto& s = MaterialValidation::spec(name);
        REQUIRE(s.hasAnisotropyGrid());
        CHECK_FALSE(s.hasSpecularGrid());
        CHECK_FALSE(s.hasClearcoatGrid());
        CHECK_FALSE(s.hasSheenGrid());
        REQUIRE(s.anisotropyStrength.size() == s.metallic.size());
        REQUIRE(s.anisotropyRotation.size() == s.roughness.size());
        CHECK(s.anisotropyStrength == std::vector<float>{0.0f, 0.5f, 1.0f, 0.0f, 0.5f, 1.0f});
        CHECK(s.metallic == std::vector<float>{0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f}); // dielectric rows, then metal rows
        CHECK(s.anisotropyRotation == std::vector<float>{0.0f, 0.78539816f, 1.57079633f, -0.78539816f});
        CHECK(s.roughness == std::vector<float>(4, 0.3f)); // base roughness 0.3: alpha 0.09, no sun quadrature mask
        CHECK(s.mirroredColumn.empty());
        CHECK(s.tangentHandedness.empty());
    }
    const auto& sun = MaterialValidation::spec("aniso-grid-sun");
    CHECK(sun.sun);
    CHECK(sun.environment == 0.0f);
    CHECK(sun.baseColor == glm::vec3(0.8f, 0.35f, 0.15f));
    CHECK(glm::length(sun.towardSun - glm::normalize(glm::vec3(-0.8f, 0.35f, -0.3f))) < 1e-6f);
    const auto& resolve = MaterialValidation::spec("aniso-grid-env-resolve");
    const auto& transport = MaterialValidation::spec("aniso-grid-env");
    CHECK_FALSE(resolve.sun);
    CHECK(resolve.environment == 0.5f);
    CHECK(transport.environment == 0.5f);
    CHECK(MaterialValidation::environmentRadiance("aniso-grid-env-resolve") == 0.5f);
    const auto& furnace = MaterialValidation::spec("furnace-aniso");
    CHECK_FALSE(furnace.sun);
    CHECK(furnace.environment == 1.0f);
    CHECK(furnace.baseColor == glm::vec3(1.0f));
    for (auto name : {"base-grid-sun", "specular-grid-sun", "clearcoat-grid-sun", "sheen-grid-sun"})
        CHECK_FALSE(MaterialValidation::spec(name).hasAnisotropyGrid());
    // Six rows of unit spacing: the grid centre is y = 2.5.
    CHECK(MaterialValidation{"aniso-grid-sun"}.getDefaultCamera().position == glm::vec3(0.0f, 2.5f, 5.5f));
}

TEST_CASE("F15.3 anisotropy grid spheres carry the strength of their row and the rotation of their column") {
    ECS ecs;
    GpuScene scene;
    test::NullTextureManager textures;
    MaterialValidation f{"aniso-grid-sun"};
    f.setup(ecs, scene, textures);
    const auto& s = MaterialValidation::spec("aniso-grid-sun");
    for (u32 row = 0; row < s.metallic.size(); ++row)
        for (u32 col = 0; col < s.roughness.size(); ++col) {
            const EntityID e = f.spheres()[row * s.roughness.size() + col];
            const auto& m = std::as_const(ecs).getComponent<MaterialComponent>(e);
            CHECK(m.metallicFactor == s.metallic[row]);
            CHECK(m.roughnessFactor == s.roughness[col]);
            if (s.anisotropyStrength[row] == 0.0f) { // strength 0: the feature is inactive, the neutral material
                CHECK((!m.extension || featureMask(*m.extension) == 0));
                continue;
            }
            REQUIRE(m.extension);
            REQUIRE(m.extension->anisotropy);
            CHECK(m.extension->anisotropy->strength == s.anisotropyStrength[row]);
            CHECK(m.extension->anisotropy->rotation == s.anisotropyRotation[col]);
            CHECK_FALSE(m.extension->anisotropy->texture.present());
            CHECK_FALSE(m.extension->specular);
            CHECK_FALSE(m.extension->clearcoat);
            CHECK_FALSE(m.extension->sheen);
            CHECK(featureMask(*m.extension) == MATERIAL_FEATURE_ANISOTROPY);
        }
    SceneStore store;
    store.sync(ecs, scene);
    CHECK(store.verifyAgainstEcs(ecs, scene).empty());
    CHECK(store.featureUnion() == MATERIAL_FEATURE_ANISOTROPY);
    // The words the reference reads: record 1 (GPUMaterialExt1) word 0 strength, word 1 rotation, word 2 texture.
    const u32 row = 4, col = 3; // strength 0.5, rotation -pi/4 on the metal row
    const EntityID e = f.spheres()[row * s.roughness.size() + col];
    const u32 group = store.materials()[store.materialIndexOf(e)].extensionIndex;
    const auto& rec = store.extensions()[group + 1];
    float strength, rotation;
    std::memcpy(&strength, &rec.w[0], 4);
    std::memcpy(&rotation, &rec.w[1], 4);
    CHECK(strength == 0.5f);
    CHECK(rotation == -0.78539816f);
    CHECK(rec.w[2] == INVALID_TEXTURE_INDEX);
    f.teardown(ecs, scene);
}

TEST_CASE("F15.3 anisotropy grids are rendered on combed spheres whose tangents come from the mesh (handedness -1)") {
    ECS ecs;
    GpuScene scene;
    test::NullTextureManager textures;
    MaterialValidation f{"aniso-grid-sun"};
    f.setup(ecs, scene, textures);
    CHECK(scene.getMeshCount() == 1);
    // Every sphere of the grid is instanced from that one mesh, not mirrored.
    SceneStore store;
    store.sync(ecs, scene);
    for (EntityID e : f.spheres()) CHECK((store.instances()[store.slotOf(e)].flags & INSTANCE_FLAG_MIRRORED) == 0u);
    f.teardown(ecs, scene);
}

TEST_CASE("F15.3 aniso-mirrored-sun: plain and mirrored carriers over both tangent handedness values") {
    const auto& s = MaterialValidation::spec("aniso-mirrored-sun");
    CHECK(s.sun);
    CHECK(s.environment == 0.0f);
    CHECK(s.sunIrradiance == 3.0f);
    CHECK(s.roughness == std::vector<float>(4, 0.3f));
    CHECK(s.metallic == std::vector<float>{0.0f, 1.0f});
    CHECK(s.anisotropyStrength == std::vector<float>{1.0f, 1.0f});
    CHECK(s.anisotropyRotation == std::vector<float>(4, 0.78539816f));
    CHECK(s.mirroredColumn == std::vector<bool>{false, true, false, true});
    CHECK(s.tangentHandedness == std::vector<float>{-1.0f, -1.0f, 1.0f, 1.0f});
    ECS ecs;
    GpuScene scene;
    test::NullTextureManager textures;
    MaterialValidation f{"aniso-mirrored-sun"};
    f.setup(ecs, scene, textures);
    CHECK(scene.getMeshCount() == 2); // handedness -1 and +1
    SceneStore store;
    store.sync(ecs, scene);
    for (u32 row = 0; row < 2; ++row)
        for (u32 col = 0; col < 4; ++col) {
            const EntityID e = f.spheres()[row * 4 + col];
            const auto& t = std::as_const(ecs).getComponent<TransformComponent>(e);
            CHECK(t.scale == glm::vec3(s.mirroredColumn[col] ? -1.0f : 1.0f, 1.0f, 1.0f));
            const auto& instance = store.instances()[store.slotOf(e)];
            // The engine flags a negative determinant (glTF: mirrored nodes flip the winding), nothing else does.
            CHECK(((instance.flags & INSTANCE_FLAG_MIRRORED) != 0u) == s.mirroredColumn[col]);
            const auto& mesh = std::as_const(ecs).getComponent<MeshInstanceComponent>(e);
            CHECK(mesh.meshHandle == std::as_const(ecs).getComponent<MeshInstanceComponent>(f.spheres()[col < 2 ? 0 : 2]).meshHandle);
        }
    const auto handleMinus = std::as_const(ecs).getComponent<MeshInstanceComponent>(f.spheres()[0]).meshHandle;
    const auto handlePlus = std::as_const(ecs).getComponent<MeshInstanceComponent>(f.spheres()[2]).meshHandle;
    CHECK(handleMinus != handlePlus);
    f.teardown(ecs, scene);
}

TEST_CASE("F15.3 aniso-grid-point: the anisotropy grid lit by one point light from behind the grid") {
    const auto& sun = MaterialValidation::spec("aniso-grid-sun");
    const auto& point = MaterialValidation::spec("aniso-grid-point");
    CHECK(point.anisotropyStrength == sun.anisotropyStrength);
    CHECK(point.anisotropyRotation == sun.anisotropyRotation);
    CHECK_FALSE(point.sun);
    CHECK(point.environment == 0.0f);
    REQUIRE(point.pointLights.size() == 1);
    const MaterialPointLight light = point.pointLights[0];
    CHECK(light.intensity == 250.0f);
    CHECK(light.range == 100.0f);
    const glm::vec3 camera = MaterialValidation{"aniso-grid-point"}.getDefaultCamera().position;
    const glm::vec3 centre(0.0f, 2.5f, 0.0f); // six rows: the grid centre is at y = 2.5
    CHECK(camera.y == doctest::Approx(centre.y));
    const float vl = glm::dot(glm::normalize(camera - centre), glm::normalize(light.position - centre));
    CHECK(vl < -0.4f);
    CHECK(vl > -0.6f);
    CHECK(glm::length(light.position - centre) == doctest::Approx(9.0f));
    for (u32 row = 0; row < point.metallic.size(); ++row)
        for (u32 col = 0; col < point.roughness.size(); ++col) {
            const float d = glm::length(light.position - MaterialValidation::sphereCenter(point, col, row));
            const float r = d / light.range, window = (1.0f - r * r * r * r) * (1.0f - r * r * r * r);
            CHECK(window > 0.9995f);
        }
}

TEST_CASE("F15.3 iridescence grids declare factor x ior rows over thickness columns on a copper base (tools/test_f15_check.py mirrors them)") {
    for (auto name : {"iridescence-grid-sun", "iridescence-grid-env-resolve", "iridescence-grid-env", "iridescence-grid-point",
                      "furnace-iridescence"}) {
        const auto& s = MaterialValidation::spec(name);
        REQUIRE(s.hasIridescenceGrid());
        CHECK_FALSE(s.hasSpecularGrid());
        CHECK_FALSE(s.hasClearcoatGrid());
        CHECK_FALSE(s.hasSheenGrid());
        CHECK_FALSE(s.hasAnisotropyGrid());
        REQUIRE(s.iridescenceFactor.size() == s.metallic.size());
        REQUIRE(s.iridescenceIor.size() == s.metallic.size());
        REQUIRE(s.iridescenceThickness.size() == s.roughness.size());
        CHECK(s.iridescenceFactor == std::vector<float>{0.5f, 1.0f, 1.0f, 0.5f, 1.0f, 1.0f});
        CHECK(s.iridescenceIor == std::vector<float>{1.3f, 1.3f, 1.8f, 1.3f, 1.3f, 1.8f});
        CHECK(s.iridescenceThickness == std::vector<float>{0.0f, 200.0f, 400.0f, 600.0f, 800.0f, 1200.0f});
        CHECK(s.metallic == std::vector<float>{0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f}); // dielectric rows, then metal rows
        CHECK(s.roughness == std::vector<float>(6, 0.3f));
    }
    const auto& sun = MaterialValidation::spec("iridescence-grid-sun");
    CHECK(sun.sun);
    CHECK(sun.environment == 0.0f);
    CHECK(sun.baseColor == glm::vec3(0.8f, 0.35f, 0.15f));
    const auto& resolve = MaterialValidation::spec("iridescence-grid-env-resolve");
    const auto& transport = MaterialValidation::spec("iridescence-grid-env");
    CHECK_FALSE(resolve.sun);
    CHECK(resolve.environment == 0.5f);
    CHECK(transport.environment == 0.5f);
    const auto& furnace = MaterialValidation::spec("furnace-iridescence");
    CHECK(furnace.environment == 1.0f);
    CHECK(furnace.baseColor == glm::vec3(1.0f));
    for (auto name : {"base-grid-sun", "aniso-grid-sun", "sheen-grid-sun"})
        CHECK_FALSE(MaterialValidation::spec(name).hasIridescenceGrid());
    CHECK(MaterialValidation{"iridescence-grid-sun"}.getDefaultCamera().position == glm::vec3(0.0f, 2.5f, 5.5f));
}

TEST_CASE("F15.3 iridescence grid spheres carry the factor and ior of their row and the thickness of their column") {
    ECS ecs;
    GpuScene scene;
    test::NullTextureManager textures;
    MaterialValidation f{"iridescence-grid-sun"};
    f.setup(ecs, scene, textures);
    const auto& s = MaterialValidation::spec("iridescence-grid-sun");
    for (u32 row = 0; row < s.metallic.size(); ++row)
        for (u32 col = 0; col < s.roughness.size(); ++col) {
            const EntityID e = f.spheres()[row * s.roughness.size() + col];
            const auto& m = std::as_const(ecs).getComponent<MaterialComponent>(e);
            CHECK(m.metallicFactor == s.metallic[row]);
            CHECK(m.roughnessFactor == s.roughness[col]);
            REQUIRE(m.extension);
            REQUIRE(m.extension->iridescence);
            const auto& block = *m.extension->iridescence;
            CHECK(block.factor == s.iridescenceFactor[row]);
            CHECK(block.ior == s.iridescenceIor[row]);
            CHECK(block.thicknessMax == s.iridescenceThickness[col]);
            CHECK(block.thicknessMin == 100.0f); // the default, so thickness-min differs from the maximum in every column
            CHECK_FALSE(block.factorTexture.present());
            CHECK_FALSE(block.thicknessTexture.present());
            CHECK_FALSE(m.extension->specular);
            CHECK_FALSE(m.extension->clearcoat);
            CHECK_FALSE(m.extension->sheen);
            CHECK_FALSE(m.extension->anisotropy);
            CHECK(featureMask(*m.extension) == MATERIAL_FEATURE_IRIDESCENCE); // thickness 0 keeps the block, the film is off by rule
        }
    SceneStore store;
    store.sync(ecs, scene);
    CHECK(store.verifyAgainstEcs(ecs, scene).empty());
    CHECK(store.featureUnion() == MATERIAL_FEATURE_IRIDESCENCE);
    // The words the reference reads: record 1 (GPUMaterialExt1) words 3 factor, 4 ior, 5 min, 6 max, 7 factor texture, 8 thickness texture.
    const u32 row = 2, col = 4; // factor 1, ior 1.8, thickness 800 on the dielectric rows
    const EntityID e = f.spheres()[row * s.roughness.size() + col];
    const u32 group = store.materials()[store.materialIndexOf(e)].extensionIndex;
    const auto& rec = store.extensions()[group + 1];
    float factor, ior, thicknessMin, thicknessMax;
    std::memcpy(&factor, &rec.w[3], 4);
    std::memcpy(&ior, &rec.w[4], 4);
    std::memcpy(&thicknessMin, &rec.w[5], 4);
    std::memcpy(&thicknessMax, &rec.w[6], 4);
    CHECK(factor == 1.0f);
    CHECK(ior == 1.8f);
    CHECK(thicknessMin == 100.0f);
    CHECK(thicknessMax == 800.0f);
    CHECK(rec.w[7] == INVALID_TEXTURE_INDEX);
    CHECK(rec.w[8] == INVALID_TEXTURE_INDEX);
    f.teardown(ecs, scene);
}

TEST_CASE("F15.3 iridescence-grid-point: the iridescence grid lit by one point light from behind the grid") {
    const auto& sun = MaterialValidation::spec("iridescence-grid-sun");
    const auto& point = MaterialValidation::spec("iridescence-grid-point");
    CHECK(point.iridescenceFactor == sun.iridescenceFactor);
    CHECK(point.iridescenceThickness == sun.iridescenceThickness);
    CHECK_FALSE(point.sun);
    CHECK(point.environment == 0.0f);
    REQUIRE(point.pointLights.size() == 1);
    const MaterialPointLight light = point.pointLights[0];
    CHECK(light.intensity == 250.0f);
    CHECK(light.range == 100.0f);
    const glm::vec3 camera = MaterialValidation{"iridescence-grid-point"}.getDefaultCamera().position;
    const glm::vec3 centre(0.0f, 2.5f, 0.0f);
    CHECK(camera.y == doctest::Approx(centre.y));
    const float vl = glm::dot(glm::normalize(camera - centre), glm::normalize(light.position - centre));
    CHECK(vl < -0.4f);
    CHECK(vl > -0.6f);
    CHECK(glm::length(light.position - centre) == doctest::Approx(9.0f));
    for (u32 row = 0; row < point.metallic.size(); ++row)
        for (u32 col = 0; col < point.roughness.size(); ++col) {
            const float d = glm::length(light.position - MaterialValidation::sphereCenter(point, col, row));
            const float r = d / light.range, window = (1.0f - r * r * r * r) * (1.0f - r * r * r * r);
            CHECK(window > 0.9995f);
        }
}

TEST_CASE("F15.3 thin subsurface grids declare weight rows over transmission colour columns on a blue-grey base (tools/test_f15_check.py mirrors them)") {
    for (auto name : {"subsurface-thin-sun", "subsurface-thin-env-resolve", "subsurface-thin-env", "subsurface-thin-point",
                      "furnace-thin-subsurface"}) {
        const auto& s = MaterialValidation::spec(name);
        REQUIRE(s.hasSubsurfaceGrid());
        CHECK_FALSE(s.hasSpecularGrid());
        CHECK_FALSE(s.hasClearcoatGrid());
        CHECK_FALSE(s.hasSheenGrid());
        CHECK_FALSE(s.hasAnisotropyGrid());
        CHECK_FALSE(s.hasIridescenceGrid());
        REQUIRE(s.subsurfaceWeight.size() == s.metallic.size());
        REQUIRE(s.subsurfaceColor.size() == s.roughness.size());
        for (float m : s.metallic) CHECK(m == 0.0f); // metals carry no diffuse lobe, so no transmission
    }
    const std::vector<glm::vec3> colors{glm::vec3(1.0f), glm::vec3(1.0f, 0.55f, 0.25f), glm::vec3(0.1f, 0.7f, 0.6f)};
    const auto& sun = MaterialValidation::spec("subsurface-thin-sun");
    CHECK(sun.sun);
    CHECK(sun.sunIrradiance == 3.0f);
    CHECK(sun.environment == 0.0f);
    CHECK(sun.baseColor == glm::vec3(0.3f, 0.5f, 0.8f));
    CHECK(sun.subsurfaceWeight == std::vector<float>{0.0f, 0.25f, 0.5f, 1.0f});
    CHECK(sun.subsurfaceColor == colors);
    CHECK(sun.roughness == std::vector<float>(3, 0.5f));
    CHECK_FALSE(sun.quadCarriers);
    // The sun is BEHIND the grid: toward-sun has a negative z, V.L at the grid centre about -0.73.
    CHECK(glm::length(sun.towardSun - glm::normalize(glm::vec3(-0.55f, 0.4f, -0.73f))) < 1e-6f);
    const glm::vec3 camera = MaterialValidation{"subsurface-thin-sun"}.getDefaultCamera().position;
    CHECK(camera == glm::vec3(0.0f, 1.5f, 5.5f));
    const float vl = glm::dot(glm::normalize(camera - glm::vec3(0.0f, 1.5f, 0.0f)), sun.towardSun);
    CHECK(vl < -0.7f);
    CHECK(vl > -0.76f);
    for (auto name : {"subsurface-thin-env-resolve", "subsurface-thin-env"}) {
        const auto& s = MaterialValidation::spec(name);
        CHECK_FALSE(s.sun);
        CHECK(s.environment == 0.5f);
        CHECK(s.pointLights.empty());
        CHECK(s.baseColor == sun.baseColor);
        CHECK(s.subsurfaceWeight == sun.subsurfaceWeight);
        CHECK(s.subsurfaceColor == colors);
    }
    const auto& furnace = MaterialValidation::spec("furnace-thin-subsurface");
    CHECK_FALSE(furnace.sun);
    CHECK(furnace.environment == 1.0f);
    CHECK(furnace.baseColor == glm::vec3(1.0f));
    CHECK(furnace.subsurfaceColor == std::vector<glm::vec3>(3, glm::vec3(1.0f))); // white transmission: the tight bound
    CHECK(furnace.roughness == std::vector<float>{0.2f, 0.5f, 0.9f});
    CHECK(furnace.subsurfaceWeight == sun.subsurfaceWeight);
    CHECK(MaterialValidation::environmentRadiance("furnace-thin-subsurface") == 1.0f);
}

TEST_CASE("F15.3 thin subsurface carriers hold the block of their row and column; the weight-0 row carries no feature") {
    ECS ecs;
    GpuScene scene;
    test::NullTextureManager textures;
    MaterialValidation f{"subsurface-thin-sun"};
    f.setup(ecs, scene, textures);
    const auto& s = MaterialValidation::spec("subsurface-thin-sun");
    for (u32 row = 0; row < s.metallic.size(); ++row)
        for (u32 col = 0; col < s.roughness.size(); ++col) {
            const EntityID e = f.spheres()[row * s.roughness.size() + col];
            const auto& m = std::as_const(ecs).getComponent<MaterialComponent>(e);
            CHECK(m.metallicFactor == 0.0f);
            CHECK(m.roughnessFactor == 0.5f);
            if (s.subsurfaceWeight[row] == 0.0f) {
                CHECK_FALSE(m.extension); // the neutral material
                continue;
            }
            REQUIRE(m.extension);
            REQUIRE(m.extension->subsurface);
            const auto& block = *m.extension->subsurface;
            CHECK(block.mode == SubsurfaceMode::Thin);
            CHECK(block.weight == s.subsurfaceWeight[row]);
            const glm::vec3 c = s.subsurfaceColor[col];
            CHECK(block.color == std::array<float, 3>{c.x, c.y, c.z});
            CHECK(block.anisotropy == 0.0f);
            CHECK_FALSE(block.weightTexture.present());
            CHECK_FALSE(block.colorTexture.present());
            CHECK_FALSE(block.radiusScaleTexture.present());
            CHECK_FALSE(m.extension->specular);
            CHECK_FALSE(m.extension->clearcoat);
            CHECK_FALSE(m.extension->sheen);
            CHECK_FALSE(m.extension->anisotropy);
            CHECK_FALSE(m.extension->iridescence);
            CHECK_FALSE(m.extension->transmission);
            CHECK(featureMask(*m.extension) == MATERIAL_FEATURE_SUBSURFACE);
        }
    SceneStore store;
    store.sync(ecs, scene);
    CHECK(store.verifyAgainstEcs(ecs, scene).empty());
    CHECK(store.featureUnion() == MATERIAL_FEATURE_SUBSURFACE);
    // The words the reference reads: record 2 (GPUMaterialExt2) word 0 mode, 1 weight, 2-4 colour, 9 anisotropy, 10-12 textures.
    const u32 row = 2, col = 1; // weight 0.5, warm
    const EntityID e = f.spheres()[row * s.roughness.size() + col];
    const u32 group = store.materials()[store.materialIndexOf(e)].extensionIndex;
    const auto& rec = store.extensions()[group + 2];
    float weight, r, g, b, anisotropy;
    std::memcpy(&weight, &rec.w[1], 4);
    std::memcpy(&r, &rec.w[2], 4);
    std::memcpy(&g, &rec.w[3], 4);
    std::memcpy(&b, &rec.w[4], 4);
    std::memcpy(&anisotropy, &rec.w[9], 4);
    CHECK(rec.w[0] == 0u); // thin
    CHECK(weight == 0.5f);
    CHECK(r == 1.0f);
    CHECK(g == 0.55f);
    CHECK(b == 0.25f);
    CHECK(anisotropy == 0.0f);
    CHECK(rec.w[10] == INVALID_TEXTURE_INDEX);
    CHECK(rec.w[11] == INVALID_TEXTURE_INDEX);
    CHECK(rec.w[12] == INVALID_TEXTURE_INDEX);
    f.teardown(ecs, scene);
}

TEST_CASE("F15.3 subsurface-thin-point: single-sided quads facing the camera, lit by one point light BEHIND them") {
    const auto& point = MaterialValidation::spec("subsurface-thin-point");
    CHECK(point.quadCarriers);
    CHECK_FALSE(point.sun);
    CHECK(point.environment == 0.0f);
    // No weight-0 row: nothing reaches a quad with no transmission from behind (the gate would have no lit pixels).
    CHECK(point.subsurfaceWeight == std::vector<float>{0.25f, 0.5f, 1.0f});
    CHECK(point.subsurfaceColor.size() == 3);
    REQUIRE(point.pointLights.size() == 1);
    const MaterialPointLight light = point.pointLights[0];
    CHECK(light.intensity == 250.0f);
    CHECK(light.range == 100.0f);
    const glm::vec3 camera = MaterialValidation{"subsurface-thin-point"}.getDefaultCamera().position;
    const glm::vec3 centre(0.0f, 1.0f, 0.0f);
    CHECK(camera.y == doctest::Approx(centre.y));
    CHECK(glm::length(light.position - centre) == doctest::Approx(9.0f));
    // The quads face +z (the camera): N.L < 0 for every quad, V.L about -0.73 at the centre.
    CHECK(light.position.z < 0.0f);
    const float vl = glm::dot(glm::normalize(camera - centre), glm::normalize(light.position - centre));
    CHECK(vl < -0.7f);
    CHECK(vl > -0.76f);
    for (u32 row = 0; row < point.metallic.size(); ++row)
        for (u32 col = 0; col < point.roughness.size(); ++col) {
            const float d = glm::length(light.position - MaterialValidation::sphereCenter(point, col, row));
            const float r = d / light.range, window = (1.0f - r * r * r * r) * (1.0f - r * r * r * r);
            CHECK(window > 0.9995f);
        }
    ECS ecs;
    GpuScene scene;
    test::NullTextureManager textures;
    MaterialValidation f{"subsurface-thin-point"};
    f.setup(ecs, scene, textures);
    CHECK(scene.getMeshCount() == 1u);
    REQUIRE(f.spheres().size() == 9);
    for (EntityID e : f.spheres()) {
        const auto& t = std::as_const(ecs).getComponent<TransformComponent>(e);
        const glm::vec3 normal = glm::mat3_cast(t.rotation) * glm::vec3(0.0f, 1.0f, 0.0f); // generatePlane: normal +Y
        CHECK(glm::length(normal - glm::vec3(0.0f, 0.0f, 1.0f)) < 1e-5f);
        CHECK(glm::dot(normal, glm::normalize(light.position - t.position)) < 0.0f);
        const auto& m = std::as_const(ecs).getComponent<MaterialComponent>(e);
        REQUIRE(m.extension);
        REQUIRE(m.extension->subsurface);
    }
    FrameScene frame;
    extractFrameScene(ecs, scene, frame);
    CHECK(frame.lights.size() == 1u);
    f.teardown(ecs, scene);
}
