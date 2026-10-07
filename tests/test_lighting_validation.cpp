#include "testbench/lighting_validation.h"
#include "renderer/gpu_scene.h"
#include "renderer/scene_extract.h"
#include "scene/ecs.h"
#include "scene/components.h"
#include "null_texture_manager.h"
#include <doctest/doctest.h>
#include <glm/gtc/matrix_transform.hpp>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

using namespace phosphor;
namespace {
struct Fixture {
    ECS ecs;
    GpuScene scene;
    test::NullTextureManager textures;
    LightingValidation bench;
    explicit Fixture(std::string scenario) : bench(std::move(scenario)) { bench.setup(ecs, scene, textures); }
    const ECS& read() const { return ecs; }
};
}

TEST_CASE("lighting fixture: all named scenarios are explicit and invalid choices fail") {
    CHECK(LightingValidation::scenarios().size() == 10);
    for (std::string_view name : LightingValidation::scenarios()) {
        CHECK(LightingValidation::validScenario(name));
        LightingValidation scene{std::string(name)};
        CHECK(scene.scenario() == name);
        CHECK(std::string(scene.getName()).find(name) != std::string::npos);
    }
    CHECK_FALSE(LightingValidation::validScenario(""));
    CHECK_FALSE(LightingValidation::validScenario("Cornell"));
    CHECK_THROWS_AS(LightingValidation("unknown"), std::invalid_argument);
}

TEST_CASE("lighting fixture: corpus uses bounded shared geometry and exact material fallbacks") {
    for (std::string_view name : LightingValidation::scenarios()) {
        CAPTURE(name);
        Fixture f{std::string(name)};
        REQUIRE(f.scene.getMeshCount() == 2);
        REQUIRE(f.scene.vertices().size() == 28); // four plane +24 cube vertices
        REQUIRE(f.scene.indices().size() == 42); // two plane +12 cube triangles
        CHECK(f.ecs.entityCount() <= 40);
        CHECK(f.textures.textureCount() == (name == "alpha-mip-shadow" ? 5 : 4));
        FrameScene frame;
        extractFrameScene(f.ecs, f.scene, frame);
        REQUIRE(!frame.instances.empty());
        u32 triangles = 0;
        for (const auto& instance : frame.instances) {
            REQUIRE(instance.meshIndex < f.scene.getMeshCount());
            REQUIRE(instance.materialIndex < frame.materials.size());
            triangles += f.scene.meshInfos()[instance.meshIndex].indexCount / 3;
        }
        CHECK(triangles < 512);
        for (const auto& material : frame.materials) {
            CHECK(material.metallic == 0);
            CHECK(material.roughness == 1);
            if (name != "alpha-mip-shadow" || material.alphaCutoff == 0)
                CHECK(material.baseColorTex == INVALID_TEXTURE_INDEX);
            for (u32 index : {material.normalTex, material.metallicRoughnessTex,
                               material.occlusionTex, material.emissiveTex}) CHECK(index == INVALID_TEXTURE_INDEX);
        }
        const auto camera = f.bench.getDefaultCamera();
        CHECK(glm::length(camera.position - camera.target) > 1);
        CHECK_FALSE(camera.orbit);
    }
}

TEST_CASE("lighting fixture: Cornell emits only its physical downward-facing panel") {
    Fixture f("cornell");
    REQUIRE(f.bench.emissivePanel() != INVALID_ENTITY);
    CHECK(f.read().getArray<LightComponent>().size() == 0); // no hidden point proxy
    const auto& panel = f.read().getComponent<MaterialComponent>(f.bench.emissivePanel());
    CHECK(panel.emissiveFactor == glm::vec3(12));
    CHECK_FALSE(panel.doubleSided);
    const auto& transform = f.read().getComponent<TransformComponent>(f.bench.emissivePanel());
    CHECK(transform.position.y == doctest::Approx(3.98));
    CHECK(transform.scale.x * transform.scale.z == doctest::Approx(0.8));
    const auto normal = glm::mat3(transform.worldMatrix) * glm::vec3(0, 1, 0);
    CHECK(normal.y < -0.99f);
    f.ecs.endFrame(); f.bench.update(0.5f, f.ecs);
    CHECK(f.read().getArray<TransformComponent>().changes().empty());
    CHECK(f.read().getArray<MaterialComponent>().changes().empty());
}

TEST_CASE("lighting fixture: thin wall is closed and documented embedded probe is inside solid") {
    Fixture f("thin-walls");
    REQUIRE(f.bench.thinWall() != INVALID_ENTITY);
    REQUIRE(f.bench.embeddedSolid() != INVALID_ENTITY);
    const auto& wall = f.read().getComponent<TransformComponent>(f.bench.thinWall());
    CHECK(wall.scale.x == doctest::Approx(0.01));
    const auto& wallMesh = f.read().getComponent<MeshInstanceComponent>(f.bench.thinWall());
    CHECK(wallMesh.castsShadows());
    CHECK(f.scene.meshInfos()[wallMesh.meshHandle].indexCount == 36); // six actual faces
    const auto& solid = f.read().getComponent<TransformComponent>(f.bench.embeddedSolid());
    const glm::vec4 local = glm::inverse(solid.worldMatrix) * glm::vec4(f.bench.embeddedProbeAnchor(), 1);
    for (u32 c = 0; c < 3; ++c) CHECK(std::abs(local[c]) < 0.5f);
    const auto& panel = f.read().getComponent<TransformComponent>(f.bench.emissivePanel());
    CHECK(panel.position.x < -0.5f); // explicit dark compartment behind wall
    // This verifies an anchor/solid geometry relation. It does not assert that
    // the runtime's automatically fitted probe grid includes this point.
}

TEST_CASE("lighting fixture: moving sources and disoccluder touch exact ECS state") {
    for (const std::string name : {"moving-sun", "moving-emissive", "disocclusion", "cache-stress"}) {
        CAPTURE(name);
        Fixture f(name);
        const EntityID actor = name == "moving-sun" ? f.bench.sun() : name == "moving-emissive" ? f.bench.emissivePanel() : f.bench.movingCaster();
        const glm::mat4 before = f.read().getComponent<TransformComponent>(actor).worldMatrix;
        f.ecs.endFrame();
        f.bench.update(0.75f, f.ecs);
        CHECK(f.read().getComponent<TransformComponent>(actor).worldMatrix != before);
        const auto changes = f.read().getArray<TransformComponent>().changes();
        CHECK_FALSE(changes.all);
        CHECK(changes.changed.size() == (name == "cache-stress" ? 37 : 1));
        CHECK(f.read().getArray<MaterialComponent>().changes().changed.size() ==
              (name == "moving-emissive" || name == "cache-stress" ? 1 : 0));
        if (name == "cache-stress") CHECK(f.read().getComponent<MeshInstanceComponent>(actor).isStatic());
        if (name == "moving-emissive") {
            CHECK(f.read().getComponent<MaterialComponent>(actor).emissiveFactor.x > 12);
            CHECK_FALSE(f.read().getComponent<MeshInstanceComponent>(actor).isStatic());
        }
    }
}

TEST_CASE("lighting fixture: sun rotation is visible to the shared light extraction") {
    Fixture f("moving-sun");
    std::vector<GPULight> before, after;
    extractLights(f.read(), before);
    f.bench.update(1, f.ecs);
    extractLights(f.read(), after);
    REQUIRE(before.size() == 1); REQUIRE(after.size() == 1);
    CHECK(after[0].type == LIGHT_DIRECTIONAL);
    CHECK(after[0].direction[1] < -0.6f); // travels down, toward-light is up
    const glm::vec3 a(before[0].direction[0], before[0].direction[1], before[0].direction[2]);
    const glm::vec3 b(after[0].direction[0], after[0].direction[1], after[0].direction[2]);
    CHECK(glm::dot(a, b) < 0.99f);
}

TEST_CASE("lighting fixture: offscreen caster has a visible geometric shadow target") {
    Fixture f("offscreen-caster");
    const auto camera = f.bench.getDefaultCamera();
    const auto vp = glm::perspective(glm::radians(60.0f), 16.0f / 9, 0.1f, 100.0f) *
                    glm::lookAt(camera.position, camera.target, glm::vec3(0, 1, 0));
    const auto& caster = f.read().getComponent<TransformComponent>(f.bench.movingCaster());
    for (float x : {-0.5f, 0.5f}) for (float y : {-0.5f, 0.5f}) for (float z : {-0.5f, 0.5f}) {
        const glm::vec4 clip = vp * caster.worldMatrix * glm::vec4(x, y, z, 1);
        REQUIRE(clip.w > 0);
        CHECK(clip.x / clip.w < -1); // whole caster beyond left camera plane
    }
    std::vector<GPULight> lights; extractLights(f.read(), lights);
    REQUIRE(lights.size() == 1);
    const glm::vec3 toward(-lights[0].direction[0], -lights[0].direction[1], -lights[0].direction[2]);
    const glm::vec3 projected = caster.position - toward * (caster.position.y / toward.y);
    CHECK(std::abs(projected.x) < 1e-4f); CHECK(std::abs(projected.z) < 1e-4f); CHECK(std::abs(projected.y) < 1e-4f);
    CHECK(f.read().getComponent<MeshInstanceComponent>(f.bench.movingCaster()).castsShadows());
}

TEST_CASE("lighting fixture: bias corpus freezes plate distances and nominal disk radius") {
    Fixture f("shadow-bias");
    std::array<bool, 3> found{};
    const float heights[] = {0.02f, 0.5f, 2.0f};
    for (const auto& transform : f.read().getArray<TransformComponent>().data()) {
        if (std::abs(transform.scale.y - 0.005f) > 1e-6f) continue;
        for (u32 i = 0; i < 3; ++i) found[i] |= std::abs(transform.position.y - heights[i]) < 1e-6f;
    }
    for (bool plate : found) CHECK(plate);
    CHECK(LightingValidation::NominalSunAngularRadius == doctest::Approx(0.00465));
    CHECK(2 * std::tan(LightingValidation::NominalSunAngularRadius) >
          0.5 * std::tan(LightingValidation::NominalSunAngularRadius));
}

TEST_CASE("lighting fixture: disocclusion script reports each cut once") {
    LightingValidation bench("disocclusion");
    glm::vec3 p, target; bool cut = true;
    REQUIRE(bench.scriptedCamera(0, p, target, cut)); CHECK_FALSE(cut);
    REQUIRE(bench.scriptedCamera(1, p, target, cut)); CHECK_FALSE(cut);
    REQUIRE(bench.scriptedCamera(2, p, target, cut)); CHECK(cut); CHECK(p.x == doctest::Approx(1.4));
    REQUIRE(bench.scriptedCamera(2.1, p, target, cut)); CHECK_FALSE(cut);
    REQUIRE(bench.scriptedCamera(8, p, target, cut)); CHECK(cut); CHECK(p.x == 0);
    REQUIRE(bench.scriptedCamera(8.1, p, target, cut)); CHECK_FALSE(cut);
    CHECK_FALSE(bench.scriptedCamera(-1, p, target, cut));
    LightingValidation staticBench("cornell"); CHECK_FALSE(staticBench.scriptedCamera(0, p, target, cut));
}

TEST_CASE("lighting fixture: update guards dt and teardown only removes owned entities") {
    Fixture f("moving-emissive");
    const EntityID unrelated = f.ecs.createEntity();
    TransformComponent extra; f.ecs.addComponent(unrelated, std::move(extra));
    CHECK_THROWS_AS(f.bench.setup(f.ecs, f.scene, f.textures), std::logic_error);
    f.ecs.endFrame(); f.bench.update(0, f.ecs);
    CHECK(f.read().getArray<TransformComponent>().changes().empty());
    CHECK_THROWS_AS(f.bench.update(-1, f.ecs), std::invalid_argument);
    CHECK_THROWS_AS(f.bench.update(std::numeric_limits<float>::quiet_NaN(), f.ecs), std::invalid_argument);
    f.bench.teardown(f.ecs, f.scene);
    CHECK(f.ecs.entityCount() == 1);
    CHECK(f.ecs.hasComponent<TransformComponent>(unrelated));
    CHECK(f.read().getArray<MeshInstanceComponent>().size() == 0);
    CHECK(f.read().getArray<MaterialComponent>().size() == 0);
    CHECK(f.bench.entities().empty());
    f.bench.teardown(f.ecs, f.scene); // idempotent
    f.ecs.destroyEntity(unrelated);
}

TEST_CASE("F10 alpha mip fixture distinguishes mip0 from mip1 before GPU execution") {
    Fixture f("alpha-mip-shadow");
    REQUIRE(f.bench.alphaReceiver() != INVALID_ENTITY);
    const auto& material = f.read().getComponent<MaterialComponent>(f.bench.alphaReceiver());
    CHECK(material.alphaCutoff == 0.75f);
    REQUIRE(material.baseColorTexIndex < f.textures.uploads.size());
    const auto& texture = f.textures.uploads[material.baseColorTexIndex];
    REQUIRE(texture.width == 256);
    REQUIRE(texture.height == 256);
    CHECK_FALSE(texture.sRGB);
    u32 opaque = 0;
    for (u32 y = 0; y < texture.height; y += 2) for (u32 x = 0; x < texture.width; x += 2) {
        u32 sum = 0;
        for (u32 dy = 0; dy < 2; ++dy) for (u32 dx = 0; dx < 2; ++dx) {
            const auto alpha = texture.rgba[(size_t(y + dy) * texture.width + x + dx) * 4 + 3];
            sum += alpha;
            opaque += alpha == 255;
        }
        CHECK(sum == 510);
        CHECK(float(sum) / (4 * 255.0f) < material.alphaCutoff);
    }
    CHECK(opaque == texture.width * texture.height / 2);
    CHECK(f.bench.sun() != INVALID_ENTITY);
}

TEST_CASE("F10 penumbra fixture has exact elevated plates and a vertical physical sun") {
    Fixture f("shadow-penumbra");
    std::vector<GPULight> lights;
    extractLights(f.ecs, lights);
    REQUIRE(lights.size() == 1);
    CHECK(lights[0].type == LIGHT_DIRECTIONAL);
    CHECK(lights[0].direction[1] == doctest::Approx(-1).epsilon(1e-5));
    u32 eight = 0, thirtyTwo = 0;
    const auto& transforms = f.read().getArray<TransformComponent>();
    for (const auto& transform : transforms.data()) {
        eight += transform.position.y == 8;
        thirtyTwo += transform.position.y == 32;
    }
    CHECK(eight == 1);
    CHECK(thirtyTwo == 1);
    CHECK(8 * std::tan(LightingValidation::NominalSunAngularRadius) > 0.037f);
    CHECK(32 * std::tan(LightingValidation::NominalSunAngularRadius) > 0.148f);
}
