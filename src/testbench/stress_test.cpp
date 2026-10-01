#include "testbench/stress_test.h"
#include "scene/ecs.h"
#include "scene/procedural.h"
#include "scene/texture_manager.h"
#include "renderer/gpu_scene.h"
#include "core/log.h"

#include <glm/gtc/matrix_transform.hpp>
#include <random>

namespace phosphor {

void StressTest::setup(ECS& ecs, GpuScene& gpuScene, TextureManager& textures) {
    textures.createDefaultTextures();
    LOG_INFO("StressTest: setting up %u instances...", INSTANCE_COUNT);

    // Low-poly sphere (8 slices x 6 stacks = ~96 triangles)
    auto sphereMesh = ProceduralMeshes::generateSphere(0.5f, 8, 6);
    MeshHandle sphereHandle = gpuScene.uploadMesh(
        sphereMesh.positions, sphereMesh.normals,
        sphereMesh.tangents, sphereMesh.uvs, sphereMesh.indices);

    entities_.reserve(INSTANCE_COUNT + MATERIAL_COUNT + 1);

    // --- Create random materials ---
    std::mt19937 rng(42); // deterministic seed for reproducibility
    std::uniform_real_distribution<float> colorDist(0.1f, 1.0f);
    std::uniform_real_distribution<float> metalDist(0.0f, 1.0f);
    std::uniform_real_distribution<float> roughDist(0.1f, 1.0f);

    for (u32 m = 0; m < MATERIAL_COUNT; ++m) {
        // Materials are stored implicitly by index in the GPU material array.
        // The ECS material component is used for the upload path.
        (void)m; // materials are assigned to instances below
    }

    // --- Create 100K instances ---
    std::uniform_real_distribution<float> posDist(-VOLUME_EXTENT, VOLUME_EXTENT);
    std::uniform_real_distribution<float> scaleDist(0.3f, 1.5f);
    std::uniform_int_distribution<u32> matDist(0, MATERIAL_COUNT - 1);

    for (u32 i = 0; i < INSTANCE_COUNT; ++i) {
        EntityID entity = ecs.createEntity();
        entities_.push_back(entity);

        TransformComponent xform{};
        xform.position = glm::vec3(posDist(rng), posDist(rng) * 0.3f, posDist(rng));
        xform.scale    = glm::vec3(scaleDist(rng));
        xform.updateMatrix();
        ecs.addComponent(entity, std::move(xform));

        MeshInstanceComponent inst{};
        inst.meshHandle    = sphereHandle;
        inst.materialIndex = matDist(rng);
        inst.setVisible(true);
        inst.setCastsShadows(false); // skip shadows for performance
        inst.setStatic(true);
        ecs.addComponent(entity, std::move(inst));

        MaterialComponent mat{};
        mat.baseColorFactor   = glm::vec4(colorDist(rng), colorDist(rng), colorDist(rng), 1.0f);
        mat.metallicFactor    = metalDist(rng);
        mat.roughnessFactor   = roughDist(rng);
        mat.baseColorTexIndex = textures.getDefaultWhite();
        mat.normalTexIndex    = textures.getDefaultNormal();
        mat.metallicRoughnessTexIndex = textures.getDefaultMR();
        ecs.addComponent(entity, std::move(mat));
    }

    // --- Single directional light ---
    EntityID lightEntity = ecs.createEntity();
    entities_.push_back(lightEntity);

    TransformComponent lightXform{};
    lightXform.position = glm::vec3(0.0f, 100.0f, 0.0f);
    lightXform.updateMatrix();
    ecs.addComponent(lightEntity, std::move(lightXform));

    LightComponent dirLight{};
    dirLight.type      = LightType::Directional;
    dirLight.color     = glm::vec3(1.0f, 0.95f, 0.9f);
    dirLight.intensity = 4.0f;
    ecs.addComponent(lightEntity, std::move(dirLight));

    LOG_INFO("StressTest: setup complete (%u entities)", static_cast<u32>(entities_.size()));
}

void StressTest::update([[maybe_unused]] float dt, [[maybe_unused]] ECS& ecs) {
    // Static scene -- no per-frame updates
}

void StressTest::teardown(ECS& ecs, [[maybe_unused]] GpuScene& gpuScene) {
    for (EntityID e : entities_) {
        ecs.destroyEntity(e);
    }
    entities_.clear();
}

CameraSetup StressTest::getDefaultCamera() const {
    CameraSetup cam{};
    cam.position = glm::vec3(0.0f, 30.0f, 80.0f);
    cam.target   = glm::vec3(0.0f, 0.0f, 0.0f);
    cam.distance = 80.0f;
    cam.orbit    = false; // FPS mode for exploring the volume
    return cam;
}

} // namespace phosphor
