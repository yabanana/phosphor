#include "testbench/stress_test.h"
#include "scene/ecs.h"
#include "scene/procedural.h"
#include "scene/texture_manager.h"
#include "renderer/gpu_scene.h"
#include "core/log.h"

#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>

namespace phosphor {

void StressTest::setup(ECS& ecs, GpuScene& gpuScene, TextureManager& textures) {
    textures.createDefaultTextures();
    if (const char* n = std::getenv("PHOSPHOR_SPIKE_INSTANCES")) count_ = static_cast<u32>(std::strtoul(n, nullptr, 10));
    dynamic_ = std::getenv("PHOSPHOR_SPIKE_DYNAMIC") != nullptr;
    const char* meshEnv = std::getenv("PHOSPHOR_SPIKE_MESH");
    const bool cube = meshEnv && std::strcmp(meshEnv, "cube") == 0;
    const float extent = VOLUME_EXTENT * std::cbrt(static_cast<float>(count_) / INSTANCE_COUNT);
    LOG_INFO("StressTest: setting up %u instances (%s, %s)...", count_, cube ? "cube" : "sphere",
             dynamic_ ? "dynamic" : "static");

    // Low-poly sphere (8 slices x 6 stacks = ~96 triangles)
    auto sphereMesh = cube ? ProceduralMeshes::generateCube(0.5f) : ProceduralMeshes::generateSphere(0.5f, 8, 6);
    MeshHandle sphereHandle = gpuScene.uploadMesh(
        sphereMesh.positions, sphereMesh.normals,
        sphereMesh.tangents, sphereMesh.uvs, sphereMesh.indices);
    // F5 spike S7: K copies of the mesh -> K draw batches.
    u32 meshCopies = 1;
    if (const char* k = std::getenv("PHOSPHOR_SPIKE_MESHES")) meshCopies = std::max(1u, static_cast<u32>(std::strtoul(k, nullptr, 10)));
    for (u32 m = 1; m < meshCopies; ++m) {
        gpuScene.uploadMesh(sphereMesh.positions, sphereMesh.normals, sphereMesh.tangents, sphereMesh.uvs, sphereMesh.indices);
    }

    entities_.reserve(count_ + MATERIAL_COUNT + 1);

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
    std::uniform_real_distribution<float> posDist(-extent, extent);
    std::uniform_real_distribution<float> scaleDist(0.3f, 1.5f);
    std::uniform_int_distribution<u32> matDist(0, MATERIAL_COUNT - 1);

    for (u32 i = 0; i < count_; ++i) {
        EntityID entity = ecs.createEntity();
        entities_.push_back(entity);

        TransformComponent xform{};
        xform.position = glm::vec3(posDist(rng), posDist(rng) * 0.3f, posDist(rng));
        xform.scale    = glm::vec3(scaleDist(rng));
        xform.updateMatrix();
        ecs.addComponent(entity, std::move(xform));

        MeshInstanceComponent inst{};
        inst.meshHandle    = sphereHandle + i % meshCopies;
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

    if (dynamic_) {
        base_.clear();
        for (const TransformComponent& t : ecs.getArray<TransformComponent>().data()) base_.push_back(t.position);
    }
    LOG_INFO("StressTest: setup complete (%u entities)", static_cast<u32>(entities_.size()));
}

void StressTest::update(float dt, ECS& ecs) {
    if (!dynamic_) return;
    // F5 spike S1: every instance moves (the best a CPU path can do: the
    // dense array, no entity lookups).
    time_ += dt;
    auto xforms = ecs.getArray<TransformComponent>().data();
    for (size_t i = 0; i < xforms.size() && i < base_.size(); ++i) {
        xforms[i].position = base_[i] + glm::vec3(0.0f, std::sin(time_ + static_cast<float>(i) * 0.001f), 0.0f);
        xforms[i].updateMatrix();
    }
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
