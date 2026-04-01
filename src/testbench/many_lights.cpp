#include "testbench/many_lights.h"
#include "scene/ecs.h"
#include "scene/procedural.h"
#include "scene/texture_manager.h"
#include "renderer/gpu_scene.h"
#include "core/log.h"

#include <glm/gtc/matrix_transform.hpp>
#include <cmath>
#include <random>

namespace phosphor {

void ManyLights::setup(ECS& ecs, GpuScene& gpuScene, TextureManager& textures) {
    textures.createDefaultTextures();

    // --- Room geometry: large cube (walls, floor, ceiling) ---
    auto cubeMesh = ProceduralMeshes::generateCube(1.0f);
    MeshHandle cubeHandle = gpuScene.uploadMesh(
        cubeMesh.positions, cubeMesh.normals,
        cubeMesh.tangents, cubeMesh.uvs, cubeMesh.indices);

    auto planeMesh = ProceduralMeshes::generatePlane(ROOM_SIZE * 2.0f, ROOM_SIZE * 2.0f, 1, 1);
    MeshHandle planeHandle = gpuScene.uploadMesh(
        planeMesh.positions, planeMesh.normals,
        planeMesh.tangents, planeMesh.uvs, planeMesh.indices);

    // Floor
    {
        EntityID e = ecs.createEntity();
        entities_.push_back(e);
        TransformComponent xform{};
        xform.position = glm::vec3(0.0f, 0.0f, 0.0f);
        xform.updateMatrix();
        ecs.addComponent(e, std::move(xform));
        MeshInstanceComponent inst{};
        inst.meshHandle = planeHandle; inst.materialIndex = 0;
        inst.setVisible(true); inst.setStatic(true);
        ecs.addComponent(e, std::move(inst));
        MaterialComponent mat{};
        mat.baseColorFactor = glm::vec4(0.6f, 0.6f, 0.6f, 1.0f);
        mat.roughnessFactor = 0.7f;
        mat.baseColorTexIndex = textures.getDefaultWhite();
        mat.normalTexIndex    = textures.getDefaultNormal();
        mat.metallicRoughnessTexIndex = textures.getDefaultMR();
        ecs.addComponent(e, std::move(mat));
    }

    // Ceiling
    {
        EntityID e = ecs.createEntity();
        entities_.push_back(e);
        TransformComponent xform{};
        xform.position = glm::vec3(0.0f, ROOM_HEIGHT, 0.0f);
        xform.scale    = glm::vec3(1.0f, -1.0f, 1.0f); // flip normal down
        xform.updateMatrix();
        ecs.addComponent(e, std::move(xform));
        MeshInstanceComponent inst{};
        inst.meshHandle = planeHandle; inst.materialIndex = 1;
        inst.setVisible(true); inst.setStatic(true);
        ecs.addComponent(e, std::move(inst));
        MaterialComponent mat{};
        mat.baseColorFactor = glm::vec4(0.8f, 0.8f, 0.8f, 1.0f);
        mat.roughnessFactor = 0.9f;
        mat.baseColorTexIndex = textures.getDefaultWhite();
        mat.normalTexIndex    = textures.getDefaultNormal();
        mat.metallicRoughnessTexIndex = textures.getDefaultMR();
        ecs.addComponent(e, std::move(mat));
    }

    // Some pillars for visual interest
    for (int ix = -2; ix <= 2; ix += 2) {
        for (int iz = -2; iz <= 2; iz += 2) {
            if (ix == 0 && iz == 0) continue;
            EntityID e = ecs.createEntity();
            entities_.push_back(e);
            TransformComponent xform{};
            xform.position = glm::vec3(ix * 8.0f, ROOM_HEIGHT * 0.5f, iz * 8.0f);
            xform.scale    = glm::vec3(0.5f, ROOM_HEIGHT * 0.5f, 0.5f);
            xform.updateMatrix();
            ecs.addComponent(e, std::move(xform));
            MeshInstanceComponent inst{};
            inst.meshHandle = cubeHandle; inst.materialIndex = 2;
            inst.setVisible(true); inst.setCastsShadows(true); inst.setStatic(true);
            ecs.addComponent(e, std::move(inst));
            MaterialComponent mat{};
            mat.baseColorFactor = glm::vec4(0.7f, 0.65f, 0.6f, 1.0f);
            mat.roughnessFactor = 0.5f;
            mat.metallicFactor  = 0.0f;
            mat.baseColorTexIndex = textures.getDefaultWhite();
            mat.normalTexIndex    = textures.getDefaultNormal();
            mat.metallicRoughnessTexIndex = textures.getDefaultMR();
            ecs.addComponent(e, std::move(mat));
        }
    }

    // --- 1024 point lights ---
    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> colorDist(0.2f, 1.0f);
    std::uniform_real_distribution<float> posDist(-ROOM_SIZE + 2.0f, ROOM_SIZE - 2.0f);
    std::uniform_real_distribution<float> heightDist(1.0f, ROOM_HEIGHT - 1.0f);
    std::uniform_real_distribution<float> radiusDist(1.0f, 8.0f);
    std::uniform_real_distribution<float> speedDist(0.2f, 1.5f);
    std::uniform_real_distribution<float> phaseDist(0.0f, 6.2832f);

    lightAnims_.reserve(LIGHT_COUNT);

    for (u32 i = 0; i < LIGHT_COUNT; ++i) {
        EntityID e = ecs.createEntity();
        entities_.push_back(e);

        glm::vec3 center(posDist(rng), 0.0f, posDist(rng));
        float y = heightDist(rng);
        float radius = radiusDist(rng);
        float speed  = speedDist(rng);
        float phase  = phaseDist(rng);

        TransformComponent xform{};
        xform.position = glm::vec3(
            center.x + radius * std::cos(phase),
            y,
            center.z + radius * std::sin(phase));
        xform.updateMatrix();
        ecs.addComponent(e, std::move(xform));

        LightComponent light{};
        light.type      = LightType::Point;
        light.color     = glm::vec3(colorDist(rng), colorDist(rng), colorDist(rng));
        light.intensity = 3.0f + colorDist(rng) * 5.0f;
        light.range     = 6.0f + colorDist(rng) * 6.0f;
        ecs.addComponent(e, std::move(light));

        lightAnims_.push_back({e, center, radius, speed, phase, y});
    }

    time_ = 0.0f;
    LOG_INFO("ManyLights: %u lights created", LIGHT_COUNT);
}

void ManyLights::update(float dt, ECS& ecs) {
    time_ += dt;
    for (auto& anim : lightAnims_) {
        float angle = anim.phase + time_ * anim.orbitSpeed;
        auto& xform = ecs.getComponent<TransformComponent>(anim.entity);
        xform.position.x = anim.center.x + anim.orbitRadius * std::cos(angle);
        xform.position.z = anim.center.z + anim.orbitRadius * std::sin(angle);
        xform.updateMatrix();
    }
}

void ManyLights::teardown(ECS& ecs, [[maybe_unused]] GpuScene& gpuScene) {
    for (EntityID e : entities_) {
        ecs.destroyEntity(e);
    }
    entities_.clear();
    lightAnims_.clear();
}

CameraSetup ManyLights::getDefaultCamera() const {
    CameraSetup cam{};
    cam.position = glm::vec3(0.0f, 3.0f, 15.0f);
    cam.target   = glm::vec3(0.0f, 3.0f, 0.0f);
    cam.distance = 15.0f;
    cam.orbit    = false; // FPS inside the room
    return cam;
}

} // namespace phosphor
