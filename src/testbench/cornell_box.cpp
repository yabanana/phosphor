#include "testbench/cornell_box.h"
#include "scene/ecs.h"
#include "scene/procedural.h"
#include "scene/texture_manager.h"
#include "renderer/gpu_scene.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace phosphor {

void CornellBox::setup(ECS& ecs, GpuScene& gpuScene, TextureManager& textures) {
    textures.createDefaultTextures();

    auto planeMesh = ProceduralMeshes::generatePlane(5.0f, 5.0f, 1, 1);
    MeshHandle planeHandle = gpuScene.uploadMesh(
        planeMesh.positions, planeMesh.normals,
        planeMesh.tangents, planeMesh.uvs, planeMesh.indices);

    auto cubeMesh = ProceduralMeshes::generateCube(1.0f);
    MeshHandle cubeHandle = gpuScene.uploadMesh(
        cubeMesh.positions, cubeMesh.normals,
        cubeMesh.tangents, cubeMesh.uvs, cubeMesh.indices);

    // Material indices: 0=white, 1=red, 2=green, 3=emissive, 4=white-box

    auto addWall = [&](glm::vec3 pos, glm::quat rot, u32 matIdx,
                       glm::vec4 color) -> EntityID {
        EntityID e = ecs.createEntity();
        entities_.push_back(e);

        TransformComponent xform{};
        xform.position = pos;
        xform.rotation = rot;
        xform.updateMatrix();
        ecs.addComponent(e, std::move(xform));

        MeshInstanceComponent inst{};
        inst.meshHandle = planeHandle;
        inst.materialIndex = matIdx;
        inst.setVisible(true);
        inst.setStatic(true);
        ecs.addComponent(e, std::move(inst));

        MaterialComponent mat{};
        mat.baseColorFactor = color;
        mat.metallicFactor  = 0.0f;
        mat.roughnessFactor = 0.95f;
        mat.baseColorTexIndex = textures.getDefaultWhite();
        mat.normalTexIndex    = textures.getDefaultNormal();
        mat.metallicRoughnessTexIndex = textures.getDefaultMR();
        ecs.addComponent(e, std::move(mat));

        return e;
    };

    const float S = 2.5f; // half-extent of the box
    glm::quat identity{1.0f, 0.0f, 0.0f, 0.0f};

    // Floor (white, facing up -- default for plane)
    addWall({0.0f, 0.0f, 0.0f}, identity, 0,
            {0.73f, 0.73f, 0.73f, 1.0f});

    // Ceiling (white, facing down)
    glm::quat ceilRot = glm::angleAxis(glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    addWall({0.0f, S * 2.0f, 0.0f}, ceilRot, 0,
            {0.73f, 0.73f, 0.73f, 1.0f});

    // Back wall (white, facing +Z)
    glm::quat backRot = glm::angleAxis(glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    addWall({0.0f, S, -S}, backRot, 0,
            {0.73f, 0.73f, 0.73f, 1.0f});

    // Left wall (red, facing +X)
    glm::quat leftRot = glm::angleAxis(glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    addWall({-S, S, 0.0f}, leftRot, 1,
            {0.65f, 0.05f, 0.05f, 1.0f});

    // Right wall (green, facing -X)
    glm::quat rightRot = glm::angleAxis(glm::radians(-90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    addWall({S, S, 0.0f}, rightRot, 2,
            {0.12f, 0.45f, 0.15f, 1.0f});

    // --- Tall box (slightly rotated) ---
    {
        tallBoxEntity_ = ecs.createEntity();
        entities_.push_back(tallBoxEntity_);

        TransformComponent xform{};
        xform.position = glm::vec3(1.0f, 1.5f, -0.8f);
        xform.scale    = glm::vec3(0.6f, 1.5f, 0.6f);
        xform.rotation = glm::angleAxis(glm::radians(18.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        xform.updateMatrix();
        ecs.addComponent(tallBoxEntity_, std::move(xform));

        MeshInstanceComponent inst{};
        inst.meshHandle = cubeHandle;
        inst.materialIndex = 4;
        inst.setVisible(true);
        inst.setCastsShadows(true);
        ecs.addComponent(tallBoxEntity_, std::move(inst));

        MaterialComponent mat{};
        mat.baseColorFactor = glm::vec4(0.73f, 0.73f, 0.73f, 1.0f);
        mat.roughnessFactor = 0.7f;
        mat.baseColorTexIndex = textures.getDefaultWhite();
        mat.normalTexIndex    = textures.getDefaultNormal();
        mat.metallicRoughnessTexIndex = textures.getDefaultMR();
        ecs.addComponent(tallBoxEntity_, std::move(mat));
    }

    // --- Short box ---
    {
        EntityID e = ecs.createEntity();
        entities_.push_back(e);

        TransformComponent xform{};
        xform.position = glm::vec3(-1.0f, 0.75f, 0.5f);
        xform.scale    = glm::vec3(0.75f, 0.75f, 0.75f);
        xform.rotation = glm::angleAxis(glm::radians(-15.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        xform.updateMatrix();
        ecs.addComponent(e, std::move(xform));

        MeshInstanceComponent inst{};
        inst.meshHandle = cubeHandle;
        inst.materialIndex = 4;
        inst.setVisible(true);
        inst.setCastsShadows(true);
        ecs.addComponent(e, std::move(inst));

        MaterialComponent mat{};
        mat.baseColorFactor = glm::vec4(0.73f, 0.73f, 0.73f, 1.0f);
        mat.roughnessFactor = 0.7f;
        mat.baseColorTexIndex = textures.getDefaultWhite();
        mat.normalTexIndex    = textures.getDefaultNormal();
        mat.metallicRoughnessTexIndex = textures.getDefaultMR();
        ecs.addComponent(e, std::move(mat));
    }

    // --- Ceiling light (bright point light simulating the classic area light) ---
    {
        EntityID e = ecs.createEntity();
        entities_.push_back(e);

        TransformComponent xform{};
        xform.position = glm::vec3(0.0f, S * 2.0f - 0.05f, 0.0f);
        xform.updateMatrix();
        ecs.addComponent(e, std::move(xform));

        LightComponent light{};
        light.type      = LightType::Point;
        light.color     = glm::vec3(1.0f, 0.93f, 0.78f);
        light.intensity = 30.0f;
        light.range     = 12.0f;
        ecs.addComponent(e, std::move(light));
    }

    rotationAngle_ = 0.0f;
}

void CornellBox::update(float dt, ECS& ecs) {
    // Optionally rotate the tall box slowly
    if (tallBoxEntity_ == INVALID_ENTITY) return;

    rotationAngle_ += dt * 0.3f;
    auto& xform = ecs.getComponent<TransformComponent>(tallBoxEntity_);
    xform.rotation = glm::angleAxis(18.0f * glm::radians(1.0f) + rotationAngle_,
                                     glm::vec3(0.0f, 1.0f, 0.0f));
    xform.updateMatrix();
}

void CornellBox::teardown(ECS& ecs, [[maybe_unused]] GpuScene& gpuScene) {
    for (EntityID e : entities_) {
        ecs.destroyEntity(e);
    }
    entities_.clear();
    tallBoxEntity_ = INVALID_ENTITY;
}

CameraSetup CornellBox::getDefaultCamera() const {
    CameraSetup cam{};
    cam.position = glm::vec3(0.0f, 2.5f, 7.5f);
    cam.target   = glm::vec3(0.0f, 2.5f, 0.0f);
    cam.distance = 7.5f;
    cam.orbit    = false;
    return cam;
}

} // namespace phosphor
