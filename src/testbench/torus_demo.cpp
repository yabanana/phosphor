#include "testbench/torus_demo.h"
#include "scene/ecs.h"
#include "scene/procedural.h"
#include "scene/texture_manager.h"
#include "renderer/gpu_scene.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace phosphor {

void TorusDemo::setup(ECS& ecs, GpuScene& gpuScene, TextureManager& textures) {
    textures.createDefaultTextures();

    // --- Torus mesh (64 major x 32 minor segments) ---
    auto torusMesh = ProceduralMeshes::generateTorus(1.5f, 0.5f, 64, 32);
    MeshHandle torusHandle = gpuScene.uploadMesh(
        torusMesh.positions, torusMesh.normals,
        torusMesh.tangents, torusMesh.uvs, torusMesh.indices);

    // Gold material: metallic = 1.0, roughness = 0.3
    u32 goldMaterialIdx = 0; // material index in the GPU material array

    // Torus entity
    torusEntity_ = ecs.createEntity();
    entities_.push_back(torusEntity_);

    TransformComponent torusXform{};
    torusXform.position = glm::vec3(0.0f, 1.5f, 0.0f);
    torusXform.updateMatrix();
    ecs.addComponent(torusEntity_, std::move(torusXform));

    MeshInstanceComponent torusInst{};
    torusInst.meshHandle    = torusHandle;
    torusInst.materialIndex = goldMaterialIdx;
    torusInst.setVisible(true);
    torusInst.setCastsShadows(true);
    ecs.addComponent(torusEntity_, std::move(torusInst));

    MaterialComponent goldMat{};
    goldMat.baseColorFactor   = glm::vec4(1.0f, 0.766f, 0.336f, 1.0f); // gold
    goldMat.metallicFactor    = 1.0f;
    goldMat.roughnessFactor   = 0.3f;
    goldMat.baseColorTexIndex = textures.getDefaultWhite();
    goldMat.normalTexIndex    = textures.getDefaultNormal();
    goldMat.metallicRoughnessTexIndex = textures.getDefaultMR();
    ecs.addComponent(torusEntity_, std::move(goldMat));

    // --- Ground plane ---
    auto planeMesh = ProceduralMeshes::generatePlane(20.0f, 20.0f, 1, 1);
    MeshHandle planeHandle = gpuScene.uploadMesh(
        planeMesh.positions, planeMesh.normals,
        planeMesh.tangents, planeMesh.uvs, planeMesh.indices);

    EntityID planeEntity = ecs.createEntity();
    entities_.push_back(planeEntity);

    TransformComponent planeXform{};
    planeXform.position = glm::vec3(0.0f, 0.0f, 0.0f);
    planeXform.updateMatrix();
    ecs.addComponent(planeEntity, std::move(planeXform));

    MeshInstanceComponent planeInst{};
    planeInst.meshHandle    = planeHandle;
    planeInst.materialIndex = 1;
    planeInst.setVisible(true);
    planeInst.setCastsShadows(false);
    planeInst.setStatic(true);
    ecs.addComponent(planeEntity, std::move(planeInst));

    MaterialComponent planeMat{};
    planeMat.baseColorFactor   = glm::vec4(0.4f, 0.4f, 0.4f, 1.0f);
    planeMat.metallicFactor    = 0.0f;
    planeMat.roughnessFactor   = 0.8f;
    planeMat.baseColorTexIndex = textures.getDefaultWhite();
    planeMat.normalTexIndex    = textures.getDefaultNormal();
    planeMat.metallicRoughnessTexIndex = textures.getDefaultMR();
    ecs.addComponent(planeEntity, std::move(planeMat));

    // --- Directional light ---
    EntityID lightEntity = ecs.createEntity();
    entities_.push_back(lightEntity);

    TransformComponent lightXform{};
    lightXform.position = glm::vec3(5.0f, 10.0f, 5.0f);
    lightXform.rotation = glm::quatLookAt(
        glm::normalize(glm::vec3(-0.5f, -1.0f, -0.5f)),
        glm::vec3(0.0f, 1.0f, 0.0f));
    lightXform.updateMatrix();
    ecs.addComponent(lightEntity, std::move(lightXform));

    LightComponent dirLight{};
    dirLight.type      = LightType::Directional;
    dirLight.color     = glm::vec3(1.0f, 0.95f, 0.9f);
    dirLight.intensity = 5.0f;
    ecs.addComponent(lightEntity, std::move(dirLight));

    rotationAngle_ = 0.0f;
}

void TorusDemo::update(float dt, ECS& ecs) {
    if (torusEntity_ == INVALID_ENTITY) return;

    rotationAngle_ += dt * 0.5f; // radians per second
    auto& xform = ecs.getComponent<TransformComponent>(torusEntity_);
    xform.rotation = glm::angleAxis(rotationAngle_, glm::vec3(0.0f, 1.0f, 0.0f));
    xform.updateMatrix();
}

void TorusDemo::teardown(ECS& ecs, [[maybe_unused]] GpuScene& gpuScene) {
    for (EntityID e : entities_) {
        ecs.destroyEntity(e);
    }
    entities_.clear();
    torusEntity_ = INVALID_ENTITY;
}

CameraSetup TorusDemo::getDefaultCamera() const {
    CameraSetup cam{};
    cam.position = glm::vec3(0.0f, 3.0f, 6.0f);
    cam.target   = glm::vec3(0.0f, 1.0f, 0.0f);
    cam.distance = 5.0f;
    cam.orbit    = true;
    return cam;
}

} // namespace phosphor
