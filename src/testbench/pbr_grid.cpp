#include "testbench/pbr_grid.h"
#include "scene/ecs.h"
#include "scene/procedural.h"
#include "scene/texture_manager.h"
#include "renderer/gpu_scene.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace phosphor {

void PBRGrid::setup(ECS& ecs, GpuScene& gpuScene, TextureManager& textures) {
    textures.createDefaultTextures();

    // Single shared sphere mesh (32 slices x 16 stacks)
    auto sphereMesh = ProceduralMeshes::generateSphere(0.4f, 32, 16);
    MeshHandle sphereHandle = gpuScene.uploadMesh(
        sphereMesh.positions, sphereMesh.normals,
        sphereMesh.tangents, sphereMesh.uvs, sphereMesh.indices);

    const float spacing = 1.2f;
    const float gridOffset = spacing * static_cast<float>(GRID_SIZE - 1) * 0.5f;

    u32 materialIndex = 0;

    for (u32 row = 0; row < GRID_SIZE; ++row) {
        float metallic = static_cast<float>(row) / static_cast<float>(GRID_SIZE - 1);

        for (u32 col = 0; col < GRID_SIZE; ++col) {
            float roughness = static_cast<float>(col) / static_cast<float>(GRID_SIZE - 1);
            roughness = glm::max(roughness, 0.05f); // avoid zero roughness singularity

            EntityID entity = ecs.createEntity();
            entities_.push_back(entity);
            sphereEntities_.push_back(entity);

            TransformComponent xform{};
            xform.position = glm::vec3(
                col * spacing - gridOffset,
                row * spacing + 0.5f,
                0.0f);
            xform.updateMatrix();
            ecs.addComponent(entity, std::move(xform));

            MeshInstanceComponent inst{};
            inst.meshHandle    = sphereHandle;
            inst.materialIndex = materialIndex;
            inst.setVisible(true);
            inst.setCastsShadows(true);
            ecs.addComponent(entity, std::move(inst));

            MaterialComponent mat{};
            mat.baseColorFactor   = glm::vec4(0.9f, 0.2f, 0.2f, 1.0f);
            mat.metallicFactor    = metallic;
            mat.roughnessFactor   = roughness;
            mat.baseColorTexIndex = textures.getDefaultWhite();
            mat.normalTexIndex    = textures.getDefaultNormal();
            mat.metallicRoughnessTexIndex = textures.getDefaultMR();
            ecs.addComponent(entity, std::move(mat));

            materialIndex++;
        }
    }

    // --- Four point lights at the corners ---
    glm::vec3 lightPositions[] = {
        {-6.0f, 8.0f,  6.0f},
        { 6.0f, 8.0f,  6.0f},
        {-6.0f, 8.0f, -6.0f},
        { 6.0f, 8.0f, -6.0f},
    };
    glm::vec3 lightColors[] = {
        {1.0f, 1.0f, 1.0f},
        {1.0f, 0.9f, 0.8f},
        {0.8f, 0.9f, 1.0f},
        {1.0f, 1.0f, 0.9f},
    };

    for (int i = 0; i < 4; ++i) {
        EntityID lightEntity = ecs.createEntity();
        entities_.push_back(lightEntity);

        TransformComponent lightXform{};
        lightXform.position = lightPositions[i];
        lightXform.updateMatrix();
        ecs.addComponent(lightEntity, std::move(lightXform));

        LightComponent light{};
        light.type      = LightType::Point;
        light.color     = lightColors[i];
        light.intensity = 80.0f;
        light.range     = 30.0f;
        ecs.addComponent(lightEntity, std::move(light));
    }

    rotationAngle_ = 0.0f;
}

void PBRGrid::update(float dt, ECS& ecs) {
    rotationAngle_ += dt * 0.15f; // slow rotation of the entire grid

    for (EntityID e : sphereEntities_) {
        auto& xform = ecs.getComponent<TransformComponent>(e);
        // Rotate around the center of the grid
        glm::vec3 origPos = xform.position;
        float cosA = std::cos(rotationAngle_);
        float sinA = std::sin(rotationAngle_);
        // Rotate around Y axis relative to grid center (0, *, 0)
        xform.rotation = glm::angleAxis(rotationAngle_, glm::vec3(0.0f, 1.0f, 0.0f));
        xform.updateMatrix();
    }
}

void PBRGrid::teardown(ECS& ecs, [[maybe_unused]] GpuScene& gpuScene) {
    for (EntityID e : entities_) {
        ecs.destroyEntity(e);
    }
    entities_.clear();
    sphereEntities_.clear();
}

CameraSetup PBRGrid::getDefaultCamera() const {
    CameraSetup cam{};
    cam.position = glm::vec3(0.0f, 5.0f, 16.0f);
    cam.target   = glm::vec3(0.0f, 5.0f, 0.0f);
    cam.distance = 15.0f;
    cam.orbit    = true;
    return cam;
}

} // namespace phosphor
