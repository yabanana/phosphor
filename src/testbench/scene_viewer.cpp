#include "testbench/scene_viewer.h"
#include "scene/ecs.h"
#include "scene/procedural.h"
#include "scene/texture_manager.h"
#include "scene/gltf_loader.h"
#include "renderer/gpu_scene.h"
#include "core/log.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <filesystem>

namespace phosphor {

void SceneViewer::setup(ECS& ecs, GpuScene& gpuScene, TextureManager& textures) {
    textures.createDefaultTextures();

    // Try to load known glTF assets in order of preference
    static const char* kAssetPaths[] = {
        "assets/Sponza.gltf",
        "assets/sponza/Sponza.gltf",
        "assets/DamagedHelmet.glb",
        "assets/damaged_helmet/DamagedHelmet.glb",
        "assets/FlightHelmet.gltf",
    };

    loadedGltf_ = false;

    for (const char* path : kAssetPaths) {
        if (std::filesystem::exists(path)) {
            LOG_INFO("SceneViewer: loading '%s'...", path);
            GltfLoader loader(gpuScene, textures, ecs);
            if (loader.loadFromFile(path)) {
                loadedGltf_ = true;
                LOG_INFO("SceneViewer: loaded '%s' successfully", path);
                break;
            } else {
                LOG_WARN("SceneViewer: failed to load '%s'", path);
            }
        }
    }

    if (!loadedGltf_) {
        LOG_INFO("SceneViewer: no glTF assets found, creating procedural fallback");
        setupProceduralFallback(ecs, gpuScene, textures);
    }

    // --- Directional light ---
    EntityID lightEntity = ecs.createEntity();
    entities_.push_back(lightEntity);

    TransformComponent lightXform{};
    lightXform.position = glm::vec3(10.0f, 20.0f, 10.0f);
    lightXform.rotation = glm::quatLookAt(
        glm::normalize(glm::vec3(-0.4f, -1.0f, -0.4f)),
        glm::vec3(0.0f, 1.0f, 0.0f));
    lightXform.updateMatrix();
    ecs.addComponent(lightEntity, std::move(lightXform));

    LightComponent dirLight{};
    dirLight.type      = LightType::Directional;
    dirLight.color     = glm::vec3(1.0f, 0.96f, 0.92f);
    dirLight.intensity = 5.0f;
    ecs.addComponent(lightEntity, std::move(dirLight));
}

void SceneViewer::update([[maybe_unused]] float dt, [[maybe_unused]] ECS& ecs) {
    // Static scene, no animations
}

void SceneViewer::teardown(ECS& ecs, [[maybe_unused]] GpuScene& gpuScene) {
    for (EntityID e : entities_) {
        ecs.destroyEntity(e);
    }
    entities_.clear();
}

CameraSetup SceneViewer::getDefaultCamera() const {
    CameraSetup cam{};
    if (loadedGltf_) {
        // Sponza-style interior
        cam.position = glm::vec3(0.0f, 3.0f, 0.0f);
        cam.target   = glm::vec3(0.0f, 3.0f, -5.0f);
        cam.distance = 10.0f;
        cam.orbit    = false;
    } else {
        // Procedural fallback
        cam.position = glm::vec3(0.0f, 4.0f, 10.0f);
        cam.target   = glm::vec3(0.0f, 1.0f, 0.0f);
        cam.distance = 10.0f;
        cam.orbit    = false;
    }
    return cam;
}

void SceneViewer::setupProceduralFallback(ECS& ecs, GpuScene& gpuScene, TextureManager& textures) {
    // Generate a few cubes and spheres as a fallback scene
    auto cubeMesh = ProceduralMeshes::generateCube(1.0f);
    MeshHandle cubeHandle = gpuScene.uploadMesh(
        cubeMesh.positions, cubeMesh.normals,
        cubeMesh.tangents, cubeMesh.uvs, cubeMesh.indices);

    auto sphereMesh = ProceduralMeshes::generateSphere(0.8f, 32, 16);
    MeshHandle sphereHandle = gpuScene.uploadMesh(
        sphereMesh.positions, sphereMesh.normals,
        sphereMesh.tangents, sphereMesh.uvs, sphereMesh.indices);

    // Ground plane
    auto planeMesh = ProceduralMeshes::generatePlane(30.0f, 30.0f, 1, 1);
    MeshHandle planeHandle = gpuScene.uploadMesh(
        planeMesh.positions, planeMesh.normals,
        planeMesh.tangents, planeMesh.uvs, planeMesh.indices);

    // -- Ground --
    {
        EntityID e = ecs.createEntity();
        entities_.push_back(e);
        TransformComponent xform{};
        xform.updateMatrix();
        ecs.addComponent(e, std::move(xform));
        MeshInstanceComponent inst{};
        inst.meshHandle = planeHandle;
        inst.materialIndex = 0;
        inst.setVisible(true);
        inst.setStatic(true);
        ecs.addComponent(e, std::move(inst));
        MaterialComponent mat{};
        mat.baseColorFactor = glm::vec4(0.5f, 0.5f, 0.5f, 1.0f);
        mat.roughnessFactor = 0.9f;
        mat.baseColorTexIndex = textures.getDefaultWhite();
        mat.normalTexIndex    = textures.getDefaultNormal();
        mat.metallicRoughnessTexIndex = textures.getDefaultMR();
        ecs.addComponent(e, std::move(mat));
    }

    // -- Cubes --
    glm::vec3 cubePositions[] = {
        {-3.0f, 1.0f, 0.0f},
        { 0.0f, 1.0f, -3.0f},
        { 3.0f, 1.5f, 0.0f},
    };
    glm::vec4 cubeColors[] = {
        {0.8f, 0.2f, 0.2f, 1.0f},
        {0.2f, 0.8f, 0.2f, 1.0f},
        {0.2f, 0.2f, 0.8f, 1.0f},
    };

    for (int i = 0; i < 3; ++i) {
        EntityID e = ecs.createEntity();
        entities_.push_back(e);
        TransformComponent xform{};
        xform.position = cubePositions[i];
        xform.updateMatrix();
        ecs.addComponent(e, std::move(xform));
        MeshInstanceComponent inst{};
        inst.meshHandle = cubeHandle;
        inst.materialIndex = static_cast<u32>(i + 1);
        inst.setVisible(true);
        inst.setCastsShadows(true);
        ecs.addComponent(e, std::move(inst));
        MaterialComponent mat{};
        mat.baseColorFactor = cubeColors[i];
        mat.roughnessFactor = 0.5f;
        mat.metallicFactor  = 0.1f;
        mat.baseColorTexIndex = textures.getDefaultWhite();
        mat.normalTexIndex    = textures.getDefaultNormal();
        mat.metallicRoughnessTexIndex = textures.getDefaultMR();
        ecs.addComponent(e, std::move(mat));
    }

    // -- Spheres --
    glm::vec3 spherePositions[] = {
        {-1.5f, 0.8f, 3.0f},
        { 1.5f, 0.8f, 3.0f},
    };

    for (int i = 0; i < 2; ++i) {
        EntityID e = ecs.createEntity();
        entities_.push_back(e);
        TransformComponent xform{};
        xform.position = spherePositions[i];
        xform.updateMatrix();
        ecs.addComponent(e, std::move(xform));
        MeshInstanceComponent inst{};
        inst.meshHandle = sphereHandle;
        inst.materialIndex = static_cast<u32>(i + 4);
        inst.setVisible(true);
        inst.setCastsShadows(true);
        ecs.addComponent(e, std::move(inst));
        MaterialComponent mat{};
        mat.baseColorFactor = glm::vec4(0.9f, 0.85f, 0.7f, 1.0f);
        mat.metallicFactor  = 1.0f;
        mat.roughnessFactor = 0.2f + i * 0.4f;
        mat.baseColorTexIndex = textures.getDefaultWhite();
        mat.normalTexIndex    = textures.getDefaultNormal();
        mat.metallicRoughnessTexIndex = textures.getDefaultMR();
        ecs.addComponent(e, std::move(mat));
    }
}

} // namespace phosphor
