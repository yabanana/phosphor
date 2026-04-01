#include "testbench/culling_viz.h"
#include "scene/ecs.h"
#include "scene/procedural.h"
#include "scene/texture_manager.h"
#include "renderer/gpu_scene.h"
#include "core/log.h"

#include <glm/gtc/matrix_transform.hpp>
#include <random>

namespace phosphor {

void CullingViz::setup(ECS& ecs, GpuScene& gpuScene, TextureManager& textures) {
    textures.createDefaultTextures();
    LOG_INFO("CullingViz: generating %u buildings in city grid...", GRID_DIM * GRID_DIM);

    auto cubeMesh = ProceduralMeshes::generateCube(1.0f);
    MeshHandle cubeHandle = gpuScene.uploadMesh(
        cubeMesh.positions, cubeMesh.normals,
        cubeMesh.tangents, cubeMesh.uvs, cubeMesh.indices);

    // Ground plane
    float gridTotalSize = GRID_DIM * (BLOCK_SIZE + STREET_WIDTH);
    auto planeMesh = ProceduralMeshes::generatePlane(gridTotalSize, gridTotalSize, 1, 1);
    MeshHandle planeHandle = gpuScene.uploadMesh(
        planeMesh.positions, planeMesh.normals,
        planeMesh.tangents, planeMesh.uvs, planeMesh.indices);

    entities_.reserve(GRID_DIM * GRID_DIM + 2);

    // Ground
    {
        EntityID e = ecs.createEntity();
        entities_.push_back(e);
        TransformComponent xform{};
        xform.updateMatrix();
        ecs.addComponent(e, std::move(xform));
        MeshInstanceComponent inst{};
        inst.meshHandle = planeHandle; inst.materialIndex = 0;
        inst.setVisible(true); inst.setStatic(true);
        ecs.addComponent(e, std::move(inst));
        MaterialComponent mat{};
        mat.baseColorFactor = glm::vec4(0.3f, 0.3f, 0.32f, 1.0f);
        mat.roughnessFactor = 0.95f;
        mat.baseColorTexIndex = textures.getDefaultWhite();
        mat.normalTexIndex    = textures.getDefaultNormal();
        mat.metallicRoughnessTexIndex = textures.getDefaultMR();
        ecs.addComponent(e, std::move(mat));
    }

    // City buildings
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> heightDist(2.0f, 25.0f);
    std::uniform_real_distribution<float> colorVal(0.3f, 0.8f);
    float halfGrid = gridTotalSize * 0.5f;
    float cellSize = BLOCK_SIZE + STREET_WIDTH;

    u32 matIdx = 1;
    for (u32 iz = 0; iz < GRID_DIM; ++iz) {
        for (u32 ix = 0; ix < GRID_DIM; ++ix) {
            float height = heightDist(rng);
            float x = ix * cellSize - halfGrid + BLOCK_SIZE * 0.5f;
            float z = iz * cellSize - halfGrid + BLOCK_SIZE * 0.5f;

            EntityID e = ecs.createEntity();
            entities_.push_back(e);

            TransformComponent xform{};
            xform.position = glm::vec3(x, height * 0.5f, z);
            xform.scale    = glm::vec3(BLOCK_SIZE * 0.45f, height * 0.5f, BLOCK_SIZE * 0.45f);
            xform.updateMatrix();
            ecs.addComponent(e, std::move(xform));

            MeshInstanceComponent inst{};
            inst.meshHandle = cubeHandle;
            inst.materialIndex = matIdx;
            inst.setVisible(true);
            inst.setCastsShadows(true);
            inst.setStatic(true);
            ecs.addComponent(e, std::move(inst));

            float c = colorVal(rng);
            MaterialComponent mat{};
            mat.baseColorFactor = glm::vec4(c * 0.8f, c * 0.85f, c, 1.0f);
            mat.roughnessFactor = 0.6f + colorVal(rng) * 0.3f;
            mat.metallicFactor  = 0.0f;
            mat.baseColorTexIndex = textures.getDefaultWhite();
            mat.normalTexIndex    = textures.getDefaultNormal();
            mat.metallicRoughnessTexIndex = textures.getDefaultMR();
            ecs.addComponent(e, std::move(mat));

            matIdx++;
        }
    }

    // Directional light (sun)
    {
        EntityID e = ecs.createEntity();
        entities_.push_back(e);
        TransformComponent xform{};
        xform.position = glm::vec3(0.0f, 50.0f, 0.0f);
        xform.updateMatrix();
        ecs.addComponent(e, std::move(xform));
        LightComponent light{};
        light.type = LightType::Directional;
        light.color = glm::vec3(1.0f, 0.95f, 0.9f);
        light.intensity = 5.0f;
        ecs.addComponent(e, std::move(light));
    }

    LOG_INFO("CullingViz: setup complete (%u entities)", static_cast<u32>(entities_.size()));
}

void CullingViz::update([[maybe_unused]] float dt, [[maybe_unused]] ECS& ecs) {
    // Static scene -- culling is the focus, not animation
}

void CullingViz::teardown(ECS& ecs, [[maybe_unused]] GpuScene& gpuScene) {
    for (EntityID e : entities_) {
        ecs.destroyEntity(e);
    }
    entities_.clear();
}

CameraSetup CullingViz::getDefaultCamera() const {
    CameraSetup cam{};
    cam.position = glm::vec3(0.0f, 30.0f, 50.0f);
    cam.target   = glm::vec3(0.0f, 0.0f, 0.0f);
    cam.distance = 50.0f;
    cam.orbit    = false; // FPS mode to fly through the city
    return cam;
}

} // namespace phosphor
