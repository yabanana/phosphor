#include "testbench/culling_viz.h"
#include "scene/ecs.h"
#include "scene/procedural.h"
#include "scene/texture_manager.h"
#include "renderer/gpu_scene.h"
#include "core/log.h"

#include <glm/gtc/matrix_transform.hpp>
#include <cmath>
#include <random>

namespace phosphor {

namespace {

/// Box of half extent 1 whose faces are n x n quad grids (outward CCW faces,
/// the conventions of ProceduralMeshes::generateCube).
MeshData subdividedBox(u32 n) {
    MeshData mesh;
    struct Face {
        glm::vec3 normal, tangent;
        glm::vec3 corners[4]; // BL, BR, TR, TL
    };
    const float h = 1.0f;
    const Face faces[6] = {
        {{1, 0, 0}, {0, 0, -1}, {{h, -h, h}, {h, -h, -h}, {h, h, -h}, {h, h, h}}},
        {{-1, 0, 0}, {0, 0, 1}, {{-h, -h, -h}, {-h, -h, h}, {-h, h, h}, {-h, h, -h}}},
        {{0, 1, 0}, {1, 0, 0}, {{-h, h, h}, {h, h, h}, {h, h, -h}, {-h, h, -h}}},
        {{0, -1, 0}, {1, 0, 0}, {{-h, -h, -h}, {h, -h, -h}, {h, -h, h}, {-h, -h, h}}},
        {{0, 0, 1}, {1, 0, 0}, {{-h, -h, h}, {h, -h, h}, {h, h, h}, {-h, h, h}}},
        {{0, 0, -1}, {-1, 0, 0}, {{h, -h, -h}, {-h, -h, -h}, {-h, h, -h}, {h, h, -h}}},
    };
    for (const Face& f : faces) {
        const u32 base = static_cast<u32>(mesh.positions.size());
        for (u32 j = 0; j <= n; ++j) {
            for (u32 i = 0; i <= n; ++i) {
                const float u = static_cast<float>(i) / n, v = static_cast<float>(j) / n;
                const glm::vec3 bottom = glm::mix(f.corners[0], f.corners[1], u);
                const glm::vec3 top    = glm::mix(f.corners[3], f.corners[2], u);
                mesh.positions.push_back(glm::mix(bottom, top, v));
                mesh.normals.push_back(f.normal);
                mesh.tangents.emplace_back(f.tangent, 1.0f);
                mesh.uvs.emplace_back(u, 1.0f - v);
            }
        }
        for (u32 j = 0; j < n; ++j) {
            for (u32 i = 0; i < n; ++i) {
                const u32 v00 = base + j * (n + 1) + i, v10 = v00 + 1, v01 = v00 + (n + 1), v11 = v01 + 1;
                mesh.indices.insert(mesh.indices.end(), {v00, v10, v11, v00, v11, v01});
            }
        }
    }
    return mesh;
}

u64 nextRandom(u64& s) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
}

float unit(u64& s) { return static_cast<float>(nextRandom(s) >> 40) / static_cast<float>(1u << 24); }

constexpr double kLoop = 20.0;

} // namespace

EntityID CullingViz::addBuilding(ECS& ecs, u32 mesh, glm::vec3 position, float height, float color, float roughness) {
    EntityID e = ecs.createEntity();
    TransformComponent xform{};
    xform.position = position + glm::vec3(0.0f, height * 0.5f, 0.0f);
    xform.scale    = glm::vec3(BLOCK_SIZE * 0.45f, height * 0.5f, BLOCK_SIZE * 0.45f);
    xform.updateMatrix();
    ecs.addComponent(e, std::move(xform));
    MeshInstanceComponent inst{};
    inst.meshHandle    = mesh;
    inst.materialIndex = 1;
    inst.setVisible(true);
    inst.setCastsShadows(true);
    ecs.addComponent(e, std::move(inst));
    MaterialComponent mat{};
    mat.baseColorFactor = glm::vec4(color * 0.8f, color * 0.85f, color, 1.0f);
    mat.roughnessFactor = roughness;
    mat.metallicFactor  = 0.0f;
    ecs.addComponent(e, std::move(mat));
    return e;
}

EntityID CullingViz::addWall(ECS& ecs) {
    EntityID e = ecs.createEntity();
    TransformComponent xform{};
    xform.position = glm::vec3(6.5f, 20.0f, 30.0f);
    xform.scale    = glm::vec3(60.0f, 20.0f, 0.5f);
    xform.updateMatrix();
    ecs.addComponent(e, std::move(xform));
    MeshInstanceComponent inst{};
    inst.meshHandle    = cubeMesh_;
    inst.materialIndex = 0;
    inst.setVisible(true);
    ecs.addComponent(e, std::move(inst));
    MaterialComponent mat{};
    mat.baseColorFactor = glm::vec4(0.75f, 0.25f, 0.2f, 1.0f);
    mat.roughnessFactor = 0.8f;
    ecs.addComponent(e, std::move(mat));
    return e;
}

void CullingViz::setupScript(ECS& ecs, GpuScene& gpuScene, TextureManager& textures) {
    (void)textures;
    const MeshData box = subdividedBox(12);
    buildingMesh_ = gpuScene.uploadMesh(box.positions, box.normals, box.tangents, box.uvs, box.indices);
    const MeshData cube = ProceduralMeshes::generateCube(1.0f);
    cubeMesh_ = gpuScene.uploadMesh(cube.positions, cube.normals, cube.tangents, cube.uvs, cube.indices);
    const MeshData sphere = ProceduralMeshes::generateSphere(2.0f, 32, 16);
    sphereMesh_ = gpuScene.uploadMesh(sphere.positions, sphere.normals, sphere.tangents, sphere.uvs, sphere.indices);
    const float gridTotalSize = GRID_DIM * (BLOCK_SIZE + STREET_WIDTH);
    const MeshData plane = ProceduralMeshes::generatePlane(gridTotalSize, gridTotalSize, 1, 1);
    const u32 planeMesh = gpuScene.uploadMesh(plane.positions, plane.normals, plane.tangents, plane.uvs, plane.indices);
    {
        EntityID e = ecs.createEntity();
        entities_.push_back(e);
        TransformComponent xform{};
        xform.updateMatrix();
        ecs.addComponent(e, std::move(xform));
        MeshInstanceComponent inst{};
        inst.meshHandle = planeMesh;
        inst.materialIndex = 0;
        inst.setVisible(true);
        inst.setStatic(true);
        ecs.addComponent(e, std::move(inst));
        MaterialComponent mat{};
        mat.baseColorFactor = glm::vec4(0.3f, 0.3f, 0.32f, 1.0f);
        mat.roughnessFactor = 0.95f;
        ecs.addComponent(e, std::move(mat));
    }
    u64 rng = 42;
    const float halfGrid = gridTotalSize * 0.5f;
    const float cellSize = BLOCK_SIZE + STREET_WIDTH;
    buildings_.reserve(GRID_DIM * GRID_DIM);
    cells_.clear();
    cells_.reserve(GRID_DIM * GRID_DIM);
    for (u32 iz = 0; iz < GRID_DIM; ++iz) {
        for (u32 ix = 0; ix < GRID_DIM; ++ix) {
            const float height = 2.0f + unit(rng) * 23.0f;
            const glm::vec3 p(ix * cellSize - halfGrid + BLOCK_SIZE * 0.5f, 0.0f, iz * cellSize - halfGrid + BLOCK_SIZE * 0.5f);
            buildings_.push_back(addBuilding(ecs, buildingMesh_, p, height, 0.3f + unit(rng) * 0.5f, 0.6f + unit(rng) * 0.3f));
            cells_.push_back(p);
        }
    }
    wall_ = addWall(ecs);
    {
        sphere_ = ecs.createEntity();
        TransformComponent xform{};
        xform.position = glm::vec3(-150.0f, 3.0f, 40.0f);
        xform.updateMatrix();
        ecs.addComponent(sphere_, std::move(xform));
        MeshInstanceComponent inst{};
        inst.meshHandle = sphereMesh_;
        inst.materialIndex = 0;
        inst.setVisible(true);
        ecs.addComponent(sphere_, std::move(inst));
        MaterialComponent mat{};
        mat.baseColorFactor = glm::vec4(0.95f, 0.85f, 0.2f, 1.0f);
        mat.roughnessFactor = 0.3f;
        ecs.addComponent(sphere_, std::move(mat));
    }
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
    time_      = 0.0;
    nextChurn_ = 0.25;
    churnRng_  = 0x9E3779B97F4A7C15ull;
    lastSegment_ = -1;
    LOG_INFO("CullingViz script: %zu buildings (1728 triangles each), wall, fast sphere, churn 20 / 0.25 s",
             buildings_.size());
}

void CullingViz::updateScript(float dt, ECS& ecs) {
    time_ += dt;
    const double phase = std::fmod(time_, kLoop);
    // The wall: present in [0, 3) and [10, 13) of the loop.
    const bool wallWanted = phase < 3.0 || (phase >= 10.0 && phase < 13.0);
    if (wallWanted && wall_ == INVALID_ENTITY) wall_ = addWall(ecs);
    if (!wallWanted && wall_ != INVALID_ENTITY) {
        ecs.destroyEntity(wall_);
        wall_ = INVALID_ENTITY;
    }
    // The fast sphere: 150 units/s along x in front of the first street.
    if (sphere_ != INVALID_ENTITY) {
        TransformComponent& t = ecs.getComponent<TransformComponent>(sphere_);
        t.position = glm::vec3(-150.0f + 300.0f * static_cast<float>(std::fmod(time_ * 0.5, 1.0)), 3.0f, 40.0f);
        t.updateMatrix();
    }
    // Churn: destroy 20 buildings and re-create them in their cells with a
    // new height (recycled entities, reused slots, no overlap).
    while (time_ >= nextChurn_ && !buildings_.empty()) {
        nextChurn_ += 0.25;
        for (u32 k = 0; k < 20; ++k) {
            const size_t i = static_cast<size_t>(nextRandom(churnRng_) % buildings_.size());
            ecs.destroyEntity(buildings_[i]);
            buildings_[i] = addBuilding(ecs, buildingMesh_, cells_[i], 2.0f + unit(churnRng_) * 23.0f,
                                        0.3f + unit(churnRng_) * 0.5f, 0.7f);
        }
    }
}

bool CullingViz::scriptedCamera(double t, glm::vec3& position, glm::vec3& target, bool& cut) const {
    if (!script_) return false;
    const double phase = std::fmod(t, kLoop);
    int segment = 0;
    const float streetX = 6.5f, eye = 2.5f;
    if (phase < 6.0) {
        // Towards the wall, down the street x = 6.5 (z from 70 to -30).
        const float z = 70.0f - static_cast<float>(phase) * 16.0f;
        position = glm::vec3(streetX, eye, z);
        target   = position + glm::vec3(0.0f, -0.05f, -1.0f);
        segment  = 0;
    } else if (phase < 12.0) {
        // Another district, along the street z = -193.5 towards -x.
        const float x = 300.0f - static_cast<float>(phase - 6.0) * 20.0f;
        position = glm::vec3(x, eye, -193.5f);
        target   = position + glm::vec3(-1.0f, 0.0f, 0.1f);
        segment  = 1;
    } else if (phase < 16.0) {
        // Rising wide view over the centre.
        const float k = static_cast<float>(phase - 12.0) / 4.0f;
        position = glm::vec3(-120.0f, 5.0f + 75.0f * k, 150.0f);
        target   = glm::vec3(0.0f, 0.0f, 0.0f);
        segment  = 2;
    } else {
        // 360 degree pan at street level.
        const float a = static_cast<float>(phase - 16.0) / 4.0f * 6.2831853f;
        position = glm::vec3(-89.5f, eye, -89.5f);
        target   = position + glm::vec3(std::cos(a), -0.02f, std::sin(a));
        segment  = 3;
    }
    cut = lastSegment_ >= 0 && segment != lastSegment_;
    lastSegment_ = segment;
    return true;
}

void CullingViz::setup(ECS& ecs, GpuScene& gpuScene, TextureManager& textures) {
    textures.createDefaultTextures();
    if (script_) {
        setupScript(ecs, gpuScene, textures);
        return;
    }
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
    // Static scene -- culling is the focus, not animation (except the F6 script)
    if (script_) updateScript(dt, ecs);
}

void CullingViz::teardown(ECS& ecs, [[maybe_unused]] GpuScene& gpuScene) {
    for (EntityID e : entities_) {
        ecs.destroyEntity(e);
    }
    entities_.clear();
    for (EntityID e : buildings_) ecs.destroyEntity(e);
    buildings_.clear();
    for (EntityID* e : {&wall_, &sphere_}) {
        if (*e != INVALID_ENTITY) ecs.destroyEntity(*e);
        *e = INVALID_ENTITY;
    }
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
