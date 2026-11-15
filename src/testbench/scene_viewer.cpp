#include "testbench/scene_viewer.h"
#include "scene/ecs.h"
#include "scene/procedural.h"
#include "scene/texture_manager.h"
#include "scene/gltf_loader.h"
#include "renderer/gpu_scene.h"
#include "renderer/material_description_io.h"
#include "core/log.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <cmath>
#include <filesystem>
#include <limits>
#include <fstream>
#include <sstream>
#include <stdexcept>

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

    const auto load = [&](const std::string &path) {
        LOG_INFO("SceneViewer: loading '%s'...", path.c_str());
        GltfLoader loader(gpuScene, textures, ecs);
        // A declared rig (--corpus-rig) lights every model with the same sun: the file's own punctual lights are not imported.
        loader.setImportPunctualLights(!rig_.enabled);
        if (!loader.loadFromFile(path))
            return false;
        importedLights_ = loader.importedLightCount();
        entities_.assign(loader.createdEntities().begin(), loader.createdEntities().end());
        loadedGltf_ = true;
        loadedPath_ = path;
        LOG_INFO("SceneViewer: loaded '%s' successfully", path.c_str());
        return true;
    };
    if (!scenePath_.empty() && scenePath_ != "procedural") {
        if (!load(scenePath_))
            throw std::runtime_error("Required scene failed to load: " + scenePath_);
    } else if (scenePath_.empty()) {
        for (const char *path : kAssetPaths) {
            if (std::filesystem::exists(path) && load(path))
                break;
        }
    }

    // F15.1 (ticket 05): --material-overrides edits the loaded materials before the
    // scene store mirrors the library, so the edited records are the scene's first
    // upload (later edits go through the same setMaterialDescription path as deltas).
    if (!materialOverridesPath_.empty()) {
        if (!loadedGltf_)
            throw std::runtime_error("--material-overrides needs a loaded glTF scene: " + materialOverridesPath_);
        std::ifstream file(materialOverridesPath_, std::ios::binary);
        if (!file)
            throw std::runtime_error("cannot read material overrides " + materialOverridesPath_);
        std::stringstream json;
        json << file.rdbuf();
        // Texture paths in the file are relative to its directory.
        const std::filesystem::path dir = std::filesystem::path(materialOverridesPath_).parent_path();
        const TexturePathResolver resolve = [&](const std::string& path, bool sRGB) {
            return textures.loadTexture((dir / path).string(), sRGB);
        };
        std::string error;
        const u64 before = gpuScene.materialEditVersion();
        if (!applyMaterialOverridesToScene(json.str(), gpuScene, resolve, error))
            throw std::runtime_error("material overrides " + materialOverridesPath_ + ": " + error);
        LOG_INFO("SceneViewer: applied material overrides '%s' (%llu material edits)", materialOverridesPath_.c_str(),
                 static_cast<unsigned long long>(gpuScene.materialEditVersion() - before));
    }

    if (!loadedGltf_) {
        LOG_INFO("SceneViewer: no glTF assets found, creating procedural fallback");
        setupProceduralFallback(ecs, gpuScene, textures);
    }

    // F15.3 (ticket 22): --corpus-rig frames the loaded model from its world bounds (the mesh-local boxes through each instance's world
    // matrix) and replaces the default sun by the declared one (testbench/corpus_rig.h).
    if (rig_.enabled) {
        glm::vec3 minimum(std::numeric_limits<float>::max()), maximum(std::numeric_limits<float>::lowest());
        const auto& bounds = gpuScene.meshLocalBounds();
        u32 counted = 0;
        for (const EntityID e : entities_) {
            if (!ecs.hasComponent<MeshInstanceComponent>(e) || !ecs.hasComponent<TransformComponent>(e)) continue;
            const MeshHandle mesh = ecs.getComponent<MeshInstanceComponent>(e).meshHandle;
            if (mesh >= bounds.size()) continue;
            const glm::mat4& world = ecs.getComponent<TransformComponent>(e).worldMatrix;
            for (u32 corner = 0; corner < 8; ++corner) {
                const glm::vec3 local((corner & 1) ? bounds[mesh].maximum.x : bounds[mesh].minimum.x,
                                      (corner & 2) ? bounds[mesh].maximum.y : bounds[mesh].minimum.y,
                                      (corner & 4) ? bounds[mesh].maximum.z : bounds[mesh].minimum.z);
                const glm::vec3 point = glm::vec3(world * glm::vec4(local, 1.0f));
                minimum = glm::min(minimum, point);
                maximum = glm::max(maximum, point);
            }
            ++counted;
        }
        constexpr float fovY = glm::radians(60.0f); // the engine camera's (Engine::init)
        rigView_ = frameCorpusRig(minimum, maximum, rig_.yawDegrees, rig_.pitchDegrees, fovY, rig_.aspect);
        LOG_INFO("CORPUS_RIG: %u instances, bounds min (%.4f %.4f %.4f) max (%.4f %.4f %.4f), camera distance %.4f", counted, minimum.x, minimum.y,
                 minimum.z, maximum.x, maximum.y, maximum.z, rigView_.distance);
    }

    // --- Directional light ---
    // A file that brought its own KHR_lights_punctual lights is lit by them, not by the default sun.
    if (importedLights_ > 0) return;
    EntityID lightEntity = ecs.createEntity();
    entities_.push_back(lightEntity);

    TransformComponent lightXform{};
    lightXform.position = glm::vec3(10.0f, 20.0f, 10.0f);
    const glm::vec3 travel = rig_.enabled ? -rigView_.towardSun : glm::normalize(glm::vec3(-0.4f, -1.0f, -0.4f));
    lightXform.rotation = glm::quatLookAt(travel, std::abs(travel.y) > 0.999f ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(0.0f, 1.0f, 0.0f));
    lightXform.updateMatrix();
    ecs.addComponent(lightEntity, std::move(lightXform));

    LightComponent dirLight{};
    dirLight.type      = LightType::Directional;
    dirLight.color     = rig_.enabled ? glm::vec3(1.0f) : glm::vec3(1.0f, 0.96f, 0.92f);
    dirLight.intensity = rig_.enabled ? CorpusRig::sunIrradiance : 5.0f;
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
    if (rig_.enabled && loadedGltf_) {
        cam.position = rigView_.position;
        cam.target   = rigView_.target;
        cam.distance = rigView_.distance;
        cam.orbit    = false;
    } else if (loadedGltf_) {
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
