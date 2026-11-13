#include "testbench/material_validation.h"
#include "renderer/gpu_scene.h"
#include "scene/components.h"
#include "scene/ecs.h"
#include "scene/procedural.h"
#include "scene/texture_manager.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <glm/gtc/quaternion.hpp>

namespace phosphor {
namespace {
constexpr std::array<std::string_view, 4> kScenarios{"base-grid-sun", "base-grid-env", "furnace-base",
                                                     "roughness-extremes"};
constexpr float pi = 3.14159265358979323846f;

glm::quat orientLight(glm::vec3 towardLight) {
    // Same convention as the lighting fixtures: local -Z through the entity
    // matrix is the direction the light travels; the renderer negates it.
    const glm::vec3 direction = -glm::normalize(towardLight), forward(0, 0, -1);
    const float cosine = std::clamp(glm::dot(forward, direction), -1.0f, 1.0f);
    if (cosine > 0.999999f) return glm::quat(1, 0, 0, 0);
    if (cosine < -0.999999f) return glm::angleAxis(pi, glm::vec3(0, 1, 0));
    return glm::angleAxis(std::acos(cosine), glm::normalize(glm::cross(forward, direction)));
}

const std::vector<MaterialScenarioSpec>& specs() {
    // roughness 0.1 and 0.2 have alpha < 0.05: the sun disk uses the declared
    // spiral quadrature there (the reference's quadrature mask).
    static const std::vector<float> roughness{0.1f, 0.2f, 0.35f, 0.5f, 0.7f, 0.85f, 1.0f};
    static const std::vector<float> metallic{0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
    static const glm::vec3 copper(0.8f, 0.35f, 0.15f);
    static const glm::vec3 towardSun = glm::normalize(glm::vec3(-0.35f, 0.5f, 0.8f));
    static const std::vector<MaterialScenarioSpec> all{
        {"base-grid-sun", roughness, metallic, copper, true, 3.0f, towardSun, 0.0f},
        {"base-grid-env", roughness, metallic, copper, false, 0.0f, towardSun, 0.5f},
        {"furnace-base", roughness, metallic, glm::vec3(1.0f), false, 0.0f, towardSun, 1.0f},
        // Roughness 0 is the declared clamp (0.04); 1 is the upper bound.
        {"roughness-extremes", {0.0f, 1.0f}, {0.0f, 0.5f, 1.0f}, copper, true, 3.0f, towardSun, 0.3f},
    };
    return all;
}
}

MaterialValidation::MaterialValidation(std::string scenario) : scenario_(std::move(scenario)) {
    if (!validScenario(scenario_)) throw std::invalid_argument("Unknown material validation scenario: " + scenario_);
    name_ = "Material validation: " + scenario_;
}
bool MaterialValidation::validScenario(std::string_view scenario) {
    return std::find(kScenarios.begin(), kScenarios.end(), scenario) != kScenarios.end();
}
std::span<const std::string_view> MaterialValidation::scenarios() { return kScenarios; }
const MaterialScenarioSpec& MaterialValidation::spec(std::string_view scenario) {
    for (const auto& s : specs())
        if (s.name == scenario) return s;
    throw std::invalid_argument("Unknown material validation scenario: " + std::string(scenario));
}
float MaterialValidation::environmentRadiance(std::string_view scenario) {
    return validScenario(scenario) ? spec(scenario).environment : 0.0f;
}
glm::vec3 MaterialValidation::sphereCenter(const MaterialScenarioSpec& s, u32 column, u32 row) {
    const float x0 = -0.5f * Spacing * float(s.roughness.size() - 1);
    return {x0 + Spacing * float(column), Spacing * float(row), 0.0f};
}

void MaterialValidation::setup(ECS& ecs, GpuScene& scene, TextureManager& textures) {
    if (!spheres_.empty()) throw std::logic_error("Material fixture already set up");
    textures.createDefaultTextures(); // idempotent; materials reference none of them
    const auto& s = spec(scenario_);
    const auto sphere = ProceduralMeshes::generateSphere(SphereRadius, SphereSlices, SphereStacks);
    const u32 handle = scene.uploadMesh(sphere.positions, sphere.normals, sphere.tangents, sphere.uvs, sphere.indices);
    u32 materialIndex = 0;
    for (u32 row = 0; row < s.metallic.size(); ++row)
        for (u32 column = 0; column < s.roughness.size(); ++column) {
            const EntityID e = ecs.createEntity();
            spheres_.push_back(e);
            TransformComponent transform;
            transform.position = sphereCenter(s, column, row);
            transform.updateMatrix();
            ecs.addComponent(e, std::move(transform));
            MeshInstanceComponent instance;
            instance.meshHandle = handle;
            instance.materialIndex = materialIndex++;
            instance.setVisible(true);
            instance.setCastsShadows(true);
            instance.setStatic(true);
            ecs.addComponent(e, std::move(instance));
            MaterialComponent material;
            material.baseColorFactor = glm::vec4(s.baseColor, 1.0f);
            material.metallicFactor = s.metallic[row];
            material.roughnessFactor = s.roughness[column];
            // INVALID texture indices intentionally use the exact shader
            // fallbacks: the default normal/MR textures would perturb factors.
            ecs.addComponent(e, std::move(material));
        }
    if (s.sun) {
        sun_ = ecs.createEntity();
        TransformComponent transform;
        transform.rotation = orientLight(s.towardSun);
        transform.updateMatrix();
        ecs.addComponent(sun_, std::move(transform));
        LightComponent light;
        light.type = LightType::Directional;
        light.color = glm::vec3(1.0f);
        light.intensity = s.sunIrradiance;
        ecs.addComponent(sun_, std::move(light));
    }
}

void MaterialValidation::teardown(ECS& ecs, GpuScene&) {
    for (EntityID e : spheres_) ecs.destroyEntity(e);
    if (sun_ != INVALID_ENTITY) ecs.destroyEntity(sun_);
    spheres_.clear();
    sun_ = INVALID_ENTITY;
}

CameraSetup MaterialValidation::getDefaultCamera() const {
    // 16:9 frame, 60 degree vertical field of view (engine default): the
    // distance keeps the whole grid plus a margin in view.
    const auto& s = spec(scenario_);
    const float centreY = 0.5f * Spacing * float(s.metallic.size() - 1);
    const float distance = s.roughness.size() > 2 ? 5.5f : 4.0f;
    return {{0.0f, centreY, distance}, {0.0f, centreY, 0.0f}, distance, false};
}
} // namespace phosphor
