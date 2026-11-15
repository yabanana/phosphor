#include "testbench/glass_receiver_validation.h"
#include "renderer/gpu_scene.h"
#include "renderer/material_description.h"
#include "scene/components.h"
#include "scene/ecs.h"
#include "scene/procedural.h"
#include "scene/texture_manager.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>
#include <glm/gtc/quaternion.hpp>

namespace phosphor {
namespace {
constexpr std::array<std::string_view, 4> kAll{"glass-receiver-stack", "glass-receiver-single", "glass-btdf-sun", "glass-btdf-point"};
constexpr float pi = 3.14159265358979323846f;
constexpr glm::vec3 kTint{0.4f, 0.9f, 0.45f};

glm::quat orientLight(glm::vec3 towardLight) {
    // Same convention as the material fixtures: local -Z through the entity matrix is the direction the light travels.
    const glm::vec3 direction = -glm::normalize(towardLight), forward(0, 0, -1);
    const float cosine = std::clamp(glm::dot(forward, direction), -1.0f, 1.0f);
    if (cosine > 0.999999f) return glm::quat(1, 0, 0, 0);
    if (cosine < -0.999999f) return glm::angleAxis(pi, glm::vec3(0, 1, 0));
    return glm::angleAxis(std::acos(cosine), glm::normalize(glm::cross(forward, direction)));
}

// Camera 38 degrees above the horizontal, the sun at the mirror direction (also 38 degrees above the horizontal, on the other side of the
// vertical): the sun reflection of every horizontal pane falls on the camera.  A: a tinted-clear pane (0, 1.0, z in [-2.5, 0]); B: a rough tinted
// pane 0.5 m below it, 1.6 x 0.6 m at z in [-0.2, 0.4]: the camera sees B directly (the view ray reaches A's plane 0.64 m nearer the camera,
// outside A) while the sun ray from B reaches A's plane 0.64 m farther from the camera, well inside A.
GlassReceiverSpec stackSpec(const char* name, bool withA) {
    GlassReceiverSpec s;
    s.name = name;
    constexpr float el = 38.0f * pi / 180.0f;
    s.sun = true;
    s.sunIrradiance = 3.0f;
    s.towardSun = {0.0f, std::sin(el), -std::cos(el)};
    s.cameraTarget = {0.0f, 0.5f, 0.1f};
    s.cameraDistance = 4.5f;
    s.cameraElevationDegrees = 38.0f;
    if (withA) {
        GlassReceiverPane a;
        a.horizontal = true;
        a.center = {0.0f, 1.0f, -1.25f};
        a.size = {3.0f, 2.5f};
        a.roughness = 0.0f;
        a.baseColor = {0.6f, 0.8f, 0.9f}; // clear glass of a visible tint: every channel is attenuated by more than the 5 % gate
        s.panes.push_back(a);
    }
    GlassReceiverPane b;
    b.horizontal = true;
    b.center = {0.0f, 0.5f, 0.1f};
    b.size = {1.6f, 0.6f};
    b.roughness = 0.3f;
    b.baseColor = kTint;
    s.panes.push_back(b);
    return s;
}

// Seven vertical thin-walled panes facing the camera (+Z), the light BEHIND them: columns roughness {0.1, 0.3, 0.6}, row 0 white, row 1
// tinted; the seventh (x = 2.4) is white, roughness 0.3, transmission 0.5.
GlassReceiverSpec btdfSpec(const char* name, bool sun) {
    GlassReceiverSpec s;
    s.name = name;
    const float roughness[3] = {0.1f, 0.3f, 0.6f};
    for (int row = 0; row < 2; ++row)
        for (int c = 0; c < 3; ++c) {
            GlassReceiverPane p;
            p.center = {(float(c) - 1.0f) * 1.2f, row == 0 ? 0.6f : 1.8f, 0.0f};
            p.size = {1.0f, 1.0f};
            p.roughness = roughness[c];
            p.baseColor = row == 0 ? glm::vec3(1.0f) : kTint;
            s.panes.push_back(p);
        }
    GlassReceiverPane half;
    half.center = {2.4f, 0.6f, 0.0f};
    half.roughness = 0.3f;
    half.transmission = 0.5f;
    s.panes.push_back(half);
    if (sun) {
        s.sun = true;
        s.sunIrradiance = 3.0f;
        s.towardSun = glm::normalize(glm::vec3(0.05f, 0.08f, -1.0f));
    } else {
        s.point = true;
        s.pointPosition = {0.4f, 1.4f, -2.0f};
        s.pointIntensity = 60.0f;
    }
    s.cameraTarget = {0.0f, 1.2f, 0.0f};
    s.cameraDistance = 5.5f;
    s.cameraElevationDegrees = 0.0f;
    return s;
}

const std::vector<GlassReceiverSpec>& table() {
    static const std::vector<GlassReceiverSpec> all = {stackSpec("glass-receiver-stack", true), stackSpec("glass-receiver-single", false),
                                                       btdfSpec("glass-btdf-sun", true), btdfSpec("glass-btdf-point", false)};
    return all;
}
} // namespace

GlassReceiverValidation::GlassReceiverValidation(std::string scenario) : scenario_(std::move(scenario)) {
    if (!validScenario(scenario_)) throw std::invalid_argument("Unknown glass receiver validation scenario: " + scenario_);
    name_ = "Glass receiver validation: " + scenario_;
}
bool GlassReceiverValidation::validScenario(std::string_view scenario) { return std::find(kAll.begin(), kAll.end(), scenario) != kAll.end(); }
std::span<const std::string_view> GlassReceiverValidation::scenarios() { return kAll; }
const GlassReceiverSpec& GlassReceiverValidation::spec(std::string_view scenario) {
    for (const auto& s : table())
        if (s.name == scenario) return s;
    throw std::invalid_argument("Unknown glass receiver validation scenario: " + std::string(scenario));
}

void GlassReceiverValidation::setup(ECS& ecs, GpuScene& scene, TextureManager& textures) {
    if (!panes_.empty() || light_ != INVALID_ENTITY || anchor_ != INVALID_ENTITY) throw std::logic_error("Glass receiver fixture already set up");
    textures.createDefaultTextures(); // idempotent; materials reference none of them
    const auto& s = spec(scenario_);
    const MeshData plane = ProceduralMeshes::generatePlane(1.0f, 1.0f, 1, 1); // unit square facing +Y, scaled per instance
    const u32 planeMesh = scene.uploadMesh(plane.positions, plane.normals, plane.tangents, plane.uvs, plane.indices);
    u32 materialIndex = 0;
    for (const GlassReceiverPane& pane : s.panes) {
        const EntityID e = ecs.createEntity();
        panes_.push_back(e);
        TransformComponent transform;
        transform.position = pane.center;
        transform.rotation = pane.horizontal ? glm::quat(1, 0, 0, 0) : glm::angleAxis(pi * 0.5f, glm::vec3(1, 0, 0)); // +Y turned to +Z
        transform.scale = {pane.size.x, 1.0f, pane.size.y};
        transform.updateMatrix();
        ecs.addComponent(e, std::move(transform));
        MeshInstanceComponent instance;
        instance.meshHandle = planeMesh;
        instance.materialIndex = materialIndex++;
        instance.setVisible(true);
        instance.setCastsShadows(true);
        instance.setStatic(true);
        ecs.addComponent(e, std::move(instance));
        MaterialComponent material;
        auto description = std::make_shared<MaterialDescription>();
        description->specular.emplace();
        description->specular->ior = pane.ior;
        description->transmission.emplace();
        description->transmission->factor = pane.transmission;
        description->volume.emplace();
        description->volume->thickness = 0.0f; // thin-walled
        material.extension = std::move(description);
        material.baseColorFactor = glm::vec4(pane.baseColor, 1.0f);
        material.metallicFactor = 0.0f;
        material.roughnessFactor = pane.roughness;
        material.doubleSided = true;
        ecs.addComponent(e, std::move(material));
    }
    {   // The anchor: a black opaque cube so that the scene is not one flat pane (the reflection probe capture rejects a zero-volume scene).
        const MeshData cube = ProceduralMeshes::generateCube(AnchorHalfExtent);
        const u32 cubeMesh = scene.uploadMesh(cube.positions, cube.normals, cube.tangents, cube.uvs, cube.indices);
        anchor_ = ecs.createEntity();
        TransformComponent transform;
        transform.position = AnchorCenter;
        transform.updateMatrix();
        ecs.addComponent(anchor_, std::move(transform));
        MeshInstanceComponent instance;
        instance.meshHandle = cubeMesh;
        instance.materialIndex = materialIndex++;
        instance.setVisible(true);
        instance.setCastsShadows(true);
        instance.setStatic(true);
        ecs.addComponent(anchor_, std::move(instance));
        MaterialComponent material;
        material.baseColorFactor = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        material.metallicFactor = 0.0f;
        material.roughnessFactor = 1.0f;
        ecs.addComponent(anchor_, std::move(material));
    }
    if (s.sun || s.point) {
        light_ = ecs.createEntity();
        TransformComponent transform;
        LightComponent light;
        light.color = glm::vec3(1.0f);
        if (s.sun) {
            transform.rotation = orientLight(s.towardSun);
            light.type = LightType::Directional;
            light.intensity = s.sunIrradiance;
        } else {
            transform.position = s.pointPosition;
            light.type = LightType::Point;
            light.intensity = s.pointIntensity;
            light.range = 100.0f;
        }
        transform.updateMatrix();
        ecs.addComponent(light_, std::move(transform));
        ecs.addComponent(light_, std::move(light));
    }
}

void GlassReceiverValidation::teardown(ECS& ecs, GpuScene&) {
    for (EntityID e : panes_) ecs.destroyEntity(e);
    if (light_ != INVALID_ENTITY) ecs.destroyEntity(light_);
    if (anchor_ != INVALID_ENTITY) ecs.destroyEntity(anchor_);
    panes_.clear();
    light_ = anchor_ = INVALID_ENTITY;
}

CameraSetup GlassReceiverValidation::getDefaultCamera() const {
    const auto& s = spec(scenario_);
    const float elevation = glm::radians(s.cameraElevationDegrees);
    const glm::vec3 offset = s.cameraDistance * glm::vec3(0.0f, std::sin(elevation), std::cos(elevation));
    return {s.cameraTarget + offset, s.cameraTarget, s.cameraDistance, false};
}
} // namespace phosphor
