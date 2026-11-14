#include "testbench/transmissive_shadow_validation.h"
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
// The names are mirrored by tools/f15_check.py SCENARIOS (tests/test_transmissive_shadow_validation.cpp compares both).
// transmissive-shadow-curved (the last) is the report-only curved-glass case.
constexpr std::array<std::string_view, 5> kAll{"transmissive-shadow-pane-sun", "transmissive-shadow-water-sun", "transmissive-shadow-volume-pane",
                                               "transmissive-shadow-point", "transmissive-shadow-curved"};
constexpr float pi = 3.14159265358979323846f;

glm::quat orientLight(glm::vec3 towardLight) {
    // Same convention as the material fixtures: local -Z through the entity matrix is the direction the light travels.
    const glm::vec3 direction = -glm::normalize(towardLight), forward(0, 0, -1);
    const float cosine = std::clamp(glm::dot(forward, direction), -1.0f, 1.0f);
    if (cosine > 0.999999f) return glm::quat(1, 0, 0, 0);
    if (cosine < -0.999999f) return glm::angleAxis(pi, glm::vec3(0, 1, 0));
    return glm::angleAxis(std::acos(cosine), glm::normalize(glm::cross(forward, direction)));
}

// A horizontal thin-walled pane (one quad facing +Y, double sided: it is seen from below by the shadow rays).
ShadowCaster thinPane(float x, float side, glm::vec3 tint) {
    ShadowCaster c;
    c.glass.shape = GlassObject::Shape::Pane;
    c.glass.center = {x, TransmissiveShadowValidation::CasterHeight, 0.0f};
    c.glass.scale = {side, 1.0f, side};
    c.glass.baseColor = tint;
    c.glass.thickness = 0.0f;
    c.glass.doubleSided = true;
    return c;
}
// A horizontal slab (a scaled cube, SINGLE SIDED closed mesh: the shadow ray enters the front face of its underside and leaves
// through the back face of its top).  `thickness` = the world thickness (mesh-local thicknessFactor 1 after the scale).
ShadowCaster slab(float x, float y, float side, float thickness, float ior, glm::vec3 attenuationColor = glm::vec3(1.0f),
                  float attenuationDistance = std::numeric_limits<float>::infinity()) {
    ShadowCaster c;
    c.glass.shape = GlassObject::Shape::Cube;
    c.glass.center = {x, y, 0.0f};
    c.glass.scale = {side, thickness, side};
    c.glass.ior = ior;
    c.glass.thickness = 1.0f;
    c.glass.attenuationColor = attenuationColor;
    c.glass.attenuationDistance = attenuationDistance;
    return c;
}
ShadowCaster opaqueSlab(float x, float y, float side, float thickness) {
    ShadowCaster c = slab(x, y, side, thickness, 1.5f);
    c.opaque = true;
    c.glass.thickness = 0.0f;
    return c;
}
ShadowCaster glassSphere(float x, float y, float ior, glm::vec3 attenuationColor = glm::vec3(1.0f),
                         float attenuationDistance = std::numeric_limits<float>::infinity()) {
    ShadowCaster c;
    c.glass.shape = GlassObject::Shape::Sphere;
    c.glass.center = {x, y, 0.0f};
    c.glass.ior = ior;
    c.glass.thickness = 0.8f;   // the path through the centre: 2 * SphereRadius
    c.glass.attenuationColor = attenuationColor;
    c.glass.attenuationDistance = attenuationDistance;
    return c;
}

glm::vec3 towardSun(float degreesFromVertical) {
    // The sun sits behind the casters as seen by the camera (towards -Z): the shadows fall towards the camera, in front of the
    // glass on the screen, so the camera never looks at the floor through a caster.
    const float a = glm::radians(degreesFromVertical);
    return {0.0f, std::cos(a), -std::sin(a)};
}

std::vector<ShadowScenarioSpec> specs() {
    std::vector<ShadowScenarioSpec> all;
    const glm::vec3 green(0.4f, 0.9f, 0.45f);
    const float H = TransmissiveShadowValidation::CasterHeight;
    // 1. Sun 25 degrees from the pane normal.  Thin-walled clear pane (one Fresnel surface), thin-walled GREEN tinted pane,
    //    thick clear pane (two Fresnel surfaces, no absorption) and an opaque slab for comparison.
    {
        ShadowScenarioSpec s;
        s.name = "transmissive-shadow-pane-sun";
        s.casters = {thinPane(-2.1f, 0.9f, glm::vec3(1.0f)), thinPane(-0.7f, 0.9f, green), slab(0.7f, H, 0.9f, 0.06f, 1.5f),
                     opaqueSlab(2.1f, H, 0.9f, 0.06f)};
        s.sun = true;
        s.sunIrradiance = 3.0f;
        s.towardSun = towardSun(25.0f);
        all.push_back(std::move(s));
    }
    // 2. Flat shallow water (ior 1.33 slabs 0.12 m deep), sun 20 degrees from the normal: clear water, murky water
    //    (attenuation colour (0.45, 0.75, 0.9) at 0.3 m: transmittance (0.72, 0.89, 0.96) at normal incidence over the depth) and an
    //    opaque slab.  Fresnel is only 2 % per surface here: the Fresnel-ignored control is dropped for this key (4 % < the 5 % bound).
    {
        ShadowScenarioSpec s;
        s.name = "transmissive-shadow-water-sun";
        s.casters = {slab(-1.4f, 0.5f, 0.9f, 0.12f, 1.33f), slab(0.0f, 0.5f, 0.9f, 0.12f, 1.33f, {0.45f, 0.75f, 0.9f}, 0.3f),
                     opaqueSlab(1.4f, 0.5f, 0.9f, 0.12f)};
        s.sun = true;
        s.sunIrradiance = 3.0f;
        s.towardSun = towardSun(20.0f);
        all.push_back(std::move(s));
    }
    // 3. Thick glass panes (0.1 m) with volume attenuation (colour, distance), sun 20 degrees from the normal.  The attenuation
    //    is moderate: the straight-line Beer-Lambert path is 3.6 % longer than the refracted one at this angle, which costs at
    //    most 1.9 % of the transmittance at the strongest channel (the declared-model difference, measured by the reference).
    {
        ShadowScenarioSpec s;
        s.name = "transmissive-shadow-volume-pane";
        s.casters = {slab(-1.4f, H, 0.9f, 0.1f, 1.5f, {0.5f, 0.8f, 1.0f}, 0.2f), slab(0.0f, H, 0.9f, 0.1f, 1.5f, {1.0f, 0.6f, 0.35f}, 0.2f),
                     slab(1.4f, H, 0.9f, 0.1f, 1.5f, {0.65f, 0.65f, 0.65f}, 0.15f)};
        s.sun = true;
        s.sunIrradiance = 3.0f;
        s.towardSun = towardSun(20.0f);
        all.push_back(std::move(s));
    }
    // 4. One local point light 4.5 m above the floor (DI visibility through glass: lights a floor point through a caster at up to
    //    26 degrees from the vertical): the panes of key 1 at 0.7 m.
    {
        ShadowScenarioSpec s;
        s.name = "transmissive-shadow-point";
        s.casters = {thinPane(-1.8f, 0.7f, glm::vec3(1.0f)), thinPane(-0.6f, 0.7f, green), slab(0.6f, H, 0.7f, 0.06f, 1.5f),
                     opaqueSlab(1.8f, H, 0.7f, 0.06f)};
        s.pointLights = {{{0.0f, 4.5f, -0.3f}, 60.0f, 100.0f}};
        all.push_back(std::move(s));
    }
    // 5. Curved glass (qualitative, report only: caustics are excluded by declaration): a clear and an absorbing glass sphere.
    {
        ShadowScenarioSpec s;
        s.name = "transmissive-shadow-curved";
        s.casters = {glassSphere(-1.0f, 0.9f, 1.5f), glassSphere(1.0f, 0.9f, 1.5f, {0.5f, 0.8f, 1.0f}, 0.6f)};
        s.sun = true;
        s.sunIrradiance = 3.0f;
        s.towardSun = towardSun(25.0f);
        all.push_back(std::move(s));
    }
    return all;
}
const std::vector<ShadowScenarioSpec>& table() {
    static const std::vector<ShadowScenarioSpec> all = specs();
    return all;
}
} // namespace

TransmissiveShadowValidation::TransmissiveShadowValidation(std::string scenario) : scenario_(std::move(scenario)) {
    if (!validScenario(scenario_)) throw std::invalid_argument("Unknown transmissive shadow validation scenario: " + scenario_);
    name_ = "Transmissive shadow validation: " + scenario_;
}
bool TransmissiveShadowValidation::validScenario(std::string_view scenario) {
    return std::find(kAll.begin(), kAll.end(), scenario) != kAll.end();
}
std::span<const std::string_view> TransmissiveShadowValidation::scenarios() { return kAll; }
const ShadowScenarioSpec& TransmissiveShadowValidation::spec(std::string_view scenario) {
    for (const auto& s : table())
        if (s.name == scenario) return s;
    throw std::invalid_argument("Unknown transmissive shadow validation scenario: " + std::string(scenario));
}

void TransmissiveShadowValidation::setup(ECS& ecs, GpuScene& scene, TextureManager& textures) {
    if (!casters_.empty() || floor_ != INVALID_ENTITY) throw std::logic_error("Transmissive shadow fixture already set up");
    textures.createDefaultTextures(); // idempotent; materials reference none of them
    const auto& s = spec(scenario_);
    auto upload = [&](const MeshData& m) { return scene.uploadMesh(m.positions, m.normals, m.tangents, m.uvs, m.indices); };
    const u32 sphereMesh = upload(ProceduralMeshes::generateSphere(TransmissionValidation::SphereRadius, TransmissionValidation::SphereSlices,
                                                                   TransmissionValidation::SphereStacks));
    const u32 cubeMesh = upload(ProceduralMeshes::generateCube(SlabHalfExtent));
    const u32 planeMesh = upload(ProceduralMeshes::generatePlane(1.0f, 1.0f, 1, 1)); // unit square facing +Y, scaled per instance
    u32 materialIndex = 0;
    {
        floor_ = ecs.createEntity();
        TransformComponent transform;
        transform.position = {0.0f, 0.0f, 0.0f};
        transform.scale = {FloorSize, 1.0f, FloorSize};
        transform.updateMatrix();
        ecs.addComponent(floor_, std::move(transform));
        MeshInstanceComponent instance;
        instance.meshHandle = planeMesh;
        instance.materialIndex = materialIndex++;
        instance.setVisible(true);
        instance.setCastsShadows(true); // the snapshot reader requires visible AND shadow roles to be equal
        instance.setStatic(true);
        ecs.addComponent(floor_, std::move(instance));
        MaterialComponent material;
        material.baseColorFactor = glm::vec4(s.floorBase, 1.0f);
        material.metallicFactor = 0.0f;
        material.roughnessFactor = s.floorRoughness;
        ecs.addComponent(floor_, std::move(material));
    }
    for (const ShadowCaster& caster : s.casters) {
        const GlassObject& g = caster.glass;
        const EntityID e = ecs.createEntity();
        casters_.push_back(e);
        TransformComponent transform;
        transform.position = g.center;
        transform.scale = g.scale;
        transform.updateMatrix();
        ecs.addComponent(e, std::move(transform));
        MeshInstanceComponent instance;
        instance.meshHandle = g.shape == GlassObject::Shape::Sphere ? sphereMesh : g.shape == GlassObject::Shape::Cube ? cubeMesh : planeMesh;
        instance.materialIndex = materialIndex++;
        instance.setVisible(true);
        instance.setCastsShadows(true);
        instance.setStatic(true);
        ecs.addComponent(e, std::move(instance));
        MaterialComponent material;
        if (caster.opaque) {
            material.baseColorFactor = glm::vec4(0.5f, 0.5f, 0.5f, 1.0f);
            material.metallicFactor = 0.0f;
            material.roughnessFactor = 1.0f;
        } else {
            auto description = std::make_shared<MaterialDescription>();
            description->specular.emplace();
            description->specular->ior = g.ior;
            description->transmission.emplace();
            description->transmission->factor = g.transmission;
            description->volume.emplace();
            description->volume->thickness = g.thickness;
            description->volume->attenuationColor = {g.attenuationColor.x, g.attenuationColor.y, g.attenuationColor.z};
            description->volume->attenuationDistance = g.attenuationDistance;
            material.extension = std::move(description);
            material.baseColorFactor = glm::vec4(g.baseColor, 1.0f);
            material.metallicFactor = 0.0f;
            material.roughnessFactor = g.roughness;
            material.doubleSided = g.doubleSided;
        }
        // INVALID texture indices intentionally use the exact shader fallbacks (factors stay untouched).
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
    for (const ShadowPointLight& point : s.pointLights) {
        const EntityID e = ecs.createEntity();
        pointLights_.push_back(e);
        TransformComponent transform;
        transform.position = point.position;
        transform.updateMatrix();
        ecs.addComponent(e, std::move(transform));
        LightComponent light;
        light.type = LightType::Point;
        light.color = glm::vec3(1.0f);
        light.intensity = point.intensity;
        light.range = point.range;
        ecs.addComponent(e, std::move(light));
    }
}

void TransmissiveShadowValidation::teardown(ECS& ecs, GpuScene&) {
    for (EntityID e : casters_) ecs.destroyEntity(e);
    for (EntityID e : pointLights_) ecs.destroyEntity(e);
    if (floor_ != INVALID_ENTITY) ecs.destroyEntity(floor_);
    if (sun_ != INVALID_ENTITY) ecs.destroyEntity(sun_);
    casters_.clear();
    pointLights_.clear();
    floor_ = sun_ = INVALID_ENTITY;
}

CameraSetup TransmissiveShadowValidation::getDefaultCamera() const {
    const auto& s = spec(scenario_);
    const float elevation = glm::radians(s.cameraElevationDegrees);
    const glm::vec3 offset = s.cameraDistance * glm::vec3(0.0f, std::sin(elevation), std::cos(elevation));
    return {s.cameraTarget + offset, s.cameraTarget, s.cameraDistance, false};
}
} // namespace phosphor
