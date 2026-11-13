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
// The names are mirrored by tools/f15_check.py SCENARIOS (tests/test_material_validation.cpp compares both).
constexpr std::array<std::string_view, 19> kScenarios{"base-grid-sun",  "base-grid-env",       "base-grid-env-resolve",
                                                      "furnace-base",   "furnace-resolve",     "roughness-extremes",
                                                      "roughness-extremes-sun", "specular-grid-sun",
                                                      "specular-grid-env-resolve", "specular-grid-point", "specular-grid-env", "furnace-specular",
                                                      "clearcoat-grid-sun", "clearcoat-grid-env-resolve", "clearcoat-grid-env",
                                                      "clearcoat-grid-point", "furnace-clearcoat",
                                                      "specular-grid-edit", "specular-grid-rough"};
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
    // Side-lit and slightly from behind (V.L about -0.33): the half vector sits
    // far from the view direction, so Schlick's (1 - V.H)^5 is non-negligible
    // and a missing or wrong Fresnel term is visible in the direct grid.
    static const glm::vec3 towardSun = glm::normalize(glm::vec3(-0.8f, 0.35f, -0.3f));
    // F15.1 (ticket 05): KHR_materials_ior x KHR_materials_specular on texture-free spheres.  Columns
    // are IORs (0 = the permanent Fresnel = 1 mode, 1.0 f0 = 0, 1.33 water, 1.5 the default, 2.4
    // diamond), rows are (specularFactor, specularColor): the neutral pair, a weight, a colour above 1
    // (f0 clamped before the weight), both; the last row is metallic 1, on which both extensions must
    // have no effect.  One roughness (0.35, alpha 0.12: wide lobe, no sun quadrature) keeps the grid small.
    static const std::vector<float> specIor{0.0f, 1.0f, 1.33f, 1.5f, 2.4f};
    static const std::vector<float> specRoughness(5, 0.35f);
    static const std::vector<float> specMetallic{0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    static const std::vector<float> specFactor{1.0f, 0.5f, 1.0f, 0.5f, 0.5f};
    static const std::vector<glm::vec3> specColor{glm::vec3(1.0f), glm::vec3(1.0f), glm::vec3(2.0f),
                                                  glm::vec3(2.0f, 1.0f, 0.5f), glm::vec3(2.0f)};
    // F15.1 (ticket 07): the same grid lit by one point light, side-lit from behind like the sun (V.L about
    // -0.5 at the grid centre, so Fresnel discriminates f90 and the specular weight).  Spheres may shadow
    // their neighbours: DI traces that visibility (the reference casts the same shadow rays), the legacy
    // resolve's local lights are unshadowed (reference without visibility).  9 m from the grid centre,
    // 250 W/sr gives about the sun's 3 W/m^2; range 100 m keeps the engine's window within 5e-4 of 1
    // (declared exactly by the reference).
    static const glm::vec3 gridCentre(0.0f, 2.0f, 0.0f);
    static const MaterialPointLight backLight{gridCentre + 9.0f * glm::normalize(glm::vec3(-0.65f, 0.55f, -0.52f)),
                                              250.0f, 100.0f};
    // F15.3 (ticket 09): KHR_materials_clearcoat on texture-free spheres.  Rows are the coat factor (0 = no coat, the
    // neutral material; 0.25, 0.5, 1), then coat 1 over a base of ior 2.4 (a different base F0 than the coat's fixed
    // 0.04: the Sample Renderer deviation "coat F0 from the material ior" is only visible there), then a metal under a
    // full coat.  Columns are the coat roughness: 0 and 0.05 (alpha 0.002 and 0.0025) and 0.15 (alpha 0.0225) exercise the
    // sun disk's narrow-lobe quadrature, 0.4 and 1.0 are wide.  The base is dielectric copper at roughness 0.6: a rough
    // base under a smooth coat, so the coat reflection is the sharp feature the denoiser must keep.
    static const std::vector<float> coatRoughness{0.0f, 0.05f, 0.15f, 0.4f, 1.0f};
    static const std::vector<float> coatFactor{0.0f, 0.25f, 0.5f, 1.0f, 1.0f, 1.0f};
    static const std::vector<float> coatBaseIor{0.0f, 0.0f, 0.0f, 0.0f, 2.4f, 0.0f}; // 0 = no ior block in that row
    static const std::vector<float> coatBaseRoughness(5, 0.6f);
    static const std::vector<float> coatMetallic{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    static const glm::vec3 coatGridCentre(0.0f, 2.5f, 0.0f);
    static const MaterialPointLight coatBackLight{coatGridCentre + 9.0f * glm::normalize(glm::vec3(-0.65f, 0.55f, -0.52f)),
                                                  250.0f, 100.0f};
    static const std::vector<MaterialScenarioSpec> all{
        {"base-grid-sun", roughness, metallic, copper, true, 3.0f, towardSun, 0.0f},
        {"base-grid-env", roughness, metallic, copper, false, 0.0f, towardSun, 0.5f},
        // The same scene as base-grid-env without the specular transport: the resolve
        // evaluates the declared environment itself (run with --reflections off
        // --lighting-denoise custom); gated against the declared-model reference.
        {"base-grid-env-resolve", roughness, metallic, copper, false, 0.0f, towardSun, 0.5f},
        {"furnace-base", roughness, metallic, glm::vec3(1.0f), false, 0.0f, towardSun, 1.0f},
        // Same scene without the specular transport: the resolve evaluates the
        // declared environment deterministically (run with --reflections off).
        {"furnace-resolve", roughness, metallic, glm::vec3(1.0f), false, 0.0f, towardSun, 1.0f},
        // Roughness 0 is the declared clamp (0.04); 1 is the upper bound.
        {"roughness-extremes", {0.0f, 1.0f}, {0.0f, 0.5f, 1.0f}, copper, true, 3.0f, towardSun, 0.3f},
        // The same spheres lit by the sun only (environment 0): the direct gate applies.
        {"roughness-extremes-sun", {0.0f, 1.0f}, {0.0f, 0.5f, 1.0f}, copper, true, 3.0f, towardSun, 0.0f},
        // Sun only: the direct gate applies (the extension lobes of the resolve and forward paths).
        {"specular-grid-sun", specRoughness, specMetallic, copper, true, 3.0f, towardSun, 0.0f, specIor, specFactor,
         specColor},
        // Environment only, deterministic resolve (--reflections off --lighting-denoise custom): the
        // declared split-sum model f0 * A + f90 * B per lobe.
        {"specular-grid-env-resolve", specRoughness, specMetallic, copper, false, 0.0f, towardSun, 0.5f, specIor,
         specFactor, specColor},
        // One point light only (no sun, environment 0): the resolve's local lights with --lighting legacy,
        // ReSTIR or clustered DI otherwise (the DI grid of ticket 07).
        {"specular-grid-point", specRoughness, specMetallic, copper, false, 0.0f, towardSun, 0.0f, specIor,
         specFactor, specColor, {backLight}},
        // F15.1 (ticket 08): the same grid and environment as specular-grid-env-resolve with the specular transport on
        // (SSR by default; the runner also captures it with RT reflections and with the probe path).
        {"specular-grid-env", specRoughness, specMetallic, copper, false, 0.0f, towardSun, 0.5f, specIor, specFactor,
         specColor},
        // Furnace over EXTENDED materials: the ior x specular grid at base colour 1 under a uniform environment of
        // radiance 1 (the runner also captures it without the transport, deterministic resolve terms).
        {"furnace-specular", specRoughness, specMetallic, glm::vec3(1.0f), false, 0.0f, towardSun, 1.0f, specIor,
         specFactor, specColor},
        // F15.3 (ticket 09) clearcoat grids.  Sun only: the direct gate (coat lobe in the sun-disk quadrature).
        {"clearcoat-grid-sun", coatBaseRoughness, coatMetallic, copper, true, 3.0f, towardSun, 0.0f, {}, {}, {}, {},
         coatFactor, coatRoughness, coatBaseIor},
        // Environment only, deterministic resolve (--reflections off --lighting-denoise custom): the declared coat term.
        {"clearcoat-grid-env-resolve", coatBaseRoughness, coatMetallic, copper, false, 0.0f, towardSun, 0.5f, {}, {}, {},
         {}, coatFactor, coatRoughness, coatBaseIor},
        // The same scene with the specular transport on: the coat term in the environment, the coat lobe in the transport.
        {"clearcoat-grid-env", coatBaseRoughness, coatMetallic, copper, false, 0.0f, towardSun, 0.5f, {}, {}, {}, {},
         coatFactor, coatRoughness, coatBaseIor},
        // One point light only (no sun, environment 0), side-lit from behind like specular-grid-point.
        {"clearcoat-grid-point", coatBaseRoughness, coatMetallic, copper, false, 0.0f, towardSun, 0.0f, {}, {}, {},
         {coatBackLight}, coatFactor, coatRoughness, coatBaseIor},
        // Furnace over coated materials: albedo 1 under a uniform environment of radiance 1 (the runner also captures it
        // without the transport).
        {"furnace-clearcoat", coatBaseRoughness, coatMetallic, glm::vec3(1.0f), false, 0.0f, towardSun, 1.0f, {}, {}, {},
         {}, coatFactor, coatRoughness, coatBaseIor},
        // Clip fixture of ticket 08 (not gated against the reference): the grid and environment of specular-grid-env
        // with the strong runtime edit of --material-toggle-every (tools/f15_clip_check.py: recovery after an edit).
        {"specular-grid-edit", specRoughness, specMetallic, copper, false, 0.0f, towardSun, 0.5f, specIor, specFactor,
         specColor, {}, {}, {}, {}, true},
        // Adaptive-shading fixture of ticket 08 (tools/f15_clip_check.py): rough (>= 0.75) dielectric spheres, the
        // reuse domain of visibility_adaptive, with every extension of the grid, lit by the sun (plain resolve).
        {"specular-grid-rough", std::vector<float>(5, 0.85f), std::vector<float>(5, 0.0f), copper, true, 3.0f, towardSun,
         0.0f, specIor, specFactor, specColor},
    };
    return all;
}
}

MaterialValidation::MaterialValidation(std::string scenario, u32 toggleEvery)
    : scenario_(std::move(scenario)), toggleEvery_(toggleEvery) {
    if (!validScenario(scenario_)) throw std::invalid_argument("Unknown material validation scenario: " + scenario_);
    if (toggleEvery_ > 0 && !spec(scenario_).hasSpecularGrid())
        throw std::invalid_argument("material toggling needs a specular grid scenario");
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
            if (s.hasSpecularGrid()) {
                auto description = std::make_shared<MaterialDescription>();
                description->specular.emplace();
                description->specular->ior            = s.ior[column];
                description->specular->specularFactor = s.specularFactor[row];
                const glm::vec3 c = s.specularColor[row];
                description->specular->specularColor = {c.x, c.y, c.z};
                material.extension = std::move(description);
            }
            if (s.hasClearcoatGrid()) {
                const bool hasIor = !s.baseIor.empty() && s.baseIor[row] > 0.0f;
                if (s.clearcoatFactor[row] > 0.0f || hasIor) {
                    auto description = std::make_shared<MaterialDescription>();
                    description->clearcoat.emplace();
                    description->clearcoat->factor    = s.clearcoatFactor[row];
                    description->clearcoat->roughness = s.clearcoatRoughness[column];
                    if (hasIor) {
                        description->specular.emplace();
                        description->specular->ior = s.baseIor[row];
                    }
                    material.extension = std::move(description);
                }
            }
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
    for (const MaterialPointLight& point : s.pointLights) {
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

void MaterialValidation::update(float, ECS& ecs) {
    if (toggleEvery_ == 0 || spheres_.empty()) return;
    if (++updates_ % toggleEvery_ != 0) return;
    const u32 event = updates_ / toggleEvery_; // 1, 2, 3 ...
    auto edited = [&](u32 index, auto&& change) {
        auto& m = ecs.getComponent<MaterialComponent>(spheres_[index]); // mutable access marks the change
        auto d = std::make_shared<MaterialDescription>(*m.extension);
        change(*d->specular);
        m.extension = std::move(d);
    };
    // Odd events: the feature sphere becomes active (ior 1.33), even events neutral again.
    const bool strong = spec(scenario_).strongEdit, odd = event % 2 == 1;
    edited(toggleFeatureSphere(), [&](SpecularBlock& b) { b.ior = strong ? (odd ? 2.4f : 1.5f) : (odd ? 1.33f : 1.5f); });
    // The parameter sphere alternates its weight 0.5 / 0.6 (a record rewrite without a bit change).
    edited(toggleParameterSphere(), [&](SpecularBlock& b) {
        if (strong) b.specularColor = odd ? std::array<float, 3>{25.0f, 25.0f, 25.0f} : std::array<float, 3>{1.0f, 1.0f, 1.0f};
        else b.specularFactor = odd ? 0.6f : 0.5f;
    });
}

void MaterialValidation::teardown(ECS& ecs, GpuScene&) {
    for (EntityID e : spheres_) ecs.destroyEntity(e);
    if (sun_ != INVALID_ENTITY) ecs.destroyEntity(sun_);
    for (EntityID e : pointLights_) ecs.destroyEntity(e);
    pointLights_.clear();
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
