#include "testbench/transmission_validation.h"
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
// The names are mirrored by tools/f15_check.py SCENARIOS (tests/test_transmission_validation.cpp compares both).
// transmission-rtoff (the last) is the declared --rt off model (see below).
constexpr std::array<std::string_view, 8> kAll{
    "transmission-roughness-ior", "transmission-attenuation", "transmission-factor-tint", "transmission-thin-volume",
    "transmission-single-sided", "transmission-glass-behind-glass", "transmission-emissive-panel", "transmission-rtoff"};
constexpr float pi = 3.14159265358979323846f;

using Glass = GlassObject;

GlassObject sphere(glm::vec3 centre, float roughness, float ior, float thickness = 0.8f) {
    GlassObject g;
    g.shape = Glass::Shape::Sphere;
    g.center = centre;
    g.roughness = roughness;
    g.ior = ior;
    g.thickness = thickness;
    return g;
}

std::vector<TransmissionScenarioSpec> specs() {
    static const glm::vec3 tint(0.9f, 0.55f, 0.25f);
    std::vector<TransmissionScenarioSpec> all;

    // 1. roughness {0, 0.1, 0.3, 0.6} (rows, bottom to top) x ior {1.33, 1.5, 2.0} (columns): volume spheres without
    // absorption (attenuationDistance +inf), factor 1, white base: a lens image of the checkerboard whose sharpness
    // follows the roughness and whose bend and Fresnel follow the ior.  Spacing 1.2 keeps the neighbours' reflections small.
    {
        TransmissionScenarioSpec s;
        s.name = "transmission-roughness-ior";
        const float roughness[4] = {0.0f, 0.1f, 0.3f, 0.6f}, ior[3] = {1.33f, 1.5f, 2.0f};
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 3; ++c) s.glass.push_back(sphere({(float(c) - 1.0f) * 1.2f, 0.4f + 1.2f * float(r), 0.0f}, roughness[r], ior[c]));
        s.cameraTarget = {0.0f, 2.2f, 0.0f};
        all.push_back(std::move(s));
    }
    // 2. AttenuationTest-like: attenuation colour (rows) x attenuation distance (columns) on clear ior 1.5 spheres of
    // roughness 0.1; the path through the sphere centre is 0.8 m, so the transmittance there is colour^(0.8 / distance).
    {
        TransmissionScenarioSpec s;
        s.name = "transmission-attenuation";
        // Moderate absorption: a strongly absorbing smooth sphere (distance 0.3, blue 0.15) makes the glTF Schlick Fresnel differ
        // from Mitsuba's exact one by 8 % mean / 19 % p95 on the dark pixels (measured, tools/f15_transmission_reference.py
        // model-differences), so extreme attenuation would gate the Fresnel model, not the volume; here at most 1.2 % / 3.1 %.
        const glm::vec3 colour[2] = {{0.5f, 0.75f, 1.0f}, {1.0f, 0.65f, 0.4f}};
        const float distance[3] = {0.6f, 1.2f, 3.0f};
        for (int r = 0; r < 2; ++r)
            for (int c = 0; c < 3; ++c) {
                GlassObject g = sphere({(float(c) - 1.0f) * 1.2f, 0.4f + 1.2f * float(r), 0.0f}, 0.1f, 1.5f);
                g.attenuationColor = colour[r];
                g.attenuationDistance = distance[c];
                s.glass.push_back(g);
            }
        s.cameraTarget = {0.0f, 1.0f, 0.0f};
        all.push_back(std::move(s));
    }
    // 3. Transmission factor x tint, thin pane (column 0, ONE crossing: the tint is unambiguous) and volume slab
    // (column 1, white base, attenuating).  Rows: (factor 1, white), (1, tint), (0.5, white), (0.5, tint); the slab rows
    // alternate the attenuation distance {0.2, 0.6} and ignore the tint (white base).  Slabs are cubes scaled to
    // 0.9 x 0.9 x 0.3 and yawed 20 degrees (parallel faces: a shifted, not inverted image).
    {
        TransmissionScenarioSpec s;
        s.name = "transmission-factor-tint";
        const float factor[4] = {1.0f, 1.0f, 0.5f, 0.5f};
        const bool tinted[4] = {false, true, false, true};
        const float distance[4] = {0.2f, 0.6f, 0.2f, 0.6f};
        for (int r = 0; r < 4; ++r) {
            GlassObject pane;
            pane.shape = Glass::Shape::Pane;
            pane.center = {-1.0f, 0.5f + 1.2f * float(r), 0.0f};
            pane.scale = {0.9f, 1.0f, 0.9f};
            pane.transmission = factor[r];
            pane.baseColor = tinted[r] ? tint : glm::vec3(1.0f);
            pane.doubleSided = true; // a one-quad thin-walled surface is seen from both sides
            s.glass.push_back(pane);
            GlassObject slab;
            slab.shape = Glass::Shape::Cube;
            slab.center = {1.0f, 0.5f + 1.2f * float(r), 0.0f};
            slab.scale = {0.9f, 0.9f, 0.3f};
            slab.yawDegrees = 20.0f;
            slab.transmission = factor[r];
            slab.thickness = 1.0f; // mesh-local: 0.3 m in the world after the scale
            slab.attenuationColor = {0.6f, 0.85f, 1.0f};
            slab.attenuationDistance = distance[r];
            s.glass.push_back(slab);
        }
        s.cameraTarget = {0.0f, 2.0f, 0.0f};
        all.push_back(std::move(s));
    }
    // 4. Thin-walled vs volume: a thin-walled sphere (no volume: no bend, two Fresnel surfaces), a clear volume sphere
    // and an attenuating volume sphere, all smooth, ior 1.5.
    {
        TransmissionScenarioSpec s;
        s.name = "transmission-thin-volume";
        s.glass.push_back(sphere({-1.2f, 1.0f, 0.0f}, 0.0f, 1.5f, 0.0f)); // thickness 0: thin-walled
        s.glass.push_back(sphere({0.0f, 1.0f, 0.0f}, 0.0f, 1.5f, 0.8f));
        GlassObject absorbing = sphere({1.2f, 1.0f, 0.0f}, 0.0f, 1.5f, 0.8f);
        absorbing.attenuationColor = {0.4f, 0.8f, 1.0f};
        absorbing.attenuationDistance = 0.8f;
        s.glass.push_back(absorbing);
        s.cameraDistance = 4.5f;
        all.push_back(std::move(s));
    }
    // 5. SINGLE-SIDED closed meshes (doubleSided = false, the default): a lens sphere, a rotated cube and an absorbing
    // sphere.  The exit surface of each is a back face of its own mesh: today's back-face rejection never finds it.
    {
        TransmissionScenarioSpec s;
        s.name = "transmission-single-sided";
        s.glass.push_back(sphere({-1.3f, 1.0f, 0.0f}, 0.0f, 1.5f, 0.8f));
        GlassObject cube;
        cube.shape = Glass::Shape::Cube;
        cube.center = {0.0f, 1.0f, 0.0f};
        cube.scale = {0.8f, 0.8f, 0.8f};
        cube.yawDegrees = 25.0f;
        cube.pitchDegrees = 15.0f;
        cube.thickness = 1.0f;
        s.glass.push_back(cube);
        GlassObject absorbing = sphere({1.3f, 1.0f, 0.0f}, 0.0f, 1.5f, 0.8f);
        absorbing.attenuationColor = {0.5f, 0.8f, 1.0f};
        absorbing.attenuationDistance = 0.6f;
        s.glass.push_back(absorbing);
        s.cameraDistance = 4.5f;
        all.push_back(std::move(s));
    }
    // 6. Glass behind glass: a clear volume sphere in front of an attenuating one (they overlap on the screen: four
    // interface crossings along the central rays), and a thin pane in front of an ior 2.0 sphere.
    {
        TransmissionScenarioSpec s;
        s.name = "transmission-glass-behind-glass";
        s.glass.push_back(sphere({-0.25f, 1.0f, 0.7f}, 0.0f, 1.5f, 0.8f));
        GlassObject back = sphere({0.25f, 1.0f, -0.7f}, 0.0f, 1.5f, 0.8f);
        back.attenuationColor = {0.5f, 0.8f, 1.0f};
        back.attenuationDistance = 0.6f;
        s.glass.push_back(back);
        GlassObject pane;
        pane.shape = Glass::Shape::Pane;
        pane.center = {1.7f, 2.4f, 0.6f};
        pane.scale = {0.9f, 1.0f, 0.9f};
        pane.doubleSided = true;
        s.glass.push_back(pane);
        s.glass.push_back(sphere({1.7f, 2.4f, -0.4f}, 0.0f, 2.0f, 0.8f));
        s.cameraTarget = {0.3f, 1.7f, 0.0f};
        all.push_back(std::move(s));
    }
    // 7. Emissive panels behind thin panes: a clear pane and a tinted pane each in front of an emissive panel (radiance 3)
    // that is smaller than the pane, so both the panel and the environment are seen through it.  Lit with ReSTIR DI
    // (the panels are sampled area lights): the thin-walled DI BTDF covers only delta lights, the transport owns the
    // panels, the radiance must be counted once.
    {
        TransmissionScenarioSpec s;
        s.name = "transmission-emissive-panel";
        s.backdrop = TransmissionScenarioSpec::Backdrop::Panels;
        s.emission = 3.0f;
        const glm::vec3 colours[2] = {glm::vec3(1.0f), tint};
        for (int c = 0; c < 2; ++c) {
            GlassObject pane;
            pane.shape = Glass::Shape::Pane;
            pane.center = {(c == 0 ? -1.0f : 1.0f), 1.2f, 0.2f};
            pane.scale = {1.2f, 1.0f, 1.2f};
            pane.baseColor = colours[c];
            pane.doubleSided = true;
            s.glass.push_back(pane);
            s.panels.push_back({pane.center.x, 1.2f, 0.5f, 0.5f});
        }
        s.cameraTarget = {0.0f, 1.2f, 0.0f};
        s.cameraDistance = 4.5f;
        all.push_back(std::move(s));
    }
    // 8. The declared --rt off model: no backdrop (without RT the engine sees only the probes/environment along the
    // refracted ray).  Rows factor {1, 0.5}; columns: thin white, thin tinted, attenuating volume, clear volume.  Roughness 0.1.
    {
        TransmissionScenarioSpec s;
        s.name = "transmission-rtoff";
        s.backdrop = TransmissionScenarioSpec::Backdrop::None;
        const float factor[2] = {1.0f, 0.5f};
        for (int r = 0; r < 2; ++r)
            for (int c = 0; c < 4; ++c) {
                GlassObject g = sphere({(float(c) - 1.5f) * 1.1f, 0.4f + 1.1f * float(r), 0.0f}, 0.1f, 1.5f, c >= 2 ? 0.8f : 0.0f);
                g.transmission = factor[r];
                if (c == 1) g.baseColor = tint;
                if (c == 2) {
                    g.attenuationColor = {0.4f, 0.8f, 1.0f};
                    g.attenuationDistance = 0.8f;
                }
                s.glass.push_back(g);
            }
        s.cameraTarget = {0.0f, 0.95f, 0.0f};
        all.push_back(std::move(s));
    }
    return all;
}
const std::vector<TransmissionScenarioSpec>& table() {
    static const std::vector<TransmissionScenarioSpec> all = specs();
    return all;
}

glm::quat orientation(const GlassObject& g) {
    glm::quat q = glm::angleAxis(glm::radians(g.yawDegrees), glm::vec3(0, 1, 0)) *
                  glm::angleAxis(glm::radians(g.pitchDegrees), glm::vec3(1, 0, 0));
    if (g.shape == GlassObject::Shape::Pane) q = q * glm::angleAxis(pi * 0.5f, glm::vec3(1, 0, 0)); // the plane faces +Y: turn it to +Z
    return q;
}
} // namespace

TransmissionValidation::TransmissionValidation(std::string scenario) : scenario_(std::move(scenario)) {
    if (!validScenario(scenario_)) throw std::invalid_argument("Unknown transmission validation scenario: " + scenario_);
    name_ = "Transmission validation: " + scenario_;
}
bool TransmissionValidation::validScenario(std::string_view scenario) {
    return std::find(kAll.begin(), kAll.end(), scenario) != kAll.end();
}
std::span<const std::string_view> TransmissionValidation::scenarios() { return kAll; }
const TransmissionScenarioSpec& TransmissionValidation::spec(std::string_view scenario) {
    for (const auto& s : table())
        if (s.name == scenario) return s;
    throw std::invalid_argument("Unknown transmission validation scenario: " + std::string(scenario));
}
float TransmissionValidation::environmentRadiance(std::string_view scenario) {
    return validScenario(scenario) ? spec(scenario).environment : 0.0f;
}

void TransmissionValidation::setup(ECS& ecs, GpuScene& scene, TextureManager& textures) {
    if (!glass_.empty() || !emitters_.empty()) throw std::logic_error("Transmission fixture already set up");
    textures.createDefaultTextures(); // idempotent; materials reference none of them
    const auto& s = spec(scenario_);
    auto upload = [&](const MeshData& m) { return scene.uploadMesh(m.positions, m.normals, m.tangents, m.uvs, m.indices); };
    const u32 sphereMesh = upload(ProceduralMeshes::generateSphere(SphereRadius, SphereSlices, SphereStacks));
    const u32 cubeMesh = upload(ProceduralMeshes::generateCube(CubeHalfExtent));
    const u32 planeMesh = upload(ProceduralMeshes::generatePlane(1.0f, 1.0f, 1, 1)); // unit square, scaled per instance
    u32 materialIndex = 0;
    for (const GlassObject& g : s.glass) {
        const EntityID e = ecs.createEntity();
        glass_.push_back(e);
        TransformComponent transform;
        transform.position = g.center;
        transform.rotation = orientation(g);
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
        // INVALID texture indices intentionally use the exact shader fallbacks (factors stay untouched).
        ecs.addComponent(e, std::move(material));
    }
    auto emitter = [&](glm::vec3 centre, float width, float height) {
        const EntityID e = ecs.createEntity();
        emitters_.push_back(e);
        TransformComponent transform;
        transform.position = centre;
        transform.rotation = glm::angleAxis(pi * 0.5f, glm::vec3(1, 0, 0)); // faces +Z, towards the camera
        transform.scale = {width, 1.0f, height};
        transform.updateMatrix();
        ecs.addComponent(e, std::move(transform));
        MeshInstanceComponent instance;
        instance.meshHandle = planeMesh;
        instance.materialIndex = materialIndex++;
        instance.setVisible(true);
        instance.setCastsShadows(true); // the snapshot reader requires visible AND shadow roles to be equal (equivalent in Mitsuba)
        instance.setStatic(true);
        ecs.addComponent(e, std::move(instance));
        MaterialComponent material;
        material.baseColorFactor = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f); // purely emissive: no diffuse base to light through the glass
        material.metallicFactor = 0.0f;
        material.roughnessFactor = 1.0f;
        material.emissiveFactor = glm::vec3(s.emission);
        ecs.addComponent(e, std::move(material));
    };
    if (s.backdrop == TransmissionScenarioSpec::Backdrop::Checker) {
        for (int row = 0; row < CheckerRows; ++row)
            for (int column = 0; column < CheckerColumns; ++column)
                if ((row + column) % 2 == 0)
                    emitter({CheckerOriginX + (float(column) + 0.5f) * CellSize, CheckerOriginY + (float(row) + 0.5f) * CellSize, BackdropZ},
                            CellSize, CellSize);
    } else if (s.backdrop == TransmissionScenarioSpec::Backdrop::Panels) {
        for (const glm::vec4& panel : s.panels) emitter({panel.x, panel.y, PanelZ}, 2.0f * panel.z, 2.0f * panel.w);
    }
}

void TransmissionValidation::teardown(ECS& ecs, GpuScene&) {
    for (EntityID e : glass_) ecs.destroyEntity(e);
    for (EntityID e : emitters_) ecs.destroyEntity(e);
    glass_.clear();
    emitters_.clear();
}

CameraSetup TransmissionValidation::getDefaultCamera() const {
    const auto& s = spec(scenario_);
    return {{s.cameraTarget.x, s.cameraTarget.y, s.cameraTarget.z + s.cameraDistance}, s.cameraTarget, s.cameraDistance, false};
}
} // namespace phosphor
