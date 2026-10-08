#include "testbench/lighting_validation.h"
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

namespace phosphor {
namespace {
constexpr std::array<std::string_view, 11> kScenarios{
    "cornell", "thin-walls", "moving-sun", "moving-emissive",
    "offscreen-caster", "shadow-bias", "disocclusion", "cache-stress",
    "shadow-penumbra", "alpha-mip-shadow", "emissive-step"};
constexpr float pi = 3.14159265358979323846f;
const glm::quat identity(1, 0, 0, 0);
glm::quat orientLight(glm::vec3 towardLight) {
    // extractLights maps local -Z through the entity matrix to the direction
    // light travels. The renderer takes its negative as toward-light.
    const glm::vec3 direction = -glm::normalize(towardLight), forward(0, 0, -1);
    const float cosine = std::clamp(glm::dot(forward, direction), -1.0f, 1.0f);
    if (cosine > 0.999999f) return identity;
    if (cosine < -0.999999f) return glm::angleAxis(pi, glm::vec3(0, 1, 0));
    return glm::angleAxis(std::acos(cosine), glm::normalize(glm::cross(forward, direction)));
}
}

LightingValidation::LightingValidation(std::string scenario) : scenario_(std::move(scenario)) {
    if (!validScenario(scenario_)) throw std::invalid_argument("Unknown lighting validation scenario: " + scenario_);
    name_ = "Lighting validation: " + scenario_;
}
bool LightingValidation::validScenario(std::string_view scenario) {
    return std::find(kScenarios.begin(), kScenarios.end(), scenario) != kScenarios.end();
}
std::span<const std::string_view> LightingValidation::scenarios() { return kScenarios; }

EntityID LightingValidation::mesh(ECS& ecs, u32 handle, glm::vec3 position, glm::vec3 fullSize,
                                  glm::quat rotation, glm::vec3 color, glm::vec3 emission, bool isStatic) {
    const EntityID e = ecs.createEntity(); entities_.push_back(e);
    TransformComponent transform;
    transform.position = position; transform.rotation = rotation; transform.scale = fullSize;
    // Shared cube/plane have unit full extent, so fullSize is physical size.
    transform.updateMatrix(); ecs.addComponent(e, std::move(transform));
    MeshInstanceComponent instance;
    instance.meshHandle = handle; instance.materialIndex = 0;
    instance.setVisible(true); instance.setCastsShadows(true); instance.setStatic(isStatic);
    ecs.addComponent(e, std::move(instance));
    MaterialComponent material;
    material.baseColorFactor = glm::vec4(color, 1);
    material.metallicFactor = 0; material.roughnessFactor = 1;
    material.emissiveFactor = emission;
    // INVALID indices intentionally use exact shader fallback values: the
    // default normal/MR textures would change the analytic material numerics.
    // Room planes are one-sided and oriented inward, as are emitting panels.
    material.doubleSided = false;
    ecs.addComponent(e, std::move(material));
    return e;
}
EntityID LightingValidation::directional(ECS& ecs, glm::vec3 towardLight, float intensity) {
    const EntityID e = ecs.createEntity(); entities_.push_back(e);
    TransformComponent transform;
    transform.rotation = orientLight(towardLight); transform.updateMatrix();
    ecs.addComponent(e, std::move(transform));
    LightComponent light;
    light.type = LightType::Directional; light.intensity = intensity; light.color = glm::vec3(1);
    ecs.addComponent(e, std::move(light));
    return e;
}

void LightingValidation::room(ECS& ecs) {
    // 4x4x4 box, front (+Z) open. All five walls have inward normals.
    const glm::vec3 white(0.73f), red(0.65f, 0.05f, 0.05f), green(0.12f, 0.45f, 0.15f);
    mesh(ecs, plane_, {0, 0, 0}, {4, 1, 4}, identity, white);
    mesh(ecs, plane_, {0, 4, 0}, {4, 1, 4}, glm::angleAxis(pi, glm::vec3(0, 0, 1)), white);
    mesh(ecs, plane_, {0, 2, -2}, {4, 1, 4}, glm::angleAxis(pi * 0.5f, glm::vec3(1, 0, 0)), white);
    mesh(ecs, plane_, {-2, 2, 0}, {4, 1, 4}, glm::angleAxis(-pi * 0.5f, glm::vec3(0, 0, 1)), red);
    mesh(ecs, plane_, {2, 2, 0}, {4, 1, 4}, glm::angleAxis(pi * 0.5f, glm::vec3(0, 0, 1)), green);
    const bool thin = scenario_ == "thin-walls";
    if (thin) {
        // CLOSED 10mm wall; actual separate faces, not a doubled zero-thickness
        // plane. Left-side panel lights left compartment, exposing GI leaks.
        thinWall_ = mesh(ecs, cube_, {0, 2, -0.35f}, {ThinWallThickness, 4, 3.3f}, identity, white);
        embeddedSolid_ = mesh(ecs, cube_, embeddedProbeAnchor(), {0.8f, 1.0f, 0.8f}, identity, white);
        panel_ = mesh(ecs, plane_, {-1.1f, 3.98f, -0.5f}, {0.8f, 1, 0.8f},
                      glm::angleAxis(pi, glm::vec3(0, 0, 1)), glm::vec3(0.8f), glm::vec3(12));
    } else {
        mesh(ecs, cube_, {-0.85f, 0.6f, 0.45f}, {1.0f, 1.2f, 1.0f},
             glm::angleAxis(-0.2f, glm::vec3(0, 1, 0)), white);
        mesh(ecs, cube_, {0.85f, 1.0f, -0.65f}, {0.8f, 2, 0.8f},
             glm::angleAxis(0.25f, glm::vec3(0, 1, 0)), white);
        panel_ = mesh(ecs, plane_, {0, 3.98f, -0.3f}, {1.0f, 1, 0.8f},
                      glm::angleAxis(pi, glm::vec3(0, 0, 1)), glm::vec3(0.8f), glm::vec3(12),
                      scenario_ != "moving-emissive");
    }
    // Physical emissive mesh ONLY; no co-located point approximation, which
    // would double count the light in F11/F12 and invalidate a GI reference.
    if (scenario_ == "disocclusion")
        mover_ = mesh(ecs, cube_, {0, 1.8f, 1.1f}, {1.7f, 3.4f, 0.08f}, identity, glm::vec3(0.6f), {}, false);
}

void LightingValidation::exterior(ECS& ecs) {
    const bool offscreen = scenario_ == "offscreen-caster";
    mesh(ecs, plane_, {0, 0, 0}, offscreen ? glm::vec3(30, 1, 20) : glm::vec3(12, 1, 12), identity, glm::vec3(0.7f));
    if (offscreen) {
        // For a 60deg vertical,16:9 camera at (0,2,7), this caster is outside
        // the camera frustum. Toward-light(-2,1,0) projects its centre onto
        // receiver origin; independent sun/caster culling must retain it.
        mover_ = mesh(ecs, cube_, {-12, 6, 0}, {1, 2, 1}, identity, glm::vec3(0.4f));
        sun_ = directional(ecs, {-2, 1, 0}, 3);
    } else if (scenario_ == "shadow-penumbra") {
        // Exact zero-thickness plates, outside the main camera. Their straight
        // X edges yield a disk-CDF shadow profile on the ground. The Z extent
        // keeps the protocol's |z|<=0.2 receiver band away from end effects.
        mesh(ecs, plane_, {-1.2f, 8.0f, 0}, {0.8f, 1, 2}, identity, glm::vec3(0.5f));
        mesh(ecs, plane_, {1.2f, 32.0f, 0}, {0.8f, 1, 2}, identity, glm::vec3(0.5f));
        sun_ = directional(ecs, {0, 1, 0}, 3);
    } else if (scenario_ == "alpha-mip-shadow") {
        const auto facingCamera = glm::angleAxis(pi * 0.5f, glm::vec3(1, 0, 0));
        mesh(ecs, plane_, {0, 1.5f, -0.2f}, {3, 1, 3}, facingCamera, glm::vec3(0.2f));
        alphaReceiver_ = mesh(ecs, plane_, {0, 1.5f, 0}, {3, 1, 3}, facingCamera, glm::vec3(0.7f));
        // Light projects this off-axis caster onto the alpha receiver. Missing
        // receiver pixels cannot hide behind an everywhere-lit shadow mask.
        mover_ = mesh(ecs, cube_, {-4, 2, 1}, {0.8f, 0.8f, 0.2f}, identity, glm::vec3(0.4f));
        sun_ = directional(ecs, {-4, 0.5f, 1}, 3);
    } else if (scenario_ == "shadow-bias") {
        // Horizontal 5mm plates at distinct receiver distances. With vertical
        // nominal sun, expected geometric disk penumbra radius ~h*tan(alpha).
        // This is an analytic geometry relation, NOT a measured output claim.
        for (u32 i = 0; i < 3; ++i) {
            const float heights[] = {0.02f, 0.5f, 2.0f};
            mesh(ecs, cube_, {float(i) * 2 - 2, heights[i], 0}, {0.8f, 0.005f, 0.8f}, identity, glm::vec3(0.5f));
            mesh(ecs, cube_, {float(i) * 2 - 2, 0.6f, -1.5f}, {0.005f, 1.2f, 0.8f}, identity, glm::vec3(0.5f));
        }
        sun_ = directional(ecs, {0, 1, 0}, 3);
    } else if (scenario_ == "cache-stress") {
        // 36 small STATIC casters whose transforms are explicitly revised.
        // This forces cache invalidation despite static semantic classification.
        for (u32 z = 0; z < 6; ++z) for (u32 x = 0; x < 6; ++x) {
            const glm::vec3 origin((float(x) - 2.5f) * 1.2f, 0.35f, (float(z) - 2.5f) * 1.2f);
            cacheOrigins_.push_back(origin);
            cacheCasters_.push_back(mesh(ecs, cube_, origin, {0.35f, 0.7f, 0.35f}, identity, glm::vec3(0.6f)));
        }
        mover_ = cacheCasters_.front(); sun_ = directional(ecs, {-0.4f, 1, 0.2f}, 3);
    } else {
        mesh(ecs, cube_, {-1.4f, 0.5f, 0}, {0.8f, 1, 0.8f}, identity, glm::vec3(0.7f));
        mover_ = mesh(ecs, cube_, {0.7f, 1.1f, -0.5f}, {1, 2.2f, 1}, identity, glm::vec3(0.6f));
        mesh(ecs, cube_, {2, 0.025f, 1}, {1, 0.005f, 1}, identity, glm::vec3(0.5f));
        sun_ = directional(ecs, {-0.4f, 1, 0.2f}, 3);
    }
}

void LightingValidation::setup(ECS& ecs, GpuScene& scene, TextureManager& textures) {
    if (!entities_.empty()) throw std::logic_error("Lighting fixture already set up");
    textures.createDefaultTextures(); // idempotent; no extra texture assets
    const auto plane = ProceduralMeshes::generatePlane(1, 1, 1, 1), cube = ProceduralMeshes::generateCube(0.5f);
    plane_ = scene.uploadMesh(plane.positions, plane.normals, plane.tangents, plane.uvs, plane.indices);
    cube_ = scene.uploadMesh(cube.positions, cube.normals, cube.tangents, cube.uvs, cube.indices);
    time_ = 0; cameraSegment_ = ~0u; materialStep_ = 0; positiveStepUpdates_ = 0;
    if (scenario_ == "cornell" || scenario_ == "thin-walls" || scenario_ == "moving-emissive" || scenario_ == "disocclusion" || scenario_ == "emissive-step") room(ecs);
    else exterior(ecs);
    if (scenario_ == "alpha-mip-shadow") {
        std::vector<u8> rgba(size_t(AlphaTextureSide) * AlphaTextureSide * 4, 255);
        for (u32 y = 0; y < AlphaTextureSide; ++y)
            for (u32 x = 0; x < AlphaTextureSide; ++x)
                rgba[(size_t(y) * AlphaTextureSide + x) * 4 + 3] = ((x ^ y) & 1u) ? 255 : 0;
        auto& material = ecs.getComponent<MaterialComponent>(alphaReceiver_);
        material.baseColorTexIndex = textures.loadTextureFromMemory(rgba.data(), AlphaTextureSide, AlphaTextureSide, 4, false);
        material.alphaCutoff = AlphaMipCutoff;
        ecs.markChanged<MaterialComponent>(alphaReceiver_);
    }
}

void LightingValidation::move(ECS& ecs, EntityID entity, glm::vec3 position, glm::quat rotation) {
    auto& transform = ecs.getComponent<TransformComponent>(entity); // tracked write
    transform.position = position; transform.rotation = rotation; transform.updateMatrix();
    ecs.markChanged<TransformComponent>(entity); // explicit ECS touch for the fixture contract
}
void LightingValidation::update(float dt, ECS& ecs) {
    if (!std::isfinite(dt) || dt < 0) throw std::invalid_argument("Lighting fixture dt must be finite and nonnegative");
    if (dt == 0 || entities_.empty()) return;
    time_ += dt;
    if (scenario_ == "emissive-step") {
        // First positive update precedes render frame0, hence update257 is
        // frame256. Saturation makes this a one-shot even in a long run.
        const bool transition = positiveStepUpdates_ == EmissiveStepFrame;
        if (positiveStepUpdates_ <= EmissiveStepFrame) ++positiveStepUpdates_;
        if (transition) {
            auto& material = ecs.getComponent<MaterialComponent>(panel_);
            material.emissiveFactor = glm::vec3(6.0f);
            ecs.markChanged<MaterialComponent>(panel_);
        }
        return; // Geometry/camera stay identical to static Cornell thereafter.
    }
    // All animation frequencies are integer multiples of 1/20 rad/s; keep
    // trigonometric inputs bounded even for a large finite scripted dt.
    const float t = float(std::fmod(time_, 40.0 * double(pi)));
    if (scenario_ == "moving-sun") {
        move(ecs, sun_, {}, orientLight({0.6f * std::sin(t * 0.5f), 1, 0.6f * std::cos(t * 0.5f)}));
    } else if (scenario_ == "moving-emissive") {
        move(ecs, panel_, {0.9f * std::sin(t * 0.7f), 3.98f, -0.3f + 0.45f * std::sin(t * 0.4f)},
             glm::angleAxis(pi, glm::vec3(0, 0, 1)));
        auto& material = ecs.getComponent<MaterialComponent>(panel_);
        material.emissiveFactor = glm::vec3(12.0f + 4.0f * std::sin(t * 0.35f));
        ecs.markChanged<MaterialComponent>(panel_);
    } else if (scenario_ == "disocclusion") {
        move(ecs, mover_, {1.3f * std::sin(t * 1.2f), 1.8f, 1.1f}, identity);
    } else if (scenario_ == "cache-stress") {
        for (u32 i = 0; i < cacheCasters_.size(); ++i) {
            auto position = cacheOrigins_[i]; position.x += 0.15f * std::sin(t + float(i));
            move(ecs, cacheCasters_[i], position, glm::angleAxis(t * 0.25f + float(i) * 0.1f, glm::vec3(0, 1, 0)));
        }
        move(ecs, sun_, {}, orientLight({-0.4f + 0.2f * std::sin(t * 0.5f), 1, 0.2f}));
        const u32 step = u32(std::fmod(std::floor(time_ / 0.5), double(cacheCasters_.size())));
        if (step != materialStep_) {
            materialStep_ = step;
            const EntityID e = cacheCasters_[step % cacheCasters_.size()];
            auto& material = ecs.getComponent<MaterialComponent>(e);
            material.baseColorFactor = glm::vec4(step % 2 ? glm::vec3(0.4f) : glm::vec3(0.8f), 1);
            ecs.markChanged<MaterialComponent>(e);
        }
    }
    // Cornell/thin-walls/offscreen/bias remain deterministic static references.
}

void LightingValidation::teardown(ECS& ecs, GpuScene&) {
    for (EntityID e : entities_) ecs.destroyEntity(e);
    entities_.clear(); cacheCasters_.clear(); cacheOrigins_.clear();
    panel_ = sun_ = mover_ = thinWall_ = embeddedSolid_ = alphaReceiver_ = INVALID_ENTITY;
    plane_ = cube_ = ~0u; time_ = 0; cameraSegment_ = ~0u; materialStep_ = 0; positiveStepUpdates_ = 0;
}
CameraSetup LightingValidation::getDefaultCamera() const {
    if (scenario_ == "offscreen-caster") return {{0, 2, 7}, {0, 0, 0}, 7, false};
    if (scenario_ == "shadow-bias") return {{0, 4, 7}, {0, 0.5f, 0}, 8, false};
    if (scenario_ == "shadow-penumbra") return {{0, 3, 6}, {0, 0, 0}, 6.708204f, false};
    if (scenario_ == "alpha-mip-shadow") return {{0, 1.5f, 4}, {0, 1.5f, 0}, 4, false};
    if (scenario_ == "moving-sun" || scenario_ == "cache-stress") return {{0, 5, 9}, {0, 0.5f, 0}, 10, false};
    if (scenario_ == "thin-walls") return {{1.4f, 2, 7.5f}, {0.2f, 1.9f, 0}, 7.5f, false};
    return {{0, 2, 7.5f}, {0, 2, 0}, 7.5f, false};
}
bool LightingValidation::scriptedCamera(double time, glm::vec3& position, glm::vec3& target, bool& cut) const {
    cut = false;
    if (scenario_ != "disocclusion" || !std::isfinite(time) || time < 0) return false;
    const double phase = std::fmod(time, 8.0);
    const u32 segment = phase >= 2 ? 1u : 0u;
    // Teleport at t=2 and each eight-second loop boundary. Report exactly once
    // for each transition; first script sample is covered by normal scene reset.
    cut = cameraSegment_ != ~0u && cameraSegment_ != segment;
    cameraSegment_ = segment;
    position = phase < 2 ? glm::vec3(0.15f * std::sin(float(phase)), 2, 7.5f) : glm::vec3(1.4f, 2, 7.5f);
    target = {0, 2, 0};
    return true;
}
} // namespace phosphor
