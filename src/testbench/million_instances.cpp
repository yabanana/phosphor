#include "testbench/million_instances.h"
#include "scene/ecs.h"
#include "scene/procedural.h"
#include "scene/texture_manager.h"
#include "renderer/gpu_scene.h"
#include "core/log.h"

#include <glm/gtc/constants.hpp>
#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <cmath>

namespace phosphor {

namespace {

constexpr u32 kShapeCount = 8;

/// Non-uniform scale of a mesh (positive factors): positions scale, normals
/// by the inverse (so they stay perpendicular), tangents with the matrix.
void scaleMesh(MeshData& m, const glm::vec3& s) {
    for (glm::vec3& p : m.positions) p *= s;
    for (glm::vec3& n : m.normals) n = glm::normalize(n / s);
    for (glm::vec4& t : m.tangents) {
        const glm::vec3 v = glm::normalize(glm::vec3(t) * s);
        t = glm::vec4(v, t.w);
    }
}

/// Shape `shape` (0..7) at variant `variant`: variant 0 is the base shape,
/// later variants change size and proportions so every one of the K meshes
/// is distinct.  All shapes have 8-48 triangles.
MeshData makeShape(u32 shape, u32 variant) {
    using namespace ProceduralMeshes;
    MeshData m;
    switch (shape % kShapeCount) {
    case 0: m = generateCube(0.5f); break;                          // 12 triangles
    case 1: m = generateIcosahedron(0.5f, 0, true); break;          // 20
    case 2: m = generateOctahedron(0.6f, true); break;              // 8
    case 3: m = generateTorus(0.35f, 0.15f, 6, 4); break;           // 48
    case 4: m = generateCube(0.5f); scaleMesh(m, {0.6f, 1.4f, 0.6f}); break;   // tall box, 12
    case 5: m = generateIcosahedron(0.5f, 0, false); scaleMesh(m, {1.2f, 0.5f, 1.2f}); break; // 20
    case 6: m = generateOctahedron(0.5f, false); scaleMesh(m, {0.6f, 1.3f, 0.6f}); break;     // 8
    default: m = generateTorus(0.3f, 0.12f, 5, 3); break;           // 30
    }
    if (variant > 0) {
        // Deterministic, different for every variant (up to 128 per shape).
        const float uniform = 0.7f + 0.05f * static_cast<float>(variant % 13);
        const float ax = 1.0f + 0.04f * static_cast<float>(variant % 7);
        const float az = 1.0f - 0.03f * static_cast<float>(variant % 5) + 0.002f * static_cast<float>(variant);
        scaleMesh(m, glm::vec3(ax, 1.0f, az) * uniform);
    }
    return m;
}

} // namespace

MillionInstances::MillionInstances(const TestBenchParams& params) {
    if (params.instances > 0) instances_ = params.instances;
    if (params.meshes > 0) meshes_ = std::min(params.meshes, MAX_MESHES);
    if (params.dynamicCpuPercent >= 0.0f) dynamicPercent_ = std::min(params.dynamicCpuPercent, 100.0f);
    churn_ = params.churn;
    // The slab keeps the instance density of the 1M default whatever N is.
    extent_ = std::max(20.0f, 300.0f * std::cbrt(static_cast<float>(instances_) / static_cast<float>(DEFAULT_INSTANCES)));
}

void MillionInstances::mirrorSome(std::mt19937& rng, TransformComponent& xf) const {
    // ~5 %: one negative scale component (the instance is drawn mirrored).
    if (std::uniform_int_distribution<u32>(0, 99)(rng) < 5) xf.scale.x = -xf.scale.x;
}

EntityID MillionInstances::createInstance(ECS& ecs, std::mt19937& rng, TransformComponent xf,
                                          const MotionComponent* motion, EntityID parent) {
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    const EntityID e = ecs.createEntity();
    xf.updateMatrix();
    ecs.addComponent(e, std::move(xf));

    MeshInstanceComponent inst{};
    inst.meshHandle    = meshBase_ + std::uniform_int_distribution<u32>(0, meshes_ - 1)(rng);
    inst.materialIndex = libraryBase_ + std::uniform_int_distribution<u32>(0, MATERIAL_COUNT - 1)(rng);
    inst.setVisible(true);
    inst.setCastsShadows(false);
    ecs.addComponent(e, std::move(inst));

    if (motion) {
        MotionComponent m = *motion;
        ecs.addComponent(e, std::move(m));
    }
    if (parent != INVALID_ENTITY) {
        HierarchyComponent h{};
        h.parent = parent;
        ecs.addComponent(e, std::move(h));
    }
    // ~1 %: an entity with its own material (instead of the library).
    if (std::uniform_int_distribution<u32>(0, 99)(rng) == 0) {
        MaterialComponent mat{};
        mat.baseColorFactor = glm::vec4(0.1f + 0.9f * unit(rng), 0.1f + 0.9f * unit(rng), 0.1f + 0.9f * unit(rng), 1.0f);
        mat.metallicFactor  = unit(rng);
        mat.roughnessFactor = 0.1f + 0.9f * unit(rng);
        mat.baseColorTexIndex         = whiteTex_;
        mat.normalTexIndex            = normalTex_;
        mat.metallicRoughnessTexIndex = mrTex_;
        ecs.addComponent(e, std::move(mat));
    }
    return e;
}

namespace {

glm::quat randomRotation(std::mt19937& rng) {
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    for (;;) {
        const glm::vec4 v(d(rng), d(rng), d(rng), d(rng));
        const float len = glm::length(v);
        if (len > 0.1f) return glm::quat(v.w / len, v.x / len, v.y / len, v.z / len);
    }
}

} // namespace

EntityID MillionInstances::createLeafRoot(ECS& ecs, std::mt19937& rng) {
    std::uniform_real_distribution<float> posXZ(-extent_, extent_);
    std::uniform_real_distribution<float> posY(-0.3f * extent_, 0.3f * extent_);
    std::uniform_real_distribution<float> scaleDist(0.5f, 1.5f);
    TransformComponent xf{};
    xf.position = glm::vec3(posXZ(rng), posY(rng), posXZ(rng));
    xf.rotation = randomRotation(rng);
    xf.scale    = glm::vec3(scaleDist(rng));
    mirrorSome(rng, xf);
    return createInstance(ecs, rng, xf, nullptr, INVALID_ENTITY);
}

void MillionInstances::setup(ECS& ecs, GpuScene& gpuScene, TextureManager& textures) {
    textures.createDefaultTextures();
    whiteTex_  = textures.getDefaultWhite();
    normalTex_ = textures.getDefaultNormal();
    mrTex_     = textures.getDefaultMR();
    LOG_INFO("MillionInstances: setting up %u instances, %u meshes, %.2f%% CPU-dynamic, churn %u...", instances_,
             meshes_, dynamicPercent_, churn_);

    // --- Meshes: K distinct ones (shape = k % 8, variant = k / 8) ---
    for (u32 k = 0; k < meshes_; ++k) {
        const MeshData m = makeShape(k % kShapeCount, k / kShapeCount);
        const MeshHandle h = gpuScene.uploadMesh(m.positions, m.normals, m.tangents, m.uvs, m.indices);
        if (k == 0) meshBase_ = h;
    }

    // --- Material library: 256 entries, every 20th double sided (5 %) ---
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    for (u32 i = 0; i < MATERIAL_COUNT; ++i) {
        GPUMaterial gm{};
        gm.baseColor[0] = 0.1f + 0.9f * unit(rng);
        gm.baseColor[1] = 0.1f + 0.9f * unit(rng);
        gm.baseColor[2] = 0.1f + 0.9f * unit(rng);
        gm.baseColor[3] = 1.0f;
        gm.metallic             = unit(rng);
        gm.roughness            = 0.1f + 0.9f * unit(rng);
        gm.normalScale          = 1.0f;
        gm.occlusionStrength    = 1.0f;
        gm.baseColorTex         = whiteTex_;
        gm.normalTex            = normalTex_;
        gm.metallicRoughnessTex = mrTex_;
        gm.occlusionTex         = INVALID_TEXTURE_INDEX;
        gm.emissiveTex          = INVALID_TEXTURE_INDEX;
        gm.alphaCutoff = 0.0f;
        gm.flags                = (i % 20 == 7) ? MATERIAL_FLAG_DOUBLE_SIDED : 0u;
        const u32 idx = gpuScene.addMaterial(gm);
        if (i == 0) libraryBase_ = idx;
    }

    // --- Instances ---
    const u32 satellites = static_cast<u32>(static_cast<u64>(instances_) / 10u);
    const u32 roots      = instances_ - satellites;
    // Moving roots: ~90 % of the roots; at least one when there are roots to move.
    u32 moving = static_cast<u32>(static_cast<u64>(roots) * 9u / 10u);
    if (satellites > 0 && moving == 0) moving = std::min(roots, 1u);
    const u32 staticRoots = roots - moving;

    records_.clear();
    records_.reserve(static_cast<size_t>(instances_) + churn_);
    leaves_.clear();

    std::uniform_real_distribution<float> posXZ(-extent_, extent_);
    std::uniform_real_distribution<float> scaleDist(0.5f, 1.5f);
    std::uniform_real_distribution<float> angle(0.0f, glm::two_pi<float>());

    // Moving roots: translation 0 in the TransformComponent, the motion
    // (centre, radius, phase, height, speed class) places them.
    std::vector<u32> movingIdx; // indices into records_
    movingIdx.reserve(moving);
    for (u32 i = 0; i < moving; ++i) {
        TransformComponent xf{};
        xf.rotation = randomRotation(rng);
        xf.scale    = glm::vec3(scaleDist(rng));
        mirrorSome(rng, xf);
        MotionComponent mo{};
        mo.centre     = glm::vec3(posXZ(rng), (unit(rng) - 0.5f) * 0.6f * extent_, posXZ(rng));
        mo.radius     = 0.5f + 5.5f * unit(rng);
        mo.phase      = angle(rng);
        mo.height     = (unit(rng) - 0.5f) * 6.0f;
        mo.speedClass = std::uniform_int_distribution<u32>(0, SCENE_MOTION_CLASSES - 1)(rng);
        movingIdx.push_back(static_cast<u32>(records_.size()));
        records_.push_back({createInstance(ecs, rng, xf, &mo, INVALID_ENTITY)});
    }
    // Plain roots without motion: leaves (nothing is parented to them), the
    // pool the churn draws from.
    for (u32 i = 0; i < staticRoots; ++i) {
        leaves_.push_back(static_cast<u32>(records_.size()));
        records_.push_back({createLeafRoot(ecs, rng)});
    }

    // Satellites: children of moving roots (70 %) or of earlier satellites
    // (30 %) down to depth MAX_SATELLITE_DEPTH.  TransformComponent = local.
    struct Sat { u32 record; u32 depth; };
    std::vector<Sat> eligible; // satellites that may still take children
    for (u32 i = 0; i < satellites && !movingIdx.empty(); ++i) {
        EntityID parent;
        u32 depth;
        if (!eligible.empty() && std::uniform_int_distribution<u32>(0, 99)(rng) < 30) {
            const Sat& p = eligible[std::uniform_int_distribution<size_t>(0, eligible.size() - 1)(rng)];
            parent = records_[p.record].entity;
            depth  = p.depth + 1;
        } else {
            parent = records_[movingIdx[std::uniform_int_distribution<size_t>(0, movingIdx.size() - 1)(rng)]].entity;
            depth  = 1;
        }
        TransformComponent xf{};
        const float r = 1.0f + 2.0f * unit(rng);
        const float a = angle(rng), b = angle(rng);
        xf.position = r * glm::vec3(std::cos(a) * std::cos(b), std::sin(b), std::sin(a) * std::cos(b));
        xf.rotation = randomRotation(rng);
        xf.scale    = glm::vec3(0.3f + 0.4f * unit(rng));
        mirrorSome(rng, xf);
        const u32 rec = static_cast<u32>(records_.size());
        records_.push_back({createInstance(ecs, rng, xf, nullptr, parent)});
        if (depth < MAX_SATELLITE_DEPTH) eligible.push_back({rec, depth});
    }
    // Instances the loops above could not place as satellites (no moving root
    // to hang them on) become leaf roots, so the requested count is exact.
    while (records_.size() < instances_) {
        leaves_.push_back(static_cast<u32>(records_.size()));
        records_.push_back({createLeafRoot(ecs, rng)});
    }

    // --- Lights: one sun and a few point lights over the slab ---
    auto addLight = [&](LightComponent light, glm::vec3 position) {
        const EntityID e = ecs.createEntity();
        lights_.push_back(e);
        TransformComponent xf{};
        xf.position = position;
        xf.updateMatrix();
        ecs.addComponent(e, std::move(xf));
        ecs.addComponent(e, std::move(light));
    };
    LightComponent sun{};
    sun.type      = LightType::Directional;
    sun.color     = glm::vec3(1.0f, 0.95f, 0.9f);
    sun.intensity = 4.0f;
    addLight(sun, glm::vec3(0.0f, 100.0f, 0.0f));
    for (int i = 0; i < 4; ++i) {
        LightComponent pl{};
        pl.type      = LightType::Point;
        pl.color     = glm::vec3(i % 2 ? 1.0f : 0.4f, 0.6f, i % 2 ? 0.4f : 1.0f);
        pl.intensity = 2000.0f;
        pl.range     = 0.5f * extent_;
        addLight(pl, glm::vec3((i & 1 ? 0.5f : -0.5f) * extent_, 0.15f * extent_, (i & 2 ? 0.5f : -0.5f) * extent_));
    }

    frame_ = 0;
    LOG_INFO("MillionInstances: setup complete (%u instances: %u moving roots, %u static leaves, %u satellites)",
             static_cast<u32>(records_.size()), moving, staticRoots,
             static_cast<u32>(records_.size()) - roots);
}

void MillionInstances::update(float dt, ECS& ecs) {
    const u32 n = static_cast<u32>(records_.size());
    if (n == 0) return;

    // CPU-driven transform changes: a window of `count` instances that
    // advances by `count` every frame (every instance is touched in turn).
    // getComponent on the non-const ECS marks the transform changed (F5 ECS
    // tracking contract); the rotation keeps the translation untouched, so
    // moving roots (translation 0, placed by their motion) stay valid.
    const u32 count = std::min(n, static_cast<u32>(std::llround(static_cast<double>(n) * dynamicPercent_ / 100.0)));
    if (count > 0) {
        const glm::quat spin = glm::angleAxis(dt * 0.5f, glm::vec3(0.0f, 1.0f, 0.0f));
        u64 at = (frame_ * count) % n;
        for (u32 i = 0; i < count; ++i) {
            TransformComponent& xf = ecs.getComponent<TransformComponent>(records_[at].entity);
            xf.rotation = glm::normalize(spin * xf.rotation);
            xf.updateMatrix();
            if (++at == n) at = 0;
        }
    }

    // Churn: destroy and create `churn_` leaf roots (never satellites, never
    // a parent), so the instance count stays constant.
    const u32 churn = std::min<u32>(churn_, static_cast<u32>(leaves_.size()));
    if (churn > 0) {
        std::mt19937 rng(static_cast<u32>(0x9E3779B9u * (frame_ + 1)));
        // Distinct victims: a partial Fisher-Yates shuffle of the pool.
        for (u32 i = 0; i < churn; ++i) {
            const size_t j = i + std::uniform_int_distribution<size_t>(0, leaves_.size() - 1 - i)(rng);
            std::swap(leaves_[i], leaves_[j]);
            const u32 rec = leaves_[i];
            ecs.destroyEntity(records_[rec].entity);
            records_[rec].entity = createLeafRoot(ecs, rng);
        }
    }
    ++frame_;
}

void MillionInstances::teardown(ECS& ecs, [[maybe_unused]] GpuScene& gpuScene) {
    for (const Record& r : records_) ecs.destroyEntity(r.entity);
    for (EntityID e : lights_) ecs.destroyEntity(e);
    records_.clear();
    leaves_.clear();
    lights_.clear();
}

CameraSetup MillionInstances::getDefaultCamera() const {
    // Inside the slab, near its +Z edge, looking at the middle: about a third
    // of the instances are in the frustum.
    CameraSetup cam{};
    cam.position = glm::vec3(0.0f, 0.2f * extent_, 0.9f * extent_);
    cam.target   = glm::vec3(0.0f, 0.0f, 0.0f);
    cam.distance = 0.9f * extent_;
    cam.orbit    = false;
    return cam;
}

} // namespace phosphor
