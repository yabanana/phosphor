#include <doctest/doctest.h>

#include "null_texture_manager.h"
#include "renderer/gpu_scene.h"
#include "renderer/scene_extract.h"
#include "scene/ecs.h"
#include "testbench/million_instances.h"
#include "testbench/testbench.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <utility>
#include <string>

using namespace phosphor;
using phosphor::test::NullTextureManager;

TEST_CASE("every test bench builds a renderable scene") {
    for (int i = 0; i < testBenchCount(); ++i) {
        const auto type = static_cast<TestBenchType>(i);
        CAPTURE(testBenchName(type));

        ECS ecs;
        GpuScene scene;
        NullTextureManager textures;

        TestBenchParams params;
        params.instances = 2000; // bench 8 defaults to 1M instances: too slow for a unit test
        auto bench = createTestBench(type, params);
        REQUIRE(bench);
        bench->setup(ecs, scene, textures);
        bench->update(1.0f / 60.0f, ecs);

        FrameScene frame;
        extractFrameScene(ecs, scene, frame);

        CHECK(scene.getMeshCount() > 0);
        CHECK(!frame.instances.empty());
        CHECK(!frame.lights.empty());
        u32 drawn = 0;
        for (const DrawBatch& b : frame.batches) {
            CHECK(b.meshIndex < scene.getMeshCount());
            drawn += b.instanceCount;
        }
        CHECK(drawn == frame.instances.size());
        for (const GPUInstance& inst : frame.instances) {
            CHECK(inst.materialIndex < frame.materials.size());
        }
        for (const GPUMaterial& m : frame.materials) {
            for (u32 tex : {m.baseColorTex, m.normalTex, m.metallicRoughnessTex}) {
                CHECK((tex == INVALID_TEXTURE_INDEX || tex < textures.textureCount()));
            }
        }

        bench->teardown(ecs, scene);
    }
}

// ---------------------------------------------------------------------------
// Bench 8, "1M Instances (dynamic)" (F5.6), with small parameters.
// ---------------------------------------------------------------------------

namespace {

struct Bench8 {
    ECS ecs;
    GpuScene scene;
    NullTextureManager textures;
    std::unique_ptr<TestBench> bench;

    explicit Bench8(const TestBenchParams& params) {
        bench = createTestBench(TestBenchType::MillionInstances, params);
        bench->setup(ecs, scene, textures);
    }
};

TestBenchParams smallParams(u32 instances = 2000) {
    TestBenchParams p;
    p.instances = instances;
    return p;
}

u32 hierarchyDepth(ECS& ecs, EntityID e) {
    u32 depth = 0;
    while (ecs.hasComponent<HierarchyComponent>(e)) {
        e = ecs.getComponent<HierarchyComponent>(e).parent;
        ++depth;
        if (depth > 64) break; // a cycle would never end
    }
    return depth;
}

} // namespace

TEST_CASE("bench 8: factory and name table") {
    CHECK(testBenchCount() == 8);
    CHECK(static_cast<int>(TestBenchType::MillionInstances) == 7);
    CHECK(std::string(testBenchName(TestBenchType::MillionInstances)) == "1M Instances (dynamic)");
    CHECK(std::string(testBenchName(7)) == "1M Instances (dynamic)");
    auto bench = createTestBench(TestBenchType::MillionInstances);
    REQUIRE(bench);
    CHECK(std::string(bench->getName()) == "1M Instances (dynamic)");
    // The default parameters are the 1M scene.
    CHECK(dynamic_cast<MillionInstances&>(*bench).instances() == 1000000);
    CHECK(dynamic_cast<MillionInstances&>(*bench).meshes() == 8);
    CHECK(dynamic_cast<MillionInstances&>(*bench).dynamicCpuPercent() == doctest::Approx(1.0f));
}

TEST_CASE("bench 8: setup creates the requested population") {
    Bench8 b(smallParams(2000));
    ECS& ecs = b.ecs;
    auto& inst = ecs.getArray<MeshInstanceComponent>();
    auto& motion = ecs.getArray<MotionComponent>();
    auto& hier = ecs.getArray<HierarchyComponent>();
    auto& xf = ecs.getArray<TransformComponent>();

    CHECK(inst.size() == 2000);
    CHECK(b.scene.getMeshCount() == 8);
    CHECK(b.scene.materials().size() == 256);

    // Satellites ~10 %, depth 1..3, parents are instances; motion only on roots.
    CHECK(hier.size() == 200);
    u32 maxDepth = 0, depth2 = 0;
    for (u32 i = 0; i < hier.size(); ++i) {
        const EntityID e = hier.entities()[i];
        const EntityID parent = hier.data()[i].parent;
        CHECK(ecs.hasComponent<MeshInstanceComponent>(parent));
        const u32 d = hierarchyDepth(ecs, e);
        maxDepth = std::max(maxDepth, d);
        depth2 += d == 2;
        CHECK(d >= 1);
        CHECK(d <= 3);
        CHECK_FALSE(ecs.hasComponent<MotionComponent>(e));
    }
    CHECK(maxDepth <= 3);
    CHECK(depth2 > 0); // satellites of satellites exist
    // The root of every chain moves.
    for (const EntityID e : hier.entities()) {
        EntityID root = e;
        while (ecs.hasComponent<HierarchyComponent>(root)) root = ecs.getComponent<HierarchyComponent>(root).parent;
        CHECK(ecs.hasComponent<MotionComponent>(root));
    }

    // Motion: ~90 % of the roots, translation 0 in the transform, valid class.
    const u32 roots = inst.size() - hier.size();
    CHECK(motion.size() == roots * 9 / 10);
    for (u32 i = 0; i < motion.size(); ++i) {
        const EntityID e = motion.entities()[i];
        CHECK_FALSE(ecs.hasComponent<HierarchyComponent>(e));
        CHECK(motion.data()[i].speedClass < 64);
        CHECK(motion.data()[i].radius > 0.0f);
        CHECK(ecs.getComponent<TransformComponent>(e).position == glm::vec3(0.0f));
    }

    // ~5 % mirrored, ~1 % own materials, ~5 % of the library double sided.
    u32 mirrored = 0;
    for (const TransformComponent& t : xf.data()) {
        if (t.scale.x < 0.0f || t.scale.y < 0.0f || t.scale.z < 0.0f) ++mirrored;
    }
    CHECK(mirrored > 20);
    CHECK(mirrored < 250);
    CHECK(ecs.getArray<MaterialComponent>().size() > 0);
    CHECK(ecs.getArray<MaterialComponent>().size() < 100);
    u32 doubleSided = 0;
    for (const GPUMaterial& m : b.scene.materials()) doubleSided += (m.flags & MATERIAL_FLAG_DOUBLE_SIDED) != 0;
    CHECK(doubleSided > 0);
    CHECK(doubleSided < 40);
    for (const MeshInstanceComponent& m : inst.data()) {
        CHECK(m.meshHandle < b.scene.getMeshCount());
        CHECK(m.materialIndex < b.scene.materials().size());
        CHECK(m.isVisible());
    }

    // Lights: one directional plus point lights.
    u32 directional = 0, point = 0;
    for (const LightComponent& l : ecs.getArray<LightComponent>().data()) {
        directional += l.type == LightType::Directional;
        point += l.type == LightType::Point;
    }
    CHECK(directional == 1);
    CHECK(point >= 1);

    // The camera sits inside / next to the slab.
    const CameraSetup cam = b.bench->getDefaultCamera();
    const float e = dynamic_cast<MillionInstances&>(*b.bench).extent();
    CHECK(std::abs(cam.position.x) <= e);
    CHECK(std::abs(cam.position.z) <= 1.5f * e);

    // The extracted frame is consistent.
    FrameScene frame;
    extractFrameScene(ecs, b.scene, frame);
    CHECK(frame.instances.size() == 2000);
    u32 mirroredInstances = 0;
    for (const GPUInstance& gi : frame.instances) mirroredInstances += (gi.flags & INSTANCE_FLAG_MIRRORED) != 0;
    CHECK(mirroredInstances > 0);

    b.bench->teardown(ecs, b.scene);
    CHECK(inst.size() == 0);
    CHECK(xf.size() == 0);
    CHECK(motion.size() == 0);
    CHECK(hier.size() == 0);
    CHECK(ecs.getArray<LightComponent>().size() == 0);
    CHECK(ecs.getArray<MaterialComponent>().size() == 0);
}

TEST_CASE("bench 8: --scene-meshes gives K distinct low-poly meshes") {
    for (const u32 k : {1u, 8u, 100u, 1024u}) {
        CAPTURE(k);
        TestBenchParams p = smallParams(3000);
        p.meshes = k;
        Bench8 b(p);
        REQUIRE(b.scene.getMeshCount() == k);
        const auto& infos = b.scene.meshInfos();
        std::set<u64> hashes;
        for (u32 m = 0; m < k; ++m) {
            const u32 tris = infos[m].indexCount / 3;
            CHECK(tris >= 8);
            CHECK(tris <= 48);
            const u32 begin = infos[m].vertexOffset;
            const u32 end = m + 1 < k ? infos[m + 1].vertexOffset : static_cast<u32>(b.scene.vertices().size());
            u64 h = 1469598103934665603ull; // FNV-1a over the vertex bytes
            const auto* bytes = reinterpret_cast<const unsigned char*>(&b.scene.vertices()[begin]);
            for (size_t i = 0; i < static_cast<size_t>(end - begin) * sizeof(GPUVertex); ++i) {
                h = (h ^ bytes[i]) * 1099511628211ull;
            }
            hashes.insert(h);
        }
        CHECK(hashes.size() == k); // no two meshes alike
    }
    // More than 1024 is clamped.
    TestBenchParams p = smallParams(100);
    p.meshes = 5000;
    CHECK(dynamic_cast<MillionInstances&>(*createTestBench(TestBenchType::MillionInstances, p)).meshes() == 1024);
}

TEST_CASE("bench 8: the CPU-dynamic subset changes exactly dynamicCpuPercent of the transforms") {
    auto snapshot = [](ECS& ecs) {
        std::map<EntityID, glm::quat> out;
        auto& a = ecs.getArray<TransformComponent>();
        for (u32 i = 0; i < a.size(); ++i) out[a.entities()[i]] = a.data()[i].rotation;
        return out;
    };
    for (const float pct : {0.0f, 1.0f, 5.0f, 100.0f}) {
        CAPTURE(pct);
        TestBenchParams p = smallParams(2000);
        p.dynamicCpuPercent = pct;
        Bench8 b(p);
        const auto before = snapshot(b.ecs);
        b.bench->update(1.0f / 60.0f, b.ecs);
        const auto after = snapshot(b.ecs);
        u32 changed = 0;
        for (const auto& [e, q] : before) changed += after.at(e) != q;
        CHECK(changed == static_cast<u32>(std::llround(2000.0 * pct / 100.0)));
    }
    // The window advances: after 100 frames at 1 % every instance was touched.
    Bench8 b(smallParams(2000));
    std::set<EntityID> touched;
    auto prev = snapshot(b.ecs);
    for (int f = 0; f < 100; ++f) {
        b.bench->update(1.0f / 60.0f, b.ecs);
        const auto now = snapshot(b.ecs);
        for (const auto& [e, q] : now) {
            if (q != prev.at(e)) touched.insert(e);
        }
        prev = now;
    }
    // 2000 instances + 5 lights (lights are never touched).
    CHECK(touched.size() == 2000);
}

TEST_CASE("bench 8: deterministic, and churn keeps the count with leaf-only turnover") {
    TestBenchParams p = smallParams(2000);
    p.churn = 25;
    Bench8 a(p), c(p);
    for (int f = 0; f < 3; ++f) {
        a.bench->update(1.0f / 60.0f, a.ecs);
        c.bench->update(1.0f / 60.0f, c.ecs);
    }
    auto& xa = a.ecs.getArray<TransformComponent>();
    auto& xc = c.ecs.getArray<TransformComponent>();
    REQUIRE(xa.size() == xc.size());
    for (u32 i = 0; i < xa.size(); ++i) {
        CHECK(xa.entities()[i] == xc.entities()[i]);
        CHECK(xa.data()[i].worldMatrix == xc.data()[i].worldMatrix);
    }

    // One more frame on `a`: what disappeared were plain leaves.
    struct Info { bool motion, hier, parent; };
    std::map<EntityID, Info> before;
    std::set<EntityID> parents;
    for (const HierarchyComponent& h : a.ecs.getArray<HierarchyComponent>().data()) parents.insert(h.parent);
    for (const EntityID e : a.ecs.getArray<MeshInstanceComponent>().entities()) {
        before[e] = {a.ecs.hasComponent<MotionComponent>(e), a.ecs.hasComponent<HierarchyComponent>(e),
                     parents.count(e) > 0};
    }
    a.ecs.endFrame(); // only this frame's changes below
    a.bench->update(1.0f / 60.0f, a.ecs);
    auto& inst = a.ecs.getArray<MeshInstanceComponent>();
    CHECK(inst.size() == 2000);
    // Ids are recycled (ECS free list): count the removals and additions
    // the ECS change lists report for this frame.
    const auto changes = std::as_const(a.ecs).getArray<MeshInstanceComponent>().changes();
    const u32 destroyed = static_cast<u32>(changes.removed.size());
    const u32 created   = static_cast<u32>(changes.added.size());
    for (const EntityID e : changes.removed) {
        REQUIRE(before.count(e));
        CHECK_FALSE(before[e].motion);
        CHECK_FALSE(before[e].hier);
        CHECK_FALSE(before[e].parent);
    }
    for (const EntityID e : changes.added) {
        CHECK_FALSE(a.ecs.hasComponent<MotionComponent>(e));
        CHECK_FALSE(a.ecs.hasComponent<HierarchyComponent>(e));
        CHECK(a.ecs.hasComponent<TransformComponent>(e));
    }
    CHECK(destroyed == 25);
    CHECK(created == 25);
    // Every remaining satellite still has a live parent.
    for (const HierarchyComponent& h : a.ecs.getArray<HierarchyComponent>().data()) {
        CHECK(a.ecs.hasComponent<MeshInstanceComponent>(h.parent));
    }
    a.bench->teardown(a.ecs, a.scene);
    CHECK(inst.size() == 0);
}
