#include <doctest/doctest.h>

#include "renderer/gpu_scene.h"
#include "renderer/scene_extract.h"
#include "scene/components.h"
#include "scene/ecs.h"
#include "scene/procedural.h"

#include <algorithm>
#include <utility>
#include <vector>

using namespace phosphor;

namespace {

std::vector<EntityID> sorted(std::span<const EntityID> s) {
    std::vector<EntityID> v(s.begin(), s.end());
    std::sort(v.begin(), v.end());
    return v;
}

using V = std::vector<EntityID>;

struct Value {
    int v = 0;
};

ComponentArray<Value> makeArray(int n) {
    ComponentArray<Value> a;
    for (int i = 0; i < n; ++i) a.add(static_cast<EntityID>(i), Value{i});
    a.clearChanges();
    return a;
}

} // namespace

TEST_CASE("ecs changes: add is reported in `added`, not `changed`") {
    ComponentArray<Value> a;
    a.add(0, Value{1});
    a.add(1, Value{2});
    (void)a.get(1); // an added entity needs no second listing
    const auto ch = a.changes();
    CHECK(sorted(ch.added) == V{0, 1});
    CHECK(ch.changed.empty());
    CHECK(ch.removed.empty());
    CHECK_FALSE(ch.all);
    a.clearChanges();
    CHECK(a.changes().empty());
}

TEST_CASE("ecs changes: non-const get/modify/markChanged mark once, const does not") {
    ComponentArray<Value> a = makeArray(4);
    const ComponentArray<Value>& c = a;

    (void)c.get(1);
    (void)c.tryGet(2);
    (void)c.data();
    CHECK(a.changes().empty());

    a.get(1).v = 10;
    a.get(1).v = 11;       // deduplicated
    a.modify(1).v = 12;
    a.modify(3).v = 13;
    a.markChanged(2);
    a.markChanged(2);
    a.markChanged(99);     // unknown entity: ignored
    CHECK(sorted(a.changes().changed) == V{1, 2, 3});
    CHECK(a.changes().added.empty());
    CHECK(a.changes().removed.empty());

    a.clearChanges();
    CHECK(a.changes().empty());
    a.modify(1).v = 5; // marks again after the clear
    CHECK(sorted(a.changes().changed) == V{1});
}

TEST_CASE("ecs changes: non-const data() sets the all flag, const data() does not") {
    ComponentArray<Value> a = makeArray(3);
    const ComponentArray<Value>& c = a;
    (void)c.data();
    CHECK_FALSE(a.changes().all);
    for (Value& v : a.data()) v.v += 1;
    CHECK(a.changes().all);
    CHECK_FALSE(a.changes().empty());
    a.clearChanges();
    CHECK_FALSE(a.changes().all);
}

TEST_CASE("ecs changes: removal drops the entity from changed and reports it") {
    ComponentArray<Value> a = makeArray(4);
    (void)a.modify(1);
    (void)a.modify(2);
    a.remove(1);
    const auto ch = a.changes();
    CHECK(sorted(ch.changed) == V{2});
    CHECK(sorted(ch.removed) == V{1});
    a.remove(1); // already gone: no second record
    a.remove(77);
    CHECK(sorted(a.changes().removed) == V{1});
}

TEST_CASE("ecs changes: add then remove in one frame cancels") {
    ComponentArray<Value> a = makeArray(2);
    a.add(5, Value{5});
    a.add(6, Value{6});
    a.remove(5);
    const auto ch = a.changes();
    CHECK(sorted(ch.added) == V{6});
    CHECK(ch.removed.empty());
    CHECK_FALSE(a.has(5));
}

TEST_CASE("ecs changes: remove then add is reported in both lists") {
    ComponentArray<Value> a = makeArray(3);
    a.remove(1);
    a.add(1, Value{100});
    auto ch = a.changes();
    CHECK(sorted(ch.removed) == V{1});
    CHECK(sorted(ch.added) == V{1});
    CHECK(ch.changed.empty());
    CHECK(std::as_const(a).get(std::as_const(a).entities()[0]).v == 0);

    // remove, add, remove: only `removed`
    a.remove(1);
    ch = a.changes();
    CHECK(sorted(ch.removed) == V{1});
    CHECK(ch.added.empty());

    a.clearChanges();
    CHECK(a.changes().empty());
    CHECK(a.size() == 2);
}

TEST_CASE("ecs changes: the change flag moves with swap-and-pop") {
    ComponentArray<Value> a = makeArray(5); // dense index == entity id
    (void)a.modify(4);                            // last element, changed
    a.add(5, Value{5});                     // added (dense index 5)
    a.remove(0);                            // 5 swaps into index 0 (pending add moves with it)
    a.remove(1);                            // 4 swaps into index 1 (pending change moves with it)
    CHECK(sorted(a.changes().changed) == V{4});
    CHECK(sorted(a.changes().added) == V{5});
    CHECK(sorted(a.changes().removed) == V{0, 1});

    // The moved entities are still deduplicated at their new index.
    (void)a.modify(4);
    (void)a.modify(5);
    CHECK(sorted(a.changes().changed) == V{4});
    CHECK(a.changes().added.size() == 1);

    // Removing a moved, changed entity unlists it correctly (position fix-up).
    a.remove(4);
    CHECK(a.changes().changed.empty());
    a.clearChanges();
    (void)a.modify(2);
    (void)a.modify(3);
    CHECK(sorted(a.changes().changed) == V{2, 3});
    a.remove(2);
    CHECK(sorted(a.changes().changed) == V{3});
    a.clearChanges();
    (void)a.modify(3);
    CHECK(sorted(a.changes().changed) == V{3});
}

TEST_CASE("ecs changes: randomised bookkeeping matches a reference model") {
    ComponentArray<Value> a;
    std::vector<int> present(64, 0), changed(64, 0), added(64, 0), removed(64, 0);
    u32 seed = 12345;
    auto rnd = [&](u32 n) { seed = seed * 1664525u + 1013904223u; return (seed >> 8) % n; };
    for (int step = 0; step < 4000; ++step) {
        const u32 e = rnd(64);
        switch (rnd(4)) {
        case 0:
            if (!present[e]) {
                a.add(e, Value{});
                present[e] = 1;
                added[e] = 1;
                changed[e] = 0; // an added entity is not listed as changed
            }
            break;
        case 1:
            if (present[e]) {
                a.remove(e);
                present[e] = 0;
                changed[e] = 0;
                if (added[e]) added[e] = 0; // cancels
                else removed[e] = 1;
            }
            break;
        case 2:
            if (present[e]) {
                (void)a.modify(e);
                if (!added[e]) changed[e] = 1;
            }
            break;
        default:
            if (rnd(50) == 0) {
                a.clearChanges();
                std::fill(changed.begin(), changed.end(), 0);
                std::fill(added.begin(), added.end(), 0);
                std::fill(removed.begin(), removed.end(), 0);
            }
            break;
        }
        const auto ch = a.changes();
        std::vector<int> c(64, 0), ad(64, 0), rm(64, 0);
        for (EntityID x : ch.changed) { REQUIRE(c[x] == 0); c[x] = 1; }
        for (EntityID x : ch.added)   { REQUIRE(ad[x] == 0); ad[x] = 1; }
        for (EntityID x : ch.removed) { REQUIRE(rm[x] == 0); rm[x] = 1; }
        REQUIRE(c == changed);
        REQUIRE(ad == added);
        REQUIRE(rm == removed);
    }
}

TEST_CASE("ecs: const access marks nothing, non-const getComponent marks") {
    ECS ecs;
    const EntityID e = ecs.createEntity();
    ecs.addComponent(e, TransformComponent{});
    ecs.endFrame();
    const ECS& c = ecs;
    (void)c.getComponent<TransformComponent>(e);
    (void)c.getArray<TransformComponent>().get(e);
    (void)c.hasComponent<TransformComponent>(e);
    CHECK(c.getArray<TransformComponent>().changes().empty());

    ecs.getComponent<TransformComponent>(e).position.x = 3.0f;
    CHECK(c.getArray<TransformComponent>().changes().changed.size() == 1);
    ecs.endFrame();
    CHECK(c.getArray<TransformComponent>().changes().empty());

    ecs.markChanged<TransformComponent>(e);
    CHECK(c.getArray<TransformComponent>().changes().changed.size() == 1);
}

TEST_CASE("ecs: const getArray of an unregistered type is an empty view and does not register it") {
    ECS ecs;
    const ECS& c = ecs;
    const auto& arr = c.getArray<MotionComponent>();
    CHECK(arr.size() == 0);
    CHECK(arr.changes().empty());
    CHECK_FALSE(c.hasComponent<MotionComponent>(0));
    ecs.endFrame(); // nothing registered: no-op
}

TEST_CASE("ecs: destroyEntity records removals in every array, endFrame clears them") {
    ECS ecs;
    const EntityID a = ecs.createEntity();
    const EntityID b = ecs.createEntity();
    for (EntityID e : {a, b}) {
        ecs.addComponent(e, TransformComponent{});
        ecs.addComponent(e, MeshInstanceComponent{});
    }
    ecs.addComponent(a, MaterialComponent{}); // only `a` has it
    ecs.endFrame();

    ecs.destroyEntity(a);
    const ECS& c = ecs;
    CHECK(c.getArray<TransformComponent>().changes().removed.size() == 1);
    CHECK(c.getArray<MeshInstanceComponent>().changes().removed.size() == 1);
    CHECK(c.getArray<MaterialComponent>().changes().removed.size() == 1);
    CHECK(c.getArray<TransformComponent>().changes().removed[0] == a);

    ecs.destroyEntity(b);
    CHECK(c.getArray<TransformComponent>().changes().removed.size() == 2);
    CHECK(c.getArray<MaterialComponent>().changes().removed.size() == 1); // b had none

    ecs.endFrame();
    CHECK(c.getArray<TransformComponent>().changes().empty());
    CHECK(c.getArray<MeshInstanceComponent>().changes().empty());
    CHECK(c.getArray<MaterialComponent>().changes().empty());
}

TEST_CASE("ecs: a full scene extraction pass marks nothing (static scene = zero changes)") {
    GpuScene scene;
    const MeshData cube = ProceduralMeshes::generateCube(1.0f);
    const MeshHandle mesh = scene.uploadMesh(cube.positions, cube.normals, cube.tangents, cube.uvs, cube.indices);
    scene.addMaterial(GPUMaterial{});

    ECS ecs;
    for (int i = 0; i < 20; ++i) {
        const EntityID e = ecs.createEntity();
        TransformComponent xf{};
        xf.position = glm::vec3(static_cast<float>(i), 0, 0);
        xf.updateMatrix();
        ecs.addComponent(e, std::move(xf));
        MeshInstanceComponent mi{};
        mi.meshHandle = mesh;
        mi.materialIndex = 0;
        mi.setVisible(true);
        ecs.addComponent(e, std::move(mi));
        if (i % 2) ecs.addComponent(e, MaterialComponent{});
        if (i % 5 == 0) {
            LightComponent lc{};
            lc.type = LightType::Point;
            ecs.addComponent(e, std::move(lc));
        }
    }
    ecs.endFrame();

    FrameScene frame;
    extractFrameScene(ecs, scene, frame);
    std::vector<GPULight> lights;
    extractLights(ecs, lights);

    CHECK(frame.instances.size() == 20);
    CHECK(lights.size() == 4);
    const ECS& c = ecs;
    CHECK(c.getArray<TransformComponent>().changes().empty());
    CHECK(c.getArray<MeshInstanceComponent>().changes().empty());
    CHECK(c.getArray<MaterialComponent>().changes().empty());
    CHECK(c.getArray<LightComponent>().changes().empty());

    // Negative control: a mutable read of the same arrays does mark them.
    auto& xf = ecs.getArray<TransformComponent>();
    for (u32 i = 0; i < xf.size(); ++i) (void)xf.get(xf.entities()[i]);
    CHECK(c.getArray<TransformComponent>().changes().changed.size() == 20);
}
