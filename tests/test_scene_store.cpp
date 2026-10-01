#include <doctest/doctest.h>

#include "renderer/gpu_scene.h"
#include "renderer/scene_extract.h"
#include "renderer/scene_store.h"
#include "scene/components.h"
#include "scene/ecs.h"
#include "scene/procedural.h"
#include "testbench/testbench.h"
#include "null_texture_manager.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace phosphor;

// Allocation counter: the global operator new replacement lives in
// test_worker_pool.cpp (counting is only on inside the measured region).
extern std::atomic<bool> g_countAllocs;
extern std::atomic<u64>  g_allocs;

namespace {

constexpr u32 NONE = ~0u;

glm::mat4 toMat(const float* p) {
    glm::mat4 m;
    std::memcpy(&m[0][0], p, 16 * sizeof(float));
    return m;
}

// Stand-in for the GPU motion kernel (same expression on both sides).
glm::mat4 motionWorld(glm::vec3 centre, float r, float cp, float sp, float h, const glm::mat4& base) {
    return glm::translate(glm::mat4(1.0f), centre + glm::vec3(r * cp, h, r * sp)) * base;
}

glm::mat4 expectedWorld(const ECS& ecs, EntityID e) {
    const TransformComponent& xf = ecs.getComponent<TransformComponent>(e);
    if (ecs.hasComponent<HierarchyComponent>(e)) {
        return expectedWorld(ecs, ecs.getComponent<HierarchyComponent>(e).parent) * xf.worldMatrix;
    }
    if (ecs.hasComponent<MotionComponent>(e)) {
        const MotionComponent& m = ecs.getComponent<MotionComponent>(e);
        return motionWorld(m.centre, m.radius, std::cos(m.phase), std::sin(m.phase), m.height, xf.worldMatrix);
    }
    return xf.worldMatrix;
}

// ---------------------------------------------------------------------------
// A scene + ECS + store, with helpers to build entities.
// ---------------------------------------------------------------------------
struct World {
    GpuScene scene;
    ECS      ecs;
    SceneStore store;
    std::vector<MeshHandle> meshes;
    std::vector<EntityID>   all;

    explicit World(u32 libMaterials = 2) {
        const MeshData cube = ProceduralMeshes::generateCube(1.0f);
        const MeshData plane = ProceduralMeshes::generatePlane(1.0f, 1.0f, 1, 1);
        const MeshData sphere = ProceduralMeshes::generateSphere(0.5f, 8, 6);
        for (const MeshData* m : {&cube, &plane, &sphere}) {
            meshes.push_back(scene.uploadMesh(m->positions, m->normals, m->tangents, m->uvs, m->indices));
        }
        for (u32 i = 0; i < libMaterials; ++i) {
            GPUMaterial g{};
            g.roughness = 0.1f * static_cast<float>(i + 1);
            if (i == 1) g.flags = MATERIAL_FLAG_DOUBLE_SIDED;
            scene.addMaterial(g);
        }
    }

    static TransformComponent xform(glm::vec3 pos, glm::vec3 scale = glm::vec3(1.0f), float angle = 0.0f) {
        TransformComponent xf{};
        xf.position = pos;
        xf.scale = scale;
        xf.rotation = glm::angleAxis(angle, glm::vec3(0, 1, 0));
        xf.updateMatrix();
        return xf;
    }

    EntityID add(u32 meshIdx, glm::vec3 pos = glm::vec3(0.0f), glm::vec3 scale = glm::vec3(1.0f), u32 mat = 0,
                 bool ownMat = false, bool doubleSided = false) {
        const EntityID e = ecs.createEntity();
        all.push_back(e);
        ecs.addComponent(e, xform(pos, scale, 0.1f * static_cast<float>(e)));
        MeshInstanceComponent mi{};
        mi.meshHandle = meshIdx < meshes.size() ? meshes[meshIdx] : 1000 + meshIdx;
        mi.materialIndex = mat;
        mi.setVisible(true);
        ecs.addComponent(e, std::move(mi));
        if (ownMat) {
            MaterialComponent mc{};
            mc.doubleSided = doubleSided;
            mc.roughnessFactor = 0.01f * static_cast<float>(e % 90);
            ecs.addComponent(e, std::move(mc));
        }
        return e;
    }

    EntityID child(EntityID parent, u32 meshIdx, glm::vec3 pos, glm::vec3 scale = glm::vec3(1.0f), u32 mat = 0,
                   bool ownMat = false) {
        const EntityID e = add(meshIdx, pos, scale, mat, ownMat);
        ecs.addComponent(e, HierarchyComponent{parent});
        return e;
    }

    void addMotion(EntityID e, float phase = 0.5f, u32 speedClass = 3) {
        MotionComponent m{};
        m.centre = glm::vec3(1.0f, 2.0f, 3.0f);
        m.radius = 4.0f;
        m.phase = phase;
        m.height = 0.5f;
        m.speedClass = speedClass;
        ecs.addComponent(e, std::move(m));
    }

    // Many per-entity materials so the material buffer is big enough for delta records
    // (a buffer whose changed records exceed 1/8 of its size is copied in full).
    void padMaterials(u32 n) {
        for (u32 i = 0; i < n; ++i) add(2, glm::vec3(1000.0f + static_cast<float>(i), 0, 0), glm::vec3(1.0f), 0, true);
    }

    void sync() { store.sync(ecs, scene); }
    std::string verify() const { return store.verifyAgainstEcs(ecs, scene); }
    // sync + verify + clear the ECS change lists (what the engine does per frame)
    void frame() {
        sync();
        const std::string v = verify();
        REQUIRE_MESSAGE(v.empty(), v);
        ecs.endFrame();
    }
    void move(EntityID e, glm::vec3 pos) {
        auto& xf = ecs.getComponent<TransformComponent>(e);
        xf.position = pos;
        xf.updateMatrix();
    }
};

// ---------------------------------------------------------------------------
// Simulated GPU: applies deltas / full copies / structure buffers and runs a
// reference of the motion + hierarchy kernels driven by dirtyRoots().
// ---------------------------------------------------------------------------
struct Sim {
    std::vector<GPUInstance> instRaw; // only CPU data (deltas / copies): compared byte for byte
    std::vector<GPUInstance> inst;    // the GPU's view: matrices of children / motion roots are computed
    std::vector<GPUMaterial> mat;
    std::vector<GPUTransformNode> node;
    std::vector<GPUMotion> motion;
    std::vector<GPUDrawBucket> buckets;
    std::vector<u32> cmds, csrOff, csrSlots, motionSlots;
    bool dropRecord = false; // negative control: lose one instance record per frame
    bool skipDirty  = false; // negative control: ignore dirtyRoots()

    template <typename T>
    static bool applyBuf(std::vector<T>& dst, std::span<const T> mir, bool full, std::span<const GPUDeltaRecord> recs,
                         bool drop) {
        if (full) {
            dst.assign(mir.begin(), mir.end());
        } else {
            if (dst.size() != mir.size()) return false;
            bool dropped = false;
            for (const GPUDeltaRecord& r : recs) {
                if (r.slot >= dst.size()) return false;
                if (drop && !dropped) { dropped = true; continue; }
                std::memcpy(&dst[r.slot], r.payload, sizeof(T));
            }
        }
        return dst.size() == mir.size() && (dst.empty() || std::memcmp(dst.data(), mir.data(), dst.size() * sizeof(T)) == 0);
    }

    template <typename T>
    static bool same(const std::vector<T>& a, std::span<const T> b) {
        return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
    }

    struct Result { bool bytesOk = true, worldOk = true; };

    void visit(u32 p) {
        for (u32 i = csrOff[p]; i < csrOff[p + 1]; ++i) {
            const u32 c = csrSlots[i];
            const glm::mat4 w = toMat(inst[p].modelMatrix) * toMat(node[c].local);
            std::memcpy(inst[c].modelMatrix, &w[0][0], sizeof(inst[c].modelMatrix));
            visit(c);
        }
    }

    Result step(const SceneStore& s, const ECS& ecs, const std::vector<EntityID>& all) {
        Result r;
        const SceneSyncStats& st = s.stats();
        r.bytesOk &= applyBuf(instRaw, s.instances(), st.fullInstances, s.instanceDeltas(), dropRecord);
        applyBuf(inst, s.instances(), st.fullInstances, s.instanceDeltas(), dropRecord);
        r.bytesOk &= applyBuf(mat, s.materials(), st.fullMaterials, s.materialDeltas(), false);
        r.bytesOk &= applyBuf(node, s.nodes(), st.fullNodes, s.nodeDeltas(), false);
        r.bytesOk &= applyBuf(motion, s.motions(), st.fullMotions, s.motionDeltas(), false);
        if (st.structure) {
            buckets.assign(s.gpuBuckets().begin(), s.gpuBuckets().end());
            cmds.assign(s.commandBuckets().begin(), s.commandBuckets().end());
            csrOff.assign(s.childOffsets().begin(), s.childOffsets().end());
            csrSlots.assign(s.childSlots().begin(), s.childSlots().end());
            motionSlots.assign(s.motionSlots().begin(), s.motionSlots().end());
        }
        r.bytesOk &= same(buckets, s.gpuBuckets()) && same(cmds, s.commandBuckets()) && same(csrOff, s.childOffsets()) &&
                     same(csrSlots, s.childSlots()) && same(motionSlots, s.motionSlots());

        // GPU: motion, then hierarchy from the dirty roots.
        for (u32 slot : motionSlots) {
            const GPUMotion& m = motion[slot];
            glm::mat4 base(1.0f);
            for (int c = 0; c < 4; ++c) {
                for (int rr = 0; rr < 3; ++rr) base[c][rr] = m.base[c * 3 + rr];
            }
            const glm::mat4 w = motionWorld(glm::vec3(m.centre[0], m.centre[1], m.centre[2]), m.radius, m.cosPhase,
                                            m.sinPhase, m.height, base);
            std::memcpy(inst[slot].modelMatrix, &w[0][0], sizeof(inst[slot].modelMatrix));
        }
        if (!skipDirty) {
            std::vector<u32> seen(s.dirtyRoots().begin(), s.dirtyRoots().end());
            std::sort(seen.begin(), seen.end());
            if (std::adjacent_find(seen.begin(), seen.end()) != seen.end()) r.worldOk = false; // not deduplicated
            for (u32 root : s.dirtyRoots()) {
                if (root >= inst.size()) { r.worldOk = false; continue; }
                visit(root);
            }
            // The motion roots with children: their own persistent GPU queue.
            for (u32 root : s.motionParentSlots()) {
                if (root >= inst.size()) { r.worldOk = false; continue; }
                visit(root);
            }
        }
        for (EntityID e : all) {
            const u32 slot = s.slotOf(e);
            if (slot == NONE) continue;
            const glm::mat4 w = expectedWorld(ecs, e);
            if (toMat(inst[slot].modelMatrix) != w) r.worldOk = false; // numeric (-0 == +0)
        }
        return r;
    }
};

// ---------------------------------------------------------------------------
// Random churn over hundreds of frames.
// ---------------------------------------------------------------------------
struct ChurnResult {
    u32 bytesBad = 0, worldBad = 0, verifyBad = 0;
    u32 relocations = 0, deltaFrames = 0, fullFrames = 0, csrFrames = 0, dirtyFrames = 0, motionFrames = 0;
    std::string firstVerify;
};

struct Churn {
    World& w;
    u32 seed;
    std::vector<EntityID> alive;
    std::unordered_map<EntityID, u32> gdepth;

    Churn(World& world, u32 s) : w(world), seed(s) {}
    u32 rnd(u32 n) { seed = seed * 1664525u + 1013904223u; return (seed >> 8) % n; }
    float frnd() { return static_cast<float>(rnd(2001)) / 1000.0f - 1.0f; }

    TransformComponent randomXf() {
        glm::vec3 scale(1.0f);
        if (rnd(100) < 15) scale[rnd(3)] = -1.0f - 0.2f * static_cast<float>(rnd(3)); // mirrored
        return World::xform(glm::vec3(frnd() * 20, frnd() * 20, frnd() * 20), scale, frnd());
    }

    EntityID spawn() {
        const EntityID e = w.ecs.createEntity();
        w.all.push_back(e);
        alive.push_back(e);
        w.ecs.addComponent(e, randomXf());
        MeshInstanceComponent mi{};
        mi.meshHandle = rnd(100) < 5 ? 777u : w.meshes[rnd(static_cast<u32>(w.meshes.size()))];
        static const u32 mats[] = {0, 1, 5, ~0u};
        mi.materialIndex = mats[rnd(4)];
        mi.flags = rnd(8);
        mi.setVisible(rnd(100) < 90);
        w.ecs.addComponent(e, std::move(mi));
        if (rnd(100) < 25) w.ecs.addComponent(e, randomMaterial());
        gdepth[e] = 0;
        if (rnd(100) < 35 && alive.size() > 1) {
            const EntityID p = alive[rnd(static_cast<u32>(alive.size() - 1))];
            if (gdepth[p] < 4) {
                w.ecs.addComponent(e, HierarchyComponent{p});
                gdepth[e] = gdepth[p] + 1;
            }
        }
        if (gdepth[e] == 0 && rnd(100) < 8) w.ecs.addComponent(e, randomMotion());
        return e;
    }

    MaterialComponent randomMaterial() {
        MaterialComponent mc{};
        mc.doubleSided = rnd(100) < 30;
        mc.roughnessFactor = 0.01f * static_cast<float>(rnd(100));
        if (rnd(100) < 20) mc.emissiveFactor = glm::vec3(1.0f, 0.5f, 0.0f);
        return mc;
    }

    MotionComponent randomMotion() {
        MotionComponent m{};
        m.centre = glm::vec3(frnd(), frnd(), frnd());
        m.radius = 1.0f + frnd();
        m.phase = frnd() * 3.0f;
        m.height = frnd();
        m.speedClass = rnd(SCENE_MOTION_CLASSES);
        return m;
    }

    void destroy() {
        if (alive.empty()) return;
        const u32 i = rnd(static_cast<u32>(alive.size()));
        w.ecs.destroyEntity(alive[i]);
        alive[i] = alive.back();
        alive.pop_back();
    }

    void op(u32 growBias) {
        const u32 r = rnd(100);
        if (r < growBias) { spawn(); return; }
        if (alive.empty()) return;
        const EntityID e = alive[rnd(static_cast<u32>(alive.size()))];
        ECS& ecs = w.ecs;
        switch (rnd(12)) {
        case 0: destroy(); break;
        case 1: case 2: case 3: { // move
            auto& xf = ecs.getComponent<TransformComponent>(e);
            xf = randomXf();
            break;
        }
        case 4: ecs.getComponent<MeshInstanceComponent>(e).meshHandle = rnd(100) < 5 ? 555u : w.meshes[rnd(3)]; break;
        case 5: { auto& mi = ecs.getComponent<MeshInstanceComponent>(e); mi.setVisible(!mi.isVisible()); break; }
        case 6: ecs.getComponent<MeshInstanceComponent>(e).flags ^= 2u; break;
        case 7: // material component: add / remove / modify
            if (!ecs.hasComponent<MaterialComponent>(e)) ecs.addComponent(e, randomMaterial());
            else if (rnd(2)) ecs.getArray<MaterialComponent>().remove(e);
            else ecs.getComponent<MaterialComponent>(e) = randomMaterial();
            break;
        case 8: // hierarchy: reparent under a root / detach
            if (ecs.hasComponent<HierarchyComponent>(e)) {
                if (rnd(2)) { ecs.getArray<HierarchyComponent>().remove(e); gdepth[e] = 0; }
                else {
                    const EntityID p = alive[rnd(static_cast<u32>(alive.size()))];
                    if (gdepth[p] == 0 && p != e) { ecs.getComponent<HierarchyComponent>(e).parent = p; gdepth[e] = 1; }
                }
            }
            break;
        case 9: // motion on a root
            if (gdepth[e] == 0 && !ecs.hasComponent<HierarchyComponent>(e)) {
                if (!ecs.hasComponent<MotionComponent>(e)) ecs.addComponent(e, randomMotion());
                else if (rnd(2)) ecs.getArray<MotionComponent>().remove(e);
                else ecs.getComponent<MotionComponent>(e) = randomMotion();
            }
            break;
        case 10: // bulk writer: the whole transform array is "changed"
            if (rnd(40) == 0) {
                for (TransformComponent& t : ecs.getArray<TransformComponent>().data()) {
                    if (rnd(4) == 0) { t.position.x += 0.25f; t.updateMatrix(); }
                }
            }
            break;
        default: spawn(); break;
        }
    }
};

ChurnResult runChurn(u32 frames, u32 seed, bool dropRecord, bool skipDirty) {
    World w;
    Churn churn(w, seed);
    Sim sim;
    sim.dropRecord = dropRecord;
    sim.skipDirty = skipDirty;
    ChurnResult res;
    for (u32 i = 0; i < 300; ++i) churn.spawn();
    u64 lastVersion = 0;
    for (u32 f = 0; f < frames; ++f) {
        const u32 bias = f < 150 ? 55 : (f < 300 ? 25 : 8);
        const u32 n = 1 + churn.rnd(f % 100 == 99 ? 40 : 6);
        for (u32 k = 0; k < n; ++k) churn.op(bias);

        w.sync();
        const SceneSyncStats& st = w.store.stats();
        if (f > 0 && w.store.structureVersion() != lastVersion) ++res.relocations;
        lastVersion = w.store.structureVersion();
        if (st.fullInstances) ++res.fullFrames; else if (st.instanceRecords) ++res.deltaFrames;
        if (st.csrRebuilt) ++res.csrFrames;
        if (st.dirtyRoots) ++res.dirtyFrames;
        if (!w.store.motionSlots().empty()) ++res.motionFrames;
        const std::string v = w.verify();
        if (!v.empty()) {
            if (res.verifyBad++ == 0) res.firstVerify = "frame " + std::to_string(f) + ": " + v;
        }
        const Sim::Result r = sim.step(w.store, w.ecs, w.all);
        if (!r.bytesOk) ++res.bytesBad;
        if (!r.worldOk) ++res.worldBad;
        w.ecs.endFrame();
    }
    return res;
}

} // namespace

// ===========================================================================
// (a) random churn
// ===========================================================================
TEST_CASE("scene store: random churn, deltas applied to a simulated GPU equal the mirror byte for byte") {
    for (u32 seed : {1u, 7u}) {
        const ChurnResult r = runChurn(450, seed, false, false);
        INFO("seed " << seed << " " << r.firstVerify);
        CHECK(r.verifyBad == 0);
        CHECK(r.bytesBad == 0);
        CHECK(r.worldBad == 0);
        // The run must actually exercise the interesting paths.
        CHECK(r.relocations > 3);
        CHECK(r.deltaFrames > 100);
        CHECK(r.fullFrames > 2);
        CHECK(r.csrFrames > 5);
        CHECK(r.dirtyFrames > 50);
        CHECK(r.motionFrames > 100);
    }
}

TEST_CASE("scene store: negative control, a dropped delta record is detected by the byte comparison") {
    const ChurnResult r = runChurn(200, 1, /*dropRecord*/ true, false);
    CHECK(r.bytesBad > 10);
}

TEST_CASE("scene store: negative control, ignoring dirtyRoots() leaves stale child matrices") {
    const ChurnResult r = runChurn(200, 1, false, /*skipDirty*/ true);
    CHECK(r.worldBad > 10);
}

TEST_CASE("scene store: negative control, verifyAgainstEcs reports a change the ECS did not report") {
    World w;
    const EntityID a = w.add(0, glm::vec3(1, 0, 0));
    const EntityID b = w.add(1, glm::vec3(2, 0, 0), glm::vec3(1.0f), 0, true);
    w.addMotion(a);
    w.ecs.getArray<MotionComponent>().remove(a);
    w.frame();

    // Unmarked write through a const_cast: the store does not see it.
    auto& xf = const_cast<TransformComponent&>(std::as_const(w.ecs).getComponent<TransformComponent>(a));
    xf.position.x = 99.0f;
    xf.updateMatrix();
    w.sync();
    CHECK(w.store.stats().instanceRecords == 0);
    CHECK_FALSE(w.verify().empty());
    w.ecs.endFrame();
    w.ecs.markChanged<TransformComponent>(a); // marking it afterwards repairs the mirror
    w.sync();
    CHECK(w.verify().empty());
    w.ecs.endFrame();

    // Unmarked material and visibility changes are caught too.
    const_cast<MaterialComponent&>(std::as_const(w.ecs).getComponent<MaterialComponent>(b)).metallicFactor = 0.9f;
    w.sync();
    CHECK_FALSE(w.verify().empty());
    w.ecs.endFrame();
    w.ecs.markChanged<MaterialComponent>(b);
    w.sync();
    CHECK(w.verify().empty());
    w.ecs.endFrame();
    auto& mi = const_cast<MeshInstanceComponent&>(std::as_const(w.ecs).getComponent<MeshInstanceComponent>(a));
    mi.setVisible(false);
    w.sync();
    CHECK_FALSE(w.verify().empty());
}

// ===========================================================================
// (b) static scene
// ===========================================================================
TEST_CASE("scene store: a static scene produces no records and no bytes after the first sync") {
    World w;
    for (u32 i = 0; i < 150; ++i) w.add(i % 3, glm::vec3(static_cast<float>(i), 0, 0), glm::vec3(1.0f), i % 2, i % 5 == 0);
    const EntityID root = w.add(0, glm::vec3(0), glm::vec3(1.0f), 0, true);
    w.child(root, 1, glm::vec3(1, 1, 1));
    const EntityID mover = w.add(2);
    w.addMotion(mover);
    w.child(mover, 0, glm::vec3(0, 2, 0));

    w.sync();
    CHECK(w.store.stats().structure);
    CHECK(w.store.stats().fullInstances);
    CHECK(w.store.stats().uploadBytes > 0);
    CHECK(w.verify().empty());
    w.ecs.endFrame();

    const u64 version = w.store.structureVersion();
    for (int f = 0; f < 4; ++f) {
        w.sync();
        const SceneSyncStats& st = w.store.stats();
        CHECK(st.instanceRecords == 0);
        CHECK(st.materialRecords == 0);
        CHECK(st.nodeRecords == 0);
        CHECK(st.motionRecords == 0);
        CHECK_FALSE(st.fullInstances);
        CHECK_FALSE(st.fullMaterials);
        CHECK_FALSE(st.fullNodes);
        CHECK_FALSE(st.fullMotions);
        CHECK_FALSE(st.structure);
        CHECK(st.uploadBytes == 0);
        CHECK(w.store.structureVersion() == version);
        // The motion root with a child is recomputed every frame from the
        // persistent motion parent queue: nothing in queue 0, nothing re-sent.
        CHECK(w.store.dirtyRoots().empty());
        CHECK(w.store.motionParentSlots().size() == 1);
        CHECK_FALSE(st.motionParentsChanged);
        w.ecs.endFrame();
    }
}

TEST_CASE("scene store: steady state does not allocate (static and animated)") {
    World w;
    std::vector<EntityID> movers;
    for (u32 i = 0; i < 200; ++i) {
        const EntityID e = w.add(i % 3, glm::vec3(static_cast<float>(i), 0, 0), glm::vec3(1.0f), i % 2, i % 4 == 0);
        if (i % 20 == 0) movers.push_back(e);
    }
    const EntityID root = w.add(0);
    const EntityID kid = w.child(root, 1, glm::vec3(1, 0, 0));
    w.child(kid, 2, glm::vec3(0, 1, 0));
    const EntityID mover = w.add(2);
    w.addMotion(mover);
    w.child(mover, 0, glm::vec3(0, 2, 0));
    movers.push_back(root);
    movers.push_back(kid);

    auto frame = [&](u32 i) {
        for (EntityID e : movers) {
            auto& xf = w.ecs.getComponent<TransformComponent>(e);
            xf.position.y = (i & 1) ? 1.0f : 2.0f;
            xf.updateMatrix();
        }
        w.store.sync(w.ecs, w.scene);
        w.ecs.endFrame();
    };
    for (u32 i = 0; i < 8; ++i) frame(i); // warm-up: buffers reach their working size

    g_allocs.store(0);
    g_countAllocs.store(true);
    for (u32 i = 0; i < 30; ++i) frame(i);
    g_countAllocs.store(false);
    CHECK(g_allocs.load() == 0);
    CHECK(w.store.stats().instanceRecords > 0);
    CHECK(w.verify().empty());
}

TEST_CASE("scene store: a change costs records for the changed entities only") {
    World w;
    std::vector<EntityID> es;
    for (u32 i = 0; i < 2000; ++i) es.push_back(w.add(i % 3, glm::vec3(static_cast<float>(i), 0, 0)));
    w.frame();
    w.move(es[17], glm::vec3(0, 5, 0));
    w.move(es[1500], glm::vec3(0, 6, 0));
    w.sync();
    CHECK(w.store.stats().instanceRecords == 2);
    CHECK(w.store.stats().nodeRecords == 0);
    CHECK(w.store.stats().uploadBytes == 2 * sizeof(GPUDeltaRecord));
    CHECK_FALSE(w.store.stats().structure);
    CHECK(w.verify().empty());
    w.ecs.endFrame();

    // Writing identical values marks the entity but produces no record.
    w.move(es[17], glm::vec3(0, 5, 0));
    w.sync();
    CHECK(w.store.stats().instanceRecords == 0);
    w.ecs.endFrame();

    // More than 1/8 of the buffer: a full copy instead of records.
    for (u32 i = 0; i < 1000; ++i) w.move(es[i], glm::vec3(0, 7, static_cast<float>(i)));
    w.sync();
    CHECK(w.store.stats().fullInstances);
    CHECK(w.store.stats().instanceRecords == 0);
    CHECK(w.store.instanceDeltas().empty());
    CHECK(w.verify().empty());
}

TEST_CASE("scene store: a bulk non-const data() write is treated as all changed") {
    World w;
    for (u32 i = 0; i < 40; ++i) w.add(i % 3, glm::vec3(static_cast<float>(i), 0, 0));
    w.frame();
    // Takes the mutable span but changes only entity-0's position: still correct.
    auto xs = w.ecs.getArray<TransformComponent>().data();
    xs[0].position.y = 3.0f;
    xs[0].updateMatrix();
    w.sync();
    CHECK(w.verify().empty());
    CHECK(w.store.stats().instanceRecords == 1);
    w.ecs.endFrame();
}

// ===========================================================================
// (c) order, buckets, command layout
// ===========================================================================
TEST_CASE("scene store: bucket order, in-bucket order and ICB layout after a full build") {
    World w;
    // Interleaved meshes, classes (mirrored, double-sided) and an invisible entity.
    for (u32 i = 0; i < 40; ++i) {
        glm::vec3 scale(1.0f);
        if (i % 7 == 3) scale.x = -1.0f;
        const EntityID e = w.add((i * 5) % 3, glm::vec3(static_cast<float>(i), 1, 2), scale, i % 4 == 0 ? 1 : 0);
        if (i == 11) w.ecs.getComponent<MeshInstanceComponent>(e).setVisible(false);
    }
    w.frame();
    const SceneStore& s = w.store;

    // Buckets sorted by (class, mesh).
    auto buckets = s.buckets();
    REQUIRE(buckets.size() >= 3);
    for (size_t i = 1; i < buckets.size(); ++i) {
        const auto a = std::pair(static_cast<u32>(buckets[i - 1].cull), buckets[i - 1].mesh);
        const auto b = std::pair(static_cast<u32>(buckets[i].cull), buckets[i].mesh);
        CHECK(a < b);
    }

    // In-bucket order = dense MeshInstanceComponent order.
    const auto& miArr = std::as_const(w.ecs).getArray<MeshInstanceComponent>();
    std::unordered_map<EntityID, u32> dense;
    for (u32 i = 0; i < miArr.size(); ++i) dense[miArr.entities()[i]] = i;
    u32 live = 0;
    for (const SceneBucket& b : buckets) {
        std::vector<std::pair<u32, EntityID>> inBucket;
        for (EntityID e : w.all) {
            const u32 slot = s.slotOf(e);
            if (slot != NONE && slot >= b.firstSlot && slot < b.firstSlot + b.count) inBucket.push_back({slot, e});
        }
        CHECK(inBucket.size() == b.count);
        std::sort(inBucket.begin(), inBucket.end());
        for (size_t i = 1; i < inBucket.size(); ++i) CHECK(dense[inBucket[i - 1].second] < dense[inBucket[i].second]);
        CHECK(b.capacity >= std::max<u32>(64, b.count));
        live += b.count;
    }
    CHECK(live == 39);
    CHECK(s.instanceCount() == 39);
    CHECK(s.slotCapacity() >= 39);

    // Command layout: per class its buckets then one sentinel.
    const auto cmds = s.commandBuckets();
    const auto ranges = s.classRanges();
    CHECK(s.commandCount() == buckets.size() + SCENE_CULL_CLASSES);
    CHECK(cmds.size() == s.commandCount());
    u32 at = 0;
    for (u32 c = 0; c < SCENE_CULL_CLASSES; ++c) {
        CHECK(ranges[c].firstCommand == at);
        u32 n = 0;
        for (u32 i = 0; i < buckets.size(); ++i) {
            if (static_cast<u32>(buckets[i].cull) != c) continue;
            CHECK(cmds[at + n] == i);
            CHECK(buckets[i].command == at + n);
            ++n;
        }
        CHECK(cmds[at + n] == NONE);
        CHECK(ranges[c].commandCount == n + 1);
        at += n + 1;
    }
    CHECK(at == cmds.size());
    // GPU bucket table mirrors the mesh infos.
    const auto& infos = w.scene.meshInfos();
    for (u32 i = 0; i < buckets.size(); ++i) {
        const GPUDrawBucket& g = s.gpuBuckets()[i];
        CHECK(g.firstSlot == buckets[i].firstSlot);
        CHECK(g.capacity == buckets[i].capacity);
        CHECK(g.meshIndex == buckets[i].mesh);
        CHECK(g.indexCount == infos[buckets[i].mesh].indexCount);
        CHECK(g.indexOffset == infos[buckets[i].mesh].indexOffset);
        CHECK(g.vertexOffset == infos[buckets[i].mesh].vertexOffset);
        CHECK(g.command == buckets[i].command);
    }
    // Slack slots are not valid, live ones are.
    u32 valid = 0;
    for (const GPUInstance& gi : s.instances()) valid += (gi.flags & INSTANCE_FLAG_VALID) ? 1u : 0u;
    CHECK(valid == 39);

    // Same draw sequence per (class, mesh) as the classic extraction.
    FrameScene fs;
    extractFrameScene(w.ecs, w.scene, fs);
    REQUIRE(fs.instances.size() == 39);
    for (const DrawBatch& batch : fs.batches) {
        const SceneBucket* found = nullptr;
        for (const SceneBucket& b : buckets) {
            if (b.mesh == batch.meshIndex && b.cull == batch.cull) found = &b;
        }
        REQUIRE(found != nullptr);
        REQUIRE(found->count == batch.instanceCount);
        for (u32 i = 0; i < batch.instanceCount; ++i) {
            const GPUInstance& a = fs.instances[batch.firstInstance + i];
            const GPUInstance& b = s.instances()[found->firstSlot + i];
            CHECK(std::memcmp(a.modelMatrix, b.modelMatrix, sizeof(a.modelMatrix)) == 0);
            CHECK(a.materialIndex == b.materialIndex);
            CHECK((b.flags & ~INSTANCE_FLAG_VALID) == a.flags);
        }
    }
}

TEST_CASE("scene store: material resolution matches extractFrameScene, default material without a library") {
    World w(0);
    const EntityID a = w.add(0, glm::vec3(0), glm::vec3(1.0f), 5);       // invalid index
    const EntityID b = w.add(0, glm::vec3(0), glm::vec3(1.0f), ~0u);     // unset
    const EntityID c = w.add(1, glm::vec3(0), glm::vec3(1.0f), 0, true); // own material
    w.frame();
    CHECK(w.store.materialIndexOf(a) == 0);
    CHECK(w.store.materialIndexOf(b) == 0);
    CHECK(w.store.materialIndexOf(c) == 1);
    const GPUMaterial def = toGPUMaterial(MaterialComponent{});
    CHECK(std::memcmp(&w.store.materials()[0], &def, sizeof(def)) == 0);
    CHECK_FALSE(w.store.hasEmissive());
    w.ecs.getComponent<MaterialComponent>(c).emissiveFactor = glm::vec3(0.0f, 1.0f, 0.0f);
    w.frame();
    CHECK(w.store.hasEmissive());
}

// ===========================================================================
// (d) relocation, (e) slot stability
// ===========================================================================
TEST_CASE("scene store: a full bucket relocates at the end of the slot space, keeping every instance") {
    World w;
    std::vector<EntityID> es;
    for (u32 i = 0; i < 70; ++i) es.push_back(w.add(0, glm::vec3(static_cast<float>(i), 0, 0)));
    es.push_back(w.add(1, glm::vec3(0, 9, 0)));
    w.frame();
    const u64 version = w.store.structureVersion();
    const u32 capBefore = w.store.slotCapacity();
    u32 oldBucketCap = 0;
    for (const SceneBucket& b : w.store.buckets()) if (b.mesh == w.meshes[0]) oldBucketCap = b.capacity;

    for (u32 i = 0; i < 60; ++i) es.push_back(w.add(0, glm::vec3(static_cast<float>(100 + i), 0, 0)));
    w.sync();
    CHECK(w.store.structureVersion() > version);
    CHECK(w.store.stats().structure);
    CHECK(w.store.stats().fullInstances);
    CHECK(w.store.slotCapacity() > capBefore);
    u32 newBucketCap = 0, newFirst = 0;
    for (const SceneBucket& b : w.store.buckets()) if (b.mesh == w.meshes[0]) { newBucketCap = b.capacity; newFirst = b.firstSlot; }
    CHECK(newBucketCap >= 2 * oldBucketCap);
    CHECK(newFirst >= capBefore); // moved to the end of the old slot space
    CHECK(w.verify().empty());
    for (EntityID e : es) {
        const u32 s = w.store.slotOf(e);
        REQUIRE(s != NONE);
        const glm::mat4 want = std::as_const(w.ecs).getComponent<TransformComponent>(e).worldMatrix;
        CHECK(std::memcmp(w.store.instances()[s].modelMatrix, &want[0][0], 64) == 0);
    }
    w.ecs.endFrame();
    // The next frame is quiet again.
    w.sync();
    CHECK_FALSE(w.store.stats().structure);
    CHECK(w.store.stats().uploadBytes == 0);
}

TEST_CASE("scene store: slots of unchanged entities are stable (swap-and-pop moves one, adds append)") {
    World w;
    std::vector<EntityID> es;
    for (u32 i = 0; i < 100; ++i) es.push_back(w.add(0, glm::vec3(static_cast<float>(i), 0, 0)));
    w.frame();
    std::unordered_map<EntityID, u32> slot;
    for (EntityID e : es) slot[e] = w.store.slotOf(e);

    // Changes keep slots.
    w.move(es[3], glm::vec3(0, 3, 0));
    w.ecs.getComponent<MeshInstanceComponent>(es[4]).flags |= 2u;
    w.frame();
    for (EntityID e : es) CHECK(w.store.slotOf(e) == slot[e]);

    // Remove one in the middle: it leaves a hole, nobody moves.
    w.ecs.destroyEntity(es[10]);
    w.frame();
    for (EntityID e : es) {
        if (e == es[10]) { CHECK(w.store.slotOf(e) == NONE); continue; }
        CHECK(w.store.slotOf(e) == slot[e]);
    }
    CHECK(w.store.stats().instanceRecords == 1); // the zeroed hole
    CHECK_FALSE(w.store.stats().structure);

    // Add: the hole is reused.
    const EntityID n = w.add(0, glm::vec3(5, 5, 5));
    w.frame();
    CHECK(w.store.slotOf(n) == slot[es[10]]);
    for (EntityID e : es) {
        if (e != es[10]) CHECK(w.store.slotOf(e) == slot[e]);
    }
    CHECK_FALSE(w.store.stats().structure);
    CHECK(w.store.instanceCount() == 100);
}

// ===========================================================================
// (f) materials
// ===========================================================================
TEST_CASE("scene store: per-entity materials are persistent and re-sent only on change") {
    World w;
    w.padMaterials(80);
    const EntityID a = w.add(0, glm::vec3(0), glm::vec3(1.0f), 0, true);
    const EntityID b = w.add(0, glm::vec3(1, 0, 0), glm::vec3(1.0f), 0, true);
    const EntityID c = w.add(0, glm::vec3(2, 0, 0), glm::vec3(1.0f), 1);
    w.frame();
    const u32 ia = w.store.materialIndexOf(a), ib = w.store.materialIndexOf(b);
    CHECK(ia >= 2);
    CHECK(ib >= 2);
    CHECK(ia != ib);
    CHECK(w.store.materialIndexOf(c) == 1);

    // Unrelated changes keep indices and send no materials.
    w.move(c, glm::vec3(0, 4, 0));
    w.frame();
    CHECK(w.store.stats().materialRecords == 0);
    CHECK(w.store.materialIndexOf(a) == ia);
    CHECK(w.store.materialIndexOf(b) == ib);

    // A material change sends exactly that material, not the instance.
    w.ecs.getComponent<MaterialComponent>(a).roughnessFactor = 0.77f;
    w.sync();
    CHECK(w.store.stats().materialRecords == 1);
    CHECK(w.store.stats().instanceRecords == 0);
    REQUIRE(w.store.materialDeltas().size() == 1);
    CHECK(w.store.materialDeltas()[0].slot == ia);
    CHECK(w.store.materials()[ia].roughness == doctest::Approx(0.77f));
    CHECK(w.verify().empty());
    w.ecs.endFrame();

    // Removing the component frees the slot (zeroed) and falls back to the library index.
    w.ecs.getArray<MaterialComponent>().remove(a);
    w.frame();
    CHECK(w.store.materialIndexOf(a) == 0);
    const GPUMaterial zero{};
    CHECK(std::memcmp(&w.store.materials()[ia], &zero, sizeof(zero)) == 0);
    // A new per-entity material reuses the freed slot.
    const EntityID d = w.add(0, glm::vec3(7, 0, 0), glm::vec3(1.0f), 0, true);
    w.frame();
    CHECK(w.store.materialIndexOf(d) == ia);
    CHECK(w.store.materialIndexOf(b) == ib);

    // A double-sided own material moves the instance to the None class.
    w.ecs.getComponent<MaterialComponent>(b).doubleSided = true;
    w.frame();
    const u32 sb = w.store.slotOf(b);
    bool inNone = false;
    for (const SceneBucket& bk : w.store.buckets()) {
        if (sb >= bk.firstSlot && sb < bk.firstSlot + bk.count) inNone = bk.cull == CullClass::None;
    }
    CHECK(inNone);
}

TEST_CASE("scene store: a library or geometry change rebuilds the store; clear() rebuilds identically") {
    World w;
    for (u32 i = 0; i < 30; ++i) w.add(i % 3, glm::vec3(static_cast<float>(i), 0, 0), glm::vec3(1.0f), i % 2, i % 3 == 0);
    w.frame();
    const u64 v = w.store.structureVersion();
    GPUMaterial extra{};
    extra.emissive[1] = 1.0f;
    w.scene.addMaterial(extra);
    w.sync();
    CHECK(w.store.structureVersion() > v);
    CHECK(w.store.stats().fullMaterials);
    CHECK(w.store.hasEmissive());
    CHECK(w.verify().empty());
    w.ecs.endFrame();

    const std::vector<GPUInstance> before(w.store.instances().begin(), w.store.instances().end());
    w.store.clear();
    CHECK(w.store.instanceCount() == 0);
    w.sync();
    CHECK(w.verify().empty());
    REQUIRE(before.size() == w.store.instances().size());
    CHECK(std::memcmp(before.data(), w.store.instances().data(), before.size() * sizeof(GPUInstance)) == 0);
}

// ===========================================================================
// (g) hierarchy
// ===========================================================================
TEST_CASE("scene store: hierarchy CSR, depth, mirrored propagation") {
    World w;
    const EntityID R = w.add(0, glm::vec3(0), glm::vec3(-1.0f, 1.0f, 1.0f)); // mirrored root
    const EntityID C1 = w.child(R, 0, glm::vec3(1, 0, 0));
    const EntityID G = w.child(C1, 0, glm::vec3(0, 1, 0), glm::vec3(-1.0f, 1.0f, 1.0f)); // mirrors back
    const EntityID C2 = w.child(R, 1, glm::vec3(0, 0, 1));
    w.frame();
    const SceneStore& s = w.store;
    const u32 sR = s.slotOf(R), sC1 = s.slotOf(C1), sG = s.slotOf(G), sC2 = s.slotOf(C2);
    REQUIRE(sR != NONE);
    REQUIRE(sG != NONE);

    CHECK(s.maxDepth() == 2);
    CHECK(s.nodes()[sC1].depth == 1);
    CHECK(s.nodes()[sG].depth == 2);
    CHECK(s.nodes()[sC1].parentSlot == sR);
    CHECK(s.nodes()[sG].parentSlot == sC1);
    CHECK(s.nodes()[sR].depth == 0);
    const glm::mat4 local = w.ecs.getComponent<TransformComponent>(C1).worldMatrix;
    CHECK(std::memcmp(s.nodes()[sC1].local, &local[0][0], 64) == 0);

    auto children = [&](u32 slot) {
        std::vector<u32> v(s.childSlots().begin() + s.childOffsets()[slot], s.childSlots().begin() + s.childOffsets()[slot + 1]);
        std::sort(v.begin(), v.end());
        return v;
    };
    std::vector<u32> wantR{sC1, sC2};
    std::sort(wantR.begin(), wantR.end());
    CHECK(children(sR) == wantR);
    CHECK(children(sC1) == std::vector<u32>{sG});
    CHECK(children(sG).empty());
    CHECK(s.childOffsets().size() == s.slotCapacity() + 1);

    // Mirror flags / classes: R mirrored, C1 inherits, G (local mirrored) is back to normal.
    CHECK((s.instances()[sR].flags & INSTANCE_FLAG_MIRRORED) != 0);
    CHECK((s.instances()[sC1].flags & INSTANCE_FLAG_MIRRORED) != 0);
    CHECK((s.instances()[sG].flags & INSTANCE_FLAG_MIRRORED) == 0);
    auto classOf = [&](u32 slot) {
        for (const SceneBucket& b : s.buckets()) {
            if (slot >= b.firstSlot && slot < b.firstSlot + b.count) return b.cull;
        }
        return CullClass::None;
    };
    CHECK(classOf(sR) == CullClass::BackMirrored);
    CHECK(classOf(sC1) == CullClass::BackMirrored);
    CHECK(classOf(sG) == CullClass::Back);
    // Placeholder: children carry an identity matrix.
    const glm::mat4 id(1.0f);
    CHECK(std::memcmp(s.instances()[sC1].modelMatrix, &id[0][0], 64) == 0);

    // Un-mirroring the root flips the whole subtree's classes (and moves buckets).
    w.ecs.endFrame();
    w.move(R, glm::vec3(0));
    {
        auto& xf = w.ecs.getComponent<TransformComponent>(R);
        xf.scale = glm::vec3(1.0f);
        xf.updateMatrix();
    }
    w.frame();
    CHECK((s.instances()[s.slotOf(C1)].flags & INSTANCE_FLAG_MIRRORED) == 0);
    CHECK((s.instances()[s.slotOf(G)].flags & INSTANCE_FLAG_MIRRORED) != 0);
    CHECK(classOf(s.slotOf(G)) == CullClass::BackMirrored);
    CHECK(s.csrRebuildCount() >= 2);
}

TEST_CASE("scene store: dirtyRoots rules") {
    World w;
    w.padMaterials(80);
    const EntityID P = w.add(0, glm::vec3(0));
    const EntityID K = w.child(P, 0, glm::vec3(1, 0, 0));
    const EntityID G = w.child(K, 0, glm::vec3(0, 1, 0), glm::vec3(1.0f), 0, true);
    const EntityID lone = w.add(1, glm::vec3(5, 5, 5));
    w.frame();
    const SceneStore& s = w.store;
    auto dirty = [&] { return std::vector<u32>(s.dirtyRoots().begin(), s.dirtyRoots().end()); };
    const u32 sP = s.slotOf(P), sK = s.slotOf(K);

    w.sync();
    CHECK(dirty().empty()); // static
    w.ecs.endFrame();

    w.move(lone, glm::vec3(1, 2, 3)); // root without children
    w.sync();
    CHECK(dirty().empty());
    w.ecs.endFrame();

    w.move(P, glm::vec3(0, 9, 0)); // root with children, world changed
    w.sync();
    CHECK(dirty() == std::vector<u32>{sP});
    CHECK(s.stats().dirtyRoots == 1);
    w.ecs.endFrame();

    w.move(K, glm::vec3(2, 0, 0)); // child local changed: its parent is dirty
    w.sync();
    CHECK(dirty() == std::vector<u32>{sP});
    w.ecs.endFrame();

    w.move(G, glm::vec3(0, 2, 0)); // grandchild local changed
    w.sync();
    CHECK(dirty() == std::vector<u32>{sK});
    w.ecs.endFrame();

    w.ecs.getComponent<MeshInstanceComponent>(G).meshHandle = w.meshes[2]; // instance record changed
    w.sync();
    CHECK(dirty() == std::vector<u32>{sK});
    w.ecs.endFrame();

    w.ecs.getComponent<MaterialComponent>(G).roughnessFactor = 0.3f; // material only: no matrix involved
    w.sync();
    CHECK(dirty().empty());
    CHECK(s.stats().materialRecords == 1);
    w.ecs.endFrame();

    // Two changes with the same parent are deduplicated.
    w.move(K, glm::vec3(3, 0, 0));
    w.ecs.getComponent<MeshInstanceComponent>(K).flags ^= 2u;
    w.sync();
    CHECK(dirty() == std::vector<u32>{sP});
    CHECK(w.verify().empty());
    w.ecs.endFrame();

    // Hiding K drops K and (parent not in the store) G; no crash, consistent.
    w.ecs.getComponent<MeshInstanceComponent>(K).setVisible(false);
    w.frame();
    CHECK(s.slotOf(K) == NONE);
    CHECK(s.slotOf(G) == NONE);
    CHECK(s.maxDepth() == 0);
    // Showing it again brings both back and recomputes under P.
    w.ecs.getComponent<MeshInstanceComponent>(K).setVisible(true);
    w.sync();
    CHECK(s.slotOf(G) != NONE);
    {
        std::vector<u32> want{s.slotOf(P), s.slotOf(K)}; // K's record is new (parent P), G's is new (parent K)
        std::vector<u32> got = dirty();
        std::sort(want.begin(), want.end());
        std::sort(got.begin(), got.end());
        CHECK(got == want);
    }
    CHECK(w.verify().empty());
}

TEST_CASE("scene store: a removal moves no child or parent (holes): no CSR rebuild, no dirty root") {
    SUBCASE("removal next to a parent and its child") {
        World w;
        const EntityID X = w.add(0);
        const EntityID P = w.add(0, glm::vec3(1, 0, 0));
        const EntityID Y = w.add(0, glm::vec3(2, 0, 0));
        const EntityID K = w.child(P, 0, glm::vec3(0, 1, 0));
        w.frame();
        const u64 rebuilds = w.store.csrRebuildCount();
        const u32 sP = w.store.slotOf(P), sK = w.store.slotOf(K), sX = w.store.slotOf(X);
        w.ecs.destroyEntity(X);
        w.sync();
        CHECK(w.store.slotOf(P) == sP);
        CHECK(w.store.slotOf(K) == sK);
        CHECK(w.store.nodes()[sK].parentSlot == sP);
        CHECK(w.store.dirtyRoots().empty());
        CHECK_FALSE(w.store.stats().csrRebuilt);
        CHECK(w.store.csrRebuildCount() == rebuilds);
        CHECK_FALSE(w.store.stats().structure);
        CHECK(w.verify().empty());
        w.ecs.endFrame();
        // The next add in the bucket takes X's hole.
        const EntityID Z = w.add(0, glm::vec3(3, 0, 0));
        w.sync();
        CHECK(w.store.slotOf(Z) == sX);
        (void)Y;
        CHECK(w.verify().empty());
    }
    SUBCASE("removal of a child updates its parent's CSR") {
        World w2;
        const EntityID P2 = w2.add(0, glm::vec3(1, 0, 0));
        const EntityID K2 = w2.child(P2, 0, glm::vec3(0, 1, 0));
        w2.frame();
        w2.ecs.destroyEntity(K2);
        w2.sync();
        const u32 sP = w2.store.slotOf(P2);
        CHECK(w2.store.childOffsets()[sP + 1] == w2.store.childOffsets()[sP]); // no child left
        CHECK(w2.verify().empty());
    }
}

TEST_CASE("scene store: hierarchy deeper than the GPU levels is an error and the entity is skipped") {
    World w;
    EntityID prev = w.add(0);
    std::vector<EntityID> chain{prev};
    for (u32 i = 1; i < SCENE_MAX_LEVELS + 2; ++i) {
        prev = w.child(prev, 0, glm::vec3(0, 1, 0));
        chain.push_back(prev);
    }
    w.frame(); // verify() agrees with the skip rule
    CHECK(w.store.maxDepth() == SCENE_MAX_LEVELS - 1);
    CHECK(w.store.slotOf(chain[SCENE_MAX_LEVELS - 1]) != NONE);
    CHECK(w.store.slotOf(chain[SCENE_MAX_LEVELS]) == NONE);
    CHECK(w.store.slotOf(chain[SCENE_MAX_LEVELS + 1]) == NONE);
    CHECK(w.store.instanceCount() == SCENE_MAX_LEVELS);
}

TEST_CASE("scene store: simulated GPU world matrices follow the hierarchy through changes") {
    World w;
    Sim sim;
    const EntityID P = w.add(0, glm::vec3(0));
    const EntityID K = w.child(P, 0, glm::vec3(1, 0, 0));
    const EntityID G = w.child(K, 0, glm::vec3(0, 1, 0));
    const EntityID M = w.add(1);
    w.addMotion(M);
    const EntityID MK = w.child(M, 2, glm::vec3(0, 0, 1));
    (void)MK;
    for (u32 f = 0; f < 6; ++f) {
        if (f == 1) w.move(P, glm::vec3(3, 3, 3));
        if (f == 2) w.move(K, glm::vec3(0, 0, 5));
        if (f == 3) w.move(G, glm::vec3(7, 7, 7));
        if (f == 4) w.ecs.getComponent<MotionComponent>(M).phase = 2.0f;
        if (f == 5) w.ecs.destroyEntity(P);
        w.sync();
        const Sim::Result r = sim.step(w.store, w.ecs, w.all);
        CHECK(r.bytesOk);
        CHECK(r.worldOk);
        CHECK(w.verify().empty());
        w.ecs.endFrame();
    }
}

// ===========================================================================
// (h) motion
// ===========================================================================
TEST_CASE("scene store: motion records and slot list") {
    World w;
    const EntityID a = w.add(0, glm::vec3(1, 2, 3), glm::vec3(2.0f));
    const EntityID b = w.add(0, glm::vec3(4, 5, 6));
    w.add(1);
    w.addMotion(a, 0.5f, 7);
    w.frame();
    const SceneStore& s = w.store;
    const u32 sa = s.slotOf(a);
    REQUIRE(s.motionSlots().size() == 1);
    CHECK(s.motionSlots()[0] == sa);

    const GPUMotion& m = s.motions()[sa];
    const MotionComponent& mc = std::as_const(w.ecs).getComponent<MotionComponent>(a);
    CHECK(m.centre[0] == mc.centre.x);
    CHECK(m.centre[2] == mc.centre.z);
    CHECK(m.radius == mc.radius);
    CHECK(m.cosPhase == std::cos(0.5f));
    CHECK(m.sinPhase == std::sin(0.5f));
    CHECK(m.height == mc.height);
    CHECK(m.speedClass == 7u);
    const glm::mat4 base = std::as_const(w.ecs).getComponent<TransformComponent>(a).worldMatrix;
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 3; ++r) CHECK(m.base[c * 3 + r] == base[c][r]);
    }
    // Non-motion slots have an all-zero motion record.
    const GPUMotion zero{};
    CHECK(std::memcmp(&s.motions()[s.slotOf(b)], &zero, sizeof(zero)) == 0);
    // The instance matrix is a placeholder the GPU overwrites.
    const glm::mat4 id(1.0f);
    CHECK(std::memcmp(s.instances()[sa].modelMatrix, &id[0][0], 64) == 0);
    w.ecs.endFrame();

    // Motion change: one motion record, no instance record, list unchanged.
    w.ecs.getComponent<MotionComponent>(a).radius = 9.0f;
    w.sync();
    CHECK(s.stats().motionRecords == 1);
    CHECK(s.stats().instanceRecords == 0);
    CHECK_FALSE(s.stats().structure);
    REQUIRE(s.motionDeltas().size() == 1);
    CHECK(s.motionDeltas()[0].slot == sa);
    w.ecs.endFrame();

    // Base transform change re-sends the motion record.
    w.move(a, glm::vec3(8, 8, 8));
    w.sync();
    CHECK(s.stats().motionRecords == 1);
    CHECK(s.stats().instanceRecords == 0);
    CHECK(w.verify().empty());
    w.ecs.endFrame();

    // Adding and removing motion changes the list (structure) and zeroes the record.
    w.addMotion(b);
    w.frame();
    CHECK(s.motionSlots().size() == 2);
    CHECK(s.stats().structure);
    w.ecs.getArray<MotionComponent>().remove(a);
    w.sync();
    CHECK(s.motionSlots().size() == 1);
    CHECK(s.motionSlots()[0] == s.slotOf(b));
    CHECK(std::memcmp(&s.motions()[sa], &zero, sizeof(zero)) == 0);
    CHECK(w.verify().empty());
    w.ecs.endFrame();

    // A motion root that moves slots keeps its list entry in sync.
    const EntityID c = w.add(0, glm::vec3(0, 0, 9));
    w.frame();
    w.ecs.destroyEntity(a);
    w.frame();
    CHECK(s.motionSlots().size() == 1);
    CHECK(s.motionSlots()[0] == s.slotOf(b));
    (void)c;
}

TEST_CASE("scene store: only roots move procedurally (motion on a child is ignored)") {
    World w;
    const EntityID P = w.add(0);
    const EntityID K = w.child(P, 0, glm::vec3(1, 0, 0));
    w.addMotion(K);
    w.frame();
    CHECK(w.store.motionSlots().empty());
    CHECK(w.store.slotOf(K) != NONE);
}

// ===========================================================================
// Benches through the store: only what a bench moves is marked changed
// ===========================================================================
TEST_CASE("scene store: every bench syncs consistently and marks only what it moves") {
    for (int i = 0; i < testBenchCount(); ++i) {
        const auto type = static_cast<TestBenchType>(i);
        CAPTURE(testBenchName(type));
        ECS ecs;
        GpuScene scene;
        phosphor::test::NullTextureManager textures;
        SceneStore store;
        TestBenchParams params;
        params.instances = 4000;
        auto bench = createTestBench(type, params);
        REQUIRE(bench);
        bench->setup(ecs, scene, textures);
        store.sync(ecs, scene);
        ecs.endFrame();
        CHECK(store.verifyAgainstEcs(ecs, scene).empty());

        u32 maxChanged = 0;
        for (int f = 0; f < 6; ++f) {
            bench->update(1.0f / 60.0f, ecs);
            maxChanged = std::max(maxChanged, static_cast<u32>(std::as_const(ecs).getArray<TransformComponent>().changes().changed.size()));
            CHECK_FALSE(std::as_const(ecs).getArray<TransformComponent>().changes().all);
            store.sync(ecs, scene);
            CHECK(store.verifyAgainstEcs(ecs, scene).empty());
            ecs.endFrame();
        }
        // No bench rewrites every transform each frame (bench 8: ~1 % plus churn).
        if (type == TestBenchType::MillionInstances) CHECK(maxChanged < 4000 / 20);
        bench->teardown(ecs, scene);
    }
}
