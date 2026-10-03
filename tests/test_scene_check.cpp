#include <doctest/doctest.h>

#include "renderer/cull_reference.h"
#include "renderer/gpu_scene.h"
#include "renderer/scene_check.h"
#include "renderer/scene_store.h"
#include "renderer/transform_math.h"
#include "renderer/transform_reference.h"
#include "scene/components.h"
#include "scene/ecs.h"
#include "scene/procedural.h"

#include <glm/gtc/matrix_transform.hpp>

#include <cstring>
#include <utility>
#include <vector>

using namespace phosphor;

// The --debug-gpu-scene comparison (renderer/scene_check.h) on a small scene
// whose "GPU read-back" is built from the CPU references: it must PASS, and
// every kind of corruption the self-check promises to catch must FAIL.

namespace {

struct Fixture {
    ECS ecs;
    GpuScene scene;
    SceneStore store;
    GPUCullParams cull{};
    std::vector<float> sinCos = std::vector<float>(SCENE_MOTION_CLASSES * 2);
    // Simulated GPU state.
    std::vector<GPUInstance> instances;
    std::vector<GPUMaterial> materials;
    std::vector<u32> prefix, visible, drawArgs;
    GPUSceneCounters counters{};

    Fixture() {
        const MeshData cube = ProceduralMeshes::generateCube(0.5f);
        const MeshHandle mesh = scene.uploadMesh(cube.positions, cube.normals, cube.tangents, cube.uvs, cube.indices);
        scene.addMaterial(GPUMaterial{});
        EntityID parent = INVALID_ENTITY;
        for (u32 i = 0; i < 40; ++i) {
            const EntityID e = ecs.createEntity();
            TransformComponent t;
            t.position = glm::vec3(float(i % 8) * 3.0f - 12.0f, 0.0f, -10.0f - float(i / 8) * 4.0f);
            if (i == 7) t.scale = glm::vec3(-1.0f, 1.0f, 1.0f); // mirrored
            t.updateMatrix();
            ecs.addComponent(e, std::move(t));
            MeshInstanceComponent m;
            m.meshHandle    = mesh;
            m.materialIndex = 0;
            m.setVisible(true);
            ecs.addComponent(e, std::move(m));
            if (i == 3) {
                MotionComponent mo;
                mo.radius = 2.0f;
                mo.speedClass = 5;
                ecs.addComponent(e, std::move(mo));
                parent = e;
            }
            if (i == 4) ecs.addComponent(e, HierarchyComponent{parent}); // child of the moving root
            if (i == 9) ecs.addComponent(e, MaterialComponent{});
        }
        store.sync(ecs, scene);
        ecs.endFrame();

        motionSinCosTable(1.25, sinCos.data());
        const glm::mat4 proj = glm::perspective(glm::radians(60.0f), 16.0f / 9.0f, 0.05f, 1000.0f);
        glm::mat4 rz = proj; // reverse-Z infinite like the engine's camera
        rz[2][2] = 0.0f;
        rz[3][2] = 0.05f;
        cull = makeCullParams(rz, rz[1][1], 1800, glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, -1.0f), 0.05f,
                              CULL_FLAG_FRUSTUM, 0.0f, 0.0f, store.slotCapacity());

        // What the GPU must hold: the mirror with the GPU-computed matrices.
        instances.assign(store.instances().begin(), store.instances().end());
        std::vector<float> worlds;
        referenceWorlds(store, sinCos.data(), worlds);
        for (size_t s = 0; s < instances.size(); ++s) std::memcpy(instances[s].modelMatrix, &worlds[s * 16], 64);
        materials.assign(store.materials().begin(), store.materials().end());
        std::vector<u8> result;
        cullReference(instances, scene.meshInfos(), cull, result);
        compactReference(result, visible, prefix);
        visible.resize(store.slotCapacity(), 0); // the GPU list has room for every slot
        drawArgsReference(store.gpuBuckets(), store.commandBuckets(), prefix, drawArgs);
        counters.tested  = store.instanceCount();
        counters.visible = prefix.back();
        for (u8 r : result) counters.culledFrustum += r == 1 ? 1u : 0u;
        for (size_t i = 0; i < drawArgs.size(); i += 2) counters.drawCommands += drawArgs[i] > 0 ? 1u : 0u;
    }

    SceneCheckResult check(bool gpuDriven = true) const {
        SceneReadback r;
        r.instances = instances;
        r.materials = materials;
        r.prefix    = prefix;
        r.visible   = visible;
        r.drawArgs  = drawArgs;
        r.counters  = counters;
        return compareScene(store, scene, ecs, cull, sinCos.data(), r, gpuDriven);
    }
};

} // namespace

TEST_CASE("scene check: the CPU references pass, on and off") {
    Fixture f;
    REQUIRE(f.counters.visible > 0);
    REQUIRE(f.counters.culledFrustum > 0); // some instances are behind the camera / outside
    const SceneCheckResult on = f.check(true);
    CHECK_MESSAGE(on.pass, formatSceneCheck(on));
    CHECK(f.check(false).pass);
}

TEST_CASE("scene check: negative controls, every corruption is reported") {
    {
        Fixture f; // a stale non-matrix field (a delta that did not land)
        f.instances[f.store.buckets()[0].firstSlot].generation ^= 1u;
        const SceneCheckResult r = f.check();
        CHECK_FALSE(r.pass);
        CHECK(r.instanceErrors == 1);
    }
    {
        Fixture f; // a GPU-computed matrix one ulp off
        bool done = false;
        for (u32 s : f.store.motionSlots()) {
            u32 bits;
            std::memcpy(&bits, &f.instances[s].modelMatrix[12], 4);
            ++bits;
            std::memcpy(&f.instances[s].modelMatrix[12], &bits, 4);
            done = true;
            break;
        }
        REQUIRE(done);
        const SceneCheckResult r = f.check();
        CHECK_FALSE(r.pass);
        CHECK(r.matrixErrors >= 1);
    }
    {
        Fixture f; // a material that did not land
        f.materials.back().roughness += 0.5f;
        CHECK(f.check().materialErrors == 1);
    }
    {
        Fixture f; // a visible slot the GPU dropped (prefix + list + counters disagree)
        u32 s = 0;
        while (f.prefix[s + 1] == f.prefix[s]) ++s;
        for (size_t i = s + 1; i < f.prefix.size(); ++i) --f.prefix[i];
        f.visible.erase(f.visible.begin() + f.prefix[s]);
        f.visible.push_back(0);
        f.counters.visible -= 1;
        const SceneCheckResult r = f.check();
        CHECK_FALSE(r.pass);
        CHECK(r.visibleErrors + r.counterErrors + r.argErrors > 0);
    }
    {
        Fixture f; // a draw command the GPU did not write
        f.drawArgs[f.drawArgs.size() - 2] = 0xFFFFFFFFu;
        CHECK(f.check().argErrors >= 1);
    }
    {
        Fixture f; // a change the ECS did not report (missed touch)
        const auto& transforms = std::as_const(f.ecs).getArray<TransformComponent>();
        auto& t = const_cast<TransformComponent&>(transforms.data()[0]);
        t.worldMatrix[3][0] += 1.0f;
        const SceneCheckResult r = f.check();
        CHECK_FALSE(r.pass);
        CHECK_FALSE(r.ecs.empty());
    }
    {
        Fixture f; // queue overflow
        f.counters.queueOverflow = 1;
        CHECK_FALSE(f.check().pass);
    }
}
