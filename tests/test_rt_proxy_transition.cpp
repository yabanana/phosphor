#include <doctest/doctest.h>
#include <array>
#include "renderer/rt_proxy_transition.h"
#include "renderer/gpu_scene.h"
#include "renderer/scene_store.h"
#include "scene/ecs.h"

using namespace phosphor;

namespace {
struct Fixture {
    ECS ecs;
    GpuScene scene;
    SceneStore store;
    RtProxyManifest manifest;
    RtProxyGeometry proxy;
    Fixture() {
        std::vector<glm::vec3> positions;
        std::vector<u32> indices;
        constexpr u32 n = 12;
        for (u32 y = 0; y <= n; ++y) for (u32 x = 0; x <= n; ++x) positions.emplace_back(x,0,y);
        for (u32 y = 0; y < n; ++y) for (u32 x = 0; x < n; ++x) {
            const u32 a = y*(n+1)+x, b=a+1, c=a+n+1, d=c+1;
            indices.insert(indices.end(),{a,c,b,b,c,d});
        }
        scene.uploadMesh(positions, {}, {}, {}, indices);
        auto material = GPUMaterial{};
        material.baseColor[3] = 1;
        material.baseColorTex = material.normalTex = material.metallicRoughnessTex =
            material.occlusionTex = material.emissiveTex = INVALID_TEXTURE_INDEX;
        // A library large enough to exercise the real material-delta threshold.
        for (u32 i = 0; i < 32; ++i) scene.addMaterial(material);
        const auto entity = ecs.createEntity();
        ecs.addComponent(entity, TransformComponent{});
        MeshInstanceComponent instance;
        instance.meshHandle = 0; instance.materialIndex = 0; instance.flags = 3;
        ecs.addComponent(entity, std::move(instance));
        sync();
        const std::array levels{RtProxyLevel::R10};
        // Synthetic unit fixture only; no production manifest or GPU claim.
        manifest = rtMakeProxyManifest(scene, levels, {1,1,0,0,0,0}, "unit-plane", "unit-plane-cpu-fixture");
        reload();
    }
    void sync() { store.sync(ecs,scene); ecs.endFrame(); }
    void reload() {
        proxy = rtBuildProxyGeometry(scene,&manifest,rtProxyProtectedMeshes(scene.getMeshCount(),store.instances(),store.materials()));
    }
    void stage(RtProxyTransitionCheck& control,RtProxyTransition mode) {
        REQUIRE(control.arm(mode,ecs,scene,store,proxy));
        sync(); // settle the deliberately non-protecting preparation
        REQUIRE(control.afterSync(store,proxy));
        REQUIRE(control.beforeSync(7,ecs,scene));
        CHECK_FALSE(control.status().applied);
        REQUIRE(control.beforeSync(8,ecs,scene));
        sync();
    }
};
}

TEST_CASE("RT proxy transition exercises material delta assignment and full-upload paths") {
    for (auto mode : {RtProxyTransition::Mask,RtProxyTransition::Emissive,RtProxyTransition::Reassign,RtProxyTransition::FullUpload}) {
        CAPTURE(rtProxyTransitionName(mode));
        Fixture f;
        RtProxyTransitionCheck control;
        REQUIRE(f.proxy.proxyTriangles < f.proxy.fullTriangles);
        f.stage(control,mode);
        f.reload(); // models the existing engine guard, not an action of the controller
        REQUIRE(control.afterSync(f.store,f.proxy));
        CHECK(control.status().applied);
        CHECK(control.status().promoted);
        CHECK_FALSE(control.complete());
        REQUIRE(control.afterReadback(f.store.instances(),f.store.materials(),true));
        CHECK(control.complete());
        CHECK(control.finish());
        CHECK(control.status().finalIndices == control.status().sourceIndices);
        CHECK(control.line().find(" | PASS") != std::string::npos);
        if (mode == RtProxyTransition::FullUpload) {
            CHECK(control.status().materialFull);
            CHECK(control.status().instancesFull);
            CHECK(control.status().materialRecords == 0);
            CHECK(control.status().instanceRecords == 0);
        }
    }
}

TEST_CASE("RT proxy transition cannot pass from a checker result alone") {
    Fixture f;
    RtProxyTransitionCheck control;
    REQUIRE(control.arm(RtProxyTransition::Mask,f.ecs,f.scene,f.store,f.proxy));
    CHECK(control.afterReadback(f.store.instances(),f.store.materials(),true));
    CHECK_FALSE(control.complete());
    CHECK_FALSE(control.finish());
    CHECK(control.line().find(" | FAIL") != std::string::npos);
}

TEST_CASE("RT proxy transition rejects missing promotion and stale GPU material") {
    SUBCASE("no engine reload") {
        Fixture f; RtProxyTransitionCheck control;
        f.stage(control,RtProxyTransition::Mask);
        CHECK_FALSE(control.afterSync(f.store,f.proxy));
        CHECK_FALSE(control.finish());
    }
    SUBCASE("GPU upload stale") {
        Fixture f; RtProxyTransitionCheck control;
        f.stage(control,RtProxyTransition::Emissive);
        f.reload();
        REQUIRE(control.afterSync(f.store,f.proxy));
        auto stale = std::vector<GPUMaterial>(f.store.materials().begin(),f.store.materials().end());
        for (auto& m : stale) m.emissive[0] = 0;
        CHECK_FALSE(control.afterReadback(f.store.instances(),stale,true));
        CHECK_FALSE(control.complete());
    }
    SUBCASE("actual RT checker failure") {
        Fixture f; RtProxyTransitionCheck control;
        f.stage(control,RtProxyTransition::FullUpload);
        f.reload();
        REQUIRE(control.afterSync(f.store,f.proxy));
        CHECK_FALSE(control.afterReadback(f.store.instances(),f.store.materials(),false));
    }
    SUBCASE("no measured proxy") {
        Fixture f; RtProxyTransitionCheck control;
        f.proxy = rtBuildProxyGeometry(f.scene);
        CHECK_FALSE(control.arm(RtProxyTransition::Mask,f.ecs,f.scene,f.store,f.proxy));
    }
}
