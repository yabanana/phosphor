#include <doctest/doctest.h>
#include "testbench/reflection_validation.h"
#include "renderer/gpu_scene.h"
#include "renderer/scene_extract.h"
#include "renderer/reflection_probe.h"
#include "renderer/scene_store.h"
#include "scene/components.h"
#include "scene/ecs.h"
#include "null_texture_manager.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
using namespace phosphor;
TEST_CASE("F13 reflection corpus has actual world geometry and equivalent ray roles") {
    for(auto name:ReflectionValidation::scenarios()) {
        ECS ecs;GpuScene scene;test::NullTextureManager textures;ReflectionValidation fixture{std::string(name)};
        fixture.setup(ecs,scene,textures);REQUIRE(scene.getMeshCount()==((fixture.aoTemporalControl()||name=="specular-environment")?1u:3u));
        FrameScene frame;extractFrameScene(ecs,scene,frame);CHECK_FALSE(frame.instances.empty());if(fixture.aoTemporalControl()||name=="specular-environment")CHECK(frame.lights.empty());else CHECK_FALSE(frame.lights.empty());
        for(auto e:fixture.entities())if(ecs.hasComponent<MeshInstanceComponent>(e)) {
            const auto& i=std::as_const(ecs).getComponent<MeshInstanceComponent>(e);
            CHECK(i.isVisible());CHECK(i.castsShadows());
            const auto& m=std::as_const(ecs).getComponent<MaterialComponent>(e);
            CHECK(m.baseColorTexIndex==INVALID_TEXTURE_INDEX);CHECK(m.metallicRoughnessTexIndex==INVALID_TEXTURE_INDEX);
        }
        fixture.teardown(ecs,scene);CHECK(fixture.entities().empty());
    }
    CHECK_THROWS(ReflectionValidation{"unknown"});
}
TEST_CASE("F13 roughness corpus uses exact material factors rather than default MR texture") {
    ECS ecs;GpuScene scene;test::NullTextureManager textures;ReflectionValidation f{"roughness"};f.setup(ecs,scene,textures);
    constexpr std::array<float,5> expected{0.04f,0.1f,0.25f,0.5f,0.9f};
    REQUIRE(f.roughnessSurfaces().size()==expected.size());
    for(u32 k=0;k<expected.size();++k) {
        const auto& m=std::as_const(ecs).getComponent<MaterialComponent>(f.roughnessSurfaces()[k]);
        CHECK(m.metallicFactor==1);CHECK(m.roughnessFactor==expected[k]);
    }
}
TEST_CASE("F13 mirror fixture has offscreen emitter with independently predictable virtual point") {
    ECS ecs;GpuScene scene;test::NullTextureManager textures;ReflectionValidation f{"mirror"};f.setup(ecs,scene,textures);
    const auto camera=f.getDefaultCamera();
    const auto& emitter=std::as_const(ecs).getComponent<TransformComponent>(f.emissivePanel());
    const glm::dvec3 toward=glm::normalize(glm::dvec3(camera.target-camera.position));
    CHECK(glm::dot(glm::dvec3(emitter.position-camera.position),toward)<0); // Behind eye.
    // Independent ideal plane z=0 reflection geometry: virtual emitter z=-z.
    const glm::dvec3 virtualEmitter(emitter.position.x,emitter.position.y,-emitter.position.z),eye(camera.position);
    const glm::dvec3 ray=virtualEmitter-eye;
    const double t=-eye.z/ray.z;
    const auto mirrorPoint=eye+t*ray;
    CHECK(mirrorPoint.z==doctest::Approx(0));CHECK(std::abs(mirrorPoint.x)<2);
    CHECK(mirrorPoint.y>0);CHECK(mirrorPoint.y<3.4);
    const auto incoming=glm::normalize(mirrorPoint-eye);
    const auto reflected=incoming-2*glm::dot(incoming,glm::dvec3(0,0,1))*glm::dvec3(0,0,1);
    CHECK(glm::dot(reflected,glm::normalize(glm::dvec3(emitter.position)-mirrorPoint))==doctest::Approx(1));
}
TEST_CASE("F13 emissive step and camera cut are tracked and deterministic") {
    ECS ecs;GpuScene scene;test::NullTextureManager textures;ReflectionValidation moving{"moving-light"};moving.setup(ecs,scene,textures);
    const auto before=std::as_const(ecs).getComponent<MaterialComponent>(moving.emissivePanel()).emissiveFactor;
    moving.update(2.01f,ecs);
    CHECK(std::as_const(ecs).getComponent<MaterialComponent>(moving.emissivePanel()).emissiveFactor==glm::vec3(0));
    CHECK(before!=glm::vec3(0));
    CHECK_FALSE(ecs.getArray<MaterialComponent>().changes().empty());
    ReflectionValidation camera{"disocclusion"};glm::vec3 p,t;bool cut=false;
    REQUIRE(camera.scriptedCamera(0,p,t,cut));CHECK_FALSE(cut);
    REQUIRE(camera.scriptedCamera(3.99,p,t,cut));CHECK_FALSE(cut);
    REQUIRE(camera.scriptedCamera(4.01,p,t,cut));CHECK(cut);
    REQUIRE(camera.scriptedCamera(4.1,p,t,cut));CHECK_FALSE(cut);
}

TEST_CASE("Opaque wide-emission fixture has an independent constant-radiance reference") {
    ECS ecs;GpuScene scene;test::NullTextureManager textures;ReflectionValidation f{"wide-emission"};f.setup(ecs,scene,textures);
    const auto e=f.emissivePanel();REQUIRE(e!=INVALID_ENTITY);
    const auto& m=std::as_const(ecs).getComponent<MaterialComponent>(e);
    CHECK(m.emissiveFactor==glm::vec3(368640,128,64));CHECK(m.emissiveFactor.x>65504.f);
    CHECK(m.baseColorFactor==glm::vec4(0,0,0,1));CHECK(m.emissiveTexIndex==INVALID_TEXTURE_INDEX);
    CHECK(m.alphaCutoff==0);CHECK(m.occlusionStrength==1);REQUIRE(m.occlusionTexIndex<textures.uploads.size());
    const std::vector<u8> black{0,0,0,255};CHECK(textures.uploads[m.occlusionTexIndex].rgba==black);
    FrameScene frame;extractFrameScene(ecs,scene,frame);REQUIRE(frame.instances.size()==2);
    for(const auto& light:frame.lights)CHECK(light.intensity==0);
    const auto& t=std::as_const(ecs).getComponent<TransformComponent>(e);const auto camera=f.getDefaultCamera();
    // Probe bounds consume packed SceneStore slots, including VALID flags;
    // FrameScene extraction precedes that runtime packing step.
    SceneStore store;store.sync(ecs,scene);
    std::vector<float> worlds;for(const auto& instance:store.instances())worlds.insert(worlds.end(),instance.modelMatrix,instance.modelMatrix+16);
    const auto bounds=reflectionProbeWorldBounds(store.instances(),scene.meshInfos(),worlds);REQUIRE(bounds);
    const auto capture=(bounds->minimum+bounds->maximum)*.5f;
    CHECK(capture.z>0.05f);CHECK(capture.z<20); // The emissive front is visible from the probe.
    for(auto id:f.entities())if(id!=e&&ecs.hasComponent<MeshInstanceComponent>(id)) {
        const auto& anchor=std::as_const(ecs).getComponent<TransformComponent>(id);
        CHECK(anchor.position.z>camera.position.z); // Never affects the primary radiance oracle.
    }
    // Geometric coverage is independent of renderer projection code: a 60deg
    // perspective at z=4 and 16:9 has half-width <4.2, inside the 8m plane.
    CHECK(t.scale==glm::vec3(16,1,16));CHECK(camera.position==glm::vec3(0,0,4));
    CHECK(4*std::tan(3.14159265358979323846/6)*16/9<8);
    const auto before=t.worldMatrix;f.update(1,ecs);
    CHECK(std::as_const(ecs).getComponent<TransformComponent>(e).worldMatrix==before);
    CHECK(std::as_const(ecs).getComponent<MaterialComponent>(e).emissiveFactor==glm::vec3(368640,128,64));
}

TEST_CASE("F13 physical AO temporal fixture preserves receiver and steps its only wall once") {
    ECS ecs;GpuScene scene;test::NullTextureManager textures;ReflectionValidation f{"ao-temporal-wall"};f.setup(ecs,scene,textures);
    const auto receiver=std::as_const(ecs).getComponent<TransformComponent>(f.mirror());
    const auto initialWall=std::as_const(ecs).getComponent<TransformComponent>(f.occluder());
    CHECK(receiver.position==glm::vec3(0));CHECK(initialWall.position.x==doctest::Approx(.8));
    CHECK(std::as_const(ecs).getComponent<MaterialComponent>(f.occluder()).doubleSided);
    FrameScene frame;extractFrameScene(ecs,scene,frame);CHECK(frame.instances.size()==2);CHECK(frame.lights.empty());
    for(u32 ordinal=0;ordinal<64;++ordinal){
        f.update(0,ecs);f.update(1.f/60.f,ecs);CHECK(f.aoTemporalOrdinal()==ordinal);
        const auto& wall=std::as_const(ecs).getComponent<TransformComponent>(f.occluder());
        CHECK(wall.position.x==doctest::Approx(ordinal<32?.8f:8.f));
        CHECK(std::as_const(ecs).getComponent<TransformComponent>(f.mirror()).worldMatrix==receiver.worldMatrix);
    }
    const auto camera=f.getDefaultCamera();CHECK(camera.position==glm::vec3(0,0,3));CHECK(camera.target==glm::vec3(0));CHECK_FALSE(camera.orbit);
}
