#include <doctest/doctest.h>
#include "testbench/reflection_validation.h"
#include "renderer/gpu_scene.h"
#include "renderer/scene_extract.h"
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
        fixture.setup(ecs,scene,textures);REQUIRE(scene.getMeshCount()==3);
        FrameScene frame;extractFrameScene(ecs,scene,frame);CHECK_FALSE(frame.instances.empty());CHECK_FALSE(frame.lights.empty());
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
