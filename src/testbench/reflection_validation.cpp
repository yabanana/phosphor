#include "testbench/reflection_validation.h"
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
constexpr std::array<std::string_view,8> names{"mirror","roughness","ao-cavity","probe-parallax","moving-light","disocclusion","wide-emission","ao-temporal-wall"};
constexpr float pi=3.14159265358979323846f;
const glm::quat identity(1,0,0,0);
glm::quat sunRotation(glm::vec3 towardSun) {
    const glm::vec3 a(0,0,-1),b=-glm::normalize(towardSun);
    const float c=std::clamp(glm::dot(a,b),-1.f,1.f);
    if(c>0.99999f)return identity;
    if(c<-0.99999f)return glm::angleAxis(pi,glm::vec3(0,1,0));
    return glm::angleAxis(std::acos(c),glm::normalize(glm::cross(a,b)));
}
}
ReflectionValidation::ReflectionValidation(std::string scenario):scenario_(std::move(scenario)) {
    if(!validScenario(scenario_))throw std::invalid_argument("Unknown reflection fixture: "+scenario_);
    name_="Reflection validation: "+scenario_;
}
bool ReflectionValidation::validScenario(std::string_view name){return std::find(names.begin(),names.end(),name)!=names.end();}
std::span<const std::string_view> ReflectionValidation::scenarios(){return names;}
EntityID ReflectionValidation::mesh(ECS& ecs,u32 handle,glm::vec3 p,glm::vec3 size,glm::quat q,
                                     glm::vec3 color,float metallic,float roughness,glm::vec3 emission,bool isStatic) {
    const auto e=ecs.createEntity();entities_.push_back(e);
    TransformComponent t;t.position=p;t.scale=size;t.rotation=q;t.updateMatrix();ecs.addComponent(e,std::move(t));
    MeshInstanceComponent i;i.meshHandle=handle;i.materialIndex=0;i.setVisible(true);i.setCastsShadows(true);i.setStatic(isStatic);
    ecs.addComponent(e,std::move(i));
    MaterialComponent m;m.baseColorFactor=glm::vec4(color,1);m.metallicFactor=metallic;m.roughnessFactor=roughness;
    m.emissiveFactor=emission;m.doubleSided=false;
    // No default MR/normal textures: they would alter exact analytic factors.
    ecs.addComponent(e,std::move(m));return e;
}
void ReflectionValidation::setup(ECS& ecs,GpuScene& scene,TextureManager& textures) {
    if(!entities_.empty())throw std::logic_error("Reflection fixture setup requires teardown");
    textures.createDefaultTextures(); // Existing scene fallback material refers to these slots.
    auto upload=[&](MeshData data){return scene.uploadMesh(data.positions,data.normals,data.tangents,data.uvs,data.indices);};
    plane_=upload(ProceduralMeshes::generatePlane(1,1,1,1));
    if(aoTemporalControl()) {
        const glm::quat faceCamera=glm::angleAxis(pi*.5f,glm::vec3(1,0,0));
        mirror_=mesh(ecs,plane_,{0,0,0},{8,1,8},faceCamera,{.5f,.5f,.5f},0,1);
        // Rotate the plane from +Y to -X; the receiver normal is perpendicular.
        // Size8 covers every radius2 AO hit. Both-sided wall is physical opacity.
        occluder_=mesh(ecs,plane_,{.8f,0,0},{8,1,8},glm::angleAxis(pi*.5f,glm::vec3(0,0,1)),{.5f,.5f,.5f},0,1);
        ecs.getComponent<MaterialComponent>(occluder_).doubleSided=true;
        time_=0;positiveUpdates_=0;cameraSegment_=~0u;return;
    }
    cube_=upload(ProceduralMeshes::generateCube(0.5f));
    sphere_=upload(ProceduralMeshes::generateSphere(0.5f,32,16));
    GPUMaterial fallback{};fallback.baseColor[3]=1;
    fallback.baseColorTex=fallback.normalTex=fallback.metallicRoughnessTex=fallback.occlusionTex=fallback.emissiveTex=INVALID_TEXTURE_INDEX;
    scene.addMaterial(fallback);
    if(scenario_=="wide-emission") {
        // An actual opaque raster surface, not the standalone SDK texture fixture.
        // Full camera coverage, no incident light, zero ambient occlusion and
        // black albedo make outgoing radiance exactly the declared emission.
        panel_=mesh(ecs,plane_,{0,0,0},{16,1,16},glm::angleAxis(pi*.5f,glm::vec3(1,0,0)),
                    {0,0,0},0,1,{368640,128,64});
        auto& material=ecs.getComponent<MaterialComponent>(panel_);
        material.occlusionTexIndex=textures.getDefaultBlack();material.occlusionStrength=1;
        // Move the fitted probe center off the emitter plane without changing
        // primary visibility: this black static anchor is behind the camera.
        const auto anchor=mesh(ecs,cube_,{0,0,20},{1,1,1},identity,{0,0,0},0,1);
        auto& anchorMaterial=ecs.getComponent<MaterialComponent>(anchor);
        anchorMaterial.occlusionTexIndex=textures.getDefaultBlack();anchorMaterial.occlusionStrength=1;
        sun_=ecs.createEntity();entities_.push_back(sun_);
        TransformComponent sunPose;sunPose.updateMatrix();ecs.addComponent(sun_,std::move(sunPose));
        LightComponent light;light.type=LightType::Directional;light.intensity=0;ecs.addComponent(sun_,std::move(light));
        time_=0;lightStep_=0;cameraSegment_=~0u;return;
    }
    mesh(ecs,plane_,{0,0,0},{12,1,12},identity,{0.65f,0.65f,0.65f},0,0.8f);
    // Noise-free sun: local -Z is travel direction, exactly as extractLights.
    sun_=ecs.createEntity();entities_.push_back(sun_);
    TransformComponent sunPose;sunPose.rotation=sunRotation({-0.35f,0.85f,0.4f});sunPose.updateMatrix();
    ecs.addComponent(sun_,std::move(sunPose));
    LightComponent sunLight;sunLight.type=LightType::Directional;sunLight.intensity=12;sunLight.color={1,1,1};
    ecs.addComponent(sun_,std::move(sunLight));
    const glm::quat towardCamera=glm::angleAxis(pi*0.5f,glm::vec3(1,0,0));
    if(scenario_=="roughness") {
        constexpr std::array<float,5> values{0.04f,0.1f,0.25f,0.5f,0.9f};
        for(u32 k=0;k<values.size();++k)
            roughness_.push_back(mesh(ecs,sphere_,{float(k)-2,0.65f,0},{1,1,1},identity,{0.85f,0.78f,0.62f},1,values[k]));
        panel_=mesh(ecs,plane_,{0,2.6f,2.5f},{4,1,1},glm::angleAxis(pi,glm::vec3(0,0,1)),{0.9f,0.9f,0.9f},0,1,{8,8,8});
    } else if(scenario_=="ao-cavity") {
        mesh(ecs,cube_,{-1.5f,0.7f,-0.4f},{0.25f,1.4f,2.4f},identity,{0.7f,0.7f,0.7f},0,1);
        mesh(ecs,cube_,{-0.6f,0.7f,-1.5f},{2,1.4f,0.25f},identity,{0.7f,0.7f,0.7f},0,1);
        mesh(ecs,sphere_,{2,0.7f,0},{1.2f,1.2f,1.2f},identity,{0.7f,0.7f,0.7f},0,1); // open AO control
        panel_=mesh(ecs,plane_,{0,3,0},{2,1,2},glm::angleAxis(pi,glm::vec3(0,0,1)),{1,1,1},0,1,{6,6,6});
    } else {
        mirror_=mesh(ecs,plane_,{0,1.7f,MirrorPlaneZ},{4,1,3.4f},towardCamera,{0.95f,0.95f,0.95f},1,0.04f);
        // Emissive red surface behind the default camera: primary SSR cannot
        // discover it from a visibility buffer, a true mirror RT ray can.
        panel_=mesh(ecs,cube_,{1.0f,1.3f,8.4f},{1.1f,1.6f,0.3f},identity,{0.75f,0.1f,0.06f},0,0.8f,{8,0.4f,0.2f},scenario_!="moving-light");
        mesh(ecs,cube_,{-1.2f,0.65f,3.0f},{0.9f,1.3f,0.9f},identity,{0.04f,0.12f,0.65f},0,0.8f);
        if(scenario_=="probe-parallax") {
            // Local color walls distinguish a world-space box-projected probe
            // from a direction-only lookup while the camera translates.
            mesh(ecs,plane_,{-3,1.5f,1},{6,1,3},glm::angleAxis(-pi*0.5f,glm::vec3(0,0,1)),{0.65f,0.05f,0.03f},0,1);
            mesh(ecs,plane_,{3,1.5f,1},{6,1,3},glm::angleAxis(pi*0.5f,glm::vec3(0,0,1)),{0.04f,0.6f,0.1f},0,1);
            roughness_.push_back(mesh(ecs,sphere_,{0,0.65f,2},{1,1,1},identity,{0.9f,0.9f,0.9f},1,0.35f));
        }
        if(scenario_=="disocclusion")
            occluder_=mesh(ecs,cube_,{0,1.25f,2.0f},{1.15f,2.5f,0.4f},identity,{0.5f,0.45f,0.2f},0,0.9f,glm::vec3(0),false);
    }
    time_=0;lightStep_=0;cameraSegment_=~0u;
}
void ReflectionValidation::move(ECS& ecs,EntityID e,glm::vec3 p,glm::quat q) {
    if(e==INVALID_ENTITY)return;
    const auto& old=std::as_const(ecs).getComponent<TransformComponent>(e);
    if(old.position==p&&old.rotation==q)return;
    auto& t=ecs.getComponent<TransformComponent>(e);t.position=p;t.rotation=q;t.updateMatrix();
}
void ReflectionValidation::update(float dt,ECS& ecs) {
    if(!std::isfinite(dt)||dt<0)throw std::invalid_argument("Reflection fixture dt must be finite nonnegative");
    if(dt==0)return;time_+=dt;
    if(aoTemporalControl()) {
        const auto ordinal=positiveUpdates_++;
        if(ordinal==AOWallStep)move(ecs,occluder_,{8,0,0},glm::angleAxis(pi*.5f,glm::vec3(0,0,1)));
        return;
    }
    if(scenario_=="moving-light") {
        const float phase=float(std::fmod(time_,ScriptPeriodSeconds));
        move(ecs,panel_,{1.0f+0.7f*std::sin(phase),1.3f,8.4f},identity);
        auto& sunPose=ecs.getComponent<TransformComponent>(sun_);
        sunPose.rotation=sunRotation({-0.35f+0.3f*std::sin(phase*0.6f),0.85f,0.4f});sunPose.updateMatrix();
        const u32 step=u32(time_/LightStepSeconds)%4u;
        if(step!=lightStep_) {
            auto& m=ecs.getComponent<MaterialComponent>(panel_);
            m.emissiveFactor=step==1u?glm::vec3(0):step==2u?glm::vec3(0.2f,0.5f,8):glm::vec3(8,0.4f,0.2f);
            ecs.getComponent<LightComponent>(sun_).intensity=step==1u?6.f:12.f;
            lightStep_=step;
        }
    } else if(scenario_=="disocclusion") {
        const float phase=float(std::fmod(time_,ScriptPeriodSeconds));
        const float x=phase<2?0:phase<4?1.8f*(phase-2)/2:1.8f;
        move(ecs,occluder_,{x,1.25f,2},identity);
    }
}
void ReflectionValidation::teardown(ECS& ecs,GpuScene&) {
    for(auto e:entities_)ecs.destroyEntity(e);entities_.clear();roughness_.clear();
    panel_=mirror_=sun_=occluder_=INVALID_ENTITY;plane_=cube_=sphere_=~0u;time_=0;positiveUpdates_=0;cameraSegment_=~0u;
}
CameraSetup ReflectionValidation::getDefaultCamera()const {
    if(scenario_=="wide-emission")return {{0,0,4},{0,0,0},4,false};
    if(aoTemporalControl())return {{0,0,3},{0,0,0},3,false};
    return {{0,2.2f,7},{0,1.1f,0},7,false};
}
bool ReflectionValidation::scriptedCamera(double t,glm::vec3& p,glm::vec3& target,bool& cut)const {
    cut=false;if(!std::isfinite(t)||t<0)return false;
    if(scenario_!="probe-parallax"&&scenario_!="disocclusion")return false;
    const double phase=std::fmod(t,ScriptPeriodSeconds);
    const u32 segment=u32(t/ScriptPeriodSeconds)*2u+(phase>=CameraCutSeconds?1u:0u);
    cut=cameraSegment_!=~0u&&cameraSegment_!=segment;cameraSegment_=segment;
    target={0,1.1f,0};
    if(scenario_=="probe-parallax")p={float(1.4*std::sin(phase*0.5)),2.2f,7};
    else p=phase<4?glm::vec3(float(-0.8+0.45*phase),2.2f,7):glm::vec3(-1.8f,2.5f,6.5f);
    return true;
}
} // namespace phosphor
