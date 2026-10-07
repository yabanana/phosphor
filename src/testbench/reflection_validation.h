#pragma once
#include "testbench/testbench.h"
#include <glm/gtc/quaternion.hpp>
#include <span>
#include <string_view>
#include <vector>

namespace phosphor {
// F13 analytic corpus, WRITTEN / NON VERIFIED. Root registers this subclass
// for --reflection-scene NAME on bench6. All distances are WORLD metres; RGB
// factors/emission are linear. Every mesh is visible AND casts shadows so the
// independent reference does not silently change ray-role semantics.
class ReflectionValidation final : public TestBench {
public:
    explicit ReflectionValidation(std::string scenario);
    static bool validScenario(std::string_view);
    static std::span<const std::string_view> scenarios();
    void setup(ECS&,GpuScene&,TextureManager&) override;
    void update(float dt,ECS&) override;
    void teardown(ECS&,GpuScene&) override;
    const char* getName()const override{return name_.c_str();}
    const char* assetSource()const override{return "procedural-reflection-validation-v1";}
    CameraSetup getDefaultCamera()const override;
    bool scriptedCamera(double time,glm::vec3& position,glm::vec3& target,bool& cut)const override;
    std::span<const EntityID> entities()const{return entities_;}
    std::span<const EntityID> roughnessSurfaces()const{return roughness_;}
    EntityID mirror()const{return mirror_;}
    EntityID emissivePanel()const{return panel_;}
    EntityID sun()const{return sun_;}
    EntityID occluder()const{return occluder_;}
    double elapsedSeconds()const{return time_;}
    static constexpr float UnitsInMetres=1.0f;
    static constexpr float MirrorPlaneZ=0.0f;
    static constexpr double LightStepSeconds=2.0,CameraCutSeconds=4.0,ScriptPeriodSeconds=8.0;
private:
    EntityID mesh(ECS&,u32,glm::vec3 position,glm::vec3 fullSize,glm::quat rotation,
                  glm::vec3 color,float metallic,float roughness,glm::vec3 emission=glm::vec3(0),bool isStatic=true);
    void move(ECS&,EntityID,glm::vec3,glm::quat);
    std::string scenario_,name_;
    std::vector<EntityID> entities_,roughness_;
    u32 plane_=~0u,cube_=~0u,sphere_=~0u;
    EntityID mirror_=INVALID_ENTITY,panel_=INVALID_ENTITY,sun_=INVALID_ENTITY,occluder_=INVALID_ENTITY;
    double time_=0;u32 lightStep_=0;
    mutable u32 cameraSegment_=~0u;
};
} // namespace phosphor
