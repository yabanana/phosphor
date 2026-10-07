#pragma once
#include "renderer/gpu_types.h"
#include <array>
#include <vector>
#include <glm/glm.hpp>

namespace phosphor {
// Independent implementation of Hillaire's low-dimensional atmosphere model.
// SI world metres / m^-1 throughout; RGB is linear, never pre-exposed.
struct AtmosphereSettings {
    glm::dvec3 planetCenter{0,-6360000,0};
    double bottomRadius=6360000,topRadius=6460000;
    double rayleighScaleHeight=8000,mieScaleHeight=1200,mieG=0.8;
    glm::dvec3 rayleighScattering{5.802e-6,13.558e-6,33.100e-6};
    glm::dvec3 mieScattering{3.996e-6},ozoneAbsorption{0.650e-6,1.881e-6,0.085e-6};
    double mieAbsorption=0.444e-6,ozoneCenterHeight=25000,ozoneHalfWidth=15000;
    glm::dvec3 groundAlbedo{0.3};
    u32 transmittanceWidth=256,transmittanceHeight=64,multiWidth=32,multiHeight=32,skyWidth=192,skyHeight=108;
    u32 marchSteps=64,multiDirections=64;
};
struct AtmosphereMedium { glm::dvec3 rayleigh{},mie{},extinction{}; };
struct AtmosphereSegment { double begin=0,end=0;bool ground=false,valid=false; };
struct AtmosphereIntegral {
    glm::dvec3 radiance{0},transmittance{1},scatteringFactor{0};
    double distance=0;bool ground=false;
};
void validateAtmosphere(const AtmosphereSettings& settings);
AtmosphereMedium atmosphereMedium(const AtmosphereSettings& settings,glm::dvec3 worldPoint);
AtmosphereSegment atmosphereSegment(const AtmosphereSettings&,glm::dvec3 worldPoint,glm::dvec3 direction,double limit=1e12);
glm::dvec3 atmosphereTransmittance(const AtmosphereSettings&,glm::dvec3 worldPoint,glm::dvec3 direction,u32 steps=256);
// Independent adaptive Simpson optical-depth quadrature; used by written
// reference tests, not the production midpoint LUT ray marcher.
glm::dvec3 atmosphereTransmittanceReference(const AtmosphereSettings&,glm::dvec3 worldPoint,glm::dvec3 direction,
                                           double tolerance=1e-9,u32 maxDepth=18);
AtmosphereIntegral atmosphereSingleScattering(const AtmosphereSettings&,glm::dvec3 worldPoint,glm::dvec3 direction,
                                              glm::dvec3 towardSun,glm::dvec3 irradiance,u32 steps=64,double limit=1e12);
glm::dvec2 atmosphereTransmittanceUv(const AtmosphereSettings&,double radius,double cosineZenith);
glm::dvec2 atmosphereTransmittanceRay(const AtmosphereSettings&,glm::dvec2 uv); // radius, cosine
double atmosphereRayleighPhase(double cosine);
double atmosphereMiePhase(double cosine,double g);

struct DayNightSettings {
    double dayLengthSeconds=1200,startDayFraction=0.25,latitudeRadians=0.7853981633974483,solarDeclinationRadians=0;
    double lunarOrbitDays=29.53059,lunarPhaseOffset=0.5,lunarInclinationRadians=0.08979719,jumpThresholdSeconds=2;
    glm::dvec3 sunIrradiance{25},moonFullIrradiance{0.01,0.012,0.02};
    double sunAngularRadius=0.00465,moonAngularRadius=0.0045,starIntensity=0.0001;
};
struct DayNightState {
    glm::dvec3 sunDirection{0,1,0},moonDirection{0,-1,0},sunIrradiance{},moonIrradiance{};
    double seconds=0,deltaSeconds=0,dayFraction=0,moonPhase=0,starRotation=0,exposureEv100=0;
    u64 epoch=1;bool reset=false;
};
class DayNightClock {
  public:
    explicit DayNightClock(DayNightSettings settings={});
    DayNightState sample(double clockSeconds,bool explicitJump=false);
    [[nodiscard]] const DayNightSettings& settings()const{return settings_;}
  private:
    DayNightSettings settings_;double previous_=0;u64 epoch_=1;bool valid_=false;
};
// Scene lights use atmosphere-attenuated irradiance at the reference position;
// atmosphere LUTs use the unattenuated extraterrestrial irradiance above.
std::array<GPULight,2> atmosphereDirectionalLights(const AtmosphereSettings&,const DayNightState&,glm::dvec3 referencePoint);
double proceduralStarRadiance(glm::dvec3 direction,double rotation,double angularFootprint,u32 seed=0x51a7u);

struct AtmosphereUpdate { bool transmittance=false,multiscattering=false,skyView=false;u32 parameterRevision=1,skyRevision=1; };
// Collision-free exact bit tuple equality; no parameter ages/hashes are hits.
class AtmosphereVersions {
  public:
    AtmosphereUpdate update(const AtmosphereSettings&,const DayNightState&,glm::dvec3 cameraPosition,bool force=false);
  private:
    std::vector<u64> physics_,sky_;u32 parameterRevision_=0,skyRevision_=0;
};
GPUAtmosphereParams makeAtmosphereParams(const AtmosphereSettings&,const DayNightSettings&,const DayNightState&,
                                         const AtmosphereUpdate&,glm::dvec3 cameraPosition,
                                         const float* inverseViewProjection,const float* viewProjection,
                                         u32 width,u32 height,u32 frame,u32 view);
} // namespace phosphor
