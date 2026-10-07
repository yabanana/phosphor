#include <doctest/doctest.h>
#include "renderer/atmosphere.h"
#include <algorithm>
#include <cmath>
#include <limits>
using namespace phosphor;
namespace {
AtmosphereSettings vacuum(){AtmosphereSettings a;a.rayleighScattering=a.mieScattering=a.ozoneAbsorption=glm::dvec3(0);a.mieAbsorption=0;a.groundAlbedo=glm::dvec3(0);return a;}
}
TEST_CASE("F14 atmosphere scalar ABI and SI Earth coefficients are explicit") {
    CHECK(sizeof(GPUAtmosphereParams)==384);CHECK(sizeof(GPUFogParams)==448);CHECK(sizeof(GPUCloudParams)==272);
    AtmosphereSettings a;CHECK_NOTHROW(validateAtmosphere(a));CHECK(a.bottomRadius==6360000);CHECK(a.rayleighScattering.z<1e-4);
    SUBCASE("negative scattering"){a.rayleighScattering.x=-1;}
    SUBCASE("nan scale"){a.mieScaleHeight=std::numeric_limits<double>::quiet_NaN();}
    SUBCASE("unbounded march"){a.marchSteps=513;}
    SUBCASE("opaque ground albedo"){a.groundAlbedo.x=1.1;}
    CHECK_THROWS(validateAtmosphere(a));
}
TEST_CASE("F14 vacuum RTE limit has transmittance one and zero in-scattering") {
    const auto a=vacuum();const glm::dvec3 camera(0,2,0),ray(0,1,0);
    const auto T=atmosphereTransmittance(a,camera,ray);
    const auto value=atmosphereSingleScatteringReference(a,camera,ray,ray,{1,1,1});
    CHECK(T.x==1);CHECK(T.y==1);CHECK(T.z==1);CHECK(value.radiance.x==0);CHECK(value.transmittance.x==1);
    CHECK(atmosphereMultipleScatteringReference(a,a.bottomRadius+1000,1,2,2).x==0);
}
TEST_CASE("F14 opaque endpoint on planetary ground contributes its material exactly once") {
    auto a=vacuum();a.groundAlbedo=glm::dvec3(.3,.2,.1);
    const glm::dvec3 camera(0,3,0),ray(0,-1,0),sun(0,1,0),E(5,7,11),scene(2,3,4);
    // Analytic vacuum transport is identity for any already-shaded endpoint,
    // independently of the unrelated planet-ground material and sunlight.
    const auto volume=atmosphereSingleScattering(a,camera,ray,sun,E,4,3,false);
    const auto reference=atmosphereSingleScatteringReference(a,camera,ray,sun,E,1e-8,4,3,false);
    REQUIRE(volume.ground);REQUIRE(reference.ground);
    const auto sky=atmosphereSingleScattering(a,camera,ray,sun,E,4,3);
    const auto skyReference=atmosphereSingleScatteringReference(a,camera,ray,sun,E,1e-8,4,3);
    const auto analyticGround=a.groundAlbedo*E/3.14159265358979323846;
    for(u32 c=0;c<3;++c) {
        CHECK(volume.transmittance[c]==1);CHECK(reference.transmittance[c]==1);
        CHECK(volume.radiance[c]==0);CHECK(reference.radiance[c]==0);
        CHECK(scene[c]*volume.transmittance[c]+volume.radiance[c]==scene[c]);
        CHECK(sky.radiance[c]==doctest::Approx(analyticGround[c]).epsilon(1e-7));
        CHECK(skyReference.radiance[c]==doctest::Approx(analyticGround[c]).epsilon(1e-7));
        // Old composition added the planet's material on top of this mesh.
        CHECK(scene[c]+sky.radiance[c]>scene[c]);
    }
}
TEST_CASE("F14 vertical Rayleigh ozone optical depth agrees with independent analytic integral") {
    AtmosphereSettings a;a.mieScattering=glm::dvec3(0);a.mieAbsorption=0;
    const double height=a.topRadius-a.bottomRadius;
    const glm::dvec3 tau=a.rayleighScattering*a.rayleighScaleHeight*(1-std::exp(-height/a.rayleighScaleHeight))+a.ozoneAbsorption*a.ozoneHalfWidth;
    const auto expected=glm::exp(-tau),reference=atmosphereTransmittanceReference(a,{0,0,0},{0,1,0}),midpoint=atmosphereTransmittance(a,{0,0,0},{0,1,0},1024);
    for(u32 c=0;c<3;++c){CHECK(reference[c]==doctest::Approx(expected[c]).epsilon(1e-7));CHECK(midpoint[c]==doctest::Approx(expected[c]).epsilon(2e-4));}
    // Independent negative control: treating m^-1 as km^-1 would multiply
    // tau by1000; this fixture cannot accept that image/reference.
    CHECK(std::abs(reference.z-std::exp(-tau.z*1000))>0.1);
}
TEST_CASE("F14 sun horizon space and solid planet transmittance have finite limits") {
    AtmosphereSettings a;
    for(auto ray:{glm::dvec3(0,1,0),glm::dvec3(1,0.01,0),glm::dvec3(1,0,0)}){
        const auto T=atmosphereTransmittanceReference(a,{0,2,0},ray,1e-7,14);
        for(u32 c=0;c<3;++c){CHECK(std::isfinite(T[c]));CHECK(T[c]>=0);CHECK(T[c]<=1);}}
    CHECK(atmosphereTransmittance(a,{0,2,0},{0,-1,0}).x==0);
    CHECK(atmosphereTransmittance(a,{0,200000,0},{0,1,0}).x==1);
    const auto space=atmosphereMedium(a,{0,200000,0});CHECK(space.extinction.x==0);
}
TEST_CASE("F14 transmittance LUT mapping roundtrips physical rays away from the singular tangent") {
    const AtmosphereSettings a;
    for(double h:{0.0,1000.0,20000.0,90000.0})for(double mu:{0.0,0.1,0.5,1.0}){
        const auto uv=atmosphereTransmittanceUv(a,a.bottomRadius+h,mu),ray=atmosphereTransmittanceRay(a,uv);
        CHECK(ray.x==doctest::Approx(a.bottomRadius+h).epsilon(1e-10));CHECK(ray.y==doctest::Approx(mu).epsilon(1e-7));}
}
TEST_CASE("F14 sky azimuth lookup is periodic while physical LUT coordinates stay bounded") {
    constexpr int width=192;constexpr double pi=3.14159265358979323846;
    std::array<double,width> sky{};
    for(int i=0;i<width;++i)sky[i]=1+.5*std::sin(2*pi*(i+.5)/width);
    // Independent normalized linear-texture lookup on a known continuous
    // periodic radiance field. Texel centres lie at (i+.5)/width.
    const auto lookup=[&](double u,bool repeat){
        const double coordinate=u*width-.5;const int lo=int(std::floor(coordinate));
        const double fraction=coordinate-lo;
        const auto at=[&](int i){return sky[repeat?(i%width+width)%width:std::clamp(i,0,width-1)];};
        return at(lo)*(1-fraction)+at(lo+1)*fraction;
    };
    constexpr double epsilon=1e-9;
    CHECK(lookup(epsilon,true)==doctest::Approx(lookup(1-epsilon,true)).epsilon(1e-8));
    CHECK(lookup(0,true)==doctest::Approx(1).epsilon(1e-12));
    // Old clamp addressing has a finite seam, not interpolation error.
    CHECK(lookup(epsilon,false)-lookup(1-epsilon,false)==doctest::Approx(std::sin(pi/width)).epsilon(1e-9));
    const double step=2*pi/width,analyticLinearErrorBound=.5*step*step/8;
    for(int i=0;i<65;++i){const double u=double(i)/64;
        CHECK(std::abs(lookup(u,true)-(1+.5*std::sin(2*pi*u)))<=analyticLinearErrorBound+1e-12);
        CHECK(lookup(u,true)==doctest::Approx(lookup(u+3,true)).epsilon(1e-12));}
}
TEST_CASE("F14 phase functions normalize and retain forward scattering sign") {
    constexpr u32 samples=20000;double rayleigh=0,hg=0;
    for(u32 i=0;i<samples;++i){const double mu=-1+2*(double(i)+0.5)/samples;rayleigh+=atmosphereRayleighPhase(mu)*4*3.141592653589793/samples;hg+=atmosphereMiePhase(mu,0.8)*4*3.141592653589793/samples;}
    CHECK(rayleigh==doctest::Approx(1).epsilon(1e-6));CHECK(hg==doctest::Approx(1).epsilon(1e-4));
    CHECK(atmosphereMiePhase(1,0.8)>atmosphereMiePhase(-1,0.8));
}
TEST_CASE("F14 exact LUT revisions isolate physical medium sky camera and light changes") {
    AtmosphereVersions versions;AtmosphereSettings settings;DayNightClock clock;auto state=clock.sample(0);
    const auto initial=versions.update(settings,state,{0,2,0});CHECK(initial.transmittance);CHECK(initial.multiscattering);CHECK(initial.skyView);
    const auto unchanged=versions.update(settings,state,{0,2,0});CHECK_FALSE(unchanged.transmittance);CHECK_FALSE(unchanged.skyView);
    state.sunDirection=glm::normalize(glm::dvec3(0.1,1,0));const auto light=versions.update(settings,state,{0,2,0});CHECK_FALSE(light.transmittance);CHECK(light.skyView);
    const auto moved=versions.update(settings,state,{0,3,0});CHECK_FALSE(moved.transmittance);CHECK(moved.skyView);
    settings.rayleighScattering.x=std::nextafter(settings.rayleighScattering.x,1.0);const auto exact=versions.update(settings,state,{0,3,0});
    CHECK(exact.transmittance);CHECK(exact.multiscattering);CHECK(exact.skyView);CHECK(exact.parameterRevision>initial.parameterRevision);
}
TEST_CASE("F14 one clock is deterministic for sun moon stars exposure and resets jumps") {
    DayNightSettings settings;settings.latitudeRadians=0;settings.solarDeclinationRadians=0;settings.dayLengthSeconds=1200;
    DayNightClock a(settings),b(settings);
    const auto first=a.sample(0),same=b.sample(0);CHECK(first.sunDirection==same.sunDirection);CHECK(first.exposureEv100==same.exposureEv100);
    const auto ordinary=a.sample(0.016);CHECK_FALSE(ordinary.reset);CHECK(ordinary.epoch==first.epoch);
    const auto noon=a.sample(300);CHECK(noon.reset);CHECK(noon.epoch>first.epoch);CHECK(noon.sunDirection.y>0.999);
    const auto night=a.sample(900);CHECK(night.sunDirection.y<-0.999);CHECK(std::isfinite(night.exposureEv100));CHECK(night.exposureEv100<noon.exposureEv100);
    const auto explicitJump=a.sample(900.016,true);CHECK(explicitJump.reset);
    const auto backwards=a.sample(800);CHECK(backwards.reset);CHECK(backwards.moonPhase>=0);CHECK(backwards.moonPhase<=1);
    CHECK_THROWS(a.sample(std::numeric_limits<double>::infinity()));
}
TEST_CASE("F14 directional scene lights attenuate through the same atmosphere before exposure") {
    DayNightSettings c;c.latitudeRadians=0;c.startDayFraction=0.5;DayNightClock clock(c);const auto state=clock.sample(0);
    AtmosphereSettings a;const auto lights=atmosphereDirectionalLights(a,state,{0,2,0});
    const auto expected=state.sunIrradiance*atmosphereTransmittanceReference(a,{0,2,0},state.sunDirection);
    for(u32 i=0;i<3;++i)CHECK(double(lights[0].color[i])==doctest::Approx(expected[i]).epsilon(0.001));
    CHECK(lights[0].direction[1]<0);CHECK(lights[0].intensity==1);
}
TEST_CASE("F14 procedural star field is finite deterministic and rotated by the shared clock") {
    const auto ray=glm::normalize(glm::dvec3(0.3,0.8,-0.2));const double a=proceduralStarRadiance(ray,0,0.001);
    CHECK(std::isfinite(a));CHECK(a>=0);CHECK(a==proceduralStarRadiance(ray,0,0.001));
    CHECK(proceduralStarRadiance(ray,6.283185307179586,0.001)==doctest::Approx(a));
}
