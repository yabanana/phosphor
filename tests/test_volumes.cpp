#include <doctest/doctest.h>
#include "renderer/fog_settings.h"
#include "renderer/cloud_settings.h"
#include "renderer/volume_noise.h"
#include "renderer/volume_math.h"
#include <cmath>
#include <limits>
using namespace phosphor;
TEST_CASE("F14 segment integral matches double expm1 including near-vacuum optical depths") {
    for (float distance : {0.0f, 0.001f, 1.0f, 100000.0f}) {
        CHECK(volumeIntegralFactor(0, distance) == distance);
        for (int i = -12; i <= 8; ++i) {
            const float sigma = float(std::pow(10.0, double(i)));
            const double expected = -std::expm1(-double(sigma) * distance) / sigma;
            CHECK(std::abs(double(volumeIntegralFactor(sigma, distance)) - expected) <=
                  2e-6 * std::max(expected, 1e-30));
        }
    }
    for (float tau : {0.099999f, 0.1f, 0.100001f})
        CHECK(volumeIntegralFactor(tau, 1) == doctest::Approx(-std::expm1(-double(tau)) / tau).epsilon(2e-6));
}
TEST_CASE("F14 fog exponential constant-density oracle and front-to-back composition") {
    const auto full=fogHomogeneous(0.01,{0.01,0.02,0.03},100);
    CHECK(full.transmittance==doctest::Approx(std::exp(-1.0)));
    CHECK(full.radiance.x==doctest::Approx(1-std::exp(-1.0)));CHECK(full.radiance.y==doctest::Approx(2*(1-std::exp(-1.0))));
    const auto split=fogComposite(fogHomogeneous(0.01,{0.01,0.02,0.03},30),fogHomogeneous(0.01,{0.01,0.02,0.03},70));
    CHECK(split.transmittance==doctest::Approx(full.transmittance));CHECK(split.radiance.x==doctest::Approx(full.radiance.x));
    const auto vacuum=fogHomogeneous(0,{1,2,3},2);CHECK(vacuum.transmittance==1);CHECK(vacuum.radiance.x==2);
    CHECK_THROWS(fogHomogeneous(-0.1,{1,1,1},1));CHECK_THROWS(fogHomogeneous(1,{1,1,1},-1));
    CHECK_FALSE(full.radiance.x==doctest::Approx(fogHomogeneous(10,{0.01,0.02,0.03},100).radiance.x));
}
TEST_CASE("F14 fog source compositing order cannot be replaced by multiplying masks") {
    const auto red=fogHomogeneous(0.1,{0.1,0,0},10),blue=fogHomogeneous(0.1,{0,0,0.1},10);
    const auto rb=fogComposite(red,blue),br=fogComposite(blue,red);
    CHECK(rb.transmittance==doctest::Approx(br.transmittance));CHECK(rb.radiance.x>br.radiance.x);CHECK(rb.radiance.z<br.radiance.z);
}
TEST_CASE("F14 fog slices density and budgets are world-unit bounded") {
    FogSettings s;CHECK_NOTHROW(validateFog(s));CHECK(fogDensity(s,100)<fogDensity(s,0));
    CHECK(fogSliceDistance(s,0)==s.nearDistance);CHECK(fogSliceDistance(s,1)==doctest::Approx(s.farDistance));
    for(u32 i=0;i<s.gridZ;++i)CHECK(fogSliceDistance(s,double(i+1)/s.gridZ)>fogSliceDistance(s,double(i)/s.gridZ));
    s.gridZ=129;CHECK_THROWS(validateFog(s));s.gridZ=32;s.maxLocalLights=17;CHECK_THROWS(validateFog(s));
}
TEST_CASE("F14 defined periodic noise has deterministic finite unit bounds") {
    for(int i=-16;i<=16;++i){const float x=float(i)*0.371f,y=float(i)*0.217f,z=float(i)*0.131f;
        const float value=volumeValueNoise(x,y,z,123),worley=volumeWorley(x,y,z,123),fbm=volumeFbm(x,y,z,123);
        CHECK(value>=0);CHECK(value<=1);CHECK(worley>=0);CHECK(worley<=1);CHECK(fbm>=0);CHECK(fbm<=1);
        CHECK(value==volumeValueNoise(x,y,z,123));CHECK(volumeValueNoise(x+256,y,z,123)==doctest::Approx(value).epsilon(1e-4));}
}
TEST_CASE("F14 cloud density is actual procedural content with altitude and coverage bounds") {
    CloudSettings s;CHECK_NOTHROW(validateClouds(s));s.coverage=1;
    double sum=0;for(u32 i=0;i<32;++i){const auto point=glm::dvec3(i*130.0,2500,i*97.0);const auto density=cloudDensityReference(s,point,2500,0);
        CHECK(density>=0);CHECK(density<=s.densityScale);sum+=density;}
    CHECK(sum>0);CHECK(cloudDensityReference(s,{0,0,0},0,0)==0);CHECK(cloudDensityReference(s,{0,5000,0},5000,0)==0);
    s.coverage=0;CHECK(cloudDensityReference(s,{0,2500,0},2500,0)==0);
    s.coverage=1.1;CHECK_THROWS(validateClouds(s));
}
TEST_CASE("F14 wind translates density and history with the same metre-per-second clock") {
    CloudSettings s;s.coverage=1;const glm::dvec3 p(700,2500,1300);const double seconds=3;
    CHECK(cloudDensityReference(s,p+s.wind*seconds,2500,seconds)==doctest::Approx(cloudDensityReference(s,p,2500,0)).epsilon(1e-5));
    GPUCloudParams params{};params.flags=VOLUME_HISTORY_VALID;params.viewID=2;params.generation=7;params.positionThreshold=0.001f;
    params.depthRelativeThreshold=0.01f;params.wind[0]=10;params.timeSeconds=2;params.previousTimeSeconds=1;
    GPUCloudHistory history{};history.valid=1;history.samples=1;history.viewID=2;history.generation=7;history.worldPosition[0]=0;history.opaqueDistance=100;
    CHECK(cloudHistoryCompatible(params,history,{10,0,0},100));CHECK_FALSE(cloudHistoryCompatible(params,history,{-10,0,0},100));
    SUBCASE("other view"){++history.viewID;}SUBCASE("clock/volume epoch"){++history.generation;}SUBCASE("new foreground depth"){history.opaqueDistance=10;}
    CHECK_FALSE(cloudHistoryCompatible(params,history,{10,0,0},100));
}

TEST_CASE("F14 cloud centroid correction preserves homogeneous front air and attenuates cloud radiance") {
    // Independent analytical transport through front air, a thin cloud layer,
    // then back air and an opaque scene. Coefficients are m^-1; all distances
    // are metres and RGB values remain linear before exposure.
    const double sigmaAir=0.01,frontDistance=30,backDistance=70,sigmaCloud=0.02,cloudDistance=20;
    const glm::dvec3 airSource(0.01,0.02,0.03),cloudSource(0.03,0.02,0.01),scene(10,5,1);
    const double Ta=std::exp(-sigmaAir*frontDistance),Tb=std::exp(-sigmaAir*backDistance),Tc=std::exp(-sigmaCloud*cloudDistance);
    const glm::dvec3 La=airSource/sigmaAir*(1-Ta),Lb=airSource/sigmaAir*(1-Tb),Lc=cloudSource/sigmaCloud*(1-Tc);
    const glm::dvec3 sceneAtmosphere=La+Ta*(Lb+Tb*scene);
    const glm::dvec3 expected=La+Ta*(Lc+Tc*(Lb+Tb*scene));
    const glm::dvec3 corrected=Tc*sceneAtmosphere+Ta*Lc+La*(1-Tc);
    for(u32 c=0;c<3;++c)CHECK(corrected[c]==doctest::Approx(expected[c]).epsilon(1e-12));
    const glm::dvec3 wrong=Lc+Tc*sceneAtmosphere;
    CHECK(glm::length(wrong-expected)>0.01); // old all-air-behind-cloud formula must fail
}
