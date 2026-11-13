#include <doctest/doctest.h>
#include "renderer/fog_settings.h"
#include "renderer/fog_gi.h"
#include "renderer/cloud_settings.h"
#include "renderer/volume_noise.h"
#include "renderer/volume_math.h"
#include <algorithm>
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
TEST_CASE("F14 fog temporal source conserves importance-sampled local illumination") {
    const float historyWeight=float(FogSettings{}.historyWeight);
    for(const float probability:{.1f,.25f,.9f})for(const float intensity:{1.f,7.5f}) {
        double mean=0,oldClampedMean=0,mass=0;
        // One luminous light drawn with PMF p and zero-contribution proposals
        // otherwise. Raw X=I/p or0 has mean I, including phase/extinction and
        // albedo. Enumerate the independent 3x3 proposal outcomes exactly.
        for(u32 bits=0;bits<512;++bits) {
            double likelihood=1;
            for(u32 k=0;k<9;++k)likelihood*=bits&(1u<<k)?probability:1-probability;
            const float raw=bits&1u?intensity/probability:0;
            mean+=likelihood*volumeTemporalSource(raw,intensity,historyWeight);
            const float clipped=bits==0?0:bits==511?intensity/probability:intensity;
            oldClampedMean+=likelihood*volumeTemporalSource(raw,clipped,historyWeight);
            mass+=likelihood;
        }
        CHECK(mass==doctest::Approx(1).epsilon(1e-6));
        CHECK(mean==doctest::Approx(intensity).epsilon(1e-6));
        CHECK(std::abs(oldClampedMean-intensity)>.03*intensity);
        if(probability==.1f)CHECK(oldClampedMean<.7*intensity);
    }
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

namespace {
// Midpoint quadrature over the sphere of f(w) for unit w, in double.
template<class F> double fogSphereIntegral(F f, int nTheta = 256, int nPhi = 512) {
    const double pi = 3.14159265358979323846;
    double sum = 0;
    for (int i = 0; i < nTheta; ++i) {
        const double theta = (i + 0.5) * pi / nTheta, st = std::sin(theta), ct = std::cos(theta);
        for (int j = 0; j < nPhi; ++j) {
            const double phi = (j + 0.5) * 2 * pi / nPhi;
            sum += f(glm::dvec3(st * std::cos(phi), st * std::sin(phi), ct)) * st;
        }
    }
    return sum * (pi / nTheta) * (2 * pi / nPhi);
}
double fogHg(double mu, double g) {
    const double pi = 3.14159265358979323846;
    return (1 - g * g) / (4 * pi * std::pow(1 + g * g - 2 * g * mu, 1.5));
}
// Six-orientation estimate and exact HG in-scattering of a radiance field
// L(w), w towards the source, for the view ray d.
template<class L> std::pair<double, double> fogGiCompare(L radiance, glm::dvec3 d, double g) {
    double e[6];
    for (int axis = 0; axis < 6; ++axis) {
        glm::dvec3 n(0);
        n[axis / 2] = (axis & 1) ? -1.0 : 1.0;
        e[axis] = fogSphereIntegral([&](glm::dvec3 w) { return radiance(w) * std::max(0.0, glm::dot(w, n)); });
    }
    const double estimate = fogGiInScatter(float(e[0]), float(e[1]), float(e[2]), float(e[3]), float(e[4]), float(e[5]),
                                           float(d.x), float(d.y), float(d.z), float(g));
    const double exact = fogSphereIntegral([&](glm::dvec3 w) { return radiance(w) * fogHg(glm::dot(w, d), g); });
    return {estimate, exact};
}
} // namespace

TEST_CASE("F14.2 fog GI in-scattering from six DDGI orientations is exact up to order 1") {
    const glm::dvec3 rays[] = {glm::normalize(glm::dvec3(0.3, -0.8, 0.5)), glm::dvec3(0, 0, 1), glm::normalize(glm::dvec3(-1, 0.2, 0.1))};
    for (double g : {0.0, 0.2, 0.6}) {
        for (const glm::dvec3& d : rays) {
            // Constant and order-1 fields: exact (quadrature error only).
            auto constant = fogGiCompare([](glm::dvec3) { return 1.7; }, d, g);
            CHECK(constant.first == doctest::Approx(constant.second).epsilon(1e-4));
            const glm::dvec3 b(0.3, -0.5, 0.2);
            auto linear = fogGiCompare([&](glm::dvec3 w) { return 1.0 + glm::dot(b, w); }, d, g);
            CHECK(linear.first == doctest::Approx(linear.second).epsilon(1e-4));
            // Order 2: the estimate keeps the mean, the exact value adds
            // g^2 * 0.4 * P2(d.a) -- the residual the header states.
            const glm::dvec3 a = glm::normalize(glm::dvec3(1, 2, -0.5));
            auto quadratic = fogGiCompare([&](glm::dvec3 w) { const double x = glm::dot(w, a); return 1.0 + 0.4 * (3 * x * x - 1) / 2; }, d, g);
            const double x = glm::dot(d, a);
            CHECK(quadratic.second - quadratic.first == doctest::Approx(g * g * 0.4 * (3 * x * x - 1) / 2).epsilon(1e-3).scale(1));
        }
    }
    // A narrow sun-like lobe: with forward scattering the order-1 term removes
    // most of the former isotropic estimate's error, averaged over view rays.
    const glm::dvec3 s = glm::normalize(glm::dvec3(0.2, 1, 0.3));
    auto lobe = [&](glm::dvec3 w) { return 5.0 * std::pow(std::max(0.0, glm::dot(w, s)), 8); };
    double newError = 0, oldError = 0;
    for (int k = 0; k < 16; ++k) {
        const double t = (k + 0.5) / 16 * 3.14159265358979323846;
        const glm::dvec3 d(std::sin(t), std::cos(t), 0.0);
        auto withG = fogGiCompare(lobe, d, 0.2);
        auto isotropic = fogGiCompare(lobe, d, 0.0);
        newError += std::abs(withG.first - withG.second);
        oldError += std::abs(isotropic.first - withG.second);
    }
    CHECK(newError < 0.5 * oldError);
}
