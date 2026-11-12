#include <doctest/doctest.h>
#include "renderer/volume_oracle.h"
#include <cmath>
using namespace phosphor;
namespace {
VolumeOracleInput solarInput(){VolumeOracleInput input;input.atmosphere.sunIrradiance[0]=input.atmosphere.sunIrradiance[1]=input.atmosphere.sunIrradiance[2]=25;
    input.atmosphere.sunAngularRadius=0.00465f;input.atmosphere.parameterRevision=4;input.atmosphere.skyRevision=7;return input;}
}
TEST_CASE("F14 sparse oracle derives wide HDR solar storage and orientation failures from actual records") {
    auto input=solarInput();GPUVolumeNumericSample record{};record.kind=VOLUME_NUMERIC_SOLAR;record.producedRevision=7;record.value[3]=1;
    const double expected=25/(3.141592653589793*std::pow(std::sin(double(input.atmosphere.sunAngularRadius)),2));
    CHECK(expected>65504);for(u32 i=0;i<3;++i)record.value[i]=float(expected);
    CHECK(evaluateVolumeOracle(input,std::span(&record,1))[0].passed);
    SUBCASE("half saturation"){for(u32 i=0;i<3;++i)record.value[i]=65504;}
    SUBCASE("wrong toward orientation"){for(u32 i=0;i<3;++i)record.value[i]=0;}
    SUBCASE("foreign produced epoch"){record.producedRevision=6;}
    CHECK_FALSE(evaluateVolumeOracle(input,std::span(&record,1))[0].passed);
}
TEST_CASE("F14 sparse oracle rejects a bright solar disk on the away direction") {
    const auto input=solarInput();GPUVolumeNumericSample record{};record.kind=VOLUME_NUMERIC_SOLAR;record.x=1;record.producedRevision=7;record.value[3]=1;
    REQUIRE(evaluateVolumeOracle(input,std::span(&record,1))[0].passed);record.value[0]=100;
    CHECK_FALSE(evaluateVolumeOracle(input,std::span(&record,1))[0].passed);
}
TEST_CASE("F14 lunar reference divides by phase and projects the illuminated normal") {
    auto input=solarInput();
    auto& p=input.atmosphere;
    p.flags=ATMOSPHERE_ENABLE_MOON;
    p.sunDirection[0]=0.8f;p.sunDirection[1]=-0.6f;p.moonDirection[1]=1;
    p.moonAngularRadius=0.0045f;p.moonPhase=0.25f;
    p.moonIrradiance[0]=0.0025f;p.moonIrradiance[1]=0.003f;p.moonIrradiance[2]=0.005f;
    GPUVolumeNumericSample rows[3]{};
    for(u32 i=0;i<3;++i){
        rows[i].kind=VOLUME_NUMERIC_LUNAR;rows[i].x=i;rows[i].producedRevision=7;rows[i].value[3]=1;
    }
    const double disk=(2.0/3.0)*3.141592653589793*std::pow(std::sin(double(p.moonAngularRadius)),2);
    for(u32 i=0;i<3;++i)rows[0].value[i]=float(p.moonIrradiance[i]*0.6/(0.25*disk));
    const auto reference=evaluateVolumeOracle(input,rows);
    REQUIRE(reference.size()==3);
    for(const auto& row:reference)CHECK(row.passed);
    rows[0].value[0]*=0.25f;
    CHECK_FALSE(evaluateVolumeOracle(input,rows)[0].passed);
    p.moonPhase=0;
    CHECK(evaluateVolumeOracle(input,rows)[0].expected[0]==0);
    p.moonPhase=0.25f;p.sunDirection[1]=0.6f;
    CHECK(evaluateVolumeOracle(input,rows)[0].expected[0]==0);
}
TEST_CASE("F14 stellar reference uses the same-frame ABI rotation and angular footprint") {
    auto input=solarInput();
    auto& p=input.atmosphere;
    p.flags=ATMOSPHERE_ENABLE_STARS;p.outputHeight=360;p.starIntensity=0.0001f;
    GPUVolumeNumericSample row{};row.kind=VOLUME_NUMERIC_STARS;row.x=18;row.producedRevision=7;row.value[3]=1;
    constexpr double pi=3.141592653589793;
    const double azimuth=5*pi/8,horizontal=std::sqrt(1-0.4*0.4);
    const glm::dvec3 direction(horizontal*std::cos(azimuth),0.4,horizontal*std::sin(azimuth));
    const double unrotated=p.starIntensity*proceduralStarRadiance(direction,0,pi/360);
    REQUIRE(unrotated>1e-5);
    for(u32 i=0;i<3;++i)row.value[i]=float(unrotated);
    CHECK(evaluateVolumeOracle(input,std::span(&row,1))[0].passed);
    p.starRotation=0.01f;
    const auto rotated=evaluateVolumeOracle(input,std::span(&row,1))[0];
    CHECK_FALSE(rotated.passed);
    const double expected=p.starIntensity*proceduralStarRadiance(direction,p.starRotation,pi/360);
    CHECK(rotated.expected[0]==doctest::Approx(expected).epsilon(1e-10).scale(expected));
    p.flags=0;
    CHECK(evaluateVolumeOracle(input,std::span(&row,1))[0].expected[0]==0);
}
