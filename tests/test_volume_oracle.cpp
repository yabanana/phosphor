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
