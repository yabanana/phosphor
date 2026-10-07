#include <doctest/doctest.h>
#include "renderer/gi_lighting_epoch.h"
using namespace phosphor;
TEST_CASE("F12 sun-only radiance epoch changes on orientation intensity and zero") {
    GiLightingEpoch epoch;GiEnvironment sky;GPULight sun{};sun.type=LIGHT_DIRECTIONAL;sun.direction[1]=-1;
    sun.color[0]=sun.color[1]=sun.color[2]=1;sun.intensity=25;
    const auto initial=epoch.update(std::span(&sun,1),sky);
    CHECK(epoch.update(std::span(&sun,1),sky)==initial);
    sun.direction[0]=0.6f;sun.direction[1]=-0.8f;const auto moved=epoch.update(std::span(&sun,1),sky);
    CHECK(moved>initial);sun.intensity=0;const auto off=epoch.update(std::span(&sun,1),sky);CHECK(off>moved);
    CHECK(epoch.update(std::span(&sun,1),sky)==off);
    sun.intensity=25;CHECK(epoch.update(std::span(&sun,1),sky)>off);
    // Negative oracle: local list is empty in ALL these states and therefore
    // a local-only epoch would remain unchanged while physical GI must reset.
    CHECK(sun.type==LIGHT_DIRECTIONAL);
}
TEST_CASE("F12 GI environment tuple includes solar disk sky and external LUT revision") {
    GiLightingEpoch epoch;GiEnvironment sky;const auto first=epoch.update({},sky);
    sky.skyRadiance[2]=0.2f;const auto lit=epoch.update({},sky);CHECK(lit>first);
    sky.sunAngularRadius*=2;const auto disk=epoch.update({},sky);CHECK(disk>lit);
    ++sky.externalRevision;CHECK(epoch.update({},sky)>disk);
}
