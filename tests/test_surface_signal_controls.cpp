#include "renderer/denoise_settings.h"
#include "renderer/reflection_settings.h"
#include "renderer/reservoir.h"
#include <doctest/doctest.h>
#include <array>
#include <cmath>
#include <limits>
using namespace phosphor;
namespace {
GPUDISurface surface(){GPUDISurface s{};s.valid=1;s.depth=3;s.geometricNormal[2]=s.shadingNormal[2]=s.viewDirection[2]=1;s.roughness=.2f;s.instanceSlot=8;s.instanceGeneration=3;s.materialRevision=2;return s;}
}
TEST_CASE("F13 controls foreign view is an actual history domain mismatch"){
    const auto p=denoiseParameters({},2,2,DENOISE_SIGNAL_DI,1,7,9,false);const auto s=surface();
    const auto correct=denoiseTemporal(s,glm::vec3(2),{},p,{},{});REQUIRE(correct.valid);REQUIRE(denoiseCompatible(s,correct,p));
    auto foreign=correct;foreign.viewID=0;CHECK(foreign.flags==0);CHECK(foreign.color[0]==correct.color[0]);CHECK_FALSE(denoiseCompatible(s,foreign,p));
    // Corrupting identity leaves plausible RGB: a color-only checker cannot
    // distinguish it, while the independent domain invariant must fail.
    CHECK(std::isfinite(foreign.color[0]));CHECK(foreign.viewID!=p.viewID);
}
TEST_CASE("F13 controls malformed motion counts actual receiver faults and rejects reprojection"){
    std::array<GPUDISurface,8> receivers{};std::array<glm::vec2,8> motion{};u32 expected=0;
    for(u32 i=0;i<8;++i){receivers[i]=surface();receivers[i].valid=i%2;expected+=receivers[i].valid;
        if(receivers[i].valid)motion[i].x=std::numeric_limits<float>::quiet_NaN();}
    u32 actual=0,rejected=0;for(u32 i=0;i<8;++i){const bool nonfinite=!std::isfinite(motion[i].x)||!std::isfinite(motion[i].y);
        actual+=receivers[i].valid&&nonfinite;if(receivers[i].valid)rejected+=di::historyPixel(i,0,motion[i].x,motion[i].y,8,1)==~0u;}
    CHECK(expected==4);CHECK(actual==expected);CHECK(rejected==expected);
    CHECK(di::historyPixel(3,0,0,0,8,1)==3); // independent uncorrupted control
}
TEST_CASE("F13 controls malformed normals remain distinguishable from background and plausible black"){
    const auto p=denoiseParameters({},8,1,DENOISE_SIGNAL_SPECULAR,0,1,1,false);u32 expected=0,transportRejected=0,historyFaults=0;
    for(u32 i=0;i<8;++i){auto s=surface();s.valid=i%2;if(s.valid){++expected;s.geometricNormal[0]=s.shadingNormal[0]=std::numeric_limits<float>::quiet_NaN();}
        const auto ray=sampleSpecular(s,{.3,.4});const auto h=denoiseTemporal(s,glm::vec3(0),{},p,{},{});
        transportRejected+=s.valid&&!ray.valid;historyFaults+=h.flags!=0;
        if(!s.valid)CHECK(h.flags==0);}
    CHECK(expected==4);CHECK(transportRejected==expected);CHECK(historyFaults==expected);
}
TEST_CASE("F13 controls finite HDR cannot conceal overflowing temporal moments"){
    const auto p=denoiseParameters({},1,1,DENOISE_SIGNAL_GI,0,1,1,false);const auto s=surface();
    const auto h=denoiseTemporal(s,glm::vec3(1e25f),{},p,{},{});
    CHECK(std::isfinite(h.color[0]));CHECK_FALSE(std::isfinite(h.secondMoment));
    const bool numericFault=!std::isfinite(h.firstMoment)||!std::isfinite(h.secondMoment)||!std::isfinite(h.variance)||h.flags!=0;
    CHECK(numericFault); // never trust a plausible RGB or validity bit alone
    // The always-on state checker must inspect moments/flags, not merely the
    // finite RGB that a final composition could cancel or sanitize.
}
