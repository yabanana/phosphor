#include <doctest/doctest.h>
#include "renderer/metalfx_fixture_oracle.h"
#include <array>
#include <limits>
using namespace phosphor;
TEST_CASE("F13 SDK channels invalidate only known source wraps and existing reset phases") {
    constexpr u64 sourceEpoch=17;
    for(u64 frame=0;frame<192;++frame) {
        const auto p=fxFixtureHistoryPolicy(FX_FIXTURE_CHANNELS,frame,128,sourceEpoch,false);
        const bool wrap=frame>0&&frame%64==0;
        CHECK(p.channelsWrap==wrap);CHECK(p.reset==(frame%48==0||wrap));
        CHECK(p.signalEpoch==sourceEpoch+frame/48+frame/64+1);
        CHECK(p.steady==(frame%48>=8&&frame%64>=8));
        if(frame>0){const auto previous=fxFixtureHistoryPolicy(FX_FIXTURE_CHANNELS,frame-1,128,sourceEpoch,false);
            CHECK((p.signalEpoch!=previous.signalEpoch)==p.reset);}
    }
    const auto before=fxFixtureHistoryPolicy(FX_FIXTURE_CHANNELS,63,128,sourceEpoch,false);
    const auto wrap=fxFixtureHistoryPolicy(FX_FIXTURE_CHANNELS,64,128,sourceEpoch,false);
    const auto after=fxFixtureHistoryPolicy(FX_FIXTURE_CHANNELS,65,128,sourceEpoch,false);
    CHECK_FALSE(before.reset);CHECK(wrap.reset);CHECK_FALSE(after.reset);
    CHECK(wrap.signalEpoch==before.signalEpoch+1);CHECK(after.signalEpoch==wrap.signalEpoch);
    for(u32 scenario:{FX_FIXTURE_CONSTANT,FX_FIXTURE_IMPULSE,FX_FIXTURE_LIFECYCLE,FX_FIXTURE_WIDE_HDR}){
        const auto p=fxFixtureHistoryPolicy(scenario,64,128,sourceEpoch,false);
        CHECK_FALSE(p.channelsWrap);CHECK_FALSE(p.reset);CHECK(p.signalEpoch==sourceEpoch+64/48+1);}
    CHECK(fxFixtureHistoryPolicy(FX_FIXTURE_CHANNELS,65,128,sourceEpoch,true).reset);
    CHECK_THROWS(fxFixtureHistoryPolicy(FX_FIXTURE_CHANNELS,0,0,sourceEpoch,false));
    CHECK_THROWS(fxFixtureHistoryPolicy(FX_FIXTURE_CHANNELS,0,128,std::numeric_limits<u64>::max(),false));
}
TEST_CASE("F13 SDK constant oracle distinguishes actual output-unit hypotheses") {
    const std::array<float,3> physical{.5f,.25f,.125f},scaled{.0078125f,.00390625f,.001953125f};
    const auto preserving=compareFXConstant(scaled,physical,physical,1.f/64);
    CHECK(preserving.preExposed);CHECK_FALSE(preserving.physical);CHECK(preserving.restored);
    const std::array<float,3> incorrectRestore{32,16,8};
    const auto unscaled=compareFXConstant(physical,incorrectRestore,physical,1.f/64);
    CHECK_FALSE(unscaled.preExposed);CHECK(unscaled.physical);CHECK_FALSE(unscaled.restored);
    auto poison=scaled;poison[0]=std::numeric_limits<float>::infinity();
    CHECK_FALSE(compareFXConstant(poison,physical,physical,1.f/64).finite);
}
TEST_CASE("F13 impulse metamorphic oracle does not require identity or unit energy") {
    // Independent authored blurred impulse: sum1.25, not the source energy1.
    const std::array<float,5> blurred{.1f,.2f,.65f,.2f,.1f},scaled{.0015625f,.003125f,.01015625f,.003125f,.0015625f};
    const auto r=compareFXScaledPair(blurred,scaled,1.f/64);
    CHECK(r.passed);CHECK(r.normalizedGain==doctest::Approx(1));CHECK(r.supportDisagreement==0);
    auto shifted=scaled;shifted[0]=0;shifted[4]+=.0015625f;
    CHECK_FALSE(compareFXScaledPair(blurred,shifted,1.f/64).passed);
    const std::array<float,5> black{};CHECK_FALSE(compareFXScaledPair(black,black,1.f/64).passed);
}
