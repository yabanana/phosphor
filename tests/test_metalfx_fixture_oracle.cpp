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
TEST_CASE("F13.3 owner criterion D1: absolute floor up to 4x white, SDR above") {
    const double m=.5/(368640.0/64.0); // frozen wide-HDR manual exposure: red 32x, green 0.011x, blue 0.0056x white
    // Measured frozen-fixture profile (closure brief 2.2): red -7%, green +1.46%, blue +5.18%.
    CHECK(fxD1Excess(368640,368640*.93,m)<=1);CHECK(fxD1Excess(128,128*1.0146,m)<=1);CHECK(fxD1Excess(64,64*1.0518,m)<=1);
    CHECK(fxD1Excess(64,64*1.10,m)>1);                       // floor 5e-4 white = 5.76: 10% of 64 exceeds it
    const double white=1/m;
    CHECK(fxD1Excess(white,white*1.005,m)<=1);CHECK(fxD1Excess(white,white*1.02,m)>1); // 1% applies near white
    CHECK(fxD1Excess(4*white,4*white*1.009,m)<=1);CHECK(fxD1Excess(4*white,4*white*1.011,m)>1);
    CHECK(fxD1Excess(4.5*white,9*white,m)>1);               // above 4x: 2/255 after the SDR transform
    CHECK(fxD1Excess(6*white,6*white*1.02,m)<=1);
    CHECK(std::isinf(fxD1Excess(1,std::numeric_limits<double>::quiet_NaN(),m)));
    const std::array<float,3> phys{368640,128,64},out{368640*.93f,128*1.0146f,64*1.0518f},sdk{out[0]/64,out[1]/64,out[2]/64};
    const auto r=compareFXConstant(sdk,out,phys,1.f/64,.01,m);CHECK_FALSE(r.restored);CHECK(r.restoredD1);
    CHECK_FALSE(compareFXConstant(sdk,out,phys,1.f/64).restoredD1); // no exposure scale: D1 not evaluated
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
