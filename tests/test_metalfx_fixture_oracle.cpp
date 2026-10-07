#include <doctest/doctest.h>
#include "renderer/metalfx_fixture_oracle.h"
#include <array>
#include <limits>
using namespace phosphor;
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
