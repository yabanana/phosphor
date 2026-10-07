#include <doctest/doctest.h>
#include "renderer/emissive_domain.h"
#include "renderer/reservoir.h"
#include "renderer/history_registry.h"
#include <array>
#include <cmath>
using namespace phosphor;
namespace {
std::array<float,16> identity(){std::array<float,16> m{};m[0]=m[5]=m[10]=m[15]=1;return m;}
GPUEmissiveSurface triangle(){GPUEmissiveSurface e{};e.valid=1;e.instanceGeneration=1;e.p1[0]=1;e.p2[1]=1;return e;}
}
TEST_CASE("F11 emissive area oracle and domain reject uniform scale and area-changing shear") {
    auto original=identity(),scaled=identity(),shear=identity();auto e=triangle();
    scaled[0]=scaled[5]=scaled[10]=2;shear[6]=1;
    CHECK(di::emitterWorldArea(e,original.data())==doctest::Approx(0.5));
    CHECK(di::emitterWorldArea(e,scaled.data())==doctest::Approx(2));
    CHECK(di::emitterWorldArea(e,shear.data())==doctest::Approx(std::sqrt(2.0)*0.5));
    CHECK_FALSE(di::sameEmitterAreaDomain(original.data(),scaled.data()));
    CHECK_FALSE(di::sameEmitterAreaDomain(original.data(),shear.data()));
    shear=identity();shear[4]=0.5f; // area-preserving XY shear still conservatively rejected
    CHECK(di::emitterWorldArea(e,shear.data())==doctest::Approx(0.5));
    CHECK_FALSE(di::sameEmitterAreaDomain(original.data(),shear.data()));
}
TEST_CASE("F11 rigid emitter mapping keeps domain while scale rejects previous reservoir epoch") {
    auto original=identity(),rigid=identity(),scaled=identity();auto e=triangle();
    rigid[0]=0;rigid[2]=-1;rigid[8]=1;rigid[10]=0;rigid[12]=11;rigid[14]=-8;
    CHECK(di::sameEmitterAreaDomain(original.data(),rigid.data()));
    di::EmissiveDomainTracker tracker;REQUIRE(tracker.update(std::span(&e,1),original));
    CHECK_FALSE(tracker.update(std::span(&e,1),rigid));
    scaled[0]=scaled[5]=scaled[10]=2;REQUIRE(tracker.update(std::span(&e,1),scaled));
    HistoryRegistry history;auto first=history.begin(0,{1,1,1,1},1);history.write(0,1,original.data());
    auto reset=history.begin(0,{1,1,1,1},2);CHECK(reset.reset);CHECK(reset.generation!=first.generation);
    GPUDIReservoir old{};old.valid=1;old.M=1;old.target=1;old.normalization=0.5;old.weightSum=0.5;
    old.viewID=0;old.historyEpoch=first.generation;old.lightID=7;old.lightGeneration=1;old.lightRevision=1;
    GPUDIParams p{};p.viewID=0;p.historyEpoch=reset.generation;p.lightRevision=1;p.maxHistoryAge=16;
    GPUSampledLight light{};light.id=7;light.generation=1;
    CHECK_FALSE(di::reusable(old,p,light));
    // Negative oracle: reusing old W on a 4x area domain returns only 1/4 of
    // its correct constant-integrand integral. A fresh proposal has W=Anew.
    const double wrong=old.normalization,correct=di::emitterWorldArea(e,scaled.data());
    CHECK(wrong/correct==doctest::Approx(0.25));CHECK(wrong!=doctest::Approx(correct));
}
