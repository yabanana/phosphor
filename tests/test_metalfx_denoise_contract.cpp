#include <doctest/doctest.h>
#include "renderer/metalfx_denoise_contract.h"
#include <limits>
using namespace phosphor;
using namespace phosphor::metalfx_denoise;
TEST_CASE("F13 MetalFX channel contract rejects plausible but wrong encodings") {
    Semantics s;REQUIRE(validSemantics(s));
    s.normalEncoding=NormalEncoding::Unorm;CHECK_FALSE(validSemantics(s));
    s=Semantics{};s.normalSpace=NormalSpace::View;CHECK_FALSE(validSemantics(s));
    s=Semantics{};s.motionUnits=MotionUnits::NormalizedUV;CHECK_FALSE(validSemantics(s));
    s=Semantics{};s.motionDirection=MotionDirection::PreviousToCurrent;CHECK_FALSE(validSemantics(s));
    s=Semantics{};s.motionIncludesJitter=true;CHECK_FALSE(validSemantics(s));
    s=Semantics{};s.roughness=Roughness::SquaredAlpha;CHECK_FALSE(validSemantics(s));
    s=Semantics{};s.color=Color::IsolatedSignal;CHECK_FALSE(validSemantics(s));
    s=Semantics{};s.specularAlbedoIncludesFresnel=false;CHECK_FALSE(validSemantics(s));
}
TEST_CASE("F13 denoised motion scale stays one for input pixel displacements") {
    CHECK(sdkMotionScale()==glm::vec2(1));
    CHECK(sdkJitter({0.25f,-0.375f})==glm::vec2(0.25f,-0.375f));
    const glm::vec2 current(25,40),previous(15,30);
    CHECK(current+(previous-current)*sdkMotionScale()==previous);
    CHECK_FALSE(current-(previous-current)*sdkMotionScale()==previous); // negative wrong sign
}
TEST_CASE("F13 denoised descriptor uses exact active extent and stale future is rejected") {
    REQUIRE(validExtent({640,360,1920,1080},1,3));
    CHECK_FALSE(validExtent({0,360,1920,1080},1,3));
    CHECK_FALSE(validExtent({320,180,1920,1080},1,3));
    CHECK_FALSE(validExtent({640,360,1920,1080},2,2));
    RequestKey old{{640,360,1920,1080},4,1},same=old;
    REQUIRE(acceptsResult(old,same));
    same.extent.inputWidth=648;CHECK_FALSE(acceptsResult(old,same));
    same=old;++same.pipelineGeneration;CHECK_FALSE(acceptsResult(old,same));
    same=old;same.flags^=2;CHECK_FALSE(acceptsResult(old,same));
}
TEST_CASE("F13 channel split preserves signed world normal and linear roughness") {
    GuideSample s;s.normal={-1,0,0};s.roughness=0.5f;s.motion={-10,-10};
    REQUIRE(validateSample(s)==0); // signed normal, no remap to [0,1]
    s.normal={0.5f,0.5f,1};CHECK(validateSample(s)&NormalError);
    s.normal={0,0,0};s.depth=0;CHECK_FALSE(validateSample(s)&NormalError); // deterministic far background
    s.depth=1;CHECK(validateSample(s)&NormalError);
    s=GuideSample{};s.color={MaximumHalf+1,0,0};CHECK(validateSample(s)&ColorError);
    s=GuideSample{};s.specularAlbedo={1.5f,0,0};CHECK(validateSample(s)&AlbedoError);
    s=GuideSample{};s.hitDistance=-1;CHECK(validateSample(s)&HitError);
    s=GuideSample{};s.reactive=2;CHECK(validateSample(s)&MaskError);
    s=GuideSample{};s.motion.x=std::numeric_limits<float>::quiet_NaN();CHECK(validateSample(s)&MotionError);
    CHECK(format(Channel::Depth)==rg::Format::Depth32Float);
    CHECK(format(Channel::Roughness)==rg::Format::R16Float);
    CHECK(format(Channel::Motion)==rg::Format::RG32Float);
}
TEST_CASE("F13 explicit pre-exposure scaling restores physical wide HDR units") {
    const glm::vec3 physical(368640,128,64);constexpr float scale=1.f/64.f;
    const auto packed=scaleInputRadiance(physical,scale);
    CHECK(packed==glm::vec3(5760,2,1));
    CHECK(restorePreExposedRadiance(packed,scale)==physical);
    GuideSample s;s.color=physical;
    CHECK(validateSample(s)&ColorError);
    CHECK_FALSE(validateSample(s,0.002f,scale)&ColorError);
    s.color.x=MaximumHalf*128.f;CHECK(validateSample(s,0.002f,scale)&ColorError);
    CHECK(format(Channel::RestoredOutput)==rg::Format::RGBA32Float);
    // Closed-form pack/restore identity is NOT evidence of SDK output units.
}
