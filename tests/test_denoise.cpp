#include "renderer/denoise_settings.h"
#include <doctest/doctest.h>
#include <cmath>
#include <limits>
using namespace phosphor;
namespace {
GPUDISurface receiver(){GPUDISurface s{};s.valid=1;s.depth=3;s.geometricNormal[2]=s.shadingNormal[2]=1;s.instanceSlot=7;s.instanceGeneration=3;s.materialRevision=5;s.roughness=.2f;return s;}
GPUSpecularSample spec(){GPUSpecularSample s{};s.flags=SPECULAR_SAMPLE_VALID;s.path=REFLECTION_PATH_RT;s.secondarySlot=9;s.secondaryGeneration=4;s.hitDistance=5;return s;}
}
TEST_CASE("F13 denoise independent signal view epoch and receiver rejection controls"){
    DenoiseSettings settings;settings.clampHistory=false;const auto p=denoiseParameters(settings,4,4,DENOISE_SIGNAL_DI,1,2,3,false);const auto s=receiver();
    auto h=denoiseTemporal(s,glm::vec3(2),{},p,{},{});REQUIRE(h.valid);CHECK(h.length==1);CHECK(denoiseCompatible(s,h,p));
    auto bad=p;++bad.viewID;CHECK_FALSE(denoiseCompatible(s,h,bad));bad=p;++bad.historyEpoch;CHECK_FALSE(denoiseCompatible(s,h,bad));
    bad=p;++bad.signal;CHECK_FALSE(denoiseCompatible(s,h,bad));bad=p;++bad.signalRevision;CHECK_FALSE(denoiseCompatible(s,h,bad));bad=p;bad.flags|=DENOISE_RESET;CHECK_FALSE(denoiseCompatible(s,h,bad));
    auto moved=s;++moved.instanceGeneration;CHECK_FALSE(denoiseCompatible(moved,h,p));moved=s;++moved.materialRevision;CHECK_FALSE(denoiseCompatible(moved,h,p));
    moved=s;moved.depth+=1;CHECK_FALSE(denoiseCompatible(moved,h,p));moved=s;moved.geometricNormal[2]=-1;CHECK_FALSE(denoiseCompatible(moved,h,p));
    moved=s;moved.position[2]=.5f;CHECK_FALSE(denoiseCompatible(moved,h,p));
    h.color[0]=std::numeric_limits<float>::quiet_NaN();CHECK_FALSE(denoiseCompatible(s,h,p));
}
TEST_CASE("F13 denoise moments raw signal remains unchanged and reset recovers immediately"){
    DenoiseSettings settings;settings.clampHistory=false;settings.temporalAlpha=.01f;settings.momentsAlpha=.01f;
    const auto p=denoiseParameters(settings,4,4,DENOISE_SIGNAL_GI,0,1,1,false);const auto s=receiver();
    const glm::vec3 first(2),second(4);const auto a=denoiseTemporal(s,first,{},p,{},{}),b=denoiseTemporal(s,second,a,p,{},{});
    CHECK(b.length==2);CHECK(b.color[0]==doctest::Approx(3));CHECK(b.firstMoment==doctest::Approx(3));CHECK(b.secondMoment==doctest::Approx(10));CHECK(b.variance==doctest::Approx(1));
    CHECK(first==glm::vec3(2));CHECK(second==glm::vec3(4));auto cut=p;cut.flags|=DENOISE_RESET;
    const auto recovered=denoiseTemporal(s,glm::vec3(0),b,cut,{},{});CHECK(recovered.length==1);CHECK(recovered.color[0]==0);
    GPUDenoiseHistory history=a;for(u32 i=0;i<100;++i)history=denoiseTemporal(s,first,history,p,{},{});CHECK(history.length==settings.maxHistory);CHECK(history.color[0]==doctest::Approx(2));
    const auto startup=denoiseTemporal(s,first,{},p,{},{},{},4);CHECK(startup.variance==doctest::Approx(4)); // short history uses local variance
}
TEST_CASE("F13 specular denoise rejects hit path distance and secondary reincarnation"){
    const auto p=denoiseParameters({},4,4,DENOISE_SIGNAL_SPECULAR,0,1,1,false);const auto s=receiver();const auto source=spec();
    const auto h=denoiseTemporal(s,glm::vec3(1),{},p,{}, {},source);REQUIRE(denoiseCompatible(s,h,p,source));
    auto changed=source;changed.path=REFLECTION_PATH_ENVIRONMENT;changed.hitDistance=0;changed.secondarySlot=~0u;CHECK_FALSE(denoiseCompatible(s,h,p,changed));
    changed=source;changed.hitDistance=20;CHECK_FALSE(denoiseCompatible(s,h,p,changed));changed=source;++changed.secondaryGeneration;CHECK_FALSE(denoiseCompatible(s,h,p,changed));
    auto rough=s;rough.roughness=.8f;CHECK_FALSE(denoiseCompatible(rough,h,p,source));
    auto normal=s;normal.shadingNormal[2]=-1;CHECK_FALSE(denoiseCompatible(normal,h,p,source));
}
TEST_CASE("F13 denoise clamp touches only history and AO remains bounded"){
    DenoiseSettings settings;const auto p=denoiseParameters(settings,4,4,DENOISE_SIGNAL_AO,0,1,1,false);const auto s=receiver();
    auto old=denoiseTemporal(s,glm::vec3(1),{},p,{},{});const glm::vec3 raw(.1f);
    const auto clipped=denoiseTemporal(s,raw,old,p,glm::vec3(.08f),glm::vec3(.12f));CHECK(clipped.color[0]<=.12f);CHECK(raw==glm::vec3(.1f));
    CHECK(denoiseTemporal(s,glm::vec3(7),{},p,{},{}).color[0]==1);
    auto invalid=s;invalid.valid=0;CHECK_FALSE(denoiseTemporal(invalid,raw,old,p,{},{ }).valid);
    settings.maxHistory=0;CHECK_FALSE(validDenoiseSettings(settings));
}
TEST_CASE("F13 atrous edge weights preserve identity on constant signals and stop normal edges"){
    const auto p=denoiseParameters({},4,4,DENOISE_SIGNAL_DI,0,1,1,false);auto a=receiver(),b=a;
    CHECK(denoiseSpatialWeight(a,b,2,2,1,p)==doctest::Approx(1));b.geometricNormal[2]=-1;CHECK(denoiseSpatialWeight(a,b,2,2,1,p)==0);
    b=a;b.depth=30;CHECK(denoiseSpatialWeight(a,b,2,2,1,p)<.001f);
    CHECK(denoiseSpatialWeight(a,a,2,200,1,p)<.001f);
}
