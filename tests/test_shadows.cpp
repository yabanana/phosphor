#include <doctest/doctest.h>
#include "renderer/shadow_settings.h"
#include "renderer/shadow_math.h"
#include "renderer/cull_math.h"
#include <cmath>
#include <limits>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

using namespace phosphor;
namespace {
ShadowCamera cameraAt(glm::vec3 position={0,0,0}) {
    const float f=1.0f/std::tan(0.5f);
    glm::mat4 projection(0);
    projection[0][0]=f; projection[1][1]=f;
    projection[2][3]=-1; projection[3][2]=0.1f;
    ShadowCamera c; c.position=position;
    c.inverseViewProjection=glm::inverse(projection*glm::lookAt(position,position+glm::vec3(0,0,-1),glm::vec3(0,1,0)));
    return c;
}
GPUShadowParams historyParams() {
    GPUShadowParams p{};
    p.flags=SHADOW_FLAG_HISTORY_VALID|SHADOW_FLAG_LIGHT_VALID;
    p.viewID=2; p.lightID=8; p.lightRevision=3; p.sceneRevision=4;
    p.temporalPositionThreshold=0.02f; p.temporalNormalThreshold=0.95f;
    return p;
}
GPUShadowSurface historySurface() {
    GPUShadowSurface s{};
    s.slot=4; s.generation=11; s.valid=1; s.position[0]=3; s.geometricNormal[1]=1;
    return s;
}
GPUShadowHistory historyOf(const GPUShadowParams& p,const GPUShadowSurface& s) {
    GPUShadowHistory h{}; h.slot=s.slot; h.generation=s.generation; h.valid=1;
    h.viewID=p.viewID; h.lightID=p.lightID; h.lightRevision=p.lightRevision; h.sceneRevision=p.sceneRevision;
    for(u32 i=0;i<3;++i) { h.position[i]=s.position[i]; h.geometricNormal[i]=s.geometricNormal[i]; }
    return h;
}
}
TEST_CASE("F10 shadow settings are opt-in and reject nonfinite or unbounded work") {
    ShadowSettings s;
    CHECK(s.mode==ShadowTechnique::Off); CHECK_FALSE(s.contact); CHECK_FALSE(s.staticCache);
    CHECK_NOTHROW(validateShadowSettings(s));
    SUBCASE("bad bias") { s.depthBiasWorld=-1; }
    SUBCASE("infinite bias") { s.normalBiasWorld=std::numeric_limits<float>::infinity(); }
    SUBCASE("bad samples") { s.blockerSamples=65; }
    SUBCASE("zero resolution") { s.mapResolution=0; }
    SUBCASE("nan angular radius") { s.sunAngularRadius=std::numeric_limits<float>::quiet_NaN(); }
    CHECK_THROWS(validateShadowSettings(s));
}
TEST_CASE("F10 four splits preserve endpoints and monotonicity") {
    for(float lambda:{0.0f,0.6f,1.0f}) {
        const auto s=shadowCascadeSplits(0.1f,120.0f,lambda);
        CHECK(s[0]==0.1f); CHECK(s[4]==120.0f);
        for(u32 i=0;i<4;++i) CHECK(s[i+1]>s[i]);
    }
    CHECK_THROWS(shadowCascadeSplits(0,100,0.5f));
    CHECK_THROWS(shadowCascadeSplits(1,1,0.5f));
}
TEST_CASE("F10 stabilized CSM cover all receiver corners including near overlap") {
    const auto camera=cameraAt();
    ShadowSettings s; s.mapResolution=256;
    const auto cascades=makeShadowCascades(camera,{0.3f,1.0f,0.2f},s);
    for(u32 c=0;c<4;++c) {
        const auto& cascade=cascades[c]; const glm::mat4 m=glm::make_mat4(cascade.viewProjection);
        for(u32 i=0;i<8;++i) {
            const glm::vec4 h=camera.inverseViewProjection*glm::vec4(i&1u?1:-1,i&2u?1:-1,1,1);
            const glm::vec3 ray=glm::vec3(h)/h.w;
            const float near=c==0?cascade.splitNear:cascade.splitNear-(cascade.splitNear-cascades[c-1].splitNear)*0.1f;
            const float d=i&4u?cascade.splitFar:near;
            const glm::vec3 p=ray*(d/-ray.z);
            const glm::vec4 clip=m*glm::vec4(p,1);
            CHECK(std::abs(clip.x)<=1.00001f); CHECK(std::abs(clip.y)<=1.00001f);
            CHECK(clip.z>=0); CHECK(clip.z<=1);
        }
        CHECK(shadowCascadeIndex((cascade.splitNear+cascade.splitFar)*0.5f,cascades)==c);
    }
    CHECK(shadowCascadeIndex(1000,cascades)==~0u);
}
TEST_CASE("F10 texel snapping does not follow subtexel camera translation") {
    ShadowSettings s; s.mapResolution=256;
    const auto original=makeShadowCascades(cameraAt(),{0,1,0},s);
    const auto shifted=makeShadowCascades(cameraAt({original[0].texelWorld*0.01f,0,0}),{0,1,0},s);
    CHECK(original[0].viewProjection[12]==doctest::Approx(shifted[0].viewProjection[12]).epsilon(1e-6));
    CHECK(original[0].viewProjection[13]==doctest::Approx(shifted[0].viewProjection[13]).epsilon(1e-6));
}
TEST_CASE("F10 caster culling keeps upstream off-camera casters") {
    ShadowSettings s; s.casterReach=50;
    const ShadowBounds behindCamera{{-0.5f,-0.5f,2},{0.5f,0.5f,3}};
    const auto cascades=makeShadowCascades(cameraAt(),{0,0,1},s,std::span(&behindCamera,1));
    // z>0 lies behind the camera looking -Z, but is on the light ray to the
    // receiver. A camera-visible list would incorrectly omit this caster.
    CHECK(shadowCasterIntersects(cascades[0],0,0,2.5f,0.5f));
    CHECK_FALSE(shadowCasterIntersects(cascades[0],100000,0,2.5f,0.5f));
    const glm::mat4 m=glm::make_mat4(cascades[0].viewProjection);
    const glm::vec4 outside=m*glm::vec4(100000,0,2.5f,1);
    CHECK((std::abs(outside.x)>1 || std::abs(outside.y)>1));
}
TEST_CASE("F10 conservative light culling retains a sphere intersecting map border") {
    GPUShadowCascade c{}; c.viewProjection[0]=1; c.viewProjection[5]=1; c.viewProjection[10]=1; c.viewProjection[15]=1;
    CHECK(shadowCasterIntersects(c,1.2f,0,0.5f,0.25f));
    CHECK_FALSE(shadowCasterIntersects(c,1.3f,0,0.5f,0.25f));
    CHECK(shadowCasterIntersects(c,0,0,-0.1f,0.2f));
}
TEST_CASE("F10 reverse-Z bias and solar PCSS units have negative controls") {
    CHECK_FALSE(shadowDepthVisible(0.2f,0.8f,0));
    CHECK(shadowDepthVisible(0.8f,0.2f,0));
    CHECK(shadowDepthVisible(0.2f,0.21f,0.011f));
    const float penumbra=shadowPenumbraWorld(12,2,0.01f);
    CHECK(penumbra==doctest::Approx(10*std::tan(0.01f)));
    CHECK(shadowPenumbraWorld(112,102,0.01f)==doctest::Approx(penumbra));
    CHECK(shadowPenumbraWorld(2,12,0.01f)==0);
    // These wrong implementations must be distinguishable by this fixture.
    CHECK_FALSE(penumbra==doctest::Approx(10*std::tan(0.01f)/2));
    CHECK_FALSE(shadowDepthVisible(0.2f,0.8f,0)==(0.2f<=0.8f));
}
TEST_CASE("F10 solar disk samples have unit length and uniform solid-angle distribution") {
    const float radius=0.04f;
    double mean=0,x=0,y=0;
    for(u32 i=0;i<4096;++i) {
        const float u=(float(i)+0.5f)/4096;
        const float v=float((i*1597u)%4096u)/4096;
        const auto d=shadowSolarDirection({0,0,1},radius,u,v);
        CHECK(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]==doctest::Approx(1).epsilon(1e-5));
        CHECK(d[2]>=std::cos(radius)-1e-6f);
        mean+=d[2]; x+=d[0]; y+=d[1];
    }
    CHECK(mean/4096==doctest::Approx((1+std::cos(radius))/2).epsilon(1e-6));
    CHECK(std::abs(x/4096)<1e-4); CHECK(std::abs(y/4096)<1e-4);
    CHECK_THROWS(shadowSolarDirection({0,0,0},radius,0.5f,0.5f));
}
TEST_CASE("F10 shadow history is isolated by view light incarnation revision and geometry") {
    auto p=historyParams(); auto s=historySurface(); auto h=historyOf(p,s);
    REQUIRE(shadowHistoryMatches(p,s,h));
    SUBCASE("other view") { ++h.viewID; }
    SUBCASE("other light") { ++h.lightID; }
    SUBCASE("light update") { ++h.lightRevision; }
    SUBCASE("caster or alpha-material update") { ++h.sceneRevision; }
    SUBCASE("slot reused") { ++h.generation; }
    SUBCASE("moved receiver without prior pose") { h.position[0]+=0.03f; }
    SUBCASE("normal discontinuity") { h.geometricNormal[1]=-1; }
    SUBCASE("camera cut reset") { p.flags&=~SHADOW_FLAG_HISTORY_VALID; }
    SUBCASE("invalid sky") { s.valid=0; }
    CHECK_FALSE(shadowHistoryMatches(p,s,h));
}
TEST_CASE("F10 cache never reuses stale light caster material or projection revision") {
    ShadowStaticCache cache(1); const ShadowCacheKey key{0,1,0,0};
    ShadowCacheRevision revision{1,2,3,4};
    cache.beginFrame(1,0,1);
    auto decision=cache.request(key,revision); REQUIRE(decision.action==ShadowCacheAction::Update);
    cache.publish(decision.entry,revision,1); cache.read(decision.entry,1);
    cache.beginFrame(2,1,0);
    CHECK(cache.request(key,revision).action==ShadowCacheAction::Cached);
    SUBCASE("light") { ++revision.light; }
    SUBCASE("caster") { ++revision.caster; }
    SUBCASE("material alpha") { ++revision.material; }
    SUBCASE("projection moved") { ++revision.projection; }
    CHECK(cache.request(key,revision).action==ShadowCacheAction::DynamicFallback);
}
TEST_CASE("F10 cache update budget and in-flight ownership are bounded") {
    ShadowStaticCache cache(1); ShadowCacheRevision r{1,1,1,1};
    cache.beginFrame(1,0,1);
    const auto a=cache.request({0,1,0,0},r);
    REQUIRE(a.action==ShadowCacheAction::Update);
    CHECK(cache.request({0,1,0,1},r).action==ShadowCacheAction::DynamicFallback);
    cache.publish(a.entry,r,3); cache.read(a.entry,5);
    cache.beginFrame(2,4,1);
    CHECK(cache.request({0,1,0,1},r).action==ShadowCacheAction::DynamicFallback);
    auto changed=r; ++changed.caster;
    const auto update=cache.request({0,1,0,0},changed);
    REQUIRE(update.action==ShadowCacheAction::Update);
    CHECK(update.requiredCompletion==5);
    CHECK_THROWS(cache.publish(update.entry,r,6)); // stale publication
    CHECK_THROWS(cache.publish(update.entry,changed,4)); // overwrites an active reader
    cache.publish(update.entry,changed,6);
    CHECK_THROWS(cache.clear());
    cache.beginFrame(3,6,1);
    CHECK_NOTHROW(cache.clear());
    const auto fresh=cache.request({0,1,0,1},r);
    CHECK(fresh.action==ShadowCacheAction::Update);
}
TEST_CASE("F10 tile dirty mask preserves unchanged regions and unions moved-caster regions") {
    GPUShadowCascade c{}; c.radius=1; c.texelWorld=0.01f;
    c.viewProjection[0]=1; c.viewProjection[5]=1; c.viewProjection[10]=1; c.viewProjection[15]=1;
    const auto old=shadowDirtyTiles(c,{{-0.8f,0.4f,0},{-0.7f,0.5f,1}});
    const auto moved=shadowDirtyTiles(c,{{0.7f,-0.5f,0},{0.8f,-0.4f,1}});
    CHECK(old!=0); CHECK(moved!=0); CHECK((old&moved)==0);
    CHECK((old|moved)!=~u64(0));
    CHECK(shadowDirtyTiles(c,{{10,10,0},{11,11,1}})==0);
    CHECK(shadowDirtyTiles(c,{{1,1,1},{0,0,0}})==~u64(0));
}
