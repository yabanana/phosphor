#include <doctest/doctest.h>
#include "renderer/shadow_settings.h"
#include "renderer/shadow_math.h"
#include "renderer/cull_math.h"
#include "renderer/gpu_scene.h"
#include "renderer/visibility_math.h"
#include <cmath>
#include <cstring>
#include <cstddef>
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
TEST_CASE("F15 ticket 24: the scalar shadow history is the 64-byte prefix of the RGB record and matches like it") {
    static_assert(sizeof(GPUShadowHistoryScalar) == 64 && sizeof(GPUShadowHistory) == 80);
    CHECK(offsetof(GPUShadowHistoryScalar, position) == offsetof(GPUShadowHistory, position));
    CHECK(offsetof(GPUShadowHistoryScalar, visibility) == offsetof(GPUShadowHistory, visibility));
    CHECK(offsetof(GPUShadowHistoryScalar, geometricNormal) == offsetof(GPUShadowHistory, geometricNormal));
    CHECK(offsetof(GPUShadowHistoryScalar, secondMoment) == offsetof(GPUShadowHistory, secondMoment));
    CHECK(offsetof(GPUShadowHistoryScalar, slot) == offsetof(GPUShadowHistory, slot));
    CHECK(offsetof(GPUShadowHistoryScalar, valid) == offsetof(GPUShadowHistory, valid));
    const GPUShadowParams p = historyParams();
    const GPUShadowSurface s = historySurface();
    GPUShadowHistory rgb = historyOf(p, s);
    GPUShadowHistoryScalar scalar{};
    std::memcpy(&scalar, &rgb, sizeof(scalar));
    CHECK(shadowHistoryMatches(p, s, rgb));
    CHECK(shadowHistoryMatches(p, s, scalar) == shadowHistoryMatches(p, s, rgb));
    rgb.generation ^= 1u; std::memcpy(&scalar, &rgb, sizeof(scalar));
    CHECK(!shadowHistoryMatches(p, s, scalar));
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
TEST_CASE("F10 complete caster bounds exclude empty reach from the PCSS search volume") {
    ShadowSettings settings;
    const std::array<ShadowBounds,3> bounds{{{{-6,-0.01f,-6},{6,0.01f,6}},
                                           {{-1.61f,7.99f,-1.01f},{-0.79f,8.01f,1.01f}},
                                           {{0.79f,31.99f,-1.01f},{1.61f,32.01f,1.01f}}}};
    const auto camera=cameraAt();
    const auto bounded=makeShadowCascades(camera,{0,1,0},settings,bounds);
    const auto fallback=makeShadowCascades(camera,{0,1,0},settings);
    CHECK(bounded[0].depthMax<33);
    CHECK(fallback[0].depthMax>500);
    // Moving the arbitrary fallback plane must not change a known scene's
    // projection or filter search range. The XY stabilization stays identical.
    settings.casterReach=10000;
    const auto farther=makeShadowCascades(camera,{0,1,0},settings,bounds);
    for(u32 c=0;c<4;++c) {
        CHECK(bounded[c].depthMax==farther[c].depthMax);
        CHECK(bounded[c].depthMin==farther[c].depthMin);
        CHECK(bounded[c].radius==fallback[c].radius);
        CHECK(bounded[c].center[0]==fallback[c].center[0]);
        CHECK(bounded[c].center[2]==fallback[c].center[2]);
        const glm::mat4 m=glm::make_mat4(bounded[c].viewProjection);
        for(const auto& b:bounds)for(u32 k=0;k<8;++k) {
            const glm::vec4 p(k&1u?b.maximum.x:b.minimum.x,k&2u?b.maximum.y:b.minimum.y,
                              k&4u?b.maximum.z:b.minimum.z,1);
            const float depth=(m*p).z;
            CHECK(depth>=0); CHECK(depth<=1);
        }
    }
    const float boundedSearch=bounded[0].depthMax*std::tan(settings.sunAngularRadius);
    const float fallbackSearch=fallback[0].depthMax*std::tan(settings.sunAngularRadius);
    CHECK(boundedSearch<0.16f); CHECK(fallbackSearch>2.3f);
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
TEST_CASE("F10 Bernoulli temporal mean preserves penumbra expectation at the history cap") {
    // Enumerate every 3x3 raw outcome exactly, without RNG or a rendered
    // reference. The centre and eight neighbors have the same true visibility.
    // A steady unbiased history must keep that expectation for either edge.
    for (const float visibility : {0.1f, 0.5f, 0.9f}) {
        double mean=0, clipped=0, mass=0;
        for (u32 bits=0; bits<512; ++bits) {
            double probability=1;
            for (u32 k=0; k<9; ++k)
                probability *= bits&(1u<<k) ? visibility : 1-visibility;
            const float observation=float(bits&1u);
            mean += probability*shadowTemporalMean(visibility,observation,16);
            const float old=bits==0 ? 0.0f : bits==511 ? 1.0f : visibility;
            clipped += probability*shadowTemporalMean(old,observation,16);
            mass += probability;
        }
        CHECK(mass==doctest::Approx(1).epsilon(1e-6));
        CHECK(mean==doctest::Approx(visibility).epsilon(1e-6));
        if (visibility!=0.5f) CHECK(std::abs(clipped-visibility)>0.03);
    }
    CHECK(shadowTemporalMean(0.3f,1.0f,1)==doctest::Approx(1));
    CHECK(shadowTemporalMean(0.1f,0.0f,16)==doctest::Approx(0.09375f));
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
TEST_CASE("F10 indexed and meshlet receivers decode the same world triangle and analytic pixel") {
    GpuScene scene;
    scene.uploadMesh({{-1,-1,-2},{1,-1,-2},{0,1,-2}},{},{},{},{0,1,2});
    const auto handle=scene.uploadMesh({{-2,-1,-3},{2,-1,-4},{2,1,-4},{-2,1,-3}},
                                        {},{},{},{0,1,2,0,2,3});
    const auto& mesh=scene.meshInfos()[handle];
    REQUIRE(mesh.indexOffset>0);REQUIRE(mesh.vertexOffset>0);
    const glm::mat4 vp=glm::inverse(cameraAt().inverseViewProjection);
    for(u32 primitive=0;primitive<mesh.indexCount/3;++primitive){
        std::array<u32,3> indexed{};for(u32 k=0;k<3;++k)indexed[k]=mesh.vertexOffset+scene.indices()[mesh.indexOffset+3*primitive+k];
        std::array<u32,3> cooked{};bool found=false;
        for(u32 mi=0;mi<mesh.meshletCount;++mi){const auto& m=scene.meshlets()[mesh.meshletOffset+mi];
            for(u32 ti=0;ti<m.triangleCount;++ti){std::array<u32,3> candidate{};
                for(u32 k=0;k<3;++k)candidate[k]=scene.meshletVertices()[m.vertexOffset+scene.meshletTriangles()[m.triangleOffset+3*ti+k]];
                if(candidate==indexed){cooked=candidate;found=true;}}}
        REQUIRE(found); // Original primitive_id is not the meshlet-local ID.
        for(float mirror:{1.f,-1.f}){
            const auto transform=glm::scale(glm::mat4(1),glm::vec3(mirror,1.3f,.8f));
            const auto world=[&](u32 index){const auto& v=scene.vertices()[index];return glm::vec3(transform*glm::vec4(v.px,v.py,v.pz,1));};
            std::array<glm::vec3,3> wi{},wm{};for(u32 k=0;k<3;++k){wi[k]=world(indexed[k]);wm[k]=world(cooked[k]);CHECK(wi[k]==wm[k]);}
            for(const auto extent:{glm::vec2(640,360),glm::vec2(480,270),glm::vec2(320,180)}){
                const glm::vec4 center=vp*glm::vec4((wi[0]+wi[1]+wi[2])/3.f,1);
                const float px=std::floor((center.x/center.w*.5f+.5f)*extent.x)+.5f;
                const float py=std::floor((.5f-center.y/center.w*.5f)*extent.y)+.5f;
                const auto reconstruct=[&](const std::array<glm::vec3,3>& w){
                    const auto a=vp*glm::vec4(w[0],1),b=vp*glm::vec4(w[1],1),c=vp*glm::vec4(w[2],1);
                    const auto bary=visibilityBarycentrics(a.x,a.y,a.w,b.x,b.y,b.w,c.x,c.y,c.w,px,py,extent.x,extent.y);
                    REQUIRE(bary.valid);return w[0]*bary.value[0]+w[1]*bary.value[1]+w[2]*bary.value[2];};
                const auto pi=reconstruct(wi),pm=reconstruct(wm);CHECK(pi==pm);
                // Independent double ray/plane intersection, not interpolated
                // depth or a second invocation of the barycentric formula.
                const auto h=glm::dmat4(cameraAt().inverseViewProjection)*glm::dvec4(2*double(px)/extent.x-1,1-2*double(py)/extent.y,1,1);
                const glm::dvec3 ray=glm::dvec3(h)/h.w;
                const auto n=glm::cross(glm::dvec3(wi[1]-wi[0]),glm::dvec3(wi[2]-wi[0]));
                const auto reference=ray*(glm::dot(n,glm::dvec3(wi[0]))/glm::dot(n,ray));
                CHECK(glm::length(glm::dvec3(pi)-reference)<2e-6);
            }
        }
    }
}

// F15.1 (ticket 15): the single-layer transmittance map of a CSM cascade (renderer/shadow_math.h, shared with MSL).
#include "renderer/transmission_math.h"
TEST_CASE("CSM transmittance map: which casters leave the opaque depth maps") {
    const u32 transmissive = MATERIAL_FEATURE_TRANSMISSION << MATERIAL_FEATURE_SHIFT;
    CHECK(shadowCasterTransmissive(transmissive, 0u, 0.0f));
    CHECK(shadowCasterTransmissive(transmissive | (MATERIAL_FEATURE_VOLUME << MATERIAL_FEATURE_SHIFT), 4u, 0.0f));
    CHECK_FALSE(shadowCasterTransmissive(0u, 0u, 0.0f));                                  // opaque
    CHECK_FALSE(shadowCasterTransmissive(MATERIAL_FEATURE_VOLUME << MATERIAL_FEATURE_SHIFT, 0u, 0.0f)); // volume without transmission
    CHECK_FALSE(shadowCasterTransmissive(transmissive, MATERIAL_EXTENSION_INVALID, 0.0f)); // no extension records: nothing to read
    CHECK_FALSE(shadowCasterTransmissive(transmissive, 0u, 0.5f));                         // alpha-masked: an opaque-masked caster (as in RT)
}

TEST_CASE("CSM transmittance map: receiver behind the nearest transmissive depth") {
    // Reverse-Z: larger = nearer the light.  0 = no transmissive caster in the texel.
    CHECK_FALSE(shadowTransmittanceBehind(0.2f, 0.0f, 0.001f));
    CHECK(shadowTransmittanceBehind(0.2f, 0.5f, 0.001f));
    CHECK_FALSE(shadowTransmittanceBehind(0.7f, 0.5f, 0.001f));   // in front of the glass
    CHECK_FALSE(shadowTransmittanceBehind(0.4995f, 0.5f, 0.001f)); // on the glass surface itself: the bias keeps it lit
    CHECK(shadowTransmittanceBehind(0.4985f, 0.5f, 0.001f));
    // A medium cannot amplify: positive sums clamp to transmittance 1.
    CHECK(shadowTransmittanceOfLn(0.7f) == 1.0f);
    CHECK(shadowTransmittanceOfLn(0.0f) == 1.0f);
    CHECK(shadowTransmittanceOfLn(std::log(0.25f)) == doctest::Approx(0.25f).epsilon(1e-6));
}

TEST_CASE("CSM transmittance map: facing of a fragment and the cascade distance origin") {
    // Entry = the surface faces away from the light (a back face in the light's view).  The rasteriser's clockwise winding reports
    // front_facing for such a face on an unmirrored instance; a mirrored instance reverses the world winding.
    CHECK(shadowTransmitEntry(true, false));
    CHECK_FALSE(shadowTransmitEntry(false, false));
    CHECK_FALSE(shadowTransmitEntry(true, true));
    CHECK(shadowTransmitEntry(false, true));
    CHECK(shadowTransmitDistance(0.0f, 80.0f) == 0.0f);   // the far plane of the cascade is the origin
    CHECK(shadowTransmitDistance(1.0f, 80.0f) == 80.0f);  // grows towards the light, like the distance of a ray from a receiver below
}

// A closed absorbing slab seen by the pass as two fragments (entry below, exit above in the light direction): the ln sums of the fragments are
// the ray-traced sums, and the common distance origin cancels so a cascade that is deeper or shallower than the slab gives the same T.
TEST_CASE("CSM transmittance map: the summed ln of a closed slab is independent of the cascade depth origin") {
    const float ior = 1.5f, cosine = std::cos(20.0f * float(M_PI) / 180.0f), thickness = 0.12f;
    const float attenuation[3] = {0.45f, 0.75f, 0.9f}, attenuationDistance = 0.3f;
    for (float range : {5.0f, 80.0f, 400.0f}) {
        // The slab bottom (entry, met first by a ray from below) at distance a, the top (exit) at a + thickness from the cascade far plane.
        const float a = 0.3f * range, tEntry = a, tExit = a + thickness;
        const TransmissionShadowLn entry = transmissionShadowSurfaceLn(cosine, true, true, ior, 1.0f, 1.0f, 1.0f, 1.0f,
                                                                       attenuation[0], attenuation[1], attenuation[2], attenuationDistance, tEntry);
        const TransmissionShadowLn exit = transmissionShadowSurfaceLn(cosine, false, true, ior, 1.0f, 1.0f, 1.0f, 1.0f,
                                                                      attenuation[0], attenuation[1], attenuation[2], attenuationDistance, tExit);
        const double fresnel = double(transmissionShadowLnPass(cosine, ior));
        for (int k = 0; k < 3; ++k) {
            const double sum = double(k == 0 ? entry.r : k == 1 ? entry.g : entry.b) + double(k == 0 ? exit.r : k == 1 ? exit.g : exit.b);
            const double expected = 2.0 * fresnel - double(transmissionShadowSigma(attenuation[k], attenuationDistance)) * thickness;
            CHECK(sum == doctest::Approx(expected).epsilon(1e-3).scale(1.0));
        }
    }
}

TEST_CASE("CSM transmittance map: the entry factor and tint apply once; a thin-walled surface is its own entry") {
    const float ior = 1.5f, cosine = 1.0f;
    const TransmissionShadowLn thin = transmissionShadowSurfaceLn(cosine, false, false, ior, 0.8f, 0.4f, 0.9f, 0.45f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f);
    const double pass = 0.96;
    CHECK(std::exp(thin.r) == doctest::Approx(pass * 0.8 * 0.4).epsilon(1e-5));
    CHECK(std::exp(thin.g) == doctest::Approx(pass * 0.8 * 0.9).epsilon(1e-5));
    CHECK(std::exp(thin.b) == doctest::Approx(pass * 0.8 * 0.45).epsilon(1e-5));
    // A solid's exit carries only the Fresnel pass and the medium term (no factor, no tint): the entry carries them.
    const TransmissionShadowLn solidExit = transmissionShadowSurfaceLn(cosine, false, true, ior, 0.8f, 0.4f, 0.9f, 0.45f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f);
    CHECK(std::exp(solidExit.r) == doctest::Approx(pass).epsilon(1e-5));
    const TransmissionShadowLn solidEntry = transmissionShadowSurfaceLn(cosine, true, true, ior, 0.8f, 0.4f, 0.9f, 0.45f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f);
    CHECK(std::exp(solidEntry.r) == doctest::Approx(pass * 0.8 * 0.4).epsilon(1e-5));
    // Metallic 1 (factor 0) floors at ln(1e-30): finite, T = 0 (the factor path of the ray-traced any-hit).
    const TransmissionShadowLn metal = transmissionShadowSurfaceLn(cosine, false, false, ior, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f);
    CHECK(std::isfinite(metal.r));
    CHECK(std::exp(metal.r) < 1e-29);
}

// F15.1 (ticket 14 closure): a segment that ENDS inside a closed volume.  The signed sum alone keeps +sigma_t t_entry (the exit term is missing);
// the open sigma restores the medium length endDistance - t_entry.
TEST_CASE("RT transmittance: a segment that ends inside a closed volume is attenuated over the open length") {
    const float ior = 1.5f, cosine = 0.8f, attenuation[3] = {0.6f, 0.8f, 0.95f}, attenuationDistance = 0.3f;
    const float tEntry = 0.4f, endDistance = 0.65f;
    const TransmissionShadowLn entry = transmissionShadowSurfaceLn(cosine, true, true, ior, 1.0f, 1.0f, 1.0f, 1.0f,
                                                                   attenuation[0], attenuation[1], attenuation[2], attenuationDistance, tEntry);
    const TransmissionShadowOpen open = transmissionShadowSurfaceOpen(true, true, attenuation[0], attenuation[1], attenuation[2], attenuationDistance);
    const double fresnel = double(transmissionShadowLnPass(cosine, ior));
    const float ln[3] = {entry.r, entry.g, entry.b}, o[3] = {open.r, open.g, open.b};
    for (int k = 0; k < 3; ++k) {
        const double sigma = double(transmissionShadowSigma(attenuation[k], attenuationDistance));
        CHECK(double(o[k]) == doctest::Approx(sigma).epsilon(1e-6));
        // Without the correction the entry term is positive (the pre-closure clamp T <= 1 hid the medium).
        CHECK(double(ln[k]) == doctest::Approx(fresnel + sigma * tEntry).epsilon(1e-4));
        const double corrected = double(ln[k]) + double(transmissionShadowOpenLn(o[k], endDistance));
        CHECK(corrected == doctest::Approx(fresnel - sigma * (endDistance - tEntry)).epsilon(1e-4));
    }
    // A closed pair cancels (net open 0: no correction); a segment that STARTS inside a medium has a negative net (nothing to add).
    const TransmissionShadowOpen exit = transmissionShadowSurfaceOpen(false, true, attenuation[0], attenuation[1], attenuation[2], attenuationDistance);
    CHECK(open.r + exit.r == 0.0f);
    CHECK(transmissionShadowOpenLn(exit.r, endDistance) == 0.0f);
    // A thin-walled or non-solid surface opens nothing; no attenuation (distance infinite) opens nothing.
    const TransmissionShadowOpen thin = transmissionShadowSurfaceOpen(true, false, attenuation[0], attenuation[1], attenuation[2], attenuationDistance);
    CHECK((thin.r == 0.0f && thin.g == 0.0f && thin.b == 0.0f));
    const TransmissionShadowOpen clear = transmissionShadowSurfaceOpen(true, true, attenuation[0], attenuation[1], attenuation[2], 1.0e39f);
    CHECK((clear.r == 0.0f && clear.g == 0.0f && clear.b == 0.0f));
}

TEST_CASE("CSM transmittance map: the divisor is 1, 2 or 4 and keeps a usable map") {
    ShadowSettings settings;
    for (u32 n : {1u, 2u, 4u}) { settings.transmittanceDivisor = n; CHECK_NOTHROW(validateShadowSettings(settings)); }
    settings.transmittanceDivisor = 3; CHECK_THROWS(validateShadowSettings(settings));
    settings.transmittanceDivisor = 0; CHECK_THROWS(validateShadowSettings(settings));
    settings.transmittanceDivisor = 4; settings.mapResolution = 32; CHECK_THROWS(validateShadowSettings(settings)); // 8 texels
}
