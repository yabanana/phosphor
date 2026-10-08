#include "renderer/reflection_settings.h"
#include "renderer/reflection_probe.h"
#include "renderer/reflection_probe_storage.h"
#include "renderer/transform_reference.h"
#include <glm/gtc/type_ptr.hpp>
#include <cstring>
#include <doctest/doctest.h>
#include <glm/gtc/matrix_transform.hpp>
#include <array>
#include <cmath>
#include <limits>
#include <algorithm>
#include <vector>
using namespace phosphor;
namespace {
GPUDISurface shiny(){GPUDISurface s{};s.valid=1;s.geometricNormal[2]=s.shadingNormal[2]=s.viewDirection[2]=1;s.depth=3;s.roughness=1;s.metallic=1;s.albedo[0]=s.albedo[1]=s.albedo[2]=1;return s;}
glm::dvec3 white(glm::dvec3,void*){return glm::dvec3(1);}
glm::vec3 colored(glm::vec3,void*){return {2,3,5};}
bool halfOcclusion(glm::vec3 d,float radius,void*){return radius>=.5f&&d.x>0;}
bool noOcclusion(glm::vec3,float,void*){return false;}
GPUReflectionProbe probe(){GPUReflectionProbe p{};p.enabled=1;p.boxMin[0]=-2;p.boxMin[1]=-1;p.boxMin[2]=-3;p.boxMax[0]=2;p.boxMax[1]=1;p.boxMax[2]=3;p.blendDistance=.5f;p.mipCount=6;p.generation=1;return p;}
}
TEST_CASE("F13 reflection solid-angle reference agrees with independent rough metal integral"){
    const auto s=shiny();const auto reference=reflectionReference(s,32,2048,white);
    // For N=V, roughness=1,F0=1: 2pi*integral0..1[z/(2pi*(z+1))]dz.
    CHECK(reference.x==doctest::Approx(1-std::log(2.0)).epsilon(1e-5));
    glm::dvec3 estimate(0);u32 zeroWeight=0;
    for(u32 y=0;y<128;++y)for(u32 x=0;x<128;++x){const auto d=sampleSpecular(s,{(x+.5)/128,(y+.5)/128});REQUIRE(d.valid);REQUIRE(d.pdf>0);
        estimate+=d.weight;zeroWeight+=glm::dot(glm::dvec3(0,0,1),d.direction)<=0;}
    estimate/=128*128;
    CHECK(estimate.x==doctest::Approx(reference.x).epsilon(1e-4));CHECK(zeroWeight==8192);
    // Resampling rejected NDF proposals would double energy for this case.
    CHECK(estimate.x*2!=doctest::Approx(reference.x));
}
TEST_CASE("F13 reflection roughness selection is continuous and misses remain explicit"){
    ReflectionSettings s;REQUIRE(validReflectionSettings(s));CHECK(reflectionRTWeight(s.rtRoughnessLow,s)==1);CHECK(reflectionRTWeight(s.rtRoughnessHigh,s)==0);
    CHECK(reflectionRTWeight((s.rtRoughnessLow+s.rtRoughnessHigh)*.5f,s)==doctest::Approx(.5));
    s.flags&=~REFLECTION_ENABLE_RT;CHECK(reflectionRTWeight(.01f,s)==0);
    s.rtRoughnessHigh=s.rtRoughnessLow;CHECK_FALSE(validReflectionSettings(s));
    auto surface=shiny();surface.roughness=std::numeric_limits<float>::quiet_NaN();CHECK_FALSE(sampleSpecular(surface,{.3,.4}).valid);
    surface=shiny();CHECK_FALSE(sampleSpecular(surface,{1,.4}).valid);
    GPUSpecularSample miss{};miss.path=REFLECTION_PATH_ENVIRONMENT;miss.secondarySlot=~0u;miss.flags=SPECULAR_SAMPLE_VALID;
    CHECK(miss.hitDistance==0); // environment is not a fake far geometry hit
}
TEST_CASE("F13 reflection energy is invariant under a mirrored surface frame"){
    auto a=shiny();a.roughness=.7f;const auto n=glm::normalize(glm::vec3(.4f,.2f,.9f)),v=glm::normalize(glm::vec3(.1f,.4f,1));
    for(u32 i=0;i<3;++i){a.shadingNormal[i]=n[i];a.viewDirection[i]=v[i];}
    auto b=a;b.shadingNormal[0]*=-1;b.viewDirection[0]*=-1;
    const auto ra=reflectionReference(a,64,256,white),rb=reflectionReference(b,64,256,white);
    CHECK(ra.x==doctest::Approx(rb.x).epsilon(1e-4));
    // Secondary front-face decisions require root's rtMaterialFrontFacing;
    // this frame-invariance oracle never substitutes WORLD facing for it.
}
TEST_CASE("F13 SSR brackets a known plane and rejects background offscreen and invalid depth"){
    constexpr u32 side=64;glm::mat4 vp(0);vp[0][0]=vp[1][1]=1;vp[2][3]=-1;vp[3][2]=.1f;
    const auto inverse=glm::inverse(vp);std::vector<float> depth(side*side,.1f/3);ReflectionSettings settings;settings.maxDistance=10;settings.ssrBinarySteps=12;
    const auto hit=reflectionSSR({0,0,-1},{0,0,-1},vp,inverse,glm::mat4(1),side,side,depth,settings);
    REQUIRE(hit);CHECK(hit->position.z==doctest::Approx(-3).epsilon(1e-5));CHECK(std::abs(hit->distance-2)<.01f);
    std::fill(depth.begin(),depth.end(),0);CHECK_FALSE(reflectionSSR({0,0,-1},{0,0,-1},vp,inverse,glm::mat4(1),side,side,depth,settings));
    std::fill(depth.begin(),depth.end(),.1f/3);CHECK_FALSE(reflectionSSR({0,0,-1},{1,0,0},vp,inverse,glm::mat4(1),side,side,depth,settings));
    CHECK_FALSE(reflectionSSR({0,0,-1},{0,0,0},vp,inverse,glm::mat4(1),side,side,depth,settings));
}
TEST_CASE("F13 AO oracle uses world radius and cosine visibility with no GI double count"){
    AOSettings settings;CHECK(validAOSettings(settings));CHECK(aoReference({0,1,0},1,64,noOcclusion)==1);
    CHECK(aoReference({0,1,0},1,64,halfOcclusion)==doctest::Approx(.5));CHECK(aoReference({0,1,0},.25f,64,halfOcclusion)==1);
    const auto open=aoHorizonVisibility({0,0,1},{0,0,1},{1,0,0},float(3.141592653589793),float(3.141592653589793),512);CHECK(open==1);
    const auto horizon=aoHorizonVisibility({0,0,1},{0,0,1},{1,0,0},float(3.141592653589793/3),float(3.141592653589793/3),1024);
    CHECK(std::abs(horizon-.75f)<.005f);
    const auto darkAO=composeSignalLighting(glm::vec3(2),glm::vec3(3),glm::vec3(5),glm::vec3(7),glm::vec3(11),0,true);
    const auto openAO=composeSignalLighting(glm::vec3(2),glm::vec3(3),glm::vec3(5),glm::vec3(7),glm::vec3(11),1,true);
    CHECK(darkAO==openAO);CHECK(darkAO==glm::vec3(17));
    CHECK(composeSignalLighting(glm::vec3(2),{},glm::vec3(5),glm::vec3(7),glm::vec3(11),.5f,false)==glm::vec3(19.5f));
}
TEST_CASE("F13 static reflection probe parallax cube orientation and constant energy"){
    const auto p=probe();REQUIRE(validReflectionProbe(p));CHECK(reflectionProbeWeight(p,{0,0,0})==1);CHECK(reflectionProbeWeight(p,{2,0,0})==0);
    CHECK_FALSE(reflectionProbeParallax(p,{3,0,0},{0,0,1}));const auto direction=reflectionProbeParallax(p,{1,0,0},{0,0,1});REQUIRE(direction);
    CHECK(direction->x==doctest::Approx(1/std::sqrt(10.0)));CHECK(direction->z==doctest::Approx(3/std::sqrt(10.0)));
    for(u32 face=0;face<6;++face)for(const glm::vec2 uv:{glm::vec2(.2f,.3f),glm::vec2(.7f,.8f),glm::vec2(.5f)}){
        const auto d=reflectionCubeDirection(face,uv);const auto recovered=reflectionCubeCoordinate(d);REQUIRE(recovered);CHECK(recovered->face==face);
        CHECK(recovered->uv.x==doctest::Approx(uv.x));CHECK(recovered->uv.y==doctest::Approx(uv.y));
        const auto vp=reflectionProbeViewProjection(p,face,.1f,100);const glm::vec4 clip=vp*glm::vec4(d*3.f,1);
        const glm::vec2 raster=glm::vec2(clip)/clip.w*glm::vec2(.5f,-.5f)+.5f;
        CHECK(raster.x==doctest::Approx(uv.x).epsilon(1e-5));CHECK(raster.y==doctest::Approx(uv.y).epsilon(1e-5));
    }
    for(float roughness:{0.f,.1f,.5f,1.f}){const auto L=reflectionProbePrefilter({.4f,.3f,.8f},roughness,256,colored);
        CHECK(L.x==doctest::Approx(2).epsilon(1e-5));CHECK(L.y==doctest::Approx(3).epsilon(1e-5));CHECK(L.z==doctest::Approx(5).epsilon(1e-5));}
    for(float roughness:{.04f,.1f,.5f,1.f})for(float metal:{0.f,1.f}){auto s=shiny();s.roughness=roughness;s.metallic=metal;
        const auto Lo=reflectionProbeContribution(s,{2,3,5});CHECK(Lo.x>=0);CHECK(Lo.x<=2);CHECK(Lo.y<=3);CHECK(Lo.z<=5);}
}


TEST_CASE("F13 static probe hierarchy matches independently flattened world geometry") {
    std::array<GPUInstance,2> instances{};std::array<GPUTransformNode,2> nodes{};
    std::array<GPUMotion,2> motions{};const std::array<u32,3> offsets{0,1,1};const std::array<u32,1> children{1};
    const glm::mat4 parent=glm::translate(glm::mat4(1),glm::vec3(100,2,-5));
    const glm::mat4 local=glm::translate(glm::mat4(1),glm::vec3(7,3,4))*glm::scale(glm::mat4(1),glm::vec3(-2,3,1));
    for(auto& i:instances){i.flags=INSTANCE_FLAG_VALID|1u|4u;const glm::mat4 identity(1);std::memcpy(i.modelMatrix,glm::value_ptr(identity),64);}
    std::memcpy(instances[0].modelMatrix,glm::value_ptr(parent),64);std::memcpy(nodes[1].local,glm::value_ptr(local),64);nodes[1].parentSlot=0;nodes[1].depth=1;
    std::vector<u32> selected;reflectionProbeStaticSlots(instances,nodes,{},selected);CHECK(selected==std::vector<u32>{0,1});
    const HierarchyView view{instances,nodes,motions,{},offsets,children};std::vector<float> worlds;referenceWorlds(view,nullptr,worlds);
    // Independent flattened matrix: child translation107,5,-1 and scale-2,3,1.
    const auto flatChild=glm::translate(glm::mat4(1),glm::vec3(107,5,-1))*glm::scale(glm::mat4(1),glm::vec3(-2,3,1));
    auto flat=instances;std::memcpy(flat[1].modelMatrix,glm::value_ptr(flatChild),64);
    std::vector<float> flatWorlds(32);for(u32 i=0;i<2;++i)std::memcpy(flatWorlds.data()+16*i,flat[i].modelMatrix,64);
    const std::array<GPUTransformNode,2> flatNodes{};std::vector<u32> flatSelected;reflectionProbeStaticSlots(flat,flatNodes,{},flatSelected);CHECK(flatSelected==selected);
    std::array<GPUMeshInfo,1> meshes{};meshes[0].boundingSphere[3]=1;
    const auto a=reflectionProbeWorldBounds(instances,meshes,worlds,selected),b=reflectionProbeWorldBounds(flat,meshes,flatWorlds,flatSelected);
    REQUIRE(a);REQUIRE(b);for(u32 k=0;k<3;++k){CHECK(a->minimum[k]==b->minimum[k]);CHECK(a->maximum[k]==b->maximum[k]);}
    CHECK(a->minimum.x>98);CHECK(a->maximum.x>109); // Nothing belongs near identity/origin.
    std::vector<float> wrong(32);for(u32 i=0;i<2;++i)std::memcpy(wrong.data()+16*i,instances[i].modelMatrix,64);
    const auto placeholder=reflectionProbeWorldBounds(instances,meshes,wrong);REQUIRE(placeholder);CHECK(placeholder->minimum.x<0);
    // Static flag / visibility membership changes need no slot or topology move.
    instances[0].flags&=~4u;reflectionProbeStaticSlots(instances,nodes,{},selected);CHECK(selected.empty());
    instances[0].flags|=4u;instances[1].flags&=~1u;reflectionProbeStaticSlots(instances,nodes,{},selected);CHECK(selected==std::vector<u32>{0});
    instances[1].flags|=1u;nodes[1].parentSlot=1;reflectionProbeStaticSlots(instances,nodes,{},selected);CHECK(selected==std::vector<u32>{0});
}

TEST_CASE("F13 animated ancestor excludes static children from raster and updates RT world bounds") {
    std::array<GPUInstance,2> instances{};std::array<GPUTransformNode,2> nodes{};std::array<GPUMotion,2> motions{};
    const glm::mat4 identity(1),local=glm::translate(identity,glm::vec3(2,0,0));
    for(auto& i:instances){i.flags=INSTANCE_FLAG_VALID|1u|4u;std::memcpy(i.modelMatrix,glm::value_ptr(identity),64);}
    std::memcpy(nodes[1].local,glm::value_ptr(local),64);nodes[1].parentSlot=0;nodes[1].depth=1;
    motions[0].centre[0]=10;motions[0].radius=3;motions[0].cosPhase=1;motions[0].base[0]=motions[0].base[4]=motions[0].base[8]=1;
    const std::array<u32,1> moving{0},children{1};const std::array<u32,3> offsets{0,1,1};
    std::vector<u32> selected;reflectionProbeStaticSlots(instances,nodes,moving,selected);CHECK(selected.empty());
    std::array<float,SCENE_MOTION_CLASSES*2> phases{};for(u32 k=0;k<SCENE_MOTION_CLASSES;++k)phases[k*2+1]=1;
    const HierarchyView view{instances,nodes,motions,moving,offsets,children};std::vector<float> first,second;referenceWorlds(view,phases.data(),first);
    phases[1]=-1;referenceWorlds(view,phases.data(),second);
    CHECK(first[12]==13);CHECK(first[16+12]==15);CHECK(second[12]==7);CHECK(second[16+12]==5);
    std::array<GPUMeshInfo,1> meshes{};meshes[0].boundingSphere[3]=.25f;
    const auto a=reflectionProbeWorldBounds(instances,meshes,first),b=reflectionProbeWorldBounds(instances,meshes,second);
    REQUIRE(a);REQUIRE(b);CHECK(a->minimum.x>12);CHECK(b->maximum.x<8);
    // Removing MotionComponent restores both memberships without recreating nodes.
    reflectionProbeStaticSlots(instances,nodes,{},selected);CHECK(selected==std::vector<u32>{0,1});
}

TEST_CASE("F13 probe BRDF fit stays physical without concealing invalid weights"){
    for(float roughness:{.5f,1.f})for(float nv:{.05f,.5f,1.f})for(float f0:{0.f,1.f}) {
        auto s=shiny();s.roughness=roughness;s.albedo[0]=s.albedo[1]=s.albedo[2]=f0;
        s.viewDirection[0]=std::sqrt(1-nv*nv);s.viewDirection[2]=nv;
        const auto bounded=reflectionProbeContribution(s,glm::vec3(1));
        const auto reference=reflectionReference(s,256,512,white);
        // Uniform solid-angle GGX quadrature is independent of the split-sum
        // fit. Projecting a finite fit onto [0,1] cannot increase its absolute
        // error against this physical integral; it is not an exact-fit claim.
        const glm::vec4 r=roughness*glm::vec4(-1,-.0275f,-.572f,.022f)+glm::vec4(1,.0425f,1.04f,-.04f);
        const float a=std::min(r.x*r.x,std::exp2(-9.28f*nv))*r.x+r.y;
        const float raw=f0*(-1.04f*a+r.z)+(1.04f*a+r.w);
        REQUIRE(reference.x>=0);REQUIRE(reference.x<=1);
        CHECK(bounded.x>=0);CHECK(bounded.x<=1);
        CHECK(std::abs(double(bounded.x)-reference.x)<=std::abs(double(raw)-reference.x)+1e-6);
        if(roughness==1&&f0==0){CHECK(raw==doctest::Approx(-.0024f));CHECK(bounded.x==0);CHECK(reference.x>0);}
        // A deliberately limited fit remains detectably approximate at rough
        // normal incidence, rather than claiming all clamps solve its error.
        if(roughness==1&&f0==1&&nv==1){
            CHECK(reference.x==doctest::Approx(1-std::log(2.0)).epsilon(1e-5));
            CHECK(bounded.x==doctest::Approx(.45f));
        }
    }
    auto corrupt=shiny();corrupt.albedo[0]=std::numeric_limits<float>::quiet_NaN();
    CHECK_FALSE(std::isfinite(reflectionProbeContribution(corrupt,glm::vec3(1)).x));
}

TEST_CASE("Probe storage rejects the complete out-of-range physical sum before HALF conversion") {
    constexpr u32 half=REFLECTION_CAPTURE_HALF;
    CHECK(reflectionProbeStorageAccepts(65504,128,64,half));
    CHECK_FALSE(reflectionProbeStorageAccepts(std::nextafter(65504.f,std::numeric_limits<float>::infinity()),128,64,half));
    // Individually representable emission and scattering can overflow in sum.
    CHECK(reflectionProbeStorageAccepts(40000,0,0,half));CHECK(reflectionProbeStorageAccepts(30000,0,0,half));
    CHECK_FALSE(reflectionProbeStorageAccepts(40000+30000,0,0,half));
    CHECK(reflectionProbeStorageAccepts(40000+30000,0,0,0));
    CHECK_FALSE(reflectionProbeStorageAccepts(368640,128,64,half));
    CHECK(reflectionProbeStorageAccepts(368640,128,64,REFLECTION_ENABLE_RT));
    for(u32 flags:{0u,half}) {
        CHECK_FALSE(reflectionProbeStorageAccepts(-1,0,0,flags));
        CHECK_FALSE(reflectionProbeStorageAccepts(std::numeric_limits<float>::infinity(),0,0,flags));
        CHECK_FALSE(reflectionProbeStorageAccepts(0,std::numeric_limits<float>::quiet_NaN(),0,flags));
    }
}
