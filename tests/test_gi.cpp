#include <doctest/doctest.h>
#include "renderer/probe_grid.h"
#include "renderer/radiance_cache.h"
#include "renderer/gi_reservoir.h"
#include "renderer/offline_reference.h"
#include "renderer/rt_geometry.h"
#include <array>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <stdexcept>
#include <vector>

using namespace phosphor;

TEST_CASE("F12 mirrored world winding does not invert the original material front") {
    // Independent geometric oracle: object-space CCW triangle, original normal
    // +Z. Reflect X through the model; world cross is -Z. A ray from z=2 toward
    // the outside/front still sees the material front even though WORLD facing
    // is false. No call to the shader or RT facing helper computes this oracle.
    const glm::dvec3 a(-1,-1,0),b(1,-1,0),c(0,1,0),outsideOrigin(0,0,2),outsideDirection(0,0,-1);
    auto mirrorX=[](glm::dvec3 p){return glm::dvec3(-p.x,p.y,p.z);};
    const glm::dvec3 wa=mirrorX(a),wb=mirrorX(b),wc=mirrorX(c);
    const glm::dvec3 worldCross=glm::cross(wb-wa,wc-wa);
    const bool worldFront=glm::dot(worldCross,-outsideDirection)>0;
    const glm::dvec3 objectCross=glm::cross(b-a,c-a);
    const bool materialFrontOracle=glm::dot(objectCross,mirrorX(outsideOrigin)-a)>0;
    REQUIRE(materialFrontOracle);
    REQUIRE_FALSE(worldFront);
    CHECK(rtMaterialFrontFacing(u32(worldFront),INSTANCE_FLAG_MIRRORED)==materialFrontOracle);
    // Negative old path: interpreting world facing as material front would mark
    // this outside probe as inside/backfacing and zero the reflected GI.
    CHECK(worldFront!=materialFrontOracle);
    const glm::dvec3 insideOrigin(0,0,-2),insideDirection(0,0,1);
    const bool insideWorldFront=glm::dot(worldCross,-insideDirection)>0;
    const bool insideMaterialOracle=glm::dot(objectCross,mirrorX(insideOrigin)-a)>0;
    REQUIRE_FALSE(insideMaterialOracle);
    CHECK(rtMaterialFrontFacing(u32(insideWorldFront),INSTANCE_FLAG_MIRRORED)==insideMaterialOracle);
    CHECK(rtMaterialFrontFacing(1,0));
    CHECK_FALSE(rtMaterialFrontFacing(0,0));
    CHECK(rtMaterialFrontFacing(1,INSTANCE_FLAG_VALID|3u)); // unrelated flags do not change side
}

namespace {
std::vector<GPUProbeRay> skyRays(u32 count,glm::vec3 L={1,1,1},float distance=100) {
    std::vector<GPUProbeRay> rays(count);
    for(u32 k=0;k<count;++k) {
        const auto d=probeRayDirection(k,count,0);
        for(int axis=0;axis<3;++axis) {rays[k].direction[axis]=d[axis];rays[k].radiance[axis]=L[axis];}
        rays[k].distance=distance;
    }
    return rays;
}
ProbeGridConfig smallGrid() {
    ProbeGridConfig c;c.origin={0,0,0};c.counts={2,2,2};c.spacing={2,2,2};c.raysPerProbe=1024;c.hysteresis=0;
    return c;
}
GPUGiReservoir secondary(glm::vec3 L={1,1,1}) {
    GPUGiReservoir y{};
    y.position[1]=2;y.normal[1]=-1;y.sourceNormal[1]=1;
    y.flags=GI_SAMPLE_VALID;y.slot=0;y.instanceGeneration=4;
    for(int k=0;k<3;++k) y.radiance[k]=L[k];
    return y;
}
std::string contents(const std::filesystem::path& p) {
    std::ifstream in(p,std::ios::binary);
    return {std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>()};
}
}

TEST_CASE("F12 probe octahedral atlas has unit vectors and folded seam") {
    for(const glm::vec3 n:{glm::vec3(1,0,0),glm::vec3(0,1,0),glm::vec3(0,0,-1),
                           glm::normalize(glm::vec3(-1,2,-3))}) {
        const auto decoded=probeOctDecode(probeOctEncode(n));
        CHECK(glm::length(decoded-n)<1e-5f);
    }
    for(u32 k=0;k<1024;++k) CHECK(glm::length(probeRayDirection(k,1024,31))==doctest::Approx(1).epsilon(1e-5));
    CHECK_FALSE(validProbeGrid(ProbeGridConfig{.spacing=glm::vec3(0)}));
    auto bad=smallGrid();bad.counts.x=1;CHECK_THROWS_AS(ProbeGrid{bad},std::invalid_argument);
}
TEST_CASE("F12 isotropic radiance integrates to pi irradiance independent of rays") {
    auto c=smallGrid();ProbeGrid grid(c);const auto rays=skyRays(c.raysPerProbe);
    for(u32 k=0;k<grid.probeCount();++k) REQUIRE(grid.update(k,rays).active);
    const auto E=grid.irradiance({1,1,1},{0,1,0});
    CHECK(E.x==doctest::Approx(3.14159265).epsilon(0.025));
    CHECK(E.x==doctest::Approx(E.y));CHECK(E.y==doctest::Approx(E.z));
    // Negative control: radiance and irradiance are numerically DIFFERENT.
    CHECK(std::abs(E.x-1)>2);
    grid.reset(8);CHECK(glm::length(grid.irradiance({1,1,1},{0,1,0}))==0);
    CHECK(glm::length(grid.irradiance({20,1,1},{0,1,0}))==0);
}
TEST_CASE("F12 probe inside wall becomes inactive and bounded relocation recovers") {
    auto c=smallGrid();c.raysPerProbe=64;ProbeGrid grid(c);auto rays=skyRays(c.raysPerProbe);
    for(u32 k=0;k<32;++k) {rays[k].backface=1;rays[k].distance=-0.02f;}
    auto update=grid.update(0,rays);
    CHECK_FALSE(update.active);CHECK(update.relocated);CHECK(grid.states()[0].age==0);
    for(int k=0;k<50;++k) grid.update(0,rays);
    const auto& s=grid.states()[0];
    const glm::vec3 offset(s.offset[0],s.offset[1],s.offset[2]);
    CHECK(glm::length(offset/c.spacing)<=c.maxRelocation+1e-5f);
    auto recovery=skyRays(c.raysPerProbe);CHECK(grid.update(0,recovery).active);
    CHECK(grid.states()[0].age==1);
    // Invalid backface signs and NaNs must not poison an atlas/history.
    recovery[0].backface=1;CHECK_THROWS(grid.update(0,recovery));
    recovery[0].backface=0;recovery[0].radiance[0]=std::numeric_limits<float>::quiet_NaN();
    CHECK_THROWS(grid.update(0,recovery));
}
TEST_CASE("F12 visibility moments detect thin wall leak and corruption control") {
    CHECK(probeVisibility(2,1,1)==0);CHECK(probeVisibility(0.5f,1,1)==1);
    CHECK(probeVisibility(2,1,100)>0.9f); // corrupted second moment MUST expose leak
    CHECK(probeVisibility(2,1,std::numeric_limits<float>::quiet_NaN())==0);
    CHECK(probeVisibility(1,-1,0)==0);
}
TEST_CASE("F12 radiance invalidation preserves multi-step probe escape geometry") {
    auto cfg=smallGrid();cfg.raysPerProbe=64;cfg.relocationStep=0.1f;
    ProbeGrid fixed(cfg),old(cfg);
    auto slabRays=[&](const ProbeGrid& grid,float radiance) {
        auto rays=skyRays(cfg.raysPerProbe,{0,0,0},cfg.maxDistance);
        const float x=grid.position(0).x,halfThickness=0.45f;
        const bool inside=std::abs(x)<halfThickness;
        for(auto& r:rays) {
            const float dx=r.direction[0];
            if(std::abs(dx)<1e-8f)continue;
            float boundary=0;
            if(inside)boundary=dx>0?halfThickness:-halfThickness;
            else if(x>halfThickness&&dx<0)boundary=halfThickness;
            else if(x<-halfThickness&&dx>0)boundary=-halfThickness;
            else continue;
            const float t=(boundary-x)/dx;
            if(!(t>0&&t<=cfg.maxDistance))continue;
            r.backface=inside?1u:0u;r.distance=inside?-t:t;
            if(!inside)for(float& c:r.radiance)c=radiance;
        }
        return rays;
    };
    bool recovered=false;
    for(u32 frame=0;frame<20;++frame) {
        const auto before=fixed.states()[0];
        fixed.invalidateRadiance(); // Sun/emissive changed every frame.
        CHECK(fixed.states()[0].generation==before.generation);
        CHECK(fixed.states()[0].state==before.state);
        for(u32 axis=0;axis<3;++axis)CHECK(fixed.states()[0].offset[axis]==before.offset[axis]);
        recovered|=fixed.update(0,slabRays(fixed,float(frame+1))).active;
        const auto& s=fixed.states()[0];
        const glm::vec3 offset(s.offset[0],s.offset[1],s.offset[2]);
        CHECK(glm::length(offset/cfg.spacing)<=cfg.maxRelocation+1e-5f);
        old.reset(1); // Negative previous radiometric-reset-as-geometry-reset behavior.
        CHECK_FALSE(old.update(0,slabRays(old,float(frame+1))).active);
    }
    REQUIRE(recovered);
    CHECK(std::abs(fixed.position(0).x)>0.45f);
    fixed.reset(9);CHECK(glm::length(fixed.position(0)-cfg.origin)==0);
    CHECK(fixed.states()[0].generation==9);CHECK(fixed.states()[0].relocationTravel==0);
}
TEST_CASE("F12 bounded hash compares full key, evicts, ages and resets generation") {
    RadianceCacheConfig c;c.capacity=2;c.probeLimit=2;c.maxAge=2;
    RadianceCache cache(c);cache.reset(7,{1,2,3});
    RadianceCacheKey a{},b{1,0,0,0,0},other{2,0,0,0,0};
    // Capacity2 guarantees deliberate collisions after enough distinct keys.
    while(radianceCacheHash(b)%2!=radianceCacheHash(a)%2) ++b.x;
    REQUIRE(cache.insert(a,{1,2,3},1));REQUIRE(cache.insert(b,{4,5,6},2));
    REQUIRE(cache.lookup(a,2));CHECK(cache.lookup(a,2)->x==1);
    CHECK_FALSE(cache.lookup({a.x,a.y,a.z,a.normal,a.direction+1},2)); // directional key is not optional
    CHECK(cache.size()==2);cache.insert(other,{7,8,9},3);CHECK(cache.size()==2);CHECK(cache.evictions()==1);
    CHECK_FALSE(cache.lookup(a,3));REQUIRE(cache.lookup(other,3));CHECK_FALSE(cache.lookup(other,6));
    cache.reset(8,{1,20,3});CHECK(cache.size()==0);CHECK_FALSE(cache.lookup(other,3));
    cache.insert(other,{2,2,2},0xfffffffeu);REQUIRE(cache.lookup(other,0));
    CHECK_FALSE(cache.lookup(other,1)); // modular age3 >2
    cache.reset(8,{1,20,3});CHECK_FALSE(cache.lookup(other,0)); // repeated/wrapped epoch clears
    CHECK_FALSE(cache.insert(a,{-1,0,0},1));CHECK_FALSE(radianceCacheKey({INFINITY,0,0},{0,1,0},{1,0,0},0.5f));
}
TEST_CASE("F12 radiance cache averaging remains bounded and distinct from irradiance") {
    RadianceCache cache({.capacity=4,.probeLimit=4,.maxAge=10,.maxSamples=2});
    const auto key=radianceCacheKey({0.2f,1,0.4f},{0,1,0},{1,0,0},0.5f);
    REQUIRE(key);cache.insert(*key,{1,1,1},1);cache.insert(*key,{3,3,3},2);
    REQUIRE(cache.lookup(*key,2));CHECK(cache.lookup(*key,2)->x==2);
    cache.insert(*key,{4,4,4},3);CHECK(cache.lookup(*key,3)->x==3);
    CHECK_FALSE(cache.lookup(*radianceCacheKey({0.2f,1,0.4f},{0,-1,0},{1,0,0},0.5f),3));
    cache.reset(2,{1,2,3});CHECK_FALSE(cache.lookup(*key,3)); // material/emissive reset
}
TEST_CASE("F12 fresh cosine path oracle transforms PDF from solid angle to area once") {
    GiReceiver x;auto sample=secondary({2,2,2});
    const auto c=giConnection(x,sample,true);
    REQUIRE(c.valid);CHECK(c.proposalArea==doctest::Approx(1.0/(4*3.141592653589793)));
    GPUGiReservoir r{};
    giAddCandidate(r,sample,c.target,c.proposalArea,0);giFinalize(r);
    const auto L=giShade(r,c);CHECK(L.x==doctest::Approx(2));
    GPUGiReservoir wrong{};
    giAddCandidate(wrong,sample,c.target,1.f/3.14159265359f,0);giFinalize(wrong);
    CHECK(std::abs(giShade(wrong,c).x-L.x)>1); // negative omitted area Jacobian
    CHECK_FALSE(giConnection(x,sample,false).valid);
}
TEST_CASE("F12 zero-proposal history preserves the IID two-sample Bernoulli mean") {
    // Independent exact expectation: four equally probable outcomes of two
    // Bernoulli(0.5) samples. The empirical two-sample estimator is (a+b)/2.
    // This test enumerates outcomes rather than reproducing a shader RNG.
    double correctedMean=0,oldMean=0;
    for(int a=0;a<2;++a)for(int b=0;b<2;++b) {
        auto candidate=secondary();candidate.proposalSolidAngle=1;candidate.flags|=GI_PROPOSAL_VALID;
        GPUGiReservoir receiver{},source{};
        giAddCandidate(receiver,candidate,float(a),1,0);giFinalize(receiver);
        giAddCandidate(source,candidate,float(b),1,0);giFinalize(source);
        REQUIRE(receiver.M==1);REQUIRE(source.M==1);
        const bool sourceHadPositive=bool(source.flags&GI_SAMPLE_VALID);
        auto old=receiver;
        if(sourceHadPositive)giMerge(old,source,float(b),true,0,32);giFinalize(old);
        giMerge(receiver,source,float(b),true,0,32);giFinalize(receiver);
        REQUIRE(receiver.M==2);
        const double expected=double(a+b)/2;
        const double value=(receiver.flags&GI_SAMPLE_VALID)?receiver.target*receiver.W:0;
        CHECK(value==doctest::Approx(expected));
        correctedMean+=value/4;
        oldMean+=((old.flags&GI_SAMPLE_VALID)?old.target*old.W:0)/4;
    }
    CHECK(correctedMean==doctest::Approx(0.5));
    CHECK(oldMean==doctest::Approx(0.625)); // negative old discard-zero-source rule
    GPUGiReservoir zero{};auto candidate=secondary();candidate.flags|=GI_PROPOSAL_VALID;
    giAddCandidate(zero,candidate,0,1,0);giFinalize(zero);
    CHECK((zero.flags&GI_SAMPLE_VALID)==0);CHECK((zero.flags&GI_PROPOSAL_VALID)!=0);CHECK(zero.W==0);
    GPUProbeGridParams p{};GiReceiver x;
    CHECK(giHistoryCompatible(zero,p,x,0.1f));
    ++p.viewRevision;CHECK_FALSE(giHistoryCompatible(zero,p,x,0.1f));
    auto corrupt=zero;corrupt.weightSum=std::numeric_limits<float>::infinity();giFinalize(corrupt);
    CHECK(corrupt.flags==0); // Do not reinterpret numeric poison as valid zero.
    GPUGiReservoir destination{};
    CHECK_FALSE(giMerge(destination,corrupt,0,false,0));CHECK(destination.M==0);
}
TEST_CASE("F12 reservoir zero-contribution paths count and bounded history rejects revisions") {
    GiReceiver x;auto sample=secondary();auto c=giConnection(x,sample,true);
    GPUGiReservoir r{};giAddCandidate(r,sample,c.target,c.proposalArea,0);
    giAddCandidate(r,sample,0,c.proposalArea,0);giFinalize(r);
    CHECK(r.M==2);CHECK(giShade(r,c).x==doctest::Approx(0.5));
    GPUGiReservoir merged{};giMerge(merged,r,0,false,0,2);CHECK(merged.M==2);CHECK(merged.weightSum==0);
    giMerge(merged,r,c.target,true,0,1);giFinalize(merged);CHECK(merged.M==3);
    GPUProbeGridParams p{};r.geometryRevision=1;p.geometryRevision=1;
    REQUIRE(giHistoryCompatible(r,p,x,0.1f));
    ++p.geometryRevision;CHECK_FALSE(giHistoryCompatible(r,p,x,0.1f));
    --p.geometryRevision;++p.materialRevision;CHECK_FALSE(giHistoryCompatible(r,p,x,0.1f));
    --p.materialRevision;++p.lightRevision;CHECK_FALSE(giHistoryCompatible(r,p,x,0.1f));
    --p.lightRevision;++p.viewRevision;CHECK_FALSE(giHistoryCompatible(r,p,x,0.1f));
}
TEST_CASE("F12 exact reference validation rejects missing texture or local singular world") {
    const std::array<GPUVertex,3> vertices{};
    const std::array<u32,3> indices{0,1,2};
    GPUMeshInfo mesh{};mesh.indexCount=3;GPUMaterial material{};material.baseColor[0]=material.baseColor[1]=material.baseColor[2]=1;
    GPUInstance instance{};instance.flags=INSTANCE_FLAG_VALID|1;instance.modelMatrix[0]=instance.modelMatrix[5]=instance.modelMatrix[10]=instance.modelMatrix[15]=1;
    OfflineReferenceScene s{.vertices=vertices,.indices=indices,.meshes=std::span(&mesh,1),
        .worldInstances=std::span(&instance,1),.materials=std::span(&material,1)};
    // Zero-initialised texture IDs are REAL texture indices, not absent markers.
    material.baseColorTex=material.normalTex=material.metallicRoughnessTex=material.occlusionTex=material.emissiveTex=INVALID_TEXTURE_INDEX;
    REQUIRE(validateReferenceScene(s).ok);
    material.emissiveTex=4;CHECK_FALSE(validateReferenceScene(s).ok);material.emissiveTex=INVALID_TEXTURE_INDEX;
    instance.modelMatrix[0]=0;CHECK_FALSE(validateReferenceScene(s).ok);
    instance.modelMatrix[0]=1;mesh.indexCount=4;CHECK_FALSE(validateReferenceScene(s).ok);
}
TEST_CASE("F12 offline snapshot PFM remains linear and full world/material values survive") {
    const auto dir=std::filesystem::temp_directory_path()/"phosphor_f12_reference_unit";
    std::filesystem::remove_all(dir);std::filesystem::remove_all(dir.string()+".partial");
    std::array<GPUVertex,3> v{};v[1].px=1;v[2].py=1;
    const std::array<u32,3> ids{0,1,2};GPUMeshInfo m{};m.indexCount=3;
    GPUMaterial material{};material.baseColor[0]=0.2f;material.baseColor[3]=1;material.emissive[0]=8;
    material.baseColorTex=material.normalTex=material.metallicRoughnessTex=material.occlusionTex=material.emissiveTex=INVALID_TEXTURE_INDEX;
    GPUInstance i{};i.modelMatrix[0]=i.modelMatrix[5]=i.modelMatrix[10]=i.modelMatrix[15]=1;i.modelMatrix[12]=5;i.flags=INSTANCE_FLAG_VALID|1;
    OfflineReferenceScene s{.vertices=v,.indices=ids,.meshes=std::span(&m,1),.worldInstances=std::span(&i,1),.materials=std::span(&material,1)};
    auto result=exportOfflineReference(s,dir);REQUIRE(result.ok);
    const auto manifest=contents(dir/"scene.json");
    CHECK(manifest.find("\"emissive\":[8,0,0]")!=std::string::npos);
    CHECK(manifest.find("\"world\":[1,0,0,0,0,1,0,0,0,0,1,0,5,0,0,1]")!=std::string::npos);
    CHECK(contents(dir/"mesh_0.ply").find("element face 1")!=std::string::npos);
    CHECK_FALSE(exportOfflineReference(s,dir).ok); // no overwrite reference
    const std::array<float,3> rgb{8,2,0.5f};std::string error;
    REQUIRE(writeLinearPfm(dir/"linear.pfm",1,1,rgb,error));
    const auto bytes=contents(dir/"linear.pfm");
    CHECK(bytes.starts_with("PF\n1 1\n-1.0\n")); // HDR8 stored unexposed, FLOAT32
    std::filesystem::remove_all(dir);
}
TEST_CASE("F12 reference preserves caster-only roles and standalone triangle emitters") {
    const auto dir=std::filesystem::temp_directory_path()/"phosphor_f12_reference_roles";
    std::filesystem::remove_all(dir);std::filesystem::remove_all(dir.string()+".partial");
    std::array<GPUVertex,3> vertices{};vertices[1].px=1;vertices[2].py=1;
    const std::array<u32,3> indices{0,1,2};GPUMeshInfo mesh{};mesh.indexCount=3;
    GPUMaterial m{};m.baseColor[3]=1;
    m.baseColorTex=m.normalTex=m.metallicRoughnessTex=m.occlusionTex=m.emissiveTex=INVALID_TEXTURE_INDEX;
    GPUInstance i{};i.modelMatrix[0]=i.modelMatrix[5]=i.modelMatrix[10]=i.modelMatrix[15]=1;
    i.flags=INSTANCE_FLAG_VALID|2u; // Invisible camera/indirect, valid shadow caster.
    ReferenceAreaLight light;light.type=6;light.flags=0;light.materialIndex=INVALID_TEXTURE_INDEX;
    light.position[1]=2;light.axisU[0]=1;light.axisV[2]=1;light.emission[0]=8;
    OfflineReferenceScene s{.vertices=vertices,.indices=indices,.meshes=std::span(&mesh,1),
        .worldInstances=std::span(&i,1),.materials=std::span(&m,1),.sampledLights=std::span(&light,1)};
    auto validation=validateReferenceScene(s);REQUIRE(validation.ok);CHECK(validation.instances==1);
    REQUIRE(exportOfflineReference(s,dir).ok);
    const auto json=contents(dir/"scene.json");
    CHECK(json.find("\"flags\":18")!=std::string::npos); // Both exact role bits retained.
    CHECK(json.find("\"geometry\":\"light_triangle_0.ply\"")!=std::string::npos);
    CHECK(contents(dir/"light_triangle_0.ply").find("element face 1")!=std::string::npos);
    std::filesystem::remove_all(dir);
}

TEST_CASE("F12 reference permits explicit infinite far and refuses inverted finite clip") {
    OfflineReferenceScene s;s.camera.farPlane=0;
    CHECK(validateReferenceScene(s).ok);
    s.camera.farPlane=s.camera.nearPlane*0.5f;
    CHECK_FALSE(validateReferenceScene(s).ok);
}


TEST_CASE("F12 history-chain age includes unselected zero and blocked source mass") {
    auto sample=secondary();sample.flags|=GI_PROPOSAL_VALID;
    GPUGiReservoir history{},fresh{};
    giAddCandidate(history,sample,1,1,0);giFinalize(history);history.age=31;
    giAddCandidate(fresh,sample,1,1,0);giFinalize(fresh);
    auto merged=fresh;CHECK_FALSE(giMerge(merged,history,1,true,0.999999f,32));
    CHECK(merged.age==32);CHECK(merged.M==2);
    giFinalize(merged);GPUProbeGridParams params{};GiReceiver receiver;
    CHECK_FALSE(giHistoryCompatible(merged,params,receiver,0.1f));
    GPUGiReservoir zero{};giAddCandidate(zero,sample,0,1,0);giFinalize(zero);zero.age=31;
    merged=fresh;CHECK_FALSE(giMerge(merged,zero,0,false,0,32));CHECK(merged.age==32);CHECK(merged.M==2);
    merged=fresh;CHECK_FALSE(giMerge(merged,history,0,false,0,32));CHECK(merged.age==32);CHECK(merged.M==2);
}

TEST_CASE("F12 exact same-support IID expiry is independent of endpoint selection") {
    // Shrink the caller's age threshold to1 and capM to2 for exhaustive five-frame
    // enumeration; production bounds remain32. True integral2, proposal weights
    // 1/3 with equal probability. No shift or visibility mismatch is involved.
    auto expectation=[](bool oldEndpointAge) {
        struct State { GPUGiReservoir r;double probability; };
        std::vector<State> states{{{},1.0}};
        for(u32 frame=0;frame<5;++frame) {
            std::vector<State> next;
            for(const auto& state:states)for(float weight:{1.f,3.f}) {
                GPUGiReservoir fresh{};auto sample=secondary();sample.flags|=GI_PROPOSAL_VALID;
                giAddCandidate(fresh,sample,weight*0.5f,0.5f,0);giFinalize(fresh);
                if(!state.r.M || state.r.age>=1) {next.push_back({fresh,state.probability*0.5});continue;}
                const u32 m=std::min(state.r.M,2u);
                const double oldWeight=double(state.r.target)*state.r.W*m;
                const double chooseOld=oldWeight/(oldWeight+weight);
                for(bool selectedOld:{false,true}) {
                    auto merged=fresh;
                    giMerge(merged,state.r,state.r.target,true,selectedOld?0.f:0.999999f,2);
                    if(oldEndpointAge)merged.age=selectedOld?state.r.age+1:0;
                    giFinalize(merged);
                    next.push_back({merged,state.probability*0.5*(selectedOld?chooseOld:1-chooseOld)});
                }
            }
            states=std::move(next);
        }
        double mean=0,probability=0;
        for(const auto& state:states) {mean+=state.probability*state.r.target*state.r.W;probability+=state.probability;}
        CHECK(probability==doctest::Approx(1));return mean;
    };
    CHECK(expectation(false)==doctest::Approx(2).epsilon(1e-6));
    CHECK(expectation(true)==doctest::Approx(182666318.0/91265265.0).epsilon(1e-6));
    CHECK(expectation(true)>2.001);
}


TEST_CASE("F12 probe parameter ABI has zero-default explicit physical diagnostic flags") {
    CHECK(sizeof(GPUProbeGridParams)==176);
    ProbeGrid grid(ProbeGridConfig{});const auto params=grid.parameters(17,2);
    CHECK(params.debugFlags==0);
    for (u32 word : params.debugPadding) CHECK(word==0);
    CHECK((GI_DEBUG_NO_VISIBILITY & (GI_DEBUG_NO_VISIBILITY-1u))==0);
}
