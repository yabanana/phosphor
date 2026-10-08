#include <doctest/doctest.h>
#include "renderer/shadow_dispatch.h"

#include <array>
#include <map>
#include <utility>

using namespace phosphor;

namespace {
struct Scene {
    std::array<GPUMeshInfo,3> meshes{};
    std::vector<SceneBucket> buckets;
    std::array<GPUInstance,40> instances{};
    std::array<u32,40> cascadeFlags{};
    Scene() {
        meshes[0].meshletOffset=10;meshes[0].meshletCount=3;meshes[0].indexCount=6;
        meshes[1].meshletOffset=20;meshes[1].meshletCount=5;meshes[1].indexCount=9;
        meshes[2].meshletOffset=35;meshes[2].meshletCount=2;meshes[2].indexCount=12;
        // Used is the high-water prefix, NOT live count. Ranges are deliberately
        // not ordered by slot, and one mesh occupies two winding-class buckets.
        buckets={{0,CullClass::Back,8,8,3,6,0},{1,CullClass::Back,0,8,2,5,1},
                 {0,CullClass::BackMirrored,16,8,2,4,2},{2,CullClass::None,24,8,2,5,3},
                 {1,CullClass::None,32,8,0,6,4}};
        auto live=[&](u32 slot,u32 mesh,u32 flags,u32 cascades) {
            instances[slot].meshIndex=mesh;instances[slot].flags=flags|INSTANCE_FLAG_VALID;
            cascadeFlags[slot]=cascades;
        };
        live(8,0,2|4,3); // Off-camera/invisible but casts shadows, static.
        live(10,0,3,5);
        live(13,0,1,15); // Live beyond holes but NOT a caster.
        live(1,1,2,8);live(4,1,3|4,15);
        live(16,0,2|INSTANCE_FLAG_MIRRORED,4);live(19,0,3|INSTANCE_FLAG_MIRRORED,1);
        live(24,2,3|4,7);live(28,2,2,15); // Double-sided/MASK bucket, late live slot after holes.
    }
};
using Pairs=std::map<std::pair<u32,u32>,u32>;
bool caster(const Scene& s,u32 slot,u32 cascade,u32 cacheClass) {
    const auto flags=s.instances[slot].flags;
    if(!(flags&INSTANCE_FLAG_VALID)||!(flags&2u)||!(s.cascadeFlags[slot]&(1u<<cascade)))return false;
    return cacheClass==2 || ((flags&4u)!=0)==(cacheClass==0);
}
// Independent oracle: enumerate actual live scene slots and THEIR mesh only.
// Neither buckets, dispatch chunks nor shader address helpers select the set.
Pairs expected(const Scene& s,u32 cascade,u32 cacheClass,bool meshPath) {
    Pairs pairs;
    for(u32 slot=0;slot<s.instances.size();++slot) {
        if(!caster(s,slot,cascade,cacheClass))continue;
        const auto& mesh=s.meshes[s.instances[slot].meshIndex];
        const u32 count=meshPath?mesh.meshletCount:mesh.indexCount/3;
        for(u32 element=0;element<count;++element)
            pairs[{slot,meshPath?mesh.meshletOffset+element:element}]=1;
    }
    return pairs;
}
Pairs emitted(const Scene& s,const std::vector<ShadowCasterDraw>& draws,u32 cascade,u32 cacheClass) {
    Pairs pairs;
    for(const auto& draw:draws)for(u32 localSlot=0;localSlot<draw.slotCount;++localSlot) {
        const auto slot=shadowDrawSlot(draw.firstSlot,localSlot,s.instances.size());
        REQUIRE(slot<s.instances.size());
        if(!caster(s,slot,cascade,cacheClass)||s.instances[slot].meshIndex!=draw.mesh)continue;
        if(draw.meshShader) {
            for(u32 x=0;x<draw.meshletCount;++x)
                ++pairs[{slot,shadowDrawMeshlet(draw.meshletFirst,x,draw.meshletCount)}];
        } else {
            for(u32 triangle=0;triangle<s.meshes[draw.mesh].indexCount/3;++triangle)++pairs[{slot,triangle}];
        }
    }
    return pairs;
}
} // namespace

TEST_CASE("shadow dispatch covers each caster primitive once including holes mirrored and off-camera slots") {
    const Scene scene;
    for(bool meshPath:{true,false}) {
        std::vector<ShadowCasterDraw> draws;
        // Tiny limits force chunking on BOTH axes in this small independent fixture.
        planShadowCasterDraws(scene.buckets,scene.meshes,scene.instances.size(),meshPath,draws,2,3);
        REQUIRE_FALSE(draws.empty());
        for(const auto& draw:draws) {
            CHECK(draw.slotCount<=3);
            if(meshPath)CHECK(draw.meshletCount<=2);
            CHECK(draw.meshShader==meshPath);
            const auto owner=std::find_if(scene.buckets.begin(),scene.buckets.end(),[&](const auto& bucket) {
                return draw.firstSlot>=bucket.firstSlot && draw.firstSlot-bucket.firstSlot<bucket.used;
            });
            REQUIRE(owner!=scene.buckets.end());
            CHECK(draw.mesh==owner->mesh);
            CHECK(draw.cull==owner->cull);
            CHECK(draw.firstSlot+draw.slotCount<=owner->firstSlot+owner->used);
        }
        for(u32 cascade=0;cascade<4;++cascade)for(u32 cacheClass=0;cacheClass<3;++cacheClass)
            CHECK(emitted(scene,draws,cascade,cacheClass)==expected(scene,cascade,cacheClass,meshPath));
        // In particular, using count=2 in the double-sided bucket would lose slot28.
        CHECK(emitted(scene,draws,0,2).contains({28,meshPath?35u:0u}));
    }
}

TEST_CASE("shadow dispatch work is per bucket rather than global meshlets times global slots") {
    const Scene scene;
    std::vector<ShadowCasterDraw> draws;
    planShadowCasterDraws(scene.buckets,scene.meshes,scene.instances.size(),true,draws,2,3);
    u64 meshGroups=0;
    for(const auto& draw:draws)meshGroups+=u64(draw.meshletCount)*draw.slotCount;
    CHECK(meshGroups==65); // 6*3 + 5*5 + 4*3 + 5*2; empty bucket excluded.
    CHECK(meshGroups<u64(3+5+2)*scene.instances.size());
    planShadowCasterDraws(scene.buckets,scene.meshes,scene.instances.size(),false,draws,2,3);
    u64 indexedInstances=0;
    for(const auto& draw:draws)indexedInstances+=draw.slotCount;
    CHECK(indexedInstances==20); // 6+5+4+5, not one full-scene draw for EACH mesh.
}

TEST_CASE("shadow dispatch legal mesh grids chunk large ranges without changing their coverage") {
    GPUMeshInfo mesh{};mesh.meshletOffset=7;mesh.meshletCount=65537;mesh.indexCount=3;
    SceneBucket bucket{0,CullClass::None,9,65537,1,65537,0};
    std::vector<ShadowCasterDraw> draws;
    planShadowCasterDraws(std::span(&bucket,1),std::span(&mesh,1),65546,true,draws);
    REQUIRE(draws.size()==4);
    u64 groups=0;
    for(const auto& draw:draws){CHECK(draw.meshletCount<=65535);CHECK(draw.slotCount<=65535);groups+=u64(draw.meshletCount)*draw.slotCount;}
    CHECK(groups==u64(65537)*65537);
    for(u32 slot:{9u,65543u,65544u,65545u})for(u32 index:{7u,65541u,65542u,65543u}) {
        u32 owners=0;
        for(const auto& draw:draws)owners+=slot>=draw.firstSlot && slot-draw.firstSlot<draw.slotCount &&
            index>=draw.meshletFirst && index-draw.meshletFirst<draw.meshletCount;
        CHECK(owners==1);
    }
    planShadowCasterDraws(std::span(&bucket,1),std::span(&mesh,1),65546,false,draws);
    CHECK(draws.size()==2); // Indexed fallback never repeats a whole mesh per meshlet chunk.
}

TEST_CASE("shadow dispatch preserves indexed fallback and rejects unsafe ranges") {
    GPUMeshInfo mesh{};mesh.indexCount=3;
    SceneBucket bucket{0,CullClass::Back,0,8,1,4,0};
    std::vector<ShadowCasterDraw> draws;
    planShadowCasterDraws(std::span(&bucket,1),std::span(&mesh,1),8,true,draws);
    REQUIRE(draws.size()==1);CHECK_FALSE(draws[0].meshShader);CHECK(draws[0].slotCount==4);
    bucket.count=0;
    planShadowCasterDraws(std::span(&bucket,1),std::span(&mesh,1),8,true,draws);
    CHECK(draws.empty());
    bucket.count=1;bucket.used=9;
    CHECK_THROWS_AS(planShadowCasterDraws(std::span(&bucket,1),std::span(&mesh,1),8,true,draws),std::invalid_argument);
    bucket.used=4;bucket.firstSlot=1;
    CHECK_THROWS_AS(planShadowCasterDraws(std::span(&bucket,1),std::span(&mesh,1),8,true,draws),std::invalid_argument);
    bucket.firstSlot=0;
    CHECK_THROWS_AS(planShadowCasterDraws(std::span(&bucket,1),std::span(&mesh,1),8,true,draws,0,3),std::invalid_argument);
    CHECK(shadowDrawSlot(8,0,8)==~0u);
    CHECK(shadowDrawSlot(2,7,8)==~0u);
    CHECK(shadowDrawSlot(5,2,8)==7);
    CHECK(shadowDrawMeshlet(20,2,3)==22);
    CHECK(shadowDrawMeshlet(20,3,3)==~0u);
    CHECK(shadowDrawMeshlet(~0u,1,2)==~0u);
}
