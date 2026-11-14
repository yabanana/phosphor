#include "renderer/denoise_settings.h"
#include <initializer_list>
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
    DenoiseHistory history=a;for(u32 i=0;i<100;++i)history=denoiseTemporal(s,first,history,p,{},{});CHECK(history.length==settings.maxHistory);CHECK(history.color[0]==doctest::Approx(2));
    const auto startup=denoiseTemporal(s,first,{},p,{},{},{},4);CHECK(startup.variance==doctest::Approx(4)); // short history uses local variance
}
TEST_CASE("F13 camera motion compares historical world receivers in the current view") {
    auto before=receiver();before.position[2]=-4;before.depth=4;
    auto p=denoiseParameters({},4,4,DENOISE_SIGNAL_GI,0,1,1,false);
    const auto history=denoiseTemporal(before,glm::vec3(2),{},p,{},{});
    REQUIRE(history.valid);
    auto now=before;now.depth=5; // Camera translated back one metre.
    CHECK_FALSE(denoiseCompatible(now,history,p)); // Old comparison mixed cameras.
    p.flags|=DENOISE_CURRENT_VIEW_DEPTH;p.currentViewDepthRow[2]=1;p.currentViewDepthRow[3]=-1;
    CHECK(denoiseCompatible(now,history,p));
    const auto accumulated=denoiseTemporal(now,glm::vec3(4),history,p,{},{});
    CHECK(accumulated.length==2);CHECK(accumulated.color[0]==doctest::Approx(3));
    SUBCASE("rotation also changes depth without changing the receiver") {
        p.currentViewDepthRow[0]=.5f;p.currentViewDepthRow[2]=std::sqrt(.75f);p.currentViewDepthRow[3]=0;
        now.depth=4*std::sqrt(.75f);CHECK(denoiseCompatible(now,history,p));
    }
    SUBCASE("a different depth or foreign point still rejects history") {
        now.depth=6;CHECK_FALSE(denoiseCompatible(now,history,p));
        now.depth=5;auto foreign=history;foreign.position[2]=-8;
        CHECK_FALSE(denoiseCompatible(now,foreign,p));
    }
    SUBCASE("invalid reprojection cannot make a valid history") {
        p.currentViewDepthRow[2]=std::numeric_limits<float>::quiet_NaN();
        CHECK_FALSE(denoiseCompatible(now,history,p));
    }
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
    DenoiseSettings settings;settings.clampHistory=true;const auto p=denoiseParameters(settings,4,4,DENOISE_SIGNAL_AO,0,1,1,false);const auto s=receiver();
    auto old=denoiseTemporal(s,glm::vec3(1),{},p,{},{});const glm::vec3 raw(.1f);
    const auto clipped=denoiseTemporal(s,raw,old,p,glm::vec3(.08f),glm::vec3(.12f));CHECK(clipped.color[0]<=.12f);CHECK(raw==glm::vec3(.1f));
    CHECK(denoiseTemporal(s,glm::vec3(7),{},p,{},{}).color[0]==1);
    auto invalid=s;invalid.valid=0;CHECK_FALSE(denoiseTemporal(invalid,raw,old,p,{},{ }).valid);
    settings.maxHistory=0;CHECK_FALSE(validDenoiseSettings(settings));
}
TEST_CASE("F13 baseline temporal accumulation preserves independent Bernoulli energy"){
    DenoiseSettings settings;REQUIRE_FALSE(settings.clampHistory);
    const auto s=receiver();const auto metadata=spec();
    for(u32 signal:{DENOISE_SIGNAL_DI,DENOISE_SIGNAL_GI,DENOISE_SIGNAL_SPECULAR,DENOISE_SIGNAL_AO}) {
        const auto p=denoiseParameters(settings,4,4,signal,0,1,1,false);
        REQUIRE((p.flags&DENOISE_CLAMP_HISTORY)==0);
        for(const float visibility:{0.1f,0.5f,0.9f}) {
            auto history=denoiseTemporal(s,glm::vec3(visibility),{},p,{},{},metadata);
            history.length=settings.maxHistory;history.firstMoment=visibility;
            history.secondMoment=visibility;history.variance=visibility*(1-visibility);
            double mean=0,biasedMean=0,mass=0;
            // Exact independent outcomes, no RNG, rendered images or fitted
            // reference. The centre shares the eight neighbors' probability.
            for(u32 bits=0;bits<512;++bits) {
                double probability=1;
                for(u32 k=0;k<9;++k)probability*=bits&(1u<<k)?visibility:1-visibility;
                const glm::vec3 low(bits==511?1.0f:0.0f),high(bits==0?0.0f:1.0f),raw(float(bits&1u));
                const auto next=denoiseTemporal(s,raw,history,p,low,high,metadata);
                CHECK(next.valid);CHECK(next.length==settings.maxHistory);
                mean+=probability*next.color[0];mass+=probability;
                auto experimental=p;experimental.flags|=DENOISE_CLAMP_HISTORY;
                biasedMean+=probability*denoiseTemporal(s,raw,history,experimental,low,high,metadata).color[0];
            }
            CHECK(mass==doctest::Approx(1).epsilon(1e-6));
            CHECK(mean==doctest::Approx(visibility).epsilon(1e-6));
            // Raw-extrema-only clipping already loses/gains this much. The
            // shader's extra sigma bound cannot rescue this all-equal outcome.
            if(visibility!=0.5f)CHECK(std::abs(biasedMean-visibility)>0.03);
        }
    }
}
TEST_CASE("F13 atrous edge weights preserve identity on constant signals and stop normal edges"){
    const auto p=denoiseParameters({},4,4,DENOISE_SIGNAL_DI,0,1,1,false);auto a=receiver(),b=a;
    CHECK(denoiseSpatialWeight(a,b,2,2,1,p)==doctest::Approx(1));b.geometricNormal[2]=-1;CHECK(denoiseSpatialWeight(a,b,2,2,1,p)==0);
    b=a;b.depth=30;CHECK(denoiseSpatialWeight(a,b,2,2,1,p)<.001f);
    CHECK(denoiseSpatialWeight(a,a,2,200,1,p)<.001f);
}

TEST_CASE("F13.6 specular atrous weight is bounded by the reflection lobe"){
    const auto p=denoiseParameters({},4,4,DENOISE_SIGNAL_SPECULAR,0,1,1,false);auto a=receiver();
    a.shadingNormal[0]=0;a.shadingNormal[1]=0;a.shadingNormal[2]=1;a.viewDirection[0]=0;a.viewDirection[1]=0;a.viewDirection[2]=1;
    auto b=a;const float angle=.003f;b.viewDirection[0]=std::sin(angle);b.viewDirection[2]=std::cos(angle); // next pixel on a flat mirror
    CHECK(denoiseSpatialWeight(a,a,2,2,1,p)==doctest::Approx(1));
    a.roughness=b.roughness=.04f; // alpha 0.002: the reflected image moves one lobe width per pixel
    CHECK(denoiseSpatialWeight(a,b,2,2,1,p)==doctest::Approx(std::exp(-(1-std::cos(angle))/(.002f*.002f))).epsilon(1e-3));
    CHECK(denoiseSpatialWeight(a,b,2,2,1,p)<.4f);
    auto c=a;c.viewDirection[0]=std::sin(2*angle);c.viewDirection[2]=std::cos(2*angle);CHECK(denoiseSpatialWeight(a,c,2,2,1,p)<.02f);
    a.roughness=b.roughness=.8f;  // wide lobe: unchanged filtering
    CHECK(denoiseSpatialWeight(a,b,2,2,1,p)>.99f);
    // DI keeps the normal guide: the view direction does not matter there.
    const auto d=denoiseParameters({},4,4,DENOISE_SIGNAL_DI,0,1,1,false);a.roughness=b.roughness=.04f;
    CHECK(denoiseSpatialWeight(a,b,2,2,1,d)==doctest::Approx(denoiseSpatialWeight(a,a,2,2,1,d)));
}

TEST_CASE("F13.6 soft change keeps history only inside the current neighbourhood"){
    DenoiseSettings settings;auto p=denoiseParameters(settings,4,4,DENOISE_SIGNAL_SPECULAR,0,1,1,false);const auto s=receiver();const auto sp=spec();
    auto h=denoiseTemporal(s,glm::vec3(1),{},p,{},{},sp);REQUIRE(h.valid);for(int i=0;i<8;++i)h=denoiseTemporal(s,glm::vec3(1),h,p,{},{},sp);
    const u32 length=h.length;REQUIRE(length>1);
    p.flags|=DENOISE_SOFT_CHANGE;
    auto kept=denoiseTemporal(s,glm::vec3(1.01f),h,p,{},{},sp,-1,glm::vec3(1.01f),glm::vec3(.05f));
    CHECK(kept.length==length+1);CHECK(kept.color[0]==doctest::Approx(1+.01f*2*settings.temporalAlpha).epsilon(1e-4));
    auto dropped=denoiseTemporal(s,glm::vec3(3),h,p,{},{},sp,-1,glm::vec3(3),glm::vec3(.05f));
    CHECK(dropped.length==1);CHECK(dropped.color[0]==doctest::Approx(3));
    p.flags&=~DENOISE_SOFT_CHANGE; // without the event a static history is never tested
    CHECK(denoiseTemporal(s,glm::vec3(3),h,p,{},{},sp,-1,glm::vec3(3),glm::vec3(.05f)).length==length+1);
}

TEST_CASE("Stochastic reflection outcomes preserve the mean and reduce independent sample variance") {
    DenoiseSettings settings;settings.temporalAlpha=settings.momentsAlpha=.01f;
    const auto p=denoiseParameters(settings,1,1,DENOISE_SIGNAL_SPECULAR,0,1,1,false);
    const auto s=receiver();double mean=0,second=0;
    // Exact enumeration: four IID fair outcomes, radiance0 or4. The true
    // mean is2, raw variance4; the four-sample average has variance1.
    for(u32 bits=0;bits<16;++bits){DenoiseHistory h{};
        for(u32 k=0;k<4;++k){auto m=spec();m.flags|=SPECULAR_SAMPLE_STOCHASTIC;
            const bool hit=(bits&(1u<<k))!=0;m.path=hit?REFLECTION_PATH_RT:REFLECTION_PATH_NONE;
            m.secondarySlot=hit?k:~0u;m.secondaryGeneration=k+1;m.hitDistance=hit?float(2+k):0;
            h=denoiseTemporal(s,glm::vec3(hit?4:0),h,p,{},{},m);REQUIRE(h.valid);CHECK(h.length==k+1);
        }
        mean+=h.color[0]/16.;second+=double(h.color[0])*h.color[0]/16.;
    }
    CHECK(mean==doctest::Approx(2));CHECK(second-mean*mean==doctest::Approx(1));
    auto m=spec();m.flags|=SPECULAR_SAMPLE_STOCHASTIC;
    const auto h=denoiseTemporal(s,glm::vec3(1),{},p,{},{},m);
    auto changed=m;changed.flags=SPECULAR_SAMPLE_VALID;CHECK_FALSE(denoiseCompatible(s,h,p,changed));
    changed=m;changed.flags|=SPECULAR_SAMPLE_ERROR;CHECK_FALSE(denoiseCompatible(s,h,p,changed));
    auto stale=p;++stale.signalRevision;CHECK_FALSE(denoiseCompatible(s,h,stale,m));
    stale=p;stale.flags|=DENOISE_RESET;CHECK_FALSE(denoiseCompatible(s,h,stale,m));
    auto wrong=h;wrong.flags|=1u;CHECK_FALSE(denoiseCompatible(s,wrong,p,m));
}
TEST_CASE("F13 compact denoise history round-trips every field a later frame reads"){
    // Budget closure: header + 64-byte record + 16-byte specular extension
    // replace the 112-byte record. Every value the temporal pass can write must
    // survive packing, and the reuse decision on the unpacked history must match.
    static_assert(sizeof(GPUDenoiseHistoryHeader)==64&&sizeof(GPUDenoiseHistoryRecord)==64&&sizeof(GPUDenoiseSpecularHistory)==16);
    CHECK(DenoiseSettings{}.maxHistory<=DENOISE_HISTORY_MAX_LENGTH);CHECK(REFLECTION_PATH_ENVIRONMENT<=DENOISE_HISTORY_MAX_PATH);
    CHECK((1u|DENOISE_HISTORY_STOCHASTIC)<=DENOISE_HISTORY_MAX_FLAGS);
    for(const u32 length:{0u,1u,2u,255u,256u})for(const u32 flags:{0u,1u,DENOISE_HISTORY_STOCHASTIC,1u|DENOISE_HISTORY_STOCHASTIC})
        for(u32 path=0;path<=REFLECTION_PATH_ENVIRONMENT;++path)for(const u32 valid:{0u,1u}){const u32 state=denoisePackState(length,flags,path,valid);
            CHECK(denoiseStateLength(state)==length);CHECK(denoiseStateFlags(state)==flags);CHECK(denoiseStatePath(state)==path);CHECK(denoiseStateValid(state)==valid);}
    CHECK(denoiseHistoryBytes(100,false)==64+100*64);CHECK(denoiseHistoryBytes(100,true)==64+100*80);CHECK(denoiseHistorySpecularOffset(100)==64+100*64);
    DenoiseSettings settings;const auto s=receiver();
    for(const u32 variant:{0u,1u,2u,3u,4u}){const u32 signal=variant==4?DENOISE_SIGNAL_SPECULAR:variant;
        const auto p=denoiseParameters(settings,4,4,signal,2,7,9,false);auto m=spec();if(variant==4)m.flags|=SPECULAR_SAMPLE_STOCHASTIC;
        DenoiseHistory h{};for(u32 i=0;i<5;++i)h=denoiseTemporal(s,glm::vec3(.25f*float(i+1)),h,p,{},{},m);REQUIRE(h.valid);REQUIRE(h.length==5);
        const bool specular=signal==DENOISE_SIGNAL_SPECULAR;
        const auto back=denoiseUnpack(denoisePackHeader(h),denoisePackRecord(h),specular?denoisePackSpecular(h):GPUDenoiseSpecularHistory{});
        for(u32 i=0;i<3;++i){CHECK(back.color[i]==h.color[i]);CHECK(back.normal[i]==h.normal[i]);CHECK(back.position[i]==h.position[i]);}
        CHECK(back.firstMoment==h.firstMoment);CHECK(back.secondMoment==h.secondMoment);CHECK(back.depth==h.depth);CHECK(back.length==h.length);
        CHECK(back.signal==h.signal);CHECK(back.viewID==h.viewID);CHECK(back.historyEpoch==h.historyEpoch);CHECK(back.signalRevision==h.signalRevision);
        CHECK(back.slot==h.slot);CHECK(back.instanceGeneration==h.instanceGeneration);CHECK(back.materialRevision==h.materialRevision);
        CHECK(back.valid==h.valid);CHECK(back.flags==h.flags);CHECK(back.variance==0);CHECK(h.variance>=settings.varianceFloor);
        if(specular){CHECK(back.roughness==h.roughness);CHECK(back.hitDistance==h.hitDistance);CHECK(back.path==h.path);
            CHECK(back.secondarySlot==h.secondarySlot);CHECK(back.secondaryGeneration==h.secondaryGeneration);}
        // Same reuse decision and the same next history from either form.
        CHECK(denoiseCompatible(s,back,p,m)==denoiseCompatible(s,h,p,m));CHECK(denoiseCompatible(s,back,p,m));
        const auto a=denoiseTemporal(s,glm::vec3(3),h,p,{},{},m),b=denoiseTemporal(s,glm::vec3(3),back,p,{},{},m);
        CHECK(a.color[0]==b.color[0]);CHECK(a.length==b.length);CHECK(a.variance==b.variance);CHECK(a.firstMoment==b.firstMoment);
        // The shared header carries the domain: a flipped view rejects the history.
        auto header=denoisePackHeader(h);header.viewID^=1u;
        CHECK_FALSE(denoiseCompatible(s,denoiseUnpack(header,denoisePackRecord(h),specular?denoisePackSpecular(h):GPUDenoiseSpecularHistory{}),p,m));
    }
}
TEST_CASE("F13.6 stationary specular history is a running mean up to its cap"){
    // DENOISE_STATIONARY (content and camera unchanged): alpha = 1/length, so a
    // static noisy signal converges like a plain average instead of an EMA
    // with the temporalAlpha floor; any other frame keeps the floor.
    DenoiseSettings settings;auto p=denoiseParameters(settings,1,1,DENOISE_SIGNAL_SPECULAR,0,1,1,false);p.maxHistory=128;
    auto still=p;still.flags|=DENOISE_STATIONARY;const auto s=receiver();auto m=spec();m.flags|=SPECULAR_SAMPLE_STOCHASTIC;
    DenoiseHistory a{},b{};double sum=0;
    for(u32 i=0;i<100;++i){const float value=float(i%7);sum+=value;a=denoiseTemporal(s,glm::vec3(value),a,still,{},{},m);b=denoiseTemporal(s,glm::vec3(value),b,p,{},{},m);}
    REQUIRE(a.valid);CHECK(a.length==100);CHECK(a.color[0]==doctest::Approx(sum/100).epsilon(1e-4));
    // The EMA keeps weighting recent samples: its value differs from the mean.
    CHECK(std::abs(b.color[0]-float(sum/100))>1e-3f);
    // The cap still bounds the length, and a reset restarts the mean.
    for(u32 i=0;i<100;++i)a=denoiseTemporal(s,glm::vec3(1),a,still,{},{},m);CHECK(a.length==128);
    auto cut=still;cut.flags|=DENOISE_RESET;CHECK(denoiseTemporal(s,glm::vec3(5),a,cut,{},{},m).length==1);
}
TEST_CASE("F13.6 stationary specular history weights frames by their sample count"){
    // A boosted record (SPECULAR_SAMPLE_COUNT) of four samples weighs four times
    // a single-sample frame, so the mean equals the per-sample mean.
    DenoiseSettings settings;auto p=denoiseParameters(settings,1,1,DENOISE_SIGNAL_SPECULAR,0,1,1,false);p.maxHistory=128;p.flags|=DENOISE_STATIONARY;
    const auto s=receiver();auto one=spec();one.flags|=SPECULAR_SAMPLE_STOCHASTIC;auto four=one;four.flags|=4u<<SPECULAR_SAMPLE_COUNT_SHIFT;
    auto h=denoiseTemporal(s,glm::vec3(2),{},p,{},{},four);CHECK(h.length==4);
    h=denoiseTemporal(s,glm::vec3(7),h,p,{},{},one);CHECK(h.length==5);CHECK(h.color[0]==doctest::Approx((4*2+7)/5.0));
    // Without the stationary flag the count is ignored (frame-weighted EMA).
    auto moving=p;moving.flags&=~DENOISE_STATIONARY;auto e=denoiseTemporal(s,glm::vec3(2),{},moving,{},{},four);CHECK(e.length==1);
}
TEST_CASE("F13.6 a specular history cap below the sample count keeps the weight a blend"){
    // --specular-max-history 2 with a four-sample boosted record: the length is
    // capped and the current frame replaces the history (alpha 1), never extrapolates.
    DenoiseSettings settings;auto p=denoiseParameters(settings,1,1,DENOISE_SIGNAL_SPECULAR,0,1,1,false);p.maxHistory=2;p.flags|=DENOISE_STATIONARY;
    const auto s=receiver();auto one=spec();one.flags|=SPECULAR_SAMPLE_STOCHASTIC;auto four=one;four.flags|=4u<<SPECULAR_SAMPLE_COUNT_SHIFT;
    auto h=denoiseTemporal(s,glm::vec3(2),{},p,{},{},one);h=denoiseTemporal(s,glm::vec3(4),h,p,{},{},one);CHECK(h.length==2);CHECK(h.color[0]==doctest::Approx(3));
    h=denoiseTemporal(s,glm::vec3(7),h,p,{},{},four);REQUIRE(h.valid);CHECK(h.length==2);CHECK(h.color[0]==doctest::Approx(7));
    // A single-sample frame under the cap blends normally.
    h=denoiseTemporal(s,glm::vec3(1),h,p,{},{},one);CHECK(h.length==2);CHECK(h.color[0]==doctest::Approx(4));
}

// F15.1 (ticket 08): the lobe contract of the specular denoiser. The base rule (null lobe pointers) is untouched.
namespace {
const u32 kCoat=specularSampleLobeFlags(SPECULAR_LOBE_CLEARCOAT,false),kRefract=specularSampleLobeFlags(SPECULAR_LOBE_TRANSMISSION,true);
GPUSpecularSample lobeSpec(u32 lobeFlags){auto s=spec();s.flags|=lobeFlags;return s;}
}
TEST_CASE("F15.1 specular lobe flags: key packing keeps the sample bits intact"){
    CHECK(specularSampleLobeKey(0)==0);CHECK(specularSampleLobeKey(SPECULAR_SAMPLE_VALID|SPECULAR_SAMPLE_STOCHASTIC|SPECULAR_SAMPLE_ERROR|(5u<<SPECULAR_SAMPLE_COUNT_SHIFT))==0);
    for(u32 id=0;id<4;++id)for(const bool refract:{false,true}){const u32 f=specularSampleLobeFlags(id,refract);
        CHECK(specularSampleLobeKey(f)==(id|(refract?4u:0u)));CHECK((f&SPECULAR_SAMPLE_LOBE_KEY_MASK)==f);
        // no overlap with the other sample bits (count bits 16-23, error bit 31, valid/prefiltered/mixed/stochastic bits 0-3)
        CHECK((f&(SPECULAR_SAMPLE_VALID|SPECULAR_SAMPLE_PREFILTERED|SPECULAR_SAMPLE_MIXED|SPECULAR_SAMPLE_STOCHASTIC|SPECULAR_SAMPLE_ERROR|(SPECULAR_SAMPLE_COUNT_MASK<<SPECULAR_SAMPLE_COUNT_SHIFT)))==0);}
    // the history state word carries the key without touching length, flags, path or valid
    for(u32 lobe=0;lobe<=DENOISE_HISTORY_MAX_LOBE;++lobe){const u32 state=denoisePackState(77,DENOISE_HISTORY_STOCHASTIC,REFLECTION_PATH_ENVIRONMENT,1,lobe);
        CHECK(denoiseStateLobe(state)==lobe);CHECK(denoiseStateLength(state)==77);CHECK(denoiseStateFlags(state)==DENOISE_HISTORY_STOCHASTIC);
        CHECK(denoiseStatePath(state)==REFLECTION_PATH_ENVIRONMENT);CHECK(denoiseStateValid(state)==1);}
}
TEST_CASE("F15.1 history is rejected across sampled lobe ids and axes, accepted for the same key"){
    const auto p=denoiseParameters({},4,4,DENOISE_SIGNAL_SPECULAR,0,1,1,false);const auto s=receiver();DenoiseLobeGuide guide;guide.ior=1.5f;guide.clearcoatRoughness=.2f;
    const u32 keys[]={0u,kCoat,kRefract,specularSampleLobeFlags(SPECULAR_LOBE_BASE,true),specularSampleLobeFlags(SPECULAR_LOBE_CLEARCOAT,true)};
    for(const u32 flags:keys){
        const auto sample=lobeSpec(flags);const auto h=denoiseTemporal(s,glm::vec3(1),{},p,{},{},sample,-1,glm::vec3(0),glm::vec3(-1),&guide);REQUIRE(h.valid);
        CHECK(h.lobe==specularSampleLobeKey(flags));
        CHECK(denoiseCompatible(s,h,p,sample,&guide));
        for(const u32 other:keys){
            const bool same=specularSampleLobeKey(other)==specularSampleLobeKey(flags);
            CHECK(denoiseCompatible(s,h,p,lobeSpec(other),&guide)==same);}
        // the packed record keeps the key
        const auto back=denoiseUnpack(denoisePackHeader(h),denoisePackRecord(h),denoisePackSpecular(h));CHECK(back.lobe==h.lobe);CHECK(denoiseCompatible(s,back,p,sample,&guide));
    }
    // the chain: accumulation continues across frames of one lobe and restarts at the first frame of another
    auto h=denoiseTemporal(s,glm::vec3(1),{},p,{},{},lobeSpec(0),-1,glm::vec3(0),glm::vec3(-1),&guide);
    h=denoiseTemporal(s,glm::vec3(1),h,p,{},{},lobeSpec(0),-1,glm::vec3(0),glm::vec3(-1),&guide);CHECK(h.length==2);
    h=denoiseTemporal(s,glm::vec3(1),h,p,{},{},lobeSpec(kCoat),-1,glm::vec3(0),glm::vec3(-1),&guide);CHECK(h.length==1);CHECK(h.lobe==specularSampleLobeKey(kCoat));
    // the base rule (no lobe pointer) ignores the key: byte-identical behaviour for scenes without features
    const auto base=denoiseTemporal(s,glm::vec3(1),{},p,{},{},lobeSpec(kCoat));CHECK(base.lobe==0);CHECK(denoiseCompatible(s,base,p,lobeSpec(0)));CHECK(denoiseCompatible(s,base,p,lobeSpec(kRefract)));
    // other signals never carry a lobe
    const auto di=denoiseParameters({},4,4,DENOISE_SIGNAL_DI,0,1,1,false);const auto hd=denoiseTemporal(s,glm::vec3(1),{},di,{},{},lobeSpec(kCoat),-1,glm::vec3(0),glm::vec3(-1),&guide);
    CHECK(hd.lobe==0);CHECK(denoiseCompatible(s,hd,di,lobeSpec(kRefract),&guide));
}
TEST_CASE("F15.1 neighbours are rejected across sampled lobe ids and axes"){
    const auto p=denoiseParameters({},4,4,DENOISE_SIGNAL_SPECULAR,0,1,1,false);auto a=receiver();a.roughness=.8f;a.viewDirection[2]=1;auto b=a;
    DenoiseLobeGuide guide;guide.ior=1.5f;guide.clearcoatRoughness=.8f;
    DenoiseLobePixel base,coat,refract,baseRefract;coat.sampleFlags=kCoat;refract.sampleFlags=kRefract;baseRefract.sampleFlags=specularSampleLobeFlags(SPECULAR_LOBE_BASE,true);
    for(auto* l:{&base,&coat,&refract,&baseRefract})l->guide=guide;
    CHECK(denoiseSpatialWeight(a,b,2,2,1,p,&base,&base)>.99f);CHECK(denoiseSpatialWeight(a,b,2,2,1,p,&coat,&coat)>.99f);CHECK(denoiseSpatialWeight(a,b,2,2,1,p,&refract,&refract)>.99f);
    CHECK(denoiseSpatialWeight(a,b,2,2,1,p,&base,&coat)==0);CHECK(denoiseSpatialWeight(a,b,2,2,1,p,&coat,&base)==0);CHECK(denoiseSpatialWeight(a,b,2,2,1,p,&base,&refract)==0);
    CHECK(denoiseSpatialWeight(a,b,2,2,1,p,&base,&baseRefract)==0); // same id, other axis
    CHECK(denoiseSpatialWeight(a,b,2,2,1,p,&coat,&refract)==0);
    // null lobe pointers: the base rule, identical numbers
    CHECK(denoiseSpatialWeight(a,b,2,2,1,p)==doctest::Approx(denoiseSpatialWeight(a,b,2,2,1,p,&base,&base)));
    // DI never uses lobes
    const auto d=denoiseParameters({},4,4,DENOISE_SIGNAL_DI,0,1,1,false);CHECK(denoiseSpatialWeight(a,b,2,2,1,d,&base,&coat)>.99f);
}
TEST_CASE("F15.1 the edge-stopping roughness follows the sampled lobe"){
    DenoiseLobeGuide guide;guide.ior=1.5f;guide.clearcoatRoughness=.04f;
    CHECK(denoiseLobeRoughness(.6f,0,guide)==doctest::Approx(.6f));
    CHECK(denoiseLobeRoughness(.6f,SPECULAR_LOBE_CLEARCOAT,guide)==doctest::Approx(.04f));
    CHECK(denoiseLobeRoughness(.6f,SPECULAR_LOBE_TRANSMISSION|4u,guide)==doctest::Approx(.6f)); // ior 1.5: factor 1
    guide.ior=1.25f;CHECK(denoiseLobeRoughness(.6f,SPECULAR_LOBE_TRANSMISSION|4u,guide)==doctest::Approx(.3f));
    guide.ior=1.0f;CHECK(denoiseLobeRoughness(.6f,SPECULAR_LOBE_TRANSMISSION|4u,guide)==doctest::Approx(0.f));
    guide.ior=0;CHECK(denoiseLobeRoughness(.6f,SPECULAR_LOBE_TRANSMISSION|4u,guide)==doctest::Approx(.6f)); // special mode: default ior
    CHECK(denoiseLobeRoughness(.6f,3,guide)==doctest::Approx(.6f));
    // the weight bound of a pair of reflection axes uses the centre pixel's LOBE roughness: a smooth coat lobe on a rough base
    // keeps neighbours with a slightly different reflection axis apart, the rough base lobe lets them through.
    const auto p=denoiseParameters({},4,4,DENOISE_SIGNAL_SPECULAR,0,1,1,false);auto a=receiver();a.roughness=.8f;a.viewDirection[2]=1;auto b=a;
    const float angle=.003f;b.viewDirection[0]=std::sin(angle);b.viewDirection[2]=std::cos(angle);
    DenoiseLobeGuide g;g.ior=1.5f;g.clearcoatRoughness=.04f;DenoiseLobePixel base,coat;coat.sampleFlags=kCoat;base.guide=coat.guide=g;
    CHECK(denoiseSpatialWeight(a,b,2,2,1,p,&base,&base)>.99f);
    CHECK(denoiseSpatialWeight(a,b,2,2,1,p,&coat,&coat)<.4f);
    // the history stores the lobe roughness and compares the current lobe roughness against it
    const auto s=receiver();auto rs=s;rs.roughness=.8f;const auto h=denoiseTemporal(rs,glm::vec3(1),{},p,{},{},lobeSpec(kCoat),-1,glm::vec3(0),glm::vec3(-1),&g);
    CHECK(h.roughness==doctest::Approx(.04f));CHECK(denoiseCompatible(rs,h,p,lobeSpec(kCoat),&g));
    auto g2=g;g2.clearcoatRoughness=.5f;CHECK_FALSE(denoiseCompatible(rs,h,p,lobeSpec(kCoat),&g2)); // the coat roughness changed beyond the threshold
}
TEST_CASE("F15.1 ticket 09: a significant coat takes the sharper of the coat and base roughness"){
    DenoiseLobeGuide g;g.ior=1.5f;
    g.clearcoatRoughness=.8f;CHECK(denoiseLobeRoughness(.1f,SPECULAR_LOBE_CLEARCOAT,g)==doctest::Approx(.1f)); // a rough coat over a smooth base: the base lobe is the sharper
    g.clearcoatRoughness=.1f;CHECK(denoiseLobeRoughness(.8f,SPECULAR_LOBE_CLEARCOAT,g)==doctest::Approx(.1f)); // the smooth coat over a rough base keeps its reflection
    g.clearcoatRoughness=.3f;CHECK(denoiseLobeRoughness(.3f,SPECULAR_LOBE_CLEARCOAT,g)==doctest::Approx(.3f));
    CHECK(denoiseLobeRoughness(.8f,0,g)==doctest::Approx(.8f)); // an insignificant coat: the key is the base lobe, the base roughness
    g.clearcoatRoughness=NAN;CHECK(denoiseLobeRoughness(.8f,SPECULAR_LOBE_CLEARCOAT,g)==doctest::Approx(.8f));
}
TEST_CASE("F15.1 ticket 09: the stable key keeps the history of a mixed-sample pixel"){
    // The transport picks the coat or the base lobe per sample, but the key it records is a pure function of the pixel: the
    // history of the pixel is accepted frame after frame (no reset), where a per-sample key would reject about half of the frames.
    const auto p=denoiseParameters({},4,4,DENOISE_SIGNAL_SPECULAR,0,1,1,false);auto s=receiver();s.roughness=.8f;
    DenoiseLobeGuide g;g.ior=1.5f;g.clearcoatRoughness=.05f;
    DenoiseHistory h{};u32 length=0;
    for(int frame=0;frame<8;++frame){h=denoiseTemporal(s,glm::vec3(1),h,p,{},{},lobeSpec(kCoat),-1,glm::vec3(0),glm::vec3(-1),&g);length=h.length;}
    CHECK(length==8);CHECK(h.lobe==specularSampleLobeKey(kCoat));CHECK(h.roughness==doctest::Approx(.05f));
    // the contrast: alternating sampled keys (the per-sample key) reset the history at every switch
    h=DenoiseHistory{};
    for(int frame=0;frame<8;++frame)h=denoiseTemporal(s,glm::vec3(1),h,p,{},{},lobeSpec(frame&1?kCoat:0u),-1,glm::vec3(0),glm::vec3(-1),&g);
    CHECK(h.length==1);
}
TEST_CASE("F15.1 (ticket 13) a thin-walled surface keeps the refract axis straight along -V"){
    auto s=receiver();s.shadingNormal[0]=0;s.shadingNormal[1]=0;s.shadingNormal[2]=1;
    const float angle=.5f;s.viewDirection[0]=std::sin(angle);s.viewDirection[1]=0;s.viewDirection[2]=std::cos(angle);
    DenoiseLobeGuide g;g.ior=1.5f;g.volumeThickness=0;
    const auto thin=denoiseLobeAxis(s,4,g);
    CHECK(thin.x==doctest::Approx(-std::sin(angle)));CHECK(thin.z==doctest::Approx(-std::cos(angle))); // the ray goes through unbent: -V
    g.volumeThickness=.8f;const auto volume=denoiseLobeAxis(s,4,g);
    CHECK(volume.x>thin.x);                                                                                // the volume bends it towards the normal
    CHECK(glm::dot(thin,volume)<.999f);
}
TEST_CASE("F15.1 the refract axis follows the refracted direction, not the reflection axis"){
    auto s=receiver();s.shadingNormal[0]=0;s.shadingNormal[1]=0;s.shadingNormal[2]=1;
    const float angle=.5f;s.viewDirection[0]=std::sin(angle);s.viewDirection[1]=0;s.viewDirection[2]=std::cos(angle);
    DenoiseLobeGuide g;g.ior=1.5f;
    const auto reflect=denoiseLobeAxis(s,0,g),refract=denoiseLobeAxis(s,4,g);
    // reflect(-V, N) mirrors V about N; the refracted ray continues through the surface: Snell with eta = 1/1.5.
    CHECK(reflect.x==doctest::Approx(-std::sin(angle)));CHECK(reflect.z==doctest::Approx(std::cos(angle)));
    CHECK(refract.z<0);CHECK(glm::length(refract)==doctest::Approx(1));
    CHECK(std::sin(angle)==doctest::Approx(1.5f*std::abs(refract.x)).epsilon(1e-4)); // Snell: sin(theta_i) = ior * sin(theta_t)
    CHECK(glm::dot(reflect,refract)<.5f);
    // total internal reflection falls back to the reflection axis
    DenoiseLobeGuide inverted;inverted.ior=1.0f;auto shallow=s;shallow.viewDirection[0]=std::sin(1.5f);shallow.viewDirection[2]=std::cos(1.5f);
    CHECK(glm::length(denoiseLobeAxis(shallow,4,inverted))==doctest::Approx(1));
    // The clip control "reflect axis used for a refract-axis lobe" is a filter with the wrong axis: for the same pair of pixels the two
    // axes give different weights, so the wrong axis changes the filter (ticket 13 runs the clip control once refraction exists).
    auto a=s;a.roughness=.04f;auto b=a;const float d=.003f;b.viewDirection[0]=std::sin(angle+d);b.viewDirection[2]=std::cos(angle+d);
    const auto p=denoiseParameters({},4,4,DENOISE_SIGNAL_SPECULAR,0,1,1,false);DenoiseLobePixel viaRefract,viaReflect;viaRefract.sampleFlags=specularSampleLobeFlags(SPECULAR_LOBE_BASE,true);viaRefract.guide=viaReflect.guide=g;
    const float wRefract=denoiseSpatialWeight(a,b,2,2,1,p,&viaRefract,&viaRefract),wReflect=denoiseSpatialWeight(a,b,2,2,1,p,&viaReflect,&viaReflect);
    CHECK(wRefract>wReflect*1.3f); // the refracted image moves slower per pixel than the reflected one
    CHECK(denoiseLobeAxis(a,4,g).x!=denoiseLobeAxis(a,0,g).x);
}

TEST_CASE("F15.1 ticket 11: an anisotropic pixel is filtered at the roughness of its sharper axis"){
    DenoiseLobeGuide g;g.ior=1.5f;
    for(const float r:{.2f,.3f,.6f,1.f})for(const float k:{0.f,.25f,.5f,1.f}){
        // alpha_t >= alpha_b: the sharper axis is the perpendicular one, the base roughness; the base-lobe key returns the same value.
        CHECK(denoiseAnisotropicRoughness(r,k)==doctest::Approx(r).epsilon(1e-5));
        CHECK(denoiseLobeRoughness(r,0,g)==doctest::Approx(denoiseAnisotropicRoughness(r,k)).epsilon(1e-5));
    }
    CHECK(denoiseAnisotropicRoughness(.01f,1.f)==doctest::Approx(std::sqrt(.002f)).epsilon(1e-5)); // the alpha floor
    // The rejected alternative, the effective roughness sqrt(sqrt((alpha_t^2 + alpha_b^2) / 2)), is rougher than the sharp axis for k > 0:
    const float r=.3f,k=1.f,ab=r*r,at=ab+(1-ab)*k*k,eff=std::sqrt(std::sqrt(.5f*(at*at+ab*ab)));
    CHECK(eff>denoiseAnisotropicRoughness(r,k)*1.5f);
    // With the rule the lobe bound of a pair of reflection axes is the one of the sharp axis (a stretched-axis bound would let them through).
    const auto p=denoiseParameters({},4,4,DENOISE_SIGNAL_SPECULAR,0,1,1,false);auto a=receiver();a.roughness=.3f;a.viewDirection[2]=1;auto b=a;
    const float angle=.12f;b.viewDirection[0]=std::sin(angle);b.viewDirection[2]=std::cos(angle);
    DenoiseLobePixel base;base.guide=g;
    const float sharp=denoiseSpatialWeight(a,b,2,2,1,p,&base,&base);
    auto wide=a;wide.roughness=eff;auto wideB=b;wideB.roughness=eff;
    CHECK(denoiseSpatialWeight(wide,wideB,2,2,1,p,&base,&base)>sharp);
}
TEST_CASE("F15.1 (ticket 13) a transmissive pixel has one key whichever branch of the reflect/refract estimator was drawn"){
    const auto p=denoiseParameters({},4,4,DENOISE_SIGNAL_SPECULAR,0,1,1,false);const auto s=receiver();
    DenoiseLobeGuide glass;glass.ior=1.5f;glass.transmissionFactor=1.0f;
    const u32 reflectBranch=0u,refractBranch=kRefract;
    CHECK(denoiseStableLobeKey(specularSampleLobeKey(reflectBranch),glass)==(SPECULAR_LOBE_TRANSMISSION|4u));
    CHECK(denoiseStableLobeKey(specularSampleLobeKey(refractBranch),glass)==(SPECULAR_LOBE_TRANSMISSION|4u));
    // history survives a branch change in either direction
    const auto fromRefract=denoiseTemporal(s,glm::vec3(1),{},p,{},{},lobeSpec(refractBranch),-1,glm::vec3(0),glm::vec3(-1),&glass);REQUIRE(fromRefract.valid);
    CHECK(denoiseCompatible(s,fromRefract,p,lobeSpec(reflectBranch),&glass));
    const auto fromReflect=denoiseTemporal(s,glm::vec3(1),{},p,{},{},lobeSpec(reflectBranch),-1,glm::vec3(0),glm::vec3(-1),&glass);
    CHECK(fromReflect.lobe==fromRefract.lobe);CHECK(denoiseCompatible(s,fromReflect,p,lobeSpec(refractBranch),&glass));
    auto h=denoiseTemporal(s,glm::vec3(1),fromRefract,p,{},{},lobeSpec(reflectBranch),-1,glm::vec3(0),glm::vec3(-1),&glass);CHECK(h.length==2);
    // neighbours of different branches filter together (same axis and roughness: the refracted image)
    DenoiseLobePixel a,b;a.guide=b.guide=glass;a.sampleFlags=reflectBranch|SPECULAR_SAMPLE_VALID;b.sampleFlags=refractBranch|SPECULAR_SAMPLE_VALID;
    auto sv=s;sv.viewDirection[2]=1;CHECK(denoiseSpatialWeight(sv,sv,.5f,.5f,.1f,p,&a,&b)>.99f);
    // a pixel below the factor threshold, or without a guide factor, keeps the sampled key (the clearcoat and base rules are untouched)
    DenoiseLobeGuide faint=glass;faint.transmissionFactor=.25f;
    CHECK(denoiseStableLobeKey(specularSampleLobeKey(reflectBranch),faint)==0u);CHECK(denoiseStableLobeKey(specularSampleLobeKey(refractBranch),faint)==(SPECULAR_LOBE_TRANSMISSION|4u));
    CHECK(denoiseStableLobeKey(specularSampleLobeKey(kCoat),DenoiseLobeGuide{})==SPECULAR_LOBE_CLEARCOAT);
    DenoiseLobeGuide nan=glass;nan.transmissionFactor=std::nanf("");CHECK(denoiseStableLobeKey(0u,nan)==0u);
}
