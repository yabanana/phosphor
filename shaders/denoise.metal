#include "restir_common.h"

// Each view AND signal owns previous/next history buffers. Never share DI,GI,
// SPECULAR or AO histories even if dimensions/formats coincide. Temporal output
// is stored as history; final atrous output is NOT temporal feedback.
// params0 surfaces1 previousHistory2 optionalSpecMetadata3 nextHistory4;
// textures raw0 pre-resolvePixelMotion1 temporalOUT2 momentsVarianceOUT3.
inline float3 denoiseNormal(GPUDISurface s,uint signal){return signal==DENOISE_SIGNAL_SPECULAR?diVec(s.shadingNormal):diVec(s.geometricNormal);}
inline bool denoiseCompatible(GPUDISurface s,GPUDenoiseHistory h,constant GPUDenoiseParams& p,GPUSpecularSample spec){
    if((p.flags&DENOISE_RESET)||!s.valid||!h.valid||!h.length||h.signal!=p.signal||h.viewID!=p.viewID||h.historyEpoch!=p.historyEpoch||h.signalRevision!=p.signalRevision||
        h.slot!=s.instanceSlot||h.instanceGeneration!=s.instanceGeneration||h.materialRevision!=s.materialRevision||
        !isfinite(s.depth)||!isfinite(h.depth)||s.depth<=0||h.depth<=0||!all(isfinite(diVec(h.color)))||
        !isfinite(h.firstMoment)||!isfinite(h.secondMoment)||!isfinite(h.variance)||h.variance<0)return false;
    const float3 n=denoiseNormal(s,p.signal),old=diVec(h.normal);if(!all(isfinite(n))||!all(isfinite(old))||dot(n,n)<=0||dot(old,old)<=0||dot(normalize(n),normalize(old))<p.normalThreshold)return false;
    const float scale=max(s.depth,h.depth);if(abs(s.depth-h.depth)>p.depthThreshold*scale||!all(isfinite(diVec(s.position)))||!all(isfinite(diVec(h.position)))||
        abs(dot(diVec(s.position)-diVec(h.position),normalize(n)))>p.planeThreshold*scale)return false;
    if(p.signal==DENOISE_SIGNAL_SPECULAR){if(!(spec.flags&SPECULAR_SAMPLE_VALID)||h.path!=spec.path||h.secondarySlot!=spec.secondarySlot||h.secondaryGeneration!=spec.secondaryGeneration||
        !isfinite(h.roughness)||!isfinite(s.roughness)||abs(h.roughness-s.roughness)>p.roughnessThreshold||!isfinite(h.hitDistance)||!isfinite(spec.hitDistance)||
        abs(h.hitDistance-spec.hitDistance)>p.hitDistanceThreshold*max(1.0f,max(h.hitDistance,spec.hitDistance)))return false;}
    return true;
}
inline bool denoiseNeighbor(GPUDISurface a,GPUDISurface b,constant GPUDenoiseParams& p){
    const float3 n=denoiseNormal(a,p.signal),m=denoiseNormal(b,p.signal);return a.valid&&b.valid&&all(isfinite(n))&&all(isfinite(m))&&dot(n,n)>0&&dot(m,m)>0&&
        dot(normalize(n),normalize(m))>=p.normalThreshold&&abs(a.depth-b.depth)<=p.depthThreshold*max(a.depth,b.depth);
}
kernel void denoise_temporal(constant GPUDenoiseParams& p [[buffer(0)]],const device GPUDISurface* surfaces [[buffer(1)]],
    const device GPUDenoiseHistory* previous [[buffer(2)]],const device GPUSpecularSample* specular [[buffer(3)]],device GPUDenoiseHistory* next [[buffer(4)]],
    texture2d<float,access::read> raw [[texture(0)]],texture2d<float,access::read> motion [[texture(1)]],
    texture2d<float,access::write> output [[texture(2)]],texture2d<float,access::write> moments [[texture(3)]],uint tid [[thread_position_in_grid]]){
    if(tid>=p.width*p.height)return;const uint2 pixel(tid%p.width,tid/p.width);const GPUDISurface s=surfaces[tid];
    GPUDenoiseHistory h{};h.signal=p.signal;h.viewID=p.viewID;h.historyEpoch=p.historyEpoch;h.signalRevision=p.signalRevision;
    float3 current=raw.read(pixel).rgb;const float3 normal=denoiseNormal(s,p.signal);GPUSpecularSample spec{};
    if(p.signal==DENOISE_SIGNAL_SPECULAR)spec=specular[tid];
    if(s.valid&&(!all(isfinite(current))||!all(isfinite(normal))||dot(normal,normal)<=0))h.flags=1u;
    if(s.valid&&all(isfinite(current))&&all(isfinite(normal))&&dot(normal,normal)>0){current=max(current,0.0f);if(p.signal==DENOISE_SIGNAL_AO)current=saturate(current);
        GPUDenoiseHistory old{};bool reuse=false;
        if(!(p.flags&DENOISE_RESET)){const float2 delta=motion.read(pixel).xy,center=float2(pixel)+.5f+delta;
            if(all(isfinite(delta))&&all(center>=0)&&all(center<float2(p.width,p.height))){const uint2 q=uint2(center);old=previous[q.y*p.width+q.x];reuse=denoiseCompatible(s,old,p,spec);}}
        float3 oldColor=diVec(old.color);
        float localVariance=0;
        if(!reuse||old.length<p.minHistoryForVariance||(p.flags&DENOISE_CLAMP_HISTORY)){float3 sum=0,square=0,lo=INFINITY,hi=-INFINITY;float count=0,lumSum=0,lumSquare=0;
            for(int y=-1;y<=1;++y)for(int x=-1;x<=1;++x){const int2 q=int2(pixel)+int2(x,y);if(any(q<0)||any(q>=int2(p.width,p.height)))continue;
                const uint index=uint(q.y)*p.width+uint(q.x);if(!denoiseNeighbor(s,surfaces[index],p))continue;const float3 value=raw.read(uint2(q)).rgb;
                if(!all(isfinite(value)))continue;sum+=value;square+=value*value;lo=min(lo,value);hi=max(hi,value);const float l=diLuminance(value);lumSum+=l;lumSquare+=l*l;++count;}
            if(count>0){const float3 mean=sum/count,variance=max(square/count-mean*mean,0.0f),sigma=sqrt(variance+p.varianceFloor);
                localVariance=max(0.0f,lumSquare/count-(lumSum/count)*(lumSum/count));
                const float3 low=max(lo,mean-p.clampSigma*sigma),high=min(hi,mean+p.clampSigma*sigma);if(reuse&&(p.flags&DENOISE_CLAMP_HISTORY))oldColor=clamp(oldColor,low,high);}
        }
        const uint length=reuse?min(old.length,p.maxHistory-1u)+1u:1u;const float alpha=reuse?max(p.temporalAlpha,1.0f/float(length)):1;
        const float ma=reuse?max(p.momentsAlpha,1.0f/float(length)):1,L=diLuminance(current);float3 color=reuse?mix(oldColor,current,alpha):current;if(p.signal==DENOISE_SIGNAL_AO)color=saturate(color);
        h.length=length;h.firstMoment=reuse?mix(old.firstMoment,L,ma):L;h.secondMoment=reuse?mix(old.secondMoment,L*L,ma):L*L;
        h.variance=max(p.varianceFloor,h.secondMoment-h.firstMoment*h.firstMoment);if(length<p.minHistoryForVariance)h.variance=max(h.variance,localVariance/float(length));h.depth=s.depth;h.roughness=s.roughness;
        h.slot=s.instanceSlot;h.instanceGeneration=s.instanceGeneration;h.materialRevision=s.materialRevision;
        h.hitDistance=spec.hitDistance;h.path=spec.path;h.secondarySlot=spec.secondarySlot;h.secondaryGeneration=spec.secondaryGeneration;
        for(uint i=0;i<3;++i){h.color[i]=color[i];h.position[i]=s.position[i];h.normal[i]=normalize(normal)[i];}
        h.valid=all(isfinite(color))&&isfinite(h.variance);if(!h.valid)h.flags=1u;
    }
    next[tid]=h;output.write(float4(h.color[0],h.color[1],h.color[2],1),pixel);
    moments.write(float4(h.firstMoment,h.secondMoment,h.variance,float(h.length)),pixel);
}

inline float denoiseWeight(GPUDISurface a,GPUDISurface b,float la,float lb,float variance,constant GPUDenoiseParams& p){
    if(!a.valid||!b.valid)return 0;const float3 na=denoiseNormal(a,p.signal),nb=denoiseNormal(b,p.signal);if(dot(na,na)<=0||dot(nb,nb)<=0)return 0;
    const float normal=pow(max(0.0f,dot(normalize(na),normalize(nb))),p.normalPhi);
    const float depth=exp(-abs(a.depth-b.depth)/max(1e-6f,p.depthThreshold*max(a.depth,b.depth)*float(p.atrousStep)));
    const float color=exp(-abs(la-lb)/(p.luminancePhi*sqrt(max(variance,p.varianceFloor))+1e-6f));
    const float rough=p.signal==DENOISE_SIGNAL_SPECULAR?exp(-abs(a.roughness-b.roughness)/max(p.roughnessThreshold,1e-6f)):1;
    return normal*depth*color*rough;
}
// params0/surfaces1/optionalSpecMetadata3. Textures input0, moments1, output2.
// One pass per stride1/2/4 (up to16), all separate resources/graph versions.
// Output variance is not fed into temporal raw/reference moments.
kernel void denoise_atrous(constant GPUDenoiseParams& p [[buffer(0)]],const device GPUDISurface* surfaces [[buffer(1)]],
    const device GPUSpecularSample* specular [[buffer(3)]],texture2d<float,access::read> input [[texture(0)]],
    texture2d<float,access::read> moments [[texture(1)]],texture2d<float,access::write> output [[texture(2)]],uint tid [[thread_position_in_grid]]){
    if(tid>=p.width*p.height)return;const uint2 pixel(tid%p.width,tid/p.width);const GPUDISurface center=surfaces[tid];const float3 value=input.read(pixel).rgb;
    if(!center.valid){output.write(float4(value,1),pixel);return;}
    const float variance=moments.read(pixel).z,lum=diLuminance(value);float3 sum=0;float weights=0;const float taps[5]={1,4,6,4,1};
    for(int y=-2;y<=2;++y)for(int x=-2;x<=2;++x){const int2 q=int2(pixel)+int2(x,y)*int(p.atrousStep);if(any(q<0)||any(q>=int2(p.width,p.height)))continue;
        const uint index=uint(q.y)*p.width+uint(q.x);const float3 color=input.read(uint2(q)).rgb;if(!all(isfinite(color)))continue;
        if(p.signal==DENOISE_SIGNAL_SPECULAR){const GPUSpecularSample a=specular[tid],b=specular[index];if(a.path!=b.path||a.secondarySlot!=b.secondarySlot||a.secondaryGeneration!=b.secondaryGeneration)continue;}
        const float weight=taps[x+2]*taps[y+2]*denoiseWeight(center,surfaces[index],lum,diLuminance(color),variance,p);sum+=color*weight;weights+=weight;
    }
    float3 result=weights>0?sum/weights:value;if(p.signal==DENOISE_SIGNAL_AO)result=saturate(result);output.write(float4(result,1),pixel);
}

// History corruption control: wrong view ID without setting an error flag.
// Independent compatible() rejects it and resets length to1 at the next frame.
kernel void denoise_corrupt_history_view(constant GPUDenoiseParams& p [[buffer(0)]],device GPUDenoiseHistory* history [[buffer(1)]],uint tid [[thread_position_in_grid]]){
    if(tid<p.width*p.height&&history[tid].valid)history[tid].viewID^=1u;
}

// Always-on checker: params0,nextHistory1,atomicCounts2; outputtexture0.
// Eight cleared words: pixels0,valid1,invalid state/domain2,invalid output3,
// reused valid histories4,max valid length5,actual GPU params view6/frame7.
// Raw/reference checks remain independent. One-dimensional exact pixel dispatch.
kernel void denoise_check(constant GPUDenoiseParams& p [[buffer(0)]],const device GPUDenoiseHistory* history [[buffer(1)]],
    device atomic_uint* counts [[buffer(2)]],texture2d<float,access::read> output [[texture(0)]],uint tid [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){
    if(tid>=p.width*p.height)return;const GPUDenoiseHistory h=history[tid];bool invalid=h.flags!=0;
    if(h.valid)invalid|=h.signal!=p.signal||h.viewID!=p.viewID||h.historyEpoch!=p.historyEpoch||h.signalRevision!=p.signalRevision||
        h.length<1||h.length>p.maxHistory||!isfinite(h.firstMoment)||!isfinite(h.secondMoment)||!isfinite(h.variance)||h.variance<0||!all(isfinite(diVec(h.color)));
    const float3 color=output.read(uint2(tid%p.width,tid/p.width)).rgb;const bool badOutput=!all(isfinite(color))||any(color<0)||(p.signal==DENOISE_SIGNAL_AO&&any(color>1));
    const uint values[5]={1u,h.valid?1u:0u,invalid?1u:0u,badOutput?1u:0u,h.valid&&h.length>1u?1u:0u};
    for(uint i=0;i<5;++i){const uint sum=simd_sum(values[i]);if(lane==0&&sum)atomic_fetch_add_explicit(counts+i,sum,memory_order_relaxed);}
    const uint longest=simd_max(h.valid?h.length:0u);if(lane==0&&longest)atomic_fetch_max_explicit(counts+5,longest,memory_order_relaxed);
    if(tid==0){atomic_store_explicit(counts+6,p.viewID,memory_order_relaxed);atomic_store_explicit(counts+7,p.frameIndex,memory_order_relaxed);}
}
