#include "restir_common.h"
#include "renderer/denoise_history_layout.h"

constant uint FC_DENOISE_SIGNAL [[function_constant(40)]];
constant uint kDenoiseSignal=is_function_constant_defined(FC_DENOISE_SIGNAL)?FC_DENOISE_SIGNAL:0xffffffffu;
inline uint denoiseSignal(constant GPUDenoiseParams& p){return kDenoiseSignal==0xffffffffu?p.signal:kDenoiseSignal;}

// Each view AND signal owns previous/next history buffers. Never share DI,GI,
// SPECULAR or AO histories even if dimensions/formats coincide. Temporal output
// is stored as history; final atrous output is NOT temporal feedback.
// params0 surfaces1 previousRecords2 optionalSpecMetadata3 nextRecords4
// previousHeader5 nextHeader6 previousSpecular7 nextSpecular8 (one history
// buffer per side, denoise_history_layout.h; the extension only for specular);
// textures raw0 pre-resolvePixelMotion1 temporalOUT2 momentsVarianceOUT3.
inline float3 denoiseNormal(GPUDISurface s,uint signal){return signal==DENOISE_SIGNAL_SPECULAR?diVec(s.shadingNormal):diVec(s.geometricNormal);}
// Spatial filter guide. Specular radiance varies with the reflection axis
// R = reflect(-V, N), not with N alone: on a flat mirror N is constant while
// the reflected image moves one pixel per pixel. Other signals keep N.
inline float3 denoiseGuide(GPUDISurface s,uint signal){
    const float3 n=denoiseNormal(s,signal);if(signal!=DENOISE_SIGNAL_SPECULAR)return n;
    const float3 v=diVec(s.viewDirection);
    return (dot(n,n)>0&&dot(v,v)>0&&all(isfinite(n))&&all(isfinite(v)))?reflect(-normalize(v),normalize(n)):float3(0);
}
// F13.6 lobe bound: neighbours whose reflection axes differ by more than the
// centre's GGX lobe (angle ~ alpha) see a different reflected image.
inline float denoiseLobe(float cosine,float roughness){const float a=max(roughness*roughness,0.002f);return exp(-max(0.0f,1.0f-cosine)/(a*a));}
inline DenoiseHistory denoiseLoad(constant GPUDenoiseParams& p,const device GPUDenoiseHistoryHeader* header,const device GPUDenoiseHistoryRecord* records,
    const device GPUDenoiseSpecularHistory* specular,uint index){
    return denoiseUnpack(header[0],records[index],denoiseSignal(p)==DENOISE_SIGNAL_SPECULAR?specular[index]:GPUDenoiseSpecularHistory{});}
inline bool denoiseCompatible(GPUDISurface s,DenoiseHistory h,constant GPUDenoiseParams& p,GPUSpecularSample spec){
    if((p.flags&DENOISE_RESET)||!s.valid||!h.valid||!h.length||h.signal!=denoiseSignal(p)||h.viewID!=p.viewID||h.historyEpoch!=p.historyEpoch||h.signalRevision!=p.signalRevision||
        h.slot!=s.instanceSlot||h.instanceGeneration!=s.instanceGeneration||h.materialRevision!=s.materialRevision||
        !isfinite(s.depth)||!isfinite(h.depth)||s.depth<=0||h.depth<=0||!all(isfinite(diVec(h.color)))||
        !isfinite(h.firstMoment)||!isfinite(h.secondMoment)||!isfinite(h.variance)||h.variance<0)return false;
    const float3 n=denoiseNormal(s,denoiseSignal(p)),old=diVec(h.normal);if(!all(isfinite(n))||!all(isfinite(old))||dot(n,n)<=0||dot(old,old)<=0||dot(normalize(n),normalize(old))<p.normalThreshold)return false;
    // Compare both receivers in this camera's space; stored h.depth belongs
    // to the previous camera and can change for an unchanged world surface.
    const float oldDepth=(p.flags&DENOISE_CURRENT_VIEW_DEPTH)?
        -(p.currentViewDepthRow[0]*h.position[0]+p.currentViewDepthRow[1]*h.position[1]+p.currentViewDepthRow[2]*h.position[2]+p.currentViewDepthRow[3]):h.depth;
    if(!isfinite(oldDepth)||oldDepth<=0)return false;
    const float scale=max(s.depth,oldDepth);if(abs(s.depth-oldDepth)>p.depthThreshold*scale||!all(isfinite(diVec(s.position)))||!all(isfinite(diVec(h.position)))||
        abs(dot(diVec(s.position)-diVec(h.position),normalize(n)))>p.planeThreshold*scale)return false;
    if(h.flags&~(denoiseSignal(p)==DENOISE_SIGNAL_SPECULAR?DENOISE_HISTORY_STOCHASTIC:0u))return false;
    if(denoiseSignal(p)==DENOISE_SIGNAL_SPECULAR){
        const bool stochastic=(spec.flags&SPECULAR_SAMPLE_STOCHASTIC)!=0&&!(p.flags&DENOISE_DIAGNOSTIC_ENDPOINTS);
        if(!(spec.flags&SPECULAR_SAMPLE_VALID)||(spec.flags&SPECULAR_SAMPLE_ERROR)||
           stochastic!=bool(h.flags&DENOISE_HISTORY_STOCHASTIC)||!isfinite(h.roughness)||!isfinite(s.roughness)||
           abs(h.roughness-s.roughness)>p.roughnessThreshold||!isfinite(spec.hitDistance)||spec.hitDistance<0)return false;
        if(!stochastic&&(h.path!=spec.path||h.secondarySlot!=spec.secondarySlot||h.secondaryGeneration!=spec.secondaryGeneration||
           !isfinite(h.hitDistance)||abs(h.hitDistance-spec.hitDistance)>p.hitDistanceThreshold*max(1.0f,max(h.hitDistance,spec.hitDistance))))return false;
    }
    return true;
}
inline bool denoiseNeighbor(GPUDISurface a,GPUDISurface b,constant GPUDenoiseParams& p){
    const float3 n=denoiseNormal(a,denoiseSignal(p)),m=denoiseNormal(b,denoiseSignal(p));return a.valid&&b.valid&&all(isfinite(n))&&all(isfinite(m))&&dot(n,n)>0&&dot(m,m)>0&&
        dot(normalize(n),normalize(m))>=p.normalThreshold&&abs(a.depth-b.depth)<=p.depthThreshold*max(a.depth,b.depth);
}
kernel void denoise_temporal(constant GPUDenoiseParams& p [[buffer(0)]],const device GPUDISurface* surfaces [[buffer(1)]],
    const device GPUDenoiseHistoryRecord* previous [[buffer(2)]],const device GPUSpecularSample* specular [[buffer(3)]],device GPUDenoiseHistoryRecord* next [[buffer(4)]],
    const device GPUDenoiseHistoryHeader* previousHeader [[buffer(5)]],device GPUDenoiseHistoryHeader* nextHeader [[buffer(6)]],
    const device GPUDenoiseSpecularHistory* previousSpecular [[buffer(7)]],device GPUDenoiseSpecularHistory* nextSpecular [[buffer(8)]],
    texture2d<float,access::read> raw [[texture(0)]],texture2d<float,access::read> motion [[texture(1)]],
    texture2d<float,access::write> output [[texture(2)]],texture2d<float,access::write> moments [[texture(3)]],uint tid [[thread_position_in_grid]]){
    if(tid>=p.width*p.height)return;const uint2 pixel(tid%p.width,tid/p.width);const GPUDISurface s=surfaces[tid];
    DenoiseHistory h{};h.signal=denoiseSignal(p);h.viewID=p.viewID;h.historyEpoch=p.historyEpoch;h.signalRevision=p.signalRevision;
    float3 current=raw.read(pixel).rgb;const float3 normal=denoiseNormal(s,denoiseSignal(p));GPUSpecularSample spec{};
    if(denoiseSignal(p)==DENOISE_SIGNAL_SPECULAR)spec=specular[tid];
    if(denoiseSignal(p)==DENOISE_SIGNAL_SPECULAR&&(spec.flags&SPECULAR_SAMPLE_STOCHASTIC)&&!(p.flags&DENOISE_DIAGNOSTIC_ENDPOINTS))h.flags|=DENOISE_HISTORY_STOCHASTIC;
    if(s.valid&&(!all(isfinite(current))||!all(isfinite(normal))||dot(normal,normal)<=0))h.flags=1u;
    if(s.valid&&all(isfinite(current))&&all(isfinite(normal))&&dot(normal,normal)>0){current=max(current,0.0f);if(denoiseSignal(p)==DENOISE_SIGNAL_AO)current=saturate(current);
        DenoiseHistory old{};bool reuse=false;
        if(!(p.flags&DENOISE_RESET)){const float2 delta=motion.read(pixel).xy,center=float2(pixel)+.5f+delta;
            if(all(isfinite(delta))&&all(center>=0)&&all(center<float2(p.width,p.height))){const uint2 q=uint2(center);old=denoiseLoad(p,previousHeader,previous,previousSpecular,q.y*p.width+q.x);reuse=denoiseCompatible(s,old,p,spec);}}
        float3 oldColor=diVec(old.color);
        float localVariance=0;
        const bool softTest=reuse&&(p.flags&DENOISE_SOFT_CHANGE);
        if(!reuse||old.length<p.minHistoryForVariance||(p.flags&DENOISE_CLAMP_HISTORY)||softTest){float3 sum=0,square=0,lo=INFINITY,hi=-INFINITY;float count=0,lumSum=0,lumSquare=0;
            for(int y=-1;y<=1;++y)for(int x=-1;x<=1;++x){const int2 q=int2(pixel)+int2(x,y);if(any(q<0)||any(q>=int2(p.width,p.height)))continue;
                const uint index=uint(q.y)*p.width+uint(q.x);if(!denoiseNeighbor(s,surfaces[index],p))continue;const float3 value=raw.read(uint2(q)).rgb;
                if(!all(isfinite(value)))continue;sum+=value;square+=value*value;lo=min(lo,value);hi=max(hi,value);const float l=diLuminance(value);lumSum+=l;lumSquare+=l*l;++count;}
            if(count>0){const float3 mean=sum/count,variance=max(square/count-mean*mean,0.0f),sigma=sqrt(variance+p.varianceFloor);
                localVariance=max(0.0f,lumSquare/count-(lumSum/count)*(lumSum/count));
                const float3 low=max(lo,mean-p.clampSigma*sigma),high=min(hi,mean+p.clampSigma*sigma);if(reuse&&(p.flags&DENOISE_CLAMP_HISTORY))oldColor=clamp(oldColor,low,high);
                // Soft change: drop the history only where it left the current neighbourhood.
                if(softTest&&any(abs(oldColor-mean)>p.clampSigma*sigma+.02f*max(abs(mean),float3(1e-4f))))reuse=false;}
        }
        const uint length=reuse?min(old.length,p.maxHistory-1u)+1u:1u;
        const float alpha=reuse?max((p.flags&DENOISE_SOFT_CHANGE)?2*p.temporalAlpha:p.temporalAlpha,1.0f/float(length)):1;
        const float ma=reuse?max(p.momentsAlpha,1.0f/float(length)):1,L=diLuminance(current);float3 color=reuse?mix(oldColor,current,alpha):current;if(denoiseSignal(p)==DENOISE_SIGNAL_AO)color=saturate(color);
        h.length=length;h.firstMoment=reuse?mix(old.firstMoment,L,ma):L;h.secondMoment=reuse?mix(old.secondMoment,L*L,ma):L*L;
        h.variance=max(p.varianceFloor,h.secondMoment-h.firstMoment*h.firstMoment);if(length<p.minHistoryForVariance)h.variance=max(h.variance,localVariance/float(length));h.depth=s.depth;h.roughness=s.roughness;
        h.slot=s.instanceSlot;h.instanceGeneration=s.instanceGeneration;h.materialRevision=s.materialRevision;
        h.hitDistance=spec.hitDistance;h.path=spec.path;h.secondarySlot=spec.secondarySlot;h.secondaryGeneration=spec.secondaryGeneration;
        for(uint i=0;i<3;++i){h.color[i]=color[i];h.position[i]=s.position[i];h.normal[i]=normalize(normal)[i];}
        h.valid=all(isfinite(color))&&isfinite(h.variance);if(!h.valid)h.flags=1u;
    }
    next[tid]=denoisePackRecord(h);if(denoiseSignal(p)==DENOISE_SIGNAL_SPECULAR)nextSpecular[tid]=denoisePackSpecular(h);
    if(tid==0)nextHeader[0]=denoisePackHeader(h); // the same domain identity for every pixel of this dispatch
    output.write(float4(h.color[0],h.color[1],h.color[2],1),pixel);
    moments.write(float4(h.firstMoment,h.secondMoment,h.variance,float(h.length)),pixel);
}

inline float denoiseWeight(GPUDISurface a,GPUDISurface b,float la,float lb,float variance,constant GPUDenoiseParams& p){
    if(!a.valid||!b.valid)return 0;const float3 na=denoiseGuide(a,denoiseSignal(p)),nb=denoiseGuide(b,denoiseSignal(p));if(dot(na,na)<=0||dot(nb,nb)<=0)return 0;
    const float cosine=dot(normalize(na),normalize(nb));const float normal=pow(max(0.0f,cosine),p.normalPhi);
    const float depth=exp(-abs(a.depth-b.depth)/max(1e-6f,p.depthThreshold*max(a.depth,b.depth)*float(p.atrousStep)));
    const float color=exp(-abs(la-lb)/(p.luminancePhi*sqrt(max(variance,p.varianceFloor))+1e-6f));
    const float rough=denoiseSignal(p)==DENOISE_SIGNAL_SPECULAR?exp(-abs(a.roughness-b.roughness)/max(p.roughnessThreshold,1e-6f))*denoiseLobe(cosine,a.roughness):1;
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
        if(denoiseSignal(p)==DENOISE_SIGNAL_SPECULAR){const GPUSpecularSample a=specular[tid],b=specular[index];
            if((a.flags|b.flags)&SPECULAR_SAMPLE_ERROR||((a.flags^b.flags)&SPECULAR_SAMPLE_STOCHASTIC))continue;
            if((!(a.flags&SPECULAR_SAMPLE_STOCHASTIC)||(p.flags&DENOISE_DIAGNOSTIC_ENDPOINTS))&&(a.path!=b.path||a.secondarySlot!=b.secondarySlot||a.secondaryGeneration!=b.secondaryGeneration))continue;}
        const float weight=taps[x+2]*taps[y+2]*denoiseWeight(center,surfaces[index],lum,diLuminance(color),variance,p);sum+=color*weight;weights+=weight;
    }
    float3 result=weights>0?sum/weights:value;if(denoiseSignal(p)==DENOISE_SIGNAL_AO)result=saturate(result);output.write(float4(result,1),pixel);
}

constant constexpr uint DENOISE_TILE_VALID=1u<<30;
static_assert((DENOISE_TILE_VALID&(SPECULAR_SAMPLE_VALID|SPECULAR_SAMPLE_PREFILTERED|SPECULAR_SAMPLE_ERROR|SPECULAR_SAMPLE_STOCHASTIC|SPECULAR_SAMPLE_MIXED))==0,
              "Tile validity must not alias any specular metadata flag");
// Cache the unchanged 5x5 atrous stencil in threadgroup memory. All source
// values remain Float32; sample order, rejection rules and filter weights are
// identical to denoise_atrous. Specializing the stride bounds the shared tile
// (largest: 24x16x40 = 15360 bytes), without per-pixel guide copies.
// Larger strides retain the measured-faster scalar reference kernel.
template<uint Step>
inline void denoiseAtrousTile(constant GPUDenoiseParams& p,const device GPUDISurface* surfaces,
    const device GPUSpecularSample* specular,texture2d<float,access::read> input,
    texture2d<float,access::read> moments,texture2d<float,access::write> output,
    uint2 group,uint2 local,uint lane,
    threadgroup float4* colors,threadgroup float4* guides,threadgroup uint2* metadata) {
    constexpr uint W=16+4*Step,H=8+4*Step,N=W*H;
    const int2 origin=int2(group*uint2(16,8))-int2(2*Step);
    for(uint i=lane;i<N;i+=128) {
        const int2 q=origin+int2(i%W,i/W);float4 color=0,guide=0;uint2 meta=0;
        if(all(q>=0)&&all(q<int2(p.width,p.height))) {
            const uint index=uint(q.y)*p.width+uint(q.x);const GPUDISurface surface=surfaces[index];
            color=input.read(uint2(q));const float3 n=denoiseGuide(surface,denoiseSignal(p));
            if(surface.valid&&dot(n,n)>0) {
                guide=float4(normalize(n),surface.depth);meta.x=as_type<uint>(surface.roughness);
                meta.y=DENOISE_TILE_VALID|(denoiseSignal(p)==DENOISE_SIGNAL_SPECULAR?specular[index].flags:0u);
            }
        }
        colors[i]=color;guides[i]=guide;metadata[i]=meta;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint2 pixel=group*uint2(16,8)+local;
    if(any(pixel>=uint2(p.width,p.height)))return;
    const uint centerIndex=(local.y+2*Step)*W+local.x+2*Step,tid=pixel.y*p.width+pixel.x;
    const float3 value=colors[centerIndex].rgb;const float4 center=guides[centerIndex];const uint2 centerMeta=metadata[centerIndex];
    if(!(centerMeta.y&DENOISE_TILE_VALID)){output.write(float4(value,1),pixel);return;}
    const float variance=moments.read(pixel).z,lum=diLuminance(value);
    const float taps[5]={1,4,6,4,1};float3 sum=0;float weights=0;
    for(int y=-2;y<=2;++y)for(int x=-2;x<=2;++x) {
        const int2 q=int2(pixel)+int2(x,y)*int(Step);
        if(any(q<0)||any(q>=int2(p.width,p.height)))continue;
        const uint i=uint(int(centerIndex)+y*int(Step*W)+x*int(Step));
        const float3 color=colors[i].rgb;const float4 neighbor=guides[i];const uint2 meta=metadata[i];
        if(!all(isfinite(color))||!(meta.y&DENOISE_TILE_VALID))continue;
        if(denoiseSignal(p)==DENOISE_SIGNAL_SPECULAR) {
            if((centerMeta.y|meta.y)&SPECULAR_SAMPLE_ERROR||((centerMeta.y^meta.y)&SPECULAR_SAMPLE_STOCHASTIC))continue;
            if(!(centerMeta.y&SPECULAR_SAMPLE_STOCHASTIC)||(p.flags&DENOISE_DIAGNOSTIC_ENDPOINTS)) {
                const GPUSpecularSample a=specular[tid],b=specular[uint(q.y)*p.width+uint(q.x)];
                if(a.path!=b.path||a.secondarySlot!=b.secondarySlot||a.secondaryGeneration!=b.secondaryGeneration)continue;
            }
        }
        const float cosine=dot(center.xyz,neighbor.xyz);const float normal=pow(max(0.0f,cosine),p.normalPhi);
        const float depth=exp(-abs(center.w-neighbor.w)/max(1e-6f,p.depthThreshold*max(center.w,neighbor.w)*float(p.atrousStep)));
        const float radiance=exp(-abs(lum-diLuminance(color))/(p.luminancePhi*sqrt(max(variance,p.varianceFloor))+1e-6f));
        const float rough=denoiseSignal(p)==DENOISE_SIGNAL_SPECULAR?exp(-abs(as_type<float>(centerMeta.x)-as_type<float>(meta.x))/max(p.roughnessThreshold,1e-6f))*denoiseLobe(cosine,as_type<float>(centerMeta.x)):1;
        const float weight=taps[x+2]*taps[y+2]*(normal*depth*radiance*rough);
        sum+=color*weight;weights+=weight;
    }
    float3 result=weights>0?sum/weights:value;if(denoiseSignal(p)==DENOISE_SIGNAL_AO)result=saturate(result);
    output.write(float4(result,1),pixel);
}
#define DENOISE_TILE_KERNEL(Name,Step) \
kernel void Name(constant GPUDenoiseParams& p [[buffer(0)]],const device GPUDISurface* surfaces [[buffer(1)]], \
    const device GPUSpecularSample* specular [[buffer(3)]],texture2d<float,access::read> input [[texture(0)]], \
    texture2d<float,access::read> moments [[texture(1)]],texture2d<float,access::write> output [[texture(2)]], \
    uint2 group [[threadgroup_position_in_grid]],uint2 local [[thread_position_in_threadgroup]],uint lane [[thread_index_in_threadgroup]]) { \
    threadgroup float4 colors[(16+4*Step)*(8+4*Step)],guides[(16+4*Step)*(8+4*Step)]; \
    threadgroup uint2 metadata[(16+4*Step)*(8+4*Step)]; \
    denoiseAtrousTile<Step>(p,surfaces,specular,input,moments,output,group,local,lane,colors,guides,metadata); \
}
DENOISE_TILE_KERNEL(denoise_atrous_tile1,1)
DENOISE_TILE_KERNEL(denoise_atrous_tile2,2)
#undef DENOISE_TILE_KERNEL

// Stride4 needs a large halo. Cache only geometry (13.5KiB), retaining the
// original color reads/tap order instead of exceeding validation shared memory.
kernel void denoise_atrous_guide4(constant GPUDenoiseParams& p [[buffer(0)]],const device GPUDISurface* surfaces [[buffer(1)]],
    const device GPUSpecularSample* specular [[buffer(3)]],texture2d<float,access::read> input [[texture(0)]],
    texture2d<float,access::read> moments [[texture(1)]],texture2d<float,access::write> output [[texture(2)]],
    uint2 group [[threadgroup_position_in_grid]],uint2 local [[thread_position_in_threadgroup]],uint lane [[thread_index_in_threadgroup]]) {
    constexpr uint W=24,N=24*24;
    threadgroup float4 geometry[N];threadgroup uint2 metadata[N];
    const int2 origin=int2(group*uint2(8))-int2(8);
    for(uint i=lane;i<N;i+=64){
        const int2 q=origin+int2(i%W,i/W);float4 guide=0;uint2 meta=0;
        if(all(q>=0)&&all(q<int2(p.width,p.height))){
            const uint index=uint(q.y)*p.width+uint(q.x);const GPUDISurface s=surfaces[index];const float3 n=denoiseGuide(s,denoiseSignal(p));
            if(s.valid&&dot(n,n)>0){guide=float4(normalize(n),s.depth);meta.x=as_type<uint>(s.roughness);
                meta.y=DENOISE_TILE_VALID|(denoiseSignal(p)==DENOISE_SIGNAL_SPECULAR?specular[index].flags:0u);}
        }
        geometry[i]=guide;metadata[i]=meta;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint2 pixel=group*uint2(8)+local;if(any(pixel>=uint2(p.width,p.height)))return;
    const uint centerIndex=(local.y+8)*W+local.x+8,tid=pixel.y*p.width+pixel.x;
    const float3 value=input.read(pixel).rgb;const float4 center=geometry[centerIndex];const uint2 centerMeta=metadata[centerIndex];
    if(!(centerMeta.y&DENOISE_TILE_VALID)){output.write(float4(value,1),pixel);return;}
    const float variance=moments.read(pixel).z,lum=diLuminance(value);const float taps[5]={1,4,6,4,1};float3 sum=0;float weights=0;
    for(int y=-2;y<=2;++y)for(int x=-2;x<=2;++x){
        const int2 q=int2(pixel)+int2(x,y)*4;if(any(q<0)||any(q>=int2(p.width,p.height)))continue;
        const uint index=uint(int(centerIndex)+y*int(4*W)+x*4);const float4 neighbor=geometry[index];const uint2 meta=metadata[index];
        const float3 color=input.read(uint2(q)).rgb;if(!all(isfinite(color))||!(meta.y&DENOISE_TILE_VALID))continue;
        if(denoiseSignal(p)==DENOISE_SIGNAL_SPECULAR){
            if((centerMeta.y|meta.y)&SPECULAR_SAMPLE_ERROR||((centerMeta.y^meta.y)&SPECULAR_SAMPLE_STOCHASTIC))continue;
            if(!(centerMeta.y&SPECULAR_SAMPLE_STOCHASTIC)||(p.flags&DENOISE_DIAGNOSTIC_ENDPOINTS)){
                const GPUSpecularSample a=specular[tid],b=specular[uint(q.y)*p.width+uint(q.x)];
                if(a.path!=b.path||a.secondarySlot!=b.secondarySlot||a.secondaryGeneration!=b.secondaryGeneration)continue;
            }
        }
        const float cosine=dot(center.xyz,neighbor.xyz);const float normal=pow(max(0.0f,cosine),p.normalPhi);
        const float depth=exp(-abs(center.w-neighbor.w)/max(1e-6f,p.depthThreshold*max(center.w,neighbor.w)*float(p.atrousStep)));
        const float radiance=exp(-abs(lum-diLuminance(color))/(p.luminancePhi*sqrt(max(variance,p.varianceFloor))+1e-6f));
        const float rough=denoiseSignal(p)==DENOISE_SIGNAL_SPECULAR?exp(-abs(as_type<float>(centerMeta.x)-as_type<float>(meta.x))/max(p.roughnessThreshold,1e-6f))*denoiseLobe(cosine,as_type<float>(centerMeta.x)):1;
        const float weight=taps[x+2]*taps[y+2]*(normal*depth*radiance*rough);sum+=color*weight;weights+=weight;
    }
    float3 result=weights>0?sum/weights:value;if(denoiseSignal(p)==DENOISE_SIGNAL_AO)result=saturate(result);output.write(float4(result,1),pixel);
}

// DI/GI/AO share the same geometric receiver. The final stride computes its
// geometry weights once; each signal retains its own variance/color weights.
constant bool kDenoiseDiffuseDI=kDenoiseSignal!=DENOISE_SIGNAL_GI;
kernel void denoise_atrous_diffuse4(constant GPUDenoiseParams& p [[buffer(0)]],const device GPUDISurface* surfaces [[buffer(1)]],
    texture2d<float,access::read> di [[texture(0), function_constant(kDenoiseDiffuseDI)]],texture2d<float,access::read> gi [[texture(1)]],texture2d<float,access::read> ao [[texture(2)]],
    texture2d<float,access::read> dm [[texture(3), function_constant(kDenoiseDiffuseDI)]],texture2d<float,access::read> gm [[texture(4)]],texture2d<float,access::read> am [[texture(5)]],
    texture2d<float,access::write> dout [[texture(6), function_constant(kDenoiseDiffuseDI)]],texture2d<float,access::write> gout [[texture(7)]],texture2d<float,access::write> aout [[texture(8)]],
    uint2 group [[threadgroup_position_in_grid]],uint2 local [[thread_position_in_threadgroup]],uint lane [[thread_index_in_threadgroup]]) {
    constexpr uint W=24,N=W*W;threadgroup float4 geometry[N];threadgroup uint valid[N];
    const int2 origin=int2(group*uint2(8))-int2(8);
    for(uint i=lane;i<N;i+=64){const int2 q=origin+int2(i%W,i/W);float4 guide=0;uint okay=0;
        if(all(q>=0)&&all(q<int2(p.width,p.height))){const GPUDISurface s=surfaces[uint(q.y)*p.width+uint(q.x)];const float3 n=diVec(s.geometricNormal);
            if(s.valid&&dot(n,n)>0){guide=float4(normalize(n),s.depth);okay=1;}}
        geometry[i]=guide;valid[i]=okay;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);const uint2 pixel=group*uint2(8)+local;if(any(pixel>=uint2(p.width,p.height)))return;
    const uint centerIndex=(local.y+8)*W+local.x+8;const float4 center=geometry[centerIndex];
    const float3 dv=kDenoiseDiffuseDI?di.read(pixel).rgb:float3(0),gv=gi.read(pixel).rgb,av=ao.read(pixel).rgb;
    if(!valid[centerIndex]){if(kDenoiseDiffuseDI)dout.write(float4(dv,1),pixel);gout.write(float4(gv,1),pixel);aout.write(float4(av,1),pixel);return;}
    const float3 lum(diLuminance(dv),diLuminance(gv),diLuminance(av));const float3 variance(kDenoiseDiffuseDI?dm.read(pixel).z:0.0f,gm.read(pixel).z,am.read(pixel).z);
    const float taps[5]={1,4,6,4,1};float3 ds=0,gs=0,as=0,weights=0;
    for(int y=-2;y<=2;++y)for(int x=-2;x<=2;++x){const int2 q=int2(pixel)+int2(x,y)*4;if(any(q<0)||any(q>=int2(p.width,p.height)))continue;
        const uint i=uint(int(centerIndex)+y*int(4*W)+x*4);if(!valid[i])continue;const float4 neighbor=geometry[i];
        const float normal=pow(max(0.0f,dot(center.xyz,neighbor.xyz)),p.normalPhi);
        const float depth=exp(-abs(center.w-neighbor.w)/max(1e-6f,p.depthThreshold*max(center.w,neighbor.w)*float(p.atrousStep)));
        const float3 dc=kDenoiseDiffuseDI?di.read(uint2(q)).rgb:float3(0),gc=gi.read(uint2(q)).rgb,ac=ao.read(uint2(q)).rgb;
        if(kDenoiseDiffuseDI&&all(isfinite(dc))){const float c=exp(-abs(lum.x-diLuminance(dc))/(p.luminancePhi*sqrt(max(variance.x,p.varianceFloor))+1e-6f));const float w=taps[x+2]*taps[y+2]*(normal*depth*c*1.0f);ds+=dc*w;weights.x+=w;}
        if(all(isfinite(gc))){const float c=exp(-abs(lum.y-diLuminance(gc))/(p.luminancePhi*sqrt(max(variance.y,p.varianceFloor))+1e-6f));const float w=taps[x+2]*taps[y+2]*(normal*depth*c*1.0f);gs+=gc*w;weights.y+=w;}
        if(all(isfinite(ac))){const float c=exp(-abs(lum.z-diLuminance(ac))/(p.luminancePhi*sqrt(max(variance.z,p.varianceFloor))+1e-6f));const float w=taps[x+2]*taps[y+2]*(normal*depth*c*1.0f);as+=ac*w;weights.z+=w;}
    }
    if(kDenoiseDiffuseDI)dout.write(float4(weights.x>0?ds/weights.x:dv,1),pixel);
    gout.write(float4(weights.y>0?gs/weights.y:gv,1),pixel);aout.write(float4(saturate(weights.z>0?as/weights.z:av),1),pixel);
}

// History corruption control: wrong view ID without setting an error flag.
// Independent compatible() rejects it and resets length to1 at the next frame.
// The view ID lives in the buffer header (offset 0), shared by every pixel.
kernel void denoise_corrupt_history_view(constant GPUDenoiseParams& p [[buffer(0)]],device GPUDenoiseHistoryHeader* header [[buffer(1)]],uint tid [[thread_position_in_grid]]){
    if(tid==0)header[0].viewID^=1u;
}

// Always-on checker: params0,nextRecords1,atomicCounts2,nextHeader3; outputtexture0.
// Eight cleared words: pixels0,valid1,invalid state/domain2,invalid output3,
// reused valid histories4,max valid length5,actual GPU params view6/frame7.
// Raw/reference checks remain independent. One-dimensional exact pixel dispatch.
kernel void denoise_check(constant GPUDenoiseParams& p [[buffer(0)]],const device GPUDenoiseHistoryRecord* history [[buffer(1)]],
    device atomic_uint* counts [[buffer(2)]],const device GPUDenoiseHistoryHeader* header [[buffer(3)]],texture2d<float,access::read> output [[texture(0)]],uint tid [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){
    if(tid>=p.width*p.height)return;const DenoiseHistory h=denoiseUnpack(header[0],history[tid],GPUDenoiseSpecularHistory{});bool invalid=(h.flags&~(denoiseSignal(p)==DENOISE_SIGNAL_SPECULAR?DENOISE_HISTORY_STOCHASTIC:0u))!=0;
    if(h.valid)invalid|=h.signal!=denoiseSignal(p)||h.viewID!=p.viewID||h.historyEpoch!=p.historyEpoch||h.signalRevision!=p.signalRevision||
        h.length<1||h.length>p.maxHistory||!isfinite(h.firstMoment)||!isfinite(h.secondMoment)||!isfinite(h.variance)||h.variance<0||!all(isfinite(diVec(h.color)));
    const float3 color=output.read(uint2(tid%p.width,tid/p.width)).rgb;const bool badOutput=!all(isfinite(color))||any(color<0)||(denoiseSignal(p)==DENOISE_SIGNAL_AO&&any(color>1));
    const uint values[5]={1u,h.valid?1u:0u,invalid?1u:0u,badOutput?1u:0u,h.valid&&h.length>1u?1u:0u};
    for(uint i=0;i<5;++i){const uint sum=simd_sum(values[i]);if(lane==0&&sum)atomic_fetch_add_explicit(counts+i,sum,memory_order_relaxed);}
    const uint longest=simd_max(h.valid?h.length:0u);if(lane==0&&longest)atomic_fetch_max_explicit(counts+5,longest,memory_order_relaxed);
    if(tid==0){atomic_store_explicit(counts+6,p.viewID,memory_order_relaxed);atomic_store_explicit(counts+7,p.frameIndex,memory_order_relaxed);}
}
