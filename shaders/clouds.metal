#include "atmosphere_common.h"

static float cloudDensity(float3 point,constant GPUCloudParams& p,constant GPUAtmosphereParams& atmosphere) {
    const float altitude=length(point-atmoVec(atmosphere.planetCenter))-atmosphere.bottomRadius;
    const float3 advected=point-atmoVec(p.wind)*p.timeSeconds;
    return volumeCloudShape(advected.x,advected.y,advected.z,altitude,p.baseHeight,p.topHeight,p.coverage,
                            p.densityScale,p.noiseScale,p.erosionScale,p.seed);
}
static float4 cloudSegments(float3 camera,float3 ray,float maximum,constant GPUCloudParams& p,constant GPUAtmosphereParams& a) {
    const float3 relative=camera-atmoVec(a.planetCenter);const float2 outer=atmoSphere(relative,ray,a.bottomRadius+p.topHeight);
    float begin=max(0.0f,outer.x),end=min(outer.y,maximum);
    const AtmoSegment planet=atmoSegment(camera,ray,a,maximum);if(planet.ground)end=min(end,planet.end);
    if(end<=begin)return float4(0);
    const float2 inner=atmoSphere(relative,ray,a.bottomRadius+p.baseHeight);
    if(inner.y<=begin||inner.x>=end||inner.x==inner.y)return float4(begin,end,0,0);
    // Two shell intervals preserve the far cloud sheet on a planetary tangent
    // that crosses below cloud base but does not hit the solid planet.
    const float firstEnd=clamp(inner.x,begin,end),secondBegin=clamp(inner.y,begin,end);
    return float4(begin,firstEnd,secondBegin,end);
}
static float cloudDistanceAt(float distance,float4 intervals) {
    const float first=intervals.y-intervals.x;
    return distance<first?intervals.x+distance:intervals.z+distance-first;
}
static float cloudToLight(float3 point,float3 towardLight,constant GPUCloudParams& p,constant GPUAtmosphereParams& atmosphere) {
    const float4 intervals=cloudSegments(point,towardLight,p.maxDistance,p,atmosphere);
    const float distance=(intervals.y-intervals.x)+(intervals.w-intervals.z);if(distance<=0)return 1;
    const uint count=clamp(min(p.lightSteps,uint(ceil(distance/max(p.lightStepDistance,1.0f)))),1u,64u);
    const float ds=distance/float(count);float tau=0;
    for(uint i=0;i<count;++i)tau+=cloudDensity(point+towardLight*cloudDistanceAt((float(i)+0.5f)*ds,intervals),p,atmosphere)*p.extinction*ds;
    return exp(-tau);
}
static float cloudOpaque(uint2 pixel,constant GPUCloudParams& p,texture2d<float,access::read> depth) {
    const uint2 begin=pixel*uint2(p.outputWidth,p.outputHeight)/uint2(p.width,p.height);
    const uint2 end=min(((pixel+1u)*uint2(p.outputWidth,p.outputHeight)+uint2(p.width,p.height)-1u)/uint2(p.width,p.height),uint2(p.outputWidth,p.outputHeight));
    float distance=p.maxDistance;
    // Full-rate reference is one texel. Low-rate ratios are explicitly 1..4;
    // conservative nearest scene depth covers the whole corresponding block.
    for(uint y=0;y<4;++y)for(uint x=0;x<4;++x){const uint2 at=begin+uint2(x,y);if(any(at>=end))continue;
        distance=min(distance,atmoOpaqueDistance(at,uint2(p.outputWidth,p.outputHeight),depth,p.inverseViewProjection,atmoVec(p.cameraPosition),p.maxDistance));}
    return distance;
}
kernel void clouds_march(constant GPUCloudParams& p [[buffer(0)]],constant GPUAtmosphereParams& a [[buffer(1)]],
    device GPUCloudHistory* fresh [[buffer(2)]],device atomic_uint* counters [[buffer(15)]],texture2d<float,access::read> depth [[texture(0)]],
    texture2d<float> trans [[texture(1)]],texture2d<float> multi [[texture(2)]],texture2d<float,access::write> output [[texture(5)]],
    texture2d<float,access::write> guide [[texture(7)]],uint2 pixel [[thread_position_in_grid]]) {
    if(pixel.x>=p.width||pixel.y>=p.height)return;const uint index=pixel.y*p.width+pixel.x;
    GPUCloudHistory history{};history.viewID=p.viewID;history.generation=p.generation;history.samples=1;history.transmittance=1;
    if(!p.width||!p.height||p.outputWidth>p.width*4||p.outputHeight>p.height*4){volumeCount(counters,1,1);fresh[index]=history;output.write(float4(0,0,0,1),pixel);guide.write(float4(0,p.maxDistance,0,0),pixel);return;}
    const float3 camera=atmoVec(p.cameraPosition),ray=atmoPixelRay(pixel,uint2(p.width,p.height),p.inverseViewProjection,camera);
    const float opaque=cloudOpaque(pixel,p,depth);history.opaqueDistance=opaque;
    const float4 intervals=cloudSegments(camera,ray,opaque,p,a);const float length=(intervals.y-intervals.x)+(intervals.w-intervals.z);
    float3 radiance=0;float T=1,weightedDistance=0,weightSum=0;
    if(length>0){const uint steps=clamp(p.marchSteps,1u,512u);const float ds=length/float(steps);
        const bool reference=p.width==p.outputWidth&&p.height==p.outputHeight;
        const float jitter=reference?0.5f:volumeRandom(volumeHash(index,p.frameIndex,0,p.seed));
        const float3 sun=atmoVec(a.sunDirection),moon=atmoVec(a.moonDirection);
        for(uint i=0;i<steps;++i){const float distance=cloudDistanceAt((float(i)+jitter)*ds,intervals),pointDensityDistance=distance;
            const float3 point=camera+ray*pointDensityDistance;float density=cloudDensity(point,p,a);if(p.corruption==VOLUME_CORRUPT_UNITS)density*=1000;
            const float sigma=density*p.extinction;if(sigma<=0)continue;
            float3 incident=atmoVec(a.sunIrradiance)*(atmoTransmittance(point,sun,a,trans)*cloudToLight(point,sun,p,a)*atmoHgPhase(dot(ray,sun),p.anisotropy)+atmoMultiple(point,sun,a,multi));
            incident+=atmoVec(a.moonIrradiance)*(atmoTransmittance(point,moon,a,trans)*cloudToLight(point,moon,p,a)*atmoHgPhase(dot(ray,moon),p.anisotropy)+atmoMultiple(point,moon,a,multi));
            const float segment=exp(-sigma*ds),factor=-expm1(-sigma*ds)/sigma;
            radiance+=T*incident*(sigma*p.albedo)*factor;
            const float scatteringWeight=T*(1-segment);weightedDistance+=scatteringWeight*distance;weightSum+=scatteringWeight;T*=segment;
            volumeCount(counters,5,1);if(T<=p.terminationTransmittance)break;
        }
    }
    const float cloudDistance=weightSum>1e-8f?weightedDistance/weightSum:0;
    const float3 point=camera+ray*cloudDistance;history.worldPosition[0]=point.x;history.worldPosition[1]=point.y;history.worldPosition[2]=point.z;
    history.radiance[0]=radiance.x;history.radiance[1]=radiance.y;history.radiance[2]=radiance.z;history.transmittance=T;
    history.valid=weightSum>1e-8f&&all(isfinite(radiance))&&isfinite(T)&&all(isfinite(point));
    if(!all(isfinite(radiance))||!isfinite(T))volumeCount(counters,0,1);
    fresh[index]=history;output.write(float4(radiance,T),pixel);guide.write(float4(cloudDistance,opaque,0,0),pixel);
}
kernel void clouds_temporal(constant GPUCloudParams& p [[buffer(0)]],device GPUCloudHistory* next [[buffer(2)]],
    const device GPUCloudHistory* previous [[buffer(3)]],device atomic_uint* counters [[buffer(15)]],
    texture2d<float,access::read> current [[texture(3)]],texture2d<float,access::read> guide [[texture(4)]],
    texture2d<float,access::write> output [[texture(5)]],uint2 pixel [[thread_position_in_grid]]) {
    if(pixel.x>=p.width||pixel.y>=p.height)return;const uint index=pixel.y*p.width+pixel.x;
    const float4 raw=current.read(pixel);const float2 depths=guide.read(pixel).xy;GPUCloudHistory h{};
    const float3 ray=atmoPixelRay(pixel,uint2(p.width,p.height),p.inverseViewProjection,atmoVec(p.cameraPosition));
    const float3 point=atmoVec(p.cameraPosition)+ray*depths.x;
    h.worldPosition[0]=point.x;h.worldPosition[1]=point.y;h.worldPosition[2]=point.z;h.opaqueDistance=depths.y;
    h.radiance[0]=raw.x;h.radiance[1]=raw.y;h.radiance[2]=raw.z;h.transmittance=raw.w;
    h.viewID=p.viewID;h.generation=p.generation;h.samples=1;h.valid=depths.x>0&&raw.w<1&&all(isfinite(raw));bool accepted=false;
    if((p.flags&VOLUME_HISTORY_VALID)&&h.valid){const float3 oldPoint=point-atmoVec(p.wind)*(p.timeSeconds-p.previousTimeSeconds);
        const float4 clip=atmoMatrix(p.previousViewProjection)*float4(oldPoint,1);const float2 uv=clip.xy/max(clip.w,1e-6f)*float2(0.5f,-0.5f)+0.5f;
        if(clip.w>0&&all(uv>=0)&&all(uv<1)){const uint2 at=uint2(uv*float2(p.width,p.height));GPUCloudHistory old=previous[at.y*p.width+at.x];
            if(p.corruption==VOLUME_CORRUPT_HISTORY)old.generation^=1u;
            const bool identity=old.viewID==p.viewID&&old.generation==p.generation;
            accepted=old.valid&&(identity||p.corruption==VOLUME_CORRUPT_HISTORY)&&old.samples>0&&old.samples<=64&&
                all(isfinite(float4(old.radiance[0],old.radiance[1],old.radiance[2],old.transmittance)))&&
                length(float3(old.worldPosition[0],old.worldPosition[1],old.worldPosition[2])-oldPoint)<=p.positionThreshold&&
                abs(old.opaqueDistance-depths.y)<=p.depthRelativeThreshold*max(1.0f,depths.y);
            if(accepted){if(!identity)volumeCount(counters,2,1);float4 low=raw,high=raw;
                for(int y=-1;y<=1;++y)for(int x=-1;x<=1;++x){const int2 q=int2(pixel)+int2(x,y);if(any(q<0)||any(q>=int2(p.width,p.height)))continue;
                    const float2 d=guide.read(uint2(q)).xy;if(d.x<=0||abs(d.y-depths.y)>p.depthRelativeThreshold*max(1.0f,depths.y))continue;
                    const float4 value=current.read(uint2(q));low=min(low,value);high=max(high,value);}
                const float4 value=mix(raw,clamp(float4(old.radiance[0],old.radiance[1],old.radiance[2],old.transmittance),low,high),p.historyWeight);
                h.radiance[0]=value.x;h.radiance[1]=value.y;h.radiance[2]=value.z;h.transmittance=value.w;h.samples=min(old.samples+1u,clamp(p.maxHistorySamples,1u,64u));}
        }
        volumeCount(counters,accepted?6u:7u,1u);
    }
    next[index]=h;output.write(float4(h.radiance[0],h.radiance[1],h.radiance[2],h.transmittance),pixel);
}
kernel void clouds_apply(constant GPUCloudParams& p [[buffer(0)]],texture2d<float,access::read> depth [[texture(0)]],
    texture2d<float,access::read> cloud [[texture(3)]],texture2d<float,access::read> guide [[texture(4)]],
    texture2d<float,access::write> output [[texture(5)]],texture2d<float,access::read> scene [[texture(6)]],uint2 pixel [[thread_position_in_grid]]) {
    if(pixel.x>=p.outputWidth||pixel.y>=p.outputHeight)return;
    const float opaque=atmoOpaqueDistance(pixel,uint2(p.outputWidth,p.outputHeight),depth,p.inverseViewProjection,atmoVec(p.cameraPosition),p.maxDistance);
    const float2 coordinate=(float2(pixel)+0.5f)/float2(p.outputWidth,p.outputHeight)*float2(p.width,p.height)-0.5f;
    const int2 base=int2(floor(coordinate));const float2 f=fract(coordinate);float4 value=0;float weights=0;
    for(uint i=0;i<4;++i){const uint2 bit(i&1u,(i>>1u)&1u),at=uint2(clamp(base+int2(bit),int2(0),int2(p.width-1,p.height-1)));
        const float2 d=guide.read(at).xy;
        if(d.x>opaque||abs(d.y-opaque)>p.depthRelativeThreshold*max(opaque,1.0f))continue;
        const float w=(bit.x?f.x:1-f.x)*(bit.y?f.y:1-f.y);value+=cloud.read(at)*w;weights+=w;}
    if(weights>1e-6f)value/=weights;else value=float4(0,0,0,1); // valid transparent fallback at disocclusion/thin depth edges
    output.write(float4(value.rgb+saturate(value.w)*scene.read(pixel).rgb,1),pixel);
}
