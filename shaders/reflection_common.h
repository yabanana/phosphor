#pragma once
#include "gi_cache_common.h"
#include "light_visibility.h"
#include "surface_geometry.h"
constexpr sampler kReflectionProbeSampler(filter::linear,mip_filter::linear,address::clamp_to_edge);
inline float4x4 reflectionMatrix(constant float* m){return float4x4(float4(m[0],m[1],m[2],m[3]),float4(m[4],m[5],m[6],m[7]),float4(m[8],m[9],m[10],m[11]),float4(m[12],m[13],m[14],m[15]));}
inline float3 reflectionConstantVec(constant float* p){return float3(p[0],p[1],p[2]);}
inline float3 reflectionTangent(float3 n){return normalize(cross(abs(n.z)<.999f?float3(0,0,1):float3(0,1,0),n));}
inline float3 reflectionSpecularBRDF(GPUDISurface s,float3 wi){
    float3 n=normalize(diVec(s.shadingNormal)),v=normalize(diVec(s.viewDirection));float nl=dot(n,wi),nv=dot(n,v);
    if(nl<=0||nv<=0||dot(v+wi,v+wi)<1e-20f)return 0;float3 h=normalize(v+wi);float nh=saturate(dot(n,h)),vh=saturate(dot(v,h));
    float a=max(s.roughness*s.roughness,.002f),a2=a*a,d=nh*nh*(a2-1)+1,D=a2/(M_PI_F*d*d);
    float gv=nl*sqrt(nv*nv*(1-a2)+a2),gl=nv*sqrt(nl*nl*(1-a2)+a2);
    float3 f0=mix(float3(.04f),diVec(s.albedo),saturate(s.metallic)),F=f0+(1-f0)*pow(1-vh,5.0f);
    return D*(.5f/max(gv+gl,1e-5f))*F;
}
struct ReflectionDirection {float3 direction,weight;float pdf;bool valid;};
inline ReflectionDirection reflectionDirection(GPUDISurface s,float2 u){
    ReflectionDirection out{};float3 n=diVec(s.shadingNormal),v=diVec(s.viewDirection);
    if(!s.valid||!all(isfinite(n))||!all(isfinite(v))||dot(n,n)<=0||dot(v,v)<=0)return out;
    n=normalize(n);v=normalize(v);float3 t=reflectionTangent(n),b=cross(n,t);
    float a=max(s.roughness*s.roughness,.002f),a2=a*a,c=sqrt((1-u.x)/(1+(a2-1)*u.x)),r=sqrt(max(0.0f,1-c*c)),phi=2*M_PI_F*u.y;
    float3 h=t*(r*cos(phi))+b*(r*sin(phi))+n*c;float vh=dot(v,h);if(vh<=0)return out;
    out.direction=reflect(-v,h);float d=c*c*(a2-1)+1;out.pdf=(a2/(M_PI_F*d*d))*c/(4*vh);
    out.valid=all(isfinite(out.direction))&&isfinite(out.pdf)&&out.pdf>0;
    if(out.valid&&dot(n,out.direction)>0)out.weight=reflectionSpecularBRDF(s,out.direction)*dot(n,out.direction)/out.pdf;
    return out;
}
inline float reflectionRTWeight(float roughness,constant GPUReflectionParams& p){return (p.flags&REFLECTION_ENABLE_RT)?1-smoothstep(p.rtRoughnessLow,p.rtRoughnessHigh,roughness):0;}

inline bool reflectionProbeDirection(GPUReflectionProbe p,float3 point,float3 direction,thread float3& corrected,thread float& weight){
    const float3 lo=diVec(p.boxMin),hi=diVec(p.boxMax);if(!p.enabled||p.blendDistance<=0||any(point<lo)||any(point>hi))return false;
    float exit=INFINITY;for(uint i=0;i<3;++i)if(abs(direction[i])>1e-8f)exit=min(exit,((direction[i]>0?hi[i]:lo[i])-point[i])/direction[i]);
    corrected=point+direction*exit-diVec(p.capturePosition);float3 dist=min(point-lo,hi-point);
    weight=saturate(min(min(dist.x,dist.y),dist.z)/p.blendDistance);
    if(!isfinite(exit)||!all(isfinite(corrected))||dot(corrected,corrected)<1e-20f)return false;corrected=normalize(corrected);return weight>0;
}
inline float3 reflectionProbeRadiance(float3 point,float3 direction,float roughness,bool prefiltered,
    constant GPUReflectionParams& p,const device GPUReflectionProbe* probes,texturecube_array<float> atlas,thread uint& path){
    float3 sum=0;float total=0;path=REFLECTION_PATH_ENVIRONMENT;
    if(p.flags&REFLECTION_ENABLE_PROBES)for(uint i=0;i<p.probeCount;++i){const GPUReflectionProbe probe=probes[i];float3 corrected;float weight;
        if(reflectionProbeDirection(probe,point,direction,corrected,weight)){const float lod=prefiltered?roughness*max(probe.mipCount-1,0.0f):0;
            float3 L=atlas.sample(kReflectionProbeSampler,corrected,probe.cubeIndex,level(lod)).rgb;if(all(isfinite(L))&&all(L>=0)){sum+=L*weight;total+=weight;}}
    }
    if(total>0){path=REFLECTION_PATH_PROBE;float coverage=saturate(total);return (sum/total)*coverage+reflectionConstantVec(p.environment)*(1-coverage);}
    return reflectionConstantVec(p.environment);
}
struct ReflectionSSRHit {uint pixel;float distance;float3 point;bool valid;};
inline bool reflectionSSRPoint(float3 origin,float3 direction,float distance,constant GPUReflectionParams& p,
    texture2d<float,access::read> depth,thread ReflectionSSRHit& hit,thread float& delta){
    const float3 point=origin+direction*distance;const float4 clip=reflectionMatrix(p.viewProjection)*float4(point,1);
    if(!all(isfinite(clip))||clip.w<=0)return false;const float2 uv=(clip.xy/clip.w)*float2(.5f,-.5f)+.5f;
    if(any(uv<0)||any(uv>=1))return false;const uint2 pixel=uint2(uv*float2(p.width,p.height));
    // Integer logical pixels, never normalized against a larger F8 backing.
    const float z=depth.read(pixel).x;if(!isfinite(z)||z<=0||z>1)return false;
    const float2 ndc=(float2(pixel)+.5f)*float2(2,-2)/float2(p.width,p.height)+float2(-1,1);
    const float4 world=reflectionMatrix(p.inverseViewProjection)*float4(ndc,z,1);if(!all(isfinite(world))||abs(world.w)<1e-20f)return false;
    hit.point=world.xyz/world.w;hit.pixel=pixel.y*p.width+pixel.x;hit.distance=length(hit.point-origin);
    delta=-(reflectionMatrix(p.view)*float4(point,1)).z+(reflectionMatrix(p.view)*float4(hit.point,1)).z;return isfinite(delta);
}
inline ReflectionSSRHit reflectionSSR(float3 origin,float3 direction,constant GPUReflectionParams& p,texture2d<float,access::read> depth){
    ReflectionSSRHit hit{};hit.pixel=~0u;float previous=0;bool front=false;
    if(!(p.flags&REFLECTION_ENABLE_SSR))return hit;
    const uint count=clamp(p.ssrSteps,2u,512u);
    for(uint step=1;step<=count;++step){float f=float(step)/float(count),distance=p.maxDistance*f*f,delta;
        ReflectionSSRHit candidate{};if(!reflectionSSRPoint(origin,direction,distance,p,depth,candidate,delta)){previous=distance;front=false;continue;}
        if(front&&delta>=0){float low=previous,high=distance;for(uint j=0;j<min(p.ssrBinarySteps,16u);++j){float m=(low+high)*.5f,md;ReflectionSSRHit mid;
            if(!reflectionSSRPoint(origin,direction,m,p,depth,mid,md)||md<0)low=m;else{high=m;candidate=mid;}}
            float last;if(reflectionSSRPoint(origin,direction,high,p,depth,candidate,last)&&last>=0&&last<=p.ssrThickness){candidate.valid=true;return candidate;}}
        previous=distance;front=delta<0;
    }return hit;
}
inline GPUSpecularSample reflectionEmpty(){GPUSpecularSample s{};s.secondarySlot=~0u;return s;}
inline GPUSpecularSample reflectionFallback(GPUDISurface surface,ReflectionDirection direction,
    constant GPUReflectionParams& p,const device GPUDISurface* surfaces,const device GPURadianceCacheEntry* cache,
    constant GPUProbeGridParams& gi,const device GPUReflectionProbe* probes,
    texture2d<float,access::read> baseRadiance,texture2d<float,access::read> depth,texturecube_array<float> probeAtlas){
    GPUSpecularSample out=reflectionEmpty();if(!surface.valid||!direction.valid)return out;out.flags=SPECULAR_SAMPLE_VALID;
    for(uint i=0;i<3;++i)out.direction[i]=direction.direction[i];out.proposalSolidAngle=direction.pdf;
    if(!any(direction.weight>0))return out;float3 L;
    const auto hit=reflectionSSR(diVec(surface.position),direction.direction,p,depth);
    if(hit.valid&&surfaces[hit.pixel].valid){const GPUDISurface secondary=surfaces[hit.pixel];out.path=REFLECTION_PATH_SSR;out.hitDistance=hit.distance;
        out.secondarySlot=secondary.instanceSlot;out.secondaryGeneration=secondary.instanceGeneration;
        if((p.flags&REFLECTION_ENABLE_CACHE)&&giCacheLookup(hit.point,diVec(secondary.geometricNormal),-direction.direction,gi,cache,L))out.path=REFLECTION_PATH_CACHE;
        else L=baseRadiance.read(uint2(hit.pixel%p.width,hit.pixel/p.width)).rgb;
    }else L=reflectionProbeRadiance(diVec(surface.position),direction.direction,surface.roughness,false,p,probes,probeAtlas,out.path);
    const float3 result=max(L,0.0f)*direction.weight;for(uint i=0;i<3;++i)out.radiance[i]=result[i];return out;
}
