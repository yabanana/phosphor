#pragma once
#include <metal_stdlib>
#include "renderer/gpu_types.h"
#include "renderer/volume_noise.h"
using namespace metal;
using namespace phosphor;

// Independently implemented SI radiative-transfer equations. The low-D LUT
// decomposition follows Hillaire2020; no external HLSL/noise asset is copied.
constexpr sampler kAtmosphereSampler(coord::normalized,filter::linear,address::clamp_to_edge);
static float3 atmoVec(constant float* v){return float3(v[0],v[1],v[2]);}
static float4x4 atmoMatrix(constant float* m){return float4x4(float4(m[0],m[1],m[2],m[3]),float4(m[4],m[5],m[6],m[7]),float4(m[8],m[9],m[10],m[11]),float4(m[12],m[13],m[14],m[15]));}
static float2 atmoSphere(float3 point,float3 ray,float radius) {
    const float r=length(point),b=dot(point,ray),c=(r-radius)*(r+radius),disc=b*b-c;
    if(disc<0)return float2(-1);
    const float root=sqrt(max(0.0f,disc)),q=-b-(b>=0?root:-root);
    if(abs(q)<1e-20f)return float2(-b);
    const float other=c/q;return float2(min(q,other),max(q,other));
}
struct AtmoSegment {float begin,end;bool ground,valid;};
static AtmoSegment atmoSegment(float3 world,float3 ray,constant GPUAtmosphereParams& p,float limit=1e8f) {
    const float3 point=world-atmoVec(p.planetCenter);const float r=length(point);
    if(r<p.bottomRadius-0.5f)return {0,0,true,false};
    if(r<=p.bottomRadius && dot(point,ray)<0)return {0,0,true,true};
    const float2 top=atmoSphere(point,ray,p.topRadius);if(top.y<=0)return {0,0,false,false};
    AtmoSegment s{max(0.0f,top.x),min(top.y,limit),false,true};
    const float2 bottom=atmoSphere(point,ray,p.bottomRadius);
    if(bottom.x>1e-3f && bottom.x>=s.begin && bottom.x<=s.end){s.end=bottom.x;s.ground=true;}
    s.valid=s.end>=s.begin;return s;
}
struct AtmoMedium {float3 rayleigh,mie,extinction;};
static AtmoMedium atmoMedium(float3 world,constant GPUAtmosphereParams& p) {
    const float h=max(0.0f,length(world-atmoVec(p.planetCenter))-p.bottomRadius);
    if(h>p.topRadius-p.bottomRadius)return {float3(0),float3(0),float3(0)};
    const float rayDensity=exp(-h/p.rayleighScaleHeight),mieDensity=exp(-h/p.mieScaleHeight);
    const float ozone=saturate(1.0f-abs(h-p.ozoneCenterHeight)/p.ozoneHalfWidth);
    AtmoMedium m;m.rayleigh=atmoVec(p.rayleighScattering)*rayDensity;m.mie=atmoVec(p.mieScattering)*mieDensity;
    m.extinction=m.rayleigh+m.mie+p.mieAbsorption*mieDensity+atmoVec(p.ozoneAbsorption)*ozone;return m;
}
static float3 atmoIntegralFactor(float3 sigma,float ds) {
    return float3(sigma.x>1e-8f?-expm1(-sigma.x*ds)/sigma.x:ds,
                  sigma.y>1e-8f?-expm1(-sigma.y*ds)/sigma.y:ds,sigma.z>1e-8f?-expm1(-sigma.z*ds)/sigma.z:ds);
}
static float atmoRayleighPhase(float mu){return 3.0f/(16.0f*M_PI_F)*(1+mu*mu);}
static float atmoHgPhase(float mu,float g){return (1-g*g)/(4*M_PI_F*pow(max(1e-8f,1+g*g-2*g*mu),1.5f));}
static float2 atmoTransUv(float r,float mu,constant GPUAtmosphereParams& p) {
    r=clamp(r,p.bottomRadius,p.topRadius);mu=clamp(mu,-1.0f,1.0f);
    const float H=sqrt((p.topRadius-p.bottomRadius)*(p.topRadius+p.bottomRadius));
    const float rho=sqrt(max(0.0f,(r-p.bottomRadius)*(r+p.bottomRadius)));
    const float distance=-r*mu+sqrt(max(0.0f,r*r*(mu*mu-1)+p.topRadius*p.topRadius));
    const float dmin=p.topRadius-r,dmax=rho+H;
    return float2(saturate((distance-dmin)/max(dmax-dmin,1e-6f)),rho/H);
}
static float2 atmoLutCoord(float2 unit,uint2 extent){return (saturate(unit)*float2(extent-1u)+0.5f)/float2(extent);}
static float3 atmoTransmittance(float3 world,float3 ray,constant GPUAtmosphereParams& p,texture2d<float> lut) {
    const float3 relative=world-atmoVec(p.planetCenter);const float r=length(relative);
    if(r<p.bottomRadius)return 0;
    const AtmoSegment segment=atmoSegment(world,ray,p);
    if(segment.ground)return 0;
    if(!segment.valid)return 1;
    const float3 sample=world+ray*segment.begin-atmoVec(p.planetCenter);
    const float height=length(sample),mu=dot(sample,ray)/max(height,1e-6f);
    return lut.sample(kAtmosphereSampler,atmoLutCoord(atmoTransUv(height,mu,p),uint2(p.transmittanceWidth,p.transmittanceHeight))).rgb;
}
static float3 atmoMultiple(float3 world,float3 towardLight,constant GPUAtmosphereParams& p,texture2d<float> lut) {
    const float3 relative=world-atmoVec(p.planetCenter);const float r=length(relative);
    const float2 uv(dot(relative,towardLight)/max(r,1e-6f)*0.5f+0.5f,(r-p.bottomRadius)/(p.topRadius-p.bottomRadius));
    return lut.sample(kAtmosphereSampler,atmoLutCoord(uv,uint2(p.multiWidth,p.multiHeight))).rgb;
}
struct AtmoIntegral {float3 radiance,transmittance,scatteringFactor;bool ground;};
static AtmoIntegral atmoIntegrate(float3 world,float3 ray,constant GPUAtmosphereParams& p,texture2d<float> trans,
                                  texture2d<float> multi,bool multiple,float limit=1e8f,bool isotropic=false,
                                  float3 canonicalSun=float3(0),bool unitIrradiance=false) {
    AtmoIntegral out{float3(0),float3(1),float3(0),false};
    const AtmoSegment segment=atmoSegment(world,ray,p,limit);out.ground=segment.ground;if(!segment.valid)return out;
    const float3 sun=unitIrradiance?canonicalSun:atmoVec(p.sunDirection);
    const float3 solar=unitIrradiance?float3(1):atmoVec(p.sunIrradiance),moon=atmoVec(p.moonDirection);
    const uint steps=clamp(p.marchSteps,1u,512u);
    for(uint i=0;i<steps;++i) {
        const float a=float(i)/float(steps),b=float(i+1)/float(steps),span=segment.end-segment.begin;
        const float t0=segment.begin+span*a*a,t1=segment.begin+span*b*b,ds=t1-t0;
        const float3 point=world+ray*((t0+t1)*0.5f);const AtmoMedium m=atmoMedium(point,p);
        const float3 scattering=m.rayleigh+m.mie;
        const float3 phase=isotropic?scattering/(4*M_PI_F):m.rayleigh*atmoRayleighPhase(dot(ray,sun))+m.mie*atmoHgPhase(dot(ray,sun),p.mieG);
        float3 source=solar*atmoTransmittance(point,sun,p,trans)*phase;
        if(multiple)source+=solar*atmoMultiple(point,sun,p,multi)*scattering;
        if(!unitIrradiance && (p.flags&ATMOSPHERE_ENABLE_MOON)) {
            const float3 moonPhase=m.rayleigh*atmoRayleighPhase(dot(ray,moon))+m.mie*atmoHgPhase(dot(ray,moon),p.mieG);
            source+=atmoVec(p.moonIrradiance)*(atmoTransmittance(point,moon,p,trans)*moonPhase+
                (multiple?atmoMultiple(point,moon,p,multi)*scattering:float3(0)));
        }
        const float3 factor=atmoIntegralFactor(m.extinction,ds);
        out.radiance+=out.transmittance*source*factor;out.scatteringFactor+=out.transmittance*scattering*factor;
        out.transmittance*=exp(-m.extinction*ds);
    }
    if(segment.ground) {
        const float3 point=world+ray*segment.end,up=normalize(point-atmoVec(p.planetCenter));
        out.radiance+=out.transmittance*atmoVec(p.groundAlbedo)/M_PI_F*solar*atmoTransmittance(point+up*0.5f,sun,p,trans)*max(0.0f,dot(up,sun));
        if(isotropic)out.scatteringFactor+=out.transmittance*atmoVec(p.groundAlbedo);
        if(!unitIrradiance && (p.flags&ATMOSPHERE_ENABLE_MOON))out.radiance+=out.transmittance*atmoVec(p.groundAlbedo)/M_PI_F*
            atmoVec(p.moonIrradiance)*atmoTransmittance(point+up*0.5f,moon,p,trans)*max(0.0f,dot(up,moon));
    }
    return out;
}
static void atmoBasis(float3 up,thread float3& east,thread float3& north){east=normalize(cross(abs(up.z)<0.9f?float3(0,0,1):float3(1,0,0),up));north=cross(up,east);}
static float atmoHorizon(float height,float bottom){return acos(-sqrt(max(0.0f,(height-bottom)*(height+bottom)))/max(height,1e-6f));}
static float3 atmoSkyDirection(float2 uv,float3 up,float height,constant GPUAtmosphereParams& p) {
    const float horizon=atmoHorizon(height,p.bottomRadius);
    const float theta=uv.y<0.5f?horizon*(1-pow(1-2*uv.y,2.0f)):horizon+(M_PI_F-horizon)*pow(2*uv.y-1,2.0f);
    const float phi=uv.x*2*M_PI_F;float3 east,north;atmoBasis(up,east,north);
    return up*cos(theta)+(east*cos(phi)+north*sin(phi))*sin(theta);
}
static float2 atmoSkyUv(float3 ray,float3 up,float height,constant GPUAtmosphereParams& p) {
    const float horizon=atmoHorizon(height,p.bottomRadius),theta=acos(clamp(dot(up,ray),-1.0f,1.0f));float3 east,north;atmoBasis(up,east,north);
    const float phi=atan2(dot(ray,north),dot(ray,east));
    const float v=theta<horizon?0.5f*(1-sqrt(max(0.0f,1-theta/horizon))):0.5f+0.5f*sqrt(max(0.0f,(theta-horizon)/max(M_PI_F-horizon,1e-6f)));
    return float2(fract(phi/(2*M_PI_F)+1),v);
}
static float3 atmoPixelRay(uint2 pixel,uint2 extent,constant float* inverse,float3 camera) {
    const float2 ndc=(float2(pixel)+0.5f)*float2(2,-2)/float2(extent)+float2(-1,1);
    const float4 h=atmoMatrix(inverse)*float4(ndc,1,1);return normalize(h.xyz/h.w-camera);
}
static float atmoOpaqueDistance(uint2 pixel,uint2 extent,texture2d<float,access::read> depth,constant float* inverse,float3 camera,float maximum) {
    const float z=depth.read(pixel).x;if(!(z>0&&z<=1))return maximum;
    const float2 ndc=(float2(pixel)+0.5f)*float2(2,-2)/float2(extent)+float2(-1,1);
    const float4 h=atmoMatrix(inverse)*float4(ndc,z,1);return all(isfinite(h))&&abs(h.w)>1e-20f?min(maximum,length(h.xyz/h.w-camera)):maximum;
}
static void volumeCount(device atomic_uint* c,uint word,uint value){if(value)atomic_fetch_add_explicit(c+word,value,memory_order_relaxed);}
