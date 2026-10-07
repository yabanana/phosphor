#pragma once
// Original F14 procedural field, version1. No downloaded texture, copied
// generator, or third-party noise code. Periodic lattice value noise + F1
// Worley distance; all loops and neighbourhoods have explicit finite bounds.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#define VN_FN static __attribute__((unused))
#define VN_FLOOR(x) metal::floor(x)
#define VN_SQRT(x) metal::sqrt(x)
#define VN_MIN(a,b) metal::min(a,b)
#define VN_MAX(a,b) metal::max(a,b)
#else
#include <cmath>
#include <algorithm>
#define VN_FN inline
#define VN_FLOOR(x) std::floor(x)
#define VN_SQRT(x) std::sqrt(x)
#define VN_MIN(a,b) std::min(a,b)
#define VN_MAX(a,b) std::max(a,b)
#endif
#include "renderer/gpu_types.h"
namespace phosphor {
VN_FN u32 volumeHash(u32 x,u32 y,u32 z,u32 seed) {
    u32 h=seed^(x*73856093u)^(y*19349663u)^(z*83492791u);
    h=h*1664525u+1013904223u;h^=h>>16;h*=2246822519u;return h^(h>>13);
}
VN_FN float volumeRandom(u32 h){return float(h>>8)*(1.0f/16777216.0f);}
VN_FN float volumeWrap(float x){return x-VN_FLOOR(x/256.0f)*256.0f;}
VN_FN float volumeFade(float x){return x*x*x*(x*(x*6.0f-15.0f)+10.0f);}
VN_FN float volumeLerp(float a,float b,float x){return a+(b-a)*x;}
VN_FN float volumeValueNoise(float x,float y,float z,u32 seed) {
    x=volumeWrap(x);y=volumeWrap(y);z=volumeWrap(z);
    const u32 ix=u32(VN_FLOOR(x)),iy=u32(VN_FLOOR(y)),iz=u32(VN_FLOOR(z));
    const float fx=volumeFade(x-float(ix)),fy=volumeFade(y-float(iy)),fz=volumeFade(z-float(iz));
    float value[8];
    for(u32 i=0;i<8;++i)value[i]=volumeRandom(volumeHash((ix+(i&1u))&255u,(iy+((i>>1u)&1u))&255u,(iz+((i>>2u)&1u))&255u,seed));
    return volumeLerp(volumeLerp(volumeLerp(value[0],value[1],fx),volumeLerp(value[2],value[3],fx),fy),
                      volumeLerp(volumeLerp(value[4],value[5],fx),volumeLerp(value[6],value[7],fx),fy),fz);
}
VN_FN float volumeWorley(float x,float y,float z,u32 seed) {
    x=volumeWrap(x);y=volumeWrap(y);z=volumeWrap(z);
    const int ix=int(VN_FLOOR(x)),iy=int(VN_FLOOR(y)),iz=int(VN_FLOOR(z));
    float squared=3.0f;
    for(int dz=-1;dz<=1;++dz)for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx) {
        const u32 h=volumeHash(u32(ix+dx)&255u,u32(iy+dy)&255u,u32(iz+dz)&255u,seed);
        const float px=float(ix+dx)+volumeRandom(h),py=float(iy+dy)+volumeRandom(volumeHash(h,1,0,seed)),
                    pz=float(iz+dz)+volumeRandom(volumeHash(h,2,0,seed));
        const float ax=px-x,ay=py-y,az=pz-z;squared=VN_MIN(squared,ax*ax+ay*ay+az*az);
    }
    return VN_MIN(1.0f,VN_SQRT(squared/3.0f));
}
VN_FN float volumeFbm(float x,float y,float z,u32 seed) {
    float value=0,weight=0.5f,total=0;
    for(u32 octave=0;octave<5;++octave){value+=weight*volumeValueNoise(x,y,z,seed+octave*1013u);total+=weight;x*=2;y*=2;z*=2;weight*=0.5f;}
    return value/total;
}
VN_FN float volumeSmooth(float a,float b,float x){const float t=VN_MIN(1.0f,VN_MAX(0.0f,(x-a)/(b-a)));return t*t*(3.0f-2.0f*t);}
VN_FN float volumeCloudShape(float x,float y,float z,float altitude,float base,float top,float coverage,
                              float densityScale,float shapeScale,float erosionScale,u32 seed) {
    if(coverage<=0 || altitude<=base || altitude>=top)return 0;
    const float height=(altitude-base)/(top-base);
    const float profile=volumeSmooth(0.0f,0.2f,height)*(1.0f-volumeSmooth(0.7f,1.0f,height));
    const float shape=0.75f*volumeFbm(x*shapeScale,y*shapeScale,z*shapeScale,seed)+
                      0.25f*(1.0f-volumeWorley(x*shapeScale,y*shapeScale,z*shapeScale,seed+17u));
    const float erode=volumeWorley(x*erosionScale,y*erosionScale,z*erosionScale,seed+29u)*0.35f;
    const float d=(shape-(1.0f-coverage))/VN_MAX(coverage,0.001f)-erode;
    return profile*densityScale*VN_MIN(1.0f,VN_MAX(0.0f,d));
}
} // namespace phosphor
#undef VN_FN
#undef VN_FLOOR
#undef VN_SQRT
#undef VN_MIN
#undef VN_MAX
