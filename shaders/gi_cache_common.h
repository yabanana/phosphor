#pragma once
#include "gi_common.h"

inline uint giDirectionBin(float3 d) {
    uint2 b=uint2(clamp(giOctEncode(normalize(d))*0.5f+0.5f,0.0f,0.999999f)*16.0f);
    return b.x|(b.y<<4u);
}
inline float giCellSize(constant GPUProbeGridParams& p) { return min(min(p.spacing[0],p.spacing[1]),p.spacing[2])*0.25f; }
inline bool giCacheKey(float3 point,float3 normal,float3 outgoing,constant GPUProbeGridParams& p,
                        thread GPURadianceCacheEntry& key) {
    if(!all(isfinite(point)) || !all(isfinite(normal)) || !all(isfinite(outgoing)) ||
        dot(normal,normal)<1e-20f || dot(outgoing,outgoing)<1e-20f || giCellSize(p)<=0.0f) return false;
    float3 cell=floor(point/giCellSize(p));
    // Shader float cannot represent INT_MAX exactly. Reject the boundary.
    if(any(cell<=-2147483648.0f) || any(cell>=2147483520.0f)) return false;
    uint3 bits=as_type<uint3>(int3(cell));
    key={};key.cellX=bits.x;key.cellY=bits.y;key.cellZ=bits.z;
    key.normalBin=giDirectionBin(normal);key.directionBin=giDirectionBin(outgoing);
    return true;
}
inline uint giCacheHash(GPURadianceCacheEntry k) {
    return giHash(k.cellX)^giHash(k.cellY+0x9e3779b9u)^giHash(k.cellZ+0x85ebca6bu)^
        giHash(k.normalBin+0xc2b2ae35u)^giHash(k.directionBin+0x27d4eb2fu);
}
inline bool giCacheSame(GPURadianceCacheEntry a,GPURadianceCacheEntry b) {
    return a.cellX==b.cellX && a.cellY==b.cellY && a.cellZ==b.cellZ && a.normalBin==b.normalBin && a.directionBin==b.directionBin;
}
inline bool giCacheCurrent(GPURadianceCacheEntry e,constant GPUProbeGridParams& p) {
    return !p.reset && e.state==1u && e.samples>0u && e.generation==p.cacheGeneration &&
        e.geometryRevision==p.geometryRevision && e.lightRevision==p.lightRevision &&
        e.materialRevision==p.materialRevision && p.frameIndex-e.lastFrame<=p.cacheMaxAge;
}
inline bool giCacheLookup(float3 point,float3 normal,float3 outgoing,constant GPUProbeGridParams& p,
                          const device GPURadianceCacheEntry* cache,thread float3& L) {
    if(!p.cacheCapacity || !p.cacheProbeLimit) return false;
    GPURadianceCacheEntry key;
    if(!giCacheKey(point,normal,outgoing,p,key)) return false;
    uint start=giCacheHash(key)%p.cacheCapacity;
    for(uint i=0;i<min(p.cacheProbeLimit,p.cacheCapacity);++i) {
        GPURadianceCacheEntry e=cache[(start+i)%p.cacheCapacity];
        if(giCacheCurrent(e,p) && giCacheSame(e,key)) {
            L=float3(e.radiance[0],e.radiance[1],e.radiance[2]);
            return all(isfinite(L)) && all(L>=0.0f);
        }
    }
    return false;
}
