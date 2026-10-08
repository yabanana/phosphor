#include "gi_cache_common.h"

// Correctness-first serial update, dispatch exactly ONE lane. Previous readers
// finish via the graph before this write; subsequent readers follow it.
// At most min(extra.cacheUpdateCount,256) candidates are considered per frame.
// Open addressing/eviction is bounded by cacheProbeLimit. No locks, RGB atomics,
// empty-state publish race or spin wait. A future parallel update needs evidence.
kernel void radiance_cache_update(constant GPUProbeGridParams& p [[buffer(0)]],
                                  device GPURadianceCacheEntry* cache [[buffer(1)]],
                                  const device GPUGiReservoir* candidates [[buffer(2)]],
                                  constant GPUProbeTraceExtra& extra [[buffer(3)]],
                                  uint tid [[thread_position_in_grid]]) {
    if(tid || !p.cacheCapacity) return;
    if(p.reset) for(uint lane=0;lane<p.cacheCapacity;++lane) cache[lane]={};
    uint updateCount=min(extra.cacheUpdateCount,256u),candidateCount=extra.cacheCandidateCount;
    if(!candidateCount || !p.cacheProbeLimit) return;
    uint begin=uint((ulong(p.frameIndex)*ulong(updateCount))%ulong(candidateCount));
    for(uint c=0;c<min(updateCount,candidateCount);++c) {
        GPUGiReservoir sample=candidates[(begin+c)%candidateCount];
        if(!(sample.flags&GI_SAMPLE_VALID) || sample.slot==~0u ||
            sample.geometryRevision!=p.geometryRevision || sample.lightRevision!=p.lightRevision ||
            sample.materialRevision!=p.materialRevision) continue;
        float3 point(sample.position[0],sample.position[1],sample.position[2]),normal(sample.normal[0],sample.normal[1],sample.normal[2]);
        float3 source(sample.sourcePosition[0],sample.sourcePosition[1],sample.sourcePosition[2]);
        float3 L(sample.radiance[0],sample.radiance[1],sample.radiance[2]);
        GPURadianceCacheEntry key;
        if(!all(isfinite(L)) || any(L<0.0f) || !giCacheKey(point,normal,source-point,p,key)) continue;
        uint start=giCacheHash(key)%p.cacheCapacity,victim=start,stale=~0u,oldest=0u,match=~0u;
        for(uint step=0;step<min(p.cacheProbeLimit,p.cacheCapacity);++step) {
            uint lane=(start+step)%p.cacheCapacity;
            GPURadianceCacheEntry e=cache[lane];
            // Reset clears the storage before insert; newly written lanes of
            // this reset frame may still be compared and averaged.
            bool current=e.state==1u && e.samples && e.generation==p.cacheGeneration &&
                e.geometryRevision==p.geometryRevision && e.lightRevision==p.lightRevision &&
                e.materialRevision==p.materialRevision && p.frameIndex-e.lastFrame<=p.cacheMaxAge;
            if(!current) {if(stale==~0u) stale=lane;continue;}
            if(giCacheSame(e,key)) {match=lane;break;}
            uint age=p.frameIndex-e.lastFrame;
            if(step==0u || age>oldest) {oldest=age;victim=lane;}
        }
        if(match!=~0u) {
            GPURadianceCacheEntry e=cache[match];
            float3 old(e.radiance[0],e.radiance[1],e.radiance[2]);
            L=mix(old,L,1.0f/float(min(e.samples,63u)+1u));
            e.radiance[0]=L.x;e.radiance[1]=L.y;e.radiance[2]=L.z;e.lastFrame=p.frameIndex;e.samples=min(e.samples+1u,64u);
            cache[match]=e;
        } else {
            if(stale!=~0u) victim=stale;
            key.radiance[0]=L.x;key.radiance[1]=L.y;key.radiance[2]=L.z;
            key.generation=p.cacheGeneration;key.lastFrame=p.frameIndex;key.samples=1u;key.state=1u;
            key.geometryRevision=p.geometryRevision;key.lightRevision=p.lightRevision;key.materialRevision=p.materialRevision;
            cache[victim]=key;
        }
    }
}
