#pragma once
#include <metal_stdlib>
#include "renderer/gpu_types.h"
using namespace metal;

// Independent raw specular reduction, shared by reflection_reduce (N samples
// read back from the per-sample records) and by the single-sample kernels,
// which apply it to their one sample in registers (REFLECTION_SINGLE_SAMPLE):
// the stored record, radiance, distance and error count are the same either way.
// Average contributions, not already-denoised signals. Mixed paths report
// hitDistance 0 (unknown), no fabricated mean geometry point. Every raw
// finite/error condition is recorded BEFORE replacing bad output with black.
struct ReflectionReduction {GPUSpecularSample out;float3 sum;float hitDistance;bool selected,mixed,error;};
inline ReflectionReduction reflectionReduceBegin(){ReflectionReduction r;r.out=GPUSpecularSample{};r.out.secondarySlot=~0u;r.sum=0;r.hitDistance=0;r.selected=false;r.mixed=false;r.error=false;return r;}
inline void reflectionReduceAdd(thread ReflectionReduction& r,GPUSpecularSample s,uint count){const float3 L(s.radiance[0],s.radiance[1],s.radiance[2]);
    const bool bad=(s.flags&SPECULAR_SAMPLE_ERROR)||!all(isfinite(L))||any(L<0)||!isfinite(s.hitDistance)||s.hitDistance<0;
    r.error|=bad;if(!bad)r.sum+=L/float(count);
    if(s.flags&SPECULAR_SAMPLE_VALID){if(!r.selected){r.out=s;r.selected=true;}else r.mixed|=r.out.path!=s.path||r.out.secondarySlot!=s.secondarySlot||r.out.secondaryGeneration!=s.secondaryGeneration;
        r.hitDistance+=s.hitDistance/float(count);}}
// counts[1] = raw numerical errors (independent check word).
inline GPUSpecularSample reflectionReduceEnd(thread ReflectionReduction& r,device atomic_uint* counts){GPUSpecularSample out=r.out;float3 sum=r.sum;
    const bool error=r.error||!all(isfinite(sum));if(error){atomic_fetch_add_explicit(counts+1,1u,memory_order_relaxed);sum=0;out.flags|=SPECULAR_SAMPLE_ERROR;}
    if(r.mixed){out.flags|=SPECULAR_SAMPLE_MIXED;out.secondarySlot=~0u;out.secondaryGeneration=0;out.hitDistance=0;}else out.hitDistance=r.hitDistance;
    for(uint i=0;i<3;++i)out.radiance[i]=sum[i];return out;}
// Final store of one pixel's raw specular signal: the per-sample record
// (multi-sample: reduced later) or the reduced record (single sample).
inline void reflectionStoreSample(uint flags,uint tid,uint width,GPUSpecularSample sample,device GPUSpecularSample* records,device atomic_uint* counts,
    texture2d<float,access::write> radiance,texture2d<float,access::write> distance){
    if(flags&REFLECTION_SINGLE_SAMPLE){auto r=reflectionReduceBegin();reflectionReduceAdd(r,sample,1u);sample=reflectionReduceEnd(r,counts);}
    records[tid]=sample;const uint2 pixel(tid%width,tid/width);
    radiance.write(float4(sample.radiance[0],sample.radiance[1],sample.radiance[2],1),pixel);distance.write(float4(sample.hitDistance),pixel);}
