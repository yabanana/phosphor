#include "restir_common.h"
kernel void reflection_counter_clear(device atomic_uint* counts [[buffer(0)]],uint tid [[thread_position_in_grid]]){
    if(tid<16)atomic_store_explicit(counts+tid,0u,memory_order_relaxed);
}
// Metadata is per-sample, each block stridePixels long. Average contributions,
// not already-denoised signals. Mixed paths report hitDistance0 (unknown), no
// fabricated mean geometry point. Every raw finite/error condition is recorded
// BEFORE replacing bad numerical output with black.
kernel void reflection_reduce(constant GPUReflectionReduceParams& p [[buffer(0)]],const device GPUSpecularSample* samples [[buffer(1)]],
    device GPUSpecularSample* metadata [[buffer(2)]],device atomic_uint* counts [[buffer(3)]],texture2d<float,access::write> radiance [[texture(0)]],
    texture2d<float,access::write> distance [[texture(1)]],uint tid [[thread_position_in_grid]]){
    if(tid>=p.width*p.height)return;GPUSpecularSample out{};out.secondarySlot=~0u;float3 sum=0;float hitDistance=0;bool selected=false,mixed=false,error=false;
    const uint count=clamp(p.samples,1u,8u);for(uint i=0;i<count;++i){const GPUSpecularSample s=samples[i*p.stridePixels+tid];const float3 L(s.radiance[0],s.radiance[1],s.radiance[2]);
        const bool bad=(s.flags&SPECULAR_SAMPLE_ERROR)||!all(isfinite(L))||any(L<0)||!isfinite(s.hitDistance)||s.hitDistance<0;
        error|=bad;if(!bad)sum+=L/float(count);
        if(s.flags&SPECULAR_SAMPLE_VALID){if(!selected){out=s;selected=true;}else mixed|=out.path!=s.path||out.secondarySlot!=s.secondarySlot||out.secondaryGeneration!=s.secondaryGeneration;
            hitDistance+=s.hitDistance/float(count);}}
    error|=!all(isfinite(sum));if(error){atomic_fetch_add_explicit(counts+1,1u,memory_order_relaxed);sum=0;out.flags|=SPECULAR_SAMPLE_ERROR;}
    if(mixed){out.flags|=SPECULAR_SAMPLE_MIXED;out.secondarySlot=~0u;out.secondaryGeneration=0;out.hitDistance=0;}else out.hitDistance=hitDistance;
    for(uint i=0;i<3;++i)out.radiance[i]=sum[i];metadata[tid]=out;const uint2 pixel(tid%p.width,tid/p.width);
    radiance.write(float4(sum,1),pixel);distance.write(float4(out.hitDistance),pixel);
}
// Clears inactive raw signals; metadata is not touched for disabled specular.
kernel void reflection_signal_zero(constant GPUReflectionComposeParams& p [[buffer(0)]],device GPUSpecularSample* metadata [[buffer(1)]],texture2d<float,access::write> specular [[texture(0)]],
    texture2d<float,access::write> ao [[texture(1)]],texture2d<float,access::write> distance [[texture(2)]],uint tid [[thread_position_in_grid]]){
    if(tid>=p.width*p.height)return;const uint2 q(tid%p.width,tid/p.width);GPUSpecularSample empty{};empty.secondarySlot=~0u;metadata[tid]=empty;
    specular.write(float4(0,0,0,1),q);ao.write(float4(1),q);distance.write(float4(0),q);
}
// Validates the actual raw capture or a filtered mip viewed as a 2DArray.
// Raw failure survives sanitization via alpha=-1 written by capture shaders.
kernel void reflection_probe_validate(constant GPUProbeFilterParams& p [[buffer(0)]],device atomic_uint* counts [[buffer(1)]],
    texture2d_array<float,access::read> faces [[texture(0)]],uint tid [[thread_position_in_grid]]){
    if(tid>=p.side*p.side*6u)return;const uint face=tid/(p.side*p.side),pixel=tid%(p.side*p.side);
    const float4 value=faces.read(uint2(pixel%p.side,pixel/p.side),6u*p.cubeIndex+face);
    if(!all(isfinite(value))||any(value.rgb<0)||value.a<0)atomic_fetch_add_explicit(counts+(p.mip==~0u?2u:3u),1u,memory_order_relaxed);
}
// Metadata is GPU-published only after every source face and destination mip.
// The persistent write also preserves cross-frame ordering of the parent/view
// atlas resources. source0,outMetadata1,counts2; filteredCubeArray texture0.
kernel void reflection_probe_ready(constant GPUReflectionProbe& source [[buffer(0)]],device GPUReflectionProbe* metadata [[buffer(1)]],
    device atomic_uint* counts [[buffer(2)]],device uint* fault [[buffer(3)]],texturecube_array<float> cube [[texture(0)]],uint tid [[thread_position_in_grid]]){
    if(tid)return;GPUReflectionProbe out=source;constexpr sampler sampling(filter::nearest,mip_filter::nearest);
    const float3 sample=cube.sample(sampling,float3(1,0,0),source.cubeIndex,level(0)).rgb;
    if(fault[0]!=source.generation){fault[0]=source.generation;fault[1]=0;}
    fault[1]|=atomic_load_explicit(counts+2,memory_order_relaxed)!=0||atomic_load_explicit(counts+3,memory_order_relaxed)!=0||!all(isfinite(sample));
    if(fault[1])atomic_fetch_add_explicit(counts+3,1u,memory_order_relaxed);
    out.enabled=source.enabled&&!fault[1];
    metadata[0]=out;
}
kernel void denoise_history_corrupt_safe(constant GPUDenoiseParams& p [[buffer(0)]],device GPUDenoiseHistory* history [[buffer(1)]],uint tid [[thread_position_in_grid]]){
    if(!(p.flags&DENOISE_RESET)&&tid<p.width*p.height&&history[tid].valid)history[tid].viewID^=1u;
}
// Output-history domain negative. The independent denoise_check observes the
// real view mismatch; no synthetic error flag or counter is injected here.
kernel void denoise_next_foreign_view(constant GPUDenoiseParams& p [[buffer(0)]],device GPUDenoiseHistory* history [[buffer(1)]],uint tid [[thread_position_in_grid]]){
    if(tid<p.width*p.height&&history[tid].valid)history[tid].viewID^=1u;
}
// Codes2/3 operate on F13-owned copies. Original Direct/F8 guides and motion
// remain intact. Only valid receiver pixels are poisoned; background is copied.
kernel void reflection_input_poison(constant GPUReflectionComposeParams& p [[buffer(0)]],const device GPUDISurface* input [[buffer(1)]],
    device GPUDISurface* output [[buffer(2)]],texture2d<float,access::read> motion [[texture(0)]],texture2d<float,access::write> poisonedMotion [[texture(1)]],uint tid [[thread_position_in_grid]]){
    if(tid>=p.width*p.height)return;const uint2 pixel(tid%p.width,tid/p.width);GPUDISurface s=input[tid];float2 velocity=motion.read(pixel).xy;
    if(s.valid&&p.pad[1]==2u)velocity.x=as_type<float>(0x7fc00000u);
    if(p.pad[1]==3u){if(s.valid){s.geometricNormal[0]=as_type<float>(0x7fc00000u);s.shadingNormal[0]=as_type<float>(0x7fc00000u);}output[tid]=s;}
    poisonedMotion.write(float4(velocity,0,0),pixel);
}
// Independent input validator BEFORE transport/filtering: flags are irrelevant
// to this check; it directly inspects the consumed guide and motion numerics.
kernel void reflection_input_check(constant GPUReflectionComposeParams& p [[buffer(0)]],const device GPUDISurface* surfaces [[buffer(1)]],
    device atomic_uint* counts [[buffer(2)]],texture2d<float,access::read> motion [[texture(0)]],uint tid [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){
    if(tid>=p.width*p.height)return;const GPUDISurface s=surfaces[tid];const float3 geometric(s.geometricNormal[0],s.geometricNormal[1],s.geometricNormal[2]),normal(s.shadingNormal[0],s.shadingNormal[1],s.shadingNormal[2]);
    const float2 velocity=motion.read(uint2(tid%p.width,tid/p.width)).xy;const bool bad=s.valid&&(!all(isfinite(geometric))||!all(isfinite(normal))||dot(geometric,geometric)<=0||dot(normal,normal)<=0||!all(isfinite(velocity)));
    const uint failures=simd_sum(bad?1u:0u);if(lane==0&&failures)atomic_fetch_add_explicit(counts+5,failures,memory_order_relaxed);
    // Diagnostic classification only: counts5 and its acceptance rule are
    // unchanged. counts7 OR bits: geo nonfinite1/zero2, shading nonfinite4/zero8,
    // motion nonfinite16. Never repair or exclude a malformed valid receiver.
    if(s.valid){uint reasons=0;
        if(!all(isfinite(geometric)))reasons|=1u;
        if(dot(geometric,geometric)<=0)reasons|=2u;
        if(!all(isfinite(normal)))reasons|=4u;
        if(dot(normal,normal)<=0)reasons|=8u;
        if(!all(isfinite(velocity))){reasons|=16u;
            // One winning invocation records the exact bits read from motion.
            // All writes finish before the existing per-slot CPU readback.
            if(atomic_fetch_add_explicit(counts+8,1u,memory_order_relaxed)==0u){
                atomic_store_explicit(counts+9,tid,memory_order_relaxed);
                atomic_store_explicit(counts+10,as_type<uint>(velocity.x),memory_order_relaxed);
                atomic_store_explicit(counts+11,as_type<uint>(velocity.y),memory_order_relaxed);
            }
        }
        if(reasons)atomic_fetch_or_explicit(counts+7,reasons,memory_order_relaxed);
    }
}
// residual0,DIselected2,GIselectedE4,specular5,AO6,output7.
// Root RESOLVE_EXTERNAL_DIFFUSE bit32 strips external DI/GI/hemisphere diffuse
// from the primary residual. Assemble positive terms ONCE, never subtract an
// FP32 signal from a quantized residual. Pre-reflection raw assembly uses this
// same kernel with DI/GI only flags, SPEC/AO off and unfiltered inputs.
kernel void reflection_composite(constant GPUReflectionComposeParams& p [[buffer(0)]],const device GPUDISurface* surfaces [[buffer(1)]],
    device atomic_uint* counts [[buffer(2)]],texture2d<float,access::read> base [[texture(0)]],
    texture2d<float,access::read> diSelected [[texture(2)]],texture2d<float,access::read> giSelected [[texture(4)]],
    texture2d<float,access::read> specular [[texture(5)]],texture2d<float,access::read> ao [[texture(6)]],texture2d<float,access::write> output [[texture(7)]],uint tid [[thread_position_in_grid]]){
    if(tid>=p.backingWidth*p.backingHeight)return;const uint2 pixel(tid%p.backingWidth,tid/p.backingWidth);float3 color=base.read(pixel).rgb;
    if(all(pixel<uint2(p.width,p.height))){const GPUDISurface s=surfaces[pixel.y*p.width+pixel.x];
        if(s.valid){if(p.flags&REFLECT_COMPOSE_DI)color+=diSelected.read(pixel).rgb;
            if(p.flags&REFLECT_COMPOSE_GI)color+=giSelected.read(pixel).rgb*diVec(s.albedo)*(1-s.metallic)/M_PI_F;
            if(p.flags&REFLECT_COMPOSE_SPEC)color+=specular.read(pixel).rgb;
            if(!(p.flags&REFLECT_COMPOSE_GI)){const float3 n=normalize(diVec(s.shadingNormal));
                const float3 sky(.30f,.36f,.45f),ground(.10f,.09f,.08f);const float3 ambient=mix(ground,sky,n.y*.5f+.5f)*diVec(s.albedo)*(1-s.metallic)*as_type<float>(s.pad[0]);
                color+=ambient*((p.flags&REFLECT_COMPOSE_AO)?saturate(ao.read(pixel).x):1.0f);}}
    }
    if(!all(isfinite(color))||any(color<-.00001f)||(p.pad[0]&&any(color>65504.0f))){atomic_fetch_add_explicit(counts+4,1u,memory_order_relaxed);color=0;}
    output.write(float4(max(color,0.0f),1),pixel);
}
// Capture8 exports the actual GI denoiser result E as reflected diffuse Lo.
// Current receiver material is applied once, matching gi_reference_diffuse;
// no AO, exposure, other signal, or numerical sanitization alters this export.
kernel void reflection_filtered_indirect_diffuse(constant GPUReflectionComposeParams& p [[buffer(0)]],
    const device GPUDISurface* surfaces [[buffer(1)]],texture2d<float,access::read> irradiance [[texture(0)]],
    texture2d<float,access::write> output [[texture(1)]],uint tid [[thread_position_in_grid]]){
    if(tid>=p.width*p.height)return;const uint2 pixel(tid%p.width,tid/p.width);const GPUDISurface s=surfaces[tid];
    const float3 Lo=s.valid?irradiance.read(pixel).rgb*diVec(s.albedo)*(1-s.metallic)/M_PI_F:float3(0);
    output.write(float4(Lo,1),pixel);
}
kernel void reflection_check(constant GPUReflectionComposeParams& p [[buffer(0)]],const device GPUSpecularSample* metadata [[buffer(1)]],
    device atomic_uint* counts [[buffer(2)]],texture2d<float,access::read> output [[texture(0)]],texture2d<float,access::read> specular [[texture(1)]],
    texture2d<float,access::read> ao [[texture(2)]],texture2d<float,access::read> distance [[texture(3)]],uint tid [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){
    if(tid>=p.width*p.height)return;const uint2 q(tid%p.width,tid/p.width);bool bad=false;
    if(p.flags&REFLECT_COMPOSE_SPEC){const GPUSpecularSample s=metadata[tid];bad=(s.flags&SPECULAR_SAMPLE_ERROR)||!isfinite(s.hitDistance)||s.hitDistance<0;
        if(s.flags&SPECULAR_SAMPLE_VALID)bad|=!all(isfinite(float3(s.radiance[0],s.radiance[1],s.radiance[2])));}
    const float3 final=output.read(q).rgb,raw=specular.read(q).rgb;const float a=ao.read(q).x,d=distance.read(q).x;
    const bool badOutput=!all(isfinite(final))||any(final<0)||!all(isfinite(raw))||!isfinite(a)||a<0||a>1||!isfinite(d)||d<0;
    const uint total=simd_sum(1u),invalid=simd_sum(bad?1u:0u),invalidOutput=simd_sum(badOutput?1u:0u);
    if(lane==0){atomic_fetch_add_explicit(counts,total,memory_order_relaxed);if(invalid)atomic_fetch_add_explicit(counts+1,invalid,memory_order_relaxed);if(invalidOutput)atomic_fetch_add_explicit(counts+6,invalidOutput,memory_order_relaxed);}
}
