#include <metal_stdlib>
#include "renderer/metalfx_denoise_layout.h"
#include "renderer/visibility_layout.h"
using namespace metal;
using namespace phosphor;

// Exact-active-region packing, WRITTEN / NON VERIFIED.
// params0/counters1. Input textures:
// color0, signed WORLD normal1, perceptual roughness2, diffuse albedo3,
// Fresnel specular albedo4, INPUT-PIXEL motion5, WORLD hit distance6,
// reactive7, skip-denoise strength16, reverse-Z depth19.
// Output textures: color8, normal9, roughness10, diffuse11, specular12,
// motion13, hit distance14, reactive15, strength17, unit exposure18.
// Depth is separately cropped with a graph Blit into Depth32Float; compute
// texture writes to that depth format are not assumed.
// No exposure, gamma, normal-unorm remap, roughness square or motion rescale.
// Repairs keep framework inputs finite but are NOT quality acceptance: any
// repair increments the independent readback counter and must fail validation.
inline void denoisePackError(device atomic_uint* c,uint field) {
    atomic_fetch_add_explicit(c+field,1u,memory_order_relaxed);
}
inline float3 denoisePackRgb(float3 v,float maximum) {
    return clamp(select(float3(0),v,isfinite(v)),0.0f,maximum);
}
kernel void denoise_pack_clear(device atomic_uint* counters [[buffer(0)]],uint tid [[thread_position_in_grid]]) {
    if(tid<8u)atomic_store_explicit(counters+tid,0u,memory_order_relaxed);
}
kernel void denoise_pack(constant GPUMetalfxDenoisePackParams& p [[buffer(0)]],
                         device atomic_uint* counters [[buffer(1)]],
                         texture2d<float,access::read> color [[texture(0)]],
                         texture2d<float,access::read> normal [[texture(1)]],
                         texture2d<float,access::read> roughness [[texture(2)]],
                         texture2d<float,access::read> diffuse [[texture(3)]],
                         texture2d<float,access::read> specular [[texture(4)]],
                         texture2d<float,access::read> motion [[texture(5)]],
                         texture2d<float,access::read> hit [[texture(6)]],
                         texture2d<float,access::read> reactive [[texture(7)]],
                         texture2d<float,access::write> packedColor [[texture(8)]],
                         texture2d<float,access::write> packedNormal [[texture(9)]],
                         texture2d<float,access::write> packedRoughness [[texture(10)]],
                         texture2d<float,access::write> packedDiffuse [[texture(11)]],
                         texture2d<float,access::write> packedSpecular [[texture(12)]],
                         texture2d<float,access::write> packedMotion [[texture(13)]],
                         texture2d<float,access::write> packedHit [[texture(14)]],
                         texture2d<float,access::write> packedReactive [[texture(15)]],
                         texture2d<float,access::read> strength [[texture(16)]],
                         texture2d<float,access::write> packedStrength [[texture(17)]],
                         texture2d<float,access::write> exposure [[texture(18)]],
                         texture2d<float,access::read> depth [[texture(19)]],
                         uint2 pixel [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    if(any(pixel>=uint2(p.width,p.height)))return;
    uint pixels=simd_sum(1u);
    if(lane==0u)atomic_fetch_add_explicit(counters,pixels,memory_order_relaxed);
    float3 C=color.read(pixel).rgb,N=normal.read(pixel).xyz,D=diffuse.read(pixel).rgb,S=specular.read(pixel).rgb;
    float r=roughness.read(pixel).x,z=depth.read(pixel).x;
    float2 mv=motion.read(pixel).xy;
    bool colorBad=!all(isfinite(C))||any(C<0.0f)||any(C>65504.0f);
    bool normalBad=!isfinite(z)||z<0.0f||z>1.0f||
        (z>0.0f&&(!all(isfinite(N))||abs(dot(N,N)-1.0f)>p.normalTolerance));
    bool albedoBad=!all(isfinite(D))||!all(isfinite(S))||any(D<0.0f)||any(D>1.0f)||any(S<0.0f)||any(S>1.0f);
    bool roughnessBad=!isfinite(r)||r<0.0f||r>1.0f,motionBad=!all(isfinite(mv));
    if(colorBad)denoisePackError(counters,1u);
    if(normalBad)denoisePackError(counters,2u);
    if(albedoBad)denoisePackError(counters,3u);
    if(roughnessBad)denoisePackError(counters,4u);
    if(motionBad)denoisePackError(counters,5u);
    float distance=(p.flags&METALFX_PACK_HIT_DISTANCE)?hit.read(pixel).x:0.0f;
    float react=(p.flags&METALFX_PACK_REACTIVE)?reactive.read(pixel).x:0.0f;
    float skip=(p.flags&METALFX_PACK_STRENGTH)?strength.read(pixel).x:0.0f;
    bool hitBad=!isfinite(distance)||distance<0.0f;
    bool maskBad=!isfinite(react)||react<0.0f||react>1.0f||!isfinite(skip)||skip<0.0f||skip>1.0f;
    if(hitBad)denoisePackError(counters,6u);
    if(maskBad)denoisePackError(counters,7u);
    N=(z>0.0f&&all(isfinite(N))&&dot(N,N)>1e-20f)?normalize(N):float3(0,0,1);
    bool repaired=colorBad||normalBad||albedoBad||roughnessBad||motionBad||hitBad||maskBad;
    packedColor.write(float4(denoisePackRgb(C,65504.0f),1.0f),pixel);
    packedNormal.write(float4(N,0.0f),pixel);
    packedRoughness.write(float4(isfinite(r)?clamp(r,0.0f,1.0f):1.0f),pixel);
    packedDiffuse.write(float4(denoisePackRgb(D,1.0f),1.0f),pixel);
    packedSpecular.write(float4(denoisePackRgb(S,1.0f),1.0f),pixel);
    packedMotion.write(float4(select(float2(0),mv,isfinite(mv)),0,0),pixel);
    packedHit.write(float4(isfinite(distance)?max(distance,0.0f):0.0f),pixel);
    packedReactive.write(float4(repaired?1.0f:(isfinite(react)?clamp(react,0.0f,1.0f):1.0f)),pixel);
    packedStrength.write(float4(repaired?1.0f:(isfinite(skip)?clamp(skip,0.0f,1.0f):1.0f)),pixel);
    if(all(pixel==uint2(0)))exposure.write(float4(p.exposureNormalization),uint2(0));
}

// Roughness is kept separate from signed world normal, as required by SDK.
kernel void denoise_split_roughness(constant GPUVisibilityParams& p [[buffer(0)]],
    texture2d<float,access::read> normalRoughness [[texture(0)]],texture2d<float,access::write> output [[texture(1)]],
    uint2 pixel [[thread_position_in_grid]]) {
    if(any(pixel>=uint2(p.width,p.height)))return;output.write(float4(normalRoughness.read(pixel).w),pixel);
}
