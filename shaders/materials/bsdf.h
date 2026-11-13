#pragma once
// F15 (ticket 01): the single primary-surface BSDF of Phosphor. The resolve
// (material_shading.h), DI (restir_common.h) and the specular transport
// (reflection_common.h) all evaluate the material through these functions, so
// a lobe is implemented once. Inputs are dot products already clamped by the
// caller: every caller keeps its own clamping convention. Secondary RT hits
// keep their documented base reduction (albedo * (1 - metallic)).
//
// Expression shapes are part of the contract: callers compare images bit for
// bit, so keep the association of products (specular = ((w * D) * V) * F) and
// do not fold factors differently without re-checking the exact captures.
#include <metal_stdlib>
using namespace metal;

// GGX (Trowbridge-Reitz) normal distribution; alpha = roughness^2.
inline float bsdfDGGX(float NdotH, float a) {
    const float a2 = a * a;
    const float d = NdotH * NdotH * (a2 - 1.0f) + 1.0f;
    return a2 / (M_PI_F * d * d);
}

// Height-correlated Smith visibility V = G / (4 NdotL NdotV).
inline float bsdfVSmithGGXCorrelated(float NdotV, float NdotL, float a) {
    const float a2 = a * a;
    const float gv = NdotL * sqrt(NdotV * NdotV * (1.0f - a2) + a2);
    const float gl = NdotV * sqrt(NdotL * NdotL * (1.0f - a2) + a2);
    return 0.5f / max(gv + gl, 1e-5f);
}

inline float3 bsdfFSchlick(float3 f0, float VdotH) {
    return f0 + (1.0f - f0) * pow(1.0f - VdotH, 5.0f);
}

// Roughness is clamped by the caller (material fetch); alpha has a floor so the
// distribution stays finite for mirror-like surfaces.
inline float bsdfAlpha(float roughness) { return max(roughness * roughness, 0.002f); }

inline float3 bsdfBaseF0(float3 albedo, float metallic) { return mix(float3(0.04f), albedo, metallic); }

inline float3 bsdfBaseDiffuse(float3 F, float metallic, float3 albedo) {
    return (1.0f - F) * (1.0f - metallic) * albedo / M_PI_F;
}

// Base metallic-roughness BRDF split in its two lobes (values of f, without the
// cosine and the incident radiance). specularWeight scales the specular lobe
// (F13 emitter ownership); pass 1 elsewhere.
struct BsdfBaseLobes {
    float3 diffuse;
    float3 specular;
};
inline BsdfBaseLobes bsdfBaseEval(float NdotL, float NdotV, float NdotH, float VdotH, float3 albedo, float metallic,
                                  float a, float specularWeight) {
    const float3 F = bsdfFSchlick(bsdfBaseF0(albedo, metallic), VdotH);
    BsdfBaseLobes out;
    out.diffuse = bsdfBaseDiffuse(F, metallic, albedo);
    out.specular = specularWeight * bsdfDGGX(NdotH, a) * bsdfVSmithGGXCorrelated(NdotV, NdotL, a) * F;
    return out;
}

// Specular lobe alone (reflection weights): D * V * F.
inline float3 bsdfBaseSpecular(float NdotL, float NdotV, float NdotH, float VdotH, float3 f0, float a) {
    return bsdfDGGX(NdotH, a) * bsdfVSmithGGXCorrelated(NdotV, NdotL, a) * bsdfFSchlick(f0, VdotH);
}

// Karis, "Physically Based Shading on Mobile" (2014): analytic fit of the
// split-sum environment BRDF, so ambient light keeps a specular term (and
// metals stay metallic) without an irradiance/prefiltered-environment map.
inline float3 bsdfEnvSpecularApprox(float3 f0, float roughness, float NdotV) {
    const float4 c0 = float4(-1.0, -0.0275, -0.572, 0.022);
    const float4 c1 = float4(1.0, 0.0425, 1.04, -0.04);
    const float4 r = roughness * c0 + c1;
    const float a004 = min(r.x * r.x, exp2(-9.28 * NdotV)) * r.x + r.y;
    const float2 ab = float2(-1.04, 1.04) * a004 + r.zw;
    return f0 * ab.x + ab.y;
}
