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
#include "renderer/dfg_table.h"
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

// glTF 2.0 Appendix B (ADR-0006): dielectric = mix(diffuse, specular, F(0.04)),
// metal = specular * F(baseColor), material = mix(dielectric, metal, metallic).
// Schlick is linear in f0, so the two specular lobes combine into
// D * V * F(mix(0.04, baseColor, metallic)); the diffuse lobe is weighted by the
// DIELECTRIC Fresnel only.
inline float3 bsdfDielectricFresnel(float VdotH) { return bsdfFSchlick(float3(0.04f), VdotH); }

inline float3 bsdfBaseDiffuse(float3 dielectricF, float metallic, float3 albedo) {
    return (1.0f - dielectricF) * (1.0f - metallic) * albedo / M_PI_F;
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
    out.diffuse = bsdfBaseDiffuse(bsdfDielectricFresnel(VdotH), metallic, albedo);
    out.specular = specularWeight * bsdfDGGX(NdotH, a) * bsdfVSmithGGXCorrelated(NdotV, NdotL, a) * F;
    return out;
}

// Specular lobe alone (reflection weights): D * V * F.
inline float3 bsdfBaseSpecular(float NdotL, float NdotV, float NdotH, float VdotH, float3 f0, float a) {
    return bsdfDGGX(NdotH, a) * bsdfVSmithGGXCorrelated(NdotV, NdotL, a) * bsdfFSchlick(f0, VdotH);
}

// Split-sum environment specular of the engine BRDF: directional albedo
// f0 * A + B from the tabulated DFG of tools/dfg_table.py (F13.5; the former
// Karis mobile fit was up to 14 % low at roughness 0.3-0.6 and low at grazing
// views, which let the ADR-0006 diffuse weight gain energy). Clamped to the
// physical [0, 1] domain of a directional albedo.
// Only a finite weight is bounded: invalid numerical input stays visible to the
// error counters downstream (F13.5 rule).
// `table` is a copy of renderer/dfg_table.h: FrameConstants::dfgTable in
// functions that have the frame (always in fragment functions: shader
// validation corrupts their reads of program-scope constant arrays, measured
// 2026-11-13 on the forward path), phosphor::DfgTable in compute kernels
// without the frame.
inline float3 bsdfEnvSpecular(float3 f0, float roughness, float NdotV, const constant float* table) {
    const float A = phosphor::dfgTableLookup(table, NdotV, roughness, 0u),
                B = phosphor::dfgTableLookup(table, NdotV, roughness, 1u);
    const float3 weight = f0 * A + B;
    return all(isfinite(weight)) ? clamp(weight, 0.0f, 1.0f) : weight;
}

// Environment/GI diffuse of the base model (ADR-0006): the diffuse lobe of the
// dielectric receives the energy its specular layer does not reflect, weighted
// by 1 - E, E = directional albedo of the dielectric specular (DFG table).
// Keeps a white dielectric at or below 1 in a uniform environment.
inline float bsdfEnvDiffuseWeight(float roughness, float NdotV, const constant float* table) {
    return 1.0f - bsdfEnvSpecular(float3(0.04f), roughness, NdotV, table).x;
}
