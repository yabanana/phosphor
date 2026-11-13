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

// ---------------------------------------------------------------------------
// F15.1 (ticket 05): KHR_materials_ior + KHR_materials_specular.
//
// glTF flattened form (.scratch/f15/research/khr-materials.md section 9; the
// ratified text of KHR_materials_specular with KHR_materials_ior, outside IOR 1):
//   dielectric_f0  = min(((ior - 1) / (ior + 1))^2 * specularColor, 1) * specular
//   dielectric_f90 = specular                      (f90 is NOT 1)
//   F_d            = f0 + (f90 - f0) * (1 - V.H)^5
//   dielectric     = F_d * specular_brdf + (1 - max(F_d)) * diffuse
//   material       = mix(dielectric, metal, metallic); the metal lobe is unchanged
// ior = 0 is the permanent special mode: the effective IOR is +infinity, the
// ((ior - 1) / (ior + 1))^2 factor is 1 (so with specular 1 and colour <= 1 the
// Fresnel term evaluates to 1 in every direction).
// The block is only compiled for scenes whose feature union has the bit
// (kMaterialFeatures) and only runs for materials whose flags carry it; at the
// neutral values the feature bit is clear and the base path above runs unchanged.
// ---------------------------------------------------------------------------
struct BsdfDielectricSpec {
    float3 f0;
    float f90;
};

inline float bsdfMax3(float3 v) { return max(v.x, max(v.y, v.z)); }

inline BsdfDielectricSpec bsdfSpecularParams(float ior, float specularFactor, float3 specularColor) {
    const float r = (ior - 1.0f) / (ior + 1.0f);
    const float iorTerm = ior == 0.0f ? 1.0f : r * r;
    BsdfDielectricSpec s;
    s.f0 = min(iorTerm * specularColor, float3(1.0f)) * specularFactor; // clamp BEFORE the weight
    s.f90 = specularFactor;
    return s;
}

inline float3 bsdfSpecFresnel(BsdfDielectricSpec s, float VdotH) {
    return s.f0 + (float3(s.f90) - s.f0) * pow(1.0f - VdotH, 5.0f);
}

// Diffuse weight of the dielectric: 1 - max(F_d) (colour-safe, per the specification).
inline float3 bsdfSpecDiffuse(float3 Fd, float metallic, float3 albedo) {
    return (1.0f - bsdfMax3(Fd)) * (1.0f - metallic) * albedo / M_PI_F;
}

// Specular lobe colour: mix(dielectric F_d, metal F(baseColor), metallic).
inline float3 bsdfSpecLobeColor(float3 Fd, float3 albedo, float metallic, float VdotH) {
    return mix(Fd, bsdfFSchlick(albedo, VdotH), metallic);
}

inline BsdfBaseLobes bsdfSpecEval(BsdfDielectricSpec s, float NdotL, float NdotV, float NdotH, float VdotH, float3 albedo,
                                  float metallic, float a) {
    const float3 Fd = bsdfSpecFresnel(s, VdotH);
    BsdfBaseLobes out;
    out.diffuse = bsdfSpecDiffuse(Fd, metallic, albedo);
    out.specular = bsdfDGGX(NdotH, a) * bsdfVSmithGGXCorrelated(NdotV, NdotL, a) * bsdfSpecLobeColor(Fd, albedo, metallic, VdotH);
    return out;
}

// Split-sum environment weights per lobe: E = f0 * A + f90 * B (DFG table), each clamped to the
// physical [0, 1] domain.  Dielectric lobe, metal lobe (f0 = baseColor, f90 = 1) and the diffuse
// weight 1 - max(E_dielectric).
struct BsdfSpecEnv {
    float3 specular;      // mix(E_dielectric, E_metal, metallic)
    float3 dielectric;    // E_dielectric
    float diffuseWeight;  // 1 - max(E_dielectric)
};
inline BsdfSpecEnv bsdfSpecEnv(BsdfDielectricSpec s, float3 albedo, float metallic, float roughness, float NdotV,
                               const constant float* table) {
    const float A = phosphor::dfgTableLookup(table, NdotV, roughness, 0u),
                B = phosphor::dfgTableLookup(table, NdotV, roughness, 1u);
    BsdfSpecEnv e;
    e.dielectric = clamp(s.f0 * A + s.f90 * B, 0.0f, 1.0f);
    const float3 metal = clamp(albedo * A + B, 0.0f, 1.0f);
    e.specular = mix(e.dielectric, metal, metallic);
    e.diffuseWeight = 1.0f - bsdfMax3(e.dielectric);
    return e;
}
