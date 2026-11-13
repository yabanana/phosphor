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
#include "renderer/sheen_table.h"
#include "renderer/lobe_select.h"
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

// ---------------------------------------------------------------------------
// F15.1 (ticket 09): KHR_materials_clearcoat (ratified text, research/khr-materials.md section 2).
//
//   F_c   = 0.04 + 0.96 (1 - |V.Nc|)^5     (coat IOR fixed at 1.5; the specification uses V.Nc, NOT V.H)
//   coat  = D(Nc.H, a_c) * Vis(Nc.V, Nc.L, a_c)         (no Fresnel inside the lobe)
//   f     = (1 - c F_c) f_material + c F_c coat         (the coat term has its own cosine max(Nc.L, 0))
// The material under the coat is the ior/specular dielectric or the metal of the sections above.  Emission is
// attenuated by the same (1 - c F_c).  Roughness clamps are the base ones (caller: [0.04, 1]; bsdfAlpha floors
// alpha).  The coat normal Nc is the shading normal of the coat: its own normal texture, or WITHOUT one the
// geometric shading normal without the base normal map.
// ---------------------------------------------------------------------------
inline float bsdfCoatFresnel(float NcdotV) { return phosphor::lobeCoatFresnel(NcdotV); }

// Coat specular without Fresnel and without the cosine: D * Vis.
inline float bsdfCoatSpecular(float NcdotL, float NcdotV, float NcdotH, float a) {
    return bsdfDGGX(NcdotH, a) * bsdfVSmithGGXCorrelated(NcdotV, NcdotL, a);
}

// Split-sum albedo of the coat lobe D * Vis alone (Fresnel 1): A + B of the DFG table at the coat roughness, bounded
// to the physical domain like every directional albedo.
inline float bsdfCoatEnvAlbedo(float coatRoughness, float NcdotV, const constant float* table) {
    const float A = phosphor::dfgTableLookup(table, NcdotV, coatRoughness, 0u),
                B = phosphor::dfgTableLookup(table, NcdotV, coatRoughness, 1u);
    const float e = A + B;
    return isfinite(e) ? clamp(e, 0.0f, 1.0f) : e;
}

// ---------------------------------------------------------------------------
// F15.1 (ticket 10): KHR_materials_sheen (ratified text and non-normative Charlie notes, research/khr-materials.md section 3).
//
//   alpha_g   = r^2, r = sheenRoughness clamped to [0.07, 1] (the declared lower bound of the Charlie lobe)
//   D_charlie = (2 + 1/alpha_g) sin(theta_h)^(1/alpha_g) / (2 pi)
//   V_charlie = 1 / ((1 + Lambda(N.V) + Lambda(N.L)) 4 N.V N.L)          (Estevez-Kulla fit of the notes)
//   f         = S D V + f_material * scaling                              (no Fresnel in the lobe; coat sits above the sum)
//   scaling   = min(1 - max3(S) E(N.V), 1 - max3(S) E(N.L))   analytic lights and the sun (the notes' exact form)
//             = 1 - max3(S) E(N.V)                            environment / GI (the notes' indirect form)
// E(N.V, r) is the Fresnel-free directional albedo of D V N.L, clamped to 1: the table of tools/sheen_table.py
// (renderer/sheen_table.h).  `table` is FrameConstants::sheenTable in fragment functions (shader validation corrupts their reads
// of program-scope constant arrays), phosphor::SheenTable in compute kernels without the frame.
// ---------------------------------------------------------------------------
inline float bsdfSheenRoughness(float roughness) { return clamp(roughness, 0.07f, 1.0f); }

inline float bsdfSheenL(float x, float alphaG) {
    const float t = (1.0f - alphaG) * (1.0f - alphaG);
    const float a = mix(21.5473f, 25.3245f, t), b = mix(3.82987f, 3.32435f, t), c = mix(0.19823f, 0.16801f, t),
                d = mix(-1.97760f, -1.27393f, t), e = mix(-4.32054f, -4.85967f, t);
    return a / (1.0f + b * pow(x, c)) + d * x + e;
}
inline float bsdfSheenLambda(float cosTheta, float alphaG) {
    const float x = abs(cosTheta);
    return x < 0.5f ? exp(bsdfSheenL(x, alphaG)) : exp(2.0f * bsdfSheenL(0.5f, alphaG) - bsdfSheenL(1.0f - x, alphaG));
}
inline float bsdfSheenD(float NdotH, float alphaG) {
    const float inv = 1.0f / alphaG, sin2h = max(1.0f - NdotH * NdotH, 0.0f);
    return (2.0f + inv) * pow(sin2h, 0.5f * inv) / (2.0f * M_PI_F);
}
inline float bsdfSheenV(float NdotV, float NdotL, float alphaG) {
    return 1.0f / ((1.0f + bsdfSheenLambda(NdotV, alphaG) + bsdfSheenLambda(NdotL, alphaG)) * (4.0f * NdotV * NdotL));
}
// The sheen lobe S D V (without the cosine and the incident radiance); NdotL > 0 (the caller masks).
inline float3 bsdfSheenLobe(float3 color, float roughness, float NdotL, float NdotV, float NdotH) {
    const float alphaG = bsdfSheenRoughness(roughness) * bsdfSheenRoughness(roughness);
    return color * (bsdfSheenD(NdotH, alphaG) * bsdfSheenV(max(NdotV, 1e-4f), max(NdotL, 1e-4f), alphaG));
}
inline float bsdfSheenE(float roughness, float cosine, const constant float* table) {
    return phosphor::sheenTableLookup(table, max(cosine, 1e-4f), bsdfSheenRoughness(roughness));
}
// 1 - max3(S) E(N.V): the scaling of every indirect term (environment, GI, transport weight).
inline float bsdfSheenScaleView(float3 color, float roughness, float NdotV, const constant float* table) {
    return 1.0f - bsdfMax3(color) * bsdfSheenE(roughness, NdotV, table);
}
// min(1 - max3(S) E(N.V), 1 - max3(S) E(N.L)): the scaling of an analytic light or the sun.
inline float bsdfSheenScaleLight(float3 color, float roughness, float NdotV, float NdotL, const constant float* table) {
    const float m = bsdfMax3(color);
    return min(1.0f - m * bsdfSheenE(roughness, NdotV, table), 1.0f - m * bsdfSheenE(roughness, NdotL, table));
}

// ---------------------------------------------------------------------------
// F15.1 (ticket 11): KHR_materials_anisotropy (ratified text and the README's informative GLSL, research/khr-materials.md section 4).
//
//   alpha_b = alpha = max(roughness^2, 0.002)           (perpendicular to the lobe direction)
//   alpha_t = mix(alpha, 1, strength^2)                 (along the lobe direction T': the highlight is stretched along it)
//   D(h)    = 1 / (pi alpha_t alpha_b ((h.t)^2/alpha_t^2 + (h.b)^2/alpha_b^2 + (h.n)^2)^2)      (Burley, normative)
//   V       = 0.5 / (N.L |(alpha_t T.V, alpha_b B.V, N.V)| + N.V |(alpha_t T.L, alpha_b B.L, N.L)|)   (informative; NOT clamped to 1:
//             the isotropic limit then equals bsdfDGGX * bsdfVSmithGGXCorrelated, glTF Appendix B)
//   f_spec  = F D V,  F, the diffuse weighting and the metal/dielectric mix are those of bsdfSpecEval.
// The lobe axes are world unit vectors in the tangent plane of the shading normal: T' = the rotated tangent (or the
// direction texture), B' = cross(N, T').  The resolve builds them from the mesh tangent (handedness w and the mirrored-instance
// sign are folded into the vertex tangent's w), the guide carries T' (GPULobeGuide::anisotropyDirection) for DI and the transport.
//
// Environment (declared, decision 06 amendment of 2026-11-13): the radiance fetch direction is the README's bent normal; the
// split sum (DFG table, diffuse weight included) runs at the effective isotropic roughness sqrt(alpha_eff),
// alpha_eff = sqrt((alpha_t^2 + alpha_b^2) / 2).
// ---------------------------------------------------------------------------
struct BsdfAniso {
    float3 t, b;     // T', B' unit, orthogonal to the shading normal
    float at, ab;    // alpha_t, alpha_b
};
// `tdir` is any vector near the tangent plane of `n` (the guide's quantised direction): Gram-Schmidt against n; a degenerate
// direction falls back to an arbitrary tangent (the lobe then stays well defined).
inline BsdfAniso bsdfAnisoFrame(float3 n, float3 tdir, float roughness, float strength) {
    BsdfAniso a;
    const float3 t = tdir - n * dot(n, tdir);
    const float l2 = dot(t, t);
    a.t = l2 > 1e-12f ? t * rsqrt(l2) : normalize(cross(abs(n.z) < 0.999f ? float3(0, 0, 1) : float3(0, 1, 0), n));
    a.b = cross(n, a.t);
    a.ab = bsdfAlpha(roughness);
    a.at = a.ab + (1.0f - a.ab) * (strength * strength);
    return a;
}
// Burley anisotropic GGX in the README's stable form.
inline float bsdfDAniso(float NdotH, float TdotH, float BdotH, float at, float ab) {
    const float a2 = at * ab;
    const float3 f = float3(ab * TdotH, at * BdotH, a2 * NdotH);
    const float w2 = a2 / dot(f, f);
    return a2 * w2 * w2 / M_PI_F;
}
inline float bsdfVAniso(float NdotV, float NdotL, float TdotV, float BdotV, float TdotL, float BdotL, float at, float ab) {
    const float gv = NdotL * length(float3(at * TdotV, ab * BdotV, NdotV));
    const float gl = NdotV * length(float3(at * TdotL, ab * BdotL, NdotL));
    return 0.5f / max(gv + gl, 1e-5f);
}
// D * V of the anisotropic lobe for unit v, l, h = normalize(v + l); the caller supplies the clamped dot products.
inline float bsdfAnisoDV(BsdfAniso an, float3 v, float3 l, float3 h, float NdotL, float NdotV, float NdotH) {
    return bsdfDAniso(NdotH, dot(an.t, h), dot(an.b, h), an.at, an.ab) *
           bsdfVAniso(NdotV, NdotL, dot(an.t, v), dot(an.b, v), dot(an.t, l), dot(an.b, l), an.at, an.ab);
}
// Same structure as bsdfSpecEval with the anisotropic D V.
inline BsdfBaseLobes bsdfSpecEvalAniso(BsdfDielectricSpec s, BsdfAniso an, float3 v, float3 l, float3 h, float NdotL, float NdotV,
                                       float NdotH, float VdotH, float3 albedo, float metallic) {
    const float3 Fd = bsdfSpecFresnel(s, VdotH);
    BsdfBaseLobes out;
    out.diffuse = bsdfSpecDiffuse(Fd, metallic, albedo);
    out.specular = bsdfAnisoDV(an, v, l, h, NdotL, NdotV, NdotH) * bsdfSpecLobeColor(Fd, albedo, metallic, VdotH);
    return out;
}
// Effective isotropic roughness of the split-sum environment term; the base roughness itself without anisotropy.
inline float bsdfAnisoEnvRoughness(float roughness, float strength) {
    if (!(strength > 0.0f)) return roughness;
    const float ab = bsdfAlpha(roughness), at = ab + (1.0f - ab) * (strength * strength);
    return sqrt(sqrt(0.5f * (at * at + ab * ab)));
}
// Roughness of the DIFFUSE complement 1 - max3(E_dielectric) where the specular lobe is TRANSPORTED (the composite): the transport
// carries the exact anisotropic lobe, so the complement should track its dielectric albedo.  Against the exact quadrature on a sphere
// (alpha_b 0.09, F0 0.04) the mean alpha (alpha_t + alpha_b) / 2 is within 7 % / 5 % of the albedo at strength 0.5 / 1 where the rms
// alpha of the split-sum fit is 12 % / 24 % low (a diffuse complement 0.008 too large: the furnace of the transported lobe gains
// energy beyond its 0.5 % bound).  Only a composite that adds a transported specular signal uses it; the deterministic resolve, the composite without
// a specular signal and the probe fit keep the declared rms fit (bsdfAnisoEnvRoughness).
inline float bsdfAnisoDiffuseRoughness(float roughness, float strength) {
    if (!(strength > 0.0f)) return roughness;
    const float ab = bsdfAlpha(roughness), at = ab + (1.0f - ab) * (strength * strength);
    return sqrt(0.5f * (at + ab));
}
// The README's bent normal (fetch direction of the environment): towards the plane spanned by v and the stretch direction.
inline float3 bsdfAnisoBentNormal(float3 n, float3 v, BsdfAniso an, float roughness, float strength) {
    const float3 bent = cross(cross(an.b, v), an.b);
    if (!(dot(bent, bent) > 1e-12f)) return n;
    const float k = 1.0f - strength * (1.0f - roughness), a = (k * k) * (k * k);
    const float3 m = mix(normalize(bent), n, a);
    return dot(m, m) > 1e-12f ? normalize(m) : n;
}
// Visible-normal sampling (Heitz 2018) of the anisotropic GGX: a visible half vector in world space for the uniform pair u.
inline float3 bsdfAnisoSampleVndf(BsdfAniso an, float3 n, float3 v, float2 u) {
    const float3 ve = float3(dot(v, an.t), dot(v, an.b), dot(v, n));
    const float3 vh = normalize(float3(an.at * ve.x, an.ab * ve.y, ve.z));
    const float lensq = vh.x * vh.x + vh.y * vh.y;
    const float3 t1 = lensq > 0.0f ? float3(-vh.y, vh.x, 0.0f) * rsqrt(lensq) : float3(1.0f, 0.0f, 0.0f);
    const float3 t2 = cross(vh, t1);
    const float r = sqrt(u.x), phi = 2.0f * M_PI_F * u.y;
    const float p1 = r * cos(phi);
    float p2 = r * sin(phi);
    const float s = 0.5f * (1.0f + vh.z);
    p2 = (1.0f - s) * sqrt(max(0.0f, 1.0f - p1 * p1)) + s * p2;
    const float3 nh = p1 * t1 + p2 * t2 + sqrt(max(0.0f, 1.0f - p1 * p1 - p2 * p2)) * vh;
    const float3 he = normalize(float3(an.at * nh.x, an.ab * nh.y, max(0.0f, nh.z)));
    return an.t * he.x + an.b * he.y + n * he.z;
}
// Solid-angle density of the direction wi reflected about the VNDF half vector: D(h) G1(v) / (4 N.V) = D(h) / (2 (N.V + |(at T.V, ab B.V, N.V)|)).
// 0 outside the visible hemisphere of h.
inline float bsdfAnisoVndfPdf(BsdfAniso an, float3 n, float3 v, float3 wi) {
    const float3 h = normalize(v + wi);
    const float nh = dot(n, h), vh = dot(v, h), nv = dot(n, v);
    if (!(nh > 0.0f && vh > 0.0f && nv > 0.0f)) return 0.0f;
    const float len = length(float3(an.at * dot(an.t, v), an.ab * dot(an.b, v), nv));
    return bsdfDAniso(nh, dot(an.t, h), dot(an.b, h), an.at, an.ab) / (2.0f * (nv + len));
}
