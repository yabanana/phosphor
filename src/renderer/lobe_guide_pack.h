#pragma once
// ---------------------------------------------------------------------------
// F15.1 (ticket 06): packing of the per-pixel lobe guide (GPULobeGuide in
// gpu_types.h).  Compiled by C++ (unit tests, readback checks) and by MSL
// (producers and consumers) from this single source, with integer-only
// conversions so both sides produce the same bits:
//
// - half fields: IEEE binary16, round to nearest even; magnitudes below 2^-14
//   (the binary16 normal range) and NaN pack to +0, values that round above
//   65504 pack to infinity (attenuation distance +inf stays +inf).
// - oct fields: octahedral unit vector, snorm16 per component (x in the low
//   half).  The encode divides, so MSL fast math may differ from C++ by one
//   step: compare oct fields by angle (decodeLobeOct), not by bits.
//
// packLobeGuide writes 0 into every word of a feature whose bit is clear;
// unpackLobeGuide returns the NEUTRAL value of such a feature (the defaults of
// renderer/material_description.h, checked by tests/test_lobe_guide_pack.cpp),
// so consumers always receive a complete parameter set.  `ior` is shared by
// KHR_materials_ior, transmission and volume: it is stored when any of the
// three bits is set.
// ---------------------------------------------------------------------------
#include "renderer/gpu_types.h"
#ifdef __METAL_VERSION__
#define PHOSPHOR_LOBE_FN static inline __attribute__((unused))
#define PHOSPHOR_LOBE_AS_UINT(x) as_type<uint>(x)
#define PHOSPHOR_LOBE_AS_FLOAT(x) as_type<float>(x)
#define PHOSPHOR_LOBE_FLOOR(x) metal::floor(x)
#define PHOSPHOR_LOBE_SQRT(x) metal::sqrt(x)
#else
#include <bit>
#include <cmath>
#include <cstdint>
#define PHOSPHOR_LOBE_FN inline
#define PHOSPHOR_LOBE_AS_UINT(x) std::bit_cast<uint32_t>(x)
#define PHOSPHOR_LOBE_AS_FLOAT(x) std::bit_cast<float>(x)
#define PHOSPHOR_LOBE_FLOOR(x) std::floor(x)
#define PHOSPHOR_LOBE_SQRT(x) std::sqrt(x)
#endif

namespace phosphor {
#ifdef __METAL_VERSION__
using lobe_u32 = uint;
using lobe_i32 = int;
#else
using lobe_u32 = uint32_t;
using lobe_i32 = int32_t;
#endif

// Neutral values of a feature whose bit is clear (material_description.h defaults).
PHOSPHOR_GPU_CONSTANT float LobeNeutralIor = 1.5f;
PHOSPHOR_GPU_CONSTANT float LobeNeutralIridescenceIor = 1.3f;
PHOSPHOR_GPU_CONSTANT float LobeNeutralIridescenceThickness = 400.0f; // thicknessMax: the thickness without a texture
PHOSPHOR_GPU_CONSTANT float LobeNeutralSubsurfaceColor = 0.8f;
PHOSPHOR_GPU_CONSTANT float LobeNeutralSubsurfaceRadius = 1.0f;
PHOSPHOR_GPU_CONSTANT float LobeNeutralRadiusScaleR = 1.0f;
PHOSPHOR_GPU_CONSTANT float LobeNeutralRadiusScaleG = 0.5f;
PHOSPHOR_GPU_CONSTANT float LobeNeutralRadiusScaleB = 0.25f;

/// Unpacked lobe parameters (thread-local in MSL).  Vectors are world space.
struct LobeParams {
    lobe_u32 features;  // MATERIAL_FEATURE_* | LOBE_GUIDE_FLAG_*
    float ior;
    float specularFactor;
    float specularColor[3];
    float clearcoatFactor;
    float clearcoatRoughness;
    float clearcoatNormal[3];
    float sheenColor[3];
    float sheenRoughness;
    float anisotropyStrength;
    float anisotropyDirection[3];
    float iridescenceFactor;
    float iridescenceIor;
    float iridescenceThickness; // nm
    float transmissionFactor;
    float volumeThickness;      // world units
    float attenuationColor[3];
    float attenuationDistance;  // world units, +inf = none
    float subsurfaceWeight;
    float subsurfaceRadius;     // metres
    float subsurfaceColor[3];
    float subsurfaceAnisotropy;
    float subsurfaceRadiusScale[3];
};

// --- binary16 ---------------------------------------------------------------

PHOSPHOR_LOBE_FN lobe_u32 lobeHalfBits(float value) {
    const lobe_u32 x = PHOSPHOR_LOBE_AS_UINT(value);
    const lobe_u32 sign = (x >> 16) & 0x8000u;
    const lobe_u32 a = x & 0x7FFFFFFFu;
    if (a > 0x7F800000u) return 0u;                 // NaN -> +0
    if (a >= 0x47800000u) return sign | 0x7C00u;    // >= 65536 (and +-inf) -> inf
    if (a < 0x38800000u) return 0u;                 // below 2^-14 -> +0 (no subnormals)
    lobe_u32 h = ((((a >> 23) - 127u + 15u) & 31u) << 10) | ((a >> 13) & 0x3FFu);
    const lobe_u32 rest = a & 0x1FFFu;
    if (rest > 0x1000u || (rest == 0x1000u && (h & 1u) != 0u)) h += 1u; // carries into the exponent (65520 -> inf)
    return sign | h;
}

PHOSPHOR_LOBE_FN float lobeHalfValue(lobe_u32 bits) {
    const lobe_u32 sign = (bits & 0x8000u) << 16;
    const lobe_u32 e = (bits >> 10) & 31u, m = bits & 0x3FFu;
    if (e == 0u) {
        const float magnitude = float(m) * (1.0f / 16777216.0f); // subnormal: m * 2^-24 (never produced)
        return sign != 0u ? -magnitude : magnitude;
    }
    if (e == 31u) return PHOSPHOR_LOBE_AS_FLOAT(sign | 0x7F800000u | (m << 13));
    return PHOSPHOR_LOBE_AS_FLOAT(sign | ((e - 15u + 127u) << 23) | (m << 13));
}

PHOSPHOR_LOBE_FN lobe_u32 packLobeHalf2(float low, float high) { return lobeHalfBits(low) | (lobeHalfBits(high) << 16); }
PHOSPHOR_LOBE_FN float lobeHalfLow(lobe_u32 word) { return lobeHalfValue(word & 0xFFFFu); }
PHOSPHOR_LOBE_FN float lobeHalfHigh(lobe_u32 word) { return lobeHalfValue(word >> 16); }

// --- octahedral unit vectors ------------------------------------------------

PHOSPHOR_LOBE_FN float lobeSignNotZero(float v) { return v >= 0.0f ? 1.0f : -1.0f; }
PHOSPHOR_LOBE_FN float lobeAbs(float v) { return v >= 0.0f ? v : -v; }

PHOSPHOR_LOBE_FN lobe_u32 lobeSnorm16(float v) {
    const float c = v < -1.0f ? -1.0f : (v > 1.0f ? 1.0f : v);
    const float q = c >= 0.0f ? PHOSPHOR_LOBE_FLOOR(c * 32767.0f + 0.5f) : -PHOSPHOR_LOBE_FLOOR(-c * 32767.0f + 0.5f);
    return lobe_u32(lobe_i32(q)) & 0xFFFFu;
}
PHOSPHOR_LOBE_FN float lobeSnorm16Value(lobe_u32 bits) {
    const lobe_i32 s = lobe_i32(bits & 0xFFFFu) - ((bits & 0x8000u) != 0u ? 65536 : 0);
    const float v = float(s) / 32767.0f;
    return v < -1.0f ? -1.0f : v;
}

/// Encode a (not necessarily normalised) vector; the zero vector encodes as +Z
/// (the word 0).  Clear features never decode their oct words (unpackLobeGuide).
PHOSPHOR_LOBE_FN lobe_u32 encodeLobeOct(float x, float y, float z) {
    const float l1 = lobeAbs(x) + lobeAbs(y) + lobeAbs(z);
    if (!(l1 > 0.0f)) return 0u;
    float u = x / l1, v = y / l1;
    if (z < 0.0f) {
        const float ou = u;
        u = (1.0f - lobeAbs(v)) * lobeSignNotZero(ou);
        v = (1.0f - lobeAbs(ou)) * lobeSignNotZero(v);
    }
    return lobeSnorm16(u) | (lobeSnorm16(v) << 16);
}

struct LobeVec3 {
    float x, y, z;
};

/// Decode to a unit vector (the word 0 is +Z).
PHOSPHOR_LOBE_FN LobeVec3 decodeLobeOct(lobe_u32 word) {
    LobeVec3 r;
    float u = lobeSnorm16Value(word & 0xFFFFu), v = lobeSnorm16Value(word >> 16);
    const float z = 1.0f - lobeAbs(u) - lobeAbs(v);
    if (z < 0.0f) {
        const float ou = u;
        u = (1.0f - lobeAbs(v)) * lobeSignNotZero(ou);
        v = (1.0f - lobeAbs(ou)) * lobeSignNotZero(v);
    }
    const float l = PHOSPHOR_LOBE_SQRT(u * u + v * v + z * z);
    r.x = u / l;
    r.y = v / l;
    r.z = z / l;
    return r;
}

// --- the record -------------------------------------------------------------

PHOSPHOR_LOBE_FN bool lobeHas(lobe_u32 features, lobe_u32 bit) { return (features & bit) != 0u; }

PHOSPHOR_LOBE_FN GPULobeGuide packLobeGuide(LobeParams p) {
    GPULobeGuide g;
    const lobe_u32 f = p.features;
    const bool spec = lobeHas(f, MATERIAL_FEATURE_SPECULAR_IOR), coat = lobeHas(f, MATERIAL_FEATURE_CLEARCOAT),
               sheen = lobeHas(f, MATERIAL_FEATURE_SHEEN), aniso = lobeHas(f, MATERIAL_FEATURE_ANISOTROPY),
               irid = lobeHas(f, MATERIAL_FEATURE_IRIDESCENCE), trans = lobeHas(f, MATERIAL_FEATURE_TRANSMISSION),
               volume = lobeHas(f, MATERIAL_FEATURE_VOLUME), sss = lobeHas(f, MATERIAL_FEATURE_SUBSURFACE);
    const float z = 0.0f;
    g.features = f;
    g.ior = (spec || trans || volume) ? p.ior : 0.0f;
    g.specularFactorCoat = packLobeHalf2(spec ? p.specularFactor : z, coat ? p.clearcoatFactor : z);
    g.specularColorRG = spec ? packLobeHalf2(p.specularColor[0], p.specularColor[1]) : 0u;
    g.specularColorBCoatRoughness = packLobeHalf2(spec ? p.specularColor[2] : z, coat ? p.clearcoatRoughness : z);
    g.clearcoatNormal = coat ? encodeLobeOct(p.clearcoatNormal[0], p.clearcoatNormal[1], p.clearcoatNormal[2]) : 0u;
    g.sheenColorRG = sheen ? packLobeHalf2(p.sheenColor[0], p.sheenColor[1]) : 0u;
    g.sheenColorBRoughness = sheen ? packLobeHalf2(p.sheenColor[2], p.sheenRoughness) : 0u;
    g.anisotropyDirection =
        aniso ? encodeLobeOct(p.anisotropyDirection[0], p.anisotropyDirection[1], p.anisotropyDirection[2]) : 0u;
    g.anisotropyIridescence = packLobeHalf2(aniso ? p.anisotropyStrength : z, irid ? p.iridescenceFactor : z);
    g.iridescence = irid ? packLobeHalf2(p.iridescenceIor, p.iridescenceThickness) : 0u;
    g.transmissionThickness = packLobeHalf2(trans ? p.transmissionFactor : z, volume ? p.volumeThickness : z);
    g.attenuationRG = volume ? packLobeHalf2(p.attenuationColor[0], p.attenuationColor[1]) : 0u;
    g.attenuationBDistance = volume ? packLobeHalf2(p.attenuationColor[2], p.attenuationDistance) : 0u;
    g.subsurfaceWeightRadius = sss ? packLobeHalf2(p.subsurfaceWeight, p.subsurfaceRadius) : 0u;
    g.subsurfaceColorRG = sss ? packLobeHalf2(p.subsurfaceColor[0], p.subsurfaceColor[1]) : 0u;
    g.subsurfaceColorBAnisotropy = sss ? packLobeHalf2(p.subsurfaceColor[2], p.subsurfaceAnisotropy) : 0u;
    g.subsurfaceScaleRG = sss ? packLobeHalf2(p.subsurfaceRadiusScale[0], p.subsurfaceRadiusScale[1]) : 0u;
    g.subsurfaceScaleB = sss ? packLobeHalf2(p.subsurfaceRadiusScale[2], z) : 0u;
    g.reserved = 0u;
    return g;
}

PHOSPHOR_LOBE_FN LobeParams unpackLobeGuide(GPULobeGuide g) {
    LobeParams p;
    const lobe_u32 f = g.features;
    const bool spec = lobeHas(f, MATERIAL_FEATURE_SPECULAR_IOR), coat = lobeHas(f, MATERIAL_FEATURE_CLEARCOAT),
               sheen = lobeHas(f, MATERIAL_FEATURE_SHEEN), aniso = lobeHas(f, MATERIAL_FEATURE_ANISOTROPY),
               irid = lobeHas(f, MATERIAL_FEATURE_IRIDESCENCE), trans = lobeHas(f, MATERIAL_FEATURE_TRANSMISSION),
               volume = lobeHas(f, MATERIAL_FEATURE_VOLUME), sss = lobeHas(f, MATERIAL_FEATURE_SUBSURFACE);
    const float inf = PHOSPHOR_LOBE_AS_FLOAT(0x7F800000u);
    p.features = f;
    p.ior = (spec || trans || volume) ? g.ior : LobeNeutralIor;
    p.specularFactor = spec ? lobeHalfLow(g.specularFactorCoat) : 1.0f;
    p.specularColor[0] = spec ? lobeHalfLow(g.specularColorRG) : 1.0f;
    p.specularColor[1] = spec ? lobeHalfHigh(g.specularColorRG) : 1.0f;
    p.specularColor[2] = spec ? lobeHalfLow(g.specularColorBCoatRoughness) : 1.0f;
    p.clearcoatFactor = coat ? lobeHalfHigh(g.specularFactorCoat) : 0.0f;
    p.clearcoatRoughness = coat ? lobeHalfHigh(g.specularColorBCoatRoughness) : 0.0f;
    const LobeVec3 cn = decodeLobeOct(g.clearcoatNormal); // zero vector when the feature is clear
    p.clearcoatNormal[0] = coat ? cn.x : 0.0f;
    p.clearcoatNormal[1] = coat ? cn.y : 0.0f;
    p.clearcoatNormal[2] = coat ? cn.z : 0.0f;
    p.sheenColor[0] = sheen ? lobeHalfLow(g.sheenColorRG) : 0.0f;
    p.sheenColor[1] = sheen ? lobeHalfHigh(g.sheenColorRG) : 0.0f;
    p.sheenColor[2] = sheen ? lobeHalfLow(g.sheenColorBRoughness) : 0.0f;
    p.sheenRoughness = sheen ? lobeHalfHigh(g.sheenColorBRoughness) : 0.0f;
    p.anisotropyStrength = aniso ? lobeHalfLow(g.anisotropyIridescence) : 0.0f;
    const LobeVec3 ad = decodeLobeOct(g.anisotropyDirection);
    p.anisotropyDirection[0] = aniso ? ad.x : 0.0f;
    p.anisotropyDirection[1] = aniso ? ad.y : 0.0f;
    p.anisotropyDirection[2] = aniso ? ad.z : 0.0f;
    p.iridescenceFactor = irid ? lobeHalfHigh(g.anisotropyIridescence) : 0.0f;
    p.iridescenceIor = irid ? lobeHalfLow(g.iridescence) : LobeNeutralIridescenceIor;
    p.iridescenceThickness = irid ? lobeHalfHigh(g.iridescence) : LobeNeutralIridescenceThickness;
    p.transmissionFactor = trans ? lobeHalfLow(g.transmissionThickness) : 0.0f;
    p.volumeThickness = volume ? lobeHalfHigh(g.transmissionThickness) : 0.0f;
    p.attenuationColor[0] = volume ? lobeHalfLow(g.attenuationRG) : 1.0f;
    p.attenuationColor[1] = volume ? lobeHalfHigh(g.attenuationRG) : 1.0f;
    p.attenuationColor[2] = volume ? lobeHalfLow(g.attenuationBDistance) : 1.0f;
    p.attenuationDistance = volume ? lobeHalfHigh(g.attenuationBDistance) : inf;
    p.subsurfaceWeight = sss ? lobeHalfLow(g.subsurfaceWeightRadius) : 0.0f;
    p.subsurfaceRadius = sss ? lobeHalfHigh(g.subsurfaceWeightRadius) : LobeNeutralSubsurfaceRadius;
    p.subsurfaceColor[0] = sss ? lobeHalfLow(g.subsurfaceColorRG) : LobeNeutralSubsurfaceColor;
    p.subsurfaceColor[1] = sss ? lobeHalfHigh(g.subsurfaceColorRG) : LobeNeutralSubsurfaceColor;
    p.subsurfaceColor[2] = sss ? lobeHalfLow(g.subsurfaceColorBAnisotropy) : LobeNeutralSubsurfaceColor;
    p.subsurfaceAnisotropy = sss ? lobeHalfHigh(g.subsurfaceColorBAnisotropy) : 0.0f;
    p.subsurfaceRadiusScale[0] = sss ? lobeHalfLow(g.subsurfaceScaleRG) : LobeNeutralRadiusScaleR;
    p.subsurfaceRadiusScale[1] = sss ? lobeHalfHigh(g.subsurfaceScaleRG) : LobeNeutralRadiusScaleG;
    p.subsurfaceRadiusScale[2] = sss ? lobeHalfLow(g.subsurfaceScaleB) : LobeNeutralRadiusScaleB;
    return p;
}
} // namespace phosphor
