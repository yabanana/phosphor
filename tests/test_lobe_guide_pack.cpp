// F15.1 (ticket 06): lobe guide layout and packing (renderer/lobe_guide_pack.h).
#include <doctest/doctest.h>
#include "renderer/lobe_guide_pack.h"
#include "renderer/material_description.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <limits>
#include <random>

using namespace phosphor;

namespace {
LobeParams allFeatures() {
    LobeParams p{};
    p.features = MATERIAL_FEATURES_ALL | LOBE_GUIDE_FLAG_SUBSURFACE_VOLUMETRIC;
    p.ior = 1.33f;
    p.specularFactor = 0.5f;
    p.specularColor[0] = 2.0f; p.specularColor[1] = 1.0f; p.specularColor[2] = 0.5f;
    p.clearcoatFactor = 0.75f;
    p.clearcoatRoughness = 0.125f;
    p.clearcoatNormal[0] = 0.0f; p.clearcoatNormal[1] = 0.6f; p.clearcoatNormal[2] = -0.8f;
    p.sheenColor[0] = 0.25f; p.sheenColor[1] = 0.5f; p.sheenColor[2] = 1.0f;
    p.sheenRoughness = 0.3f;
    p.anisotropyStrength = 0.9f;
    p.anisotropyDirection[0] = 1.0f; p.anisotropyDirection[1] = 0.0f; p.anisotropyDirection[2] = 0.0f;
    p.iridescenceFactor = 1.0f;
    p.iridescenceIor = 1.8f;
    p.iridescenceThickness = 1200.0f;
    p.transmissionFactor = 1.0f;
    p.volumeThickness = 0.02f;
    p.attenuationColor[0] = 0.9f; p.attenuationColor[1] = 0.5f; p.attenuationColor[2] = 0.1f;
    p.attenuationDistance = std::numeric_limits<float>::infinity();
    p.subsurfaceWeight = 1.0f;
    p.subsurfaceRadius = 0.005f;
    p.subsurfaceColor[0] = 0.8f; p.subsurfaceColor[1] = 0.4f; p.subsurfaceColor[2] = 0.2f;
    p.subsurfaceAnisotropy = -0.25f;
    p.subsurfaceRadiusScale[0] = 1.0f; p.subsurfaceRadiusScale[1] = 0.5f; p.subsurfaceRadiusScale[2] = 0.25f;
    return p;
}
// Relative half precision: 2^-11 of the value (round to nearest).
bool nearHalf(float got, float want) { return std::fabs(got - want) <= std::ldexp(std::fabs(want), -11) + 1e-30f; }
// Angle in double precision through atan2(|a x b|, a . b) (acos loses ~4e-4 rad near 1 in float).
float angle(LobeVec3 a, double x, double y, double z) {
    const double cx = a.y * z - a.z * y, cy = a.z * x - a.x * z, cz = a.x * y - a.y * x;
    return float(std::atan2(std::sqrt(cx * cx + cy * cy + cz * cz), a.x * x + a.y * y + a.z * z));
}
} // namespace

TEST_CASE("lobe guide: frozen layout") {
    CHECK(sizeof(GPULobeGuide) == 80);
    CHECK(offsetof(GPULobeGuide, features) == 0);
    CHECK(offsetof(GPULobeGuide, ior) == 4);
    CHECK(offsetof(GPULobeGuide, clearcoatNormal) == 20);
    CHECK(offsetof(GPULobeGuide, anisotropyDirection) == 32);
    CHECK(offsetof(GPULobeGuide, transmissionThickness) == 44);
    CHECK(offsetof(GPULobeGuide, subsurfaceWeightRadius) == 56);
    CHECK(offsetof(GPULobeGuide, reserved) == 76);
    CHECK((LOBE_GUIDE_FEATURE_MASK & MATERIAL_FEATURES_ALL) == MATERIAL_FEATURES_ALL);
    CHECK((LOBE_GUIDE_FLAG_SUBSURFACE_VOLUMETRIC & LOBE_GUIDE_FEATURE_MASK) == 0u);
}

TEST_CASE("lobe guide: binary16 conversion rounds to nearest even, flushes subnormals, keeps infinity") {
    CHECK(lobeHalfBits(1.0f) == 0x3C00u);
    CHECK(lobeHalfBits(1.5f) == 0x3E00u);
    CHECK(lobeHalfBits(-2.0f) == 0xC000u);
    CHECK(lobeHalfBits(0.1f) == 0x2E66u);
    CHECK(lobeHalfBits(65504.0f) == 0x7BFFu);
    CHECK(lobeHalfBits(65519.0f) == 0x7BFFu);
    CHECK(lobeHalfBits(65520.0f) == 0x7C00u); // tie to even rounds up into infinity
    CHECK(lobeHalfBits(2049.0f) == 0x6800u);  // tie -> even mantissa (2048)
    CHECK(lobeHalfBits(2051.0f) == 0x6802u);  // tie -> even mantissa (2052)
    CHECK(lobeHalfBits(std::ldexp(1.0f, -14)) == 0x0400u);
    CHECK(lobeHalfBits(std::ldexp(1.0f, -15)) == 0u); // subnormal range -> 0
    CHECK(lobeHalfBits(-1e-6f) == 0u);
    CHECK(lobeHalfBits(std::numeric_limits<float>::infinity()) == 0x7C00u);
    CHECK(lobeHalfBits(-std::numeric_limits<float>::infinity()) == 0xFC00u);
    CHECK(lobeHalfBits(std::numeric_limits<float>::quiet_NaN()) == 0u);
    CHECK(lobeHalfValue(0x7C00u) == std::numeric_limits<float>::infinity());
    CHECK(lobeHalfValue(0x3555u) == doctest::Approx(1.0f / 3.0f).epsilon(1e-3));
    // Every binary16 normal value round-trips exactly.
    for (uint32_t h = 0x0400u; h < 0x7C00u; ++h) {
        REQUIRE(lobeHalfBits(lobeHalfValue(h)) == h);
        REQUIRE(lobeHalfBits(-lobeHalfValue(h)) == (h | 0x8000u));
    }
#if defined(__FLT16_MAX__)
    // Agrees with the compiler's binary16 conversion (round to nearest even) on the normal range.
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> mantissa(1.0f, 2.0f);
    std::uniform_int_distribution<int> exponent(-14, 16);
    for (int i = 0; i < 200000; ++i) {
        const float v = std::ldexp(mantissa(rng), exponent(rng));
        const _Float16 h = static_cast<_Float16>(v);
        uint16_t bits;
        std::memcpy(&bits, &h, sizeof bits);
        REQUIRE(lobeHalfBits(v) == bits);
    }
#endif
}

TEST_CASE("lobe guide: octahedral directions stay within 1e-4 rad") {
    CHECK(encodeLobeOct(0.0f, 0.0f, 0.0f) == 0u);
    CHECK(encodeLobeOct(0.0f, 0.0f, 2.0f) == 0u);
    const LobeVec3 up = decodeLobeOct(0u);
    CHECK(up.x == 0.0f);
    CHECK(up.z == 1.0f);
    std::mt19937 rng(11);
    std::normal_distribution<float> n(0.0f, 1.0f);
    float worst = 0.0f;
    for (int i = 0; i < 100000; ++i) {
        const float x = n(rng), y = n(rng), z = n(rng);
        if (x * x + y * y + z * z < 1e-6f) continue;
        worst = std::max(worst, angle(decodeLobeOct(encodeLobeOct(x, y, z)), x, y, z));
    }
    CHECK(worst < 1e-4f);
    for (const float s : {1.0f, -1.0f}) { // axes and the octahedron's fold
        CHECK(angle(decodeLobeOct(encodeLobeOct(0, 0, s)), 0, 0, s) < 1e-4f);
        CHECK(angle(decodeLobeOct(encodeLobeOct(s, 0, 0)), s, 0, 0) < 1e-4f);
        CHECK(angle(decodeLobeOct(encodeLobeOct(0, s, 0)), 0, s, 0) < 1e-4f);
    }
}

TEST_CASE("lobe guide: clear features pack to zero words and unpack to the neutral description values") {
    LobeParams p = allFeatures();
    p.features = 0u;
    const GPULobeGuide g = packLobeGuide(p);
    const uint32_t* w = reinterpret_cast<const uint32_t*>(&g);
    for (size_t i = 0; i < sizeof(GPULobeGuide) / 4; ++i) CHECK(w[i] == 0u);

    const LobeParams n = unpackLobeGuide(g);
    const MaterialDescription d{};
    SpecularBlock spec{};
    ClearcoatBlock coat{};
    SheenBlock sheen{};
    AnisotropyBlock aniso{};
    IridescenceBlock irid{};
    TransmissionBlock trans{};
    VolumeBlock volume{};
    SubsurfaceBlock sss{};
    (void)d;
    CHECK(n.ior == spec.ior);
    CHECK(n.specularFactor == spec.specularFactor);
    for (int c = 0; c < 3; ++c) CHECK(n.specularColor[c] == spec.specularColor[c]);
    CHECK(n.clearcoatFactor == coat.factor);
    CHECK(n.clearcoatNormal[2] == 0.0f);      // zero vector for a clear feature
    CHECK(n.anisotropyDirection[0] == 0.0f);
    CHECK(n.clearcoatRoughness == coat.roughness);
    for (int c = 0; c < 3; ++c) CHECK(n.sheenColor[c] == sheen.color[c]);
    CHECK(n.sheenRoughness == sheen.roughness);
    CHECK(n.anisotropyStrength == aniso.strength);
    CHECK(n.iridescenceFactor == irid.factor);
    CHECK(n.iridescenceIor == irid.ior);
    CHECK(n.iridescenceThickness == irid.thicknessMax);
    CHECK(n.transmissionFactor == trans.factor);
    CHECK(n.volumeThickness == volume.thickness);
    for (int c = 0; c < 3; ++c) CHECK(n.attenuationColor[c] == volume.attenuationColor[c]);
    CHECK(n.attenuationDistance == volume.attenuationDistance);
    CHECK(n.subsurfaceWeight == sss.weight);
    CHECK(n.subsurfaceRadius == sss.radius);
    for (int c = 0; c < 3; ++c) {
        CHECK(n.subsurfaceColor[c] == sss.color[c]);
        CHECK(n.subsurfaceRadiusScale[c] == sss.radiusScale[c]);
    }
    CHECK(n.subsurfaceAnisotropy == sss.anisotropy);
}

TEST_CASE("lobe guide: every feature round-trips within half precision; ior is exact") {
    const LobeParams p = allFeatures();
    const LobeParams q = unpackLobeGuide(packLobeGuide(p));
    CHECK(q.features == p.features);
    CHECK(q.ior == p.ior);
    CHECK(nearHalf(q.specularFactor, p.specularFactor));
    for (int c = 0; c < 3; ++c) {
        CHECK(nearHalf(q.specularColor[c], p.specularColor[c]));
        CHECK(nearHalf(q.sheenColor[c], p.sheenColor[c]));
        CHECK(nearHalf(q.attenuationColor[c], p.attenuationColor[c]));
        CHECK(nearHalf(q.subsurfaceColor[c], p.subsurfaceColor[c]));
        CHECK(nearHalf(q.subsurfaceRadiusScale[c], p.subsurfaceRadiusScale[c]));
    }
    CHECK(nearHalf(q.clearcoatFactor, p.clearcoatFactor));
    CHECK(nearHalf(q.clearcoatRoughness, p.clearcoatRoughness));
    CHECK(nearHalf(q.sheenRoughness, p.sheenRoughness));
    CHECK(nearHalf(q.anisotropyStrength, p.anisotropyStrength));
    CHECK(nearHalf(q.iridescenceFactor, p.iridescenceFactor));
    CHECK(nearHalf(q.iridescenceIor, p.iridescenceIor));
    CHECK(nearHalf(q.iridescenceThickness, p.iridescenceThickness));
    CHECK(nearHalf(q.transmissionFactor, p.transmissionFactor));
    CHECK(nearHalf(q.volumeThickness, p.volumeThickness));
    CHECK(std::isinf(q.attenuationDistance));
    CHECK(nearHalf(q.subsurfaceWeight, p.subsurfaceWeight));
    CHECK(nearHalf(q.subsurfaceRadius, p.subsurfaceRadius));
    CHECK(nearHalf(q.subsurfaceAnisotropy, p.subsurfaceAnisotropy));
    const LobeVec3 cn{q.clearcoatNormal[0], q.clearcoatNormal[1], q.clearcoatNormal[2]};
    CHECK(angle(cn, p.clearcoatNormal[0], p.clearcoatNormal[1], p.clearcoatNormal[2]) < 1e-4f);
    const LobeVec3 ad{q.anisotropyDirection[0], q.anisotropyDirection[1], q.anisotropyDirection[2]};
    CHECK(angle(ad, p.anisotropyDirection[0], p.anisotropyDirection[1], p.anisotropyDirection[2]) < 1e-4f);

    // ior is kept for transmission or volume alone (KHR_materials_ior neutral, bit clear).
    LobeParams t = p;
    t.features = MATERIAL_FEATURE_TRANSMISSION;
    CHECK(packLobeGuide(t).ior == p.ior);
    t.features = MATERIAL_FEATURE_CLEARCOAT;
    CHECK(packLobeGuide(t).ior == 0.0f);
    CHECK(unpackLobeGuide(packLobeGuide(t)).ior == LobeNeutralIor);
}
