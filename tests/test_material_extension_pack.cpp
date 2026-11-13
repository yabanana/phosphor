#include "renderer/material_extension_pack.h"

#include <doctest/doctest.h>

#include <cmath>
#include <cstddef>
#include <cstring>

using namespace phosphor;

namespace {
float wordF(const GPUMaterialExtRecord& r, size_t i) {
    float f;
    std::memcpy(&f, &r.w[i], sizeof(f));
    return f;
}
} // namespace

// The word offsets below are the frozen layout of decision 07 / ADR-0002 written
// as literals: a reordered field fails here, not in a shader.
TEST_CASE("extension records: sizes and offsets are frozen") {
    CHECK(sizeof(GPUMaterial) == 80);
    CHECK(sizeof(GPUMaterialExt0) == 80);
    CHECK(sizeof(GPUMaterialExt1) == 80);
    CHECK(sizeof(GPUMaterialExt2) == 80);
    CHECK(sizeof(GPUMaterialExt3) == 80);
    CHECK(sizeof(GPUMaterialExtRecord) == 80);
    CHECK(MATERIAL_EXT_RECORDS == 4);
    CHECK(MATERIAL_FEATURE_SHIFT == 8);
    CHECK(offsetof(GPUMaterial, flags) == 68);
    CHECK(offsetof(GPUMaterial, extensionIndex) == 72);
    CHECK(offsetof(GPUMaterial, pad1) == 76);

    CHECK(offsetof(GPUMaterialExt0, ior) == 0);
    CHECK(offsetof(GPUMaterialExt0, specularFactor) == 4);
    CHECK(offsetof(GPUMaterialExt0, specularColor) == 8);
    CHECK(offsetof(GPUMaterialExt0, specularTex) == 20);
    CHECK(offsetof(GPUMaterialExt0, specularColorTex) == 24);
    CHECK(offsetof(GPUMaterialExt0, clearcoatFactor) == 28);
    CHECK(offsetof(GPUMaterialExt0, clearcoatNormalTex) == 48);
    CHECK(offsetof(GPUMaterialExt0, sheenColor) == 52);
    CHECK(offsetof(GPUMaterialExt0, sheenRoughness) == 64);
    CHECK(offsetof(GPUMaterialExt0, sheenRoughnessTex) == 72);

    CHECK(offsetof(GPUMaterialExt1, anisotropyStrength) == 0);
    CHECK(offsetof(GPUMaterialExt1, iridescenceFactor) == 12);
    CHECK(offsetof(GPUMaterialExt1, iridescenceThicknessTex) == 32);
    CHECK(offsetof(GPUMaterialExt1, transmissionFactor) == 36);
    CHECK(offsetof(GPUMaterialExt1, volumeThickness) == 44);
    CHECK(offsetof(GPUMaterialExt1, attenuationColor) == 52);
    CHECK(offsetof(GPUMaterialExt1, attenuationDistance) == 64);

    CHECK(offsetof(GPUMaterialExt2, weight) == 4);
    CHECK(offsetof(GPUMaterialExt2, color) == 8);
    CHECK(offsetof(GPUMaterialExt2, radius) == 20);
    CHECK(offsetof(GPUMaterialExt2, radiusScale) == 24);
    CHECK(offsetof(GPUMaterialExt2, anisotropy) == 36);
    CHECK(offsetof(GPUMaterialExt2, radiusScaleTex) == 48);

    CHECK(offsetof(GPUMaterialExt3, layerCount) == 4);
    CHECK(offsetof(GPUMaterialExt3, layerMaterial) == 8);
    CHECK(offsetof(GPUMaterialExt3, packedMaskChannels) == 24);
    CHECK(offsetof(GPUMaterialExt3, heightContrast) == 28);
}

TEST_CASE("extension records: a default description packs the neutral parameters") {
    const MaterialExtGroup g = packMaterialExtensions(MaterialDescription{});
    // Record 0 words: ior 1.5, specular 1, colour (1,1,1), textures absent.
    CHECK(wordF(g[0], 0) == doctest::Approx(1.5f));
    CHECK(wordF(g[0], 1) == 1.0f);
    CHECK(wordF(g[0], 2) == 1.0f);
    CHECK(wordF(g[0], 4) == 1.0f);
    CHECK(g[0].w[5] == INVALID_TEXTURE_INDEX);
    CHECK(g[0].w[6] == INVALID_TEXTURE_INDEX);
    CHECK(wordF(g[0], 7) == 0.0f);   // clearcoat factor
    CHECK(wordF(g[0], 9) == 1.0f);   // clearcoat normal scale
    CHECK(wordF(g[0], 13) == 0.0f);  // sheen colour
    // Record 1: iridescence ior 1.3, thickness 100..400, attenuation (1,1,1) at +inf.
    CHECK(wordF(g[1], 4) == doctest::Approx(1.3f));
    CHECK(wordF(g[1], 5) == 100.0f);
    CHECK(wordF(g[1], 6) == 400.0f);
    CHECK(std::isinf(wordF(g[1], 16)));
    // Record 3: no layers.
    CHECK(g[3].w[0] == INVALID_TEXTURE_INDEX);
    CHECK(g[3].w[1] == 0);
    CHECK(g[3].w[2] == MATERIAL_EXTENSION_INVALID);
}

TEST_CASE("extension records: specular, ior and the other blocks land at their words") {
    MaterialDescription d;
    d.specular.emplace();
    d.specular->ior            = 1.33f;
    d.specular->specularFactor = 0.5f;
    d.specular->specularColor  = {2.0f, 1.0f, 0.25f};
    d.specular->specularTexture.index      = 7;
    d.specular->specularColorTexture.index = 9;
    d.clearcoat.emplace();
    d.clearcoat->factor    = 0.8f;
    d.clearcoat->roughness = 0.1f;
    d.clearcoat->normalTexture.index = 11;
    d.sheen.emplace();
    d.sheen->color     = {0.1f, 0.2f, 0.3f};
    d.sheen->roughness = 0.6f;
    d.anisotropy.emplace();
    d.anisotropy->strength = 0.4f;
    d.anisotropy->rotation = 1.25f;
    d.iridescence.emplace();
    d.iridescence->factor       = 1.0f;
    d.iridescence->thicknessMin = 250.0f;
    d.transmission.emplace();
    d.transmission->factor = 0.9f;
    d.volume.emplace();
    d.volume->thickness           = 0.01f;
    d.volume->attenuationColor    = {0.5f, 0.6f, 0.7f};
    d.volume->attenuationDistance = 3.0f;
    d.subsurface.emplace();
    d.subsurface->mode   = SubsurfaceMode::Volumetric;
    d.subsurface->weight = 0.75f;
    d.subsurface->radius = 0.002f;

    const MaterialExtGroup g = packMaterialExtensions(d);
    CHECK(wordF(g[0], 0) == 1.33f);
    CHECK(wordF(g[0], 1) == 0.5f);
    CHECK(wordF(g[0], 2) == 2.0f);
    CHECK(wordF(g[0], 3) == 1.0f);
    CHECK(wordF(g[0], 4) == 0.25f);
    CHECK(g[0].w[5] == 7);
    CHECK(g[0].w[6] == 9);
    CHECK(wordF(g[0], 7) == 0.8f);
    CHECK(wordF(g[0], 8) == 0.1f);
    CHECK(g[0].w[12] == 11);
    CHECK(wordF(g[0], 13) == 0.1f);
    CHECK(wordF(g[0], 15) == 0.3f);
    CHECK(wordF(g[0], 16) == 0.6f);
    CHECK(wordF(g[1], 0) == 0.4f);
    CHECK(wordF(g[1], 1) == 1.25f);
    CHECK(wordF(g[1], 3) == 1.0f);
    CHECK(wordF(g[1], 5) == 250.0f);
    CHECK(wordF(g[1], 9) == 0.9f);
    CHECK(wordF(g[1], 11) == 0.01f);
    CHECK(wordF(g[1], 13) == 0.5f);
    CHECK(wordF(g[1], 15) == 0.7f);
    CHECK(wordF(g[1], 16) == 3.0f);
    CHECK(g[2].w[0] == 1u);
    CHECK(wordF(g[2], 1) == 0.75f);
    CHECK(wordF(g[2], 5) == 0.002f);
    // The typed view agrees with the raw words.
    CHECK(unpackExtRecord<GPUMaterialExt0>(g[0]).specularFactor == 0.5f);
    CHECK(unpackExtRecord<GPUMaterialExt1>(g[1]).attenuationDistance == 3.0f);
}

TEST_CASE("extension records: layer stack packs channels, materials and height contrast") {
    MaterialDescription d;
    d.layers.emplace();
    d.layers->mask.index = 5;
    MaterialLayer a, b, c;
    a.materialIndex = 3;  a.maskChannel = TextureChannel::R;
    b.materialIndex = 8;  b.maskChannel = TextureChannel::B; b.heightContrast = 0.002f;
    c.materialIndex = 12; c.maskChannel = TextureChannel::A;
    d.layers->layers = {a, b, c};
    const MaterialExtGroup g = packMaterialExtensions(d);
    CHECK(g[3].w[0] == 5);
    CHECK(g[3].w[1] == 3);
    CHECK(g[3].w[2] == 3);
    CHECK(g[3].w[3] == 8);
    CHECK(g[3].w[4] == 12);
    CHECK(g[3].w[5] == MATERIAL_EXTENSION_INVALID);
    // R=0 (bits 0-1), B=2 (bits 2-3), A=3 (bits 4-5).
    CHECK(g[3].w[6] == (0u | (2u << 2) | (3u << 4)));
    CHECK(wordF(g[3], 7) == -1.0f);
    CHECK(wordF(g[3], 8) == 0.002f);
    CHECK(wordF(g[3], 9) == -1.0f);
}

TEST_CASE("extension records: feature flags are the feature mask shifted by 8") {
    MaterialDescription d;
    CHECK(featureFlags(d) == 0);
    d.specular.emplace();
    d.specular->ior = 1.33f;
    CHECK(featureFlags(d) == (1u << 8));
    d.clearcoat.emplace();
    d.clearcoat->factor = 1.0f;
    CHECK(featureFlags(d) == ((1u << 8) | (1u << 9)));
    GPUMaterial m{};
    m.flags = MATERIAL_FLAG_DOUBLE_SIDED | featureFlags(d);
    CHECK(materialFeatureBits(m) == 3u);
}

#include "renderer/material_support.h"

TEST_CASE("support report: ior/specular is approximated with a named approximation, others stay unsupported") {
    const MaterialSupportTable& t = currentSupportTable();
    const FeatureSupportEntry& s = t.features[static_cast<u32>(MaterialFeature::SpecularIor)];
    CHECK(s.level == SupportLevel::Approximated);
    CHECK(std::string(s.approximation) == "di_and_specular_transport_use_base_model");
    for (u32 f = 1; f < MATERIAL_FEATURE_COUNT; ++f) CHECK(t.features[f].level == SupportLevel::Unsupported);

    MaterialDescription d;
    d.specular.emplace();
    d.specular->ior = 1.33f;
    MaterialSourceInfo info;
    info.extensions = {"KHR_materials_ior"};
    const MaterialSupportReport r = buildSupportReport(d, info);
    REQUIRE(r.features.size() == 1);
    CHECK(r.features[0].feature == "specular_ior");
    CHECK(r.features[0].level == SupportLevel::Approximated);
    CHECK(r.features[0].approximation == "di_and_specular_transport_use_base_model");
    CHECK(formatSupportSummary({r}, {}).find("approximated=1") != std::string::npos);
}
