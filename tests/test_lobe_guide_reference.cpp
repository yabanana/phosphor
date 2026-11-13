// F15.1 (ticket 06): CPU expectation and readback check of the lobe guide (renderer/lobe_guide_reference.h).
#include <doctest/doctest.h>
#include "renderer/lobe_guide_reference.h"

#include <cstring>
#include <vector>

using namespace phosphor;

namespace {
GPUMaterial extendedMaterial(u32 extensionIndex) {
    GPUMaterial m{};
    m.flags = MATERIAL_FEATURE_SPECULAR_IOR << MATERIAL_FEATURE_SHIFT;
    m.extensionIndex = extensionIndex;
    return m;
}

GPUMaterialExtRecord record0(float ior, float factor, float r, float g, float b, u32 specTex = INVALID_TEXTURE_INDEX,
                             u32 colorTex = INVALID_TEXTURE_INDEX) {
    GPUMaterialExt0 e{};
    e.ior = ior;
    e.specularFactor = factor;
    e.specularColor[0] = r; e.specularColor[1] = g; e.specularColor[2] = b;
    e.specularTex = specTex;
    e.specularColorTex = colorTex;
    e.clearcoatFactorTex = e.clearcoatRoughnessTex = e.clearcoatNormalTex = INVALID_TEXTURE_INDEX;
    e.sheenColorTex = e.sheenRoughnessTex = INVALID_TEXTURE_INDEX;
    GPUMaterialExtRecord rec{};
    std::memcpy(&rec, &e, sizeof(e));
    return rec;
}

// Two instances: slot 0 -> base material 0, slot 1 -> extended material 1, slot 2 -> textured material 2.
struct Fixture {
    std::vector<GPUInstance> instances = std::vector<GPUInstance>(3);
    std::vector<GPUMaterial> materials = std::vector<GPUMaterial>(3);
    std::vector<GPUMaterialExtRecord> extensions = std::vector<GPUMaterialExtRecord>(8);
    std::vector<GPUDISurface> surfaces;
    std::vector<GPULobeGuide> guides;
    LobeGuideCheckInput input;

    Fixture() {
        for (u32 i = 0; i < 3; ++i) {
            instances[i].materialIndex = i;
            instances[i].generation = 7 + i;
        }
        materials[1] = extendedMaterial(4);
        materials[2] = extendedMaterial(0);
        extensions[4] = record0(1.33f, 0.5f, 1.0f, 0.5f, 0.25f);
        extensions[0] = record0(1.5f, 1.0f, 1.0f, 1.0f, 1.0f, 3u);
        surfaces.resize(4 * 2);
        guides.resize(4 * 2);
        // pixels: 0 base, 1 extended, 2 textured, 3 background, 4..7 background
        setSurface(0, 0);
        setSurface(1, 1);
        setSurface(2, 2);
        guides[0] = packLobeGuide(expectedLobeParams(materials[0], extensions, MATERIAL_FEATURE_SPECULAR_IOR).params);
        guides[1] = packLobeGuide(expectedLobeParams(materials[1], extensions, MATERIAL_FEATURE_SPECULAR_IOR).params);
        rebind();
    }
    void setSurface(size_t pixel, u32 slot) {
        surfaces[pixel] = GPUDISurface{};
        surfaces[pixel].valid = 1;
        surfaces[pixel].instanceSlot = slot;
        surfaces[pixel].instanceGeneration = instances[slot].generation;
    }
    void rebind() {
        input = {};
        input.width = 4;
        input.height = 2;
        input.surfaces = surfaces;
        input.guides = guides;
        input.instances = instances;
        input.materials = materials;
        input.extensions = extensions;
        input.featureUnion = MATERIAL_FEATURE_SPECULAR_IOR;
    }
};
} // namespace

TEST_CASE("lobe guide reference: resolved parameters of ior/specular") {
    const GPUMaterialExtRecord ext[1] = {record0(1.33f, 0.5f, 1.0f, 0.5f, 0.25f)};
    const auto e = expectedLobeParams(extendedMaterial(0), ext, MATERIAL_FEATURE_SPECULAR_IOR);
    CHECK(e.params.features == MATERIAL_FEATURE_SPECULAR_IOR);
    CHECK(e.params.ior == 1.33f);
    CHECK(e.params.specularFactor == 0.5f);
    CHECK(e.params.specularColor[1] == 0.5f);
    CHECK_FALSE(e.unverified);
    // A base material, a union without the bit and a feature the resolve does not resolve yet: features 0.
    CHECK(expectedLobeParams(GPUMaterial{}, ext, MATERIAL_FEATURE_SPECULAR_IOR).params.features == 0);
    CHECK(expectedLobeParams(extendedMaterial(0), ext, 0).params.features == 0);
    GPUMaterial iridescent{};
    iridescent.flags = MATERIAL_FEATURE_IRIDESCENCE << MATERIAL_FEATURE_SHIFT;
    CHECK(expectedLobeParams(iridescent, ext, MATERIAL_FEATURES_ALL).params.features == 0);
    // Textures make the parameters unverifiable on the CPU.
    const GPUMaterialExtRecord textured[1] = {record0(1.5f, 1.0f, 1.0f, 1.0f, 1.0f, INVALID_TEXTURE_INDEX, 5u)};
    CHECK(expectedLobeParams(extendedMaterial(0), textured, MATERIAL_FEATURE_SPECULAR_IOR).unverified);
    // An extension group outside the mirror is reported.
    bool ok = true;
    (void)expectedLobeParams(extendedMaterial(9), ext, MATERIAL_FEATURE_SPECULAR_IOR, &ok);
    CHECK_FALSE(ok);
}

TEST_CASE("lobe guide reference: check passes on the expected guide") {
    Fixture f;
    const auto r = checkLobeGuide(f.input);
    CHECK(r.pass());
    CHECK(r.pixels == 3);
    CHECK(r.extended == 1);
    CHECK(r.base == 1);
    CHECK(r.unverified == 1);
    CHECK(r.invalid == 5);
    CHECK(formatLobeGuideCheck(12, r) == "LOBE-GUIDE check frame 12 | pixels 3 extended 1 base 1 unverified 1 | PASS");
}

TEST_CASE("lobe guide reference: every kind of corruption fails the check") {
    SUBCASE("ior shifted by 0.25 (the negative control)") {
        Fixture f;
        f.guides[1].ior += 0.25f;
        const auto r = checkLobeGuide(f.input);
        CHECK_FALSE(r.pass());
        CHECK(r.mismatches == 1);
        CHECK(formatLobeGuideCheck(3, r).find("FAIL") != std::string::npos);
    }
    SUBCASE("one half bit") {
        Fixture f;
        f.guides[1].specularFactorCoat ^= 1u;
        CHECK_FALSE(checkLobeGuide(f.input).pass());
    }
    SUBCASE("a base pixel with a record") {
        Fixture f;
        f.guides[0] = f.guides[1];
        CHECK_FALSE(checkLobeGuide(f.input).pass());
    }
    SUBCASE("an extended pixel without a record") {
        Fixture f;
        f.guides[1] = GPULobeGuide{};
        CHECK_FALSE(checkLobeGuide(f.input).pass());
    }
    SUBCASE("a background pixel with a record") {
        Fixture f;
        f.guides[5] = f.guides[1];
        const auto r = checkLobeGuide(f.input);
        CHECK_FALSE(r.pass());
        CHECK(r.mismatches == 1);
    }
    SUBCASE("a dead instance") {
        Fixture f;
        f.surfaces[1].instanceGeneration += 1;
        CHECK_FALSE(checkLobeGuide(f.input).pass());
    }
    SUBCASE("a textured pixel is not compared") {
        Fixture f;
        f.guides[2].ior = 99.0f;
        const auto r = checkLobeGuide(f.input);
        CHECK(r.pass());
        CHECK(r.unverified == 1);
    }
    SUBCASE("only four mismatches are described") {
        Fixture f;
        for (auto& g : f.guides) g.ior = 3.0f;
        const auto r = checkLobeGuide(f.input);
        CHECK(r.mismatches >= 5);
        CHECK(r.firstMismatches.size() == 4);
    }
}

TEST_CASE("lobe guide reference: a union without the bit expects a zero record") {
    Fixture f;
    f.input.featureUnion = 0;
    const auto r = checkLobeGuide(f.input);
    CHECK_FALSE(r.pass()); // pixel 1 holds a specular record the scene union forbids
    CHECK(r.extended == 0);
}

// F15.1 (ticket 09): the clearcoat fields.
namespace {
GPUMaterialExtRecord coatRecord(float factor, float roughness, u32 factorTex = INVALID_TEXTURE_INDEX, u32 normalTex = INVALID_TEXTURE_INDEX) {
    GPUMaterialExt0 e{};
    e.ior = 1.5f; e.specularFactor = 1.0f; e.specularColor[0] = e.specularColor[1] = e.specularColor[2] = 1.0f;
    e.specularTex = e.specularColorTex = INVALID_TEXTURE_INDEX;
    e.clearcoatFactor = factor; e.clearcoatRoughness = roughness;
    e.clearcoatFactorTex = factorTex; e.clearcoatRoughnessTex = INVALID_TEXTURE_INDEX; e.clearcoatNormalTex = normalTex;
    e.sheenColorTex = e.sheenRoughnessTex = INVALID_TEXTURE_INDEX;
    GPUMaterialExtRecord rec{};
    std::memcpy(&rec, &e, sizeof(e));
    return rec;
}
GPUMaterial coatMaterial(u32 extensionIndex, u32 normalTex = INVALID_TEXTURE_INDEX) {
    GPUMaterial m{};
    m.flags = MATERIAL_FEATURE_CLEARCOAT << MATERIAL_FEATURE_SHIFT;
    m.extensionIndex = extensionIndex;
    m.normalTex = normalTex;
    return m;
}
} // namespace

TEST_CASE("lobe guide reference: resolved parameters of clearcoat (factor saturated, roughness clamped to [0.04, 1])") {
    const GPUMaterialExtRecord ext[3] = {coatRecord(0.75f, 0.3f), coatRecord(2.0f, 0.0f), coatRecord(0.5f, 0.5f, 4u)};
    auto e = expectedLobeParams(coatMaterial(0), ext, MATERIAL_FEATURE_CLEARCOAT);
    CHECK(e.params.features == MATERIAL_FEATURE_CLEARCOAT);
    CHECK(e.params.clearcoatFactor == 0.75f);
    CHECK(e.params.clearcoatRoughness == 0.3f);
    CHECK_FALSE(e.unverified);
    CHECK(e.coatNormalChecked);
    CHECK(e.coatNormalIsShading);
    e = expectedLobeParams(coatMaterial(1), ext, MATERIAL_FEATURE_CLEARCOAT);
    CHECK(e.params.clearcoatFactor == 1.0f);
    CHECK(e.params.clearcoatRoughness == 0.04f);
    CHECK(expectedLobeParams(coatMaterial(2), ext, MATERIAL_FEATURE_CLEARCOAT).unverified); // factor texture
    // A base normal map (or a coat normal texture) leaves the coat normal unverifiable by angle (unit length only).
    CHECK_FALSE(expectedLobeParams(coatMaterial(0, 9u), ext, MATERIAL_FEATURE_CLEARCOAT).coatNormalIsShading);
    const GPUMaterialExtRecord normalTex[1] = {coatRecord(1.0f, 0.2f, INVALID_TEXTURE_INDEX, 6u)};
    CHECK_FALSE(expectedLobeParams(coatMaterial(0), normalTex, MATERIAL_FEATURE_CLEARCOAT).coatNormalIsShading);
    // The union gates the feature like the shader.
    CHECK(expectedLobeParams(coatMaterial(0), ext, MATERIAL_FEATURE_SPECULAR_IOR).params.features == 0);
}

TEST_CASE("lobe guide reference: the coat normal is compared by angle to the shading normal") {
    Fixture f;
    f.materials[1] = coatMaterial(4);
    f.extensions[4] = coatRecord(1.0f, 0.2f);
    f.input.featureUnion = MATERIAL_FEATURE_CLEARCOAT;
    f.surfaces[1].shadingNormal[0] = 0.6f; f.surfaces[1].shadingNormal[1] = 0.0f; f.surfaces[1].shadingNormal[2] = 0.8f;
    auto write = [&](float nx, float ny, float nz) {
        LobeParams p = expectedLobeParams(f.materials[1], f.extensions, MATERIAL_FEATURE_CLEARCOAT).params;
        p.clearcoatNormal[0] = nx; p.clearcoatNormal[1] = ny; p.clearcoatNormal[2] = nz;
        f.guides[1] = packLobeGuide(p);
    };
    f.materials[0] = GPUMaterial{};
    f.guides[0] = GPULobeGuide{};
    f.materials[2] = GPUMaterial{}; // slot 2 (pixel 2) is a base material here
    f.guides[2] = GPULobeGuide{};
    write(0.6f, 0.0f, 0.8f);
    CHECK(checkLobeGuide(f.input).pass());
    write(0.0f, 0.6f, 0.8f); // another direction: 45 degrees off
    CHECK_FALSE(checkLobeGuide(f.input).pass());
    write(0.3f, 0.0f, 0.4f); // not a unit vector decodes to a unit vector, equal in direction: passes
    CHECK(checkLobeGuide(f.input).pass());
    write(0.0f, 0.0f, 0.0f); // the zero vector decodes to +Z, not the shading normal
    CHECK_FALSE(checkLobeGuide(f.input).pass());
    // The factor and the roughness are compared bit for bit.
    write(0.6f, 0.0f, 0.8f);
    f.guides[1].specularFactorCoat ^= 1u << 16;
    CHECK_FALSE(checkLobeGuide(f.input).pass());
}

// F15.1 (ticket 10): the sheen fields (colour as authored, roughness clamped to [0.07, 1]).
namespace {
GPUMaterialExtRecord sheenRecord(float r, float g, float b, float roughness, u32 colorTex = INVALID_TEXTURE_INDEX) {
    GPUMaterialExt0 e{};
    e.ior = 1.5f; e.specularFactor = 1.0f; e.specularColor[0] = e.specularColor[1] = e.specularColor[2] = 1.0f;
    e.specularTex = e.specularColorTex = INVALID_TEXTURE_INDEX;
    e.clearcoatFactorTex = e.clearcoatRoughnessTex = e.clearcoatNormalTex = INVALID_TEXTURE_INDEX;
    e.sheenColor[0] = r; e.sheenColor[1] = g; e.sheenColor[2] = b; e.sheenRoughness = roughness;
    e.sheenColorTex = colorTex; e.sheenRoughnessTex = INVALID_TEXTURE_INDEX;
    GPUMaterialExtRecord rec{};
    std::memcpy(&rec, &e, sizeof(e));
    return rec;
}
GPUMaterial sheenMaterial(u32 extensionIndex) {
    GPUMaterial m{};
    m.flags = MATERIAL_FEATURE_SHEEN << MATERIAL_FEATURE_SHIFT;
    m.extensionIndex = extensionIndex;
    return m;
}
} // namespace

TEST_CASE("lobe guide reference: resolved parameters of sheen (roughness clamped to [0.07, 1])") {
    const GPUMaterialExtRecord ext[3] = {sheenRecord(1.0f, 0.5f, 0.25f, 0.3f), sheenRecord(0.5f, 0.5f, 0.5f, 0.0f),
                                         sheenRecord(0.5f, 0.5f, 0.5f, 0.4f, 5u)};
    auto e = expectedLobeParams(sheenMaterial(0), ext, MATERIAL_FEATURE_SHEEN);
    CHECK(e.params.features == MATERIAL_FEATURE_SHEEN);
    CHECK(e.params.sheenColor[0] == 1.0f);
    CHECK(e.params.sheenColor[1] == 0.5f);
    CHECK(e.params.sheenColor[2] == 0.25f);
    CHECK(e.params.sheenRoughness == 0.3f);
    CHECK_FALSE(e.unverified);
    e = expectedLobeParams(sheenMaterial(1), ext, MATERIAL_FEATURE_SHEEN);
    CHECK(e.params.sheenRoughness == 0.07f);
    CHECK(expectedLobeParams(sheenMaterial(2), ext, MATERIAL_FEATURE_SHEEN).unverified); // colour texture
    // A scene union without the bit resolves nothing.
    CHECK(expectedLobeParams(sheenMaterial(0), ext, MATERIAL_FEATURE_SPECULAR_IOR).params.features == 0);
    // The packed record round-trips the half-precision sheen fields.
    const GPULobeGuide g = packLobeGuide(expectedLobeParams(sheenMaterial(0), ext, MATERIAL_FEATURE_SHEEN).params);
    const LobeParams u = unpackLobeGuide(g);
    CHECK(u.sheenColor[1] == 0.5f);
    CHECK(u.sheenRoughness == doctest::Approx(0.3f).epsilon(1e-3));
}

// F15.1 (ticket 11): anisotropy.  Record 1 carries the strength; the direction depends on the mesh tangent frame and is compared by angle.
namespace {
GPUMaterialExtRecord anisotropyRecord(float strength, u32 texture = INVALID_TEXTURE_INDEX) {
    GPUMaterialExt1 e{};
    e.anisotropyStrength = strength;
    e.anisotropyRotation = 0.5f;
    e.anisotropyTex = texture;
    e.iridescenceFactorTex = e.iridescenceThicknessTex = e.transmissionTex = e.volumeThicknessTex = INVALID_TEXTURE_INDEX;
    GPUMaterialExtRecord rec{};
    std::memcpy(&rec, &e, sizeof(e));
    return rec;
}
GPUMaterial anisotropicMaterial(u32 extensionIndex) {
    GPUMaterial m{};
    m.flags = MATERIAL_FEATURE_ANISOTROPY << MATERIAL_FEATURE_SHIFT;
    m.extensionIndex = extensionIndex;
    return m;
}
} // namespace

TEST_CASE("lobe guide reference: resolved strength of anisotropy, direction checked by angle") {
    const GPUMaterialExtRecord ext[4] = {GPUMaterialExtRecord{}, anisotropyRecord(0.6f), GPUMaterialExtRecord{}, anisotropyRecord(0.5f, 5u)};
    auto e = expectedLobeParams(anisotropicMaterial(0), ext, MATERIAL_FEATURE_ANISOTROPY);
    CHECK(e.params.features == MATERIAL_FEATURE_ANISOTROPY);
    CHECK(e.params.anisotropyStrength == 0.6f);
    CHECK(e.anisotropyDirectionChecked);
    CHECK_FALSE(e.unverified);
    CHECK(expectedLobeParams(anisotropicMaterial(2), ext, MATERIAL_FEATURE_ANISOTROPY).unverified); // the direction texture
    CHECK(expectedLobeParams(anisotropicMaterial(0), ext, MATERIAL_FEATURE_SPECULAR_IOR).params.features == 0);
    // The record of the last group is out of range for the strength record.
    bool valid = true;
    CHECK(expectedLobeParams(anisotropicMaterial(3), ext, MATERIAL_FEATURE_ANISOTROPY, &valid).unverified);
    CHECK_FALSE(valid);
    const LobeParams u = unpackLobeGuide(packLobeGuide(e.params));
    CHECK(u.anisotropyStrength == doctest::Approx(0.6f).epsilon(1e-3));
}

TEST_CASE("lobe guide reference: the anisotropy direction is compared by angle (tangent plane), the strength bit for bit") {
    Fixture f;
    f.materials[1] = anisotropicMaterial(4);
    f.extensions[4] = GPUMaterialExtRecord{};
    f.extensions[5] = anisotropyRecord(0.6f);
    f.input.featureUnion = MATERIAL_FEATURE_ANISOTROPY;
    f.surfaces[1].shadingNormal[0] = 0.6f; f.surfaces[1].shadingNormal[1] = 0.0f; f.surfaces[1].shadingNormal[2] = 0.8f;
    f.materials[0] = GPUMaterial{};
    f.guides[0] = GPULobeGuide{};
    f.materials[2] = GPUMaterial{};
    f.guides[2] = GPULobeGuide{};
    auto write = [&](float dx, float dy, float dz) {
        LobeParams p = expectedLobeParams(f.materials[1], f.extensions, MATERIAL_FEATURE_ANISOTROPY).params;
        p.anisotropyDirection[0] = dx; p.anisotropyDirection[1] = dy; p.anisotropyDirection[2] = dz;
        f.guides[1] = packLobeGuide(p);
    };
    write(0.0f, 1.0f, 0.0f); // orthogonal to the shading normal
    CHECK(checkLobeGuide(f.input).pass());
    write(0.8f, 0.0f, -0.6f); // also in the tangent plane, any rotation passes
    CHECK(checkLobeGuide(f.input).pass());
    write(0.6f, 0.0f, 0.8f); // along the shading normal
    CHECK_FALSE(checkLobeGuide(f.input).pass());
    write(0.0f, 0.0f, 0.0f); // decodes to +Z: not in the tangent plane
    CHECK_FALSE(checkLobeGuide(f.input).pass());
    write(0.0f, 1.0f, 0.0f);
    f.guides[1].anisotropyIridescence ^= 1u; // the strength is compared bit for bit
    CHECK_FALSE(checkLobeGuide(f.input).pass());
}
