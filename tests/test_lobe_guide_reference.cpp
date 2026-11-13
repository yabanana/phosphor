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
    GPUMaterial coat{};
    coat.flags = MATERIAL_FEATURE_CLEARCOAT << MATERIAL_FEATURE_SHIFT;
    CHECK(expectedLobeParams(coat, ext, MATERIAL_FEATURES_ALL).params.features == 0);
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
