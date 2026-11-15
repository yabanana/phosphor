// Seam 1 (F15 ticket 04): small glTF JSON strings -> material descriptions and
// the support report.  Expected values come from the specification text cited
// in .scratch/f15/research/khr-materials.md, not from the code.
#include "renderer/material_description.h"
#include "renderer/material_support.h"
#include "scene/gltf_material_extensions.h"

#include <doctest/doctest.h>
#include <json.hpp>
#include <tiny_gltf.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>

using namespace phosphor;

namespace {

tinygltf::Model loadModel(const std::string& json) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;
    REQUIRE_MESSAGE(loader.LoadASCIIFromString(&model, &err, &warn, json.c_str(),
                                               static_cast<unsigned>(json.size()), ""),
                    err);
    return model;
}

// A model with one material (materialJson) and an optional extra top level.
tinygltf::Model modelWithMaterial(const std::string& materialJson, const std::string& top = "") {
    return loadModel(R"({"asset":{"version":"2.0"},)" + top + R"("materials":[)" + materialJson + "]}");
}

GltfMaterialParse parseOne(const std::string& materialJson) {
    const tinygltf::Model model = modelWithMaterial(materialJson);
    // Identity resolver, marked +100 so a resolved index differs from the glTF one.
    return parseGltfMaterial(model.materials.at(0), [](int index, bool) { return static_cast<u32>(index + 100); });
}

std::string withExt(const std::string& name, const std::string& body) {
    return R"({"extensions":{")" + name + R"(":)" + body + "}}";
}

bool contains(const std::vector<std::string>& v, const std::string& s) {
    return std::find(v.begin(), v.end(), s) != v.end();
}

bool anyWarning(const GltfMaterialParse& p, const std::string& needle) {
    for (const auto& w : p.info.warnings)
        if (w.find(needle) != std::string::npos) return true;
    return false;
}

} // namespace

TEST_CASE("glTF materials: a core-only material has an empty feature mask") {
    const auto p = parseOne(R"({"name":"stone","pbrMetallicRoughness":{"baseColorFactor":[0.5,0.25,1,1],
        "metallicFactor":0.2,"roughnessFactor":0.7,"baseColorTexture":{"index":3}},
        "normalTexture":{"index":4,"scale":0.5},"occlusionTexture":{"index":5,"strength":0.25},
        "emissiveFactor":[1,0.5,0],"alphaMode":"MASK","alphaCutoff":0.3,"doubleSided":true})");
    CHECK(featureMask(p.description) == 0);
    CHECK(p.info.extensions.empty());
    CHECK(p.info.warnings.empty());
    const auto& b = p.description.base;
    CHECK(b.baseColor[0] == doctest::Approx(0.5f));
    CHECK(b.baseColor[1] == doctest::Approx(0.25f));
    CHECK(b.metallic == doctest::Approx(0.2f));
    CHECK(b.roughness == doctest::Approx(0.7f));
    CHECK(b.normalScale == doctest::Approx(0.5f));
    CHECK(b.occlusionStrength == doctest::Approx(0.25f));
    CHECK(b.emissive[1] == doctest::Approx(0.5f));
    CHECK(b.alphaCutoff == doctest::Approx(0.3f));
    CHECK(b.doubleSided);
    // Resolved bindless indices and colour spaces of the core slots.
    CHECK(b.baseColorTexture.index == 103u);
    CHECK(b.baseColorTexture.colorSpace == TextureColorSpace::SRGB);
    CHECK(b.normalTexture.index == 104u);
    CHECK(b.normalTexture.colorSpace == TextureColorSpace::Linear);
    CHECK(b.occlusionTexture.index == 105u);
    CHECK_FALSE(b.emissiveTexture.present());
    CHECK(b.metallicRoughnessTexture.index == INVALID_TEXTURE_INDEX);

    const auto report = buildSupportReport(p.description, p.info);
    REQUIRE(report.valid());
    CHECK(report.base.feature == "base");
    CHECK(report.base.level == SupportLevel::Implemented);
    CHECK(report.features.empty());
    CHECK(report.encodings.empty());
}

TEST_CASE("glTF materials: alpha cutoff only applies to MASK") {
    CHECK(parseOne(R"({"alphaMode":"OPAQUE","alphaCutoff":0.7})").description.base.alphaCutoff == 0.0f);
    CHECK(parseOne(R"({"alphaMode":"MASK","alphaCutoff":0.7})").description.base.alphaCutoff == doctest::Approx(0.7f));
}

TEST_CASE("glTF materials: KHR_materials_clearcoat") {
    SUBCASE("defaults of an empty extension") {
        const auto p = parseOne(withExt("KHR_materials_clearcoat", "{}"));
        REQUIRE(p.description.clearcoat);
        CHECK(p.description.clearcoat->factor == 0.0f);
        CHECK(p.description.clearcoat->roughness == 0.0f);
        CHECK(p.description.clearcoat->normalScale == 1.0f);
        CHECK_FALSE(p.description.clearcoat->normalTexture.present());
        // "If clearcoatFactor is zero, the whole clearcoat layer is disabled."
        CHECK(featureMask(p.description) == 0);
    }
    SUBCASE("parameters, texture channels and colour spaces") {
        const auto p = parseOne(withExt("KHR_materials_clearcoat", R"({"clearcoatFactor":0.8,
            "clearcoatRoughnessFactor":0.25,"clearcoatTexture":{"index":1},
            "clearcoatRoughnessTexture":{"index":2},
            "clearcoatNormalTexture":{"index":3,"scale":0.6}})"));
        const auto& c = *p.description.clearcoat;
        CHECK(c.factor == doctest::Approx(0.8f));
        CHECK(c.roughness == doctest::Approx(0.25f));
        CHECK(c.normalScale == doctest::Approx(0.6f));
        // clearcoatTexture.r, clearcoatRoughnessTexture.g, normal map; all linear.
        CHECK(c.factorTexture.index == 101u);
        CHECK(c.factorTexture.channel == TextureChannel::R);
        CHECK(c.factorTexture.colorSpace == TextureColorSpace::Linear);
        CHECK(c.roughnessTexture.index == 102u);
        CHECK(c.roughnessTexture.channel == TextureChannel::G);
        CHECK(c.roughnessTexture.colorSpace == TextureColorSpace::Linear);
        CHECK(c.normalTexture.index == 103u);
        CHECK(c.normalTexture.colorSpace == TextureColorSpace::Linear);
        CHECK(featureMask(p.description) == featureBit(MaterialFeature::Clearcoat));
        CHECK(p.info.warnings.empty());
    }
    SUBCASE("out of range values are kept and reported") {
        const auto p = parseOne(withExt("KHR_materials_clearcoat", R"({"clearcoatFactor":2})"));
        CHECK(p.description.clearcoat->factor == 2.0f);
        CHECK(anyWarning(p, "clearcoat.factor"));
    }
}

TEST_CASE("glTF materials: KHR_materials_sheen") {
    const auto d = parseOne(withExt("KHR_materials_sheen", R"({"sheenColorFactor":[0.1,0.2,0.3],
        "sheenRoughnessFactor":0.4,"sheenColorTexture":{"index":7},"sheenRoughnessTexture":{"index":8}})"));
    const auto& s = *d.description.sheen;
    CHECK(s.color[0] == doctest::Approx(0.1f));
    CHECK(s.color[2] == doctest::Approx(0.3f));
    CHECK(s.roughness == doctest::Approx(0.4f));
    // sheenColorTexture.rgb is sRGB-encoded, sheenRoughnessTexture.a is linear.
    CHECK(s.colorTexture.colorSpace == TextureColorSpace::SRGB);
    CHECK(s.colorTexture.channel == TextureChannel::Rgb);
    CHECK(s.roughnessTexture.colorSpace == TextureColorSpace::Linear);
    CHECK(s.roughnessTexture.channel == TextureChannel::A);
    CHECK(featureMask(d.description) == featureBit(MaterialFeature::Sheen));

    // Default sheen colour is zero: the layer is disabled.
    CHECK(featureMask(parseOne(withExt("KHR_materials_sheen", "{}")).description) == 0);
}

TEST_CASE("glTF materials: KHR_materials_anisotropy") {
    const auto p = parseOne(withExt("KHR_materials_anisotropy",
                                    R"({"anisotropyStrength":0.5,"anisotropyRotation":1.5,"anisotropyTexture":{"index":2}})"));
    const auto& a = *p.description.anisotropy;
    CHECK(a.strength == doctest::Approx(0.5f));
    CHECK(a.rotation == doctest::Approx(1.5f));
    CHECK(a.texture.channel == TextureChannel::Rgb); // rg direction, b strength
    CHECK(a.texture.colorSpace == TextureColorSpace::Linear);
    CHECK(featureMask(p.description) == featureBit(MaterialFeature::Anisotropy));

    const auto def = parseOne(withExt("KHR_materials_anisotropy", "{}"));
    CHECK(def.description.anisotropy->strength == 0.0f);
    CHECK(def.description.anisotropy->rotation == 0.0f);
}

TEST_CASE("glTF materials: KHR_materials_transmission and volume") {
    const auto t = parseOne(withExt("KHR_materials_transmission",
                                    R"({"transmissionFactor":0.9,"transmissionTexture":{"index":4}})"));
    CHECK(t.description.transmission->factor == doctest::Approx(0.9f));
    CHECK(t.description.transmission->texture.channel == TextureChannel::R);
    CHECK(t.description.transmission->texture.colorSpace == TextureColorSpace::Linear);

    // volume defaults: thickness 0, attenuationColor [1,1,1], attenuationDistance +inf.
    const auto vd = parseOne(withExt("KHR_materials_volume", "{}"));
    CHECK(vd.description.volume->thickness == 0.0f);
    CHECK(vd.description.volume->attenuationColor == std::array<float, 3>{1, 1, 1});
    CHECK(std::isinf(vd.description.volume->attenuationDistance));
    CHECK(featureMask(vd.description) == 0); // thickness 0 = thin-walled, volume absent

    const auto v = parseOne(R"({"extensions":{"KHR_materials_transmission":{"transmissionFactor":1},
        "KHR_materials_volume":{"thicknessFactor":0.5,"attenuationColor":[0.9,0.5,0.1],
        "attenuationDistance":0.25,"thicknessTexture":{"index":6}}}})");
    const auto& vol = *v.description.volume;
    CHECK(vol.thickness == doctest::Approx(0.5f));
    CHECK(vol.attenuationColor[1] == doctest::Approx(0.5f));
    CHECK(vol.attenuationDistance == doctest::Approx(0.25f));
    CHECK(vol.thicknessTexture.channel == TextureChannel::G); // thicknessTexture.g, linear
    CHECK(vol.thicknessTexture.colorSpace == TextureColorSpace::Linear);
    CHECK(featureMask(v.description) ==
          (featureBit(MaterialFeature::Transmission) | featureBit(MaterialFeature::Volume)));
    CHECK(v.info.warnings.empty());

    // volume without a transmission-like companion is reported.
    const auto lonely = parseOne(withExt("KHR_materials_volume", R"({"thicknessFactor":1})"));
    CHECK(anyWarning(lonely, "volume requires transmission"));
}

TEST_CASE("glTF materials: KHR_materials_ior and KHR_materials_specular") {
    const auto def = parseOne(withExt("KHR_materials_specular", "{}"));
    const auto& d = *def.description.specular;
    CHECK(d.ior == doctest::Approx(1.5f));
    CHECK(d.specularFactor == 1.0f);
    CHECK(d.specularColor == std::array<float, 3>{1, 1, 1});
    CHECK(featureMask(def.description) == 0); // neutral

    const auto p = parseOne(R"({"extensions":{"KHR_materials_ior":{"ior":1.33},
        "KHR_materials_specular":{"specularFactor":0.5,"specularColorFactor":[2,1,0.5],
        "specularTexture":{"index":1},"specularColorTexture":{"index":2}}}})");
    const auto& s = *p.description.specular;
    CHECK(s.ior == doctest::Approx(1.33f));
    CHECK(s.specularFactor == doctest::Approx(0.5f));
    CHECK(s.specularColor[0] == doctest::Approx(2.0f)); // may exceed 1
    // specularTexture.a linear; specularColorTexture.rgb sRGB.
    CHECK(s.specularTexture.channel == TextureChannel::A);
    CHECK(s.specularTexture.colorSpace == TextureColorSpace::Linear);
    CHECK(s.specularColorTexture.channel == TextureChannel::Rgb);
    CHECK(s.specularColorTexture.colorSpace == TextureColorSpace::SRGB);
    CHECK(featureMask(p.description) == featureBit(MaterialFeature::SpecularIor));
    CHECK(p.info.warnings.empty());

    // ior = 0 is the permanent special mode and is valid; 0.5 is not.
    const auto zero = parseOne(withExt("KHR_materials_ior", R"({"ior":0})"));
    CHECK(zero.description.specular->ior == 0.0f);
    CHECK(zero.info.warnings.empty());
    CHECK(featureMask(zero.description) == featureBit(MaterialFeature::SpecularIor));
    CHECK(anyWarning(parseOne(withExt("KHR_materials_ior", R"({"ior":0.5})")), "specular.ior"));
}

TEST_CASE("glTF materials: KHR_materials_iridescence") {
    const auto def = parseOne(withExt("KHR_materials_iridescence", "{}"));
    const auto& d = *def.description.iridescence;
    CHECK(d.factor == 0.0f);
    CHECK(d.ior == doctest::Approx(1.3f));
    CHECK(d.thicknessMin == doctest::Approx(100.0f));
    CHECK(d.thicknessMax == doctest::Approx(400.0f));

    const auto p = parseOne(withExt("KHR_materials_iridescence",
        R"({"iridescenceFactor":1,"iridescenceIor":1.8,"iridescenceThicknessMinimum":200,
            "iridescenceThicknessMaximum":800,"iridescenceTexture":{"index":1},
            "iridescenceThicknessTexture":{"index":2}})"));
    const auto& i = *p.description.iridescence;
    CHECK(i.ior == doctest::Approx(1.8f));
    CHECK(i.thicknessMin == doctest::Approx(200.0f));
    CHECK(i.thicknessMax == doctest::Approx(800.0f));
    CHECK(i.factorTexture.channel == TextureChannel::R);
    CHECK(i.thicknessTexture.channel == TextureChannel::G);
    CHECK(i.factorTexture.colorSpace == TextureColorSpace::Linear);
    CHECK(i.thicknessTexture.colorSpace == TextureColorSpace::Linear);
    CHECK(featureMask(p.description) == featureBit(MaterialFeature::Iridescence));
    // Minimum may exceed maximum: not a warning.
    const auto swapped = parseOne(withExt("KHR_materials_iridescence",
        R"({"iridescenceFactor":1,"iridescenceThicknessMinimum":900,"iridescenceThicknessMaximum":300})"));
    CHECK(swapped.info.warnings.empty());
}

TEST_CASE("glTF materials: KHR_materials_emissive_strength folds into the emissive factor") {
    const auto p = parseOne(R"({"emissiveFactor":[1,0.5,0],"extensions":{"KHR_materials_emissive_strength":{"emissiveStrength":5}}})");
    CHECK(p.description.base.emissive[0] == doctest::Approx(5.0f));
    CHECK(p.description.base.emissive[1] == doctest::Approx(2.5f));
    CHECK(p.description.base.emissive[2] == doctest::Approx(0.0f));
    CHECK(p.info.emissiveStrengthUsed);
    CHECK(featureMask(p.description) == 0);

    // Default strength is 1.
    const auto d = parseOne(R"({"emissiveFactor":[0.2,0.2,0.2],"extensions":{"KHR_materials_emissive_strength":{}}})");
    CHECK(d.description.base.emissive[0] == doctest::Approx(0.2f));

    const auto report = buildSupportReport(p.description, p.info);
    bool found = false;
    CHECK(report.features.empty()); // an encoding is not a material feature
    for (const auto& f : report.encodings)
        if (f.feature == "emissive_strength") {
            found = true;
            CHECK(f.level == SupportLevel::Implemented);
        }
    CHECK(found);
}

TEST_CASE("glTF materials: KHR_materials_diffuse_transmission (release candidate) maps to thin subsurface") {
    const auto p = parseOne(withExt("KHR_materials_diffuse_transmission",
        R"({"diffuseTransmissionFactor":0.6,"diffuseTransmissionColorFactor":[1,0.5,0.25],
            "diffuseTransmissionTexture":{"index":1},"diffuseTransmissionColorTexture":{"index":2}})"));
    const auto& s = *p.description.subsurface;
    CHECK(s.mode == SubsurfaceMode::Thin);
    CHECK(s.source == SubsurfaceSource::KhrMaterialsDiffuseTransmission);
    CHECK(s.weight == doctest::Approx(0.6f));
    CHECK(s.color[1] == doctest::Approx(0.5f));
    // diffuseTransmissionTexture.a linear, diffuseTransmissionColorTexture.rgb sRGB.
    CHECK(s.weightTexture.channel == TextureChannel::A);
    CHECK(s.weightTexture.colorSpace == TextureColorSpace::Linear);
    CHECK(s.colorTexture.channel == TextureChannel::Rgb);
    CHECK(s.colorTexture.colorSpace == TextureColorSpace::SRGB);
    CHECK_FALSE(s.draft.has_value());
    CHECK(featureMask(p.description) == featureBit(MaterialFeature::Subsurface));

    // Default colour [1,1,1], default factor 0 (disabled).
    const auto def = parseOne(withExt("KHR_materials_diffuse_transmission", "{}"));
    CHECK(def.description.subsurface->color == std::array<float, 3>{1, 1, 1});
    CHECK(featureMask(def.description) == 0);

    const auto report = buildSupportReport(p.description, p.info);
    bool found = false;
    for (const auto& f : report.features)
        if (f.feature == "subsurface") {
            found = true;
            CHECK(f.label == InputLabel::ReleaseCandidate);
            CHECK(f.source == "KHR_materials_diffuse_transmission");
            CHECK(f.level == SupportLevel::Approximated);
        }
    CHECK(found);
}

TEST_CASE("glTF materials: draft KHR_materials_scatter (PR 2579 at a6cd5b4)") {
    // Expected values are worked from the draft formulas (research/khr-materials.md 10.3):
    //   thin:       base = mix(diffuse, mix(btdf, scatter_bsdf(multiscatterColor, g), scatterStrength), transmission)
    //               -> scatter weight = transmission * scatterStrength, colour = multiscatterColor
    //   volumetric: surface unchanged -> weight = transmission; rho_ms* = scatterStrength * multiscatterColor
    SUBCASE("thin-walled without a volume: weight is transmission x scatterStrength") {
        const auto p = parseOne(R"({"extensions":{"KHR_materials_transmission":{"transmissionFactor":0.5},
            "KHR_materials_scatter":{"scatterStrengthFactor":0.7,"multiscatterColorFactor":[0.9,0.6,0.3],
            "scatterAnisotropy":0.25,"scatterStrengthTexture":{"index":1},"multiscatterColorTexture":{"index":2}}}})");
        const auto& s = *p.description.subsurface;
        CHECK(s.mode == SubsurfaceMode::Thin);
        CHECK(s.source == SubsurfaceSource::KhrMaterialsScatterDraft);
        REQUIRE(s.draft.has_value());
        CHECK(s.draft->scatterStrength == doctest::Approx(0.7f)); // raw draft values are kept
        CHECK(s.draft->multiscatterColor[1] == doctest::Approx(0.6f));
        CHECK(s.draft->scatterAnisotropy == doctest::Approx(0.25f));
        CHECK(s.weight == doctest::Approx(0.35f)); // 0.5 * 0.7
        CHECK(s.color[0] == doctest::Approx(0.9f));
        CHECK(s.color[1] == doctest::Approx(0.6f));
        CHECK(s.color[2] == doctest::Approx(0.3f));
        CHECK(s.anisotropy == doctest::Approx(0.25f));
        CHECK(s.weightTexture.channel == TextureChannel::A);
        CHECK(s.colorTexture.colorSpace == TextureColorSpace::SRGB);
        CHECK(p.info.warnings.empty());
        CHECK((featureMask(p.description) & featureBit(MaterialFeature::Subsurface)) != 0u);

        const auto report = buildSupportReport(p.description, p.info);
        bool found = false;
        for (const auto& f : report.features)
            if (f.feature == "subsurface") {
                found = true;
                CHECK(f.label == InputLabel::Draft);
                CHECK(f.source == "KHR_materials_scatter");
            }
        CHECK(found);
    }
    SUBCASE("no effect without KHR_materials_transmission: warning, no subsurface block") {
        const auto p = parseOne(R"({"extensions":{"KHR_materials_scatter":{"scatterStrengthFactor":1,
            "multiscatterColorFactor":[0.9,0.6,0.3]}}})");
        CHECK_FALSE(p.description.subsurface.has_value());
        CHECK(featureMask(p.description) == 0);
        CHECK(anyWarning(p, "no effect without KHR_materials_transmission"));
        const auto volumetric = parseOne(R"({"extensions":{"KHR_materials_volume":{"thicknessFactor":1,
            "attenuationDistance":2,"attenuationColor":[0.5,0.5,0.5]},
            "KHR_materials_scatter":{"scatterStrengthFactor":1}}})");
        CHECK_FALSE(volumetric.description.subsurface.has_value());
    }
    SUBCASE("defaults of the draft with transmission: strength 0 (weight 0), colour 1, anisotropy 0") {
        const auto p = parseOne(R"({"extensions":{"KHR_materials_transmission":{"transmissionFactor":1},
            "KHR_materials_scatter":{}}})");
        REQUIRE(p.description.subsurface.has_value());
        CHECK(p.description.subsurface->weight == 0.0f);
        CHECK(p.description.subsurface->color == std::array<float, 3>{1, 1, 1});
        CHECK(p.description.subsurface->anisotropy == 0.0f);
        CHECK((featureMask(p.description) & featureBit(MaterialFeature::Subsurface)) == 0u); // zero weight: disabled
    }
    SUBCASE("a transmission factor of 0 disables the scatter lobe") {
        const auto p = parseOne(R"({"extensions":{"KHR_materials_transmission":{},
            "KHR_materials_scatter":{"scatterStrengthFactor":1}}})");
        REQUIRE(p.description.subsurface.has_value());
        CHECK(p.description.subsurface->weight == 0.0f);
    }
    SUBCASE("thickness 0 is thin-walled") {
        const auto p = parseOne(R"({"extensions":{"KHR_materials_transmission":{"transmissionFactor":0.8},
            "KHR_materials_volume":{"thicknessFactor":0},
            "KHR_materials_scatter":{"scatterStrengthFactor":0.5}}})");
        CHECK(p.description.subsurface->mode == SubsurfaceMode::Thin);
        CHECK(p.description.subsurface->weight == doctest::Approx(0.4f)); // 0.8 * 0.5
    }
    SUBCASE("volumetric: weight is the transmission, colour is strength x multiscatter, mean free path from the medium") {
        // sigma_t = -ln(c)/d, mean free path 1/sigma_t = d / -ln(c).
        // d = 2, c = (0.5, 0.25, 1): 2/ln 2 = 2.885390, 2/ln 4 = 1.442695, c = 1 -> no attenuation.
        const auto p = parseOne(R"({"extensions":{"KHR_materials_transmission":{"transmissionFactor":0.5},
            "KHR_materials_volume":{"thicknessFactor":1,"attenuationDistance":2,"attenuationColor":[0.5,0.25,1]},
            "KHR_materials_scatter":{"scatterStrengthFactor":0.5,"multiscatterColorFactor":[0.8,0.5,0.2],
            "scatterAnisotropy":-0.3}}})");
        const auto& s = *p.description.subsurface;
        CHECK(s.mode == SubsurfaceMode::Volumetric);
        CHECK(s.weight == doctest::Approx(0.5f));          // transmissionFactor
        CHECK(s.color[0] == doctest::Approx(0.4f));        // 0.5 * 0.8
        CHECK(s.color[1] == doctest::Approx(0.25f));       // 0.5 * 0.5
        CHECK(s.color[2] == doctest::Approx(0.1f));        // 0.5 * 0.2
        const auto mfp = meanFreePath(s);
        CHECK(mfp[0] == doctest::Approx(2.885390f).epsilon(1e-4));
        CHECK(mfp[1] == doctest::Approx(1.442695f).epsilon(1e-4));
        CHECK(mfp[2] >= 1.0e5f); // effectively infinite
        CHECK(s.anisotropy == doctest::Approx(-0.3f));
        REQUIRE(s.draft.has_value());
        CHECK(s.draft->scatterStrength == doctest::Approx(0.5f)); // raw values kept
        CHECK(s.draft->multiscatterColor[0] == doctest::Approx(0.8f));
        CHECK(p.info.warnings.empty());
    }
    SUBCASE("a multiscatter colour above 1 is not an error, only a warning") {
        const auto p = parseOne(R"({"extensions":{"KHR_materials_transmission":{"transmissionFactor":1},
            "KHR_materials_scatter":{"scatterStrengthFactor":1,"multiscatterColorFactor":[2.5,0.5,0.5]}}})");
        CHECK(p.description.subsurface->color[0] == doctest::Approx(2.5f));
        CHECK_FALSE(anyWarning(p, "invalid value"));
        CHECK(anyWarning(p, "subsurface.color"));
    }
    SUBCASE("scatter replaces diffuse transmission, with a warning") {
        const auto p = parseOne(R"({"extensions":{"KHR_materials_transmission":{"transmissionFactor":1},
            "KHR_materials_diffuse_transmission":{"diffuseTransmissionFactor":0.2},
            "KHR_materials_scatter":{"scatterStrengthFactor":0.9}}})");
        CHECK(p.description.subsurface->source == SubsurfaceSource::KhrMaterialsScatterDraft);
        CHECK(p.description.subsurface->weight == doctest::Approx(0.9f));
        CHECK(anyWarning(p, "excludes"));
    }
}

TEST_CASE("glTF materials: unknown extensions are listed per material") {
    const auto p = parseOne(R"({"extensions":{"KHR_materials_dispersion":{"dispersion":0.5},
        "KHR_materials_unlit":{},"EXT_vendor_thing":{"x":1},"KHR_materials_clearcoat":{"clearcoatFactor":1}}})");
    CHECK(contains(p.info.unknownExtensions, "KHR_materials_dispersion"));
    CHECK(contains(p.info.unknownExtensions, "KHR_materials_unlit"));
    CHECK(contains(p.info.unknownExtensions, "EXT_vendor_thing"));
    CHECK_FALSE(contains(p.info.unknownExtensions, "KHR_materials_clearcoat"));
    const auto report = buildSupportReport(p.description, p.info);
    CHECK(report.unknownExtensions.size() == 3);
}

TEST_CASE("glTF materials: malformed extension values are skipped with a warning") {
    const auto p = parseOne(withExt("KHR_materials_clearcoat", R"({"clearcoatFactor":"high","clearcoatTexture":{"texCoord":1}})"));
    CHECK(p.description.clearcoat->factor == 0.0f);
    CHECK_FALSE(p.description.clearcoat->factorTexture.present());
    CHECK(anyWarning(p, "clearcoatFactor"));
    CHECK(anyWarning(p, "clearcoatTexture"));
}

TEST_CASE("glTF materials: KHR_texture_transform parsing and uniform analysis") {
    SUBCASE("a uniform transform on every texture is reported implemented (baked)") {
        const std::string xf = R"("extensions":{"KHR_texture_transform":{"offset":[0.5,0.25],"rotation":1.5707963,"scale":[2,3]}})";
        const auto p = parseOne(R"({"pbrMetallicRoughness":{"baseColorTexture":{"index":0,)" + xf + R"(}},
            "normalTexture":{"index":1,)" + xf + "}}");
        const auto a = analyzeTextureTransforms(p.description);
        REQUIRE(a.status == TextureTransformStatus::Uniform);
        CHECK(a.transform.offset[0] == doctest::Approx(0.5f));
        CHECK(a.transform.offset[1] == doctest::Approx(0.25f));
        CHECK(a.transform.rotation == doctest::Approx(1.5707963f));
        CHECK(a.transform.scale[0] == doctest::Approx(2.0f));
        CHECK(a.transform.scale[1] == doctest::Approx(3.0f));
        const auto report = buildSupportReport(p.description, p.info);
        bool found = false;
        CHECK(report.features.empty());
        for (const auto& f : report.encodings)
            if (f.feature == "texture_transform") {
                found = true;
                CHECK(f.level == SupportLevel::Implemented);
            }
        CHECK(found);
    }
    SUBCASE("a transform applied to only one texture is evaluated per slot") {
        const auto p = parseOne(R"({"pbrMetallicRoughness":{"baseColorTexture":{"index":0,
            "extensions":{"KHR_texture_transform":{"scale":[2,2]}}}},"normalTexture":{"index":1}})");
        CHECK(p.info.transformStatus == TextureTransformStatus::PerSlot);
        const auto report = buildSupportReport(p.description, p.info);
        bool found = false;
        for (const auto& f : report.encodings)
            if (f.feature == "texture_transform") {
                found = true;
                CHECK(f.level == SupportLevel::Implemented);
            }
        CHECK(found);
    }
    SUBCASE("different transforms on extension textures are evaluated per slot") {
        const auto p = parseOne(R"({"pbrMetallicRoughness":{"baseColorTexture":{"index":0,
            "extensions":{"KHR_texture_transform":{"scale":[2,2]}}}},
            "extensions":{"KHR_materials_clearcoat":{"clearcoatFactor":1,"clearcoatTexture":{"index":1,
            "extensions":{"KHR_texture_transform":{"scale":[4,4]}}}}}})");
        CHECK(p.info.transformStatus == TextureTransformStatus::PerSlot);
    }
    SUBCASE("a texCoord override to another UV set: the set is copied into UV0 (single set per material)") {
        const auto p = parseOne(R"({"pbrMetallicRoughness":{"baseColorTexture":{"index":0,
            "extensions":{"KHR_texture_transform":{"scale":[2,2],"texCoord":1}}}}})");
        CHECK(p.description.base.baseColorTexture.texCoord == 1u);
        CHECK(p.info.transformStatus == TextureTransformStatus::Uniform);
        CHECK(analyzeTextureTransforms(p.description).uvSet == 1u);
    }
    SUBCASE("two UV sets inside one material are unsupported and say why") {
        const auto p = parseOne(R"({"pbrMetallicRoughness":{"baseColorTexture":{"index":0,
            "extensions":{"KHR_texture_transform":{"scale":[2,2],"texCoord":1}}}},"normalTexture":{"index":1}})");
        CHECK(p.info.transformStatus == TextureTransformStatus::Unsupported);
        CHECK_FALSE(p.info.transformReason.empty());
        const auto report = buildSupportReport(p.description, p.info);
        bool found = false;
        for (const auto& f : report.encodings)
            if (f.feature == "texture_transform") {
                found = true;
                CHECK(f.level == SupportLevel::Unsupported);
            }
        CHECK(found);
        CHECK_FALSE(report.warnings.empty());
    }
    SUBCASE("no transform: not reported") {
        const auto p = parseOne(R"({"pbrMetallicRoughness":{"baseColorTexture":{"index":0}}})");
        CHECK(p.info.transformStatus == TextureTransformStatus::None);
        const auto report = buildSupportReport(p.description, p.info);
        CHECK(report.features.empty());
        CHECK(report.encodings.empty());
    }
}

TEST_CASE("glTF materials: uv bake of a worked example") {
    // uv (0.25, 0.5) with offset (0.5, 0.25), rotation pi/2, scale (2, 3)
    // -> scaled (0.5, 1.5) -> rotated (-1.5, 0.5) -> translated (-1.0, 0.75).
    std::vector<glm::vec2> uvs = {{0.25f, 0.5f}, {0.0f, 0.0f}};
    UvTransform t;
    t.offset   = {0.5f, 0.25f};
    t.rotation = 1.57079632679f;
    t.scale    = {2.0f, 3.0f};
    bakeUvTransform(t, uvs);
    CHECK(uvs[0].x == doctest::Approx(-1.0f).epsilon(1e-5));
    CHECK(uvs[0].y == doctest::Approx(0.75f).epsilon(1e-5));
    // The origin maps to the offset.
    CHECK(uvs[1].x == doctest::Approx(0.5f));
    CHECK(uvs[1].y == doctest::Approx(0.25f));
}

TEST_CASE("glTF extensions: required extensions must be rendered, not merely parsed, or the asset is refused") {
    const std::string irid = R"({"asset":{"version":"2.0"},
        "extensionsUsed":["KHR_materials_transmission"],"extensionsRequired":["KHR_materials_transmission"]})";
    SUBCASE("a required extension whose feature is unsupported today is refused with a reason") {
        MaterialSupportTable unsupported = currentSupportTable(); // transmission is rendered since F15.1 ticket 13
        unsupported.features[static_cast<size_t>(MaterialFeature::Transmission)].level = SupportLevel::Unsupported;
        const auto bad = unsupportedRequiredExtensions(loadModel(irid), unsupported);
        REQUIRE(bad.size() == 1);
        CHECK(bad[0].name == "KHR_materials_transmission");
        CHECK(bad[0].reason.find("not rendered") != std::string::npos);
    }
    SUBCASE("the same asset loads once the feature is implemented or approximated") {
        MaterialSupportTable table = currentSupportTable();
        auto& entry = table.features[static_cast<size_t>(MaterialFeature::Transmission)];
        entry.level = SupportLevel::Unsupported;
        CHECK_FALSE(unsupportedRequiredExtensions(loadModel(irid), table).empty());
        entry.level = SupportLevel::Implemented;
        CHECK(unsupportedRequiredExtensions(loadModel(irid), table).empty());
        entry.level = SupportLevel::Approximated;
        entry.approximation = "test-approx";
        CHECK(unsupportedRequiredExtensions(loadModel(irid), table).empty());
    }
    SUBCASE("every material extension maps to its own feature") {
        struct Case { const char* name; MaterialFeature feature; };
        const Case cases[] = {
            {"KHR_materials_clearcoat", MaterialFeature::Clearcoat}, {"KHR_materials_sheen", MaterialFeature::Sheen},
            {"KHR_materials_anisotropy", MaterialFeature::Anisotropy},
            {"KHR_materials_transmission", MaterialFeature::Transmission}, {"KHR_materials_volume", MaterialFeature::Volume},
            {"KHR_materials_ior", MaterialFeature::SpecularIor}, {"KHR_materials_specular", MaterialFeature::SpecularIor},
            {"KHR_materials_iridescence", MaterialFeature::Iridescence},
            {"KHR_materials_diffuse_transmission", MaterialFeature::Subsurface},
            {"KHR_materials_scatter", MaterialFeature::Subsurface}};
        for (const Case& c : cases) {
            MaterialSupportTable table = currentSupportTable();
            // (ior/specular are approximated since F15.1 ticket 05: reset the entry under test)
            table.features[static_cast<size_t>(c.feature)].level = SupportLevel::Unsupported;
            CHECK_FALSE(isExtensionRendered(c.name, table));
            table.features[static_cast<size_t>(c.feature)].level = SupportLevel::Implemented;
            CHECK_MESSAGE(isExtensionRendered(c.name, table), c.name);
        }
    }
    SUBCASE("emissive strength and a uniform texture transform are rendered today") {
        const auto m = loadModel(R"({"asset":{"version":"2.0"},
            "extensionsUsed":["KHR_materials_emissive_strength","KHR_texture_transform"],
            "extensionsRequired":["KHR_materials_emissive_strength","KHR_texture_transform"],
            "materials":[{"pbrMetallicRoughness":{"baseColorTexture":{"index":0,
              "extensions":{"KHR_texture_transform":{"scale":[2,2]}}}}}]})");
        CHECK(unsupportedRequiredExtensions(m).empty());
    }
    SUBCASE("a required KHR_texture_transform with per-slot transforms is rendered (F15.1 ticket 22)") {
        const auto m = loadModel(R"({"asset":{"version":"2.0"},
            "extensionsUsed":["KHR_texture_transform"],"extensionsRequired":["KHR_texture_transform"],
            "materials":[{"name":"mixed","pbrMetallicRoughness":{"baseColorTexture":{"index":0,
              "extensions":{"KHR_texture_transform":{"scale":[2,2]}}}},"normalTexture":{"index":1}}]})");
        CHECK(unsupportedRequiredExtensions(m).empty());
    }
    SUBCASE("a required KHR_texture_transform on a material that mixes UV sets is refused and says why") {
        const auto m = loadModel(R"({"asset":{"version":"2.0"},
            "extensionsUsed":["KHR_texture_transform"],"extensionsRequired":["KHR_texture_transform"],
            "materials":[{"name":"twosets","pbrMetallicRoughness":{"baseColorTexture":{"index":0,"texCoord":1,
              "extensions":{"KHR_texture_transform":{"scale":[2,2]}}}},"normalTexture":{"index":1}}]})");
        const auto bad = unsupportedRequiredExtensions(m);
        REQUIRE(bad.size() == 1);
        CHECK(bad[0].name == "KHR_texture_transform");
        CHECK(bad[0].reason.find("UV sets") != std::string::npos);
        CHECK(bad[0].reason.find("twosets") != std::string::npos);
    }
    SUBCASE("unsupported required extensions are named") {
        for (const char* name : {"KHR_materials_unlit", "KHR_lights_punctual", "KHR_draco_mesh_compression",
                                 "KHR_texture_basisu", "EXT_meshopt_compression", "KHR_materials_dispersion"}) {
            const auto m = loadModel(std::string(R"({"asset":{"version":"2.0"},"extensionsUsed":[")") + name +
                                     R"("],"extensionsRequired":[")" + name + R"("]})");
            const auto bad = unsupportedRequiredExtensions(m);
            REQUIRE_MESSAGE(bad.size() == 1, name);
            CHECK(bad[0].name == name);
        }
    }
    SUBCASE("optional unknown extensions are listed, not refused") {
        const auto m = loadModel(R"({"asset":{"version":"2.0"},
            "extensionsUsed":["KHR_lights_punctual","KHR_materials_sheen","KHR_materials_dispersion"]})");
        CHECK(unsupportedRequiredExtensions(m).empty());
        const auto unknown = unknownOptionalExtensions(m);
        REQUIRE(unknown.size() == 2);
        CHECK(unknown[0] == "KHR_lights_punctual");
        CHECK(unknown[1] == "KHR_materials_dispersion");
    }
}

TEST_CASE("support report: summary line, detail and JSON") {
    std::vector<MaterialSupportReport> reports;
    {
        const auto p = parseOne(R"({"name":"paint","extensions":{"KHR_materials_clearcoat":{"clearcoatFactor":1},"KHR_materials_transmission":{"transmissionFactor":1},
            "KHR_materials_scatter":{"scatterStrengthFactor":1},"KHR_materials_dispersion":{}}})");
        reports.push_back(buildSupportReport(p.description, p.info));
    }
    {
        const auto p = parseOne(R"({"name":"plain"})");
        reports.push_back(buildSupportReport(p.description, p.info));
    }
    const std::vector<std::string> assetUnknown = {"KHR_lights_punctual"};
    const std::string line = formatSupportSummary(reports, assetUnknown);
    CHECK(line.rfind("MATERIALS count=2 base=[implemented(2)] ", 0) == 0);
    CHECK(line.find("implemented=0 approximated=3 unsupported=0") != std::string::npos); // clearcoat (ticket 09), thin subsurface (ticket 16) and transmission (ticket 13) are approximated
    CHECK(line.find("encodings=[]") != std::string::npos);
    CHECK(line.find("clearcoat:approximated(1)") != std::string::npos);
    CHECK(line.find("subsurface:approximated(1)") != std::string::npos);
    CHECK(line.find("KHR_lights_punctual") != std::string::npos);
    CHECK(line.find("KHR_materials_dispersion") != std::string::npos);

    const std::string detail = formatSupportDetail(reports[0]);
    CHECK(detail.find("'paint'") != std::string::npos);
    CHECK(detail.find("subsurface=approximated(thin_mode_only;") != std::string::npos);
    CHECK(detail.find("[draft]") != std::string::npos);

    const auto json = nlohmann::json::parse(supportToJson(reports, assetUnknown));
    CHECK(json.at("count") == 2);
    CHECK(json.at("schema") == "phosphor.materials.v2");
    CHECK(json.at("features").at("unsupported") == 0);
    CHECK(json.at("features").at("approximated") == 3);
    CHECK(json.at("features").at("implemented") == 0);
    CHECK(json.at("base").at("implemented") == 2);
    CHECK(json.at("encodings").at("implemented") == 0);
    CHECK(json.at("materials").at(0).at("name") == "paint");
    CHECK(json.at("materials").at(0).at("base").at("support") == "implemented");
    CHECK(json.at("materials").at(0).at("encodings").empty());
    bool sawDraft = false;
    for (const auto& f : json.at("materials").at(0).at("features"))
        if (f.at("feature") == "subsurface") {
            CHECK(f.at("label") == "draft");
            CHECK(f.at("support") == "approximated");
            sawDraft = true;
        }
    CHECK(sawDraft);
    CHECK(json.at("asset_unknown_extensions").at(0) == "KHR_lights_punctual");
}

TEST_CASE("support report: both subsurface modes are approximated, the volumetric one with its own named approximations") {
    const auto thin = parseOne(R"({"extensions":{"KHR_materials_diffuse_transmission":{"diffuseTransmissionFactor":0.5}}})");
    const auto thinReport = buildSupportReport(thin.description, thin.info);
    REQUIRE(thinReport.features.size() == 1);
    CHECK(thinReport.features[0].level == SupportLevel::Approximated);
    CHECK(thinReport.features[0].label == InputLabel::ReleaseCandidate);
    MaterialDescription volumetric = thin.description;
    volumetric.subsurface->mode = SubsurfaceMode::Volumetric;
    const auto volumetricReport = buildSupportReport(volumetric, thin.info);
    REQUIRE(volumetricReport.features.size() == 1);
    CHECK(volumetricReport.features[0].level == SupportLevel::Approximated); // ticket 17: screen-space diffusion
    CHECK(std::string(volumetricReport.features[0].approximation).find("no_back_lit_translucency") != std::string::npos);
}

TEST_CASE("support report: an approximated feature carries its named approximation") {
    MaterialSupportTable table = currentSupportTable();
    table.features[static_cast<size_t>(MaterialFeature::Clearcoat)].level = SupportLevel::Approximated;
    table.features[static_cast<size_t>(MaterialFeature::Clearcoat)].approximation = "test-approx";
    const auto p = parseOne(withExt("KHR_materials_clearcoat", R"({"clearcoatFactor":1})"));
    const auto report = buildSupportReport(p.description, p.info, table);
    bool found = false;
    for (const auto& f : report.features)
        if (f.feature == "clearcoat") {
            found = true;
            CHECK(f.level == SupportLevel::Approximated);
            CHECK(f.approximation == "test-approx");
        }
    CHECK(found);
    CHECK(formatSupportDetail(report).find("clearcoat=approximated(test-approx)") != std::string::npos);
}

TEST_CASE("image URI containment") {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "phosphor_uri_containment";
    fs::remove_all(root);
    fs::create_directories(root / "asset" / "textures");
    fs::create_directories(root / "outside");
    { std::ofstream(root / "outside" / "secret.png") << "x"; }
    { std::ofstream(root / "asset" / "textures" / "ok.png") << "x"; }
    std::error_code ec;
    fs::create_directory_symlink(root / "outside", root / "asset" / "link", ec);
    const std::string base = (root / "asset").string() + "/";
    std::string error;

    CHECK(isImageUriContained(base, "textures/ok.png", error));
    CHECK(isImageUriContained(base, "textures/not_yet_there.png", error));
    CHECK(isImageUriContained(base, "textures%2Fok.png", error));
    CHECK(isImageUriContained(base, "data:image/png;base64,AAAA", error));
    CHECK(isImageUriContained(base, "./textures/ok.png", error));

    CHECK_FALSE(isImageUriContained(base, "", error));
    CHECK_FALSE(isImageUriContained(base, "/etc/passwd", error));
    CHECK(error.find("absolute") != std::string::npos);
    CHECK_FALSE(isImageUriContained(base, "C:/Windows/x.png", error));
    CHECK_FALSE(isImageUriContained(base, "\\\\server\\share\\x.png", error));
    CHECK_FALSE(isImageUriContained(base, "file:///etc/passwd", error));
    CHECK_FALSE(isImageUriContained(base, "http://example.com/x.png", error));
    CHECK_FALSE(isImageUriContained(base, "../outside/secret.png", error));
    CHECK(error.find("'..'") != std::string::npos);
    CHECK_FALSE(isImageUriContained(base, "textures/../../outside/secret.png", error));
    CHECK_FALSE(isImageUriContained(base, "textures\\..\\..\\outside\\secret.png", error));
    CHECK_FALSE(isImageUriContained(base, "%2E%2E/outside/secret.png", error));
    CHECK_FALSE(isImageUriContained(base, "textures/%2e%2e/%2e%2e/outside/secret.png", error));

    if (!ec) { // symlink escape: the link resolves to the sibling directory
        CHECK_FALSE(isImageUriContained(base, "link/secret.png", error));
        CHECK(error.find("outside the asset directory") != std::string::npos);
    }
    fs::remove_all(root);
}

TEST_CASE("support report: base, features and glTF encodings are reported apart") {
    const auto p = parseOne(R"({"name":"mix","emissiveFactor":[1,1,1],"pbrMetallicRoughness":{"baseColorTexture":{"index":0,
        "extensions":{"KHR_texture_transform":{"scale":[2,2]}}}},
        "extensions":{"KHR_materials_emissive_strength":{"emissiveStrength":3},
        "KHR_materials_clearcoat":{"clearcoatFactor":1}}})");
    const auto report = buildSupportReport(p.description, p.info);
    CHECK(report.base.level == SupportLevel::Implemented);
    REQUIRE(report.features.size() == 1);
    CHECK(report.features[0].feature == "clearcoat");
    REQUIRE(report.encodings.size() == 2);
    CHECK(report.encodings[0].feature == "emissive_strength");
    CHECK(report.encodings[1].feature == "texture_transform");

    const std::string line = formatSupportSummary({report}, {});
    CHECK(line.find("features=[clearcoat:approximated(1)]") != std::string::npos);
    CHECK(line.find("encodings=[emissive_strength:implemented(1) texture_transform:implemented(1)]") != std::string::npos);
    CHECK(line.find("implemented=0 approximated=1 unsupported=0") != std::string::npos); // encodings are not features

    const auto json = nlohmann::json::parse(supportToJson({report}, {}));
    CHECK(json.at("features").at("approximated") == 1);
    CHECK(json.at("encodings").at("implemented") == 2);
    CHECK(json.at("materials").at(0).at("encodings").at(1).at("source") == "KHR_texture_transform");
    const std::string detail = formatSupportDetail(report);
    CHECK(detail.find("features: clearcoat=approximated") != std::string::npos);
    CHECK(detail.find("encodings: emissive_strength=implemented") != std::string::npos);
}

TEST_CASE("glTF materials: unresolvable textures and out-of-range values are noted, not applied") {
    const tinygltf::Model model = modelWithMaterial(R"({"pbrMetallicRoughness":{"baseColorTexture":{"index":2,"texCoord":-3}},
        "emissiveTexture":{"index":5},
        "extensions":{"KHR_materials_sheen":{"sheenColorFactor":[1,1,1],"sheenColorTexture":{"index":4,"texCoord":-1}},
        "KHR_materials_emissive_strength":{"emissiveStrength":-1}},"emissiveFactor":[1,1,1]})");
    // Textures 0..3 exist; 4 and 5 do not.
    const auto p = parseGltfMaterial(model.materials.at(0), [](int index, bool) {
        return index < 4 ? static_cast<u32>(index + 100) : INVALID_TEXTURE_INDEX;
    });
    CHECK(p.description.base.baseColorTexture.index == 102u);
    CHECK(p.description.base.baseColorTexture.texCoord == 0u); // negative texCoord read as UV set 0
    CHECK(anyWarning(p, "baseColorTexture: negative texCoord -3"));
    CHECK_FALSE(p.description.base.emissiveTexture.present());
    CHECK(anyWarning(p, "emissiveTexture: texture index 5"));
    REQUIRE(p.description.sheen.has_value());
    CHECK_FALSE(p.description.sheen->colorTexture.present());
    CHECK(anyWarning(p, "sheenColorTexture: texture index 4"));
    // The authored negative strength stays in the description.
    CHECK(p.description.base.emissive[0] == doctest::Approx(-1.0f));
    CHECK(anyWarning(p, "emissiveStrength: negative"));
    // The warnings reach the support report.
    CHECK(buildSupportReport(p.description, p.info).warnings.size() == p.info.warnings.size());
}
