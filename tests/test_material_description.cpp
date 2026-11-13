#include "renderer/material_description.h"

#include <doctest/doctest.h>

#include <cmath>
#include <string>

using namespace phosphor;

namespace {
bool mentions(const std::vector<std::string>& problems, const std::string& needle) {
    for (const auto& p : problems)
        if (p.find(needle) != std::string::npos) return true;
    return false;
}
} // namespace

TEST_CASE("material description: version constant and defaults follow the specifications") {
    CHECK(MATERIAL_DESCRIPTION_VERSION == 1);
    MaterialDescription d;
    CHECK(d.version == 1);
    CHECK(featureMask(d) == 0);
    CHECK(validate(d).empty());
    // KHR_materials_clearcoat / sheen / anisotropy / iridescence / volume defaults.
    CHECK(ClearcoatBlock{}.factor == 0.0f);
    CHECK(ClearcoatBlock{}.roughness == 0.0f);
    CHECK(SheenBlock{}.roughness == 0.0f);
    CHECK(AnisotropyBlock{}.strength == 0.0f);
    CHECK(IridescenceBlock{}.ior == doctest::Approx(1.3f));
    CHECK(IridescenceBlock{}.thicknessMin == doctest::Approx(100.0f));
    CHECK(IridescenceBlock{}.thicknessMax == doctest::Approx(400.0f));
    CHECK(SpecularBlock{}.ior == doctest::Approx(1.5f));
    CHECK(std::isinf(VolumeBlock{}.attenuationDistance));
    CHECK(VolumeBlock{}.attenuationDistance > 0.0f);
}

TEST_CASE("material description: feature bits are stable") {
    CHECK(featureBit(MaterialFeature::SpecularIor) == 1u);
    CHECK(featureBit(MaterialFeature::Clearcoat) == 2u);
    CHECK(featureBit(MaterialFeature::Sheen) == 4u);
    CHECK(featureBit(MaterialFeature::Anisotropy) == 8u);
    CHECK(featureBit(MaterialFeature::Iridescence) == 16u);
    CHECK(featureBit(MaterialFeature::Transmission) == 32u);
    CHECK(featureBit(MaterialFeature::Volume) == 64u);
    CHECK(featureBit(MaterialFeature::Subsurface) == 128u);
    CHECK(featureBit(MaterialFeature::Layers) == 256u);
}

TEST_CASE("material description: featureMask counts only active blocks") {
    MaterialDescription d;
    d.clearcoat = ClearcoatBlock{};            // factor 0: "the whole clearcoat layer is disabled"
    d.specular  = SpecularBlock{};             // neutral ior 1.5, factor 1, colour 1
    CHECK(featureMask(d) == 0);

    d.clearcoat->factor = 0.5f;
    CHECK(featureMask(d) == featureBit(MaterialFeature::Clearcoat));

    d.specular->ior = 1.33f;
    CHECK(featureMask(d) == (featureBit(MaterialFeature::Clearcoat) | featureBit(MaterialFeature::SpecularIor)));

    d.specular = SpecularBlock{};
    d.specular->ior = 0.0f; // special mode is active
    CHECK((featureMask(d) & featureBit(MaterialFeature::SpecularIor)) != 0);

    MaterialDescription e;
    e.sheen = SheenBlock{};
    CHECK(featureMask(e) == 0);
    e.sheen->color = {0.0f, 0.0f, 0.2f};
    CHECK(featureMask(e) == featureBit(MaterialFeature::Sheen));
    e.anisotropy = AnisotropyBlock{};
    e.anisotropy->strength = 0.3f;
    e.iridescence = IridescenceBlock{};
    e.iridescence->factor = 1.0f;
    e.transmission = TransmissionBlock{};
    e.transmission->factor = 1.0f;
    e.subsurface = SubsurfaceBlock{};
    e.subsurface->weight = 0.1f;
    CHECK(featureMask(e) == (4u | 8u | 16u | 32u | 128u));
}

TEST_CASE("material description: validate checks ranges from the specifications") {
    MaterialDescription d;
    d.clearcoat = ClearcoatBlock{};
    d.clearcoat->factor = 1.5f;
    CHECK(mentions(validate(d), "clearcoat.factor"));
    d.clearcoat->factor = 1.0f;
    CHECK(validate(d).empty());

    d.specular = SpecularBlock{};
    d.specular->ior = 0.5f; // schema: 0 or >= 1
    CHECK(mentions(validate(d), "specular.ior"));
    d.specular->ior = 0.0f;
    CHECK(validate(d).empty());
    d.specular->ior = 1.0f;
    CHECK(validate(d).empty());
    d.specular->specularColor = {2.0f, 1.0f, 1.0f}; // specularColor may exceed 1
    CHECK(validate(d).empty());
    d.specular->specularColor = {-0.1f, 1.0f, 1.0f};
    CHECK(mentions(validate(d), "specularColor"));

    MaterialDescription s;
    s.sheen = SheenBlock{};
    s.sheen->color = {0.0f, 1.2f, 0.0f};
    CHECK(mentions(validate(s), "sheen.color"));

    MaterialDescription i;
    i.iridescence = IridescenceBlock{};
    i.iridescence->ior = 0.9f;
    CHECK(mentions(validate(i), "iridescence.ior"));
    i.iridescence->ior = 1.3f;
    i.iridescence->thicknessMin = 500.0f; // may exceed the maximum per the specification
    CHECK(validate(i).empty());

    MaterialDescription v;
    v.transmission = TransmissionBlock{};
    v.transmission->factor = 1.0f;
    v.volume = VolumeBlock{};
    v.volume->thickness = 1.0f;
    v.volume->attenuationDistance = 0.0f;
    CHECK(mentions(validate(v), "attenuationDistance"));
    v.volume->attenuationDistance = 2.0f;
    v.volume->attenuationColor = {1.0f, 2.0f, 1.0f};
    CHECK(mentions(validate(v), "attenuationColor"));

    MaterialDescription u;
    u.subsurface = SubsurfaceBlock{};
    u.subsurface->anisotropy = 1.0f; // exclusive bound
    CHECK(mentions(validate(u), "subsurface.anisotropy"));
    u.subsurface->anisotropy = 0.0f;
    u.subsurface->weight = -0.5f;
    CHECK(mentions(validate(u), "subsurface.weight"));
}

TEST_CASE("material description: companion rules") {
    MaterialDescription d;
    d.volume = VolumeBlock{};
    d.volume->thickness = 0.5f;
    CHECK(mentions(validate(d), "volume requires transmission"));

    d.transmission = TransmissionBlock{};
    d.transmission->factor = 1.0f;
    CHECK(validate(d).empty());

    // Volumetric subsurface is the other companion; thin subsurface is not.
    MaterialDescription e;
    e.volume = VolumeBlock{};
    e.volume->thickness = 0.5f;
    e.subsurface = SubsurfaceBlock{};
    e.subsurface->weight = 1.0f;
    e.subsurface->mode = SubsurfaceMode::Thin;
    CHECK(mentions(validate(e), "volume requires"));
    e.subsurface->mode = SubsurfaceMode::Volumetric;
    CHECK(validate(e).empty());

    // Volumetric subsurface needs a volume.
    MaterialDescription f;
    f.subsurface = SubsurfaceBlock{};
    f.subsurface->weight = 1.0f;
    f.subsurface->mode = SubsurfaceMode::Volumetric;
    CHECK(mentions(validate(f), "volumetric subsurface requires a volume"));
}

TEST_CASE("material description: layer stack is bounded and validated") {
    MaterialDescription d;
    d.layers = LayerStack{};
    d.layers->mask.path = "mask.png";
    CHECK(mentions(validate(d), "layer stack needs 1 to 4"));

    for (u32 i = 0; i < 4; ++i) {
        MaterialLayer l;
        l.materialName = "m" + std::to_string(i);
        l.maskChannel  = static_cast<TextureChannel>(i);
        d.layers->layers.push_back(l);
    }
    CHECK(validate(d).empty());
    CHECK(featureMask(d) == featureBit(MaterialFeature::Layers));

    d.layers->layers.push_back(MaterialLayer{"m4", ~0u, TextureChannel::R, std::nullopt});
    CHECK(mentions(validate(d), "layer stack needs 1 to 4"));
    d.layers->layers.pop_back();

    d.layers->layers[1].maskChannel = TextureChannel::R; // duplicate channel
    CHECK(mentions(validate(d), "used by another layer"));
    d.layers->layers[1].maskChannel = TextureChannel::G;

    d.layers->layers[2].maskChannel = TextureChannel::Rgb; // not a single channel
    CHECK(mentions(validate(d), "must be R, G, B or A"));
    d.layers->layers[2].maskChannel = TextureChannel::B;

    d.layers->layers[0].heightContrast = -1.0f;
    CHECK(mentions(validate(d), "heightContrast"));
    d.layers->layers[0].heightContrast = 0.05f;
    CHECK(validate(d).empty());

    d.layers->layers[3].materialName.clear();
    CHECK(mentions(validate(d), "has no material"));
    d.layers->layers[3].materialIndex = 7;
    CHECK(validate(d, 8).empty());
    CHECK(mentions(validate(d, 5), "out of range"));

    d.layers->mask = textureSlot(TextureColorSpace::Linear, TextureChannel::Rgba);
    CHECK(mentions(validate(d), "mask texture"));
}

TEST_CASE("material description: mean free path is radius times scale") {
    SubsurfaceBlock s;
    s.radius = 0.02f;
    s.radiusScale = {1.0f, 0.5f, 0.25f};
    const auto mfp = meanFreePath(s);
    CHECK(mfp[0] == doctest::Approx(0.02f));
    CHECK(mfp[1] == doctest::Approx(0.01f));
    CHECK(mfp[2] == doctest::Approx(0.005f));
}

TEST_CASE("material description: uv transform worked example") {
    // Hand-worked: uv (0.25, 0.5), scale (2, 3) -> (0.5, 1.5); rotate 90 deg
    // counter-clockwise (x' = -y, y' = x) -> (-1.5, 0.5); offset (0.5, 0.25)
    // -> (-1.0, 0.75).
    UvTransform t;
    t.offset   = {0.5f, 0.25f};
    t.rotation = 1.57079632679f;
    t.scale    = {2.0f, 3.0f};
    const auto r = applyUvTransform(t, 0.25f, 0.5f);
    CHECK(r[0] == doctest::Approx(-1.0f).epsilon(1e-5));
    CHECK(r[1] == doctest::Approx(0.75f).epsilon(1e-5));
    const auto id = applyUvTransform(UvTransform{}, 0.3f, 0.7f);
    CHECK(id[0] == doctest::Approx(0.3f));
    CHECK(id[1] == doctest::Approx(0.7f));
}

TEST_CASE("material description: texture transform analysis") {
    MaterialDescription d;
    d.base.baseColorTexture.index = 1;
    d.base.normalTexture.index = 2;
    CHECK(analyzeTextureTransforms(d).status == TextureTransformStatus::None);

    UvTransform t;
    t.scale = {2.0f, 2.0f};
    d.base.baseColorTexture.transform = t;
    // Another present texture without the transform: not uniform.
    CHECK(analyzeTextureTransforms(d).status == TextureTransformStatus::NonUniform);
    d.base.normalTexture.transform = t;
    const auto uniform = analyzeTextureTransforms(d);
    CHECK(uniform.status == TextureTransformStatus::Uniform);
    CHECK(uniform.transform == t);

    UvTransform other = t;
    other.offset = {0.1f, 0.0f};
    d.base.normalTexture.transform = other;
    CHECK(analyzeTextureTransforms(d).status == TextureTransformStatus::NonUniform);

    d.base.normalTexture.transform = t;
    d.base.normalTexture.texCoord = 1;
    CHECK(analyzeTextureTransforms(d).status == TextureTransformStatus::NonUniform);
}

TEST_CASE("material description: forEachTexture visits feature slots") {
    MaterialDescription d;
    size_t base = 0;
    forEachTexture(d, [&](const char*, const MaterialTextureRef&) { ++base; });
    CHECK(base == 5);
    d.clearcoat = ClearcoatBlock{};
    d.sheen = SheenBlock{};
    size_t n = 0;
    forEachTexture(d, [&](const char*, const MaterialTextureRef&) { ++n; });
    CHECK(n == 5 + 3 + 2);
}
