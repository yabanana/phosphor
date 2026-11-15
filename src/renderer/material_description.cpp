#include "renderer/material_description.h"

#include <string_view>

#include <algorithm>
#include <cmath>

namespace phosphor {

const char* featureName(MaterialFeature f) {
    switch (f) {
        case MaterialFeature::SpecularIor:  return "specular_ior";
        case MaterialFeature::Clearcoat:    return "clearcoat";
        case MaterialFeature::Sheen:        return "sheen";
        case MaterialFeature::Anisotropy:   return "anisotropy";
        case MaterialFeature::Iridescence:  return "iridescence";
        case MaterialFeature::Transmission: return "transmission";
        case MaterialFeature::Volume:       return "volume";
        case MaterialFeature::Subsurface:   return "subsurface";
        case MaterialFeature::Layers:       return "layers";
    }
    return "unknown";
}

bool featureActive(const MaterialDescription& d, MaterialFeature f) {
    switch (f) {
        case MaterialFeature::SpecularIor: {
            if (!d.specular) return false;
            const SpecularBlock& s = *d.specular;
            const SpecularBlock neutral{};
            return s.ior != neutral.ior || s.specularFactor != neutral.specularFactor ||
                   s.specularColor != neutral.specularColor || s.specularTexture.present() ||
                   s.specularColorTexture.present();
        }
        case MaterialFeature::Clearcoat:    return d.clearcoat && d.clearcoat->factor > 0.0f;
        case MaterialFeature::Sheen: {
            if (!d.sheen) return false;
            const auto& c = d.sheen->color;
            return c[0] != 0.0f || c[1] != 0.0f || c[2] != 0.0f;
        }
        case MaterialFeature::Anisotropy:   return d.anisotropy && d.anisotropy->strength > 0.0f;
        case MaterialFeature::Iridescence:  return d.iridescence && d.iridescence->factor > 0.0f;
        case MaterialFeature::Transmission: return d.transmission && d.transmission->factor > 0.0f;
        case MaterialFeature::Volume:       return d.volume && d.volume->thickness > 0.0f;
        case MaterialFeature::Subsurface:   return d.subsurface && d.subsurface->weight > 0.0f;
        case MaterialFeature::Layers:       return d.layers && !d.layers->layers.empty();
    }
    return false;
}

u32 featureMask(const MaterialDescription& d) {
    u32 mask = 0;
    for (u32 i = 0; i < MATERIAL_FEATURE_COUNT; ++i) {
        const auto f = static_cast<MaterialFeature>(i);
        if (featureActive(d, f)) mask |= featureBit(f);
    }
    return mask;
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

namespace {

struct Checker {
    std::vector<std::string>& out;

    void inRange(const char* name, float v, float lo, float hi) {
        if (!(v >= lo && v <= hi)) out.push_back(std::string(name) + " " + std::to_string(v) + " outside [" +
                                                std::to_string(lo) + ", " + std::to_string(hi) + "]");
    }
    void atLeast(const char* name, float v, float lo) {
        if (!(v >= lo)) out.push_back(std::string(name) + " " + std::to_string(v) + " below " + std::to_string(lo));
    }
    void finite(const char* name, float v) {
        if (!std::isfinite(v)) out.push_back(std::string(name) + " is not finite");
    }
    template <size_t N>
    void eachInRange(const char* name, const std::array<float, N>& v, float lo, float hi) {
        for (size_t i = 0; i < N; ++i) inRange(name, v[i], lo, hi);
    }
    template <size_t N>
    void eachAtLeast(const char* name, const std::array<float, N>& v, float lo) {
        for (size_t i = 0; i < N; ++i) atLeast(name, v[i], lo);
    }
    void error(std::string s) { out.push_back(std::move(s)); }
};

} // namespace

std::vector<std::string> validate(const MaterialDescription& d, u32 materialCount) {
    std::vector<std::string> out;
    Checker c{out};

    if (d.version != MATERIAL_DESCRIPTION_VERSION)
        c.error("unsupported material description version " + std::to_string(d.version));

    const BaseMaterial& b = d.base;
    c.eachInRange("base.baseColor", b.baseColor, 0.0f, 1.0f);
    c.inRange("base.metallic", b.metallic, 0.0f, 1.0f);
    c.inRange("base.roughness", b.roughness, 0.0f, 1.0f);
    c.finite("base.normalScale", b.normalScale);
    c.inRange("base.occlusionStrength", b.occlusionStrength, 0.0f, 1.0f);
    c.eachAtLeast("base.emissive", b.emissive, 0.0f);
    for (float e : b.emissive) c.finite("base.emissive", e);
    c.atLeast("base.alphaCutoff", b.alphaCutoff, 0.0f);

    if (d.specular) {
        const SpecularBlock& s = *d.specular;
        if (!(s.ior == 0.0f || s.ior >= 1.0f)) c.error("specular.ior " + std::to_string(s.ior) + " must be 0 or >= 1");
        c.finite("specular.ior", s.ior);
        c.inRange("specular.specularFactor", s.specularFactor, 0.0f, 1.0f);
        c.eachAtLeast("specular.specularColor", s.specularColor, 0.0f);
    }
    if (d.clearcoat) {
        c.inRange("clearcoat.factor", d.clearcoat->factor, 0.0f, 1.0f);
        c.inRange("clearcoat.roughness", d.clearcoat->roughness, 0.0f, 1.0f);
        c.finite("clearcoat.normalScale", d.clearcoat->normalScale);
    }
    if (d.sheen) {
        c.eachInRange("sheen.color", d.sheen->color, 0.0f, 1.0f);
        c.inRange("sheen.roughness", d.sheen->roughness, 0.0f, 1.0f);
    }
    if (d.anisotropy) {
        c.inRange("anisotropy.strength", d.anisotropy->strength, 0.0f, 1.0f);
        c.finite("anisotropy.rotation", d.anisotropy->rotation);
    }
    if (d.iridescence) {
        const IridescenceBlock& i = *d.iridescence;
        c.inRange("iridescence.factor", i.factor, 0.0f, 1.0f);
        c.atLeast("iridescence.ior", i.ior, 1.0f);
        c.atLeast("iridescence.thicknessMin", i.thicknessMin, 0.0f);
        c.atLeast("iridescence.thicknessMax", i.thicknessMax, 0.0f);
        // thicknessMin > thicknessMax is allowed: the specification says
        // "Minimum MAY exceed Maximum".
    }
    if (d.transmission) c.inRange("transmission.factor", d.transmission->factor, 0.0f, 1.0f);
    if (d.volume) {
        const VolumeBlock& v = *d.volume;
        c.atLeast("volume.thickness", v.thickness, 0.0f);
        c.finite("volume.thickness", v.thickness);
        c.eachInRange("volume.attenuationColor", v.attenuationColor, 0.0f, 1.0f);
        if (!(v.attenuationDistance > 0.0f)) c.error("volume.attenuationDistance must be > 0 (or +inf)");
    }
    if (d.subsurface) {
        const SubsurfaceBlock& s = *d.subsurface;
        c.inRange("subsurface.weight", s.weight, 0.0f, 1.0f);
        // The draft KHR_materials_scatter does not bound the multiple-scattering
        // colour above (validationWarnings reports values over 1); it cannot be negative.
        c.eachAtLeast("subsurface.color", s.color, 0.0f);
        c.atLeast("subsurface.radius", s.radius, 0.0f);
        c.finite("subsurface.radius", s.radius);
        c.eachAtLeast("subsurface.radiusScale", s.radiusScale, 0.0f);
        if (!(s.anisotropy > -1.0f && s.anisotropy < 1.0f)) c.error("subsurface.anisotropy must be in (-1, 1)");
        if (s.draft) {
            c.inRange("scatter.scatterStrength", s.draft->scatterStrength, 0.0f, 1.0f);
            if (!(s.draft->scatterAnisotropy > -1.0f && s.draft->scatterAnisotropy < 1.0f))
                c.error("scatter.scatterAnisotropy must be in (-1, 1)");
        }
    }

    // Companion rules.  A volume has an effect behind transmission, behind volumetric
    // subsurface and also with a thin subsurface / diffuse-transmission block (the
    // diffuse_transmission RC text combines it with KHR_materials_volume).
    if (d.volume && d.volume->thickness > 0.0f) {
        const bool transmissive = d.transmission && d.transmission->factor > 0.0f;
        const bool subsurface   = d.subsurface && d.subsurface->weight > 0.0f;
        if (!transmissive && !subsurface)
            c.error("volume requires transmission or subsurface to have an effect");
    }
    if (d.subsurface && d.subsurface->weight > 0.0f && d.subsurface->mode == SubsurfaceMode::Volumetric &&
        !(d.volume && d.volume->thickness > 0.0f))
        c.error("volumetric subsurface requires a volume with thickness > 0");

    if (d.layers) {
        const LayerStack& stack = *d.layers;
        if (stack.layers.empty() || stack.layers.size() > MATERIAL_LAYER_MAX)
            c.error("layer stack needs 1 to " + std::to_string(MATERIAL_LAYER_MAX) + " layers, has " +
                    std::to_string(stack.layers.size()));
        if (!stack.mask.present()) c.error("layer stack needs a mask texture");
        std::array<bool, 4> used{};
        for (size_t i = 0; i < stack.layers.size(); ++i) {
            const MaterialLayer& l = stack.layers[i];
            const std::string prefix = "layers[" + std::to_string(i) + "]";
            const u8 ch = static_cast<u8>(l.maskChannel);
            if (ch > static_cast<u8>(TextureChannel::A)) {
                c.error(prefix + ".maskChannel must be R, G, B or A");
            } else if (used[ch]) {
                c.error(prefix + ".maskChannel is used by another layer");
            } else {
                used[ch] = true;
            }
            if (l.materialName.empty() && l.materialIndex == ~0u) c.error(prefix + " has no material");
            if (l.materialIndex != ~0u && materialCount != ~0u && l.materialIndex >= materialCount)
                c.error(prefix + ".materialIndex out of range");
            if (l.heightContrast && !(*l.heightContrast >= 0.0f && std::isfinite(*l.heightContrast)))
                c.error(prefix + ".heightContrast must be a finite value >= 0");
            if (!(l.uvScale > 0.0f && std::isfinite(l.uvScale))) c.error(prefix + ".uvScale must be a finite value > 0");
        }
    }
    return out;
}

std::vector<std::string> validationWarnings(const MaterialDescription& d) {
    std::vector<std::string> out;
    if (d.subsurface) {
        const auto& color = d.subsurface->color;
        for (size_t i = 0; i < 3; ++i)
            if (color[i] > 1.0f) {
                out.push_back("subsurface.color " + std::to_string(color[i]) +
                              " is above 1 (the draft KHR_materials_scatter does not bound it; energy may exceed the "
                              "single-layer albedo)");
                break;
            }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Texture slots
// ---------------------------------------------------------------------------

namespace {

template <typename Desc, typename Ref, typename Fn>
void visitTextures(Desc& d, Fn&& fn) {
    fn("baseColorTexture", d.base.baseColorTexture);
    fn("metallicRoughnessTexture", d.base.metallicRoughnessTexture);
    fn("normalTexture", d.base.normalTexture);
    fn("occlusionTexture", d.base.occlusionTexture);
    fn("emissiveTexture", d.base.emissiveTexture);
    if (d.specular) {
        fn("specularTexture", d.specular->specularTexture);
        fn("specularColorTexture", d.specular->specularColorTexture);
    }
    if (d.clearcoat) {
        fn("clearcoatTexture", d.clearcoat->factorTexture);
        fn("clearcoatRoughnessTexture", d.clearcoat->roughnessTexture);
        fn("clearcoatNormalTexture", d.clearcoat->normalTexture);
    }
    if (d.sheen) {
        fn("sheenColorTexture", d.sheen->colorTexture);
        fn("sheenRoughnessTexture", d.sheen->roughnessTexture);
    }
    if (d.anisotropy) fn("anisotropyTexture", d.anisotropy->texture);
    if (d.iridescence) {
        fn("iridescenceTexture", d.iridescence->factorTexture);
        fn("iridescenceThicknessTexture", d.iridescence->thicknessTexture);
    }
    if (d.transmission) fn("transmissionTexture", d.transmission->texture);
    if (d.volume) fn("thicknessTexture", d.volume->thicknessTexture);
    if (d.subsurface) {
        fn("subsurfaceWeightTexture", d.subsurface->weightTexture);
        fn("subsurfaceColorTexture", d.subsurface->colorTexture);
        fn("subsurfaceRadiusScaleTexture", d.subsurface->radiusScaleTexture);
    }
    if (d.layers) {
        fn("layerMaskTexture", d.layers->mask);
        static constexpr const char* kHeightSlots[MATERIAL_LAYER_MAX] = {"layerHeightTexture0", "layerHeightTexture1",
                                                                          "layerHeightTexture2", "layerHeightTexture3"};
        for (size_t i = 0; i < d.layers->layers.size() && i < MATERIAL_LAYER_MAX; ++i)
            fn(kHeightSlots[i], d.layers->layers[i].heightTexture);
    }
}

} // namespace

void forEachTexture(MaterialDescription& desc,
                    const std::function<void(const char*, MaterialTextureRef&)>& fn) {
    visitTextures<MaterialDescription, MaterialTextureRef>(desc, fn);
}

void forEachTexture(const MaterialDescription& desc,
                    const std::function<void(const char*, const MaterialTextureRef&)>& fn) {
    visitTextures<const MaterialDescription, const MaterialTextureRef>(desc, fn);
}

u32 textureTransformSlot(const char* slotName) {
    static constexpr struct { const char* name; u32 slot; } kSlots[] = {
        {"baseColorTexture", TT_SLOT_BASE_COLOR},         {"metallicRoughnessTexture", TT_SLOT_METAL_ROUGH},
        {"normalTexture", TT_SLOT_NORMAL},                {"occlusionTexture", TT_SLOT_OCCLUSION},
        {"emissiveTexture", TT_SLOT_EMISSIVE},            {"specularTexture", TT_SLOT_SPECULAR},
        {"specularColorTexture", TT_SLOT_SPECULAR_COLOR}, {"clearcoatTexture", TT_SLOT_COAT},
        {"clearcoatRoughnessTexture", TT_SLOT_COAT_ROUGHNESS}, {"clearcoatNormalTexture", TT_SLOT_COAT_NORMAL},
        {"sheenColorTexture", TT_SLOT_SHEEN_COLOR},       {"sheenRoughnessTexture", TT_SLOT_SHEEN_ROUGHNESS},
        {"anisotropyTexture", TT_SLOT_ANISOTROPY},        {"iridescenceTexture", TT_SLOT_IRIDESCENCE},
        {"iridescenceThicknessTexture", TT_SLOT_IRIDESCENCE_THICKNESS}, {"transmissionTexture", TT_SLOT_TRANSMISSION},
        {"thicknessTexture", TT_SLOT_VOLUME_THICKNESS},   {"subsurfaceWeightTexture", TT_SLOT_SUBSURFACE_WEIGHT},
        {"subsurfaceColorTexture", TT_SLOT_SUBSURFACE_COLOR}, {"subsurfaceRadiusScaleTexture", TT_SLOT_SUBSURFACE_RADIUS},
    };
    for (const auto& e : kSlots)
        if (std::string_view(e.name) == slotName) return e.slot;
    return ~0u;
}

TextureTransformAnalysis analyzeTextureTransforms(const MaterialDescription& desc) {
    TextureTransformAnalysis result;
    bool anyTransform = false, uniform = true, haveFirst = false, haveSet = false;
    UvTransform first;
    forEachTexture(desc, [&](const char* slot, const MaterialTextureRef& ref) {
        if (!ref.present()) return;
        const UvTransform t = ref.transform.value_or(UvTransform{});
        if (textureTransformSlot(slot) == ~0u) { // layer mask / height: tiled by the layer stack, no transform
            if (!t.isIdentity() || ref.texCoord != 0) {
                result.status = TextureTransformStatus::Unsupported;
                if (result.reason.empty()) result.reason = std::string(slot) + " carries a transform or a UV set (layer textures use the layer tiling)";
            }
            return;
        }
        if (!t.isIdentity()) anyTransform = true;
        if (!haveSet) {
            result.uvSet = ref.texCoord;
            haveSet      = true;
        } else if (ref.texCoord != result.uvSet) {
            result.status = TextureTransformStatus::Unsupported;
            result.reason = "the textures of this material use different UV sets (the engine has one UV stream per mesh)";
        }
        if (!haveFirst) {
            first     = t;
            haveFirst = true;
        } else if (!(t == first)) {
            uniform = false;
        }
    });
    if (result.status == TextureTransformStatus::Unsupported) return result;
    if (!anyTransform) return result;
    result.status = uniform ? TextureTransformStatus::Uniform : TextureTransformStatus::PerSlot;
    if (uniform) result.transform = first;
    return result;
}

std::array<float, 2> applyUvTransform(const UvTransform& t, float u, float v) {
    // scale, then rotate (counter-clockwise), then translate.
    const float sx = u * t.scale[0], sy = v * t.scale[1];
    const float c = std::cos(t.rotation), s = std::sin(t.rotation);
    return {c * sx - s * sy + t.offset[0], s * sx + c * sy + t.offset[1]};
}

} // namespace phosphor
