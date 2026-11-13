#include "scene/gltf_material_extensions.h"

#include <tiny_gltf.h> // declarations only; the implementation lives in gltf_loader.cpp

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <limits>

namespace phosphor {

namespace {

using tinygltf::Value;

constexpr const char* EXT_CLEARCOAT    = "KHR_materials_clearcoat";
constexpr const char* EXT_SHEEN        = "KHR_materials_sheen";
constexpr const char* EXT_ANISOTROPY   = "KHR_materials_anisotropy";
constexpr const char* EXT_TRANSMISSION = "KHR_materials_transmission";
constexpr const char* EXT_VOLUME       = "KHR_materials_volume";
constexpr const char* EXT_IOR          = "KHR_materials_ior";
constexpr const char* EXT_SPECULAR     = "KHR_materials_specular";
constexpr const char* EXT_IRIDESCENCE  = "KHR_materials_iridescence";
constexpr const char* EXT_EMISSIVE     = "KHR_materials_emissive_strength";
constexpr const char* EXT_DIFFUSE_TRANSMISSION = "KHR_materials_diffuse_transmission";
constexpr const char* EXT_SCATTER      = "KHR_materials_scatter"; // draft, PR #2579 head a6cd5b4
constexpr const char* EXT_TEXTURE_TRANSFORM = "KHR_texture_transform";

constexpr const char* kMaterialExtensions[] = {
    EXT_CLEARCOAT, EXT_SHEEN, EXT_ANISOTROPY, EXT_TRANSMISSION, EXT_VOLUME, EXT_IOR,
    EXT_SPECULAR, EXT_IRIDESCENCE, EXT_EMISSIVE, EXT_DIFFUSE_TRANSMISSION, EXT_SCATTER,
};

// Radius reported for a medium that does not attenuate (attenuationDistance
// infinite or attenuationColor 1): effectively infinite mean free path.
constexpr float MAX_MEAN_FREE_PATH = 1.0e6f;

/// Typed access to one extension object with warnings on malformed values.
class Reader {
public:
    Reader(const Value& obj, const char* ext, std::vector<std::string>& warnings,
           const GltfTextureResolver& resolve)
        : obj_(obj), ext_(ext), warnings_(warnings), resolve_(resolve) {}

    void number(const char* key, float& out) const {
        if (!obj_.Has(key)) return;
        const Value& v = obj_.Get(key);
        if (!v.IsNumber()) return warn(key, "expected a number");
        out = static_cast<float>(v.GetNumberAsDouble());
    }

    template <size_t N>
    void vec(const char* key, std::array<float, N>& out) const {
        if (!obj_.Has(key)) return;
        const Value& v = obj_.Get(key);
        if (!v.IsArray() || v.ArrayLen() != N) return warn(key, "expected an array of " + std::to_string(N));
        std::array<float, N> tmp{};
        for (size_t i = 0; i < N; ++i) {
            const Value& e = v.Get(static_cast<int>(i));
            if (!e.IsNumber()) return warn(key, "expected numbers");
            tmp[i] = static_cast<float>(e.GetNumberAsDouble());
        }
        out = tmp;
    }

    /// A textureInfo (plain Value object) with `index`, `texCoord`, optional
    /// KHR_texture_transform and, for normal maps, `scale`.
    void texture(const char* key, MaterialTextureRef& ref, bool sRGB, float* scale = nullptr) const {
        if (!obj_.Has(key)) return;
        const Value& v = obj_.Get(key);
        if (!v.IsObject() || !v.Has("index") || !v.Get("index").IsNumber())
            return warn(key, "expected a textureInfo with an index");
        const int index = v.Get("index").GetNumberAsInt();
        if (index < 0) return warn(key, "negative texture index");
        ref.colorSpace = sRGB ? TextureColorSpace::SRGB : TextureColorSpace::Linear;
        ref.index      = resolve_(index, sRGB);
        ref.texCoord   = v.Has("texCoord") && v.Get("texCoord").IsNumber()
                             ? static_cast<u32>(std::max(0, v.Get("texCoord").GetNumberAsInt()))
                             : 0u;
        if (v.Has("extensions") && v.Get("extensions").IsObject() &&
            v.Get("extensions").Has(EXT_TEXTURE_TRANSFORM))
            applyTransform(v.Get("extensions").Get(EXT_TEXTURE_TRANSFORM), ref);
        if (scale && v.Has("scale") && v.Get("scale").IsNumber())
            *scale = static_cast<float>(v.Get("scale").GetNumberAsDouble());
    }

    static void applyTransform(const Value& t, MaterialTextureRef& ref) {
        if (!t.IsObject()) return;
        UvTransform xf;
        auto pair = [&](const char* key, std::array<float, 2>& out) {
            if (t.Has(key) && t.Get(key).IsArray() && t.Get(key).ArrayLen() == 2 &&
                t.Get(key).Get(0).IsNumber() && t.Get(key).Get(1).IsNumber())
                out = {static_cast<float>(t.Get(key).Get(0).GetNumberAsDouble()),
                       static_cast<float>(t.Get(key).Get(1).GetNumberAsDouble())};
        };
        pair("offset", xf.offset);
        pair("scale", xf.scale);
        if (t.Has("rotation") && t.Get("rotation").IsNumber())
            xf.rotation = static_cast<float>(t.Get("rotation").GetNumberAsDouble());
        if (!xf.isIdentity()) ref.transform = xf;
        // The extension's texCoord, when present, overrides the textureInfo's.
        if (t.Has("texCoord") && t.Get("texCoord").IsNumber())
            ref.texCoord = static_cast<u32>(std::max(0, t.Get("texCoord").GetNumberAsInt()));
    }

private:
    void warn(const char* key, const std::string& what) const {
        warnings_.push_back(std::string(ext_) + "." + key + ": " + what);
    }
    const Value&                obj_;
    const char*                 ext_;
    std::vector<std::string>&   warnings_;
    const GltfTextureResolver&  resolve_;
};

template <typename Info>
void coreTexture(const Info& info, MaterialTextureRef& ref, bool sRGB, const GltfTextureResolver& resolve) {
    if (info.index < 0) return;
    ref.colorSpace = sRGB ? TextureColorSpace::SRGB : TextureColorSpace::Linear;
    ref.index      = resolve(info.index, sRGB);
    ref.texCoord   = static_cast<u32>(std::max(0, info.texCoord));
    const auto it  = info.extensions.find(EXT_TEXTURE_TRANSFORM);
    if (it != info.extensions.end()) Reader::applyTransform(it->second, ref);
}

float meanFreePathFor(float attenuationDistance, float attenuationColor) {
    // sigma_t = -ln(c) / d (KHR_materials_volume); mean free path = 1 / sigma_t.
    if (std::isinf(attenuationDistance) || attenuationColor >= 1.0f) return MAX_MEAN_FREE_PATH;
    if (attenuationColor <= 0.0f) return 0.0f;
    return std::min(MAX_MEAN_FREE_PATH, attenuationDistance / -std::log(attenuationColor));
}

} // namespace

GltfMaterialParse parseGltfMaterial(const tinygltf::Material& material, const GltfTextureResolver& resolve) {
    GltfMaterialParse out;
    MaterialDescription& d = out.description;
    MaterialSourceInfo& info = out.info;
    info.name = material.name;

    // --- core (the same slot order the loader always used) -----------------
    const auto& pbr = material.pbrMetallicRoughness;
    BaseMaterial& b = d.base;
    for (size_t i = 0; i < 4 && i < pbr.baseColorFactor.size(); ++i)
        b.baseColor[i] = static_cast<float>(pbr.baseColorFactor[i]);
    b.metallic  = static_cast<float>(pbr.metallicFactor);
    b.roughness = static_cast<float>(pbr.roughnessFactor);
    coreTexture(pbr.baseColorTexture, b.baseColorTexture, true, resolve);
    coreTexture(pbr.metallicRoughnessTexture, b.metallicRoughnessTexture, false, resolve);
    coreTexture(material.normalTexture, b.normalTexture, false, resolve);
    if (material.normalTexture.index >= 0) b.normalScale = static_cast<float>(material.normalTexture.scale);
    coreTexture(material.occlusionTexture, b.occlusionTexture, false, resolve);
    if (material.occlusionTexture.index >= 0)
        b.occlusionStrength = static_cast<float>(material.occlusionTexture.strength);
    coreTexture(material.emissiveTexture, b.emissiveTexture, true, resolve);
    for (size_t i = 0; i < 3 && i < material.emissiveFactor.size(); ++i)
        b.emissive[i] = static_cast<float>(material.emissiveFactor[i]);
    b.alphaCutoff = material.alphaMode == "MASK" ? static_cast<float>(material.alphaCutoff) : 0.0f;
    b.doubleSided = material.doubleSided;

    // --- extensions ----------------------------------------------------------
    for (const auto& [name, value] : material.extensions) {
        info.extensions.push_back(name);
        const bool known = std::find_if(std::begin(kMaterialExtensions), std::end(kMaterialExtensions),
                                        [&](const char* k) { return name == k; }) != std::end(kMaterialExtensions);
        if (!known) info.unknownExtensions.push_back(name);
    }
    auto find = [&](const char* name) -> const Value* {
        const auto it = material.extensions.find(name);
        return it == material.extensions.end() ? nullptr : &it->second;
    };
    auto reader = [&](const Value& v, const char* ext) { return Reader(v, ext, info.warnings, resolve); };
    auto objectOrWarn = [&](const Value* v, const char* ext) {
        if (v && !v->IsObject()) {
            info.warnings.push_back(std::string(ext) + ": expected an object");
            return false;
        }
        return v != nullptr;
    };

    if (const Value* v = find(EXT_EMISSIVE); objectOrWarn(v, EXT_EMISSIVE)) {
        float strength = 1.0f;
        reader(*v, EXT_EMISSIVE).number("emissiveStrength", strength);
        for (float& e : b.emissive) e *= strength;
        info.emissiveStrengthUsed = true;
    }

    const Value* ior      = find(EXT_IOR);
    const Value* specular = find(EXT_SPECULAR);
    const bool hasIor = objectOrWarn(ior, EXT_IOR);
    const bool hasSpecular = objectOrWarn(specular, EXT_SPECULAR);
    if (hasIor || hasSpecular) {
        SpecularBlock s;
        if (hasIor) reader(*ior, EXT_IOR).number("ior", s.ior);
        if (hasSpecular) {
            const Reader r = reader(*specular, EXT_SPECULAR);
            r.number("specularFactor", s.specularFactor);
            r.vec("specularColorFactor", s.specularColor);
            r.texture("specularTexture", s.specularTexture, false);
            r.texture("specularColorTexture", s.specularColorTexture, true);
        }
        d.specular = s;
    }

    if (const Value* v = find(EXT_CLEARCOAT); objectOrWarn(v, EXT_CLEARCOAT)) {
        ClearcoatBlock c;
        const Reader r = reader(*v, EXT_CLEARCOAT);
        r.number("clearcoatFactor", c.factor);
        r.number("clearcoatRoughnessFactor", c.roughness);
        r.texture("clearcoatTexture", c.factorTexture, false);
        r.texture("clearcoatRoughnessTexture", c.roughnessTexture, false);
        r.texture("clearcoatNormalTexture", c.normalTexture, false, &c.normalScale);
        d.clearcoat = c;
    }

    if (const Value* v = find(EXT_SHEEN); objectOrWarn(v, EXT_SHEEN)) {
        SheenBlock s;
        const Reader r = reader(*v, EXT_SHEEN);
        r.vec("sheenColorFactor", s.color);
        r.number("sheenRoughnessFactor", s.roughness);
        r.texture("sheenColorTexture", s.colorTexture, true);
        r.texture("sheenRoughnessTexture", s.roughnessTexture, false);
        d.sheen = s;
    }

    if (const Value* v = find(EXT_ANISOTROPY); objectOrWarn(v, EXT_ANISOTROPY)) {
        AnisotropyBlock a;
        const Reader r = reader(*v, EXT_ANISOTROPY);
        r.number("anisotropyStrength", a.strength);
        r.number("anisotropyRotation", a.rotation);
        r.texture("anisotropyTexture", a.texture, false);
        d.anisotropy = a;
    }

    if (const Value* v = find(EXT_IRIDESCENCE); objectOrWarn(v, EXT_IRIDESCENCE)) {
        IridescenceBlock i;
        const Reader r = reader(*v, EXT_IRIDESCENCE);
        r.number("iridescenceFactor", i.factor);
        r.number("iridescenceIor", i.ior);
        r.number("iridescenceThicknessMinimum", i.thicknessMin);
        r.number("iridescenceThicknessMaximum", i.thicknessMax);
        r.texture("iridescenceTexture", i.factorTexture, false);
        r.texture("iridescenceThicknessTexture", i.thicknessTexture, false);
        d.iridescence = i;
    }

    if (const Value* v = find(EXT_TRANSMISSION); objectOrWarn(v, EXT_TRANSMISSION)) {
        TransmissionBlock t;
        const Reader r = reader(*v, EXT_TRANSMISSION);
        r.number("transmissionFactor", t.factor);
        r.texture("transmissionTexture", t.texture, false);
        d.transmission = t;
    }

    if (const Value* v = find(EXT_VOLUME); objectOrWarn(v, EXT_VOLUME)) {
        VolumeBlock vb;
        const Reader r = reader(*v, EXT_VOLUME);
        r.number("thicknessFactor", vb.thickness);
        r.vec("attenuationColor", vb.attenuationColor);
        r.number("attenuationDistance", vb.attenuationDistance);
        r.texture("thicknessTexture", vb.thicknessTexture, false);
        d.volume = vb;
    }

    // Subsurface: the draft scatter extension wins over the RC diffuse
    // transmission (the draft excludes it and replaces it).
    const Value* scatter = find(EXT_SCATTER);
    const Value* diffuseTransmission = find(EXT_DIFFUSE_TRANSMISSION);
    if (objectOrWarn(scatter, EXT_SCATTER)) {
        if (diffuseTransmission)
            info.warnings.push_back(std::string(EXT_SCATTER) + " excludes " + EXT_DIFFUSE_TRANSMISSION +
                                    "; the draft extension is used");
        if (!d.transmission) {
            // Draft (PR 2579 at a6cd5b4): "no effect without KHR_materials_transmission".
            info.warnings.push_back(std::string(EXT_SCATTER) + " has no effect without KHR_materials_transmission; "
                                                               "no subsurface block");
        } else {
            SubsurfaceBlock s;
            ScatterDraftParams raw;
            const Reader r = reader(*scatter, EXT_SCATTER);
            r.number("scatterStrengthFactor", raw.scatterStrength);
            r.vec("multiscatterColorFactor", raw.multiscatterColor);
            r.number("scatterAnisotropy", raw.scatterAnisotropy);
            r.texture("scatterStrengthTexture", s.weightTexture, false);
            r.texture("multiscatterColorTexture", s.colorTexture, true);
            s.source     = SubsurfaceSource::KhrMaterialsScatterDraft;
            s.draft      = raw; // the raw draft values are kept next to the mapping
            s.anisotropy = raw.scatterAnisotropy;
            const float transmission = d.transmission->factor;
            // Thin-walled without a volume (or thickness 0), volumetric otherwise.
            const bool volumetric = d.volume && d.volume->thickness > 0.0f;
            s.mode = volumetric ? SubsurfaceMode::Volumetric : SubsurfaceMode::Thin;
            if (!volumetric) {
                // base = mix(diffuse, mix(specular_btdf * baseColor, scatter_bsdf(multiscatterColor, g),
                //                         scatterStrength), transmission): the scatter lobe carries
                // transmission * scatterStrength of the base, with the colour as authored.
                s.weight = transmission * raw.scatterStrength;
                s.color  = raw.multiscatterColor;
            } else {
                // Volumetric: the surface BSDF is unchanged (transmission); the medium scatters with
                // rho_ms* = scatterStrength * multiscatterColor per channel, mean free path from the volume.
                s.weight = transmission;
                for (size_t c = 0; c < 3; ++c) s.color[c] = raw.scatterStrength * raw.multiscatterColor[c];
                s.radius = 1.0f;
                for (size_t c = 0; c < 3; ++c)
                    s.radiusScale[c] =
                        meanFreePathFor(d.volume->attenuationDistance, d.volume->attenuationColor[c]);
            }
            d.subsurface = s;
        }
    } else if (objectOrWarn(diffuseTransmission, EXT_DIFFUSE_TRANSMISSION)) {
        SubsurfaceBlock s;
        s.mode   = SubsurfaceMode::Thin;
        s.source = SubsurfaceSource::KhrMaterialsDiffuseTransmission;
        s.color  = {1.0f, 1.0f, 1.0f}; // diffuseTransmissionColorFactor default
        const Reader r = reader(*diffuseTransmission, EXT_DIFFUSE_TRANSMISSION);
        r.number("diffuseTransmissionFactor", s.weight);
        r.vec("diffuseTransmissionColorFactor", s.color);
        r.texture("diffuseTransmissionTexture", s.weightTexture, false);
        r.texture("diffuseTransmissionColorTexture", s.colorTexture, true);
        d.subsurface = s;
    }

    info.transformStatus = analyzeTextureTransforms(d).status;

    for (const std::string& problem : validate(d)) info.warnings.push_back("invalid value: " + problem);
    for (const std::string& note : validationWarnings(d)) info.warnings.push_back(note);
    return out;
}

bool isSupportedGltfExtension(const std::string& name) {
    if (name == EXT_TEXTURE_TRANSFORM) return true;
    return std::find_if(std::begin(kMaterialExtensions), std::end(kMaterialExtensions),
                        [&](const char* k) { return name == k; }) != std::end(kMaterialExtensions);
}

bool isExtensionRendered(const std::string& name, const MaterialSupportTable& table) {
    auto rendered = [](const FeatureSupportEntry& e) { return e.level != SupportLevel::Unsupported; };
    auto feature = [&](MaterialFeature f) { return rendered(table.features[static_cast<size_t>(f)]); };
    if (name == EXT_CLEARCOAT) return feature(MaterialFeature::Clearcoat);
    if (name == EXT_SHEEN) return feature(MaterialFeature::Sheen);
    if (name == EXT_ANISOTROPY) return feature(MaterialFeature::Anisotropy);
    if (name == EXT_TRANSMISSION) return feature(MaterialFeature::Transmission);
    if (name == EXT_VOLUME) return feature(MaterialFeature::Volume);
    if (name == EXT_IOR || name == EXT_SPECULAR) return feature(MaterialFeature::SpecularIor);
    if (name == EXT_IRIDESCENCE) return feature(MaterialFeature::Iridescence);
    if (name == EXT_DIFFUSE_TRANSMISSION || name == EXT_SCATTER) return feature(MaterialFeature::Subsurface);
    if (name == EXT_EMISSIVE) return rendered(table.emissiveStrength);
    if (name == EXT_TEXTURE_TRANSFORM) return rendered(table.textureTransformUniform);
    return false;
}

std::vector<RequiredExtensionProblem> unsupportedRequiredExtensions(const tinygltf::Model& model,
                                                                    const MaterialSupportTable& table) {
    std::vector<RequiredExtensionProblem> out;
    for (const std::string& name : model.extensionsRequired) {
        if (!isSupportedGltfExtension(name)) {
            out.push_back({name, "not understood by this loader"});
        } else if (!isExtensionRendered(name, table)) {
            out.push_back({name, "parsed, but its feature is not rendered yet"});
        } else if (name == EXT_TEXTURE_TRANSFORM) {
            // Only uniform transforms are baked into UV0.
            for (size_t i = 0; i < model.materials.size(); ++i) {
                const auto parsed = parseGltfMaterial(model.materials[i], [](int index, bool) { return static_cast<u32>(index); });
                if (parsed.info.transformStatus == TextureTransformStatus::NonUniform) {
                    out.push_back({name, "material " + std::to_string(i) + " ('" + model.materials[i].name +
                                             "') has a non-uniform transform"});
                    break;
                }
            }
        }
    }
    return out;
}

std::vector<std::string> unknownOptionalExtensions(const tinygltf::Model& model) {
    std::vector<std::string> out;
    for (const std::string& name : model.extensionsUsed) {
        if (isSupportedGltfExtension(name)) continue;
        if (std::find(model.extensionsRequired.begin(), model.extensionsRequired.end(), name) !=
            model.extensionsRequired.end())
            continue;
        if (std::find(out.begin(), out.end(), name) == out.end()) out.push_back(name);
    }
    return out;
}

void bakeUvTransform(const UvTransform& transform, std::vector<glm::vec2>& uvs) {
    for (glm::vec2& uv : uvs) {
        const auto r = applyUvTransform(transform, uv.x, uv.y);
        uv = glm::vec2{r[0], r[1]};
    }
}

bool isContainedRelativePath(const std::string& baseDir, const std::string& path, std::string& error) {
    namespace fs = std::filesystem;
    if (path.empty()) {
        error = "empty URI";
        return false;
    }
    if (path.find('\0') != std::string::npos) {
        error = "URI '" + path + "' contains a NUL byte";
        return false;
    }
    if (path[0] == '/' || path[0] == '\\' || (path.size() >= 2 && path[1] == ':') ||
        path.find("://") != std::string::npos || path.rfind("file:", 0) == 0) {
        error = "URI '" + path + "' is absolute or has a scheme";
        return false;
    }
    // ".." segments, on both separators.
    size_t start = 0;
    while (start <= path.size()) {
        size_t end = path.find_first_of("/\\", start);
        if (end == std::string::npos) end = path.size();
        if (path.compare(start, end - start, "..") == 0) {
            error = "URI '" + path + "' contains a '..' segment";
            return false;
        }
        start = end + 1;
    }

    std::error_code ec;
    const fs::path base = fs::weakly_canonical(baseDir.empty() ? fs::path(".") : fs::path(baseDir), ec);
    if (ec) {
        error = "cannot resolve asset directory '" + baseDir + "'";
        return false;
    }
    const fs::path resolved = fs::weakly_canonical(base / path, ec);
    if (ec) {
        error = "cannot resolve URI '" + path + "'";
        return false;
    }
    // Component-wise prefix comparison of the canonical (symlink-resolved) paths.
    auto b = base.begin();
    auto r = resolved.begin();
    for (; b != base.end(); ++b, ++r) {
        if (b->empty()) continue; // trailing separator
        if (r == resolved.end() || *r != *b) {
            error = "URI '" + path + "' resolves outside the asset directory";
            return false;
        }
    }
    return true;
}

bool isImageUriContained(const std::string& baseDir, const std::string& uri, std::string& error) {
    if (uri.empty()) {
        error = "empty image URI";
        return false;
    }
    if (uri.rfind("data:", 0) == 0) return true;

    // Percent-decode (glTF URIs are URI-encoded paths).
    std::string path;
    for (size_t i = 0; i < uri.size(); ++i) {
        if (uri[i] == '%' && i + 2 < uri.size() && std::isxdigit(static_cast<unsigned char>(uri[i + 1])) &&
            std::isxdigit(static_cast<unsigned char>(uri[i + 2]))) {
            path += static_cast<char>(std::stoi(uri.substr(i + 1, 2), nullptr, 16));
            i += 2;
        } else {
            path += uri[i];
        }
    }
    return isContainedRelativePath(baseDir, path, error);
}

} // namespace phosphor
