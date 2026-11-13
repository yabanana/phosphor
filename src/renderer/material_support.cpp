#include "renderer/material_support.h"

#include <json.hpp> // nlohmann/json, shipped with tinygltf

#include <algorithm>
#include <map>

namespace phosphor {

const char* supportLevelName(SupportLevel level) {
    switch (level) {
        case SupportLevel::Implemented:  return "implemented";
        case SupportLevel::Approximated: return "approximated";
        case SupportLevel::Unsupported:  return "unsupported";
    }
    return "unsupported";
}

const char* inputLabelName(InputLabel label) {
    switch (label) {
        case InputLabel::Ratified:         return "ratified";
        case InputLabel::ReleaseCandidate: return "release-candidate";
        case InputLabel::Draft:            return "draft";
        case InputLabel::Phosphor:         return "phosphor";
    }
    return "ratified";
}

const MaterialSupportTable& currentSupportTable() {
    // F15.1: only the base model, the folded emissive strength and the baked
    // uniform texture transform are rendered.  Each later ticket flips its
    // feature here (and names its approximation).
    static const MaterialSupportTable table = [] {
        MaterialSupportTable t;
        t.base.level                    = SupportLevel::Implemented;
        t.emissiveStrength.level        = SupportLevel::Implemented;
        t.textureTransformUniform.level = SupportLevel::Implemented;
        return t;
    }();
    return table;
}

namespace {

bool hasExtension(const std::vector<std::string>& list, const char* name) {
    return std::find(list.begin(), list.end(), name) != list.end();
}

std::string join(const std::vector<std::string>& v) {
    std::string out;
    for (size_t i = 0; i < v.size(); ++i) out += (i ? "," : "") + v[i];
    return out;
}

void describeSource(MaterialFeature f, const MaterialDescription& desc, const MaterialSourceInfo& info,
                    std::string& source, InputLabel& label) {
    std::vector<std::string> names;
    auto add = [&](const char* ext) {
        if (hasExtension(info.extensions, ext)) names.emplace_back(ext);
    };
    switch (f) {
        case MaterialFeature::SpecularIor:
            add("KHR_materials_specular");
            add("KHR_materials_ior");
            break;
        case MaterialFeature::Clearcoat:    add("KHR_materials_clearcoat"); break;
        case MaterialFeature::Sheen:        add("KHR_materials_sheen"); break;
        case MaterialFeature::Anisotropy:   add("KHR_materials_anisotropy"); break;
        case MaterialFeature::Iridescence:  add("KHR_materials_iridescence"); break;
        case MaterialFeature::Transmission: add("KHR_materials_transmission"); break;
        case MaterialFeature::Volume:       add("KHR_materials_volume"); break;
        case MaterialFeature::Subsurface:
            if (desc.subsurface) {
                switch (desc.subsurface->source) {
                    case SubsurfaceSource::KhrMaterialsScatterDraft:
                        source = "KHR_materials_scatter";
                        label  = InputLabel::Draft;
                        return;
                    case SubsurfaceSource::KhrMaterialsDiffuseTransmission:
                        source = "KHR_materials_diffuse_transmission";
                        label  = InputLabel::ReleaseCandidate;
                        return;
                    case SubsurfaceSource::DescriptionFile: break;
                }
            }
            break;
        case MaterialFeature::Layers: break;
    }
    if (names.empty()) {
        source = "material-description-file";
        label  = InputLabel::Phosphor;
    } else {
        source = join(names);
        label  = InputLabel::Ratified;
    }
}

FeatureSupport entry(const char* feature, const FeatureSupportEntry& e, std::string source, InputLabel label) {
    FeatureSupport f;
    f.feature       = feature;
    f.level         = e.level;
    f.approximation = e.level == SupportLevel::Approximated ? e.approximation : "";
    f.source        = std::move(source);
    f.label         = label;
    return f;
}

} // namespace

MaterialSupportReport buildSupportReport(const MaterialDescription& desc, const MaterialSourceInfo& info,
                                         const MaterialSupportTable& table) {
    MaterialSupportReport r;
    r.name              = info.name;
    r.unknownExtensions = info.unknownExtensions;
    r.warnings          = info.warnings;

    r.features.push_back(entry("base", table.base, "glTF 2.0 core", InputLabel::Ratified));

    const u32 mask = featureMask(desc);
    for (u32 i = 0; i < MATERIAL_FEATURE_COUNT; ++i) {
        const auto f = static_cast<MaterialFeature>(i);
        if (!(mask & featureBit(f))) continue;
        std::string source;
        InputLabel  label = InputLabel::Ratified;
        describeSource(f, desc, info, source, label);
        r.features.push_back(entry(featureName(f), table.features[i], source, label));
    }
    if (info.emissiveStrengthUsed)
        r.features.push_back(entry("emissive_strength", table.emissiveStrength, "KHR_materials_emissive_strength",
                                   InputLabel::Ratified));
    switch (info.transformStatus) {
        case TextureTransformStatus::None: break;
        case TextureTransformStatus::Uniform:
            r.features.push_back(entry("texture_transform", table.textureTransformUniform, "KHR_texture_transform",
                                       InputLabel::Ratified));
            break;
        case TextureTransformStatus::NonUniform:
            r.features.push_back(entry("texture_transform", FeatureSupportEntry{}, "KHR_texture_transform",
                                       InputLabel::Ratified));
            r.warnings.push_back("texture_transform: textures of this material use different transforms or UV sets");
            break;
    }
    return r;
}

std::string formatSupportSummary(const std::vector<MaterialSupportReport>& materials,
                                 const std::vector<std::string>& assetUnknown) {
    u32 counts[3] = {0, 0, 0};
    std::map<std::string, std::array<u32, 3>> perFeature;
    std::vector<std::string> unknown = assetUnknown;
    for (const auto& m : materials) {
        for (const auto& f : m.features) {
            if (f.feature == "base") continue;
            ++counts[static_cast<size_t>(f.level)];
            ++perFeature[f.feature][static_cast<size_t>(f.level)];
        }
        for (const auto& e : m.unknownExtensions)
            if (!hasExtension(unknown, e.c_str())) unknown.push_back(e);
    }
    std::string line = "MATERIALS count=" + std::to_string(materials.size()) +
                       " implemented=" + std::to_string(counts[0]) + " approximated=" + std::to_string(counts[1]) +
                       " unsupported=" + std::to_string(counts[2]) + " features=[";
    bool first = true;
    for (const auto& [name, c] : perFeature) {
        for (size_t level = 0; level < 3; ++level) {
            if (!c[level]) continue;
            line += (first ? "" : " ") + name + ":" + supportLevelName(static_cast<SupportLevel>(level)) + "(" +
                    std::to_string(c[level]) + ")";
            first = false;
        }
    }
    line += "] unknown_extensions=[" + join(unknown) + "]";
    return line;
}

std::string formatSupportDetail(const MaterialSupportReport& report) {
    std::string out = "MATERIAL '" + report.name + "':";
    for (const auto& f : report.features) {
        out += " " + f.feature + "=" + supportLevelName(f.level);
        if (!f.approximation.empty()) out += "(" + f.approximation + ")";
        if (f.label != InputLabel::Ratified) out += "[" + std::string(inputLabelName(f.label)) + "]";
    }
    if (!report.unknownExtensions.empty()) out += " unknown_extensions=" + join(report.unknownExtensions);
    for (const auto& w : report.warnings) out += " warning: " + w;
    return out;
}

std::string supportToJson(const std::vector<MaterialSupportReport>& materials,
                          const std::vector<std::string>& assetUnknown) {
    using Json = nlohmann::json;
    Json root;
    root["schema"] = "phosphor.materials.v1";
    root["count"]  = materials.size();
    u32 counts[3]  = {0, 0, 0};
    Json list      = Json::array();
    std::vector<std::string> unknown = assetUnknown;
    for (const auto& m : materials) {
        Json mj;
        mj["name"] = m.name;
        Json fj    = Json::array();
        for (const auto& f : m.features) {
            if (f.feature != "base") ++counts[static_cast<size_t>(f.level)];
            Json e;
            e["feature"]       = f.feature;
            e["support"]       = supportLevelName(f.level);
            e["approximation"] = f.approximation;
            e["source"]        = f.source;
            e["label"]         = inputLabelName(f.label);
            fj.push_back(std::move(e));
        }
        mj["features"]           = std::move(fj);
        mj["unknown_extensions"] = m.unknownExtensions;
        mj["warnings"]           = m.warnings;
        for (const auto& e : m.unknownExtensions)
            if (!hasExtension(unknown, e.c_str())) unknown.push_back(e);
        list.push_back(std::move(mj));
    }
    root["implemented"]        = counts[0];
    root["approximated"]       = counts[1];
    root["unsupported"]        = counts[2];
    root["unknown_extensions"] = unknown;
    root["asset_unknown_extensions"] = assetUnknown;
    root["materials"]          = std::move(list);
    return root.dump(2);
}

} // namespace phosphor
