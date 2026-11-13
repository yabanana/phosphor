#include "renderer/material_support.h"

#include <json.hpp> // nlohmann/json, shipped with tinygltf

#include <algorithm>
#include <array>
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

    r.base = entry("base", table.base, "glTF 2.0 core", InputLabel::Ratified);

    const u32 mask = featureMask(desc);
    for (u32 i = 0; i < MATERIAL_FEATURE_COUNT; ++i) {
        const auto f = static_cast<MaterialFeature>(i);
        if (!(mask & featureBit(f))) continue;
        std::string source;
        InputLabel  label = InputLabel::Ratified;
        describeSource(f, desc, info, source, label);
        r.features.push_back(entry(featureName(f), table.features[i], source, label));
    }
    // glTF encodings that carry no material feature of their own.
    if (info.emissiveStrengthUsed)
        r.encodings.push_back(entry("emissive_strength", table.emissiveStrength, "KHR_materials_emissive_strength",
                                    InputLabel::Ratified));
    switch (info.transformStatus) {
        case TextureTransformStatus::None: break;
        case TextureTransformStatus::Uniform:
            r.encodings.push_back(entry("texture_transform", table.textureTransformUniform, "KHR_texture_transform",
                                        InputLabel::Ratified));
            break;
        case TextureTransformStatus::NonUniform:
            r.encodings.push_back(entry("texture_transform", FeatureSupportEntry{}, "KHR_texture_transform",
                                        InputLabel::Ratified));
            r.warnings.push_back("texture_transform: textures of this material use different transforms or UV sets");
            break;
    }
    return r;
}

namespace {

using LevelCounts = std::array<u32, 3>;

std::string levelList(const std::map<std::string, LevelCounts>& perName) {
    std::string out;
    bool first = true;
    for (const auto& [name, c] : perName)
        for (size_t level = 0; level < 3; ++level) {
            if (!c[level]) continue;
            out += (first ? "" : " ") + name + ":" + supportLevelName(static_cast<SupportLevel>(level)) + "(" +
                   std::to_string(c[level]) + ")";
            first = false;
        }
    return out;
}

void addUnknown(std::vector<std::string>& unknown, const std::vector<std::string>& more) {
    for (const auto& e : more)
        if (!hasExtension(unknown, e.c_str())) unknown.push_back(e);
}

} // namespace

std::string formatSupportSummary(const std::vector<MaterialSupportReport>& materials,
                                 const std::vector<std::string>& assetUnknown) {
    LevelCounts totals{}, base{};
    std::map<std::string, LevelCounts> perFeature, perEncoding;
    std::vector<std::string> unknown = assetUnknown;
    for (const auto& m : materials) {
        if (m.valid()) ++base[static_cast<size_t>(m.base.level)];
        for (const auto& f : m.features) {
            ++totals[static_cast<size_t>(f.level)];
            ++perFeature[f.feature][static_cast<size_t>(f.level)];
        }
        for (const auto& e : m.encodings) ++perEncoding[e.feature][static_cast<size_t>(e.level)];
        addUnknown(unknown, m.unknownExtensions);
    }
    std::string baseText;
    for (size_t level = 0; level < 3; ++level)
        if (base[level])
            baseText += (baseText.empty() ? "" : " ") + std::string(supportLevelName(static_cast<SupportLevel>(level))) +
                        "(" + std::to_string(base[level]) + ")";
    return "MATERIALS count=" + std::to_string(materials.size()) + " base=[" + baseText + "]" +
           " implemented=" + std::to_string(totals[0]) + " approximated=" + std::to_string(totals[1]) +
           " unsupported=" + std::to_string(totals[2]) + " features=[" + levelList(perFeature) + "] encodings=[" +
           levelList(perEncoding) + "] unknown_extensions=[" + join(unknown) + "]";
}

std::string formatSupportDetail(const MaterialSupportReport& report) {
    auto one = [](const FeatureSupport& f) {
        std::string out = " " + f.feature + "=" + supportLevelName(f.level);
        if (!f.approximation.empty()) out += "(" + f.approximation + ")";
        if (f.label != InputLabel::Ratified) out += "[" + std::string(inputLabelName(f.label)) + "]";
        return out;
    };
    std::string out = "MATERIAL '" + report.name + "':" + one(report.base);
    if (!report.features.empty()) {
        out += " features:";
        for (const auto& f : report.features) out += one(f);
    }
    if (!report.encodings.empty()) {
        out += " encodings:";
        for (const auto& f : report.encodings) out += one(f);
    }
    if (!report.unknownExtensions.empty()) out += " unknown_extensions=" + join(report.unknownExtensions);
    for (const auto& w : report.warnings) out += " warning: " + w;
    return out;
}

std::string supportToJson(const std::vector<MaterialSupportReport>& materials,
                          const std::vector<std::string>& assetUnknown) {
    using Json = nlohmann::json;
    auto supportJson = [](const FeatureSupport& f) {
        Json e;
        e["feature"]       = f.feature;
        e["support"]       = supportLevelName(f.level);
        e["approximation"] = f.approximation;
        e["source"]        = f.source;
        e["label"]         = inputLabelName(f.label);
        return e;
    };
    auto countsJson = [](const LevelCounts& c) {
        return Json{{"implemented", c[0]}, {"approximated", c[1]}, {"unsupported", c[2]}};
    };
    Json root;
    root["schema"] = "phosphor.materials.v2";
    root["count"]  = materials.size();
    LevelCounts base{}, features{}, encodings{};
    Json list = Json::array();
    std::vector<std::string> unknown = assetUnknown;
    for (const auto& m : materials) {
        Json mj;
        mj["name"] = m.name;
        if (m.valid()) {
            ++base[static_cast<size_t>(m.base.level)];
            mj["base"] = supportJson(m.base);
        }
        Json fj = Json::array(), ej = Json::array();
        for (const auto& f : m.features) {
            ++features[static_cast<size_t>(f.level)];
            fj.push_back(supportJson(f));
        }
        for (const auto& f : m.encodings) {
            ++encodings[static_cast<size_t>(f.level)];
            ej.push_back(supportJson(f));
        }
        mj["features"]           = std::move(fj);
        mj["encodings"]          = std::move(ej);
        mj["unknown_extensions"] = m.unknownExtensions;
        mj["warnings"]           = m.warnings;
        addUnknown(unknown, m.unknownExtensions);
        list.push_back(std::move(mj));
    }
    root["base"]                     = countsJson(base);
    root["features"]                 = countsJson(features);
    root["encodings"]                = countsJson(encodings);
    root["unknown_extensions"]       = unknown;
    root["asset_unknown_extensions"] = assetUnknown;
    root["materials"]                = std::move(list);
    return root.dump(2);
}

} // namespace phosphor
