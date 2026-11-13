#pragma once

// F15.3 (ticket 04): the support report.
//
// Per material and feature: implemented / approximated (named approximation)
// / unsupported, plus the extensions of the material the loader does not
// understand.  Printed as a `MATERIALS` log line (per-material detail at debug
// level) and written into the benchmark report JSON.  Portable.

#include "core/types.h"
#include "renderer/material_description.h"

#include <string>
#include <vector>

namespace phosphor {

enum class SupportLevel : u8 { Implemented, Approximated, Unsupported };
[[nodiscard]] const char* supportLevelName(SupportLevel level);

/// Maturity of the specification a feature input comes from.
enum class InputLabel : u8 { Ratified, ReleaseCandidate, Draft, Phosphor };
[[nodiscard]] const char* inputLabelName(InputLabel label);

/// What the renderer currently does with one material feature.  Later tickets
/// flip entries of currentSupportTable().
struct FeatureSupportEntry {
    SupportLevel level         = SupportLevel::Unsupported;
    const char*  approximation = ""; // id, required when Approximated
};

struct MaterialSupportTable {
    FeatureSupportEntry features[MATERIAL_FEATURE_COUNT];
    FeatureSupportEntry base;               // glTF core metallic-roughness
    FeatureSupportEntry emissiveStrength;   // KHR_materials_emissive_strength (folded)
    FeatureSupportEntry textureTransformUniform; // baked into UV0 when uniform per material
};

/// The table of this build.
[[nodiscard]] const MaterialSupportTable& currentSupportTable();

/// What the source (glTF material or description file) said about a material,
/// independent of the description's parameters.
struct MaterialSourceInfo {
    std::string              name;
    std::vector<std::string> extensions;        // every extension name on the material
    std::vector<std::string> unknownExtensions; // those the loader does not parse
    bool                     emissiveStrengthUsed = false;
    TextureTransformStatus   transformStatus = TextureTransformStatus::None;
    std::vector<std::string> warnings;          // parse problems, validation messages
};

struct FeatureSupport {
    std::string  feature;       // "base", "emissive_strength", "texture_transform" or featureName()
    SupportLevel level = SupportLevel::Unsupported;
    std::string  approximation; // named approximation id (Approximated only)
    std::string  source;        // extension name(s) or "material-description-file"
    InputLabel   label = InputLabel::Ratified;
};

struct MaterialSupportReport {
    std::string                 name;
    std::vector<FeatureSupport> features; // base first; only features the material uses
    std::vector<std::string>    unknownExtensions;
    std::vector<std::string>    warnings;
};

/// Build the report of one material.
[[nodiscard]] MaterialSupportReport buildSupportReport(const MaterialDescription& desc,
                                                       const MaterialSourceInfo& info,
                                                       const MaterialSupportTable& table = currentSupportTable());

/// One line: `MATERIALS count=.. implemented=.. approximated=.. unsupported=..
/// features=[name:level(count) ...] unknown_extensions=[...]` -- counts are
/// (material, feature) pairs, base excluded.
[[nodiscard]] std::string formatSupportSummary(const std::vector<MaterialSupportReport>& materials,
                                               const std::vector<std::string>& assetUnknownExtensions);

/// Detail of one material (debug log level).
[[nodiscard]] std::string formatSupportDetail(const MaterialSupportReport& report);

/// JSON object for the benchmark report (`materials` section).
[[nodiscard]] std::string supportToJson(const std::vector<MaterialSupportReport>& materials,
                                        const std::vector<std::string>& assetUnknownExtensions);

} // namespace phosphor
