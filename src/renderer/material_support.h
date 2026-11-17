#pragma once

// F15.3 (ticket 04): the support report.
//
// Vocabulary (GLOSSARY.md): a material FEATURE is an optional part of the
// model (clearcoat, sheen, ...); an EXTENSION is its glTF file encoding.  The
// report therefore keeps three separate things per material: the status of the
// base model, the material features the material uses (implemented /
// approximated with a named approximation / unsupported) and the glTF
// ENCODINGS handled without a feature of their own (emissive_strength,
// texture_transform), plus the extensions the loader does not understand.
// Printed as a `MATERIALS` log line (per-material detail at debug level) and
// written into the benchmark report JSON (`phosphor.materials.v2`).  Portable.

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
    FeatureSupportEntry subsurfaceVolumetric; // the volumetric mode of the subsurface feature (features[Subsurface] is the thin mode)
    FeatureSupportEntry base;               // glTF core metallic-roughness
    FeatureSupportEntry emissiveStrength;   // KHR_materials_emissive_strength (folded)
    FeatureSupportEntry textureTransformUniform; // baked into UV0 when uniform per material
    FeatureSupportEntry textureTransformPerSlot; // F15.1 (ticket 22): per-slot transforms evaluated on the GPU
    FeatureSupportEntry punctualLights;     // KHR_lights_punctual (asset level, imported by the loader; spot falloff declared)
    FeatureSupportEntry unlit;              // KHR_materials_unlit (lowered onto the emission path, F15 closure)
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
    std::string              transformReason; // why the status is Unsupported
    std::vector<std::string> warnings;          // parse problems, validation messages
};

struct FeatureSupport {
    std::string  feature;       // "base", featureName(), or an encoding: "emissive_strength", "texture_transform"
    SupportLevel level = SupportLevel::Unsupported;
    std::string  approximation; // named approximation id (Approximated only)
    std::string  source;        // extension name(s) or "material-description-file"
    InputLabel   label = InputLabel::Ratified;
};

struct MaterialSupportReport {
    std::string                 name;
    FeatureSupport              base;      // the glTF core metallic-roughness model (empty feature = no report)
    std::vector<FeatureSupport> features;  // material features the material uses (featureName())
    std::vector<FeatureSupport> encodings; // glTF encodings handled without a feature (emissive_strength, texture_transform)
    std::vector<std::string>    unknownExtensions;
    std::vector<std::string>    warnings;

    /// False for materials added without a description (procedural benches).
    [[nodiscard]] bool valid() const { return !base.feature.empty(); }
};

/// Build the report of one material.
[[nodiscard]] MaterialSupportReport buildSupportReport(const MaterialDescription& desc,
                                                       const MaterialSourceInfo& info,
                                                       const MaterialSupportTable& table = currentSupportTable());

/// One line: `MATERIALS count=.. base=level(count).. implemented=..
/// approximated=.. unsupported=.. features=[name:level(count) ...]
/// encodings=[name:level(count) ...] unknown_extensions=[...]` -- the three
/// totals count (material, feature) pairs only; base and encodings are listed
/// apart.
[[nodiscard]] std::string formatSupportSummary(const std::vector<MaterialSupportReport>& materials,
                                               const std::vector<std::string>& assetUnknownExtensions);

/// Detail of one material (debug log level).
[[nodiscard]] std::string formatSupportDetail(const MaterialSupportReport& report);

/// JSON object for the benchmark report (`materials` section).
[[nodiscard]] std::string supportToJson(const std::vector<MaterialSupportReport>& materials,
                                        const std::vector<std::string>& assetUnknownExtensions);

} // namespace phosphor
