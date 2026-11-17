#pragma once

// F15.2 (ticket 21): feature-class manifests (ADR-0003).
//
// A manifest records, per scene, which material feature classes the scene uses: a class is the feature mask of one
// material (bit n = MaterialFeature n, the value of the reserved function constant MATERIAL_FEATURES), `materials` how
// many live materials carry it, and `union` the OR of the scene's masks (SceneStore::featureUnion).  The app and the
// `class_manifest` tool export them from the F15 fixtures (and the chess scene and the Khronos corpus);
// tools/variant_gen reads them together with the declared list shaders/material_classes.def and writes the class
// table, the MSL constants and the archive plan.
//
// Header-only and portable on purpose: tools/variant_gen (a build-time host tool that cannot link phosphor_core)
// uses the very same parser and validator as the exporter, so a manifest the exporter writes is exactly what the
// generator accepts, and a malformed one is rejected by both.
//
// JSON schema, version 1 (every object is closed: an unknown or a missing key is an error):
//   { "schema": "phosphor.material-class-manifest", "version": 1, "source": <string>,
//     "scenes": [ { "name": <string, unique>, "family": <string>, "union": <u32>, "volumetric": <bool>,
//                   "classes": [ { "mask": <u32>, "materials": <u32 >= 1>, "lobes": [<lobe name>...] } ] } ] }
// Rules: masks are within MATERIAL_FEATURES_ALL, unique inside a scene and sorted ascending; `union` equals the OR
// of the class masks; `lobes` lists the names of the mask's bits in bit order (a redundant, human readable copy that
// must agree); `volumetric` needs the subsurface bit in the union; a scene has at least one class.

#include "renderer/gpu_types.h"

#include <json.hpp> // nlohmann/json, shipped with tinygltf

#include <algorithm>
#include <array>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace phosphor::matclass {

inline constexpr const char* kSchemaName = "phosphor.material-class-manifest";
inline constexpr u32 kSchemaVersion = 1;
inline constexpr u32 kFeatureCount = 9; // MaterialFeature values 0..8 (renderer/material_description.h)

inline constexpr std::array<const char*, kFeatureCount> kLobeNames = {
    "specular_ior", "clearcoat", "sheen", "anisotropy", "iridescence", "transmission", "volume", "subsurface", "layers"};

struct ClassUse {
    u32 mask = 0;
    u32 materials = 0;
    bool operator==(const ClassUse&) const = default;
};

struct SceneEntry {
    std::string name;   // "<family>:<scenario>", unique inside a manifest
    std::string family; // material | transmission | shadow | chess | corpus ...
    u32 unionMask = 0;
    bool volumetric = false; // a live material carries VOLUMETRIC subsurface (the resolve also sets SSS_OUTPUT)
    std::vector<ClassUse> classes; // ascending mask
    bool operator==(const SceneEntry&) const = default;
};

struct Manifest {
    std::string source; // free text: how the manifest was produced
    std::vector<SceneEntry> scenes;
    bool operator==(const Manifest&) const = default;
};

/// Lobe names of a mask in bit order.
[[nodiscard]] inline std::vector<std::string> lobeNames(u32 mask) {
    std::vector<std::string> out;
    for (u32 f = 0; f < kFeatureCount; ++f)
        if (mask & (1u << f)) out.emplace_back(kLobeNames[f]);
    return out;
}

/// Builds a scene entry from the feature masks of its live materials (one per material, any order; 0 = no feature).
[[nodiscard]] inline SceneEntry makeSceneEntry(std::string name, std::string family, const std::vector<u32>& materialMasks, bool volumetric) {
    SceneEntry e;
    e.name = std::move(name);
    e.family = std::move(family);
    e.volumetric = volumetric;
    std::vector<u32> sorted = materialMasks;
    std::sort(sorted.begin(), sorted.end());
    for (u32 m : sorted) {
        e.unionMask |= m;
        if (e.classes.empty() || e.classes.back().mask != m) e.classes.push_back({m, 0});
        ++e.classes.back().materials;
    }
    return e;
}

[[nodiscard]] inline nlohmann::json toJson(const Manifest& m) {
    using json = nlohmann::json;
    json scenes = json::array();
    for (const SceneEntry& s : m.scenes) {
        json classes = json::array();
        for (const ClassUse& c : s.classes) classes.push_back({{"mask", c.mask}, {"materials", c.materials}, {"lobes", lobeNames(c.mask)}});
        scenes.push_back({{"name", s.name}, {"family", s.family}, {"union", s.unionMask}, {"volumetric", s.volumetric}, {"classes", classes}});
    }
    return {{"schema", kSchemaName}, {"version", kSchemaVersion}, {"source", m.source}, {"scenes", scenes}};
}

/// Deterministic text of a manifest (sorted keys, 2-space indent, trailing newline).
[[nodiscard]] inline std::string serialize(const Manifest& m) { return toJson(m).dump(2) + "\n"; }

namespace detail {
[[nodiscard]] inline bool closedObject(const nlohmann::json& j, std::initializer_list<const char*> keys, const std::string& where, std::string& error) {
    if (!j.is_object()) { error = where + ": not an object"; return false; }
    std::set<std::string> want(keys.begin(), keys.end());
    for (auto it = j.begin(); it != j.end(); ++it)
        if (!want.count(it.key())) { error = where + ": unknown key '" + it.key() + "'"; return false; }
    for (const std::string& k : want)
        if (!j.contains(k)) { error = where + ": missing key '" + k + "'"; return false; }
    return true;
}
[[nodiscard]] inline bool unsignedValue(const nlohmann::json& j, const std::string& where, u32& out, std::string& error) {
    if (!j.is_number_unsigned() || j.get<unsigned long long>() > 0xFFFFFFFFull) { error = where + ": not an unsigned 32-bit integer"; return false; }
    out = j.get<u32>();
    return true;
}
} // namespace detail

/// Parses and validates a manifest; false with a message on any schema error (out is unspecified then).
[[nodiscard]] inline bool parseManifest(std::string_view text, Manifest& out, std::string& error) {
    using json = nlohmann::json;
    out = {};
    const json j = json::parse(text.begin(), text.end(), nullptr, false);
    if (j.is_discarded()) { error = "not valid JSON"; return false; }
    if (!detail::closedObject(j, {"schema", "version", "source", "scenes"}, "manifest", error)) return false;
    if (!j["schema"].is_string() || j["schema"].get<std::string>() != kSchemaName) { error = "manifest: wrong schema name"; return false; }
    u32 version = 0;
    if (!detail::unsignedValue(j["version"], "manifest.version", version, error)) return false;
    if (version != kSchemaVersion) { error = "manifest: unsupported version " + std::to_string(version) + " (this reader knows " + std::to_string(kSchemaVersion) + ")"; return false; }
    if (!j["source"].is_string()) { error = "manifest.source: not a string"; return false; }
    out.source = j["source"].get<std::string>();
    if (!j["scenes"].is_array() || j["scenes"].empty()) { error = "manifest.scenes: not a non-empty array"; return false; }
    std::set<std::string> names;
    for (size_t si = 0; si < j["scenes"].size(); ++si) {
        const json& s = j["scenes"][si];
        const std::string where = "scenes[" + std::to_string(si) + "]";
        if (!detail::closedObject(s, {"name", "family", "union", "volumetric", "classes"}, where, error)) return false;
        SceneEntry e;
        if (!s["name"].is_string() || s["name"].get<std::string>().empty()) { error = where + ".name: not a non-empty string"; return false; }
        e.name = s["name"].get<std::string>();
        if (!names.insert(e.name).second) { error = where + ": duplicate scene name '" + e.name + "'"; return false; }
        if (!s["family"].is_string() || s["family"].get<std::string>().empty()) { error = where + ".family: not a non-empty string"; return false; }
        e.family = s["family"].get<std::string>();
        if (!detail::unsignedValue(s["union"], where + ".union", e.unionMask, error)) return false;
        if (!s["volumetric"].is_boolean()) { error = where + ".volumetric: not a boolean"; return false; }
        e.volumetric = s["volumetric"].get<bool>();
        if (!s["classes"].is_array() || s["classes"].empty()) { error = where + ".classes: not a non-empty array"; return false; }
        u32 orMask = 0, previous = 0;
        for (size_t ci = 0; ci < s["classes"].size(); ++ci) {
            const json& c = s["classes"][ci];
            const std::string cw = where + ".classes[" + std::to_string(ci) + "]";
            if (!detail::closedObject(c, {"mask", "materials", "lobes"}, cw, error)) return false;
            ClassUse use;
            if (!detail::unsignedValue(c["mask"], cw + ".mask", use.mask, error)) return false;
            if (!detail::unsignedValue(c["materials"], cw + ".materials", use.materials, error)) return false;
            if (use.mask > MATERIAL_FEATURES_ALL) { error = cw + ".mask: bits outside MATERIAL_FEATURES_ALL"; return false; }
            if (use.materials == 0) { error = cw + ".materials: must be >= 1"; return false; }
            if (ci > 0 && use.mask <= previous) { error = cw + ".mask: classes must be unique and ascending"; return false; }
            previous = use.mask;
            if (!c["lobes"].is_array()) { error = cw + ".lobes: not an array"; return false; }
            std::vector<std::string> lobes;
            for (const json& l : c["lobes"]) {
                if (!l.is_string()) { error = cw + ".lobes: not strings"; return false; }
                lobes.push_back(l.get<std::string>());
            }
            if (lobes != lobeNames(use.mask)) { error = cw + ".lobes: do not match the mask"; return false; }
            orMask |= use.mask;
            e.classes.push_back(use);
        }
        if (orMask != e.unionMask) { error = where + ".union: is not the OR of the class masks"; return false; }
        if (e.volumetric && !(e.unionMask & MATERIAL_FEATURE_SUBSURFACE)) { error = where + ".volumetric: the union has no subsurface bit"; return false; }
        out.scenes.push_back(std::move(e));
    }
    return true;
}

} // namespace phosphor::matclass
