#pragma once

// F15.1 (ticket 04): the material description file.
//
// A versioned JSON serialisation of MaterialDescription.  Loadable standalone
// (`phosphor-material-description`) or as name-keyed overrides over materials
// already loaded (`phosphor-material-overrides`).  Texture references are paths
// relative to the file; the caller resolves them to bindless indices.
// Portable; JSON via the nlohmann json.hpp shipped with tinygltf.
//
// File layout (version 1):
//   {"schema": "phosphor-material-description", "version": 1,
//    "materials": [{"name": "...", "base": {...}, "clearcoat": {...}, ...}]}
//   {"schema": "phosphor-material-overrides", "version": 1,
//    "overrides": [{"name": "...", "clearcoat": {"factor": 1}, "sheen": null}]}
// Block keys: base, specular, clearcoat, sheen, anisotropy, iridescence,
// transmission, volume, subsurface, layers.  In overrides only the listed
// fields change; `null` removes a feature block.  Unknown keys, unknown
// material names and invalid values are errors.  A float written as the string
// "inf" is +infinity (volume.attenuationDistance).

#include "renderer/material_description.h"

#include <functional>
#include <string>
#include <vector>

namespace phosphor {

class GpuScene;

struct NamedMaterial {
    std::string         name;
    MaterialDescription description;
};

/// Serialise (pretty-printed).  Fails when a texture slot holds a bindless
/// index without a path (it cannot be written).
[[nodiscard]] bool saveMaterialDescriptions(const std::vector<NamedMaterial>& materials, std::string& json,
                                            std::string& error);

/// Parse and validate a description file.  On failure `out` is untouched.
[[nodiscard]] bool loadMaterialDescriptions(const std::string& json, std::vector<NamedMaterial>& out,
                                            std::string& error);

/// Apply an overrides file over `materials` (every entry with the override's
/// name changes).  Atomic: on failure `materials` is untouched.  `changed`
/// (optional) receives the indices of changed entries.
[[nodiscard]] bool applyMaterialOverrides(const std::string& json, std::vector<NamedMaterial>& materials,
                                          std::string& error, std::vector<size_t>* changed = nullptr);

/// File helpers (paths in the file are relative to its directory; the loaded
/// `path` fields are made relative-to-cwd by prefixing that directory).
[[nodiscard]] bool loadMaterialDescriptionFile(const std::string& file, std::vector<NamedMaterial>& out,
                                               std::string& error);
[[nodiscard]] bool saveMaterialDescriptionFile(const std::string& file, const std::vector<NamedMaterial>& materials,
                                               std::string& error);

/// Resolves a texture path to a bindless index.
using TexturePathResolver = std::function<u32(const std::string& path, bool sRGB)>;

/// Apply an overrides file to the named materials of a scene: texture paths
/// are resolved with `resolve`, the descriptions replaced and the GPU base
/// records refreshed (GpuScene::setMaterialDescription).  Unnamed materials
/// cannot be addressed.
[[nodiscard]] bool applyMaterialOverridesToScene(const std::string& json, GpuScene& scene,
                                                 const TexturePathResolver& resolve, std::string& error);

} // namespace phosphor
