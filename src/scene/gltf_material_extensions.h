#pragma once

// F15.1/F15.3 (ticket 04): glTF material description parsing.
//
// tinygltf v2.9.7 keeps every KHR_materials_* extension as an unparsed
// `tinygltf::Value` tree (texture references inside extensions are plain Value
// objects, not TextureInfo).  This module converts those trees, plus the core
// material, into a MaterialDescription and the support-report input.
// Portable: no Metal, no GPU.

#include "core/types.h"
#include "renderer/material_description.h"
#include "renderer/material_support.h"

#include <glm/glm.hpp>

#include <functional>
#include <string>
#include <vector>

namespace tinygltf {
struct Material;
class Model;
} // namespace tinygltf

namespace phosphor {

/// Maps a glTF texture index to a bindless texture index (the loader loads and
/// caches the image; tests pass an identity function).  `sRGB` selects the
/// colour space the specification defines for the slot.
using GltfTextureResolver = std::function<u32(int gltfTexture, bool sRGB)>;

struct GltfMaterialParse {
    MaterialDescription description;
    MaterialSourceInfo  info;
};

/// Convert one glTF material.  Textures are resolved base slots first
/// (baseColor, metallicRoughness, normal, occlusion, emissive), then the
/// extension slots.  Out-of-range values are kept as authored (validate()
/// reports them); malformed extension JSON is skipped with a warning in
/// `info.warnings`.
[[nodiscard]] GltfMaterialParse parseGltfMaterial(const tinygltf::Material& material,
                                                  const GltfTextureResolver& resolve);

/// Extension names this loader understands (material-level KHR_materials_*
/// and KHR_texture_transform).
[[nodiscard]] bool isSupportedGltfExtension(const std::string& name);

/// glTF 2.0 section 3.2: an asset whose `extensionsRequired` lists an
/// extension the loader does not support should not be loaded.  Returns the
/// unsupported required extensions (empty = loadable).
[[nodiscard]] std::vector<std::string> unsupportedRequiredExtensions(const tinygltf::Model& model);

/// Optional (`extensionsUsed` but not required) extensions the loader does not
/// understand, in file order.
[[nodiscard]] std::vector<std::string> unknownOptionalExtensions(const tinygltf::Model& model);

/// Apply the transform to the UV0 stream in place (uv' = T * R * S * uv).
void bakeUvTransform(const UvTransform& transform, std::vector<glm::vec2>& uvs);

/// Image URI containment (decision: image URIs resolve only inside the asset
/// directory).  `data:` URIs are always accepted.  Rejects absolute paths,
/// URL schemes, `..` segments and symlink escapes (canonical prefix check).
/// On failure `error` explains why.
[[nodiscard]] bool isImageUriContained(const std::string& baseDir, const std::string& uri, std::string& error);

} // namespace phosphor
