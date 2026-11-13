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
/// colour space the specification defines for the slot.  Returning
/// INVALID_TEXTURE_INDEX means the texture cannot be resolved (index out of
/// range, no image): the slot stays empty and a warning is recorded.
using GltfTextureResolver = std::function<u32(int gltfTexture, bool sRGB)>;

struct GltfMaterialParse {
    MaterialDescription description;
    MaterialSourceInfo  info;
};

/// Convert one glTF material.  Textures are resolved base slots first
/// (baseColor, metallicRoughness, normal, occlusion, emissive), then the
/// extension slots.  Out-of-range values are kept as authored (validate()
/// reports them); malformed extension JSON, a negative emissiveStrength or
/// texCoord and unresolvable texture indices are noted in `info.warnings`
/// (a negative texCoord is read as UV set 0, an unresolvable texture as none).
[[nodiscard]] GltfMaterialParse parseGltfMaterial(const tinygltf::Material& material,
                                                  const GltfTextureResolver& resolve);

/// Extension names this loader understands (material-level KHR_materials_*
/// and KHR_texture_transform).
[[nodiscard]] bool isSupportedGltfExtension(const std::string& name);

/// One required extension the loader cannot honour, with the reason.
struct RequiredExtensionProblem {
    std::string name;
    std::string reason;
};

/// Does the renderer currently RENDER the feature this extension encodes
/// (support level implemented or approximated in `table`), as opposed to only
/// parsing it?  False for extensions the loader does not know.
[[nodiscard]] bool isExtensionRendered(const std::string& name,
                                       const MaterialSupportTable& table = currentSupportTable());

/// glTF 2.0 section 3.2: an asset whose `extensionsRequired` lists an
/// extension that cannot be honoured should not be loaded.  An extension is
/// honoured only when its feature is rendered (see isExtensionRendered), and a
/// required KHR_texture_transform additionally needs every material's
/// transforms to be uniform (bakeable into UV0).  Returns the problems (empty
/// = loadable).
[[nodiscard]] std::vector<RequiredExtensionProblem> unsupportedRequiredExtensions(
    const tinygltf::Model& model, const MaterialSupportTable& table = currentSupportTable());

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

/// The same checks for an already percent-decoded path relative to `baseDir`
/// (what tinygltf hands to its file-system callbacks).  Refuses empty, NUL,
/// absolute, scheme, `..` and symlink-escaping paths.
[[nodiscard]] bool isContainedRelativePath(const std::string& baseDir, const std::string& path, std::string& error);

} // namespace phosphor
