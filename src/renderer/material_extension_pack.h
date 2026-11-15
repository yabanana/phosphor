#pragma once

// F15.1 (ticket 05): MaterialDescription -> the GPU extension records.
//
// Portable (phosphor_core).  An extended material owns a group of
// MATERIAL_EXT_RECORDS (4) records; this packs every block of the description
// (active or not: a block's data stays valid while its feature bit is clear,
// so toggling never rewrites layout) into the frozen layouts of
// renderer/gpu_types.h.  Absent blocks pack the neutral parameter values of
// the glTF specifications.  Texture fields are the resolved bindless indices of
// the description (INVALID_TEXTURE_INDEX when absent or unresolved).

#include "renderer/gpu_types.h"
#include "renderer/material_description.h"

#include <array>
#include <cstring>
#include <type_traits>

namespace phosphor {

using MaterialExtGroup = std::array<GPUMaterialExtRecord, MATERIAL_EXT_RECORDS>;

/// The four records of `desc` (specular+ior/clearcoat/sheen, anisotropy/
/// iridescence/transmission/volume, subsurface, layer stack).
[[nodiscard]] MaterialExtGroup packMaterialExtensions(const MaterialDescription& desc);

/// F15.1 (ticket 22): the transform block of a material with per-slot KHR_texture_transform (status PerSlot): MATERIAL_TT_RECORDS
/// records, slot s at record s / 3.  Absent or identity slots keep a zero record entry and no mask bit.
using TextureTransformBlock = std::array<GPUMaterialExtRecord, MATERIAL_TT_RECORDS>;
[[nodiscard]] bool needsTextureTransformBlock(const MaterialDescription& desc);
[[nodiscard]] TextureTransformBlock packTextureTransforms(const MaterialDescription& desc);
/// uv' = M uv + o with the matrices of the KHR_texture_transform README (computed in double, stored as float).
[[nodiscard]] GPUTextureTransform makeGpuTextureTransform(const UvTransform& t);

/// Typed view of a record (copy; the layouts are scalars only).
template <typename T>
[[nodiscard]] T unpackExtRecord(const GPUMaterialExtRecord& r) {
    static_assert(sizeof(T) == sizeof(GPUMaterialExtRecord) && std::is_trivially_copyable_v<T>);
    T out;
    std::memcpy(&out, &r, sizeof(T));
    return out;
}

/// Feature bits of `desc` as they appear in GPUMaterial::flags (already
/// shifted by MATERIAL_FEATURE_SHIFT).
[[nodiscard]] inline u32 featureFlags(const MaterialDescription& desc) { return featureMask(desc) << MATERIAL_FEATURE_SHIFT; }

/// The feature mask carried by a GPUMaterial (unshifted, MaterialFeature bits).
[[nodiscard]] inline u32 materialFeatureBits(const GPUMaterial& m) { return (m.flags >> MATERIAL_FEATURE_SHIFT) & MATERIAL_FEATURES_ALL; }

} // namespace phosphor
