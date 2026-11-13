#pragma once

#include "core/types.h"
#include "renderer/meshlet_builder.h"
#include "renderer/gpu_types.h"
#include "renderer/mesh_bounds.h"
#include "renderer/material_description.h"
#include "renderer/material_support.h"

#include <glm/glm.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace phosphor {

using MeshHandle = u32;
constexpr MeshHandle INVALID_MESH_HANDLE = ~0u;

// ---------------------------------------------------------------------------
// GpuScene -- API-agnostic scene geometry and per-frame scene data.
//
// Test benches and the glTF loader append meshes here.  The graphics backend
// (see src/platform/metal) mirrors the arrays into GPU buffers whenever the
// corresponding version counter changes.  Keeping this class free of any
// graphics API lets it build and be unit-tested on every platform.
// ---------------------------------------------------------------------------

class GpuScene {
public:
    GpuScene() = default;

    GpuScene(const GpuScene&) = delete;
    GpuScene& operator=(const GpuScene&) = delete;

    /// Append a mesh: converts to GPUVertex, builds meshlets, keeps an index
    /// list for the classic indexed path.  Returns a handle usable as
    /// GPUInstance::meshIndex.
    MeshHandle uploadMesh(const std::vector<glm::vec3>& positions,
                          const std::vector<glm::vec3>& normals,
                          const std::vector<glm::vec4>& tangents,
                          const std::vector<glm::vec2>& uvs,
                          const std::vector<u32>& indices);

    /// Replace the per-frame instance list.
    void updateInstances(const std::vector<GPUInstance>& instances);

    /// Replace the light list.
    void updateLights(const std::vector<GPULight>& lights);

    void updateSampledLights(std::vector<GPUSampledLight> lights) { sampledLights_=std::move(lights); }
    [[nodiscard]] const std::vector<GPUSampledLight>& sampledLights() const { return sampledLights_; }

    /// Replace the shared material library.
    void updateMaterials(const std::vector<GPUMaterial>& materials);

    /// Append one material to the shared library and return its index.
    /// Instances whose entity has no MaterialComponent index this library.
    u32 addMaterial(const GPUMaterial& material);

    /// F15.1: append one material together with its description, name and
    /// support report (kept in vectors parallel to the library).
    u32 addMaterial(const GPUMaterial& material, MaterialDescription description, std::string name,
                    MaterialSupportReport support);

    /// F15.1: replace the description of a material and refresh the GPU base
    /// record fields it describes: the parameters (the emissive factor is
    /// clamped to >= 0 on the GPU; the description keeps the authored value),
    /// the texture of every slot (a slot without a texture, e.g. one an
    /// override removed, binds the documented default registered with
    /// setBaseTextureDefaults, or INVALID_TEXTURE_INDEX = the shader fallback
    /// when none was registered; a slot whose path is not resolved yet keeps
    /// its current binding) and the double-sided flag (the other flags are
    /// kept).  The material's support entry is recomputed from the new
    /// description (its features are then sourced from the description file).
    ///
    /// Only valid BEFORE the scene store is built: once sealMaterialLibrary()
    /// has been called the call is an error (logged) and is rejected, because
    /// the persistent GPU scene would not see the edit.  Until F15 ticket 05
    /// routes material edits through SceneStore deltas, overrides can only be
    /// applied between loading a scene and the first scene sync.  Returns
    /// false when `index` is out of range or the library is sealed.
    bool setMaterialDescription(u32 index, const MaterialDescription& description);

    /// Bindless textures a material slot falls back to without a texture (the
    /// loader registers the texture manager's defaults; cleared by clear()).
    void setBaseTextureDefaults(u32 white, u32 metallicRoughness, u32 normal);

    /// Mark the material library as mirrored by the scene store: from now on
    /// setMaterialDescription is rejected.  Called by the engine after the
    /// store's first sync of a scene; clear() reopens the library.
    void sealMaterialLibrary() { materialLibrarySealed_ = true; }
    [[nodiscard]] bool materialLibrarySealed() const { return materialLibrarySealed_; }

    /// Description, name and support report of material `index` (parallel to
    /// materials(); default-constructed for materials added without them).
    [[nodiscard]] const std::vector<MaterialDescription>&  materialDescriptions() const { return materialDescriptions_; }
    [[nodiscard]] const std::vector<std::string>&          materialNames() const { return materialNames_; }
    [[nodiscard]] const std::vector<MaterialSupportReport>& materialSupport() const { return materialSupport_; }
    /// Index of the first material with this name.
    [[nodiscard]] std::optional<u32> findMaterial(const std::string& name) const;

    /// glTF `extensionsUsed` entries of loaded assets that are not understood.
    void addAssetUnknownExtensions(const std::vector<std::string>& names);
    [[nodiscard]] const std::vector<std::string>& assetUnknownExtensions() const { return assetUnknownExtensions_; }

    /// Kept for source compatibility with loaders: geometry is flushed by the
    /// backend when geometryVersion() changes, so this is a no-op.
    void flushMeshData() {}

    /// Drop all geometry and per-frame data (used when switching benches).
    void clear();

    /// F6.1: cook options of the meshlets of meshes uploaded from now on
    /// (must pass validateMeshletOptions).  Kept across clear().
    void setMeshletOptions(const MeshletBuildOptions& options) { meshletOptions_ = options; }
    [[nodiscard]] const MeshletBuildOptions& meshletOptions() const { return meshletOptions_; }

    // --- Accessors used by the backend -----------------------------------
    [[nodiscard]] const std::vector<GPUVertex>&     vertices()          const { return vertices_; }
    [[nodiscard]] const std::vector<u32>&           indices()           const { return indices_; }
    [[nodiscard]] const std::vector<Meshlet>&       meshlets()          const { return meshlets_; }
    [[nodiscard]] const std::vector<u32>&           meshletVertices()   const { return meshletVertices_; }
    [[nodiscard]] const std::vector<u8>&            meshletTriangles()  const { return meshletTriangles_; }
    [[nodiscard]] const std::vector<MeshletBounds>& meshletBounds()     const { return meshletBounds_; }
    [[nodiscard]] const std::vector<GPUMeshInfo>&   meshInfos()         const { return meshInfos_; }
    [[nodiscard]] const std::vector<MeshLocalBounds>& meshLocalBounds() const { return meshBounds_; }
    [[nodiscard]] const std::vector<GPUInstance>&   instances()         const { return instances_; }
    [[nodiscard]] const std::vector<GPUMaterial>&   materials()         const { return materials_; }
    [[nodiscard]] const std::vector<GPULight>&      lights()            const { return lights_; }

    /// Incremented whenever mesh geometry changes (uploadMesh / clear).
    [[nodiscard]] u64 geometryVersion() const { return geometryVersion_; }

    [[nodiscard]] u32 getMeshletTotalCount() const { return static_cast<u32>(meshlets_.size()); }
    [[nodiscard]] u32 getMeshCount() const { return static_cast<u32>(meshInfos_.size()); }

private:
    std::vector<MeshLocalBounds> meshBounds_;      // parallel to meshInfos_
    std::vector<GPUVertex>     vertices_;
    std::vector<u32>           indices_;          // mesh-local indices, offset by GPUMeshInfo
    std::vector<Meshlet>       meshlets_;
    std::vector<u32>           meshletVertices_;  // global vertex indices
    std::vector<u8>            meshletTriangles_;
    std::vector<MeshletBounds> meshletBounds_;
    std::vector<GPUMeshInfo>   meshInfos_;

    std::vector<GPUInstance>   instances_;
    std::vector<GPUMaterial>   materials_;
    std::vector<MaterialDescription>   materialDescriptions_; // parallel to materials_
    std::vector<std::string>           materialNames_;        // parallel to materials_
    std::vector<MaterialSupportReport> materialSupport_;      // parallel to materials_
    std::unordered_map<std::string, u32> materialByName_;     // first material of each name
    std::vector<std::string>           assetUnknownExtensions_;
    struct BaseTextureDefaults {
        u32 white = INVALID_TEXTURE_INDEX, metallicRoughness = INVALID_TEXTURE_INDEX, normal = INVALID_TEXTURE_INDEX;
    } baseTextureDefaults_;
    bool materialLibrarySealed_ = false;
    std::vector<GPULight>      lights_;
    std::vector<GPUSampledLight> sampledLights_;

    u64 geometryVersion_ = 0;
    MeshletBuildOptions meshletOptions_;
};

} // namespace phosphor
