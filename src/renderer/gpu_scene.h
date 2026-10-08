#pragma once

#include "core/types.h"
#include "renderer/meshlet_builder.h"
#include "renderer/gpu_types.h"

#include <glm/glm.hpp>
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
    [[nodiscard]] const std::vector<GPUInstance>&   instances()         const { return instances_; }
    [[nodiscard]] const std::vector<GPUMaterial>&   materials()         const { return materials_; }
    [[nodiscard]] const std::vector<GPULight>&      lights()            const { return lights_; }

    /// Incremented whenever mesh geometry changes (uploadMesh / clear).
    [[nodiscard]] u64 geometryVersion() const { return geometryVersion_; }

    [[nodiscard]] u32 getMeshletTotalCount() const { return static_cast<u32>(meshlets_.size()); }
    [[nodiscard]] u32 getMeshCount() const { return static_cast<u32>(meshInfos_.size()); }

private:
    std::vector<GPUVertex>     vertices_;
    std::vector<u32>           indices_;          // mesh-local indices, offset by GPUMeshInfo
    std::vector<Meshlet>       meshlets_;
    std::vector<u32>           meshletVertices_;  // global vertex indices
    std::vector<u8>            meshletTriangles_;
    std::vector<MeshletBounds> meshletBounds_;
    std::vector<GPUMeshInfo>   meshInfos_;

    std::vector<GPUInstance>   instances_;
    std::vector<GPUMaterial>   materials_;
    std::vector<GPULight>      lights_;
    std::vector<GPUSampledLight> sampledLights_;

    u64 geometryVersion_ = 0;
    MeshletBuildOptions meshletOptions_;
};

} // namespace phosphor
