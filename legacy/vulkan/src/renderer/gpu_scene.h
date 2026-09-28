#pragma once

#include "core/types.h"
#include "rhi/vk_common.h"
#include "rhi/vk_allocator.h"
#include "renderer/meshlet_builder.h"

#include <glm/glm.hpp>
#include <vector>

namespace phosphor {

class VulkanDevice;
class CommandManager;

// ---------------------------------------------------------------------------
// GPU-side structs (must match shader types.glsl layout)
// ---------------------------------------------------------------------------

struct GPUVertex {
    float px, py, pz;     // position
    float nx, ny, nz;     // normal
    float tx, ty, tz, tw; // tangent + handedness
    float u, v;           // UV
};

struct GPUInstance {
    float modelMatrix[16]; // mat4 column-major
    u32 meshIndex;
    u32 materialIndex;
    u32 flags;
    u32 pad;
};

struct GPUMaterial {
    float baseColor[4];        // rgba
    float metallic;
    float roughness;
    float normalScale;
    float occlusionStrength;
    u32 baseColorTex;
    u32 normalTex;
    u32 metallicRoughnessTex;
    u32 occlusionTex;
    u32 emissiveTex;
    float emissive[3];
    float alphaCutoff;
};

struct GPUMeshInfo {
    u32 meshletCount;
    u32 meshletOffset;   // into global meshlet buffer
    u32 vertexOffset;    // into global vertex buffer
    u32 indexOffset;
    float boundingSphere[4]; // center xyz + radius
};

struct GPULight {
    u32 type;          // 0=directional, 1=point, 2=spot
    float position[3];
    float direction[3];
    float color[3];
    float intensity;
    float range;
    float innerCone;
    float outerCone;
    u32 shadowMapIndex;
};

struct SceneGlobals {
    u64 vertexBufferAddress;
    u64 meshletBufferAddress;
    u64 meshletVertexAddress;
    u64 meshletTriangleAddress;
    u64 instanceBufferAddress;
    u64 materialBufferAddress;
    u64 meshInfoBufferAddress;
    u64 lightBufferAddress;
    u64 meshletBoundsAddress;
    u32 instanceCount;
    u32 lightCount;
    u32 meshletTotalCount;
    u32 pad;
};

using MeshHandle = u32;
constexpr MeshHandle INVALID_MESH_HANDLE = ~0u;

// ---------------------------------------------------------------------------
// GpuScene — GPU-side scene representation
// ---------------------------------------------------------------------------

class GpuScene {
public:
    GpuScene(VulkanDevice& device, GpuAllocator& allocator, CommandManager& commands);
    ~GpuScene();

    GpuScene(const GpuScene&) = delete;
    GpuScene& operator=(const GpuScene&) = delete;

    /// Upload a mesh: converts to GPUVertex format, builds meshlets, uploads to GPU.
    /// Returns a handle to reference this mesh in GPUInstance::meshIndex.
    MeshHandle uploadMesh(const std::vector<glm::vec3>& positions,
                          const std::vector<glm::vec3>& normals,
                          const std::vector<glm::vec4>& tangents,
                          const std::vector<glm::vec2>& uvs,
                          const std::vector<u32>& indices);

    /// Upload instance data (call each frame with current transforms/materials).
    void updateInstances(const std::vector<GPUInstance>& instances);

    /// Upload light data.
    void updateLights(const std::vector<GPULight>& lights);

    /// Upload material data.
    void updateMaterials(const std::vector<GPUMaterial>& materials);

    /// Flush any pending mesh data to the GPU.
    /// Called automatically by getSceneGlobals() if dirty, but can be called explicitly.
    void flushMeshData();

    /// Get SceneGlobals for push constants / UBO.
    SceneGlobals getSceneGlobals();

    /// Upload SceneGlobals to a GPU buffer and return its BDA.
    /// Must be called after getSceneGlobals() and any per-frame updates
    /// so that all buffer addresses are valid.
    VkDeviceAddress uploadSceneGlobalsBuffer();

    u32 getMeshletTotalCount() const { return meshletTotalCount_; }
    u32 getMeshCount() const { return static_cast<u32>(meshInfos_.size()); }

private:
    void uploadToGPU(AllocatedBuffer& dst, const void* data, VkDeviceSize size);
    void growAndUpload(AllocatedBuffer& dst, const void* data, VkDeviceSize size,
                       VkBufferUsageFlags extraUsage = 0);
    AllocatedBuffer createDeviceBuffer(VkDeviceSize size, VkBufferUsageFlags extraUsage = 0);
    void destroyBufferIfValid(AllocatedBuffer& buf);

    VulkanDevice& device_;
    GpuAllocator& allocator_;
    CommandManager& commands_;

    // Per-mesh data accumulated across uploadMesh calls
    std::vector<GPUVertex> allVertices_;
    std::vector<Meshlet> allMeshlets_;
    std::vector<u32> allMeshletVertices_;
    std::vector<u8> allMeshletTriangles_;
    std::vector<MeshletBounds> allMeshletBounds_;
    std::vector<GPUMeshInfo> meshInfos_;

    // GPU buffers — mesh geometry (rebuilt when gpuBuffersDirty_)
    AllocatedBuffer vertexBuffer_{};
    AllocatedBuffer meshletBuffer_{};
    AllocatedBuffer meshletVertexBuffer_{};
    AllocatedBuffer meshletTriangleBuffer_{};
    AllocatedBuffer meshletBoundsBuffer_{};
    AllocatedBuffer meshInfoBuffer_{};

    // GPU buffers — per-frame dynamic data
    AllocatedBuffer instanceBuffer_{};
    AllocatedBuffer materialBuffer_{};
    AllocatedBuffer lightBuffer_{};
    AllocatedBuffer sceneGlobalsBuffer_{};

    u32 meshletTotalCount_ = 0;
    u32 instanceCount_     = 0;
    u32 lightCount_        = 0;
    bool gpuBuffersDirty_  = false;
};

} // namespace phosphor
