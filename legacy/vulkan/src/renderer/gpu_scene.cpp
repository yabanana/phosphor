#include "renderer/gpu_scene.h"
#include "rhi/vk_device.h"
#include "rhi/vk_commands.h"
#include "core/log.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>

namespace phosphor {

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------

GpuScene::GpuScene(VulkanDevice& device, GpuAllocator& allocator, CommandManager& commands)
    : device_(device), allocator_(allocator), commands_(commands) {
    LOG_INFO("GpuScene created");
}

GpuScene::~GpuScene() {
    device_.waitIdle();

    destroyBufferIfValid(vertexBuffer_);
    destroyBufferIfValid(meshletBuffer_);
    destroyBufferIfValid(meshletVertexBuffer_);
    destroyBufferIfValid(meshletTriangleBuffer_);
    destroyBufferIfValid(meshletBoundsBuffer_);
    destroyBufferIfValid(meshInfoBuffer_);
    destroyBufferIfValid(instanceBuffer_);
    destroyBufferIfValid(materialBuffer_);
    destroyBufferIfValid(lightBuffer_);
    destroyBufferIfValid(sceneGlobalsBuffer_);

    LOG_INFO("GpuScene destroyed");
}

// ---------------------------------------------------------------------------
// Mesh upload
// ---------------------------------------------------------------------------

MeshHandle GpuScene::uploadMesh(
    const std::vector<glm::vec3>& positions,
    const std::vector<glm::vec3>& normals,
    const std::vector<glm::vec4>& tangents,
    const std::vector<glm::vec2>& uvs,
    const std::vector<u32>& indices) {

    assert(!positions.empty());
    assert(indices.size() % 3 == 0);

    const size_t vertexCount = positions.size();
    const bool hasNormals  = normals.size() == vertexCount;
    const bool hasTangents = tangents.size() == vertexCount;
    const bool hasUVs      = uvs.size() == vertexCount;

    // Record offsets before appending
    const u32 vertexOffset        = static_cast<u32>(allVertices_.size());
    const u32 meshletOffset       = static_cast<u32>(allMeshlets_.size());
    const u32 meshletVertexOffset = static_cast<u32>(allMeshletVertices_.size());
    const u32 meshletTriOffset    = static_cast<u32>(allMeshletTriangles_.size());

    // Convert to GPUVertex format
    std::vector<GPUVertex> verts(vertexCount);
    for (size_t i = 0; i < vertexCount; ++i) {
        GPUVertex& v = verts[i];
        v.px = positions[i].x;
        v.py = positions[i].y;
        v.pz = positions[i].z;

        if (hasNormals) {
            v.nx = normals[i].x;
            v.ny = normals[i].y;
            v.nz = normals[i].z;
        } else {
            v.nx = 0.0f;
            v.ny = 1.0f;
            v.nz = 0.0f;
        }

        if (hasTangents) {
            v.tx = tangents[i].x;
            v.ty = tangents[i].y;
            v.tz = tangents[i].z;
            v.tw = tangents[i].w;
        } else {
            v.tx = 1.0f;
            v.ty = 0.0f;
            v.tz = 0.0f;
            v.tw = 1.0f;
        }

        if (hasUVs) {
            v.u = uvs[i].x;
            v.v = uvs[i].y;
        } else {
            v.u = 0.0f;
            v.v = 0.0f;
        }
    }

    // Build meshlets
    // meshopt expects tightly-packed float3 positions for its spatial computations,
    // but we can point it at our GPUVertex array with the correct stride since
    // position is at offset 0.
    MeshletBuildResult meshletData = MeshletBuilder::build(
        reinterpret_cast<const float*>(positions.data()),
        vertexCount,
        sizeof(glm::vec3),
        indices.data(),
        indices.size());

    // Compute a bounding sphere for the entire mesh
    float bsCenter[3] = {0.0f, 0.0f, 0.0f};
    for (size_t i = 0; i < vertexCount; ++i) {
        bsCenter[0] += positions[i].x;
        bsCenter[1] += positions[i].y;
        bsCenter[2] += positions[i].z;
    }
    const float invCount = 1.0f / static_cast<float>(vertexCount);
    bsCenter[0] *= invCount;
    bsCenter[1] *= invCount;
    bsCenter[2] *= invCount;

    float bsRadius = 0.0f;
    for (size_t i = 0; i < vertexCount; ++i) {
        const float dx = positions[i].x - bsCenter[0];
        const float dy = positions[i].y - bsCenter[1];
        const float dz = positions[i].z - bsCenter[2];
        const float distSq = dx * dx + dy * dy + dz * dz;
        if (distSq > bsRadius) {
            bsRadius = distSq;
        }
    }
    bsRadius = std::sqrt(bsRadius);

    // Append to global arrays
    allVertices_.insert(allVertices_.end(), verts.begin(), verts.end());

    // Meshlet vertex indices need to be offset by our vertex base
    // (meshopt produces local vertex indices into the input mesh,
    //  and we need them to be global into allVertices_)
    const size_t mvBase = allMeshletVertices_.size();
    allMeshletVertices_.insert(allMeshletVertices_.end(),
                               meshletData.meshletVertices.begin(),
                               meshletData.meshletVertices.end());
    // Offset the vertex indices so they point into the global vertex buffer
    for (size_t i = mvBase; i < allMeshletVertices_.size(); ++i) {
        allMeshletVertices_[i] += vertexOffset;
    }

    allMeshletTriangles_.insert(allMeshletTriangles_.end(),
                                meshletData.meshletTriangles.begin(),
                                meshletData.meshletTriangles.end());

    // Offset meshlet vertex/triangle offsets to be global
    for (auto& m : meshletData.meshlets) {
        Meshlet globalMeshlet{};
        globalMeshlet.vertexOffset   = m.vertexOffset + meshletVertexOffset;
        globalMeshlet.vertexCount    = m.vertexCount;
        globalMeshlet.triangleOffset = m.triangleOffset + meshletTriOffset;
        globalMeshlet.triangleCount  = m.triangleCount;
        allMeshlets_.push_back(globalMeshlet);
    }

    allMeshletBounds_.insert(allMeshletBounds_.end(),
                             meshletData.bounds.begin(),
                             meshletData.bounds.end());

    // Create mesh info entry
    GPUMeshInfo info{};
    info.meshletCount  = static_cast<u32>(meshletData.meshlets.size());
    info.meshletOffset = meshletOffset;
    info.vertexOffset  = vertexOffset;
    info.indexOffset    = 0; // meshlet rendering does not use a traditional index buffer
    info.boundingSphere[0] = bsCenter[0];
    info.boundingSphere[1] = bsCenter[1];
    info.boundingSphere[2] = bsCenter[2];
    info.boundingSphere[3] = bsRadius;
    meshInfos_.push_back(info);

    meshletTotalCount_ = static_cast<u32>(allMeshlets_.size());
    gpuBuffersDirty_ = true;

    const MeshHandle handle = static_cast<MeshHandle>(meshInfos_.size() - 1);
    LOG_INFO("Uploaded mesh %u: %zu vertices, %zu meshlets",
             handle, vertexCount, meshletData.meshlets.size());
    return handle;
}

// ---------------------------------------------------------------------------
// Per-frame dynamic data updates
// ---------------------------------------------------------------------------

void GpuScene::updateInstances(const std::vector<GPUInstance>& instances) {
    if (instances.empty()) {
        return;
    }
    const VkDeviceSize size = instances.size() * sizeof(GPUInstance);
    growAndUpload(instanceBuffer_, instances.data(), size);
    instanceCount_ = static_cast<u32>(instances.size());
}

void GpuScene::updateLights(const std::vector<GPULight>& lights) {
    if (lights.empty()) {
        return;
    }
    const VkDeviceSize size = lights.size() * sizeof(GPULight);
    growAndUpload(lightBuffer_, lights.data(), size);
    lightCount_ = static_cast<u32>(lights.size());
}

void GpuScene::updateMaterials(const std::vector<GPUMaterial>& materials) {
    if (materials.empty()) {
        return;
    }
    const VkDeviceSize size = materials.size() * sizeof(GPUMaterial);
    growAndUpload(materialBuffer_, materials.data(), size);
}

// ---------------------------------------------------------------------------
// Flush mesh geometry to GPU
// ---------------------------------------------------------------------------

void GpuScene::flushMeshData() {
    if (!gpuBuffersDirty_) {
        return;
    }

    device_.waitIdle();

    // Destroy old buffers
    destroyBufferIfValid(vertexBuffer_);
    destroyBufferIfValid(meshletBuffer_);
    destroyBufferIfValid(meshletVertexBuffer_);
    destroyBufferIfValid(meshletTriangleBuffer_);
    destroyBufferIfValid(meshletBoundsBuffer_);
    destroyBufferIfValid(meshInfoBuffer_);

    // Helper: only create and upload if there is data
    auto uploadIfNonEmpty = [&](AllocatedBuffer& dst, const void* data, VkDeviceSize size) {
        if (size == 0) {
            return;
        }
        dst = createDeviceBuffer(size);
        uploadToGPU(dst, data, size);
    };

    uploadIfNonEmpty(vertexBuffer_,
                     allVertices_.data(),
                     allVertices_.size() * sizeof(GPUVertex));

    uploadIfNonEmpty(meshletBuffer_,
                     allMeshlets_.data(),
                     allMeshlets_.size() * sizeof(Meshlet));

    uploadIfNonEmpty(meshletVertexBuffer_,
                     allMeshletVertices_.data(),
                     allMeshletVertices_.size() * sizeof(u32));

    uploadIfNonEmpty(meshletTriangleBuffer_,
                     allMeshletTriangles_.data(),
                     allMeshletTriangles_.size() * sizeof(u8));

    uploadIfNonEmpty(meshletBoundsBuffer_,
                     allMeshletBounds_.data(),
                     allMeshletBounds_.size() * sizeof(MeshletBounds));

    uploadIfNonEmpty(meshInfoBuffer_,
                     meshInfos_.data(),
                     meshInfos_.size() * sizeof(GPUMeshInfo));

    gpuBuffersDirty_ = false;

    LOG_INFO("Flushed mesh data to GPU: %zu vertices, %zu meshlets, %zu mesh infos",
             allVertices_.size(), allMeshlets_.size(), meshInfos_.size());
}

// ---------------------------------------------------------------------------
// Scene globals
// ---------------------------------------------------------------------------

SceneGlobals GpuScene::getSceneGlobals() {
    if (gpuBuffersDirty_) {
        flushMeshData();
    }

    SceneGlobals g{};
    g.vertexBufferAddress    = vertexBuffer_.deviceAddress;
    g.meshletBufferAddress   = meshletBuffer_.deviceAddress;
    g.meshletVertexAddress   = meshletVertexBuffer_.deviceAddress;
    g.meshletTriangleAddress = meshletTriangleBuffer_.deviceAddress;
    g.instanceBufferAddress  = instanceBuffer_.deviceAddress;
    g.materialBufferAddress  = materialBuffer_.deviceAddress;
    g.meshInfoBufferAddress  = meshInfoBuffer_.deviceAddress;
    g.lightBufferAddress     = lightBuffer_.deviceAddress;
    g.meshletBoundsAddress   = meshletBoundsBuffer_.deviceAddress;
    g.instanceCount          = instanceCount_;
    g.lightCount             = lightCount_;
    g.meshletTotalCount      = meshletTotalCount_;
    g.pad                    = 0;
    return g;
}

// ---------------------------------------------------------------------------
// Upload SceneGlobals to GPU and return BDA
// ---------------------------------------------------------------------------

VkDeviceAddress GpuScene::uploadSceneGlobalsBuffer() {
    SceneGlobals g = getSceneGlobals();

    const VkDeviceSize size = sizeof(SceneGlobals);
    growAndUpload(sceneGlobalsBuffer_, &g, size);
    return sceneGlobalsBuffer_.deviceAddress;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

AllocatedBuffer GpuScene::createDeviceBuffer(VkDeviceSize size, VkBufferUsageFlags extraUsage) {
    return allocator_.createBuffer(
        size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        extraUsage,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        0);
}

void GpuScene::uploadToGPU(AllocatedBuffer& dst, const void* data, VkDeviceSize size) {
    // Create a host-visible staging buffer
    AllocatedBuffer staging = allocator_.createBuffer(
        size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
        VMA_ALLOCATION_CREATE_MAPPED_BIT);

    // Copy data into the staging buffer
    void* mapped = allocator_.mapMemory(staging);
    std::memcpy(mapped, data, static_cast<size_t>(size));
    allocator_.unmapMemory(staging);

    // Record and submit a copy command
    commands_.immediateSubmit([&](VkCommandBuffer cmd) {
        VkBufferCopy region{};
        region.srcOffset = 0;
        region.dstOffset = 0;
        region.size      = size;
        vkCmdCopyBuffer(cmd, staging.buffer, dst.buffer, 1, &region);
    });

    allocator_.destroyBuffer(staging);
}

void GpuScene::growAndUpload(AllocatedBuffer& dst, const void* data, VkDeviceSize size,
                             VkBufferUsageFlags extraUsage) {
    // Re-create the buffer if the existing one is too small
    if (dst.buffer == VK_NULL_HANDLE || dst.size < size) {
        destroyBufferIfValid(dst);
        dst = createDeviceBuffer(size, extraUsage);
    }
    uploadToGPU(dst, data, size);
}

void GpuScene::destroyBufferIfValid(AllocatedBuffer& buf) {
    if (buf.buffer != VK_NULL_HANDLE) {
        allocator_.destroyBuffer(buf);
    }
}

} // namespace phosphor
