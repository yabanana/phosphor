#pragma once

#include "core/types.h"
#include "rhi/vk_common.h"
#include "rhi/vk_allocator.h"

#include <vector>

namespace phosphor {

class VulkanDevice;
class GpuAllocator;
class CommandManager;

// ---------------------------------------------------------------------------
// BLAS — Bottom-Level Acceleration Structure (per-mesh geometry)
// ---------------------------------------------------------------------------

struct BLAS {
    VkAccelerationStructureKHR handle        = VK_NULL_HANDLE;
    AllocatedBuffer            buffer;
    VkDeviceAddress            deviceAddress = 0;
};

// ---------------------------------------------------------------------------
// TLAS — Top-Level Acceleration Structure (scene of instances)
// ---------------------------------------------------------------------------

struct TLAS {
    VkAccelerationStructureKHR handle         = VK_NULL_HANDLE;
    AllocatedBuffer            buffer;
    AllocatedBuffer            instanceBuffer; // VkAccelerationStructureInstanceKHR array
    VkDeviceAddress            deviceAddress  = 0;
    u32                        instanceCount  = 0;
};

// ---------------------------------------------------------------------------
// AccelerationStructureManager — builds and manages BLAS/TLAS
// ---------------------------------------------------------------------------

class AccelerationStructureManager {
public:
    AccelerationStructureManager(VulkanDevice& device, GpuAllocator& allocator,
                                 CommandManager& commands);
    ~AccelerationStructureManager();

    AccelerationStructureManager(const AccelerationStructureManager&)            = delete;
    AccelerationStructureManager& operator=(const AccelerationStructureManager&) = delete;
    AccelerationStructureManager(AccelerationStructureManager&&)                 = delete;
    AccelerationStructureManager& operator=(AccelerationStructureManager&&)      = delete;

    /// Build a BLAS from interleaved vertex positions and a triangle index buffer.
    /// @param vertices       Pointer to the first float of position data.
    /// @param vertexCount    Number of vertices.
    /// @param vertexStride   Byte stride between consecutive vertices (e.g. sizeof(GPUVertex)).
    /// @param indices        Triangle index buffer (u32).
    /// @param indexCount     Number of indices (must be a multiple of 3).
    BLAS buildBLAS(const float* vertices, u32 vertexCount, u32 vertexStride,
                   const u32* indices, u32 indexCount);

    /// Build or rebuild the TLAS from scratch.
    void buildTLAS(const std::vector<VkAccelerationStructureInstanceKHR>& instances);

    /// Refit (update) the existing TLAS in-place (transforms only, no topology change).
    void updateTLAS(const std::vector<VkAccelerationStructureInstanceKHR>& instances);

    VkAccelerationStructureKHR getTLAS() const;
    VkDeviceAddress            getTLASAddress() const;

    void destroyBLAS(BLAS& blas);

private:
    VulkanDevice&   device_;
    GpuAllocator&   allocator_;
    CommandManager& commands_;

    TLAS            tlas_{};
    AllocatedBuffer scratchBuffer_{};     // shared scratch — grown as needed
    VkDeviceSize    scratchSize_ = 0;

    // --- Vulkan RT extension function pointers ---
    PFN_vkCreateAccelerationStructureKHR              vkCreateAS_      = nullptr;
    PFN_vkDestroyAccelerationStructureKHR             vkDestroyAS_     = nullptr;
    PFN_vkGetAccelerationStructureBuildSizesKHR       vkGetBuildSizes_ = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR           vkCmdBuildAS_    = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR    vkGetASAddress_  = nullptr;

    void loadFunctionPointers();

    /// Ensure the shared scratch buffer is at least @p requiredSize bytes.
    void ensureScratch(VkDeviceSize requiredSize);

    /// Upload instance data to the GPU-side instance buffer inside @p tlas_.
    void uploadInstances(const std::vector<VkAccelerationStructureInstanceKHR>& instances);
};

} // namespace phosphor
