#include "rhi/vk_rt.h"
#include "rhi/vk_device.h"
#include "rhi/vk_allocator.h"
#include "rhi/vk_commands.h"
#include "core/log.h"

#include <algorithm>
#include <cstring>

namespace phosphor {

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------

AccelerationStructureManager::AccelerationStructureManager(
    VulkanDevice& device, GpuAllocator& allocator, CommandManager& commands)
    : device_(device)
    , allocator_(allocator)
    , commands_(commands)
{
    loadFunctionPointers();
    LOG_INFO("AccelerationStructureManager initialised");
}

AccelerationStructureManager::~AccelerationStructureManager()
{
    VkDevice dev = device_.getDevice();

    // Destroy TLAS
    if (tlas_.handle != VK_NULL_HANDLE) {
        vkDestroyAS_(dev, tlas_.handle, nullptr);
    }
    if (tlas_.buffer.buffer != VK_NULL_HANDLE) {
        allocator_.destroyBuffer(tlas_.buffer);
    }
    if (tlas_.instanceBuffer.buffer != VK_NULL_HANDLE) {
        allocator_.destroyBuffer(tlas_.instanceBuffer);
    }

    // Destroy shared scratch
    if (scratchBuffer_.buffer != VK_NULL_HANDLE) {
        allocator_.destroyBuffer(scratchBuffer_);
    }

    LOG_INFO("AccelerationStructureManager destroyed");
}

// ---------------------------------------------------------------------------
// Function pointer loading
// ---------------------------------------------------------------------------

void AccelerationStructureManager::loadFunctionPointers()
{
    VkDevice dev = device_.getDevice();

    vkCreateAS_      = reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(
                           vkGetDeviceProcAddr(dev, "vkCreateAccelerationStructureKHR"));
    vkDestroyAS_     = reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(
                           vkGetDeviceProcAddr(dev, "vkDestroyAccelerationStructureKHR"));
    vkGetBuildSizes_ = reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(
                           vkGetDeviceProcAddr(dev, "vkGetAccelerationStructureBuildSizesKHR"));
    vkCmdBuildAS_    = reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(
                           vkGetDeviceProcAddr(dev, "vkCmdBuildAccelerationStructuresKHR"));
    vkGetASAddress_  = reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(
                           vkGetDeviceProcAddr(dev, "vkGetAccelerationStructureDeviceAddressKHR"));

    if (!vkCreateAS_ || !vkDestroyAS_ || !vkGetBuildSizes_ ||
        !vkCmdBuildAS_ || !vkGetASAddress_) {
        LOG_ERROR("Failed to load VK_KHR_acceleration_structure function pointers — "
                  "ensure the extension is enabled");
        std::abort();
    }
}

// ---------------------------------------------------------------------------
// Scratch buffer management
// ---------------------------------------------------------------------------

void AccelerationStructureManager::ensureScratch(VkDeviceSize requiredSize)
{
    if (requiredSize <= scratchSize_) {
        return;
    }

    if (scratchBuffer_.buffer != VK_NULL_HANDLE) {
        allocator_.destroyBuffer(scratchBuffer_);
    }

    // Round up to 256-byte alignment (required by spec)
    requiredSize = (requiredSize + 255) & ~VkDeviceSize{255};

    scratchBuffer_ = allocator_.createBuffer(
        requiredSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);

    scratchSize_ = requiredSize;
    LOG_DEBUG("RT scratch buffer grown to %llu bytes", static_cast<unsigned long long>(requiredSize));
}

// ---------------------------------------------------------------------------
// Instance upload helper
// ---------------------------------------------------------------------------

void AccelerationStructureManager::uploadInstances(
    const std::vector<VkAccelerationStructureInstanceKHR>& instances)
{
    const VkDeviceSize instanceDataSize =
        instances.size() * sizeof(VkAccelerationStructureInstanceKHR);

    // Recreate the instance buffer if it is too small
    if (tlas_.instanceBuffer.buffer == VK_NULL_HANDLE ||
        tlas_.instanceBuffer.size < instanceDataSize) {

        if (tlas_.instanceBuffer.buffer != VK_NULL_HANDLE) {
            allocator_.destroyBuffer(tlas_.instanceBuffer);
        }

        tlas_.instanceBuffer = allocator_.createBuffer(
            instanceDataSize,
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
            VMA_MEMORY_USAGE_CPU_TO_GPU);
    }

    // Map and copy
    void* mapped = allocator_.mapMemory(tlas_.instanceBuffer);
    std::memcpy(mapped, instances.data(), instanceDataSize);
    allocator_.unmapMemory(tlas_.instanceBuffer);

    tlas_.instanceCount = static_cast<u32>(instances.size());
}

// ---------------------------------------------------------------------------
// buildBLAS
// ---------------------------------------------------------------------------

BLAS AccelerationStructureManager::buildBLAS(
    const float* vertices, u32 vertexCount, u32 vertexStride,
    const u32* indices, u32 indexCount)
{
    VkDevice dev = device_.getDevice();

    // -- Upload vertex and index data to GPU buffers --
    const VkDeviceSize vertexDataSize = static_cast<VkDeviceSize>(vertexCount) * vertexStride;
    const VkDeviceSize indexDataSize  = static_cast<VkDeviceSize>(indexCount) * sizeof(u32);

    AllocatedBuffer vertexBuf = allocator_.createBuffer(
        vertexDataSize,
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        VMA_MEMORY_USAGE_CPU_TO_GPU);
    {
        void* mapped = allocator_.mapMemory(vertexBuf);
        std::memcpy(mapped, vertices, vertexDataSize);
        allocator_.unmapMemory(vertexBuf);
    }

    AllocatedBuffer indexBuf = allocator_.createBuffer(
        indexDataSize,
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        VMA_MEMORY_USAGE_CPU_TO_GPU);
    {
        void* mapped = allocator_.mapMemory(indexBuf);
        std::memcpy(mapped, indices, indexDataSize);
        allocator_.unmapMemory(indexBuf);
    }

    // -- Describe the geometry --
    VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
    triangles.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    triangles.vertexFormat  = VK_FORMAT_R32G32B32_SFLOAT;
    triangles.vertexData.deviceAddress = vertexBuf.deviceAddress;
    triangles.vertexStride  = vertexStride;
    triangles.maxVertex     = vertexCount - 1;
    triangles.indexType     = VK_INDEX_TYPE_UINT32;
    triangles.indexData.deviceAddress = indexBuf.deviceAddress;

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.flags        = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geometry.geometry.triangles = triangles;

    const u32 primitiveCount = indexCount / 3;

    // -- Query build sizes --
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type          = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries   = &geometry;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

    vkGetBuildSizes_(dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                     &buildInfo, &primitiveCount, &sizeInfo);

    // -- Allocate the AS buffer --
    BLAS blas{};
    blas.buffer = allocator_.createBuffer(
        sizeInfo.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);

    // -- Create the acceleration structure object --
    VkAccelerationStructureCreateInfoKHR createInfo{};
    createInfo.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    createInfo.buffer = blas.buffer.buffer;
    createInfo.size   = sizeInfo.accelerationStructureSize;
    createInfo.type   = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

    VK_CHECK(vkCreateAS_(dev, &createInfo, nullptr, &blas.handle));

    // -- Get device address --
    VkAccelerationStructureDeviceAddressInfoKHR addrInfo{};
    addrInfo.sType                 = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    addrInfo.accelerationStructure = blas.handle;
    blas.deviceAddress = vkGetASAddress_(dev, &addrInfo);

    // -- Ensure scratch is large enough --
    ensureScratch(sizeInfo.buildScratchSize);

    // -- Build --
    buildInfo.dstAccelerationStructure  = blas.handle;
    buildInfo.scratchData.deviceAddress = scratchBuffer_.deviceAddress;

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount  = primitiveCount;
    rangeInfo.primitiveOffset = 0;
    rangeInfo.firstVertex     = 0;
    rangeInfo.transformOffset = 0;

    const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = &rangeInfo;

    commands_.immediateSubmit([&](VkCommandBuffer cmd) {
        vkCmdBuildAS_(cmd, 1, &buildInfo, &pRangeInfo);

        // Barrier: AS build writes must complete before any read
        VkMemoryBarrier2 barrier{};
        barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        barrier.srcStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        barrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        barrier.dstStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                                VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
        barrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;

        VkDependencyInfo dep{};
        dep.sType                = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.memoryBarrierCount   = 1;
        dep.pMemoryBarriers      = &barrier;

        vkCmdPipelineBarrier2(cmd, &dep);
    });

    // Cleanup temporary upload buffers
    allocator_.destroyBuffer(vertexBuf);
    allocator_.destroyBuffer(indexBuf);

    LOG_DEBUG("Built BLAS: %u verts, %u tris, %llu bytes",
              vertexCount, primitiveCount,
              static_cast<unsigned long long>(sizeInfo.accelerationStructureSize));

    return blas;
}

// ---------------------------------------------------------------------------
// buildTLAS
// ---------------------------------------------------------------------------

void AccelerationStructureManager::buildTLAS(
    const std::vector<VkAccelerationStructureInstanceKHR>& instances)
{
    VkDevice dev = device_.getDevice();

    // Destroy old TLAS if present
    if (tlas_.handle != VK_NULL_HANDLE) {
        vkDestroyAS_(dev, tlas_.handle, nullptr);
        tlas_.handle = VK_NULL_HANDLE;
    }
    if (tlas_.buffer.buffer != VK_NULL_HANDLE) {
        allocator_.destroyBuffer(tlas_.buffer);
        tlas_.buffer = {};
    }

    if (instances.empty()) {
        LOG_WARN("buildTLAS called with zero instances — TLAS not created");
        return;
    }

    uploadInstances(instances);

    // -- Describe geometry (instances) --
    VkAccelerationStructureGeometryInstancesDataKHR instancesData{};
    instancesData.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    instancesData.arrayOfPointers    = VK_FALSE;
    instancesData.data.deviceAddress = tlas_.instanceBuffer.deviceAddress;

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.flags        = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geometry.geometry.instances = instancesData;

    const u32 instanceCount = tlas_.instanceCount;

    // -- Query build sizes --
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type          = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                              VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    buildInfo.mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries   = &geometry;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

    vkGetBuildSizes_(dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                     &buildInfo, &instanceCount, &sizeInfo);

    // -- Allocate TLAS buffer --
    tlas_.buffer = allocator_.createBuffer(
        sizeInfo.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);

    // -- Create TLAS handle --
    VkAccelerationStructureCreateInfoKHR createInfo{};
    createInfo.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    createInfo.buffer = tlas_.buffer.buffer;
    createInfo.size   = sizeInfo.accelerationStructureSize;
    createInfo.type   = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;

    VK_CHECK(vkCreateAS_(dev, &createInfo, nullptr, &tlas_.handle));

    // -- Get device address --
    VkAccelerationStructureDeviceAddressInfoKHR addrInfo{};
    addrInfo.sType                 = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    addrInfo.accelerationStructure = tlas_.handle;
    tlas_.deviceAddress = vkGetASAddress_(dev, &addrInfo);

    // -- Ensure scratch --
    ensureScratch(sizeInfo.buildScratchSize);

    // -- Build --
    buildInfo.dstAccelerationStructure  = tlas_.handle;
    buildInfo.scratchData.deviceAddress = scratchBuffer_.deviceAddress;

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount  = instanceCount;
    rangeInfo.primitiveOffset = 0;
    rangeInfo.firstVertex     = 0;
    rangeInfo.transformOffset = 0;

    const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = &rangeInfo;

    commands_.immediateSubmit([&](VkCommandBuffer cmd) {
        vkCmdBuildAS_(cmd, 1, &buildInfo, &pRangeInfo);

        VkMemoryBarrier2 barrier{};
        barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        barrier.srcStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        barrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        barrier.dstStageMask  = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
        barrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;

        VkDependencyInfo dep{};
        dep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers    = &barrier;

        vkCmdPipelineBarrier2(cmd, &dep);
    });

    LOG_INFO("Built TLAS: %u instances, %llu bytes",
             instanceCount,
             static_cast<unsigned long long>(sizeInfo.accelerationStructureSize));
}

// ---------------------------------------------------------------------------
// updateTLAS — refit in-place
// ---------------------------------------------------------------------------

void AccelerationStructureManager::updateTLAS(
    const std::vector<VkAccelerationStructureInstanceKHR>& instances)
{
    if (tlas_.handle == VK_NULL_HANDLE) {
        LOG_WARN("updateTLAS called before buildTLAS — falling back to full build");
        buildTLAS(instances);
        return;
    }

    if (instances.empty()) {
        LOG_WARN("updateTLAS called with zero instances — skipping");
        return;
    }

    uploadInstances(instances);

    // -- Describe geometry --
    VkAccelerationStructureGeometryInstancesDataKHR instancesData{};
    instancesData.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    instancesData.arrayOfPointers    = VK_FALSE;
    instancesData.data.deviceAddress = tlas_.instanceBuffer.deviceAddress;

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.flags        = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geometry.geometry.instances = instancesData;

    const u32 instanceCount = tlas_.instanceCount;

    // -- Query update scratch size --
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type          = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                              VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    buildInfo.mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
    buildInfo.srcAccelerationStructure = tlas_.handle;
    buildInfo.dstAccelerationStructure = tlas_.handle;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries   = &geometry;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

    vkGetBuildSizes_(device_.getDevice(),
                     VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                     &buildInfo, &instanceCount, &sizeInfo);

    ensureScratch(sizeInfo.updateScratchSize);

    buildInfo.scratchData.deviceAddress = scratchBuffer_.deviceAddress;

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount  = instanceCount;
    rangeInfo.primitiveOffset = 0;
    rangeInfo.firstVertex     = 0;
    rangeInfo.transformOffset = 0;

    const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = &rangeInfo;

    commands_.immediateSubmit([&](VkCommandBuffer cmd) {
        vkCmdBuildAS_(cmd, 1, &buildInfo, &pRangeInfo);

        VkMemoryBarrier2 barrier{};
        barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        barrier.srcStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        barrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        barrier.dstStageMask  = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
        barrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;

        VkDependencyInfo dep{};
        dep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers    = &barrier;

        vkCmdPipelineBarrier2(cmd, &dep);
    });

    LOG_DEBUG("Updated TLAS: %u instances (refit)", instanceCount);
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

VkAccelerationStructureKHR AccelerationStructureManager::getTLAS() const
{
    return tlas_.handle;
}

VkDeviceAddress AccelerationStructureManager::getTLASAddress() const
{
    return tlas_.deviceAddress;
}

// ---------------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------------

void AccelerationStructureManager::destroyBLAS(BLAS& blas)
{
    if (blas.handle != VK_NULL_HANDLE) {
        vkDestroyAS_(device_.getDevice(), blas.handle, nullptr);
        blas.handle = VK_NULL_HANDLE;
    }
    if (blas.buffer.buffer != VK_NULL_HANDLE) {
        allocator_.destroyBuffer(blas.buffer);
        blas.buffer = {};
    }
    blas.deviceAddress = 0;
}

} // namespace phosphor
