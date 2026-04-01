#include "diagnostics/breadcrumbs.h"
#include "rhi/vk_device.h"
#include "rhi/vk_allocator.h"
#include "core/log.h"

#include <cstring>

namespace phosphor {

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------

GpuBreadcrumbs::GpuBreadcrumbs(VulkanDevice& device, GpuAllocator& allocator)
    : device_(device)
    , markerBuffer_{}
{
    // Try to load NV checkpoint extension functions
    VkDevice dev = device_.getDevice();
    vkCmdSetCheckpointNV_ = reinterpret_cast<PFN_vkCmdSetCheckpointNV>(
        vkGetDeviceProcAddr(dev, "vkCmdSetCheckpointNV"));
    vkGetQueueCheckpointDataNV_ = reinterpret_cast<PFN_vkGetQueueCheckpointDataNV>(
        vkGetDeviceProcAddr(dev, "vkGetQueueCheckpointDataNV"));

    if (vkCmdSetCheckpointNV_ && vkGetQueueCheckpointDataNV_) {
        // Allocate a small host-visible buffer to hold marker indices.
        // The buffer itself is not strictly needed for the NV checkpoint API
        // (which uses pCheckpointMarker as an opaque pointer), but we keep it
        // for future AMD breadcrumb-buffer based implementations.
        markerBuffer_ = allocator.createBuffer(
            4096,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_TO_CPU,
            VMA_ALLOCATION_CREATE_MAPPED_BIT);

        LOG_INFO("GPU breadcrumbs enabled (VK_NV_device_diagnostic_checkpoints)");
    } else {
        vkCmdSetCheckpointNV_ = nullptr;
        vkGetQueueCheckpointDataNV_ = nullptr;
        LOG_INFO("GPU breadcrumbs not available (VK_NV_device_diagnostic_checkpoints not supported)");
    }
}

GpuBreadcrumbs::~GpuBreadcrumbs() {
    if (markerBuffer_.buffer != VK_NULL_HANDLE) {
        // Deliberately not destroying here -- GpuAllocator owns the lifetime.
        // The allocator will be destroyed after this object.
    }
}

// ---------------------------------------------------------------------------
// Insert a checkpoint
// ---------------------------------------------------------------------------

void GpuBreadcrumbs::insertCheckpoint(VkCommandBuffer cmd, const char* passName) {
    if (!vkCmdSetCheckpointNV_) {
        return; // no-op: extension not available
    }

    u32 markerValue = nextMarker_++;
    markerNames_[markerValue] = passName;

    // The NV checkpoint API takes an opaque const void*.
    // We encode our marker value directly as a pointer-sized integer.
    // This avoids needing persistent storage for the pointer target.
    const void* marker = reinterpret_cast<const void*>(
        static_cast<uintptr_t>(markerValue));
    vkCmdSetCheckpointNV_(cmd, marker);
}

// ---------------------------------------------------------------------------
// Retrieve checkpoints after device lost
// ---------------------------------------------------------------------------

std::vector<std::string> GpuBreadcrumbs::retrieveCheckpoints(VkQueue queue) {
    std::vector<std::string> result;

    if (!vkGetQueueCheckpointDataNV_) {
        return result; // no-op: extension not available
    }

    u32 count = 0;
    vkGetQueueCheckpointDataNV_(queue, &count, nullptr);
    if (count == 0) {
        return result;
    }

    std::vector<VkCheckpointDataNV> checkpoints(count);
    for (auto& cp : checkpoints) {
        cp.sType = VK_STRUCTURE_TYPE_CHECKPOINT_DATA_NV;
        cp.pNext = nullptr;
    }
    vkGetQueueCheckpointDataNV_(queue, &count, checkpoints.data());

    result.reserve(count);
    for (const auto& cp : checkpoints) {
        u32 markerValue = static_cast<u32>(
            reinterpret_cast<uintptr_t>(cp.pCheckpointMarker));
        auto it = markerNames_.find(markerValue);
        if (it != markerNames_.end()) {
            result.push_back(it->second);
        } else {
            result.push_back("unknown_marker_" + std::to_string(markerValue));
        }
    }

    return result;
}

} // namespace phosphor
