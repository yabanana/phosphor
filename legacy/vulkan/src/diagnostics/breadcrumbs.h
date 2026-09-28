#pragma once

#include "core/types.h"
#include "rhi/vk_common.h"
#include "rhi/vk_allocator.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace phosphor {

class VulkanDevice;
class GpuAllocator;

/// GPU breadcrumbs using VK_NV_device_diagnostic_checkpoints.
///
/// Insert lightweight checkpoints into command buffers during recording.
/// After a device-lost event, retrieve the last checkpoint(s) that the GPU
/// actually reached -- narrowing down which pass caused the crash.
///
/// All methods are no-ops if the extension is not available on the device.
class GpuBreadcrumbs {
public:
    GpuBreadcrumbs(VulkanDevice& device, GpuAllocator& allocator);
    ~GpuBreadcrumbs();

    GpuBreadcrumbs(const GpuBreadcrumbs&) = delete;
    GpuBreadcrumbs& operator=(const GpuBreadcrumbs&) = delete;

    /// Insert a named checkpoint into the command buffer.
    /// The marker value is an integer that maps to @p passName in our table.
    void insertCheckpoint(VkCommandBuffer cmd, const char* passName);

    /// After a device-lost event, query the queue for the checkpoints the GPU
    /// actually reached.  Returns the names of all reached checkpoints.
    std::vector<std::string> retrieveCheckpoints(VkQueue queue);

    bool isAvailable() const { return vkCmdSetCheckpointNV_ != nullptr; }

private:
    VulkanDevice&   device_;
    AllocatedBuffer markerBuffer_;

    std::unordered_map<u32, std::string> markerNames_;
    u32 nextMarker_ = 0;

    PFN_vkCmdSetCheckpointNV        vkCmdSetCheckpointNV_        = nullptr;
    PFN_vkGetQueueCheckpointDataNV  vkGetQueueCheckpointDataNV_  = nullptr;
};

} // namespace phosphor
