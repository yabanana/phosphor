#include "diagnostics/tracy_integration.h"
#include "rhi/vk_device.h"
#include "core/log.h"

namespace phosphor {

// ---------------------------------------------------------------------------
// When Tracy is enabled, create a real VkCtx for GPU profiling.
// ---------------------------------------------------------------------------

#ifdef TRACY_ENABLE

void TracyGpuContext::init(VulkanDevice& device, VkQueue queue, VkCommandPool cmdPool) {
    // Allocate a transient command buffer for the calibration commands
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool        = cmdPool;
    allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(device.getDevice(), &allocInfo, &cmd));

    // Tracy creates its own query pool and calibration internally
    context_ = TracyVkContext(
        device.getPhysicalDevice(),
        device.getDevice(),
        queue,
        cmd);

    vkFreeCommandBuffers(device.getDevice(), cmdPool, 1, &cmd);

    LOG_INFO("Tracy GPU context created");
}

void TracyGpuContext::destroy() {
    if (context_) {
        TracyVkDestroy(static_cast<tracy::VkCtx*>(context_));
        context_ = nullptr;
        LOG_INFO("Tracy GPU context destroyed");
    }
}

void TracyGpuContext::collect(VkCommandBuffer cmd) {
    if (context_) {
        TracyVkCollect(static_cast<tracy::VkCtx*>(context_), cmd);
    }
}

#else // Tracy disabled -- no-op implementations

void TracyGpuContext::init(VulkanDevice& /*device*/, VkQueue /*queue*/,
                           VkCommandPool /*cmdPool*/) {
    LOG_DEBUG("Tracy GPU context: compiled without TRACY_ENABLE (no-op)");
}

void TracyGpuContext::destroy() {
    // nothing to do
}

void TracyGpuContext::collect(VkCommandBuffer /*cmd*/) {
    // nothing to do
}

#endif // TRACY_ENABLE

} // namespace phosphor
