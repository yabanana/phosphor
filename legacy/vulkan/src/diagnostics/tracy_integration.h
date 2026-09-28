#pragma once

// ---------------------------------------------------------------------------
// Tracy Profiler integration for Phosphor.
//
// When PHOSPHOR_ENABLE_TRACY (mapped to TRACY_ENABLE in CMake) is defined,
// these macros expand to real Tracy instrumentation calls.  Otherwise they
// compile away to nothing, adding zero overhead.
//
// Usage in code:
//   PHOSPHOR_CPU_ZONE("MyFunction");
//   PHOSPHOR_FRAME_MARK();
//   PHOSPHOR_GPU_ZONE(cmd, "ShadowPass");
// ---------------------------------------------------------------------------

#ifdef TRACY_ENABLE

#include <tracy/Tracy.hpp>
#include <tracy/TracyVulkan.hpp>

#define PHOSPHOR_CPU_ZONE(name)      ZoneScopedN(name)
#define PHOSPHOR_FRAME_MARK()        FrameMark
#define PHOSPHOR_GPU_ZONE(ctx, cmd, name) \
    TracyVkZone(ctx, cmd, name)

#else // Tracy disabled

#define PHOSPHOR_CPU_ZONE(name)            ((void)0)
#define PHOSPHOR_FRAME_MARK()              ((void)0)
#define PHOSPHOR_GPU_ZONE(ctx, cmd, name)  ((void)0)

#endif // TRACY_ENABLE

#include <vulkan/vulkan.h>
#include "core/types.h"

namespace phosphor {

class VulkanDevice;

// ---------------------------------------------------------------------------
// TracyGpuContext -- manages Tracy's Vulkan GPU profiling context.
// When Tracy is disabled, all methods are harmless no-ops.
// ---------------------------------------------------------------------------

class TracyGpuContext {
public:
    /// Create the GPU context.  Must be called after device and queue creation.
    /// @param device    The Vulkan device wrapper
    /// @param queue     The graphics queue to profile
    /// @param cmdPool   A command pool from which to allocate calibration commands
    void init(VulkanDevice& device, VkQueue queue, VkCommandPool cmdPool);

    /// Destroy the GPU context.  Safe to call even if init() was never called.
    void destroy();

    /// Collect completed GPU timestamp queries.  Call once per frame.
    void collect(VkCommandBuffer cmd);

    /// Returns the raw Tracy context pointer (or nullptr when disabled).
    void* getContext() const { return context_; }

private:
    void* context_ = nullptr; // tracy::VkCtx* when enabled, nullptr otherwise
};

} // namespace phosphor
