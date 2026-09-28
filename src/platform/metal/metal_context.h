#pragma once

#include "core/types.h"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <array>
#include <atomic>
#include <functional>
#include <vector>

namespace phosphor {

constexpr u32 METAL_FRAMES_IN_FLIGHT = 3;

// ---------------------------------------------------------------------------
// MetalContext -- owns the Metal 4 device-level objects and the frame loop.
//
// Frame protocol (one MTLSharedEvent as the timeline):
//   value 2n+1  "scene done"  signalled by the MTL4 queue after frame n
//   value 2n+2  "frame done"  signalled by the overlay command buffer
// The CPU waits for "frame done" of frame n - METAL_FRAMES_IN_FLIGHT before
// reusing that slot's command allocator, upload memory and deferred releases.
//
// Metal 4 command buffers neither retain resources nor make them resident:
// every long-lived allocation is added to residencySet(), and objects that
// may still be referenced by in-flight GPU work go through deferRelease().
// ---------------------------------------------------------------------------

class MetalContext {
public:
    struct Frame {
        MTL4::CommandBuffer* commandBuffer = nullptr;
        CA::MetalDrawable*   drawable      = nullptr;
        u32                  slot          = 0;
        u64                  index         = 0;
    };

    explicit MetalContext(CA::MetalLayer* layer);
    ~MetalContext();

    MetalContext(const MetalContext&) = delete;
    MetalContext& operator=(const MetalContext&) = delete;

    [[nodiscard]] MTL::Device*        device()       const { return device_; }
    [[nodiscard]] MTL4::CommandQueue* queue()        const { return queue_; }
    [[nodiscard]] MTL::CommandQueue*  legacyQueue()  const { return legacyQueue_; }
    [[nodiscard]] MTL4::Compiler*     compiler()     const { return compiler_; }
    [[nodiscard]] CA::MetalLayer*     layer()        const { return layer_; }
    [[nodiscard]] MTL::SharedEvent*   frameEvent()   const { return frameEvent_; }
    [[nodiscard]] MTL::PixelFormat    colorFormat()  const { return MTL::PixelFormatBGRA8Unorm_sRGB; }
    [[nodiscard]] const char*         gpuName()      const;
    [[nodiscard]] bool                isApple9OrLater() const { return apple9_; }
    /// GPU time of the most recently completed scene command buffer.
    [[nodiscard]] float               lastGpuMs() const { return lastGpuMs_.load(std::memory_order_relaxed); }

    /// Register a long-lived allocation for residency (committed lazily).
    void makeResident(const MTL::Allocation* allocation);
    /// Remove an allocation from the residency set (committed lazily).
    void evict(const MTL::Allocation* allocation);
    /// Release an object once all frames currently in flight have finished.
    void deferRelease(NS::Object* object);

    /// Resize the drawable to match the window's pixel size.
    void resize(u32 width, u32 height);
    [[nodiscard]] u32 width()  const { return width_; }
    [[nodiscard]] u32 height() const { return height_; }

    /// Wait for the frame slot, begin its command buffer and acquire a drawable.
    /// Returns false if no drawable is available (e.g. minimised window).
    bool beginFrame(Frame& frame);

    /// End and commit the scene command buffer and signal "scene done".
    /// Returns a Metal 3 command buffer (from legacyQueue()) that already
    /// waits for the scene; the caller encodes the overlay (ImGui) into it and
    /// then calls presentFrame().
    MTL::CommandBuffer* submitScene(Frame& frame);

    /// Present the drawable from `overlay`, signal "frame done" and commit.
    void presentFrame(Frame& frame, MTL::CommandBuffer* overlay);

    /// Record commands into a one-off command buffer, submit, and block until
    /// the GPU has finished.  Used for loading-time uploads only.
    void submitAndWait(const std::function<void(MTL4::ComputeCommandEncoder*)>& record);

    /// Block until every submitted frame has completed.
    void waitIdle();

private:
    void flushResidency();
    void waitForValue(u64 value);

    CA::MetalLayer*        layer_       = nullptr;
    MTL::Device*           device_      = nullptr;
    MTL4::CommandQueue*    queue_       = nullptr;
    MTL::CommandQueue*     legacyQueue_ = nullptr;
    MTL4::Compiler*        compiler_    = nullptr;
    MTL::ResidencySet*     residency_   = nullptr;
    MTL::SharedEvent*      frameEvent_  = nullptr;
    MTL::SharedEvent*      uploadEvent_ = nullptr;
    MTL4::CommandBuffer*   uploadCommandBuffer_ = nullptr;
    MTL4::CommandAllocator* uploadAllocator_ = nullptr;

    std::array<MTL4::CommandAllocator*, METAL_FRAMES_IN_FLIGHT> allocators_{};
    std::array<MTL4::CommandBuffer*, METAL_FRAMES_IN_FLIGHT> commandBuffers_{};
    // Objects released once the frame recorded in that slot has completed.
    std::array<std::vector<NS::Object*>, METAL_FRAMES_IN_FLIGHT> pendingReleases_{};

    std::atomic<float> lastGpuMs_{0.0f};
    bool residencyDirty_ = false;
    bool apple9_         = false;
    u64  frameIndex_     = 0;
    u64  uploadValue_    = 0;
    u32  width_          = 0;
    u32  height_         = 0;
};

} // namespace phosphor
