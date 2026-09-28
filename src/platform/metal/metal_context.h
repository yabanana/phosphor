#pragma once

#include "core/memory/memory_budget.h"
#include "core/types.h"
#include "platform/metal/residency_manager.h"
#include "platform/metal/upload_ring.h"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <array>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace phosphor {

class GpuMemory;

constexpr u32 METAL_FRAMES_IN_FLIGHT = 3;

// ---------------------------------------------------------------------------
// MetalContext -- owns the Metal 4 device-level objects and the frame loop.
//
// Frame protocol (one MTLSharedEvent as the timeline): every frame is a single
// MTL4 command buffer (scene + overlay) and the queue signals value n+1 once
// frame n has finished.  The CPU waits for frame n - METAL_FRAMES_IN_FLIGHT
// before reusing that slot's command allocator, upload memory and deferred
// releases.
//
// Metal 4 command buffers neither retain resources nor make them resident:
// GPU allocations go through memory() (which makes them resident), and
// objects that may still be referenced by in-flight GPU work go through
// deferRelease().
//
// CPU-written data goes through two upload rings: frameUploads() for data
// consumed by the frame being recorded, stagingAllocate() + enqueueUpload() +
// flushUploads() for loading-time copies into private resources.
// ---------------------------------------------------------------------------

class MetalContext {
public:
    struct Frame {
        MTL4::CommandBuffer* commandBuffer = nullptr;
        CA::MetalDrawable*   drawable      = nullptr;
        u32                  slot          = 0;
        u64                  index         = 0;
    };

    /// `libraryPath` is the compiled phosphor.metallib shared by every pass.
    MetalContext(CA::MetalLayer* layer, const std::string& libraryPath);
    ~MetalContext();

    MetalContext(const MetalContext&) = delete;
    MetalContext& operator=(const MetalContext&) = delete;

    [[nodiscard]] MTL::Device*        device()       const { return device_; }
    [[nodiscard]] MTL4::CommandQueue* queue()        const { return queue_; }
    [[nodiscard]] MTL4::Compiler*     compiler()     const { return compiler_; }
    [[nodiscard]] MTL::Library*       library()      const { return library_; }
    [[nodiscard]] CA::MetalLayer*     layer()        const { return layer_; }
    [[nodiscard]] MTL::SharedEvent*   frameEvent()   const { return frameEvent_; }
    [[nodiscard]] MTL::PixelFormat    colorFormat()  const { return MTL::PixelFormatBGRA8Unorm_sRGB; }
    [[nodiscard]] const char*         gpuName()      const;
    [[nodiscard]] bool                isApple9OrLater() const { return apple9_; }
    [[nodiscard]] bool                isApple10OrLater() const { return apple10_; }
    [[nodiscard]] const MemoryBudget& budget() const { return budget_; }
    /// GPU time of the most recently completed frame command buffer.
    [[nodiscard]] float               lastGpuMs() const { return lastGpuMs_.load(std::memory_order_relaxed); }
    /// Index of the frame being recorded, or of the next one between frames.
    [[nodiscard]] u64                 frameIndex() const { return frameIndex_; }

    [[nodiscard]] GpuMemory&  memory()       { return *memory_; }
    [[nodiscard]] const ResidencyManager& residency() const { return *residency_; }
    [[nodiscard]] UploadRing& frameUploads() { return *frameUploads_; }
    [[nodiscard]] const UploadRing& frameUploads() const { return *frameUploads_; }
    [[nodiscard]] const UploadRing& staging() const { return *staging_; }

    /// Register a long-lived allocation for residency; committed before the
    /// next command buffer commit, so it may be used by the frame being recorded.
    void makeResident(const MTL::Allocation* allocation, ResidencyClass cls = ResidencyClass::Static);
    /// Remove an allocation from the residency set (committed lazily).
    void evict(const MTL::Allocation* allocation);
    /// Release an object once all frames currently in flight have finished
    /// (and, if `evict`, remove it from the residency set at that point).
    void deferRelease(NS::Object* object);
    void deferRelease(MTL::Resource* resource, bool evict);
    /// Same, removing `evict` (e.g. a heap) from the residency set first.
    void deferRelease(NS::Object* object, const MTL::Allocation* evict);

    /// Resize the drawable to match the window's pixel size.
    void resize(u32 width, u32 height);
    [[nodiscard]] u32 width()  const { return width_; }
    [[nodiscard]] u32 height() const { return height_; }

    /// Wait for the frame slot, begin its command buffer and acquire a drawable.
    /// Returns false if no drawable is available (e.g. minimised window).
    bool beginFrame(Frame& frame);

    /// End and commit the frame's command buffer, present its drawable and
    /// signal "frame done".
    void submitFrame(Frame& frame);

    /// Record the GPU time of the next `frames` frames (benchmark mode).
    void beginGpuTimeCapture(u32 frames);
    /// Wait for the captured frames and return their GPU times in ms.
    std::vector<float> endGpuTimeCapture();

    /// Staging memory for a loading-time upload.  If the staging ring is full
    /// the uploads queued so far are flushed first; oversized requests get a
    /// one-off buffer.  Valid until the flushUploads() that consumes it.
    [[nodiscard]] UploadRing::Slice stagingAllocate(u64 size, u64 alignment = 256);
    /// Queue GPU work (copies, mip generation) that reads staging slices.
    void enqueueUpload(std::function<void(MTL4::ComputeCommandEncoder*)> record);
    /// Run every queued upload in one command buffer, wait for it and recycle
    /// the staging ring.  Loading time only (blocks).
    void flushUploads();

    /// Record commands into a one-off command buffer, submit, and block until
    /// the GPU has finished.  Used for loading-time uploads only.
    void submitAndWait(const std::function<void(MTL4::ComputeCommandEncoder*)>& record);

    /// Block until every submitted frame has completed.
    void waitIdle();

    /// waitIdle() and release every deferred object now, then commit the
    /// residency sets so the memory is actually returned.  Only between frames
    /// (nothing recorded yet may reference them), e.g. on a bench switch.
    void collectGarbage();

    /// Commit pending residency changes now (they are otherwise committed
    /// with the next command buffer).  Residency sets keep removed
    /// allocations alive until this commit.
    void commitResidency() { flushResidency(); }

private:
    void flushResidency();
    void onFrameFeedback(MTL4::CommitFeedback* feedback);
    void waitForValue(u64 value);

    CA::MetalLayer*        layer_       = nullptr;
    MTL::Device*           device_      = nullptr;
    MTL4::CommandQueue*    queue_       = nullptr;
    MTL4::Compiler*        compiler_    = nullptr;
    MTL::Library*          library_     = nullptr;
    MTL::SharedEvent*      frameEvent_  = nullptr;
    MTL::SharedEvent*      uploadEvent_ = nullptr;
    MTL4::CommandBuffer*   uploadCommandBuffer_ = nullptr;
    MTL4::CommandAllocator* uploadAllocator_ = nullptr;

    std::array<MTL4::CommandAllocator*, METAL_FRAMES_IN_FLIGHT> allocators_{};
    std::array<MTL4::CommandBuffer*, METAL_FRAMES_IN_FLIGHT> commandBuffers_{};
    // Objects released once frame `afterFrame` has completed: the frame being
    // recorded when deferRelease() was called, or -- between frames -- the
    // next one, which is conservative for every frame still in flight.
    struct PendingRelease {
        NS::Object*            object     = nullptr;
        const MTL::Allocation* evict      = nullptr; // removed from residency first
        u64                    afterFrame = 0;
    };
    std::vector<PendingRelease> pendingReleases_;
    /// Release every entry whose frame is <= `completedFrame` (all if ~0).
    void releaseCompleted(u64 completedFrame);

    std::unique_ptr<ResidencyManager> residency_;
    std::unique_ptr<GpuMemory>  memory_;
    std::unique_ptr<UploadRing> frameUploads_;
    std::unique_ptr<UploadRing> staging_;
    std::vector<std::function<void(MTL4::ComputeCommandEncoder*)>> queuedUploads_;

    std::atomic<u64>     feedbackCount_{0};
    std::atomic<float>   lastGpuMs_{0.0f};
    // Benchmark capture: frame index -> GPU ms, filled by commit feedback
    // handlers on a Metal thread.  Feedback can arrive after the frame event,
    // so the reader waits on the received count, not on waitIdle().
    std::mutex              gpuTimesMutex_;
    std::condition_variable gpuTimesCv_;
    std::vector<float>      gpuTimes_;
    u64                     gpuTimesFirst_    = 0;
    u32                     gpuTimesReceived_ = 0;
    bool apple9_         = false;
    bool apple10_        = false;
    MemoryBudget budget_;
    u64  frameIndex_     = 0;
    u64  uploadValue_    = 0;
    u32  width_          = 0;
    u32  height_         = 0;
};

} // namespace phosphor
