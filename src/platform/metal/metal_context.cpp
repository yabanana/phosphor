#include <cstdlib>
#include <thread>
#include "platform/metal/metal_context.h"

#include <sys/sysctl.h>
#include "platform/metal/gpu_memory.h"
#include "platform/metal/residency_manager.h"
#include "core/log.h"

#include <chrono>
#include <stdexcept>

namespace phosphor {

namespace {

NS::String* str(const char* s) {
    return NS::String::string(s, NS::UTF8StringEncoding);
}

constexpr u64 kWaitTimeoutMs = 5000;


u64 frameDoneValue(u64 frameIndex) { return frameIndex + 1; }

} // namespace

MetalContext::MetalContext(CA::MetalLayer* layer, const std::string& libraryPath) : layer_(layer) {
    device_ = MTL::CreateSystemDefaultDevice();
    if (!device_) {
        throw std::runtime_error("No Metal device available");
    }
    // The Metal 4 programming model requires Apple7 (M1/A14) and OS 26.
    if (!device_->supportsFamily(MTL::GPUFamilyApple7) || !device_->supportsFamily(MTL::GPUFamilyMetal4)) {
        throw std::runtime_error("Phosphor requires a Metal 4 capable Apple GPU (Apple7+, macOS 26+)");
    }
    apple9_  = device_->supportsFamily(MTL::GPUFamilyApple9);
    apple10_ = device_->supportsFamily(MTL::GPUFamilyApple10);
    budget_  = MemoryBudget(device_->recommendedMaxWorkingSetSize(), detectTier(gpuName(), apple10_));
    LOG_INFO("Metal device: %s (Apple9+: %s, Apple10+: %s), tier %s, working set %.1f GiB, engine budget %.1f GiB",
             gpuName(), apple9_ ? "yes" : "no", apple10_ ? "yes" : "no", tierName(budget_.tier()),
             static_cast<double>(budget_.workingSet()) / (1ull << 30),
             static_cast<double>(budget_.engineLimit()) / (1ull << 30));
    if (!apple9_) {
        LOG_WARN("Below the Apple9 (M3) baseline: ray tracing and mesh-shader ICBs will be unavailable");
    }

    NS::Error* error = nullptr;

    MTL4::CommandQueueDescriptor* queueDesc = MTL4::CommandQueueDescriptor::alloc()->init();
    queueDesc->setLabel(str("Phosphor MTL4 queue"));
    queue_ = device_->newMTL4CommandQueue(queueDesc, &error);
    queueDesc->release();
    if (!queue_) {
        throw std::runtime_error("Failed to create MTL4CommandQueue");
    }
    MTL4::CommandQueueDescriptor* asyncDesc = MTL4::CommandQueueDescriptor::alloc()->init();
    asyncDesc->setLabel(str("Phosphor MTL4 async compute queue"));
    asyncQueue_ = device_->newMTL4CommandQueue(asyncDesc, &error);
    asyncDesc->release();
    if (!asyncQueue_) {
        throw std::runtime_error("Failed to create the async compute MTL4CommandQueue");
    }

    library_ = device_->newLibrary(str(libraryPath.c_str()), &error);
    if (!library_) {
        const char* reason = error ? error->localizedDescription()->utf8String() : "unknown error";
        throw std::runtime_error(std::string("Failed to load shader library ") + libraryPath + ": " + reason);
    }

    residency_ = std::make_unique<ResidencyManager>(device_, queue_, asyncQueue_);

    for (u32 i = 0; i < METAL_FRAMES_IN_FLIGHT; ++i) {
        allocators_[i]     = device_->newCommandAllocator();
        commandBuffers_[i] = device_->newCommandBuffer();
        commandBuffers_[i]->setLabel(str("Frame"));
    }
    uploadAllocator_     = device_->newCommandAllocator();
    uploadCommandBuffer_ = device_->newCommandBuffer();

    frameEvent_  = device_->newSharedEvent();
    graphicsTimeline_ = device_->newSharedEvent();
    asyncTimeline_    = device_->newSharedEvent();
    uploadEvent_ = device_->newSharedEvent();

    layer_->setDevice(device_);
    layer_->setPixelFormat(MTL::PixelFormatBGRA8Unorm_sRGB);
    layer_->setFramebufferOnly(false); // --capture copies the drawable into a buffer
    layer_->setMaximumDrawableCount(3);
    // Drawables must be resident for the MTL4 queue; the layer owns this set.
    queue_->addResidencySet(layer_->residencySet());

    memory_       = std::make_unique<GpuMemory>(*this);
    // Ring sizes come from the budget; the frame ring still grows after an
    // overflow and the staging ring flushes when full.
    frameUploads_ = std::make_unique<UploadRing>(*memory_, budget_.frameUploadRingSize(), METAL_FRAMES_IN_FLIGHT,
                                                 "Frame uploads");
    staging_      = std::make_unique<UploadRing>(*memory_, budget_.stagingRingSize(), 1, "Staging");
    LOG_INFO("Memory pools: frame ring %llu MiB, staging %llu MiB, heap pages %llu MiB",
             static_cast<unsigned long long>(budget_.frameUploadRingSize() >> 20),
             static_cast<unsigned long long>(budget_.stagingRingSize() >> 20),
             static_cast<unsigned long long>(budget_.heapPageSize() >> 20));
}

MetalContext::~MetalContext() {
    waitIdle();
    for (auto *target : offscreenTargets_)
        if (memory_)
            memory_->release(target, MemoryCategory::RenderTargets);
    staging_.reset();
    frameUploads_.reset();
    releaseCompleted(~u64{0});
    memory_.reset();
    for (u32 i = 0; i < METAL_FRAMES_IN_FLIGHT; ++i) {
        commandBuffers_[i]->release();
        allocators_[i]->release();
    }
    uploadCommandBuffer_->release();
    uploadAllocator_->release();
    uploadEvent_->release();
    frameEvent_->release();
    graphicsTimeline_->release();
    asyncTimeline_->release();
    residency_.reset();
    library_->release();
    asyncQueue_->release();
    queue_->release();
    device_->release();
}

u64 MetalContext::physicalMemoryBytes() const {
    u64 bytes = 0;
    size_t size = sizeof(bytes);
    if (sysctlbyname("hw.memsize", &bytes, &size, nullptr, 0) != 0) return 0;
    return bytes;
}

const char* MetalContext::gpuName() const {
    return device_->name()->utf8String();
}

void MetalContext::makeResident(const MTL::Allocation* allocation, ResidencyClass cls) {
    residency_->add(allocation, cls);
}

void MetalContext::evict(const MTL::Allocation* allocation) {
    if (!allocation) return;
    residency_->remove(allocation);
}

void MetalContext::deferRelease(NS::Object* object) {
    if (!object) return;
    pendingReleases_.push_back({object, nullptr, frameIndex_});
}

void MetalContext::deferRelease(MTL::Resource* resource, bool evict) {
    if (!resource) return;
    pendingReleases_.push_back({resource, evict ? resource : nullptr, frameIndex_});
}

void MetalContext::deferRelease(NS::Object* object, const MTL::Allocation* evict) {
    if (!object) return;
    pendingReleases_.push_back({object, evict, frameIndex_});
}

void MetalContext::releaseCompleted(u64 completedFrame) {
    size_t kept = 0;
    for (size_t i = 0; i < pendingReleases_.size(); ++i) {
        const PendingRelease& p = pendingReleases_[i];
        if (completedFrame != ~u64{0} && p.afterFrame > completedFrame) {
            pendingReleases_[kept++] = p;
            continue;
        }
        if (p.evict) evict(p.evict);
        p.object->release();
    }
    pendingReleases_.resize(kept);
    if (memory_) memory_->releaseCompleted(completedFrame);
}

void MetalContext::flushResidency() {
    residency_->commit();
}

void MetalContext::resize(u32 width, u32 height) {
    width_  = width;
    height_ = height;
    layer_->setDrawableSize(CGSize{static_cast<CGFloat>(width), static_cast<CGFloat>(height)});
}

void MetalContext::waitForValue(u64 value) {
    if (frameEvent_->signaledValue() >= value) return;
    if (!frameEvent_->waitUntilSignaledValue(value, kWaitTimeoutMs)) {
        LOG_ERROR("GPU timeout waiting for frame event value %llu (current %llu)",
                  static_cast<unsigned long long>(value),
                  static_cast<unsigned long long>(frameEvent_->signaledValue()));
        std::fflush(nullptr);
        std::_Exit(EXIT_FAILURE);
    }
}

void MetalContext::setFramesInFlight(u32 count) {
    if (count < 1 || count > METAL_FRAMES_IN_FLIGHT)
        throw std::invalid_argument("Frames in flight must be 1..3");
    framesInFlight_ = count;
}

bool MetalContext::beginFrame(Frame& frame) {
    const u64 index = frameIndex_;
    const u32 slot  = static_cast<u32>(index % METAL_FRAMES_IN_FLIGHT);

    if (index >= framesInFlight_) {
        waitForValue(frameDoneValue(index - framesInFlight_));
        waitForAsync(asyncDone_[static_cast<u32>((index - framesInFlight_) % METAL_FRAMES_IN_FLIGHT)]);
    }
    if (index >= METAL_FRAMES_IN_FLIGHT) {
        waitForValue(frameDoneValue(index - METAL_FRAMES_IN_FLIGHT));
        waitForAsync(asyncDone_[slot]);
        // Frames up to index - N have finished: recycle what they could use.
        releaseCompleted(index - METAL_FRAMES_IN_FLIGHT);
    }

    if (offscreen_ &&
        (!offscreenTargets_[0] || offscreenTargets_[0]->width() != width_ ||
         offscreenTargets_[0]->height() != height_ || offscreenTargets_[0]->pixelFormat() != colorFormat())) {
        auto *descriptor = MTL::TextureDescriptor::texture2DDescriptor(colorFormat(), width_, height_, false);
        descriptor->setStorageMode(MTL::StorageModePrivate);
        descriptor->setUsage(MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead);
        for (auto *&texture : offscreenTargets_) {
            memory_->release(texture, MemoryCategory::RenderTargets);
            texture = memory_->newTexture(descriptor, MemoryCategory::RenderTargets, "Offscreen benchmark target");
        }
    }
    CA::MetalDrawable *drawable = offscreen_ ? nullptr : layer_->nextDrawable();
    if (!offscreen_ && !drawable) {
        LOG_WARN("No drawable available for frame %llu (window hidden or occluded?)",
                 static_cast<unsigned long long>(index));
        return false;
    }
    frameUploads_->beginFrame(index);

    allocators_[slot]->reset();
    MTL4::CommandBuffer* cmd = commandBuffers_[slot];
    cmd->beginCommandBuffer(allocators_[slot]);

    frame.commandBuffer   = cmd;
    frame.drawable        = drawable;
    frame.target = offscreen_ ? offscreenTargets_[slot] : drawable->texture();
    frame.slot            = slot;
    frame.index           = index;
    frame.buffers[0]      = cmd;
    frame.bufferCount     = 1;
    frame.submissionCount = 0;
    frame.asyncDoneValue  = 0;
    return true;
}

void MetalContext::submitFrame(Frame& frame) {
    frame.commandBuffer->endCommandBuffer();
    frameUploads_->endFrame();
    // Allocations registered while recording this frame (grown upload
    // buffers, capture readback) must be resident before the commit.
    flushResidency();

    if (frame.submissionCount == 0) {
        frame.submissions[0] = Submission{SubmitQueue::Graphics, 0, frame.bufferCount, 0, 0, 0};
        frame.submissionCount = 1;
    }
    u32 lastGraphics = 0;
    for (u32 i = 0; i < frame.submissionCount; ++i) {
        if (frame.submissions[i].queue == SubmitQueue::Graphics) lastGraphics = i;
    }
    bool drawableWaited = false;
    for (u32 i = 0; i < frame.submissionCount; ++i) {
        const Submission& sub = frame.submissions[i];
        MTL4::CommandQueue* q = sub.queue == SubmitQueue::Async ? asyncQueue_ : queue_;
        if (frame.drawable && sub.queue == SubmitQueue::Graphics && !drawableWaited) {
            // A drawable must be waited on before the MTL4 queue renders to
            // it, and signalled after the work that renders to it.
            queue_->wait(frame.drawable);
            drawableWaited = true;
        }
        const bool async = sub.queue == SubmitQueue::Async;
        if (sub.waitFrame) q->wait(frameEvent_, sub.waitFrame);
        if (sub.waitValue) q->wait(async ? graphicsTimeline_ : asyncTimeline_, sub.waitValue);
        if (sub.fenceEvent && sub.fenceWait) q->wait(sub.fenceEvent, sub.fenceWait);
        const MTL4::CommandBuffer* const* buffers = frame.buffers.data() + sub.firstBuffer;
        // Track every submission, including async/split commits. GPU event
        // completion alone does not drain CPU feedback callbacks.
        auto *options = MTL4::CommitOptions::alloc()->init();
        {
            std::lock_guard lock(feedbackMutex_);
            ++pendingFeedback_;
        }
        const u64 index = frame.index;
        const bool timed = i == lastGraphics;
        options->addFeedbackHandler(
            [this, index, timed](MTL4::CommitFeedback *feedback) { onFrameFeedback(index, feedback, timed); });
        q->commit(buffers, sub.bufferCount, options);
        options->release();
        if (timed && frame.drawable) {
            queue_->signalDrawable(frame.drawable);
            frame.drawable->present();
        }
        if (sub.signalValue) q->signalEvent(async ? asyncTimeline_ : graphicsTimeline_, sub.signalValue);
        if (sub.fenceEvent && sub.fenceSignal) q->signalEvent(sub.fenceEvent, sub.fenceSignal);
    }
    asyncDone_[frame.slot] = frame.asyncDoneValue;
    if (frame.asyncDoneValue) lastAsyncDone_ = frame.asyncDoneValue;
    queue_->signalEvent(frameEvent_, frameDoneValue(frame.index));
    ++frameIndex_;
}

void MetalContext::onFrameFeedback(u64 index, MTL4::CommitFeedback *feedback, bool timed) {
    if (feedbackDelayMs_)
        std::this_thread::sleep_for(std::chrono::milliseconds(feedbackDelayMs_));
    if (timed)
        feedbackCount_.fetch_add(1, std::memory_order_relaxed);
    const bool injected = timed && feedbackFailFrame_ != 0 && index + 1 == feedbackFailFrame_;
    if (NS::Error *error = feedback->error(); error || injected) {
        gpuFailures_.fetch_add(1);
        LOG_ERROR("Frame %llu GPU feedback failure: %s", static_cast<unsigned long long>(index),
                  error ? error->localizedDescription()->utf8String() : "injected negative control");
        gpuTimesCv_.notify_all();
    } else if (timed) {
        const float ms = static_cast<float>((feedback->GPUEndTime() - feedback->GPUStartTime()) * 1000.0);
        lastGpuMs_.store(ms, std::memory_order_relaxed);
        std::lock_guard lock(gpuTimesMutex_);
        if (index >= gpuTimesFirst_ && index - gpuTimesFirst_ < gpuTimes_.size()) {
            gpuTimes_[index - gpuTimesFirst_] = ms;
            ++gpuTimesReceived_;
            gpuTimesCv_.notify_all();
        }
    }
    // Last access to this context. waitIdle also waits for this CPU boundary
    // before the owning context/mutexes can be destroyed.
    std::lock_guard done(feedbackMutex_);
    --pendingFeedback_;
    feedbackDone_.notify_all();
}

void MetalContext::beginGpuTimeCapture(u32 frames) {
    std::lock_guard lock(gpuTimesMutex_);
    gpuTimes_.assign(frames, 0.0f);
    gpuTimesFirst_    = frameIndex_;
    gpuTimesReceived_ = 0;
}

std::vector<float> MetalContext::endGpuTimeCapture() {
    waitIdle();
    std::unique_lock lock(gpuTimesMutex_);
    const size_t expected = gpuTimes_.size();
    if (!gpuTimesCv_.wait_for(lock, std::chrono::milliseconds(kWaitTimeoutMs),
                              [&] { return gpuTimesReceived_ >= expected || gpuFailures_.load() != 0; })) {
        LOG_WARN("GPU timing feedback missing for %zu of %zu frames", expected - gpuTimesReceived_, expected);
    }
    std::vector<float> times = std::move(gpuTimes_);
    gpuTimes_.clear();
    return times;
}

UploadRing::Slice MetalContext::stagingAllocate(u64 size, u64 alignment) {
    if (UploadRing::Slice slice = staging_->tryAllocate(size, alignment)) return slice;
    flushUploads(); // frees the whole staging ring
    if (UploadRing::Slice slice = staging_->tryAllocate(size, alignment)) return slice;
    return staging_->allocate(size, alignment); // larger than the ring: one-off buffer
}

void MetalContext::enqueueUpload(std::function<void(MTL4::ComputeCommandEncoder*)> record) {
    queuedUploads_.push_back(std::move(record));
}

void MetalContext::flushUploads() {
    if (queuedUploads_.empty()) {
        staging_->reset();
        return;
    }
    submitAndWait([this](MTL4::ComputeCommandEncoder* enc) {
        for (const auto& record : queuedUploads_) record(enc);
    });
    queuedUploads_.clear();
    staging_->reset();
}

void MetalContext::submitAndWait(const std::function<void(MTL4::ComputeCommandEncoder*)>& record) {
    flushResidency();

    uploadAllocator_->reset();
    uploadCommandBuffer_->beginCommandBuffer(uploadAllocator_);
    uploadCommandBuffer_->setLabel(str("Upload"));
    MTL4::ComputeCommandEncoder* enc = uploadCommandBuffer_->computeCommandEncoder();
    record(enc);
    enc->endEncoding();
    uploadCommandBuffer_->endCommandBuffer();

    const MTL4::CommandBuffer* buffers[] = {uploadCommandBuffer_};
    queue_->commit(buffers, 1);
    ++uploadValue_;
    queue_->signalEvent(uploadEvent_, uploadValue_);
    if (!uploadEvent_->waitUntilSignaledValue(uploadValue_, kWaitTimeoutMs)) {
        LOG_ERROR("GPU timeout waiting for upload; refusing staging reuse");
        std::fflush(nullptr);
        std::_Exit(EXIT_FAILURE);
    }
}

void MetalContext::collectGarbage() {
    waitIdle();
    releaseCompleted(~u64{0});
    flushResidency();
}

void MetalContext::waitIdle() {
    if (frameIndex_ > 0) {
        waitForValue(frameDoneValue(frameIndex_ - 1));
    }
    waitForAsync(lastAsyncDone_);
    std::unique_lock lock(feedbackMutex_);
    if (!feedbackDone_.wait_for(lock, std::chrono::milliseconds(kWaitTimeoutMs),
                                [this] { return pendingFeedback_ == 0; })) {
        LOG_ERROR("GPU feedback callbacks did not drain; refusing unsafe context destruction");
        std::fflush(nullptr);
        std::_Exit(EXIT_FAILURE);
    }
}

void MetalContext::waitForAsync(u64 value) {
    if (value == 0 || asyncTimeline_->signaledValue() >= value) return;
    if (!asyncTimeline_->waitUntilSignaledValue(value, kWaitTimeoutMs)) {
        LOG_ERROR("GPU timeout waiting for async compute value %llu (current %llu)",
                  static_cast<unsigned long long>(value),
                  static_cast<unsigned long long>(asyncTimeline_->signaledValue()));
        std::fflush(nullptr);
        std::_Exit(EXIT_FAILURE);
    }
}

} // namespace phosphor
