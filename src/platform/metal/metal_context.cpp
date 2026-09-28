#include "platform/metal/metal_context.h"
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

// Initial ring sizes; UploadRing grows the frame ring after an overflow and
// the staging ring flushes when full.  Per-tier values arrive with F1.4.
constexpr u64 kFrameUploadCapacity = 64ull << 20;
constexpr u64 kStagingCapacity     = 64ull << 20;

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

    MTL4::CompilerDescriptor* compilerDesc = MTL4::CompilerDescriptor::alloc()->init();
    compilerDesc->setLabel(str("Phosphor compiler"));
    compiler_ = device_->newCompiler(compilerDesc, &error);
    compilerDesc->release();
    if (!compiler_) {
        throw std::runtime_error("Failed to create MTL4Compiler");
    }

    library_ = device_->newLibrary(str(libraryPath.c_str()), &error);
    if (!library_) {
        const char* reason = error ? error->localizedDescription()->utf8String() : "unknown error";
        throw std::runtime_error(std::string("Failed to load shader library ") + libraryPath + ": " + reason);
    }

    residency_ = std::make_unique<ResidencyManager>(device_, queue_);

    for (u32 i = 0; i < METAL_FRAMES_IN_FLIGHT; ++i) {
        allocators_[i]     = device_->newCommandAllocator();
        commandBuffers_[i] = device_->newCommandBuffer();
        commandBuffers_[i]->setLabel(str("Frame"));
    }
    uploadAllocator_     = device_->newCommandAllocator();
    uploadCommandBuffer_ = device_->newCommandBuffer();

    frameEvent_  = device_->newSharedEvent();
    uploadEvent_ = device_->newSharedEvent();

    layer_->setDevice(device_);
    layer_->setPixelFormat(colorFormat());
    layer_->setFramebufferOnly(false); // --capture copies the drawable into a buffer
    layer_->setMaximumDrawableCount(3);
    // Drawables must be resident for the MTL4 queue; the layer owns this set.
    queue_->addResidencySet(layer_->residencySet());

    memory_       = std::make_unique<GpuMemory>(*this);
    frameUploads_ = std::make_unique<UploadRing>(*memory_, kFrameUploadCapacity, METAL_FRAMES_IN_FLIGHT,
                                                 "Frame uploads");
    staging_      = std::make_unique<UploadRing>(*memory_, kStagingCapacity, 1, "Staging");
}

MetalContext::~MetalContext() {
    waitIdle();
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
    residency_.reset();
    library_->release();
    compiler_->release();
    queue_->release();
    device_->release();
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
    }
}

bool MetalContext::beginFrame(Frame& frame) {
    const u64 index = frameIndex_;
    const u32 slot  = static_cast<u32>(index % METAL_FRAMES_IN_FLIGHT);

    if (index >= METAL_FRAMES_IN_FLIGHT) {
        waitForValue(frameDoneValue(index - METAL_FRAMES_IN_FLIGHT));
        // Frames up to index - N have finished: recycle what they could use.
        releaseCompleted(index - METAL_FRAMES_IN_FLIGHT);
    }

    CA::MetalDrawable* drawable = layer_->nextDrawable();
    if (!drawable) {
        return false;
    }
    frameUploads_->beginFrame(index);

    allocators_[slot]->reset();
    MTL4::CommandBuffer* cmd = commandBuffers_[slot];
    cmd->beginCommandBuffer(allocators_[slot]);

    frame.commandBuffer = cmd;
    frame.drawable      = drawable;
    frame.slot          = slot;
    frame.index         = index;
    return true;
}

void MetalContext::submitFrame(Frame& frame) {
    frame.commandBuffer->endCommandBuffer();
    frameUploads_->endFrame();
    // Allocations registered while recording this frame (grown upload
    // buffers, capture readback) must be resident before the commit.
    flushResidency();

    // A drawable must be waited on before the MTL4 queue renders to it, and
    // signalled after the work that renders to it, before present().
    queue_->wait(frame.drawable);
    const MTL4::CommandBuffer* buffers[] = {frame.commandBuffer};
    // A fresh options object per commit: a reused MTL4CommitOptions stops
    // delivering feedback after its first commit (measured: 0 of 600 frames).
    // This small per-frame allocation is the one accepted exception to O7.
    MTL4::CommitOptions* options = MTL4::CommitOptions::alloc()->init();
    options->addFeedbackHandler([this](MTL4::CommitFeedback* feedback) { onFrameFeedback(feedback); });
    queue_->commit(buffers, 1, options);
    options->release();
    queue_->signalDrawable(frame.drawable);
    frame.drawable->present();
    queue_->signalEvent(frameEvent_, frameDoneValue(frame.index));
    ++frameIndex_;
}

void MetalContext::onFrameFeedback(MTL4::CommitFeedback* feedback) {
    // Only frame commits register this handler, and feedback for one queue
    // arrives in commit order, so the n-th feedback belongs to frame n.
    const u64 index = feedbackCount_.fetch_add(1, std::memory_order_relaxed);
    if (feedback->error()) return;
    const float ms = static_cast<float>((feedback->GPUEndTime() - feedback->GPUStartTime()) * 1000.0);
    lastGpuMs_.store(ms, std::memory_order_relaxed);
    std::lock_guard lock(gpuTimesMutex_);
    if (index >= gpuTimesFirst_ && index - gpuTimesFirst_ < gpuTimes_.size()) {
        gpuTimes_[index - gpuTimesFirst_] = ms;
        ++gpuTimesReceived_;
        gpuTimesCv_.notify_all();
    }
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
                              [&] { return gpuTimesReceived_ >= expected; })) {
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
        LOG_ERROR("GPU timeout waiting for upload");
    }
}

void MetalContext::collectGarbage() {
    waitIdle();
    releaseCompleted(~u64{0});
}

void MetalContext::waitIdle() {
    if (frameIndex_ > 0) {
        waitForValue(frameDoneValue(frameIndex_ - 1));
    }
}

} // namespace phosphor
