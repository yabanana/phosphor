#include "platform/metal/metal_context.h"
#include "core/log.h"

#include <stdexcept>

namespace phosphor {

namespace {

NS::String* str(const char* s) {
    return NS::String::string(s, NS::UTF8StringEncoding);
}

constexpr u64 kWaitTimeoutMs = 5000;

u64 sceneDoneValue(u64 frameIndex) { return 2 * frameIndex + 1; }
u64 frameDoneValue(u64 frameIndex) { return 2 * frameIndex + 2; }

} // namespace

MetalContext::MetalContext(CA::MetalLayer* layer) : layer_(layer) {
    device_ = MTL::CreateSystemDefaultDevice();
    if (!device_) {
        throw std::runtime_error("No Metal device available");
    }
    // The Metal 4 programming model requires Apple7 (M1/A14) and OS 26.
    if (!device_->supportsFamily(MTL::GPUFamilyApple7) || !device_->supportsFamily(MTL::GPUFamilyMetal4)) {
        throw std::runtime_error("Phosphor requires a Metal 4 capable Apple GPU (Apple7+, macOS 26+)");
    }
    apple9_ = device_->supportsFamily(MTL::GPUFamilyApple9);
    LOG_INFO("Metal device: %s (Apple9+: %s)", gpuName(), apple9_ ? "yes" : "no");
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
    legacyQueue_ = device_->newCommandQueue();
    legacyQueue_->setLabel(str("Phosphor overlay queue"));

    MTL4::CompilerDescriptor* compilerDesc = MTL4::CompilerDescriptor::alloc()->init();
    compilerDesc->setLabel(str("Phosphor compiler"));
    compiler_ = device_->newCompiler(compilerDesc, &error);
    compilerDesc->release();
    if (!compiler_) {
        throw std::runtime_error("Failed to create MTL4Compiler");
    }

    MTL::ResidencySetDescriptor* rsDesc = MTL::ResidencySetDescriptor::alloc()->init();
    rsDesc->setLabel(str("Phosphor static residency"));
    rsDesc->setInitialCapacity(256);
    residency_ = device_->newResidencySet(rsDesc, &error);
    rsDesc->release();
    if (!residency_) {
        throw std::runtime_error("Failed to create residency set");
    }
    queue_->addResidencySet(residency_);

    for (u32 i = 0; i < METAL_FRAMES_IN_FLIGHT; ++i) {
        allocators_[i]     = device_->newCommandAllocator();
        commandBuffers_[i] = device_->newCommandBuffer();
    }
    uploadAllocator_     = device_->newCommandAllocator();
    uploadCommandBuffer_ = device_->newCommandBuffer();

    frameEvent_  = device_->newSharedEvent();
    uploadEvent_ = device_->newSharedEvent();

    layer_->setDevice(device_);
    layer_->setPixelFormat(colorFormat());
    layer_->setFramebufferOnly(false); // the overlay pass loads the scene result
    layer_->setMaximumDrawableCount(3);
    // Drawables must be resident for the MTL4 queue; the layer owns this set.
    queue_->addResidencySet(layer_->residencySet());
}

MetalContext::~MetalContext() {
    waitIdle();
    for (auto& list : pendingReleases_) {
        for (NS::Object* obj : list) obj->release();
        list.clear();
    }
    for (u32 i = 0; i < METAL_FRAMES_IN_FLIGHT; ++i) {
        commandBuffers_[i]->release();
        allocators_[i]->release();
    }
    uploadCommandBuffer_->release();
    uploadAllocator_->release();
    uploadEvent_->release();
    frameEvent_->release();
    residency_->release();
    compiler_->release();
    legacyQueue_->release();
    queue_->release();
    device_->release();
}

const char* MetalContext::gpuName() const {
    return device_->name()->utf8String();
}

void MetalContext::makeResident(const MTL::Allocation* allocation) {
    if (!allocation) return;
    residency_->addAllocation(allocation);
    residencyDirty_ = true;
}

void MetalContext::evict(const MTL::Allocation* allocation) {
    if (!allocation) return;
    residency_->removeAllocation(allocation);
    residencyDirty_ = true;
}

void MetalContext::deferRelease(NS::Object* object) {
    if (!object) return;
    pendingReleases_[frameIndex_ % METAL_FRAMES_IN_FLIGHT].push_back(object);
}

void MetalContext::flushResidency() {
    if (residencyDirty_) {
        residency_->commit();
        residencyDirty_ = false;
    }
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
    }

    // The frame that last used this slot has finished: recycle its memory.
    for (NS::Object* obj : pendingReleases_[slot]) obj->release();
    pendingReleases_[slot].clear();

    CA::MetalDrawable* drawable = layer_->nextDrawable();
    if (!drawable) {
        return false;
    }

    flushResidency();

    allocators_[slot]->reset();
    MTL4::CommandBuffer* cmd = commandBuffers_[slot];
    cmd->beginCommandBuffer(allocators_[slot]);
    cmd->setLabel(str("Scene"));

    frame.commandBuffer = cmd;
    frame.drawable      = drawable;
    frame.slot          = slot;
    frame.index         = index;
    return true;
}

MTL::CommandBuffer* MetalContext::submitScene(Frame& frame) {
    frame.commandBuffer->endCommandBuffer();

    // A drawable must be waited on before the MTL4 queue renders to it.
    queue_->wait(frame.drawable);
    const MTL4::CommandBuffer* buffers[] = {frame.commandBuffer};
    MTL4::CommitOptions* options = MTL4::CommitOptions::alloc()->init();
    options->addFeedbackHandler([this](MTL4::CommitFeedback* feedback) {
        if (!feedback->error()) {
            const double ms = (feedback->GPUEndTime() - feedback->GPUStartTime()) * 1000.0;
            lastGpuMs_.store(static_cast<float>(ms), std::memory_order_relaxed);
        }
    });
    queue_->commit(buffers, 1, options);
    options->release();
    queue_->signalEvent(frameEvent_, sceneDoneValue(frame.index));

    MTL::CommandBuffer* overlay = legacyQueue_->commandBuffer();
    overlay->setLabel(str("Overlay"));
    overlay->encodeWait(frameEvent_, sceneDoneValue(frame.index));
    return overlay;
}

void MetalContext::presentFrame(Frame& frame, MTL::CommandBuffer* overlay) {
    overlay->presentDrawable(frame.drawable);
    overlay->encodeSignalEvent(frameEvent_, frameDoneValue(frame.index));
    overlay->commit();
    ++frameIndex_;
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

void MetalContext::waitIdle() {
    if (frameIndex_ > 0) {
        waitForValue(frameDoneValue(frameIndex_ - 1));
    }
}

} // namespace phosphor
