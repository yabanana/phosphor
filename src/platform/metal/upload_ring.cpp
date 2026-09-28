#include "platform/metal/upload_ring.h"
#include "platform/metal/gpu_memory.h"
#include "core/log.h"

#include <algorithm>

namespace phosphor {

namespace {

constexpr MTL::ResourceOptions kUploadOptions =
    MTL::ResourceStorageModeShared | MTL::ResourceCPUCacheModeWriteCombined;

} // namespace

UploadRing::UploadRing(GpuMemory& memory, u64 capacity, u32 framesInFlight, const char* label)
    : memory_(memory), label_(label), ring_(capacity, framesInFlight) {
    buffer_ = memory_.newBuffer(capacity, kUploadOptions, MemoryCategory::Upload, label_);
}

UploadRing::~UploadRing() {
    memory_.release(buffer_, MemoryCategory::Upload);
}

UploadRing::Slice UploadRing::sliceAt(MTL::Buffer* buffer, u64 offset) const {
    return {static_cast<u8*>(buffer->contents()) + offset, buffer->gpuAddress() + offset, buffer, offset};
}

void UploadRing::beginFrame(u64 frameIndex) {
    if (growTo_ > 0) {
        // Frames in flight may still read the old buffer: release it deferred
        // and start the new, larger ring empty.
        LOG_WARN("UploadRing '%s': growing %llu -> %llu bytes after an overflow", label_,
                 static_cast<unsigned long long>(ring_.capacity()), static_cast<unsigned long long>(growTo_));
        memory_.release(buffer_, MemoryCategory::Upload);
        buffer_ = memory_.newBuffer(growTo_, kUploadOptions, MemoryCategory::Upload, label_);
        ring_.resize(growTo_);
        growTo_ = 0;
    }
    ring_.beginFrame(frameIndex);
}

void UploadRing::endFrame() {
    ring_.endFrame();
}

UploadRing::Slice UploadRing::tryAllocate(u64 size, u64 alignment) {
    const auto offset = ring_.allocate(size, alignment);
    return offset ? sliceAt(buffer_, *offset) : Slice{};
}

UploadRing::Slice UploadRing::allocate(u64 size, u64 alignment) {
    if (Slice slice = tryAllocate(size, alignment)) return slice;

    // Full: serve this request from a one-off buffer released after the frame,
    // and ask for a ring that holds the in-flight frames plus this peak.
    const LinearRing::Stats s = ring_.stats();
    growTo_ = std::max(growTo_, std::max(ring_.capacity() * 2, (s.frameBytes + size) * 4));
    ++oneOffBuffers_;
    MTL::Buffer* oneOff = memory_.newBuffer(std::max<u64>(size, 1), kUploadOptions, MemoryCategory::Upload,
                                            "Upload overflow");
    memory_.release(oneOff, MemoryCategory::Upload); // alive until the frame completes
    return sliceAt(oneOff, 0);
}

void UploadRing::reset() {
    ring_.reset();
}

} // namespace phosphor
