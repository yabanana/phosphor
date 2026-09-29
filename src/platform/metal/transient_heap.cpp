#include "platform/metal/transient_heap.h"
#include "platform/metal/gpu_memory.h"
#include "platform/metal/metal_context.h"
#include "core/log.h"

namespace phosphor {

namespace {

constexpr MTL::ResourceOptions kTransientOptions =
    MTL::ResourceStorageModePrivate | MTL::ResourceHazardTrackingModeUntracked;

NS::String* str(const char* s) {
    return NS::String::string(s, NS::UTF8StringEncoding);
}

} // namespace

TransientHeap::TransientHeap(MetalContext& context) : context_(context) {}

TransientHeap::~TransientHeap() {
    context_.memory().releaseHeap(heap_, MemoryCategory::Transient);
}

bool TransientHeap::reserve(u64 bytes) {
    if (heap_ && heap_->size() >= bytes) return true;
    context_.memory().releaseHeap(heap_, MemoryCategory::Transient);
    heap_ = context_.memory().newPlacementHeap(bytes, MemoryCategory::Transient, "Transient heap");
    return heap_ != nullptr;
}

MTL::SizeAndAlign TransientHeap::sizeAndAlign(const MTL::TextureDescriptor* descriptor) const {
    return context_.device()->heapTextureSizeAndAlign(descriptor);
}

MTL::SizeAndAlign TransientHeap::sizeAndAlign(u64 length) const {
    return context_.device()->heapBufferSizeAndAlign(length, kTransientOptions);
}

bool TransientHeap::fits(u64 offset, const MTL::SizeAndAlign& sa, const char* label) const {
    if (!heap_ || offset % sa.align != 0 || offset + sa.size > heap_->size()) {
        LOG_ERROR("TransientHeap: '%s' does not fit at offset %llu (size %llu, align %llu, heap %llu)", label,
                  static_cast<unsigned long long>(offset), static_cast<unsigned long long>(sa.size),
                  static_cast<unsigned long long>(sa.align), static_cast<unsigned long long>(size()));
        return false;
    }
    return true;
}

MTL::Texture* TransientHeap::createTexture(const MTL::TextureDescriptor* descriptor, u64 offset,
                                           const char* label) {
    if (descriptor->storageMode() != MTL::StorageModePrivate) {
        LOG_ERROR("TransientHeap: '%s' must use private storage", label);
        return nullptr;
    }
    if (!fits(offset, sizeAndAlign(descriptor), label)) return nullptr;
    MTL::Texture* texture = heap_->newTexture(descriptor, offset);
    if (texture) texture->setLabel(str(label));
    return texture;
}

MTL::Buffer* TransientHeap::createBuffer(u64 length, u64 offset, const char* label) {
    if (!fits(offset, sizeAndAlign(length), label)) return nullptr;
    MTL::Buffer* buffer = heap_->newBuffer(length, kTransientOptions, offset);
    if (buffer) buffer->setLabel(str(label));
    return buffer;
}

void TransientHeap::release(MTL::Resource* resource, bool evict) {
    // Resident through the heap: nothing to evict individually, unless the
    // caller registered the resource itself.
    context_.deferRelease(resource, evict);
}

} // namespace phosphor
