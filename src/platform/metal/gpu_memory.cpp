#include "platform/metal/gpu_memory.h"
#include "platform/metal/metal_context.h"
#include "core/log.h"

namespace phosphor {

namespace {

NS::String* str(const char* s) {
    return NS::String::string(s, NS::UTF8StringEncoding);
}

bool isMemoryless(const MTL::Resource* resource) {
    return resource->storageMode() == MTL::StorageModeMemoryless;
}

} // namespace

GpuMemory::GpuMemory(MetalContext& context) : context_(context) {}

GpuMemory::~GpuMemory() {
    for (u32 c = 0; c < MEMORY_CATEGORY_COUNT; ++c) {
        if (stats_[c].count != 0) {
            LOG_WARN("GpuMemory: %u %s allocations (%llu bytes) still alive at shutdown", stats_[c].count,
                     memoryCategoryName(static_cast<MemoryCategory>(c)),
                     static_cast<unsigned long long>(stats_[c].bytes));
        }
    }
}

void GpuMemory::account(MTL::Resource* resource, MemoryCategory category) {
    CategoryStats& s = stats_[static_cast<u32>(category)];
    s.bytes += resource->allocatedSize();
    ++s.count;
    ++allocationCount_;
}

void GpuMemory::unaccount(MTL::Resource* resource, MemoryCategory category) {
    CategoryStats& s = stats_[static_cast<u32>(category)];
    s.bytes -= resource->allocatedSize();
    --s.count;
}

MTL::Buffer* GpuMemory::newBuffer(u64 length, MTL::ResourceOptions options, MemoryCategory category,
                                  const char* label) {
    MTL::Buffer* buffer = context_.device()->newBuffer(length, options);
    if (!buffer) {
        LOG_ERROR("GpuMemory: failed to allocate %llu-byte buffer '%s'", static_cast<unsigned long long>(length),
                  label);
        return nullptr;
    }
    buffer->setLabel(str(label));
    context_.makeResident(buffer);
    account(buffer, category);
    return buffer;
}

MTL::Texture* GpuMemory::newTexture(const MTL::TextureDescriptor* descriptor, MemoryCategory category,
                                    const char* label) {
    MTL::Texture* texture = context_.device()->newTexture(descriptor);
    if (!texture) {
        LOG_ERROR("GpuMemory: failed to allocate %lux%lu texture '%s'",
                  static_cast<unsigned long>(descriptor->width()), static_cast<unsigned long>(descriptor->height()),
                  label);
        return nullptr;
    }
    texture->setLabel(str(label));
    // Memoryless textures live in tile memory only: nothing to make resident.
    if (!isMemoryless(texture)) context_.makeResident(texture);
    account(texture, category);
    return texture;
}

void GpuMemory::release(MTL::Resource* resource, MemoryCategory category) {
    if (!resource) return;
    unaccount(resource, category);
    context_.deferRelease(resource, /*evict*/ !isMemoryless(resource));
}

void GpuMemory::releaseNow(MTL::Resource* resource, MemoryCategory category) {
    if (!resource) return;
    unaccount(resource, category);
    if (!isMemoryless(resource)) context_.evict(resource);
    resource->release();
}

u64 GpuMemory::totalBytes() const {
    u64 total = 0;
    for (const CategoryStats& s : stats_) total += s.bytes;
    return total;
}

} // namespace phosphor
