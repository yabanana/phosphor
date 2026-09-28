#include "platform/metal/metal_texture_manager.h"
#include "platform/metal/gpu_memory.h"
#include "platform/metal/metal_context.h"
#include "core/log.h"

#include <cstring>

namespace phosphor {

MetalTextureManager::MetalTextureManager(MetalContext& context) : context_(context) {
    table_ = context_.memory().newBuffer(sizeof(MTL::ResourceID) * MAX_TEXTURES,
                                         MTL::ResourceStorageModeShared | MTL::ResourceCPUCacheModeWriteCombined,
                                         MemoryCategory::Textures, "Texture table");
    std::memset(table_->contents(), 0, table_->length());
}

MetalTextureManager::~MetalTextureManager() {
    // Queued copies may reference these textures: run them before releasing.
    context_.flushUploads();
    for (MTL::Texture* tex : textures_) {
        context_.memory().release(tex, MemoryCategory::Textures);
    }
    context_.memory().release(table_, MemoryCategory::Textures);
}

u32 MetalTextureManager::createTexture(const u8* rgba, u32 width, u32 height, bool sRGB) {
    if (textures_.size() >= MAX_TEXTURES) {
        LOG_ERROR("Bindless texture table full (%u)", MAX_TEXTURES);
        return getDefaultWhite();
    }

    const bool mipmapped = width > 1 || height > 1;
    MTL::TextureDescriptor* desc = MTL::TextureDescriptor::texture2DDescriptor(
        sRGB ? MTL::PixelFormatRGBA8Unorm_sRGB : MTL::PixelFormatRGBA8Unorm,
        width, height, mipmapped);
    desc->setUsage(MTL::TextureUsageShaderRead);
    desc->setStorageMode(MTL::StorageModePrivate);

    MTL::Texture* texture = context_.memory().newTexture(desc, MemoryCategory::Textures, "Material texture");
    if (!texture) {
        return getDefaultWhite();
    }

    const size_t rowBytes = static_cast<size_t>(width) * 4;
    const UploadRing::Slice staging = context_.stagingAllocate(rowBytes * height);
    std::memcpy(staging.cpu, rgba, rowBytes * height);
    // Blit upload: the private texture stays eligible for lossless compression (O6).
    context_.enqueueUpload([staging, rowBytes, width, height, texture](MTL4::ComputeCommandEncoder* enc) {
        enc->copyFromBuffer(staging.buffer, staging.offset, rowBytes, 0, MTL::Size::Make(width, height, 1),
                            texture, 0, 0, MTL::Origin::Make(0, 0, 0));
    });

    const u32 index = static_cast<u32>(textures_.size());
    textures_.push_back(texture);
    static_cast<MTL::ResourceID*>(table_->contents())[index] = texture->gpuResourceID();
    if (texture->mipmapLevelCount() > 1) pendingMips_.push_back(texture);
    return index;
}

void MetalTextureManager::flushUploads() {
    if (!pendingMips_.empty()) {
        // Queued after every copy; a staging flush may already have run some
        // copies, in which case the barrier is merely redundant.
        context_.enqueueUpload([mips = pendingMips_](MTL4::ComputeCommandEncoder* enc) {
            // Metal 4 does not track hazards: order the copies before mip generation.
            enc->barrierAfterEncoderStages(MTL::StageBlit, MTL::StageBlit, MTL4::VisibilityOptionDevice);
            for (MTL::Texture* texture : mips) enc->generateMipmaps(texture);
        });
    }
    context_.flushUploads();
    pendingMips_.clear();
    LOG_INFO("Texture uploads flushed (%zu textures)", textures_.size());
}

} // namespace phosphor
