#include "platform/metal/metal_texture_manager.h"
#include "platform/metal/metal_context.h"
#include "core/log.h"

#include <algorithm>
#include <bit>
#include <cstring>

namespace phosphor {

MetalTextureManager::MetalTextureManager(MetalContext& context) : context_(context) {
    table_ = context_.device()->newBuffer(sizeof(MTL::ResourceID) * MAX_TEXTURES,
                                          MTL::ResourceStorageModeShared | MTL::ResourceCPUCacheModeWriteCombined);
    table_->setLabel(NS::String::string("Texture table", NS::UTF8StringEncoding));
    std::memset(table_->contents(), 0, table_->length());
    context_.makeResident(table_);
}

MetalTextureManager::~MetalTextureManager() {
    context_.waitIdle();
    for (PendingUpload& p : pending_) {
        p.staging->release();
    }
    for (MTL::Texture* tex : textures_) {
        context_.evict(tex);
        tex->release();
    }
    context_.evict(table_);
    table_->release();
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

    MTL::Texture* texture = context_.device()->newTexture(desc);
    if (!texture) {
        LOG_ERROR("Failed to create %ux%u texture", width, height);
        return getDefaultWhite();
    }

    const size_t bytes = static_cast<size_t>(width) * height * 4;
    MTL::Buffer* staging = context_.device()->newBuffer(bytes, MTL::ResourceStorageModeShared);
    std::memcpy(staging->contents(), rgba, bytes);
    context_.makeResident(staging);
    context_.makeResident(texture);

    const u32 index = static_cast<u32>(textures_.size());
    textures_.push_back(texture);
    static_cast<MTL::ResourceID*>(table_->contents())[index] = texture->gpuResourceID();

    pending_.push_back({texture, staging, width, height});
    return index;
}

void MetalTextureManager::flushUploads() {
    if (pending_.empty()) return;

    context_.submitAndWait([this](MTL4::ComputeCommandEncoder* enc) {
        for (const PendingUpload& p : pending_) {
            enc->copyFromBuffer(p.staging, 0, p.width * 4, 0,
                                MTL::Size::Make(p.width, p.height, 1),
                                p.texture, 0, 0, MTL::Origin::Make(0, 0, 0));
        }
        // Metal 4 does not track hazards: order the copies before mip generation.
        enc->barrierAfterEncoderStages(MTL::StageBlit, MTL::StageBlit, MTL4::VisibilityOptionDevice);
        for (const PendingUpload& p : pending_) {
            if (p.texture->mipmapLevelCount() > 1) {
                enc->generateMipmaps(p.texture);
            }
        }
    });

    for (PendingUpload& p : pending_) {
        context_.evict(p.staging);
        p.staging->release();
    }
    LOG_INFO("Uploaded %zu textures", pending_.size());
    pending_.clear();
}

} // namespace phosphor
