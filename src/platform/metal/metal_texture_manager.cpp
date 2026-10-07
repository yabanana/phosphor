#include "platform/metal/metal_texture_manager.h"
#include "platform/metal/gpu_memory.h"
#include "platform/metal/metal_context.h"
#include "core/log.h"

#include <cstring>
#include <algorithm>
#include <stdexcept>

namespace phosphor {

MetalTextureManager::MetalTextureManager(MetalContext& context, bool keepCpuTextures)
    : keepCpuTextures_(keepCpuTextures), context_(context) {
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
    if (keepCpuTextures_) cpuTextures_.push_back(rtMakeCpuTexture(std::span(rgba, rowBytes * height), width, height));
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
    if (keepCpuTextures_) readbackCpuMips();
    LOG_INFO("Texture uploads flushed (%zu textures)", textures_.size());
}


void MetalTextureManager::readbackCpuMips() {
    if (cpuMipReadbackCount_ == textures_.size()) return;
    struct Copy { size_t offset, rowBytes; u32 width, height, level; };
    const auto layout = [](MTL::Texture* texture) {
        std::vector<Copy> copies;
        size_t offset = 0;
        for (u32 level = 1; level < texture->mipmapLevelCount(); ++level) {
            const auto width = std::max(1u, u32(texture->width()) >> level);
            const auto height = std::max(1u, u32(texture->height()) >> level);
            const size_t rowBytes = (size_t(width) * 4 + 255) & ~size_t(255);
            copies.push_back({offset, rowBytes, width, height, level});
            offset += rowBytes * height;
        }
        return copies;
    };
    // One reusable buffer bounds diagnostic GPU memory to the largest texture's
    // mip chain. All submissions below complete before this buffer is reused.
    size_t maxBytes = 0;
    for (size_t i = cpuMipReadbackCount_; i < textures_.size(); ++i) {
        const auto copies = layout(textures_[i]);
        if (!copies.empty()) maxBytes = std::max(maxBytes, copies.back().offset + copies.back().rowBytes * copies.back().height);
    }
    MTL::Buffer* readback = maxBytes ? context_.memory().newBuffer(maxBytes, MTL::ResourceStorageModeShared,
        MemoryCategory::Other, "RT diagnostic alpha mip readback") : nullptr;
    if (maxBytes && !readback) throw std::runtime_error("RT alpha mip readback allocation failed");
    struct ReadbackRelease {
        GpuMemory& memory;
        MTL::Buffer* buffer;
        ~ReadbackRelease() { if (buffer) memory.release(buffer, MemoryCategory::Other); }
    } release{context_.memory(), readback};
    for (size_t i = cpuMipReadbackCount_; i < textures_.size(); ++i) {
        auto* texture = textures_[i];
        const auto copies = layout(texture);
        if (!copies.empty()) {
            context_.submitAndWait([texture, readback, &copies](MTL4::ComputeCommandEncoder* enc) {
                // Prior upload/mipmap submission completed synchronously.
                for (const auto& copy : copies) {
                    enc->copyFromTexture(texture, 0, copy.level, MTL::Origin::Make(0, 0, 0),
                        MTL::Size::Make(copy.width, copy.height, 1), readback, copy.offset, copy.rowBytes, 0);
                }
            });
            auto& cpu = cpuTextures_[i];
            for (const auto& copy : copies) {
                RtCpuMip mip{copy.width, copy.height, std::vector<u8>(size_t(copy.width) * copy.height * 4)};
                const auto* source = static_cast<const u8*>(readback->contents()) + copy.offset;
                for (u32 y = 0; y < copy.height; ++y)
                    std::memcpy(mip.rgba8.data() + size_t(y) * copy.width * 4, source + y * copy.rowBytes, size_t(copy.width) * 4);
                cpu.mips.push_back(std::move(mip));
            }
        }
        cpuTextures_[i].exactMips = true;
    }
    cpuMipReadbackCount_ = textures_.size();
}

} // namespace phosphor
