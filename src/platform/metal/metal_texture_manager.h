#pragma once

#include "scene/texture_manager.h"

#include <Metal/Metal.hpp>
#include <vector>

namespace phosphor {

class MetalContext;

// ---------------------------------------------------------------------------
// MetalTextureManager -- bindless textures on Metal 4.
//
// Every texture is private (GPU-optimal layout, lossless compression on
// Apple GPUs) and gets a slot in a shared "texture table" buffer holding its
// MTLResourceID; shaders index that table (argument buffers tier 2).
// Uploads are batched: createTexture() stages pixels, flushUploads() copies
// them and generates mipmaps in one command buffer.
// ---------------------------------------------------------------------------

class MetalTextureManager final : public TextureManager {
public:
    static constexpr u32 MAX_TEXTURES = 16384;

    explicit MetalTextureManager(MetalContext& context);
    ~MetalTextureManager() override;

    /// Upload all textures created since the last flush (blocking).
    void flushUploads();

    [[nodiscard]] u32 textureCount() const override { return static_cast<u32>(textures_.size()); }

    /// GPU address of the MTLResourceID table (bind at buffer index 5).
    [[nodiscard]] MTL::GPUAddress tableAddress() const { return table_->gpuAddress(); }

protected:
    u32 createTexture(const u8* rgba, u32 width, u32 height, bool sRGB) override;

private:
    struct PendingUpload {
        MTL::Texture* texture;
        MTL::Buffer*  staging;
        u32 width;
        u32 height;
    };

    MetalContext&              context_;
    MTL::Buffer*               table_ = nullptr;
    std::vector<MTL::Texture*> textures_;
    std::vector<PendingUpload> pending_;
};

} // namespace phosphor
