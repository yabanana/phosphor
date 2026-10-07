#pragma once

#include "scene/texture_manager.h"
#include "renderer/rt_check.h"
#include "renderer/offline_reference.h"

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
// Uploads are batched: createTexture() copies pixels into the context's
// staging ring and queues the blit; flushUploads() generates the mipmaps and
// runs everything in one command buffer.  A full staging ring flushes the
// copies early; mip generation always follows at flushUploads().
// ---------------------------------------------------------------------------

class MetalTextureManager final : public TextureManager {
public:
    static constexpr u32 MAX_TEXTURES = 16384;

    explicit MetalTextureManager(MetalContext& context, bool keepCpuTextures = false);
    ~MetalTextureManager() override;

    /// Upload all textures created since the last flush (blocking).
    void flushUploads();

    [[nodiscard]] u32 textureCount() const override { return static_cast<u32>(textures_.size()); }

    /// GPU address of the MTLResourceID table (bind at buffer index 5).
    [[nodiscard]] MTL::GPUAddress tableAddress() const { return table_->gpuAddress(); }
    [[nodiscard]] MTL::Buffer* tableBuffer() const { return table_; } // F9 intersection-function table binding

    /// Present only for explicitly enabled diagnostics. Indexed exactly like
    /// the GPU bindless table; flushUploads() reads back generated mip bytes.
    [[nodiscard]] std::span<const RtCpuTexture> cpuTextures() const { return cpuTextures_; }
    [[nodiscard]] std::vector<ReferenceTexture> referenceTextures() const;

protected:
    u32 createTexture(const u8* rgba, u32 width, u32 height, bool sRGB) override;

private:
    void readbackCpuMips(); // Loading time only, never on the normal render path.
    bool keepCpuTextures_ = false;
    size_t cpuMipReadbackCount_ = 0;
    std::vector<RtCpuTexture> cpuTextures_;
    std::vector<bool> srgb_;
    MetalContext&              context_;
    MTL::Buffer*               table_ = nullptr;
    std::vector<MTL::Texture*> textures_;
    std::vector<MTL::Texture*> pendingMips_; // copied (or queued), mips not generated yet
};

} // namespace phosphor
