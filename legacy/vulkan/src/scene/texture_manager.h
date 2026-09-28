#pragma once

#include "core/types.h"
#include "rhi/vk_common.h"
#include "rhi/vk_allocator.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace phosphor {

class VulkanDevice;
class GpuAllocator;
class BindlessDescriptorManager;
class CommandManager;

class TextureManager {
public:
    TextureManager(VulkanDevice& device, GpuAllocator& allocator,
                   BindlessDescriptorManager& descriptors, CommandManager& commands);
    ~TextureManager();

    TextureManager(const TextureManager&) = delete;
    TextureManager& operator=(const TextureManager&) = delete;

    /// Load a texture from disk. Returns a bindless descriptor index.
    /// Deduplicates by path: loading the same path twice returns the cached index.
    u32 loadTexture(const std::string& path, bool sRGB = true);

    /// Load a texture from raw pixel data already in memory.
    /// @param data       Pointer to tightly packed RGBA/RGB/RG/R pixel data
    /// @param width      Image width in pixels
    /// @param height     Image height in pixels
    /// @param components Number of components per pixel (1-4)
    /// @param sRGB       Whether to use sRGB format
    /// @return Bindless descriptor index
    u32 loadTextureFromMemory(const u8* data, u32 width, u32 height, u32 components, bool sRGB = true);

    /// Create the four default 1x1 textures used as fallbacks.
    ///   - white:  (255, 255, 255, 255) for base color
    ///   - normal: (128, 128, 255, 255) for flat tangent-space normal
    ///   - black:  (0, 0, 0, 255) for emissive / occlusion
    ///   - MR:     (0, 128, 0, 255) for metallic=0.0, roughness=0.5
    void createDefaultTextures();

    u32 getDefaultWhite()  const { return defaultWhite_; }
    u32 getDefaultNormal() const { return defaultNormal_; }
    u32 getDefaultBlack()  const { return defaultBlack_; }
    u32 getDefaultMR()     const { return defaultMR_; }

private:
    u32 uploadTexture(const u8* pixels, u32 width, u32 height, u32 components, bool sRGB);
    void generateMipmaps(VkCommandBuffer cmd, VkImage image, VkFormat format,
                         u32 width, u32 height, u32 mipLevels);

    VulkanDevice&              device_;
    GpuAllocator&              allocator_;
    BindlessDescriptorManager& descriptors_;
    CommandManager&            commands_;

    std::vector<AllocatedImage> textures_;
    std::vector<VkImageView>    views_;
    std::unordered_map<std::string, u32> loadedPaths_; // path -> bindless index (dedup)

    u32 defaultWhite_  = 0;
    u32 defaultNormal_ = 0;
    u32 defaultBlack_  = 0;
    u32 defaultMR_     = 0;
};

} // namespace phosphor
