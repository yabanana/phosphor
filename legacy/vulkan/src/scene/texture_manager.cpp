#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "scene/texture_manager.h"
#include "rhi/vk_device.h"
#include "rhi/vk_allocator.h"
#include "rhi/vk_descriptors.h"
#include "rhi/vk_commands.h"
#include "core/log.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace phosphor {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static u32 computeMipLevels(u32 width, u32 height) {
    return static_cast<u32>(std::floor(std::log2(std::max(width, height)))) + 1;
}

static VkFormat pickFormat(u32 components, bool sRGB) {
    // Always upload as RGBA; expand data if needed before upload.
    // This keeps things simple and avoids format compatibility issues.
    (void)components;
    return sRGB ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
}

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------

TextureManager::TextureManager(VulkanDevice& device, GpuAllocator& allocator,
                               BindlessDescriptorManager& descriptors,
                               CommandManager& commands)
    : device_(device)
    , allocator_(allocator)
    , descriptors_(descriptors)
    , commands_(commands)
{
    LOG_INFO("TextureManager created");
}

TextureManager::~TextureManager() {
    VkDevice dev = device_.getDevice();
    vkDeviceWaitIdle(dev);

    size_t count = textures_.size();

    for (auto view : views_) {
        allocator_.destroyImageView(view);
    }
    for (auto& img : textures_) {
        allocator_.destroyImage(img);
    }
    views_.clear();
    textures_.clear();
    loadedPaths_.clear();

    LOG_INFO("TextureManager destroyed (%zu textures freed)", count);
}

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

u32 TextureManager::loadTexture(const std::string& path, bool sRGB) {
    // Dedup: return cached bindless index if already loaded
    auto it = loadedPaths_.find(path);
    if (it != loadedPaths_.end()) {
        return it->second;
    }

    int w = 0, h = 0, channels = 0;
    // Force 4 components (RGBA) for uniform handling
    stbi_uc* pixels = stbi_load(path.c_str(), &w, &h, &channels, 4);
    if (!pixels) {
        LOG_ERROR("Failed to load texture: %s (%s)", path.c_str(), stbi_failure_reason());
        return defaultWhite_;
    }

    u32 bindlessIndex = uploadTexture(pixels, static_cast<u32>(w),
                                      static_cast<u32>(h), 4, sRGB);
    stbi_image_free(pixels);

    loadedPaths_[path] = bindlessIndex;
    LOG_INFO("Loaded texture: %s (%dx%d, %d ch) -> bindless %u",
             path.c_str(), w, h, channels, bindlessIndex);
    return bindlessIndex;
}

u32 TextureManager::loadTextureFromMemory(const u8* data, u32 width, u32 height,
                                           u32 components, bool sRGB) {
    if (!data || width == 0 || height == 0) {
        LOG_WARN("loadTextureFromMemory: invalid parameters");
        return defaultWhite_;
    }

    // Expand to RGBA if fewer than 4 components
    std::vector<u8> rgba;
    const u8* uploadData = data;

    if (components < 4) {
        rgba.resize(static_cast<size_t>(width) * height * 4);
        for (u32 i = 0; i < width * height; ++i) {
            const u8* src = data + i * components;
            u8* dst = rgba.data() + i * 4;
            dst[0] = (components >= 1) ? src[0] : 0;
            dst[1] = (components >= 2) ? src[1] : 0;
            dst[2] = (components >= 3) ? src[2] : 0;
            dst[3] = (components >= 4) ? src[3] : 255;
        }
        uploadData = rgba.data();
    }

    return uploadTexture(uploadData, width, height, 4, sRGB);
}

void TextureManager::createDefaultTextures() {
    // 1x1 white (base color fallback)
    {
        const u8 px[] = { 255, 255, 255, 255 };
        defaultWhite_ = uploadTexture(px, 1, 1, 4, true);
    }
    // 1x1 flat normal (tangent-space: (0,0,1) encoded as (128,128,255))
    {
        const u8 px[] = { 128, 128, 255, 255 };
        defaultNormal_ = uploadTexture(px, 1, 1, 4, false);
    }
    // 1x1 black (emissive / occlusion fallback)
    {
        const u8 px[] = { 0, 0, 0, 255 };
        defaultBlack_ = uploadTexture(px, 1, 1, 4, true);
    }
    // 1x1 metallic-roughness default: green channel=roughness=0.5 (128),
    // blue channel=metallic=0.0 (0). glTF convention: R=occlusion, G=roughness, B=metallic
    {
        const u8 px[] = { 0, 128, 0, 255 };
        defaultMR_ = uploadTexture(px, 1, 1, 4, false);
    }

    LOG_INFO("Default textures created (white=%u, normal=%u, black=%u, MR=%u)",
             defaultWhite_, defaultNormal_, defaultBlack_, defaultMR_);
}

// ---------------------------------------------------------------------------
// Upload pipeline: staging -> copy -> mipmaps -> shader read -> bindless
// ---------------------------------------------------------------------------

u32 TextureManager::uploadTexture(const u8* pixels, u32 width, u32 height,
                                   u32 components, bool sRGB) {
    VkFormat format = pickFormat(components, sRGB);
    u32 mipLevels = computeMipLevels(width, height);
    VkDeviceSize imageSize = static_cast<VkDeviceSize>(width) * height * 4;

    // --- Create staging buffer ---
    AllocatedBuffer staging = allocator_.createBuffer(
        imageSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_CPU_ONLY,
        VMA_ALLOCATION_CREATE_MAPPED_BIT);

    void* mapped = allocator_.mapMemory(staging);
    std::memcpy(mapped, pixels, imageSize);
    allocator_.unmapMemory(staging);

    // --- Create the GPU image ---
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = format;
    imageInfo.extent = { width, height, 1 };
    imageInfo.mipLevels = mipLevels;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                    | VK_IMAGE_USAGE_TRANSFER_DST_BIT
                    | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    AllocatedImage image = allocator_.createImage(imageInfo, VMA_MEMORY_USAGE_GPU_ONLY);

    // --- Copy staging -> image, generate mipmaps, transition to shader read ---
    commands_.immediateSubmit([&](VkCommandBuffer cmd) {
        // Transition mip 0 to TRANSFER_DST
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image.image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = mipLevels;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);

        // Copy staging buffer to mip level 0
        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = { 0, 0, 0 };
        region.imageExtent = { width, height, 1 };

        vkCmdCopyBufferToImage(cmd, staging.buffer, image.image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               1, &region);

        // Generate mipmaps (also transitions all levels to SHADER_READ_ONLY)
        generateMipmaps(cmd, image.image, format, width, height, mipLevels);
    });

    allocator_.destroyBuffer(staging);

    // --- Create image view ---
    VkImageView view = allocator_.createImageView(
        image.image, format, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels);

    // --- Register in bindless descriptor set ---
    u32 bindlessIndex = descriptors_.registerTexture(view);

    textures_.push_back(image);
    views_.push_back(view);

    return bindlessIndex;
}

// ---------------------------------------------------------------------------
// Mipmap generation via vkCmdBlitImage
// ---------------------------------------------------------------------------

void TextureManager::generateMipmaps(VkCommandBuffer cmd, VkImage image,
                                      VkFormat format, u32 width, u32 height,
                                      u32 mipLevels) {
    // Each iteration:
    //  1. Transition current mip from TRANSFER_DST to TRANSFER_SRC
    //  2. Blit current mip into next mip (which is already TRANSFER_DST)
    //  3. After loop, transition all mips to SHADER_READ_ONLY

    i32 mipWidth  = static_cast<i32>(width);
    i32 mipHeight = static_cast<i32>(height);

    for (u32 i = 1; i < mipLevels; ++i) {
        // Transition mip (i-1) from TRANSFER_DST to TRANSFER_SRC
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = i - 1;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);

        // Blit from mip (i-1) to mip (i)
        VkImageBlit blit{};
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.mipLevel = i - 1;
        blit.srcSubresource.baseArrayLayer = 0;
        blit.srcSubresource.layerCount = 1;
        blit.srcOffsets[0] = { 0, 0, 0 };
        blit.srcOffsets[1] = { mipWidth, mipHeight, 1 };

        i32 nextWidth  = (mipWidth  > 1) ? (mipWidth  / 2) : 1;
        i32 nextHeight = (mipHeight > 1) ? (mipHeight / 2) : 1;

        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.mipLevel = i;
        blit.dstSubresource.baseArrayLayer = 0;
        blit.dstSubresource.layerCount = 1;
        blit.dstOffsets[0] = { 0, 0, 0 };
        blit.dstOffsets[1] = { nextWidth, nextHeight, 1 };

        vkCmdBlitImage(cmd,
            image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &blit, VK_FILTER_LINEAR);

        mipWidth  = nextWidth;
        mipHeight = nextHeight;
    }

    // Transition the last mip from TRANSFER_DST to TRANSFER_SRC
    // (so the final barrier can handle all mips uniformly)
    {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = mipLevels - 1;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);
    }

    // Transition ALL mip levels from TRANSFER_SRC to SHADER_READ_ONLY
    {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = mipLevels;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);
    }
}

} // namespace phosphor
