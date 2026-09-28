#include "scene/texture_manager.h"
#include "core/log.h"

// Declarations only: the stb_image implementation is compiled in gltf_loader.cpp
// together with tinygltf.
#include <stb_image.h>

#include <vector>

namespace phosphor {

u32 TextureManager::loadTexture(const std::string& path, bool sRGB) {
    auto it = loadedPaths_.find(path);
    if (it != loadedPaths_.end()) {
        return it->second;
    }

    int w = 0, h = 0, channels = 0;
    stbi_uc* pixels = stbi_load(path.c_str(), &w, &h, &channels, 4);
    if (!pixels) {
        LOG_ERROR("Failed to load texture: %s (%s)", path.c_str(), stbi_failure_reason());
        return defaultWhite_;
    }

    const u32 index = createTexture(pixels, static_cast<u32>(w), static_cast<u32>(h), sRGB);
    stbi_image_free(pixels);

    loadedPaths_[path] = index;
    LOG_INFO("Loaded texture: %s (%dx%d, %d ch) -> bindless %u", path.c_str(), w, h, channels, index);
    return index;
}

u32 TextureManager::loadTextureFromMemory(const u8* data, u32 width, u32 height,
                                          u32 components, bool sRGB) {
    if (!data || width == 0 || height == 0 || components == 0 || components > 4) {
        LOG_WARN("loadTextureFromMemory: invalid parameters");
        return defaultWhite_;
    }
    if (components == 4) {
        return createTexture(data, width, height, sRGB);
    }

    std::vector<u8> rgba(static_cast<size_t>(width) * height * 4);
    for (size_t i = 0; i < static_cast<size_t>(width) * height; ++i) {
        const u8* src = data + i * components;
        u8* dst = rgba.data() + i * 4;
        const bool hasColor = components >= 3;
        dst[0] = src[0];
        dst[1] = hasColor ? src[1] : src[0];
        dst[2] = hasColor ? src[2] : src[0];
        dst[3] = components == 2 ? src[1] : 255; // luminance + alpha
    }
    return createTexture(rgba.data(), width, height, sRGB);
}

void TextureManager::createDefaultTextures() {
    if (defaultsCreated_) return;
    const u8 white[]  = {255, 255, 255, 255};
    const u8 normal[] = {128, 128, 255, 255};
    const u8 black[]  = {0, 0, 0, 255};
    // glTF multiplies the metallic/roughness factors by the texture (G = roughness,
    // B = metallic): the neutral default is white so the factors pass through.
    const u8 mr[]     = {255, 255, 255, 255};
    defaultWhite_  = createTexture(white, 1, 1, true);
    defaultNormal_ = createTexture(normal, 1, 1, false);
    defaultBlack_  = createTexture(black, 1, 1, true);
    defaultMR_     = createTexture(mr, 1, 1, false);
    defaultsCreated_ = true;
}

} // namespace phosphor
