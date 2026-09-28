#pragma once

#include "core/types.h"
#include <string>
#include <unordered_map>

namespace phosphor {

// ---------------------------------------------------------------------------
// TextureManager -- API-agnostic front end for bindless textures.
//
// Decoding, RGBA expansion, path de-duplication and the default textures live
// here; a graphics backend derives from it and implements createTexture(),
// which uploads tightly packed RGBA8 pixels and returns the bindless index.
// ---------------------------------------------------------------------------

class TextureManager {
public:
    virtual ~TextureManager() = default;

    TextureManager(const TextureManager&) = delete;
    TextureManager& operator=(const TextureManager&) = delete;

    /// Load a texture from disk. Returns a bindless index.
    /// Loading the same path twice returns the cached index.
    u32 loadTexture(const std::string& path, bool sRGB = true);

    /// Load a texture from pixel data already in memory (1-4 components).
    u32 loadTextureFromMemory(const u8* data, u32 width, u32 height, u32 components, bool sRGB = true);

    /// Create the four default 1x1 textures used as fallbacks (idempotent).
    ///   - white:  (255, 255, 255, 255) for base color / occlusion
    ///   - normal: (128, 128, 255, 255) for a flat tangent-space normal
    ///   - black:  (0, 0, 0, 255) for emissive
    ///   - MR:     (0, 128, 0, 255) for metallic = 0, roughness = 0.5
    void createDefaultTextures();

    u32 getDefaultWhite()  const { return defaultWhite_; }
    u32 getDefaultNormal() const { return defaultNormal_; }
    u32 getDefaultBlack()  const { return defaultBlack_; }
    u32 getDefaultMR()     const { return defaultMR_; }

    /// Number of textures created so far.
    [[nodiscard]] virtual u32 textureCount() const = 0;

protected:
    TextureManager() = default;

    /// Upload width * height RGBA8 pixels; return the bindless index.
    virtual u32 createTexture(const u8* rgba, u32 width, u32 height, bool sRGB) = 0;

private:
    std::unordered_map<std::string, u32> loadedPaths_;
    bool defaultsCreated_ = false;
    u32 defaultWhite_  = 0;
    u32 defaultNormal_ = 0;
    u32 defaultBlack_  = 0;
    u32 defaultMR_     = 0;
};

} // namespace phosphor
