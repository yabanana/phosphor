#pragma once

#include "scene/texture_manager.h"
#include <vector>

namespace phosphor::test {

/// TextureManager that records uploads instead of talking to a GPU.
class NullTextureManager final : public TextureManager {
public:
    struct Upload {
        u32 width, height;
        bool sRGB;
        std::vector<u8> rgba;
    };

    [[nodiscard]] u32 textureCount() const override { return static_cast<u32>(uploads.size()); }

    std::vector<Upload> uploads;

protected:
    u32 createTexture(const u8* rgba, u32 width, u32 height, bool sRGB) override {
        uploads.push_back({width, height, sRGB,
                           std::vector<u8>(rgba, rgba + static_cast<size_t>(width) * height * 4)});
        return static_cast<u32>(uploads.size() - 1);
    }
};

} // namespace phosphor::test
