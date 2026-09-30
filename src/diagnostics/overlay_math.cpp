#include "diagnostics/overlay_math.h"

#include <algorithm>

namespace phosphor::overlay {

std::vector<float> tileCostReference(const float* overdraw, const float* lights, u32 width, u32 height) {
    const u32 tilesX = tileCount(width);
    const u32 tilesY = tileCount(height);
    std::vector<float> tiles(static_cast<size_t>(tilesX) * tilesY, 0.0f);
    for (u32 ty = 0; ty < tilesY; ++ty) {
        for (u32 tx = 0; tx < tilesX; ++tx) {
            float sum   = 0.0f;
            float count = 0.0f;
            const u32 y1 = std::min((ty + 1) * TILE_SIZE, height);
            const u32 x1 = std::min((tx + 1) * TILE_SIZE, width);
            for (u32 y = ty * TILE_SIZE; y < y1; ++y) {
                for (u32 x = tx * TILE_SIZE; x < x1; ++x) {
                    const size_t i = static_cast<size_t>(y) * width + x;
                    sum += overdraw[i] * (1.0f + std::max(lights[i], 0.0f));
                    count += 1.0f;
                }
            }
            tiles[static_cast<size_t>(ty) * tilesX + tx] = count > 0.0f ? sum / count : 0.0f;
        }
    }
    return tiles;
}

} // namespace phosphor::overlay
