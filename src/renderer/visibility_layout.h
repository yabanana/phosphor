#pragma once
#include "renderer/gpu_types.h"

namespace phosphor {
struct GPUShadingHistory {
    u32 generation, primitive;
    float luminance, roughness;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUShadingHistory) == 16, "adaptive shading history layout");
PHOSPHOR_GPU_CONSTANT u32 VISIBILITY_TILE = 16;
PHOSPHOR_GPU_CONSTANT u32 VISIBILITY_CLASSES = 4;
struct GPUVisibilityParams {
    u32 width, height, tilesX, tilesY;
    u32 candidateCapacity, materialCount, defaultNormal, debugMode;
    float exposure, mipBias;
    u32 binning, pad;
    u32 outputWidth, outputHeight, pad2, pad3;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUVisibilityParams) == 64, "visibility parameters");
} // namespace phosphor
