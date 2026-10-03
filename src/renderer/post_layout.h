#pragma once
#include "renderer/gpu_types.h"
namespace phosphor {
struct GPUPostParams {
    u32 inputWidth, inputHeight, outputWidth, outputHeight;
    u32 tonemap, autoExposure, historyReset, temporal;
    float deltaTime, manualExposure, adaptSpeed, sharpening;
    float headroom, minExposure, maxExposure, whitePoint;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUPostParams) == 64, "post parameters");
} // namespace phosphor
