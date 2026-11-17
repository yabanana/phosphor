#pragma once
#include "renderer/gpu_types.h"
namespace phosphor {
struct GPUTemporalParams {
    float currentViewProjection[16];  // unjittered
    float previousViewProjection[16]; // unjittered, same view
    float renderSize[2];
    float previousRenderSize[2];
    float jitter[2]; // raster displacement in pixels (+right, +down); MetalFX adapter validates the mapping
    float deltaTime;
    u32 historyValid;
    u32 viewIndex;
    u32 frameIndex;
    float mipBias;
    float manualExposure;
    u32 debugFlags, pad0;
    GPUMaterialExtPtr materialExtensions; // F15.4 (ticket 23, texxform): the extension buffer for the alpha tests of layer stacks and per-slot transforms; 0 without one
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUTemporalParams) == 192, "temporal parameters layout");
} // namespace phosphor
