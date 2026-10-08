#pragma once
#include "renderer/gpu_types.h"
namespace phosphor {
PHOSPHOR_GPU_CONSTANT u32 METALFX_PACK_HIT_DISTANCE = 1u;
PHOSPHOR_GPU_CONSTANT u32 METALFX_PACK_REACTIVE = 2u;
PHOSPHOR_GPU_CONSTANT u32 METALFX_PACK_STRENGTH = 4u;
struct GPUMetalfxDenoisePackParams {
    u32 width, height, flags, pad;
    float exposureNormalization, normalTolerance, colorScale, padFloat;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUMetalfxDenoisePackParams)==32,"MetalFX guide pack parameters");
struct GPUMetalfxDenoisePackCounters {
    u32 pixels, colorErrors, normalErrors, albedoErrors;
    u32 roughnessErrors, motionErrors, hitErrors, maskErrors;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUMetalfxDenoisePackCounters)==32,"MetalFX guide pack counters");
struct GPUMetalfxRestoreParams {
    u32 width,height;float inversePreExposure;u32 pad;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUMetalfxRestoreParams)==16,"MetalFX physical radiance restore");
} // namespace phosphor
