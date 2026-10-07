#pragma once
#include "renderer/gpu_types.h"
namespace phosphor {
PHOSPHOR_GPU_CONSTANT u32 FX_FIXTURE_CONSTANT=0u;
PHOSPHOR_GPU_CONSTANT u32 FX_FIXTURE_IMPULSE=1u;
PHOSPHOR_GPU_CONSTANT u32 FX_FIXTURE_CHANNELS=2u;
PHOSPHOR_GPU_CONSTANT u32 FX_FIXTURE_LIFECYCLE=3u;
PHOSPHOR_GPU_CONSTANT u32 FX_FIXTURE_WIDE_HDR=4u;
struct GPUFXFixtureParams {
    u32 width,height,outputWidth,outputHeight;
    u32 scenario,phase,frame,flags;
    float color[3],preExposure;
    float nearPlane,planeDistance,motionPixels,impulseAmplitude;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUFXFixtureParams)==64,"SDK fixture parameters");
// Twelve scalar words/sample. The channel readback samples actual generated
// color, normal, roughness, motion, depth, albedo and flags; SDK image readbacks
// remain separate full-frame arrays, so an authored guide is not native output.
struct GPUFXFixtureSample {
    float color[3],roughness;
    float normal[3],depth;
    float motion[2],diffuseR,specularR;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUFXFixtureSample)==48,"SDK fixture sample");
} // namespace phosphor
