#pragma once
#include "renderer/gpu_types.h"
#include <glm/glm.hpp>
namespace phosphor {
struct DenoiseSettings {
    float temporalAlpha=.08f, momentsAlpha=.2f, depthThreshold=.02f, normalThreshold=.95f;
    float roughnessThreshold=.08f, hitDistanceThreshold=.1f, luminancePhi=4, normalPhi=64;
    float varianceFloor=1e-6f, clampSigma=2, planeThreshold=.02f;
    u32 maxHistory=32, atrousIterations=3;
    bool clampHistory=true;
};
[[nodiscard]] bool validDenoiseSettings(const DenoiseSettings&);
[[nodiscard]] GPUDenoiseParams denoiseParameters(const DenoiseSettings&,u32 width,u32 height,u32 signal,u32 view,u32 epoch,u32 revision,bool reset);
[[nodiscard]] bool denoiseCompatible(const GPUDISurface&,const GPUDenoiseHistory&,const GPUDenoiseParams&,const GPUSpecularSample& specular={});
[[nodiscard]] GPUDenoiseHistory denoiseTemporal(const GPUDISurface&,glm::vec3 current,const GPUDenoiseHistory&,
    const GPUDenoiseParams&,glm::vec3 neighborhoodLow,glm::vec3 neighborhoodHigh,const GPUSpecularSample& specular={});
[[nodiscard]] float denoiseSpatialWeight(const GPUDISurface&,const GPUDISurface&,float centerLuminance,float neighborLuminance,
    float variance,const GPUDenoiseParams&);
} // namespace phosphor
