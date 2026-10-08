#pragma once
#include "renderer/gpu_types.h"
#include <glm/glm.hpp>
namespace phosphor {
struct CloudSettings {
    double baseHeight=1500,topHeight=4000,coverage=0.45,densityScale=1;
    double noiseScale=1.0/4000,erosionScale=1.0/500,extinction=0.001,albedo=0.99,anisotropy=0.6;
    double maxDistance=100000,terminationTransmittance=0.005,historyWeight=0.9;
    double depthRelativeThreshold=0.05,positionThreshold=100,lightStepDistance=400;
    glm::dvec3 wind{10,0,0};u32 marchSteps=128,lightSteps=8,seed=0xc10du,maxHistorySamples=16;
};
void validateClouds(const CloudSettings&);
// Original bounded procedural density: periodic value fBm and 27-cell Worley
// erosion, defined in volume_noise.h. No assets or third-party generator.
double cloudDensityReference(const CloudSettings&,glm::dvec3 worldPoint,double altitude,double seconds);
bool cloudHistoryCompatible(const GPUCloudParams&,const GPUCloudHistory&,glm::dvec3 worldPoint,double opaqueDistance);
} // namespace phosphor
