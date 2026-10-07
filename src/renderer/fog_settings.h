#pragma once
#include "renderer/gpu_types.h"
#include <span>
#include <glm/glm.hpp>
namespace phosphor {
struct FogSettings {
    u32 gridX=64,gridY=36,gridZ=32,maxLocalLights=64,maxHistoryAge=16;
    double nearDistance=0.1,farDistance=500,densityAtBase=0.001,heightBase=0,heightFalloff=0.01,maxDensity=0.1;
    glm::dvec3 albedo{0.9};double anisotropy=0.2,historyWeight=0.9,positionThreshold=1,depthRelativeThreshold=0.05;
};
struct FogIntegral { glm::dvec3 radiance{0};double transmittance=1; };
void validateFog(const FogSettings&);
double fogDensity(const FogSettings&,double worldHeight);
double fogSliceDistance(const FogSettings&,double normalizedSlice);
// Exact constant-source/constant-extinction integration, including sigma=0.
FogIntegral fogHomogeneous(double extinction,glm::dvec3 source,double distance);
FogIntegral fogComposite(FogIntegral front,FogIntegral back);
FogIntegral fogIntegrateReference(std::span<const double> extinction,std::span<const glm::dvec3> source,
                                  std::span<const double> segmentLengths);
} // namespace phosphor
