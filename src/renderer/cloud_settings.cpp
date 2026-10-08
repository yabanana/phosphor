#include "renderer/cloud_settings.h"
#include "renderer/volume_noise.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>
namespace phosphor {
void validateClouds(const CloudSettings& s) {
    if(!(s.baseHeight>=0&&s.topHeight>s.baseHeight&&std::isfinite(s.topHeight))||!(s.coverage>=0&&s.coverage<=1)||
       !(s.densityScale>=0&&s.densityScale<=10)||!(s.noiseScale>0&&std::isfinite(s.noiseScale))||!(s.erosionScale>0&&std::isfinite(s.erosionScale))||
       !(s.extinction>=0&&std::isfinite(s.extinction))||!(s.albedo>=0&&s.albedo<=1)||!(std::abs(s.anisotropy)<0.99)||
       !(s.maxDistance>0&&s.maxDistance<=1e7)||!(s.terminationTransmittance>0&&s.terminationTransmittance<1)||
       !(s.historyWeight>=0&&s.historyWeight<1)||!(s.positionThreshold>=0&&std::isfinite(s.positionThreshold))||
       !(s.depthRelativeThreshold>=0&&s.depthRelativeThreshold<=1)||!(s.lightStepDistance>0&&std::isfinite(s.lightStepDistance))||
       !s.marchSteps||s.marchSteps>512||!s.lightSteps||s.lightSteps>64||!s.maxHistorySamples||s.maxHistorySamples>64)
        throw std::invalid_argument("invalid procedural cloud preset");
    for(double c:{s.wind.x,s.wind.y,s.wind.z})if(!std::isfinite(c))throw std::invalid_argument("invalid cloud wind");
}
double cloudDensityReference(const CloudSettings& s,glm::dvec3 p,double altitude,double seconds) {
    if(!std::isfinite(seconds)||!std::isfinite(altitude))throw std::invalid_argument("invalid cloud clock/height");
    p-=s.wind*seconds;
    for(double c:{p.x,p.y,p.z})if(!std::isfinite(c)||std::abs(c)>std::numeric_limits<float>::max())throw std::invalid_argument("invalid cloud world point");
    return volumeCloudShape(float(p.x),float(p.y),float(p.z),float(altitude),float(s.baseHeight),float(s.topHeight),float(s.coverage),
                            float(s.densityScale),float(s.noiseScale),float(s.erosionScale),s.seed);
}
bool cloudHistoryCompatible(const GPUCloudParams& p,const GPUCloudHistory& h,glm::dvec3 point,double depth) {
    if(!(p.flags&VOLUME_HISTORY_VALID)||!h.valid||h.viewID!=p.viewID||h.generation!=p.generation||
       !std::isfinite(depth)||!std::isfinite(h.opaqueDistance)||!h.samples||h.samples>64)return false;
    const glm::dvec3 old(h.worldPosition[0],h.worldPosition[1],h.worldPosition[2]);
    const glm::dvec3 advected=point-glm::dvec3(p.wind[0],p.wind[1],p.wind[2])*double(p.timeSeconds-p.previousTimeSeconds);
    for(double c:{advected.x,advected.y,advected.z,old.x,old.y,old.z})if(!std::isfinite(c))return false;
    return glm::length(advected-old)<=p.positionThreshold && std::abs(depth-h.opaqueDistance)<=p.depthRelativeThreshold*std::max(1.0,depth);
}
} // namespace phosphor
