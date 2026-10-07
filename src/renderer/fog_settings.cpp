#include "renderer/fog_settings.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
namespace phosphor {
void validateFog(const FogSettings& s) {
    if(!s.gridX||!s.gridY||!s.gridZ||s.gridX>256||s.gridY>256||s.gridZ>128||s.maxLocalLights>16||!s.maxHistoryAge||s.maxHistoryAge>64||
       !(s.nearDistance>0&&s.farDistance>s.nearDistance&&std::isfinite(s.farDistance))||
       !(s.densityAtBase>=0&&std::isfinite(s.densityAtBase))||!(s.heightFalloff>=0&&std::isfinite(s.heightFalloff))||
       !std::isfinite(s.heightBase)||!(s.maxDensity>=0&&std::isfinite(s.maxDensity))||
       !(s.historyWeight>=0&&s.historyWeight<1)||!(std::abs(s.anisotropy)<0.99)||
       !(s.positionThreshold>=0&&std::isfinite(s.positionThreshold))||!(s.depthRelativeThreshold>=0&&s.depthRelativeThreshold<=1)||
       glm::any(glm::lessThan(s.albedo,glm::dvec3(0)))||glm::any(glm::greaterThan(s.albedo,glm::dvec3(1))))
        throw std::invalid_argument("invalid metre-based fog preset");
    for(double c:{s.albedo.x,s.albedo.y,s.albedo.z})if(!std::isfinite(c))throw std::invalid_argument("nonfinite fog albedo");
}
double fogDensity(const FogSettings& s,double y){if(!std::isfinite(y))throw std::invalid_argument("invalid fog height");return std::min(s.maxDensity,s.densityAtBase*std::exp(std::clamp(-(y-s.heightBase)*s.heightFalloff,-80.0,80.0)));}
double fogSliceDistance(const FogSettings& s,double slice){return s.nearDistance*std::pow(s.farDistance/s.nearDistance,std::clamp(slice,0.0,1.0));}
FogIntegral fogHomogeneous(double sigma,glm::dvec3 source,double distance) {
    if(!(sigma>=0&&std::isfinite(sigma)&&distance>=0&&std::isfinite(distance)))throw std::invalid_argument("invalid homogeneous volume");
    const double tau=sigma*distance,T=std::exp(-tau),integral=sigma>1e-12?-std::expm1(-tau)/sigma:distance;
    return {source*integral,T};
}
FogIntegral fogComposite(FogIntegral front,FogIntegral back){return {front.radiance+front.transmittance*back.radiance,front.transmittance*back.transmittance};}
FogIntegral fogIntegrateReference(std::span<const double> sigma,std::span<const glm::dvec3> source,std::span<const double> ds) {
    if(sigma.size()!=source.size()||sigma.size()!=ds.size())throw std::invalid_argument("volume reference arrays mismatch");
    FogIntegral out;for(size_t i=0;i<sigma.size();++i)out=fogComposite(out,fogHomogeneous(sigma[i],source[i],ds[i]));return out;
}
} // namespace phosphor
