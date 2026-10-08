#include "renderer/denoise_settings.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace phosphor {
namespace {
glm::vec3 v3(const float* p){return {p[0],p[1],p[2]};}
bool finite(glm::vec3 p){return std::isfinite(p.x)&&std::isfinite(p.y)&&std::isfinite(p.z);}
float luminance(glm::vec3 c){return glm::dot(c,glm::vec3(.2126f,.7152f,.0722f));}
glm::vec3 signalNormal(const GPUDISurface& s,u32 signal){return v3(signal==DENOISE_SIGNAL_SPECULAR?s.shadingNormal:s.geometricNormal);}
}
bool validDenoiseSettings(const DenoiseSettings& s){return std::isfinite(s.temporalAlpha)&&s.temporalAlpha>0&&s.temporalAlpha<=1&&
    std::isfinite(s.momentsAlpha)&&s.momentsAlpha>0&&s.momentsAlpha<=1&&std::isfinite(s.depthThreshold)&&s.depthThreshold>0&&
    std::isfinite(s.normalThreshold)&&s.normalThreshold>=-1&&s.normalThreshold<=1&&std::isfinite(s.roughnessThreshold)&&s.roughnessThreshold>=0&&
    std::isfinite(s.hitDistanceThreshold)&&s.hitDistanceThreshold>=0&&std::isfinite(s.luminancePhi)&&s.luminancePhi>0&&
    std::isfinite(s.normalPhi)&&s.normalPhi>0&&std::isfinite(s.varianceFloor)&&s.varianceFloor>0&&std::isfinite(s.clampSigma)&&s.clampSigma>=0&&
    std::isfinite(s.planeThreshold)&&s.planeThreshold>0&&s.maxHistory>=1&&s.maxHistory<=256&&s.atrousIterations<=5;}
GPUDenoiseParams denoiseParameters(const DenoiseSettings& s,u32 w,u32 h,u32 signal,u32 view,u32 epoch,u32 rev,bool reset){
    if(!validDenoiseSettings(s)||!w||!h||signal>DENOISE_SIGNAL_AO)throw std::invalid_argument("invalid denoise configuration");
    GPUDenoiseParams p{};p.width=w;p.height=h;p.signal=signal;p.viewID=view;p.historyEpoch=epoch;p.signalRevision=rev;
    p.flags=(reset?DENOISE_RESET:0u)|(s.clampHistory?DENOISE_CLAMP_HISTORY:0u);
    p.temporalAlpha=s.temporalAlpha;p.momentsAlpha=s.momentsAlpha;p.depthThreshold=s.depthThreshold;p.normalThreshold=s.normalThreshold;
    p.roughnessThreshold=s.roughnessThreshold;p.hitDistanceThreshold=s.hitDistanceThreshold;p.luminancePhi=s.luminancePhi;p.normalPhi=s.normalPhi;
    p.maxHistory=s.maxHistory;p.atrousStep=1;p.minHistoryForVariance=4;p.varianceFloor=s.varianceFloor;p.clampSigma=s.clampSigma;p.planeThreshold=s.planeThreshold;return p;
}
bool denoiseCompatible(const GPUDISurface& s,const GPUDenoiseHistory& h,const GPUDenoiseParams& p,const GPUSpecularSample& spec){
    if((p.flags&DENOISE_RESET)||!s.valid||!h.valid||!h.length||h.signal!=p.signal||h.viewID!=p.viewID||h.historyEpoch!=p.historyEpoch||
        h.signalRevision!=p.signalRevision||h.slot!=s.instanceSlot||h.instanceGeneration!=s.instanceGeneration||h.materialRevision!=s.materialRevision||
        !std::isfinite(s.depth)||!std::isfinite(h.depth)||s.depth<=0||h.depth<=0||!finite(v3(h.color))||!std::isfinite(h.firstMoment)||
        !std::isfinite(h.secondMoment)||!std::isfinite(h.variance)||h.variance<0)return false;
    auto n=signalNormal(s,p.signal),old=v3(h.normal);if(!finite(n)||!finite(old)||glm::dot(n,n)<=0||glm::dot(old,old)<=0||
        glm::dot(glm::normalize(n),glm::normalize(old))<p.normalThreshold)return false;
    const float scale=std::max(s.depth,h.depth);
    if(std::abs(s.depth-h.depth)>p.depthThreshold*scale||!finite(v3(s.position))||!finite(v3(h.position))||
        std::abs(glm::dot(v3(s.position)-v3(h.position),glm::normalize(n)))>p.planeThreshold*scale)return false;
    if(h.flags&~(p.signal==DENOISE_SIGNAL_SPECULAR?DENOISE_HISTORY_STOCHASTIC:0u))return false;
    if(p.signal==DENOISE_SIGNAL_SPECULAR){
        const bool stochastic=(spec.flags&SPECULAR_SAMPLE_STOCHASTIC)!=0&&!(p.flags&DENOISE_DIAGNOSTIC_ENDPOINTS);
        if(!(spec.flags&SPECULAR_SAMPLE_VALID)||(spec.flags&SPECULAR_SAMPLE_ERROR)||
           stochastic!=bool(h.flags&DENOISE_HISTORY_STOCHASTIC)||!std::isfinite(h.roughness)||!std::isfinite(s.roughness)||
           std::abs(h.roughness-s.roughness)>p.roughnessThreshold||!std::isfinite(spec.hitDistance)||spec.hitDistance<0)return false;
        // A fresh GGX ray can legitimately hit another object, miss, or have
        // zero contribution. Primary geometry and source revisions above own
        // validity; conditioning accumulation on the random endpoint defeats it.
        if(!stochastic&&(h.path!=spec.path||h.secondarySlot!=spec.secondarySlot||h.secondaryGeneration!=spec.secondaryGeneration||
           !std::isfinite(h.hitDistance)||std::abs(h.hitDistance-spec.hitDistance)>p.hitDistanceThreshold*std::max(1.f,std::max(h.hitDistance,spec.hitDistance))))return false;
    }
    return true;
}
GPUDenoiseHistory denoiseTemporal(const GPUDISurface& s,glm::vec3 current,const GPUDenoiseHistory& old,const GPUDenoiseParams& p,glm::vec3 low,glm::vec3 high,const GPUSpecularSample& spec,float neighborhoodVariance){
    GPUDenoiseHistory h{};h.signal=p.signal;h.viewID=p.viewID;h.historyEpoch=p.historyEpoch;h.signalRevision=p.signalRevision;
    if(p.signal==DENOISE_SIGNAL_SPECULAR&&(spec.flags&SPECULAR_SAMPLE_STOCHASTIC)&&!(p.flags&DENOISE_DIAGNOSTIC_ENDPOINTS))h.flags|=DENOISE_HISTORY_STOCHASTIC;
    if(!s.valid)return h;
    const auto rawNormal=signalNormal(s,p.signal);if(!finite(current)||!finite(rawNormal)||glm::dot(rawNormal,rawNormal)<=0){h.flags=1;return h;}
    current=glm::max(current,glm::vec3(0));if(p.signal==DENOISE_SIGNAL_AO)current=glm::clamp(current,glm::vec3(0),glm::vec3(1));
    bool reuse=denoiseCompatible(s,old,p,spec);glm::vec3 previous=v3(old.color);
    if(reuse&&(p.flags&DENOISE_CLAMP_HISTORY)&&finite(low)&&finite(high))previous=glm::clamp(previous,glm::min(low,high),glm::max(low,high));
    const u32 count=reuse?std::min(old.length,p.maxHistory-1)+1:1;
    const float alpha=reuse?std::max(p.temporalAlpha,1.f/count):1,ma=reuse?std::max(p.momentsAlpha,1.f/count):1;
    glm::vec3 color=reuse?glm::mix(previous,current,alpha):current;if(p.signal==DENOISE_SIGNAL_AO)color=glm::clamp(color,glm::vec3(0),glm::vec3(1));const float L=luminance(current);
    h.firstMoment=reuse?old.firstMoment*(1-ma)+L*ma:L;h.secondMoment=reuse?old.secondMoment*(1-ma)+L*L*ma:L*L;
    h.variance=std::max(p.varianceFloor,h.secondMoment-h.firstMoment*h.firstMoment);h.length=count;
    if(count<p.minHistoryForVariance&&std::isfinite(neighborhoodVariance)&&neighborhoodVariance>=0)h.variance=std::max(h.variance,neighborhoodVariance/count);
    const auto n=glm::normalize(signalNormal(s,p.signal));
    for(u32 i=0;i<3;++i){h.color[i]=color[i];h.position[i]=s.position[i];h.normal[i]=n[i];}
    h.depth=s.depth;h.roughness=s.roughness;h.slot=s.instanceSlot;h.instanceGeneration=s.instanceGeneration;h.materialRevision=s.materialRevision;
    h.hitDistance=spec.hitDistance;h.path=spec.path;h.secondarySlot=spec.secondarySlot;h.secondaryGeneration=spec.secondaryGeneration;
    h.valid=finite(color)&&std::isfinite(h.variance)&&finite(n);if(!h.valid)h.flags=1;return h;
}
float denoiseSpatialWeight(const GPUDISurface& a,const GPUDISurface& b,float la,float lb,float variance,const GPUDenoiseParams& p){
    if(!a.valid||!b.valid||!std::isfinite(la)||!std::isfinite(lb)||!std::isfinite(variance)||variance<0)return 0;
    const auto na=signalNormal(a,p.signal),nb=signalNormal(b,p.signal);if(!finite(na)||!finite(nb)||glm::dot(na,na)<=0||glm::dot(nb,nb)<=0)return 0;
    const float normal=std::pow(std::max(0.f,glm::dot(glm::normalize(na),glm::normalize(nb))),p.normalPhi);
    const float depth=std::exp(-std::abs(a.depth-b.depth)/std::max(1e-6f,p.depthThreshold*std::max(a.depth,b.depth)*p.atrousStep));
    const float color=std::exp(-std::abs(la-lb)/(p.luminancePhi*std::sqrt(std::max(variance,p.varianceFloor))+1e-6f));
    const float rough=p.signal==DENOISE_SIGNAL_SPECULAR?std::exp(-std::abs(a.roughness-b.roughness)/std::max(p.roughnessThreshold,1e-6f)):1;
    return normal*depth*color*rough;
}
} // namespace phosphor
