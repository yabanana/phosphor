#pragma once
#include "renderer/metalfx_denoise_fixture_layout.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>

namespace phosphor {
struct FXFixtureHistoryPolicy {
    u64 signalEpoch=0;
    bool reset=false,channelsWrap=false,steady=false;
};
inline FXFixtureHistoryPolicy fxFixtureHistoryPolicy(u32 scenario,u64 frame,u32 width,u64 sourceEpoch,bool sourceReset) {
    const bool channels=scenario==FX_FIXTURE_CHANNELS;
    if(channels&&width<2)throw std::invalid_argument("invalid channel fixture period");
    const u64 period=std::max(1u,width/2u),wraps=channels?frame/period:0;
    const u64 phaseEpoch=frame/48u+1u;
    if(wraps>std::numeric_limits<u64>::max()-phaseEpoch||sourceEpoch>std::numeric_limits<u64>::max()-phaseEpoch-wraps)
        throw std::invalid_argument("SDK fixture signal epoch overflow");
    FXFixtureHistoryPolicy result;
    result.channelsWrap=channels&&frame>0&&frame%period==0;
    result.signalEpoch=sourceEpoch+phaseEpoch+wraps;
    result.reset=sourceReset||frame%48u==0||result.channelsWrap;
    result.steady=frame%48u>=8u&&(!channels||frame%period>=8u);
    return result;
}
struct FXConstantComparison {
    bool finite=true,preExposed=false,physical=false,restored=false;
    double preExposedRelativeError=0,physicalRelativeError=0,restoredRelativeError=0;
};
inline FXConstantComparison compareFXConstant(std::span<const float> sdk,std::span<const float> restored,
                                               std::array<float,3> physical,float preExposure,double tolerance=0.01) {
    if(sdk.empty()||sdk.size()%3||sdk.size()!=restored.size()||!std::isfinite(preExposure)||preExposure<=0||
       !std::isfinite(tolerance)||tolerance<=0||!std::all_of(physical.begin(),physical.end(),[](float v){return std::isfinite(v)&&v>=0;}))
        throw std::invalid_argument("invalid SDK constant oracle arrays");
    FXConstantComparison r;
    for(size_t i=0;i<sdk.size();++i) {
        r.finite=r.finite&&std::isfinite(sdk[i])&&std::isfinite(restored[i])&&sdk[i]>=0&&restored[i]>=0;
        const double target=physical[i%3],scale=std::max(std::abs(target),1e-6);
        r.preExposedRelativeError=std::max(r.preExposedRelativeError,std::abs(double(sdk[i])/preExposure-target)/scale);
        r.physicalRelativeError=std::max(r.physicalRelativeError,std::abs(double(sdk[i])-target)/scale);
        r.restoredRelativeError=std::max(r.restoredRelativeError,std::abs(double(restored[i])-target)/scale);
    }
    r.preExposed=r.finite&&r.preExposedRelativeError<=tolerance;
    r.physical=r.finite&&r.physicalRelativeError<=tolerance;
    r.restored=r.finite&&r.restoredRelativeError<=tolerance;
    return r; // Hypotheses are reported; this function never changes SDK policy.
}
struct FXMetamorphicComparison {
    bool finite=true,passed=false;
    double normalizedGain=0,relativeShapeError=0,supportDisagreement=0;
};
inline FXMetamorphicComparison compareFXScaledPair(std::span<const float> unit,std::span<const float> scaled,
                                                   float factor,double gainTolerance=0.02,double supportTolerance=0.02) {
    if(unit.empty()||unit.size()!=scaled.size()||!std::isfinite(factor)||factor<=0||
       !std::isfinite(gainTolerance)||gainTolerance<=0||!std::isfinite(supportTolerance)||supportTolerance<=0)
        throw std::invalid_argument("invalid SDK metamorphic pair");
    double a=0,b=0,error=0,peak=0;FXMetamorphicComparison r;
    for(size_t i=0;i<unit.size();++i) {
        r.finite=r.finite&&std::isfinite(unit[i])&&std::isfinite(scaled[i])&&unit[i]>=0&&scaled[i]>=0;
        a+=std::max(0.f,unit[i]);b+=std::max(0.f,scaled[i])/factor;
        error+=std::abs(double(scaled[i])/factor-unit[i]);peak=std::max(peak,double(std::max(0.f,unit[i])));
    }
    if(!(a>1e-12)){r.passed=false;return r;} // A black image cannot prove scaling.
    r.normalizedGain=b/a;r.relativeShapeError=error/a;
    u32 different=0,populated=0;
    for(size_t i=0;i<unit.size();++i) {
        const bool x=unit[i]>peak*0.001,y=double(scaled[i])/factor>peak*0.001;
        if(x||y)++populated;if(x!=y)++different;
    }
    r.supportDisagreement=populated?double(different)/populated:1;
    r.passed=r.finite&&std::abs(r.normalizedGain-1)<=gainTolerance&&r.relativeShapeError<=gainTolerance&&r.supportDisagreement<=supportTolerance;
    return r; // No assumption that the native impulse is identity/unit-energy.
}
} // namespace phosphor
