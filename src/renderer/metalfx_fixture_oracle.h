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
// Owner decision D1 (2026-11-10, closure brief section 4): the MetalFX SDK
// carries an absolute error floor of about 1e-4..3e-4 of exposure white, so a
// pure 1% relative cap is unreachable for dark channels and for channels far
// above white. A channel with physical target t and exposure scale e (physical
// -> exposed, white = 1) passes when
//   x = t*e <= 4:  |restored - t| <= max(0.01*t, 5e-4/e)
//   x > 4:         |sdr(restored*e) - sdr(x)| <= 2/255
// where sdr is the product display transform (Narkowicz ACES fit of
// post.metal, then the sRGB transfer the swapchain applies).
inline double fxProductSdr(double x) {
    x = std::max(x, 0.0);
    const double a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    const double t = std::clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
    return t <= 0.0031308 ? 12.92 * t : 1.055 * std::pow(t, 1.0 / 2.4) - 0.055;
}
// Returns the error over its allowance (<= 1 passes).
inline double fxD1Excess(double target, double restored, double exposureScale) {
    if (!std::isfinite(target) || !std::isfinite(restored) || !(exposureScale > 0) || !std::isfinite(exposureScale))
        return std::numeric_limits<double>::infinity();
    const double x = target * exposureScale;
    if (x <= 4.0) return std::abs(restored - target) / std::max(0.01 * std::abs(target), 5e-4 / exposureScale);
    return std::abs(fxProductSdr(restored * exposureScale) - fxProductSdr(x)) / (2.0 / 255.0);
}
struct FXConstantComparison {
    bool finite=true,preExposed=false,physical=false,restored=false,restoredD1=false;
    double preExposedRelativeError=0,physicalRelativeError=0,restoredRelativeError=0,restoredD1Excess=0;
};
inline FXConstantComparison compareFXConstant(std::span<const float> sdk,std::span<const float> restored,
                                               std::array<float,3> physical,float preExposure,double tolerance=0.01,
                                               double exposureScale=0) {
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
        if(exposureScale>0)r.restoredD1Excess=std::max(r.restoredD1Excess,fxD1Excess(target,restored[i],exposureScale));
    }
    r.preExposed=r.finite&&r.preExposedRelativeError<=tolerance;
    r.physical=r.finite&&r.physicalRelativeError<=tolerance;
    r.restored=r.finite&&r.restoredRelativeError<=tolerance;
    r.restoredD1=r.finite&&exposureScale>0&&r.restoredD1Excess<=1.0;
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
