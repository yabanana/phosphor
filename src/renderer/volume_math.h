#pragma once
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#define VM_FN static __attribute__((unused))
#define VM_EXP(x) metal::exp(x)
#else
#include <cmath>
#define VM_FN inline
#define VM_EXP(x) std::exp(x)
#endif
namespace phosphor {
// The injected source already includes the inverse light-selection proposal.
// Linear temporal accumulation preserves its expectation; stochastic raw
// neighbor extrema are not bounds on the mean incident radiance.
VM_FN float volumeTemporalSource(float current, float previous, float historyWeight) {
    return current + (previous - current) * historyWeight;
}
// Integral of exp(-sigma*t), t in [0,distance], for nonnegative finite
// extinction and distance. Metal has no expm1. The small optical-depth
// polynomial avoids subtraction cancellation and includes the vacuum limit.
VM_FN float volumeIntegralFactor(float sigma, float distance) {
    const float tau = sigma * distance;
    if (tau < 0.1f)
        return distance * (1.0f + tau * (-0.5f + tau * (1.0f / 6.0f +
               tau * (-1.0f / 24.0f + tau / 120.0f))));
    return (1.0f - VM_EXP(-tau)) / sigma;
}
}
#undef VM_FN
#undef VM_EXP
