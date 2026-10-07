#pragma once
#include "renderer/gpu_types.h"
#include <array>
#include <span>
#include <vector>
namespace phosphor::di {
// Area-domain reuse with J=1 is allowed only for an isometry of the emitter's
// affine map. Translation is irrelevant; scale/shear change its linear metric.
// Relative tolerance is an FP32 arithmetic tolerance, not a quality preset.
std::array<double,7> emitterLinearMetric(const float* world);
bool sameEmitterAreaDomain(const float* previous,const float* current,double relativeTolerance=2e-5);
double emitterWorldArea(const GPUEmissiveSurface&,const float* world);
class EmissiveDomainTracker {
public:
    bool update(std::span<const GPUEmissiveSurface>,std::span<const float> worldMatrices);
    void clear(){entries_.clear();}
private:
    struct Entry {std::array<double,7> metric{};u32 slot=~0u,generation=0;bool valid=false;};
    std::vector<Entry> entries_;
};
}
