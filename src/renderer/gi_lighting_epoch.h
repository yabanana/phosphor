#pragma once
#include "renderer/gpu_types.h"
#include <array>
#include <span>
#include <vector>
namespace phosphor {
struct GiEnvironment {
    std::array<float,3> skyRadiance{};
    float sunAngularRadius=0.00465f;
    u64 externalRevision=0;
};
// Collision-free exact semantic tuple. Directional lights belong to the GI
// radiance domain even though F11's sampled local list deliberately omits them.
class GiLightingEpoch {
public:
    u64 update(std::span<const GPULight>,const GiEnvironment&,u32 sampledRadianceRevision=0);
    [[nodiscard]] u64 value()const{return epoch_;}
private:
    std::vector<u32> previous_;
    u64 external_=~u64{0},epoch_=0;
};
}
