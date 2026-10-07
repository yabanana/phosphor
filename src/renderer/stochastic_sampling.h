#pragma once
#include "core/types.h"
#include <vector>

namespace phosphor::di {

inline constexpr u32 STOCHASTIC_GENERATOR_VERSION = 1;
struct StbnConfig {
    u32 width = 8, height = 8, frames = 16, dimensions = 64;
    u32 seed = 0x5eeda11u;
    float sigmaSpatial = 1.9f, sigmaTemporal = 1.9f;
    u32 maxRelaxationSwaps = 4096;
};
struct StbnMask {
    StbnConfig config;
    // Independent scalar dimensions; ranks [0,N), N=width*height*frames.
    // index = (((dimension*frames+t)*height+y)*width+x).
    std::vector<u32> ranks;
    bool relaxationConverged = false; // no quality/adoption claim follows from this
    [[nodiscard]] float sample(u32 x, u32 y, u32 frame, u32 dimension) const;
};

// Original implementation of scalar void-and-cluster with the STBN same-z or
// same-xy toroidal Gaussian energy, described by Wolfe et al. (EGSR 2022).
// Generation is explicit LOADING/OFFLINE work. Never call this every frame.
// No external code or texture assets are copied; project-authored source.
[[nodiscard]] StbnMask generateStbn(const StbnConfig& config);
// Independent full-range fallback. This is WHITE noise, never labelled STBN.
[[nodiscard]] u32 stochasticHash(u32 value);
[[nodiscard]] float whiteSample(u32 x, u32 y, u32 frame, u32 dimension, u32 seed);

} // namespace phosphor::di
