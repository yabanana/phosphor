#pragma once
#include "renderer/light_sampling.h"
#include <span>
namespace phosphor {
class GpuScene;class SceneStore;
// CPU proposal data preserves local emitter geometry. The graph transforms
// emitters using the SAME FRAME GPU poses before any lighting consumer reads it.
class LocalLightScene {
public:
    void rebuild(const GpuScene&,const SceneStore&,std::span<const GPULight> punctual);
    std::vector<GPUSampledLight> lights;
    std::vector<GPUEmissiveSurface> emitters;
    di::AliasTable alias;
    u32 revision=1, radianceRevision=1;
private:
    std::vector<double> weights_;
    u64 hash_=0, radianceHash_=0;
};
} // namespace phosphor
