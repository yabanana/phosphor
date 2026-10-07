#pragma once
#include "platform/metal/direct_lighting_passes.h"
#include "renderer/gi_lighting_epoch.h"
namespace phosphor {
class GiPasses {
public:
    GiPasses(MetalContext&,PipelineCache&,SceneRenderer&,AccelerationStructures&,ShadowPasses&,DirectLightingPasses&,const LaunchOptions&);
    ~GiPasses();
    void loadScene(const GpuScene&,const SceneStore&);
    void prepareFrame(const GpuScene&,const SceneStore&,std::span<const GPULight>,const ShadowPasses::Frame&);
    void setEnvironment(const GiEnvironment&);
    void addToGraph(rg::RenderGraph&);
    void bindFrame(MetalGraphExecutor&);
    [[nodiscard]] rg::TextureRef irradiance() const;
    [[nodiscard]] rg::TextureRef referenceDiffuse() const;
    [[nodiscard]] u64 version() const;
    [[nodiscard]] bool check(u32 slot) const;
private:
    struct Impl;std::unique_ptr<Impl> impl_;
};
} // namespace phosphor
