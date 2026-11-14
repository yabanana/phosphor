#pragma once
#include "platform/metal/direct_lighting_passes.h"
#include "renderer/gi_lighting_epoch.h"
namespace phosphor {
class GiPasses {
public:
    struct ReadResources {
        MTL::Buffer *states=nullptr,*cache=nullptr;
        MTL::GPUAddress params=0,extra=0;
        rg::BufferRef stateRef{},cacheRef{};
        rg::TextureRef irradiance{},moments{};
        GPUProbeGridParams parameters{};
    };
    [[nodiscard]] ReadResources readResources() const;
    GiPasses(MetalContext&,PipelineCache&,SceneRenderer&,AccelerationStructures&,ShadowPasses&,DirectLightingPasses&,const LaunchOptions&);
    ~GiPasses();
    void loadScene(const GpuScene&,const SceneStore&);
    void prepareFrame(const GpuScene&,const SceneStore&,std::span<const GPULight>,const ShadowPasses::Frame&);
    void setEnvironment(const GiEnvironment&);
    void addToGraph(rg::RenderGraph&);
    void bindFrame(MetalGraphExecutor&);
    [[nodiscard]] rg::TextureRef irradiance() const;
    /// F15.1 (ticket 19): E(-N) of the thin-walled subsurface pixels (DDGI with a valid lobe guide), else an invalid ref.
    [[nodiscard]] rg::TextureRef irradianceBack() const;
    [[nodiscard]] rg::TextureRef referenceDiffuse() const;
    [[nodiscard]] u64 version() const;
    [[nodiscard]] bool check(u32 slot) const;
private:
    struct Impl;std::unique_ptr<Impl> impl_;
};
} // namespace phosphor
