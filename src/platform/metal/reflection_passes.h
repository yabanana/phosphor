#pragma once
#include "platform/metal/direct_lighting_passes.h"
#include <memory>
namespace phosphor {
class GiPasses;
// SOURCE ONLY / NON VERIFIED. Complete connected F13 graph host. prepareFrame
// follows DirectLighting/GI preparation; addToGraph follows their producers and
// the base HDR resolve (old hemisphere specular omitted by caller flag8).
class ReflectionPasses {
public:
    ReflectionPasses(MetalContext&,PipelineCache&,SceneRenderer&,DirectLightingPasses&,
                     AccelerationStructures*,GiPasses*,const LaunchOptions&);
    ~ReflectionPasses();
    void loadScene(const GpuScene&,const SceneStore&);
    void prepareFrame(const SceneStore&,const ShadowPasses::Frame&,bool useCustom,u64 signalEpoch);
    rg::TextureRef addToGraph(rg::RenderGraph&,rg::TextureRef baseHDR,rg::TextureRef depth);
    void bindFrame(MetalGraphExecutor&);
    [[nodiscard]] u64 version() const;
    [[nodiscard]] bool check(u32 slot) const;
    [[nodiscard]] bool ready() const;
    [[nodiscard]] rg::TextureRef hitDistance() const;
    [[nodiscard]] rg::TextureRef rawSpecular() const;
    [[nodiscard]] rg::TextureRef rawAO() const;
    [[nodiscard]] rg::TextureRef filteredSpecular() const;
    [[nodiscard]] rg::TextureRef filteredAO() const;
    // Capture8: actual selected GI E converted once at the current receiver.
    [[nodiscard]] rg::TextureRef filteredIndirectDiffuse() const;
    [[nodiscard]] rg::BufferRef metadataRef() const;
    [[nodiscard]] const char* probeSource() const;
private:
    struct Impl;std::unique_ptr<Impl> impl_;
};
} // namespace phosphor
