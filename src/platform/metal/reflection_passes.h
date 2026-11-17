#pragma once
#include "platform/metal/direct_lighting_passes.h"
#include <array>
#include <memory>
#include <string>
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
    // The scene's single declared isotropic environment (GiEnvironment sky):
    // reflection misses, probe-capture misses and the GI-off ambient use it.
    void setEnvironment(std::array<float,3> skyRadiance);
    // F15.1 (ticket 17, RT tier): before prepareFrame, whether the subsurface walk is effective (the composite reads its gain target); after the
    // walk pass is in the graph, the gain target (addToGraph reads it).
    void setSubsurfaceWalk(bool active);
    void setSubsurfaceGain(rg::TextureRef gain);
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
    [[nodiscard]] std::string completedAOCheckpoint(u32 slot) const;
    [[nodiscard]] std::string completedSignalCheckpoint(u32 slot,u32 signal) const;
    // Capture8: actual selected GI E converted once at the current receiver.
    [[nodiscard]] rg::TextureRef filteredIndirectDiffuse() const;
    [[nodiscard]] rg::BufferRef metadataRef() const;
    [[nodiscard]] const char* probeSource() const;
private:
    struct Impl;std::unique_ptr<Impl> impl_;
};
} // namespace phosphor
