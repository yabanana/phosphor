#pragma once
#include "platform/metal/shadow_passes.h"
#include "renderer/offline_reference.h"
namespace phosphor {
class MetalTextureManager;class DirectLightingPasses;
// Offline export captures WORLD poses/materials after GPU scene producers. It
// does not launch a renderer or manufacture a reference image.
class ReferenceSnapshot {
public:
    ReferenceSnapshot(MetalContext&,SceneRenderer&,const std::string& destination, u64 captureFrame=0, AccelerationStructures* rt=nullptr);
    ~ReferenceSnapshot();
    void loadScene(const GpuScene&,const SceneStore&,const MetalTextureManager&);
    void prepareFrame(const ShadowPasses::Frame&,ReferenceCamera,std::span<const GPULight>,const SceneStore&,DirectLightingPasses*);
    void addToGraph(rg::RenderGraph&);
    void bindFrame(MetalGraphExecutor&);
    void consume(u32 slot); // AFTER completion, BEFORE reload/teardown
    [[nodiscard]] u64 version() const;
private:
    struct Impl;std::unique_ptr<Impl> impl_;
};
} // namespace phosphor
