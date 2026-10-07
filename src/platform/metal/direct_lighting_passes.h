#pragma once
#include "core/launch_options.h"
#include "platform/metal/shadow_passes.h"
#include "renderer/gpu_types.h"
#include "rendergraph/render_graph.h"
#include <memory>
namespace phosphor {
class DirectLightingPasses {
public:
    DirectLightingPasses(MetalContext&,PipelineCache&,SceneRenderer&,MeshRenderer&,VisibilityRenderer&,AccelerationStructures&,
                         ShadowPasses&,const LaunchOptions&);
    ~DirectLightingPasses();
    void loadScene(const GpuScene&,const SceneStore&);
    void prepareFrame(const GpuScene&,const SceneStore&,std::span<const GPULight>,const ShadowPasses::Frame&);
    void addToGraph(rg::RenderGraph&,rg::TextureRef visibility,rg::TextureRef depth);
    void bindFrame(MetalGraphExecutor&);
    [[nodiscard]] rg::TextureRef direct() const;
    [[nodiscard]] rg::TextureRef motion() const;
    [[nodiscard]] rg::BufferRef surfaceRef() const;
    [[nodiscard]] rg::BufferRef lightsRef() const;
    [[nodiscard]] rg::BufferRef emittersRef() const;
    [[nodiscard]] MTL::Buffer* lightsBuffer() const;
    [[nodiscard]] MTL::Buffer* emittersBuffer() const;
    [[nodiscard]] u32 lightCount() const;
    [[nodiscard]] u32 lightRevision() const;
    [[nodiscard]] u64 version() const;
    [[nodiscard]] bool check(u32 slot) const;
private:
    struct Impl;std::unique_ptr<Impl> impl_;
};
} // namespace phosphor
