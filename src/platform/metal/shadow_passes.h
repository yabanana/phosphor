#pragma once
#include "core/launch_options.h"
#include "platform/metal/rt_consumer.h"
#include "renderer/history_registry.h"
#include "renderer/shadow_settings.h"
#include "rendergraph/render_graph.h"
#include <memory>
namespace phosphor {
class SceneRenderer; class MeshRenderer; class VisibilityRenderer; class SceneStore; class GpuScene;
class MetalGraphExecutor;
class ShadowPasses {
public:
    struct Frame {
        u32 slot=0, view=0, width=0, height=0, backingWidth=0, backingHeight=0;
        u64 index=0, scene=0;
        bool cut=false, reset=false;
        FrameConstants constants{};
        float unjitteredVP[16]{};
        float nearPlane=0.1f;
    };
    ShadowPasses(MetalContext&, PipelineCache&, SceneRenderer&, MeshRenderer&, VisibilityRenderer&,
                 AccelerationStructures*, const LaunchOptions&);
    ~ShadowPasses();
    void loadScene(const GpuScene&, const SceneStore&);
    void prepareFrame(const SceneStore&, std::span<const GPULight>, const Frame&);
    void addToGraph(rg::RenderGraph&, rg::TextureRef visibility, rg::TextureRef depth);
    void bindFrame(MetalGraphExecutor&);
    [[nodiscard]] rg::BufferRef surfaces() const;
    [[nodiscard]] rg::TextureRef worldPosition() const;
    [[nodiscard]] rg::TextureRef geometricNormal() const;
    [[nodiscard]] rg::TextureRef receiverKeys() const;
    [[nodiscard]] rg::TextureRef mask() const;
    [[nodiscard]] rg::TextureRef zeroLighting() const;
    [[nodiscard]] u32 sunIndex() const;
    [[nodiscard]] u64 version() const;
    [[nodiscard]] GPUShadowCounters counters(u32 slot) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace phosphor
