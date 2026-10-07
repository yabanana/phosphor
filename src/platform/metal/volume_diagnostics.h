#pragma once
#include "platform/metal/metal_context.h"
#include "platform/metal/pipeline_cache.h"
#include "renderer/volume_oracle.h"
#include "rendergraph/render_graph.h"
#include <memory>
namespace phosphor {
class MetalGraphExecutor;
class VolumeDiagnostics {
  public:
    struct Config {std::string path;u32 every=1;bool homogeneousFog=false;u32 corruption=0;};
    struct Sources {
        rg::TextureRef transmittance{},multiple{},sky{};
        rg::BufferRef fogCells{},fogIntegrated{};
        MTL::Buffer *fogCellsBuffer=nullptr,*fogIntegratedBuffer=nullptr;
        MTL::Buffer* counters=nullptr;
    };
    VolumeDiagnostics(MetalContext&,PipelineCache&,Config);
    ~VolumeDiagnostics();
    bool prepare(u32 slot,u64 frame,u32 view,const GPUAtmosphereParams& expected,const GPUAtmosphereParams& submitted,const GPUFogParams& fog,bool armed);
    void beginGraph(rg::RenderGraph&);
    void stamp(rg::RenderGraph&,rg::TextureRef actualProducerOutput,u32 kind);
    void homogeneous(rg::PassContext&,MTL::Buffer* cells,MTL::GPUAddress fogParams);
    void foreignHistory(rg::RenderGraph&,rg::BufferRef& history,MTL::Buffer* physical,u32 cellCount,bool cloud);
    void collect(rg::RenderGraph&,const Sources&);
    void bindFrame(MetalGraphExecutor&);
    bool consume(u32 completedSlot); // true also when no matching capture exists
    [[nodiscard]] bool selected()const;
    [[nodiscard]] bool homogeneousFog()const;
    [[nodiscard]] u32 corruption()const;
  private:
    struct Impl;std::unique_ptr<Impl> impl_;
};
} // namespace phosphor
