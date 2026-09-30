#pragma once

#include "core/types.h"
#include "pipeline/pipeline_registry.h"
#include "core/memory/memory_budget.h"
#include "platform/metal/metal_context.h"
#include "rendergraph/scenario.h"

#include <array>
#include <string>
#include <vector>

namespace phosphor {

class MetalGraphExecutor;
class PipelineCache;

// ---------------------------------------------------------------------------
// ScenarioPasses -- --graph-scenario N (OPT-1): runs a portable graph scenario
// (rendergraph/scenario.h) of synthetic passes (shaders/scenario.metal) as the
// frame graph, through the real executor.  The scenario's image goes to the
// drawable (its "Present" pass); the engine may add the ImGui overlay and the
// frame capture after it.
//
// Raster pipelines depend on the render group a pass ends up in (every
// attachment of the group is declared, unwritten slots with write mask 0), so
// they are requested after each compilation (onCompiled) and waited for:
// graph compilations are loading time.  Persistent textures (TAA history
// ping-pong, probe atlas) are created and filled once, then bound every frame.
// ---------------------------------------------------------------------------

class ScenarioPasses {
public:
    ScenarioPasses(MetalContext& context, PipelineCache& pipelines, u32 index, const rg::ScenarioParams& params);
    ~ScenarioPasses();

    ScenarioPasses(const ScenarioPasses&) = delete;
    ScenarioPasses& operator=(const ScenarioPasses&) = delete;

    /// Build the scenario into `graph` (reset first) with its present pass
    /// writing `drawable` (imported by the caller after reset: pass a
    /// callback-free graph whose only resource is the drawable).
    void build(rg::RenderGraph& graph, rg::TextureRef drawable);
    /// Request (and wait for) the pipelines of the compiled graph.
    void onCompiled(const rg::RenderGraph& graph, const rg::CompiledGraph& compiled);
    /// Bind the persistent textures of frame `frameIndex`.
    void bind(MetalGraphExecutor& executor, u64 frameIndex);

    [[nodiscard]] const rg::Scenario& scenario() const { return scenario_; }
    /// Plan family of the scenario (rg::scenarioFamily, without build choices).
    [[nodiscard]] std::string family() const { return rg::scenarioFamily(index_, defaults_); }
    /// Build choices of an OPT-1 plan for the next build(); reset: the
    /// launch parameters.
    void setBuildChoices(const std::vector<std::string>& remat, const std::vector<std::string>& async);
    void resetBuildChoices() { params_ = defaults_; }
    [[nodiscard]] const rg::ScenarioParams& params() const { return params_; }

private:
    struct PassState {
        pipe::PipelineHandle pipeline = pipe::INVALID_PIPELINE;
        std::vector<u8>      args;       // SynthArgs bytes (copied per frame)
        u32                  vertexCount = 0;
    };

    void execute(u32 pass, rg::PassContext& ctx);
    void createPersistent();
    void fill(MTL::Texture* texture, u32 seed);
    [[nodiscard]] MTL::Texture* newTexture(const rg::TextureDesc& desc, MemoryCategory category, const char* label);

    MetalContext&      context_;
    PipelineCache&     pipelines_;
    u32                index_;
    rg::ScenarioParams params_;
    rg::ScenarioParams defaults_;
    rg::Scenario       scenario_;
    std::vector<PassState> passes_;

    pipe::PipelineHandle compute_ = pipe::INVALID_PIPELINE;
    MTL4::ArgumentTable* rasterArgs_  = nullptr;
    MTL4::ArgumentTable* computeArgs_ = nullptr;
    MTL::Texture*        dummyColor_  = nullptr;
    MTL::Texture*        dummyDepth_  = nullptr;
    MTL::DepthStencilState* depthWrite_ = nullptr;
    MTL::DepthStencilState* depthTest_  = nullptr;
    MTL::DepthStencilState* depthOff_   = nullptr;
    // Persistent textures: history ping-pong pair and static imports, by import index.
    std::array<MTL::Texture*, 2> history_{};
    std::vector<MTL::Texture*>   statics_;
    // Depth state of the render encoder being recorded (the validation layer
    // rejects redundant state; a new encoder starts from Metal's default).
    const void*             encoder_ = nullptr;
    MTL::DepthStencilState* bound_   = nullptr;
    const void*                 computeEncoder_ = nullptr; // same for the compute pipeline
    MTL::ComputePipelineState*  boundCompute_   = nullptr;
};

} // namespace phosphor
