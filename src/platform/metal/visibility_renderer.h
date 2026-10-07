#pragma once
#include "platform/metal/metal_context.h"
#include "pipeline/pipeline_registry.h"
#include "renderer/visibility_layout.h"
#include "renderer/temporal_layout.h"
#include "renderer/exposure.h"
#include "rendergraph/render_graph.h"
#include <array>

namespace phosphor {
class PipelineCache;
class SceneRenderer;
class MeshRenderer;
class SceneStore;
class GpuScene;
namespace rg {
class PassContext;
}

// F7: material-class tile lists, linear resolve, denoiser guides and an
// indexed overflow fallback. Resources and dependencies belong to the graph.
class VisibilityRenderer {
  public:
    VisibilityRenderer(MetalContext &context, PipelineCache &pipelines, SceneRenderer &scene, MeshRenderer &mesh,
                       bool binning, bool checks = false, bool tileResolve = false, bool adaptive = false, bool lighting = false);
    ~VisibilityRenderer();
    void prepareFrame(u32 slot, u32 width, u32 height, u32 outputWidth, u32 outputHeight, const SceneStore &store,
                      std::array<u32, 3> defaultTextures, float exposure, u32 debugMode,
                      const GPUTemporalParams &temporal, const FrameConstants &constants);
    rg::TextureRef addResolve(rg::RenderGraph &graph, rg::TextureRef visibility, rg::TextureRef depth);
    void addChecks(rg::RenderGraph &graph);
    bool check(const GpuScene &geometry);
    [[nodiscard]] bool needsPoseReset(u32 view, u64 bytes) const {
        return view >= previousInstances_.size() || !previousInstances_[view] || bytes > poseCapacity_;
    }
    [[nodiscard]] u64 binnedFrames() const { return binnedFrames_; }
    [[nodiscard]] u64 genericFrames() const { return genericFrames_; }
    [[nodiscard]] u32 checkCount() const { return checksCount_; }
    [[nodiscard]] u32 checkFailures() const { return checkFailures_; }
    [[nodiscard]] std::array<u32, 2> adaptiveStats() const;
    [[nodiscard]] const std::array<u32, ExposureBins> &checkedHistogram() const { return checkedHistogram_; }
    [[nodiscard]] const std::array<u32, ExposureBins> &checkedHistogramLow() const { return checkedHistogramLow_; }
    [[nodiscard]] const std::array<u32, ExposureBins> &checkedHistogramHigh() const { return checkedHistogramHigh_; }
    void addPoseSnapshot(rg::RenderGraph &graph);
    void resetGraphRefs() { poses_ = {}; }
    void prepareLighting(u32 flags, u32 sunIndex);
    void setLightingTextures(rg::TextureRef sun, rg::TextureRef direct, rg::TextureRef gi);
    rg::BufferRef importPoseHistory(rg::RenderGraph& graph);
    [[nodiscard]] MTL::GPUAddress paramsAddress() const { return paramsAddress_; }
    [[nodiscard]] MTL::GPUAddress temporalAddress() const { return temporalAddress_; }
    [[nodiscard]] MTL::GPUAddress previousPoseAddress() const { return previousInstances_[view_]->gpuAddress(); }
    rg::TextureRef addPresent(rg::RenderGraph &graph, rg::TextureRef drawable);
    void replaceColor(rg::TextureRef value) { outputs_[0]=value; }
    [[nodiscard]] rg::TextureRef color() const { return outputs_[0]; }
    [[nodiscard]] rg::TextureRef visibility() const { return visibility_; }
    [[nodiscard]] rg::TextureRef normalRoughness() const { return outputs_[1]; }
    [[nodiscard]] rg::TextureRef diffuseAlbedo() const { return outputs_[2]; }
    [[nodiscard]] rg::TextureRef specularAlbedo() const { return outputs_[3]; }
    [[nodiscard]] rg::TextureRef reactiveMask() const { return outputs_[5]; }
    [[nodiscard]] rg::TextureRef depth() const { return depth_; }
    [[nodiscard]] rg::TextureRef motion() const { return outputs_[4]; }

  private:
    void bind(rg::PassContext &ctx, MTL4::ArgumentTable *table);
    MetalContext &context_;
    PipelineCache &pipelines_;
    SceneRenderer &scene_;
    MeshRenderer &mesh_;
    bool binning_;
    bool lighting_ = false;
    MTL::DepthStencilState* lightingFallbackDepth_ = nullptr;
    MTL::GPUAddress lightingAddress_ = 0;
    rg::TextureRef sun_, direct_, indirect_;
    bool tileResolve_ = false;
    bool adaptive_ = false;
    std::array<MTL::Buffer *, 4> shadingHistory_{};
    std::array<u64, 4> shadingHistorySize_{};
    std::array<bool, 4> shadingHistoryValid_{};
    rg::BufferRef shadingHistoryRef_;
    bool checks_ = false;
    u32 checksCount_ = 0, checkFailures_ = 0;
    FrameConstants constants_{};
    std::array<u32, 3> defaultTextures_{}; // white, normal, metallic-roughness
    std::vector<GPUMaterial> checkMaterials_;
    std::array<u32, ExposureBins> checkedHistogram_{};
    std::array<u32, ExposureBins> checkedHistogramLow_{}, checkedHistogramHigh_{};
    GPUTemporalParams temporal_{};
    std::array<std::vector<GPUInstance>, 4> checkedPreviousPoses_{};
    std::array<MTL::Buffer *, 8> readbacks_{};
    std::array<u64, 8> pitches_{};
    MTL::Buffer *currentReadback_ = nullptr;
    MTL::Buffer *previousReadback_ = nullptr;
    u32 readWidth_ = 0, readHeight_ = 0;
    u64 readPoseCapacity_ = 0;
    bool useBinning_ = false;
    u64 binnedFrames_ = 0, genericFrames_ = 0;
    GPUVisibilityParams params_{};
    u32 slot_ = 0;
    u32 view_ = 0;
    u64 poseCapacity_ = 0;
    std::array<MTL::Buffer *, 4> previousInstances_{};
    MTL::GPUAddress temporalAddress_ = 0;
    rg::BufferRef poses_;
    u64 tileCapacity_ = 0;
    MTL::GPUAddress paramsAddress_ = 0;
    struct Buffers {
        MTL::Buffer *tiles = nullptr;
        MTL::Buffer *args = nullptr;
    };
    std::array<Buffers, METAL_FRAMES_IN_FLIGHT> frames_{};
    // Separate tables: mutation after encoding must not change earlier work.
    std::array<MTL4::ArgumentTable *, 5> tables_{};
    pipe::PipelineHandle clear_, classify_, generic_, present_, fallback_, tile_;
    pipe::PipelineHandle adaptivePipeline_, saveHistory_;
    std::array<pipe::PipelineHandle, VISIBILITY_CLASSES> resolve_{};
    rg::TextureRef visibility_, depth_;
    std::array<rg::TextureRef, 6> outputs_{};
    rg::BufferRef bins_;
};
} // namespace phosphor
