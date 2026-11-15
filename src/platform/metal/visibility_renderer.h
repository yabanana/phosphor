#pragma once
#include "platform/metal/feature_pipelines.h"
#include "platform/metal/metal_context.h"
#include "pipeline/pipeline_registry.h"
#include "renderer/visibility_layout.h"
#include "renderer/material_class_table.h"
#include "renderer/temporal_layout.h"
#include "renderer/exposure.h"
#include "rendergraph/render_graph.h"
#include <array>
#include <memory>
#include <unordered_map>

namespace phosphor {
class PipelineCache;
class SceneRenderer;
class MeshRenderer;
class SceneStore;
class GpuScene;
class AccelerationStructures;
class RtConsumer;
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
    // exposureInput names the SAME graph version used by Post histogram.
    // Raw resolve/guide checks keep their independent original attachments.
    void addChecks(rg::RenderGraph &graph,rg::TextureRef exposureInput);
    bool check(const GpuScene &geometry);
    [[nodiscard]] bool needsPoseReset(u32 view, u64 bytes) const {
        return view >= previousInstances_.size() || !previousInstances_[view] || bytes > poseCapacity_;
    }
    /// F15.2 (ticket 21): --resolve-features all (the resolve of a scene with features uses the all-lobes pipeline, no feature bins:
    /// the exact-equivalence reference) and --debug-class-drop-lobes (every class bin pipeline compiled without those lobes).
    /// `classBudget` (--debug-class-budget) shrinks the class table so that the overflow bin is exercised on small fixtures.
    void setFeatureDebug(bool forceAllLobes, u32 dropLobes, u32 classBudget = VISIBILITY_FEATURE_CLASSES) {
        forceAllLobes_ = forceAllLobes; dropLobes_ = dropLobes; classBudget_ = classBudget;
    }
    /// Frames resolved through per-class feature bins, and the table of the latest of them.
    [[nodiscard]] u64 featureBinnedFrames() const { return featureBinnedFrames_; }
    [[nodiscard]] const MaterialClassTable& classTable() const { return classTable_; }
    [[nodiscard]] u64 binnedFrames() const { return binnedFrames_; }
    [[nodiscard]] u64 genericFrames() const { return genericFrames_; }
    [[nodiscard]] u32 checkCount() const { return checksCount_; }
    [[nodiscard]] u32 checkFailures() const { return checkFailures_; }
    [[nodiscard]] std::array<u32, 2> adaptiveStats() const;
    [[nodiscard]] const std::array<u32, ExposureBins> &checkedHistogram() const { return checkedHistogram_; }
    [[nodiscard]] const std::array<u32, ExposureBins> &checkedHistogramLow() const { return checkedHistogramLow_; }
    [[nodiscard]] const std::array<u32, ExposureBins> &checkedHistogramHigh() const { return checkedHistogramHigh_; }
    void addPoseSnapshot(rg::RenderGraph &graph);
    void resetGraphRefs() { poses_ = {}; composedColor_ = {}; }
    // environment: the scene's declared isotropic radiance, used by the F13
    // path (flag 16) for ambient fallbacks. Background: pixels without geometry.
    void prepareLighting(u32 flags, u32 sunIndex, std::array<float, 3> environment = {}, float sunAngularRadius = 0);
    void setBackground(std::array<float, 3> radiance) { background_ = radiance; }
    /// F15.1 (ticket 17): called before prepareFrame.  allowed = the resolve owns the diffuse term (no external diffuse composite); the camera
    /// is the unjittered reverse-Z infinite projection (tan of the half field of view per axis).  The screen-space diffusion of volumetric
    /// subsurface is active when allowed, the scene has a volumetric material and the compute resolve runs (not the on-tile experiment).
    void setSubsurface(bool allowed, float nearPlane, float tanHalfFovX, float tanHalfFovY) {
        subsurfaceAllowed_ = allowed; sssNear_ = nearPlane; sssTanX_ = tanHalfFovX; sssTanY_ = tanHalfFovY;
    }
    [[nodiscard]] bool subsurfaceActive() const { return subsurface_; }
    /// F15.1 (ticket 17/18): the OPTIONAL RT tier of the volumetric subsurface (--subsurface-mode rt-walk).  `rt` = the acceleration structures
    /// (null = the tier is not available: the screen-space diffusion serves, the reason is the caller's to log); `samples` walks per pixel and frame,
    /// `maxSteps` the hard bound of one walk; `flags` = SSS_WALK_* (gpu_types.h).  Call once after construction (it requests the kernel).
    void setSubsurfaceWalk(AccelerationStructures* rt, u32 samples, u32 maxSteps, u32 flags);
    /// After AccelerationStructures::prepareFrame: binds the walk's intersection function table for this frame slot.
    void prepareSubsurfaceWalk(u32 slot);
    [[nodiscard]] bool subsurfaceWalkActive() const { return walk_; }
    /// The walk's gain target (rgb = weight x exiting radiance per unit uniform dome radiance, a = weight): the reflection composite multiplies its
    /// ambient / GI diffuse term by it for walk pixels.  Valid after addResolve in a graph with the walk pass.
    [[nodiscard]] rg::TextureRef subsurfaceGain() const { return walk_ ? sssGain_ : rg::TextureRef{}; }
    /// Graph-key value of the subsurface path: 0 none, 1 screen-space diffusion, 2 RT walk.
    [[nodiscard]] u32 subsurfaceKey() const { return subsurface_ ? (walk_ ? 2u : 1u) : 0u; }
    void setLightingTextures(rg::TextureRef sun, rg::TextureRef direct, rg::TextureRef gi, rg::TextureRef giBack = {});
    rg::BufferRef importPoseHistory(rg::RenderGraph& graph);
    [[nodiscard]] MTL::GPUAddress paramsAddress() const { return paramsAddress_; }
    [[nodiscard]] MTL::GPUAddress temporalAddress() const { return temporalAddress_; }
    [[nodiscard]] MTL::GPUAddress previousPoseAddress() const { return previousInstances_[view_]->gpuAddress(); }
    rg::TextureRef addPresent(rg::RenderGraph &graph, rg::TextureRef drawable);
    // Post-lighting selection must never retarget the already-declared resolve
    // writes. Encode callbacks bind outputs_ after the whole graph is built.
    void replaceColor(rg::TextureRef value) { composedColor_=value; }
    [[nodiscard]] rg::TextureRef color() const { return composedColor_.valid() ? composedColor_ : outputs_[0]; }
    [[nodiscard]] rg::TextureRef visibility() const { return visibility_; }
    [[nodiscard]] rg::TextureRef normalRoughness() const { return outputs_[1]; }
    [[nodiscard]] rg::TextureRef diffuseAlbedo() const { return outputs_[2]; }
    [[nodiscard]] rg::TextureRef specularAlbedo() const { return outputs_[3]; }
    [[nodiscard]] rg::TextureRef reactiveMask() const { return outputs_[5]; }
    [[nodiscard]] rg::TextureRef depth() const { return depth_; }
    [[nodiscard]] rg::TextureRef motion() const { return outputs_[4]; }

  private:
    void bind(rg::PassContext &ctx, MTL4::ArgumentTable *table, bool withSss = false);
    [[nodiscard]] pipe::PipelineDesc fallbackDesc(u32 sss = 0) const;
    MetalContext &context_;
    PipelineCache &pipelines_;
    SceneRenderer &scene_;
    MeshRenderer &mesh_;
    // F15.1: the pipelines the frame encodes with (the scene's feature union applied by prepareFrame).
    FeaturePipelines features_;
    bool textureTransform_ = false; // F15.1 (ticket 22): the scene has per-slot texture transforms (bin pipelines compile them)
    std::array<pipe::PipelineHandle, VISIBILITY_CLASSES> servedResolve_{};
    pipe::PipelineHandle servedGeneric_ = pipe::INVALID_PIPELINE, servedTile_ = pipe::INVALID_PIPELINE,
                         servedFallback_ = pipe::INVALID_PIPELINE;
    bool binning_;
    // F15.2: per-class feature bins of a scene with material features (binning == 2 in the params).
    pipe::PipelineHandle binPipeline(u32 mask, bool overflow);
    bool forceAllLobes_ = false, featureBins_ = false;
    u32 dropLobes_ = 0, classBudget_ = VISIBILITY_FEATURE_CLASSES;
    MaterialClassTable classTable_;
    std::array<pipe::PipelineHandle, VISIBILITY_FEATURE_CLASSES> servedBins_{};
    pipe::PipelineHandle servedOverflow_ = pipe::INVALID_PIPELINE;
    std::unordered_map<u64, pipe::PipelineHandle> binHandles_;
    u64 featureBinnedFrames_ = 0;
    u32 loggedOverflowClasses_ = 0;
    bool lighting_ = false;
    MTL::DepthStencilState* lightingFallbackDepth_ = nullptr;
    MTL::GPUAddress lightingAddress_ = 0;
    rg::TextureRef sun_, direct_, indirect_, indirectBack_;
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
    std::array<MTL::Buffer *, 9> readbacks_{};
    std::array<u64, 9> pitches_{};
    MTL::Buffer *currentReadback_ = nullptr;
    MTL::Buffer *previousReadback_ = nullptr;
    u32 readWidth_ = 0, readHeight_ = 0;
    bool readColorFloat32_ = false,readExposureFloat32_ = false;
    u64 readPoseCapacity_ = 0;
    bool useBinning_ = false;
    u64 binnedFrames_ = 0, genericFrames_ = 0;
    std::array<float, 3> background_{0.02f, 0.025f, 0.035f}; // legacy clear colour
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
    std::array<MTL4::ArgumentTable *, 7> tables_{};
    // F15.1 (ticket 17): diffuse-irradiance target, clear and diffusion passes (tables_ 5 and 6).
    bool subsurfaceAllowed_ = false, subsurface_ = false;
    float sssNear_ = 0.1f, sssTanX_ = 1.0f, sssTanY_ = 1.0f;
    pipe::PipelineHandle sssClear_, sssDiffuse_, fallbackSss_;
    rg::TextureRef sss_, sssResult_, sssGain_;
    // Ticket 17/18: the RT walk tier (SSS_OUTPUT == 2): the kernel replaces the diffusion pass.
    AccelerationStructures* walkRt_ = nullptr;
    std::unique_ptr<RtConsumer> walkConsumer_;
    pipe::PipelineHandle sssWalk_ = pipe::INVALID_PIPELINE;
    bool walkWanted_ = false, walk_ = false;
    u32 walkSamples_ = 4, walkSteps_ = 512, walkFlags_ = 0;
    MTL::GPUAddress walkParamsAddress_ = 0, walkTableAddress_ = 0;
    [[nodiscard]] u32 sssOutput() const { return subsurface_ ? (walk_ ? 2u : 1u) : 0u; }
    MTL::GPUAddress sssParamsAddress_ = 0;
    pipe::PipelineHandle clear_, classify_, generic_, present_, fallback_, tile_;
    pipe::PipelineHandle adaptivePipeline_, saveHistory_;
    std::array<pipe::PipelineHandle, VISIBILITY_CLASSES> resolve_{};
    rg::TextureRef visibility_, depth_;
    rg::TextureRef composedColor_;
    std::array<rg::TextureRef, 6> outputs_{};
    rg::BufferRef bins_;
};
} // namespace phosphor
