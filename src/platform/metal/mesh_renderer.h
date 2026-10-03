#pragma once

#include "core/launch_options.h"
#include "core/types.h"
#include "pipeline/pipeline_registry.h"
#include "platform/metal/hiz_builder.h"
#include "platform/metal/metal_context.h"
#include "renderer/gpu_types.h"
#include "rendergraph/render_graph.h"

#include <array>
#include <memory>
#include <optional>
#include <vector>

namespace phosphor {

class GpuScene;
class PipelineCache;
class SceneRenderer;
class SceneStore;

// ---------------------------------------------------------------------------
// MeshRenderer -- the F6 mesh path (--geometry-path mesh): object + mesh
// shaders over meshlets, conservative meshlet culling and two-phase Hi-Z
// occlusion.  Contract and pass order: renderer/meshlet_layout.h; culling
// maths: renderer/meshlet_cull_math.h; CPU references:
// renderer/meshlet_cull_reference.h.
//
// Builds on the F5 GPU scene (SceneRenderer: persistent instances, Instance
// cull, ICB draw build) and keeps its contracts: same instances / materials /
// textures / lights, same forward fragment shader and variants, same cull
// classes and states (the mesh draws of class c run with class c's state).
// The F5 ICB stays in the frame as the OVERFLOW fallback: Draw build writes
// draws only when the meshlet candidates exceeded the capacity (gate word),
// so a frame is never truncated.
//
// Resources: meshlet geometry (private, rebuilt on a geometry change), per
// frame slot the candidate / phase-B lists, flags, ranges, indirect
// arguments, counters and gate (shared: the self-check reads them after the
// frame), and the Hi-Z pyramids (HiZBuilder).  The candidate capacity is
// meshletCandidateCapacity(buckets) (an upper bound of any frame's
// candidates), recomputed at structure changes; lists only grow (x1.5).
//
// History (F6.5): per view (one view) the history pyramid, the view it was
// rendered with and a generation (engine); invalid after creation, resize,
// bench switch, camera cut or --history-reset-every.  The history
// only orders work between phase A and phase B: every candidate it rejects
// is retested in phase B against the CURRENT frame's depth.
// ---------------------------------------------------------------------------

class MeshRenderer {
public:
    struct Options {
        MeshletCull        cull      = MeshletCull::TwoPhase;
        HiZBuilder::Backend hiz      = HiZBuilder::Backend::Compute;
        MeshletDebugView   debugView = MeshletDebugView::None;
        u32                debugHiZLevel = 3; // --debug-view hiz: pyramid level shown
        /// --debug-meshlets: the graph carries the read-back passes (only
        /// then: an empty blit encoder is dropped by Metal and would leave
        /// the next timed unit without a valid timestamp, F4.1).
        bool               checks = false;
        bool visibility = false; // F7: R32Uint opaque + masked raster
        /// F6.2 APPROXIMATE size cull: meshlets whose projected bound covers
        /// less than minPixels^2 pixels (0 = off, the exact preset).
        float              minPixels = 0.0f;
        /// F6.3 option: per-triangle facing cull + compaction in the mesh
        /// shader (meshlet_mesh_tricull).  Off by default: slower on M5 Max.
        bool               triangleCull = false;
        /// Spike S2: mesh-only pipeline (no object stage, no culling).
        bool               objectStage = true;
        u32                salt      = 0;
        bool               genericOnly = false;
        std::optional<u32> forceVariant;
    };

    struct FrameParams {
        u32                  slot   = 0;
        u32 viewIndex = 0;
        u32                  width  = 0;
        u32                  height = 0;
        u32 allocationWidth = 0, allocationHeight = 0; // F8 backing extent, 0 = logical
        GPUMeshletCullParams cull{}; // viewProj, planes, camera, near filled by the engine
        float                prevViewProj[16]{}; // view of the history (when valid)
        bool                 historyValid = false;
        MeshletCorruption    corrupt = MeshletCorruption::None; // this frame only (self-check)
    };

    MeshRenderer(MetalContext& context, PipelineCache& pipelines, SceneRenderer& scene, const Options& options);
    ~MeshRenderer();
    MeshRenderer(const MeshRenderer&) = delete;
    MeshRenderer& operator=(const MeshRenderer&) = delete;

    /// Upload the meshlet geometry if the scene geometry changed (blocking).
    void syncGeometry(const GpuScene& scene);
    /// Bench switch (after SceneRenderer::loadScene, before the switch's
    /// collectGarbage): the lists are re-sized to the new scene right away,
    /// so the old ones are freed by that collection and no allocation or
    /// deferred release lands in the frames after the switch (hitch check).
    void loadScene(const SceneStore& store, const GpuScene& scene);
    /// Frame: capacities, parameters into the frame ring, bindings, variant.
    /// After SceneRenderer::prepareFrame (uses its constants/lights addresses).
    void prepareFrame(const SceneStore& store, const GpuScene& scene, const FrameParams& params);
    void requestAllVariants();
    void setTemporalInputs(MTL::GPUAddress params) {
        for (auto *table : drawTables_)
            table->setAddress(params, 16);
    }

    // --- Render graph -------------------------------------------------------------
    /// The candidate pass (SceneRenderer::addPassesToGraph's afterCull hook);
    /// returns the buffer holding the draw gate.
    rg::BufferRef addCandidatePass(rg::RenderGraph& graph);
    /// Phase A ("Forward"), and with two-phase culling Hi-Z A, Meshlet B,
    /// Forward B and Hi-Z final.  `color` is updated to the last version.
    /// Returns the depth texture's last version.
    rg::TextureRef addRasterPasses(rg::RenderGraph& graph, rg::TextureRef& color, u32 width, u32 height);
    /// Import the per-frame resources of the graph for this frame (call
    /// every frame after the graph is compiled, before execute).
    void bindFrame(class MetalGraphExecutor& executor, u32 slot) const;

    // --- Diagnostics ----------------------------------------------------------------
    struct FrameSet {
        MTL::Buffer* candidates = nullptr; // GPUMeshletCandidate[capacity]
        MTL::Buffer* bFlags     = nullptr; // u32[capacity]
        MTL::Buffer* bList      = nullptr; // GPUMeshletCandidate[capacity]
        MTL::Buffer* groupSums  = nullptr; // u32[groups * 3]
        MTL::Buffer* bSums      = nullptr; // u32[candidateGroups * 3]
        MTL::Buffer* ranges     = nullptr; // GPUMeshletDrawRange[6]
        MTL::Buffer* args       = nullptr; // u32[6 * 3]
        MTL::Buffer* counters   = nullptr; // GPUMeshletCounters
        MTL::Buffer* gate       = nullptr; // u32 (overflow)
        MTL::Buffer* cullParams = nullptr; // GPUMeshletCullParams of the frame (copy, self-check)
        MTL::Buffer* decisions  = nullptr; // u32[2 * capacity], written on self-check frames
    };
    [[nodiscard]] const FrameSet& frame(u32 slot) const { return frames_[slot]; }
    [[nodiscard]] GPUMeshletCounters counters(u32 slot) const;
    [[nodiscard]] u64 capacity() const { return capacity_; }
    [[nodiscard]] u64 version() const { return version_; }
    [[nodiscard]] const HiZBuilder *hiz() const { return hiz_; }
    [[nodiscard]] bool twoPhase() const { return options_.cull == MeshletCull::TwoPhase; }
    [[nodiscard]] rg::BufferRef frameListsRef() const { return graphFrame_; }
    [[nodiscard]] MTL::Buffer *meshletBuffer() const { return meshlets_; }
    [[nodiscard]] MTL::Buffer *meshletVertexBuffer() const { return meshletVertices_; }
    [[nodiscard]] MTL::Buffer *meshletTriangleBuffer() const { return meshletTriangles_; }
    [[nodiscard]] const Options& options() const { return options_; }
    [[nodiscard]] u32 meshletCount() const { return meshletCount_; }
    [[nodiscard]] bool ready() const;
    [[nodiscard]] bool usingFallbackPipeline() const { return usingFallback_; }
    void setDebugView(MeshletDebugView view) { options_.debugView = view; }
    /// Self-check: this frame copies the history pyramid phase A tests
    /// against (at frame start), and at the end the final depth, the current
    /// pyramid (phase B's) and the new history (Hi-Z final's).
    void requestCheckReadback(bool on) { readback_ = on; }
    struct CheckReadback {
        MTL::Buffer* history = nullptr; // pyramid used by phase A
        MTL::Buffer* current = nullptr; // pyramid of phase A's depth (phase B)
        MTL::Buffer* next    = nullptr; // pyramid of the final depth (next frame's history)
        MTL::Buffer* depth   = nullptr; // final depth, width x height floats
    };
    [[nodiscard]] const CheckReadback& checkReadback() const { return check_; }
    [[nodiscard]] u32 depthWidth() const { return width_; }
    [[nodiscard]] u32 depthHeight() const { return height_; }
    [[nodiscard]] const MTL::Buffer* meshletBoundsBuffer() const { return bounds_; }

private:
    void ensureCapacity(u64 capacity, u32 slotCount);
    void releaseFrames();
    void releaseGeometry();
    MTL::Buffer* privateBuffer(const void* data, size_t size, const char* label);
    MTL::Buffer* sharedBuffer(u64 size, const char* label);
    void encodeCandidates(MTL4::ComputeCommandEncoder* enc) const;
    void encodePhaseB(MTL4::ComputeCommandEncoder* enc) const;
    void encodeRaster(MTL4::RenderCommandEncoder *enc, u32 phase, bool alpha = false) const;
    MTL::ComputePipelineState* kernel(pipe::PipelineHandle h) const;

    MetalContext&  context_;
    PipelineCache& pipelines_;
    SceneRenderer& scene_;
    Options        options_;
    std::array<std::unique_ptr<HiZBuilder>, 4> hizViews_{};
    HiZBuilder *hiz_ = nullptr;

    // Pipelines.
    pipe::PipelineHandle generic_ = pipe::INVALID_PIPELINE, debug_ = pipe::INVALID_PIPELINE;
    std::vector<pipe::PipelineHandle> variants_;
    MTL::RenderPipelineState* pipeline_ = nullptr;
    bool usingFallback_ = false;
    pipe::PipelineHandle visibilityOpaque_ = pipe::INVALID_PIPELINE, visibilityAlpha_ = pipe::INVALID_PIPELINE;
    pipe::PipelineHandle kCandCount_ = pipe::INVALID_PIPELINE, kCandScan_ = pipe::INVALID_PIPELINE,
                         kCandWrite_ = pipe::INVALID_PIPELINE, kBCount_ = pipe::INVALID_PIPELINE,
                         kBScan_ = pipe::INVALID_PIPELINE, kBWrite_ = pipe::INVALID_PIPELINE;
    pipe::PipelineHandle hizView_ = pipe::INVALID_PIPELINE; // --debug-view hiz
    MTL4::ArgumentTable* hizViewTable_ = nullptr;
    MTL::Buffer* hizViewParams_ = nullptr;
    MTL4::ArgumentTable* candTable_ = nullptr;
    MTL4::ArgumentTable* bTable_    = nullptr;
    std::array<MTL4::ArgumentTable*, 2> drawTables_{}; // phase A / B

    // Geometry.
    MTL::Buffer* meshlets_ = nullptr;
    MTL::Buffer* meshletVertices_ = nullptr;
    MTL::Buffer* meshletTriangles_ = nullptr;
    MTL::Buffer* bounds_ = nullptr;
    u64 geometryVersion_ = ~u64{0};
    u32 meshletCount_ = 0;

    // Per frame slot lists.
    std::array<FrameSet, METAL_FRAMES_IN_FLIGHT> frames_{};
    u64 capacity_ = 0;
    u32 slotCap_  = 0;
    u64 version_  = 0;
    u64 structureVersion_ = ~u64{0};
    CheckReadback check_;
    bool readback_ = false;

    // Graph.
    rg::BufferRef  graphFrame_;          // the slot's FrameSet (one graph resource)
    rg::TextureRef hizHistoryRef_, hizCurrentRef_;
    rg::BufferRef  readbackRef_;
    rg::TextureRef depthRef_;

    // Frame state.
    FrameParams frame_{};
    u32 width_ = 0, height_ = 0;
    u32 slotCount_ = 0, groups_ = 0, candidateGroups_ = 0;
    MTL::GPUAddress params_ = 0; // GPUMeshletCullParams in the frame ring
    bool corruptDepth_ = false;
};

} // namespace phosphor
