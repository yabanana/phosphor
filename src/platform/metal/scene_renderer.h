#pragma once

#include "core/launch_options.h"
#include "core/types.h"
#include "pipeline/pipeline_registry.h"
#include "platform/metal/gpu_scene_buffers.h"
#include "platform/metal/metal_context.h"
#include "renderer/gpu_types.h"
#include "rendergraph/render_graph.h"

#include <array>
#include <atomic>
#include <optional>
#include <span>
#include <vector>

namespace phosphor {

class GpuScene;
class PipelineCache;
class SceneStore;

// ---------------------------------------------------------------------------
// SceneRenderer -- the scene's GPU work on Metal 4.
//
// F0-F4: forward pass with per-frame instance uploads.  Since F5 the scene
// is persistent on the GPU (GpuSceneBuffers mirrors SceneStore) and the
// frame runs, as render-graph passes declared by Engine::buildFrameGraph:
//   Scene update      delta scatters / full copies (D1), queue clear
//   Scene transforms  procedural motion + hierarchy queues (F5.2, F5.4)
//   Instance cull     (gpu-driven on) frustum/distance/size + stable scan
//   Draw build        (on) one ICB command per bucket (F5.3)
//   Forward           off: one direct draw per bucket; on: per cull class
//                     the class's cull state + executeCommandsInBuffer
// Both modes draw in the order (cull class, mesh, slot) and the vertex
// shader reads instances[visible[instance_id]] (identity list when off), so
// off and on give the same image.  The CPU command count of the scene is
// constant in mode on (O8).
//
// Geometry lives in private buffers rebuilt when the GpuScene's geometry
// version changes.  A dispatch chain inside one scene pass is ordered with
// Dispatch -> Dispatch encoder barriers between its own dispatches (the
// graph orders passes, not the dispatches inside a pass).
//
// Since F3.3 the forward pipeline is a specialised variant chosen per frame
// (light types, emissive materials, debug mode); a variant not compiled yet
// is drawn with the generic pipeline until the cache swaps it in.
// ---------------------------------------------------------------------------

class SceneRenderer {
public:
    struct FrameParams {
        u32         slot   = 0;     // frame slot (per-frame buffers)
        u32         width  = 0;
        u32         height = 0;
        GpuDrivenMode mode = GpuDrivenMode::Off;
        GPUCullParams cull{};       // planes etc. (slotCount filled here)
        const float* motionSinCos = nullptr; // SCENE_MOTION_CLASSES * 2 values
        SceneCorruption corrupt = SceneCorruption::None; // self-check negative controls (this frame only)
    };

    SceneRenderer(MetalContext& context, PipelineCache& pipelines, u32 salt = 0, bool genericOnly = false,
                  std::optional<u32> forceVariant = std::nullopt);
    ~SceneRenderer();

    SceneRenderer(const SceneRenderer&) = delete;
    SceneRenderer& operator=(const SceneRenderer&) = delete;

    /// Upload vertex/index/mesh-info data if the scene geometry changed (blocking).
    void syncGeometry(const GpuScene& scene);
    /// Bench switch: drop the scene buffers and upload `store` in full (blocking).
    void loadScene(const SceneStore& store);

    /// Frame: stage the store's deltas, write constants/lights/parameters into
    /// the frame ring, bind everything, choose the forward variant.  `store`
    /// must stay alive until the frame is encoded.  Returns the bytes the CPU
    /// wrote for the scene (deltas, copies, constants, lights).
    u64 prepareFrame(const SceneStore& store, std::span<const GPULight> lights, const FrameConstants& constants,
                     MTL::GPUAddress textureTable, const FrameParams& params);

    // --- Render graph (Engine::declareFrameGraph) ----------------------------------
    /// Import the scene's buffers and add the scene passes for `mode` (Scene
    /// update, Scene transforms, and with on Instance cull + Draw build).
    void addPassesToGraph(rg::RenderGraph& graph, GpuDrivenMode mode);
    /// Accesses of a pass that draws the scene (Forward, overlays): instances,
    /// materials (vertex + fragment) and, with on, the ICB / visible list as
    /// indirect arguments at the Vertex stage (spike S5).
    void declareDrawReads(rg::PassBuilder& builder) const;

    // --- Pass callbacks -------------------------------------------------------------
    void encodeUpdate(MTL4::ComputeCommandEncoder* encoder) const;
    void encodeTransforms(MTL4::ComputeCommandEncoder* encoder) const;
    void encodeCull(MTL4::ComputeCommandEncoder* encoder) const;
    void encodeDrawBuild(MTL4::ComputeCommandEncoder* encoder) const;
    /// Forward draws, chunk `chunk` of `chunks` (F2.5): off splits the bucket
    /// list, on splits each class range.
    void encode(MTL4::RenderCommandEncoder* encoder, u32 chunk = 0, u32 chunks = 1) const;
    /// F4.7: the same draws (same bindings, viewport, cull state, ICB) with
    /// another pipeline.  With `depthTest` the reverse-Z depth state is used.
    void encodeOverlay(MTL4::RenderCommandEncoder* encoder, pipe::PipelineHandle pipeline, bool depthTest) const;

    // --- Diagnostics -------------------------------------------------------------
    [[nodiscard]] u32 lastTriangleCount() const { return lastTriangles_; }
    [[nodiscard]] bool usingFallback() const { return usingFallback_; }
    void requestAllVariants();
    [[nodiscard]] static MTL::PixelFormat depthFormat() { return MTL::PixelFormatDepth32Float; }
    [[nodiscard]] const GpuSceneBuffers& buffers() const { return buffers_; }
    /// Commands the CPU encoded for the scene in the last encoded frame (all
    /// scene passes + forward + overlays).
    [[nodiscard]] u32 cpuCommands() const;
    /// Counters of the frame that last used `slot` (call once that frame is
    /// complete, i.e. at the slot's next beginFrame).
    [[nodiscard]] GPUSceneCounters counters(u32 slot) const;
    /// True once every scene kernel pipeline is usable (the scene passes do
    /// nothing before that).
    [[nodiscard]] bool kernelsReady() const;

private:
    MTL::Buffer* createPrivateBuffer(const void* data, size_t size, const char* label);
    void releaseGeometry();
    void encodeForward(MTL4::RenderCommandEncoder* encoder, MTL::RenderPipelineState* pipeline,
                       MTL::DepthStencilState* depthState, u32 chunk, u32 chunks, u32 counterIndex) const;
    MTL::ComputePipelineState* kernel(pipe::PipelineHandle h) const;
    void count(u32 index, u32 n) const { cpuCommands_[index].fetch_add(n, std::memory_order_relaxed); }

    MetalContext& context_;
    PipelineCache& pipelines_;
    u32                       salt_ = 0;
    bool                      genericOnly_ = false;
    std::optional<u32>        forceVariant_;
    pipe::PipelineHandle      generic_ = pipe::INVALID_PIPELINE;
    std::vector<pipe::PipelineHandle> variants_;
    MTL::RenderPipelineState* pipeline_ = nullptr;
    bool                      usingFallback_ = false;
    MTL::DepthStencilState*   depthState_ = nullptr;
    MTL4::ArgumentTable*      arguments_  = nullptr; // forward + overlays

    // Scene kernels (shaders/gpu_scene.metal, shaders/transforms.metal).
    pipe::PipelineHandle kClear_ = pipe::INVALID_PIPELINE, kScatter_ = pipe::INVALID_PIPELINE,
                         kMotion_ = pipe::INVALID_PIPELINE, kQueueArgs_ = pipe::INVALID_PIPELINE,
                         kHier_ = pipe::INVALID_PIPELINE, kCullFlags_ = pipe::INVALID_PIPELINE,
                         kCullScan_ = pipe::INVALID_PIPELINE, kCullWrite_ = pipe::INVALID_PIPELINE,
                         kDrawBuild_ = pipe::INVALID_PIPELINE;
    // One table per scene pass (passes of one compute encoder may run back to
    // back: separate tables keep their bindings independent).
    MTL4::ArgumentTable* updateTable_    = nullptr;
    MTL4::ArgumentTable* transformTable_ = nullptr;
    MTL4::ArgumentTable* cullTable_      = nullptr;
    MTL4::ArgumentTable* drawTable_      = nullptr;

    MTL::Buffer* vertexBuffer_ = nullptr;
    MTL::Buffer* indexBuffer_  = nullptr;
    MTL::Buffer* meshBuffer_   = nullptr; // GPUMeshInfo[]
    u64          geometryVersion_ = ~u64{0};

    GpuSceneBuffers buffers_;
    // Graph resources: persistent scene data and the per-frame lists/ICB
    // (latest versions, after the scene passes).
    rg::BufferRef   graphData_;
    rg::BufferRef   graphFrame_;
    GpuDrivenMode   graphMode_ = GpuDrivenMode::Off;

    // Frame state (prepareFrame -> encode*).
    const SceneStore* store_ = nullptr;
    FrameParams       frame_{};
    u32               slotCount_   = 0;
    u32               cullGroups_  = 0;
    u32               motionCount_ = 0;
    u32               commandCount_ = 0;
    MTL::GPUAddress   clearParams_   = 0;
    MTL::GPUAddress   scatterParams_ = 0; // 4 x GPUScatterParams
    std::array<MTL::GPUAddress, 4> scatterRecords_{};
    std::array<u32, 4>             scatterCounts_{};
    MTL::GPUAddress   motionFrame_   = 0;
    MTL::GPUAddress   hierParams_    = 0; // SCENE_MAX_LEVELS x GPUHierParams
    MTL::GPUAddress   queue0_        = 0;
    MTL::GPUAddress   cullParams_    = 0;
    MTL::GPUAddress   drawParams_    = 0;
    u32 lastTriangles_ = 0;
    // CPU commands per encoding thread: [0] scene passes + overlays, [1 + c] forward chunk c.
    static constexpr u32 kMaxChunks = 16;
    mutable std::array<std::atomic<u32>, kMaxChunks + 1> cpuCommands_{};
};

} // namespace phosphor
