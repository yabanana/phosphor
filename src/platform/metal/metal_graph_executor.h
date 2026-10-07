#pragma once

#include "core/types.h"
#include "core/worker_pool.h"
#include "platform/metal/metal_context.h"
#include "platform/metal/transient_heap.h"
#include "rendergraph/render_graph.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace phosphor {

class GpuTimestamps;
class PipelineCache;

[[nodiscard]] MTL::PixelFormat toMetalFormat(rg::Format format);
[[nodiscard]] MTL::Stages      toMetalStages(rg::Stages stages);

// ---------------------------------------------------------------------------
// MetalGraphExecutor -- runs a compiled render graph (F2) on Metal 4.
//
// compile() is the slow path, taken only when the graph changes (resize,
// bench switch, debug flags): it compiles the portable graph, creates the
// transient resources (memoryless textures through GpuMemory, the others
// placed in the TransientHeap at the offsets of the aliasing plan) and one
// persistent MTL4::RenderPassDescriptor per render group with the deduced
// load/store actions.  execute() is the per-frame path and allocates
// nothing (O7): it opens one encoder per EncoderPlan, encodes the planned
// barriers and calls every pass's execute callback.
//
// F2.5: a raster pass with parallelChunks > 1 is encoded by several threads
// (WorkerPool).  Its render group becomes one render pass split across
// command buffers: the part before the pass in the frame's command buffer
// (encoder Suspending), one command buffer per chunk (Resuming|Suspending),
// and a tail command buffer (Resuming) with the rest of the group and every
// later encoder.  Each extra command buffer has its own allocator per frame
// slot; all are committed together, in order, by MetalContext::submitFrame.
//
// F2.6: passes on Queue::AsyncCompute run on the context's second queue.
// The frame becomes a list of submissions cut at the graph's QueueSync
// points (compile() splits encoders there): a submission waits on the other
// queue's timeline before its commit and signals its own after it.  The
// first async submission of frame N also waits for graphics frame N-1, and
// the first graphics submission of frame N for the async work of frame N-1
// (the two queues share transient memory and persistent buffers across
// frames).
//
// F4.1: with setTimestamps(), compile() builds the graph's rg::TimingPlan and
// execute() writes a command-buffer timestamp at the start of every commit
// and one timestamp at the end of every timed unit (render group, compute
// pass).  Nothing else changes in the encoded frame.
//
// Imported resources (drawable, readback buffers) are bound every frame
// with bindTexture()/bindBuffer().  Imported attachments are removed from
// the persistent descriptors right after the encoder is created: the
// descriptor would otherwise keep the drawable alive.
// ---------------------------------------------------------------------------

class MetalGraphExecutor {
public:
  explicit MetalGraphExecutor(MetalContext &context, PipelineCache *pipelines = nullptr);
  ~MetalGraphExecutor();

  MetalGraphExecutor(const MetalGraphExecutor &) = delete;
  MetalGraphExecutor &operator=(const MetalGraphExecutor &) = delete;

  /// Compile `graph` and rebuild the physical resources.  `graph` must
  /// outlive the executor's use of it (execute() calls its callbacks).
  /// Returns false (and logs the errors) if the graph is invalid.
  bool compile(const rg::RenderGraph &graph, const rg::CompileOptions &options = {});

  [[nodiscard]] const rg::CompiledGraph &compiled() const { return compiled_; }
  [[nodiscard]] bool valid() const { return graph_ && compiled_.ok; }
  /// Number of successful compile() calls (graph cache diagnostics).
  [[nodiscard]] u32 compileCount() const { return compileCount_; }

  /// F4.1: time the units of the graph (null: no timestamps).  Takes effect
  /// at the next compile().
  void setTimestamps(GpuTimestamps *timestamps) { timestamps_ = timestamps; }

  void bindTexture(rg::TextureRef texture, MTL::Texture *physical);
  void bindBuffer(rg::BufferRef buffer, MTL::Buffer *physical);
  void bindAccelerationStructure(rg::AccelerationStructureRef structure, MTL::AccelerationStructure *physical);

  /// Encode the whole graph into the frame's command buffer.
  void execute(MetalContext::Frame &frame);

  [[nodiscard]] MTL::Texture *texture(u32 resource) const { return textures_[resource]; }
  [[nodiscard]] MTL::Buffer *buffer(u32 resource) const { return buffers_[resource]; }

private:
    class Sizer;
    class Context;

    void releaseResources();
    bool createResources();
    void buildPassDescriptors();
    void encodeBarriers(MTL4::CommandEncoder* encoder, u32 position) const;
    void runPass(MTL4::CommandEncoder* encoder, u32 position, const MetalContext::Frame& frame, u32 chunk = 0,
                 u32 chunks = 1) const;
    void setImportedAttachments(u32 group, bool bind);
    /// Encode a render group whose pass at `splitPosition` is split in chunks;
    /// returns the command buffer that later encoders must use.
    MTL4::CommandBuffer* encodeSplitGroup(MetalContext::Frame& frame, MTL4::CommandBuffer* cmd, u32 group,
                                          u32 splitPosition);
    /// Begin extra command buffer `index` of the frame (F2.5/F2.6) for the
    /// graphics queue (or the async queue) and record it in the submissions.
    MTL4::CommandBuffer* beginExtraCommandBuffer(MetalContext::Frame& frame, u32 index, bool async = false);
    u32  openSubmission(MetalContext::Frame& frame, MetalContext::SubmitQueue queue, u64 waitValue, u64 waitFrame);
    void addBuffer(MetalContext::Frame& frame, bool async, MTL4::CommandBuffer* cmd);
    void ensureParallelResources(u32 maxChunks);
    static void encodeChunkJob(void* user, u32 chunk);
    /// F2.5: end the current graphics submission before a split render pass
    /// (it signals the split fence) and open the next (it waits for it).
    void cutSubmissionForSplit(MetalContext::Frame& frame);

    PipelineCache *pipelines_ = nullptr;
    u32 externalAnchor_ = ~0u;
    MTL::Fence *externalFence_ = nullptr;
    MetalContext&            context_;
    TransientHeap            heap_;
    GpuTimestamps*           timestamps_ = nullptr;
    std::vector<u32>         unitOfPosition_;   // F4.1: timed unit ending at a position, or ~0u
    const rg::RenderGraph*   graph_ = nullptr;
    rg::CompiledGraph        compiled_;
    u32                      compileCount_ = 0;

    std::vector<MTL::TextureUsage> usage_;     // per resource, from the accesses
    std::vector<MTL::Texture*> textures_;      // per resource (imported: bound per frame)
    std::vector<MTL::Buffer*>  buffers_;
    std::vector<MTL::AccelerationStructure*> accelerationStructures_; // imported, owned by RT subsystem
    std::vector<bool>          owned_;         // created here (released on recompile)
    std::vector<MTL4::RenderPassDescriptor*> passDescriptors_; // per render group
    std::vector<NS::String*>   groupLabels_;   // per render group
    std::vector<NS::String*>   passLabels_;    // per pass
    std::vector<NS::String*>   encoderLabels_; // per compute encoder (null for raster)
    std::vector<u32>           barrierIndex_;  // per position: index into compiled_.barriers or ~0u
    std::vector<u32>           splitPosition_; // per render group: position of its split pass or ~0u
    // F2.6: timeline offsets (QueueSync::value) per position, 0 = none.
    std::vector<u32>           waitBefore_;
    std::vector<u32>           signalAfter_;
    bool                       segmented_ = false; // the graph uses the async queue
    // F2.5: the graph has a split render pass: same-queue fence between the
    // commits before / from it, and between frames (values monotonic).
    bool                       hasSplit_  = false;
    MTL::SharedEvent*          splitFence_ = nullptr;
    u64                        splitFenceValue_ = 0;
    // Per-frame submission state (execute()).
    u32                        graphicsSub_ = 0;
    u32                        asyncSub_    = ~0u;

    // F2.5 parallel encoding: extra command buffers [index][slot] with their
    // allocators, and the threads that encode the chunks.
    struct ExtraCommandBuffer {
        std::array<u64, METAL_FRAMES_IN_FLIGHT> residencyGenerations{};
        std::array<MTL4::CommandAllocator*, METAL_FRAMES_IN_FLIGHT> allocators{};
        std::array<MTL4::CommandBuffer*, METAL_FRAMES_IN_FLIGHT>    buffers{};
    };
    std::vector<ExtraCommandBuffer> extra_;
    std::unique_ptr<WorkerPool>     workers_;
    // Per-frame state of the chunk jobs (no allocation per frame).
    struct ChunkJob {
        const MetalGraphExecutor*    executor = nullptr;
        const MetalContext::Frame*   frame    = nullptr;
        u32                          position = 0;
        u32                          chunks   = 1;
        std::array<MTL4::RenderCommandEncoder*, MetalContext::MAX_FRAME_COMMAND_BUFFERS> encoders{};
    };
    ChunkJob chunkJob_;
};

} // namespace phosphor
