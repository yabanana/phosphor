#pragma once

#include "core/types.h"
#include "platform/metal/metal_context.h"
#include "platform/metal/transient_heap.h"
#include "rendergraph/render_graph.h"

#include <string>
#include <vector>

namespace phosphor {

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
// Imported resources (drawable, readback buffers) are bound every frame
// with bindTexture()/bindBuffer().  Imported attachments are removed from
// the persistent descriptors right after the encoder is created: the
// descriptor would otherwise keep the drawable alive.
// ---------------------------------------------------------------------------

class MetalGraphExecutor {
public:
    explicit MetalGraphExecutor(MetalContext& context);
    ~MetalGraphExecutor();

    MetalGraphExecutor(const MetalGraphExecutor&) = delete;
    MetalGraphExecutor& operator=(const MetalGraphExecutor&) = delete;

    /// Compile `graph` and rebuild the physical resources.  `graph` must
    /// outlive the executor's use of it (execute() calls its callbacks).
    /// Returns false (and logs the errors) if the graph is invalid.
    bool compile(const rg::RenderGraph& graph, const rg::CompileOptions& options = {});

    [[nodiscard]] const rg::CompiledGraph& compiled() const { return compiled_; }
    [[nodiscard]] bool                     valid()    const { return graph_ && compiled_.ok; }
    /// Number of successful compile() calls (graph cache diagnostics).
    [[nodiscard]] u32                      compileCount() const { return compileCount_; }

    void bindTexture(rg::TextureRef texture, MTL::Texture* physical);
    void bindBuffer(rg::BufferRef buffer, MTL::Buffer* physical);

    /// Encode the whole graph into the frame's command buffer.
    void execute(MetalContext::Frame& frame);

    [[nodiscard]] MTL::Texture* texture(u32 resource) const { return textures_[resource]; }
    [[nodiscard]] MTL::Buffer*  buffer(u32 resource)  const { return buffers_[resource]; }

private:
    class Sizer;
    class Context;

    void releaseResources();
    bool createResources();
    void buildPassDescriptors();
    void encodeBarriers(MTL4::CommandEncoder* encoder, u32 position) const;
    void runPass(MTL4::CommandEncoder* encoder, u32 position, const MetalContext::Frame& frame) const;

    MetalContext&            context_;
    TransientHeap            heap_;
    const rg::RenderGraph*   graph_ = nullptr;
    rg::CompiledGraph        compiled_;
    u32                      compileCount_ = 0;

    std::vector<MTL::TextureUsage> usage_;     // per resource, from the accesses
    std::vector<MTL::Texture*> textures_;      // per resource (imported: bound per frame)
    std::vector<MTL::Buffer*>  buffers_;
    std::vector<bool>          owned_;         // created here (released on recompile)
    std::vector<MTL4::RenderPassDescriptor*> passDescriptors_; // per render group
    std::vector<NS::String*>   groupLabels_;   // per render group
    std::vector<NS::String*>   passLabels_;    // per pass
    std::vector<NS::String*>   encoderLabels_; // per compute encoder (null for raster)
    std::vector<u32>           barrierIndex_;  // per position: index into compiled_.barriers or ~0u
};

} // namespace phosphor
