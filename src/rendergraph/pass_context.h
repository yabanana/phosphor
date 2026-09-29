#pragma once

#include "rendergraph/render_graph.h"

namespace phosphor::rg {

// ---------------------------------------------------------------------------
// PassContext -- what a pass's execute callback receives from the backend.
// Pointers are backend objects (on Metal: MTL4::RenderCommandEncoder* or
// MTL4::ComputeCommandEncoder*, MTL::Texture*, MTL::Buffer*), kept opaque
// so that phosphor_core stays free of Apple headers.
// ---------------------------------------------------------------------------

class PassContext {
public:
    virtual ~PassContext() = default;

    /// Encoder of the pass's type, already inside the (fused) render pass.
    [[nodiscard]] virtual void* encoder() const = 0;
    [[nodiscard]] virtual void* texture(TextureRef texture) const = 0;
    [[nodiscard]] virtual void* buffer(BufferRef buffer) const = 0;
    /// F2.5: which piece of a pass with parallelChunks > 1 this call encodes.
    [[nodiscard]] virtual u32 chunk() const = 0;
    [[nodiscard]] virtual u32 chunkCount() const = 0;
    /// Index of the frame being recorded.
    [[nodiscard]] virtual u64 frameIndex() const = 0;
};

} // namespace phosphor::rg
