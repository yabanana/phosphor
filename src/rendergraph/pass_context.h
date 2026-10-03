#pragma once

#include "rendergraph/render_graph.h"
#include <stdexcept>

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
    /// Only for External passes: borrowed native command buffer and fence.
    /// The executor owns encoder boundaries, dependencies and fence lifetime.
    [[nodiscard]] virtual void *commandBuffer() const { return nullptr; }
    [[nodiscard]] virtual void *externalFence() const { return nullptr; }

    struct ExternalDependency {
        void *inputReady = nullptr, *outputReady = nullptr;
        u64 value = 0;
        void (*resume)(PassContext &, void *) = nullptr;
        void *user = nullptr;
        void (*submitted)(void *, u64) = nullptr;
        void *submissionUser = nullptr;
    };
    // Split after the producer commands, signal inputReady, wait outputReady,
    // then call resume to encode the consumer. Events and resources must live
    // through the frame; the backend owns the submission boundary.
    virtual void externalDependency(const ExternalDependency &) {
        throw std::logic_error("External queue dependencies unsupported by this context");
    }

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
