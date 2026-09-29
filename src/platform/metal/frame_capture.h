#pragma once

#include "core/types.h"
#include "platform/metal/metal_context.h"

#include <string>

namespace phosphor {

// ---------------------------------------------------------------------------
// FrameCapture -- copies a frame's drawable into a shared buffer and writes
// it as a PNG (--capture).  Used for visual checks of the test benches.
//
// The copy is a blit pass of the render graph with a side effect (F2.7): the
// graph orders it after every pass that renders to the drawable and emits
// the barrier.  The readback buffer is created with the graph (resize), not
// in the frame.
// ---------------------------------------------------------------------------

class FrameCapture {
public:
    explicit FrameCapture(MetalContext& context) : context_(context) {}
    ~FrameCapture();

    FrameCapture(const FrameCapture&) = delete;
    FrameCapture& operator=(const FrameCapture&) = delete;

    /// Create (or resize) the readback buffer for a width x height drawable.
    void prepare(u32 width, u32 height);
    [[nodiscard]] MTL::Buffer* readback() const { return readback_; }
    [[nodiscard]] u64 readbackSize() const { return static_cast<u64>(width_) * height_ * 4; }

    /// Encode the copy of `source` (the drawable) into the readback buffer.
    void encode(MTL4::ComputeCommandEncoder* encoder, MTL::Texture* source);

    /// Write the captured frame to `path`; the frame must have completed
    /// (call MetalContext::waitIdle() first).  Returns false on failure.
    bool writePng(const std::string& path);

    /// Raw BGRA8 pixels of the captured frame (after waitIdle), width x height.
    [[nodiscard]] const u8* pixels() const { return readback_ ? static_cast<const u8*>(readback_->contents()) : nullptr; }
    [[nodiscard]] u32 width() const { return width_; }
    [[nodiscard]] u32 height() const { return height_; }

private:
    MetalContext& context_;
    MTL::Buffer*  readback_ = nullptr;
    u32           width_    = 0;
    u32           height_   = 0;
};

} // namespace phosphor
