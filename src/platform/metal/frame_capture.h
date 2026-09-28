#pragma once

#include "core/types.h"
#include "platform/metal/metal_context.h"

#include <string>

namespace phosphor {

// ---------------------------------------------------------------------------
// FrameCapture -- copies a frame's drawable into a shared buffer and writes
// it as a PNG (--capture).  Used for visual checks of the test benches; not
// meant for per-frame use.
// ---------------------------------------------------------------------------

class FrameCapture {
public:
    explicit FrameCapture(MetalContext& context) : context_(context) {}
    ~FrameCapture();

    FrameCapture(const FrameCapture&) = delete;
    FrameCapture& operator=(const FrameCapture&) = delete;

    /// Encode the copy of the drawable after every pass that renders to it.
    void encode(MetalContext::Frame& frame);

    /// Write the captured frame to `path`; the frame must have completed
    /// (call MetalContext::waitIdle() first).  Returns false on failure.
    bool writePng(const std::string& path);

private:
    MetalContext& context_;
    MTL::Buffer*  readback_ = nullptr;
    u32           width_    = 0;
    u32           height_   = 0;
};

} // namespace phosphor
