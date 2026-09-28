#pragma once

#include "core/types.h"
#include "renderdoc/renderdoc_app.h"

namespace phosphor {

// ---------------------------------------------------------------------------
// RenderDocCapture -- optional integration with the RenderDoc in-app API.
//
// Attempts to load the RenderDoc shared library at construction time
// (which succeeds only when the process was launched from within RenderDoc).
// If not available, all methods are safe no-ops.
// ---------------------------------------------------------------------------

class RenderDocCapture {
public:
    RenderDocCapture();
    ~RenderDocCapture() = default;

    RenderDocCapture(const RenderDocCapture&) = delete;
    RenderDocCapture& operator=(const RenderDocCapture&) = delete;

    /// Returns true if the RenderDoc API was successfully loaded.
    [[nodiscard]] bool isAvailable() const;

    /// Request a single-frame capture.  The actual capture will be started
    /// on the next beginFrame() call.
    void triggerCapture();

    /// Call at the start of every frame.  Begins a RenderDoc capture if one
    /// was previously requested via triggerCapture().
    void beginFrame();

    /// Call at the end of every frame.  Ends the in-progress capture.
    void endFrame();

    [[nodiscard]] bool isCapturing()    const { return capturing_; }
    [[nodiscard]] u32  getCaptureCount() const { return captureCount_; }

private:
    void tryLoadAPI();

    ::RENDERDOC_API_1_6_0* api_ = nullptr;
    bool captureRequested_           = false;
    bool capturing_                  = false;
    u32  captureCount_               = 0;
};

} // namespace phosphor
