#pragma once

#include "core/types.h"
#include "platform/metal/metal_context.h"

#include <string>

namespace phosphor {

// ---------------------------------------------------------------------------
// GpuCapture (F4.3) -- .gputrace documents of whole frames from the CLI app.
//
// Measured (opt-log, F4 spike 4): a process not launched by Xcode has no
// capture layer ("Capture layer is not inserted") unless
// MTL_CAPTURE_ENABLED=1 is set before the Metal device is created; setting
// it with setenv() in the process works.  enableLayer() does that, and the
// engine calls it only when a capture option is given, so a run without
// capture options is unchanged.
//
// A capture covers one frame: startCapture (device scope: both queues)
// before the frame is encoded, stopCapture after it is submitted.  Captures
// cannot be taken after the fact, so a "slow frame" trigger arms the NEXT
// frame (documented in the log line).
// ---------------------------------------------------------------------------

class GpuCapture {
public:
    /// Insert the capture layer: call before creating the MetalContext.
    static void enableLayer();

    GpuCapture(MetalContext& context, std::string directory, u32 maxCaptures);

    GpuCapture(const GpuCapture&) = delete;
    GpuCapture& operator=(const GpuCapture&) = delete;

    /// Capture the next frame; `reason` goes into the log and the file name
    /// ("key", "frame", "over").  Ignored once maxCaptures were taken.
    void request(const char* reason);
    [[nodiscard]] bool armed() const { return armed_; }

    /// Before the frame is encoded: starts the capture if armed.
    void beginFrame(u64 frameIndex);
    /// After MetalContext::submitFrame: ends a capture in progress.
    void endFrame();

    [[nodiscard]] u32 captures() const { return captures_; }
    [[nodiscard]] u32 failures() const { return failures_; }
    [[nodiscard]] const std::string& lastPath() const { return lastPath_; }

private:
    MetalContext& context_;
    std::string   directory_;
    u32           maxCaptures_ = 1;
    u32           captures_    = 0;
    u32           failures_    = 0;
    bool          armed_       = false;
    bool          capturing_   = false;
    const char*   reason_      = "";
    std::string   lastPath_;
};

} // namespace phosphor
