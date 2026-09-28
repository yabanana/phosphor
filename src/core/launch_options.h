#pragma once

#include "core/types.h"

#include <optional>
#include <string>

namespace phosphor {

// ---------------------------------------------------------------------------
// LaunchOptions -- command-line switches of the app.
//
//   --bench N          start on test bench N (1-based, like the hotkeys)
//   --frames N         benchmark mode: measure N frames, print a summary, exit
//   --warmup N         frames skipped before measuring (default 120)
//   --no-vsync         disable display sync (uncapped frame rate)
//   --no-ui            do not draw the ImGui overlay
//   --fixed-timestep   advance the simulation by 1/60 s per frame regardless
//                      of real time (deterministic captures)
//   --capture FILE     write a frame to FILE (PNG): the last measured frame in
//                      benchmark mode, otherwise the first frame
//   --report FILE      write the benchmark summary to FILE (JSON)
//
// Arguments not starting with "--" are ignored: macOS may add its own
// (e.g. -NSDocumentRevisionsDebugMode when launched from Xcode).
// ---------------------------------------------------------------------------

struct LaunchOptions {
    std::optional<int> bench;  // 0-based TestBenchType index
    u32         frames = 0;    // 0 = interactive
    u32         warmup = 120;
    bool        vsync  = true;
    bool        ui     = true;
    bool        fixedTimestep = false;
    std::string capturePath;
    std::string reportPath;

    /// True when the app runs a fixed number of frames and then exits.
    [[nodiscard]] bool benchmark() const { return frames > 0; }
};

/// Parse argv into `out`.  Returns false and sets `error` on malformed input.
/// `benchCount` bounds the accepted --bench values.
bool parseLaunchOptions(int argc, const char* const* argv, int benchCount,
                        LaunchOptions& out, std::string& error);

} // namespace phosphor
