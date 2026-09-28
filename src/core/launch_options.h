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
//   --switch-every N   switch to the next test bench every N frames, through
//                      the same path as the 1-7 hotkeys (switching tests)
//   --simulate-pressure  inject a memory-pressure warning at frame 10 and a
//                      critical notification at frame 20 (F1.5 tests)
//   --memory-stress N  create/destroy N GPU resources, check that device
//                      memory does not grow, exit (F1.6); exit code 1 on failure
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
    u32         switchEvery   = 0; // 0 = never
    bool        simulatePressure = false;
    u32         memoryStress  = 0; // cycles; 0 = off
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
