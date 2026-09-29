#pragma once

#include "core/types.h"

#include <optional>
#include <string>

namespace phosphor {

// ---------------------------------------------------------------------------
// LaunchOptions -- command-line switches of the app.
//
//   --bench N          start on test bench N (1-based, like the hotkeys)
//   --frames N         benchmark mode: measure N frames, print a summary, exit;
//                      keyboard and mouse input is ignored (reproducible runs)
//   --warmup N         frames skipped before measuring (default 120)
//   --no-vsync         disable display sync (uncapped frame rate)
//   --no-ui            do not draw the ImGui overlay
//   --fixed-timestep   advance the simulation by 1/60 s per frame regardless
//                      of real time (deterministic captures)
//   --switch-every N   switch to the next test bench every N frames, through
//                      the same path as the 1-7 hotkeys (switching tests)
//   --resize-every N   resize the window every N frames, alternating between
//                      two sizes (tests the render graph recompilation)
//   --simulate-pressure  inject a memory-pressure warning at frame 10 and a
//                      critical notification at frame 20 (F1.5 tests)
//   --memory-stress N  create/destroy N GPU resources, check that device
//                      memory does not grow, exit (F1.6); exit code 1 on failure
//   --transient-test   aliasing self-test of the transient heap, exit (F1.1)
//   --debug-graph-transients  add a synthetic chain of compute/raster passes on
//                      transient resources (F2.2) whose result is read back and
//                      checked exactly on the CPU; prints a GRAPH-TRANSIENTS line
//                      and exits with code 1 on any mismatch or if nothing was
//                      aliased.  The drawable is untouched.
//   --inject-input     push synthetic key/mouse events every frame (tests
//                      that benchmark mode is immune to user input)
//   --capture FILE     write a frame to FILE (PNG): the last measured frame in
//                      benchmark mode, otherwise the first frame
//   --report FILE      write the benchmark summary to FILE (JSON)
//   --dump-graph FILE  write the frame's render graph (Graphviz dot, with the
//                      estimated DRAM bytes per resource) whenever it is compiled
//   --debug-split-encoding  encode the forward pass on several threads, in
//                      render passes suspended/resumed across command buffers (F2.5)
//   --debug-async-compute   add a synthetic compute pass on the second queue,
//                      synchronised with events (F2.6)
//   --pipeline-archive FILE  MTL4Archive to look pipelines up in (F3.4); by
//                      default shaders/phosphor-archive.metallib next to the
//                      shader library, if it exists
//   --no-pipeline-archive    ignore any archive: every pipeline is compiled
//   --harvest-pipelines FILE  record every pipeline descriptor the app creates
//                      (plus the whole variant table) into FILE (.mtl4-json)
//   --pipeline-sync    compile requested pipelines on the render thread
//                      (negative control for the async path, F3.1/F3.2)
//   --compile-qos Q    QoS of the compile threads: utility (default) or
//                      interactive (negative control, F3.1)
//   --pipeline-salt N  add a salt constant to every specialized variant so
//                      the OS shader cache cannot serve it (cold compiles)
//   --frame-trace FILE write per-frame times and bench-switch phases (CSV)
//   --debug-pipeline-fallback  the forward pass never requests its
//                      specialised variants and draws with the generic
//                      pipeline, the fallback a new variant uses while it
//                      compiles (F3.2/F3.3 check: same pixels as before F3)
//   --debug-flexible-pipelines  render pipelines stay on their Metal 4
//                      flexible fallback (never the final object): measures
//                      what the flexible path changes (F3.2)
//   --debug-compile-storm  F3.1 spike: after 60 measured frames request every
//                      forward variant at once (with --pipeline-salt: real
//                      compiles) while the frames keep being measured
//   --shader-dir DIR   Debug: watch DIR's .metal files and hot-reload the
//                      pipelines when they change (F3.6)
//   --debug-hot-reload FILE  Debug self-test: after a few frames swap in the
//                      pipelines of the probe library FILE (.metallib) through
//                      the hot-reload path, then check the drawable exactly
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
    u32         resizeEvery   = 0; // 0 = never
    bool        simulatePressure = false;
    u32         memoryStress  = 0; // cycles; 0 = off
    bool        transientTest = false;
    bool        debugGraphTransients = false; // F2.2 aliasing self-test on the GPU
    bool        injectInput   = false; // test: synthetic keys/mouse every frame
    std::string capturePath;
    std::string reportPath;
    std::string dumpGraphPath;
    bool        debugSplitEncoding = false;
    bool        debugAsyncCompute  = false;
    // F3: pipelines.
    std::string pipelineArchivePath;          // empty: default location
    bool        noPipelineArchive  = false;
    std::string harvestPipelinesPath;
    bool        pipelineSync       = false;
    bool        compileQosInteractive = false; // --compile-qos interactive
    u32         pipelineSalt       = 0;        // 0 = no salt
    std::string frameTracePath;
    bool        debugCompileStorm  = false;
    bool        debugPipelineFallback = false;
    bool        debugFlexiblePipelines = false;
    std::string shaderDir;                     // hot reload (Debug)
    std::string debugHotReloadPath;

    /// True when the app runs a fixed number of frames and then exits.
    [[nodiscard]] bool benchmark() const { return frames > 0; }
};

/// Parse argv into `out`.  Returns false and sets `error` on malformed input.
/// `benchCount` bounds the accepted --bench values.
bool parseLaunchOptions(int argc, const char* const* argv, int benchCount,
                        LaunchOptions& out, std::string& error);

} // namespace phosphor
