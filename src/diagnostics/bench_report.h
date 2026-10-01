#pragma once

#include "core/types.h"

#include <string>
#include <vector>

namespace phosphor {

// ---------------------------------------------------------------------------
// Benchmark summary for `--frames N` runs (baseline numbers in perf-log.md).
//
// Per frame four times are recorded:
//   frameMs  wall time between consecutive frames (what FPS is derived from)
//   cpuMs    CPU time spent producing the frame, excluding the waits for a
//            free frame slot and for a drawable
//   gpuMs    GPU start-to-end span of the frame's command buffer; when the
//            GPU overlaps consecutive frames this includes the overlap, so it
//            is an upper bound of the frame's GPU cost (the per-pass times
//            below are the measured one)
//   waitMs   CPU time blocked in beginFrame() (free frame slot + drawable)
//
// F4.1/F4.6: with GPU timing on, `passes` holds one entry per timed unit of
// the render graph (rg::TimingPlan: a fused render group is one unit) with
// the distribution of its GPU time over the measured frames, and
// gpuPassSumMs the per-frame sum of the units.  JSON schema version 2 adds
// "schema_version", "gpu_timing", "gpu_timing_unfused", "passes" (array of
// {"name", "queue", "fused", "passes": [..], "shaders": [..],
// "dram_bytes", "frames", "gpu_ms": {summary}}), "gpu_pass_sum_ms" and
// "gpu_frame_span_ms" (summaries); version 1 fields are unchanged.
// OPT-0.4, version 3: each pass entry may carry "work": [{"pass", "draws",
// "instances", "indices", "vertices", "pixels", "threads", "lights"}]
// (PassWork); nothing else changes.
// ---------------------------------------------------------------------------

struct FrameSample {
    float frameMs = 0.0f;
    float cpuMs   = 0.0f;
    float gpuMs   = 0.0f;
    float waitMs  = 0.0f;
};

struct TimingSummary {
    float mean = 0.0f;
    float min  = 0.0f;
    float p50  = 0.0f;
    float p99  = 0.0f;
    float max  = 0.0f;
};

// OPT-0.4: work of one render-graph pass over one frame, from data the CPU
// already has (no GPU counters): what the SoC cost model multiplies the
// shaders' op counts by (tools/soc_model).  `vertices` = unique vertices of
// the drawn meshes x instances (a lower bound of vertex shader invocations;
// `indices` is the upper bound without post-transform reuse).
struct PassWork {
    std::string pass;          // render graph pass name
    u64         draws     = 0;
    u64         instances = 0;
    u64         indices   = 0; // index count x instances
    u64         vertices  = 0; // unique vertices x instances
    u64         pixels    = 0; // render area of the pass
    u64         threads   = 0; // compute threads dispatched
    u32         lights    = 0; // lights the fragment shader loops over
};

struct PassReport {
    std::string              name;       // unit name ("A + B" for a fused group)
    std::string              queue;      // "graphics" / "async"
    bool                     fused = false;
    std::vector<std::string> passes;     // render graph passes covered
    std::vector<std::string> shaders;    // their profile shaders (PassBuilder::setProfileShaders)
    u64                      dramBytes = 0; // estimated DRAM bytes per frame
    u32                      frames    = 0; // measured frames with a valid time for this unit
    TimingSummary            gpuMs;
    std::vector<PassWork>    work;       // OPT-0.4 (schema 3): passes with known work
};

// OPT-1 (schema 4): how the frame graph was compiled and what it costs.
struct GraphReport {
    bool        present = false;
    std::string mode;          // --graph-opt: off | greedy | plan
    std::string family;        // plan family of the graph (scenarios), or empty
    std::string plan;          // "applied", "none", or "rejected: <reason>"
    std::string alias;         // alias policy used
    std::string barriers;      // barrier policy used
    u32         passes       = 0; // live passes
    u32         renderPasses = 0;
    u32         memoryless   = 0;
    u32         barrierCount = 0;
    u64         dramBytes    = 0; // estimated DRAM bytes per frame (O1)
    u64         heapBytes    = 0; // transient heap (memory peak of the transients)
    u64         heapUnaliasedBytes = 0;
    u64         maxLiveBytes = 0; // lower bound of the heap for this order
};

// F5 (schema 5): the persistent GPU scene and its submission over the
// measured frames.  Per-frame quantities are summaries over those frames;
// GPU counters (visible, culled, draw commands) come from the frame's
// GPUSceneCounters read back METAL_FRAMES_IN_FLIGHT frames later.
struct SceneReport {
    bool          present = false;
    std::string   mode;            // gpu-driven "off" / "on"
    u32           instances = 0;   // live instances (last frame)
    u32           slots     = 0;
    u32           buckets   = 0;
    u32           materials = 0;
    u32           commands  = 0;   // ICB commands (buckets + sentinels)
    u32           structureChanges = 0; // structure events during the measured frames
    u32           queueOverflow    = 0; // GPU queue entries dropped (must be 0)
    TimingSummary uploadBytes;     // bytes written by the CPU for the scene per frame
    TimingSummary deltaRecords;    // delta records per frame (all buffers)
    TimingSummary visible;
    TimingSummary culledFrustum;
    TimingSummary culledDistance;
    TimingSummary culledSize;
    TimingSummary drawCommands;    // non-empty draws (on: ICB commands written; off: CPU draws)
    TimingSummary cpuCommands;     // commands the CPU encoded for the scene per frame (O8)
};

// F5 (schema 5): CPU ms per frame phase (replaces the F5 spike S1 line).
struct CpuPhasesReport {
    bool          present = false;
    TimingSummary sim;       // bench update
    TimingSummary sceneSync; // SceneStore::sync + lights
    TimingSummary prepare;   // renderer prepareFrame (delta upload)
    TimingSummary ui;
    TimingSummary graph;     // graph execution (encoding)
    TimingSummary submit;
};

constexpr u32 BENCH_REPORT_SCHEMA_VERSION = 5;

struct BenchReport {
    std::string   bench;
    std::string   device;
    u32           width  = 0;
    u32           height = 0;
    bool          vsync  = true;
    bool          ui     = true;
    u32           frames = 0;
    float         fps    = 0.0f; // 1000 / mean frame time
    u64           gpuAllocations = 0; // GPU buffers/textures created while measuring (O7: must be 0)
    // CPU heap growth over the measured frames (all malloc zones): live
    // blocks and bytes at the end minus at the start.  ~0 = flat (O7).
    i64           cpuHeapBlocksDelta = 0;
    i64           cpuHeapBytesDelta  = 0;
    TimingSummary frameMs;
    TimingSummary cpuMs;
    TimingSummary gpuMs;
    TimingSummary waitMs;
    std::string   pipelinesJson; // F3.5: PipelineStats as a JSON object (empty: omitted)
    // F4.1/F4.6: per-pass GPU timing (empty `passes` when timing is off).
    bool          gpuTiming        = false;
    bool          gpuTimingUnfused = false; // graph compiled without raster fusion
    std::vector<PassReport> passes;
    TimingSummary gpuPassSumMs;
    TimingSummary gpuFrameSpanMs;
    GraphReport   graph;         // OPT-1 (schema 4)
    SceneReport   scene;         // F5 (schema 5)
    CpuPhasesReport cpuPhases;   // F5 (schema 5)
};

/// Nearest-rank statistics of `values` (empty input gives all zeros).
TimingSummary summarize(std::vector<float> values);

/// Fill the timing fields of `report` from the recorded samples.
void summarizeSamples(const std::vector<FrameSample>& samples, BenchReport& report);

/// One-line human-readable summary for the log.
std::string formatReportLine(const BenchReport& report);

/// JSON object with every field of the report.
std::string reportToJson(const BenchReport& report);

} // namespace phosphor
