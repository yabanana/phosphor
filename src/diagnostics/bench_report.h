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

struct PassReport {
    std::string              name;       // unit name ("A + B" for a fused group)
    std::string              queue;      // "graphics" / "async"
    bool                     fused = false;
    std::vector<std::string> passes;     // render graph passes covered
    std::vector<std::string> shaders;    // their profile shaders (PassBuilder::setProfileShaders)
    u64                      dramBytes = 0; // estimated DRAM bytes per frame
    u32                      frames    = 0; // measured frames with a valid time for this unit
    TimingSummary            gpuMs;
};

constexpr u32 BENCH_REPORT_SCHEMA_VERSION = 2;

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
