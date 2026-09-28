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
//            is an upper bound of the frame's GPU cost (per-pass timestamps
//            arrive with F4.1)
//   waitMs   CPU time blocked in beginFrame() (free frame slot + drawable)
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
