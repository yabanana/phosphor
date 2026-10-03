#pragma once

#include "core/types.h"

#include <string>
#include <vector>

namespace phosphor {

// ---------------------------------------------------------------------------
// FrameTrace -- per-frame timings and bench-switch phases (F3 "no hitch"
// measurement, before F4 brings GPU timestamps and Tracy).
//
// The engine records one FrameRecord per presented frame (benchmark mode) and
// one SwitchRecord per bench switch, with the switch split into phases so
// compilation on the render thread is separated from the synchronous
// loading work (waitIdle, uploads, GC -- by design until F22 streaming).
//
// Hitch analysis (analyzeHitches):
//   * the frame that performed the switch (FrameBenchSwitch) carries the
//     loading cost and is reported separately, not as a hitch;
//   * "steady" frames: not flagged and not within `window` frames after a
//     switch or a resize;
//   * threshold = max(p99Factor * steady CPU p99, steady CPU p99 + marginMs),
//     computed per bench (FrameRecord::bench) since benches differ ~20x in CPU
//     ms; a window frame is compared with the threshold of its own bench, and a
//     bench without steady frames falls back to the global threshold;
//   * a switch is a PIPELINE HITCH if the render thread compiled anything
//     for it (renderThreadCompileMs > 0) or any of the `window` frames after
//     it has CPU ms above the threshold.
// ---------------------------------------------------------------------------

enum FrameFlags : u32 {
    FrameBenchSwitch   = 1u << 0, // a bench switch ran before this frame
    FrameResize        = 1u << 1, // drawable size changed
    FrameGraphCompile  = 1u << 2, // render graph recompiled
    FramePipelineSwap  = 1u << 3, // the pipeline cache swapped objects at frame start
    FrameFallbackDraw  = 1u << 4, // some pass drew with a fallback pipeline
};

struct FrameRecord {
    u32   index  = 0;  // presented-frame index (0-based, benchmark mode)
    u32   bench  = 0;  // 0-based TestBenchType
    u32   flags  = 0;  // FrameFlags
    float frameMs = 0.0f;
    float cpuMs   = 0.0f;
    float waitMs  = 0.0f;
    float gpuMs   = 0.0f; // filled at the end of the run (commit feedback)
    float eventMs = 0.0f; // measured SDL/AppKit event pump, separate from renderer CPU work
};

struct SwitchRecord {
    u32   frame     = 0;  // index of the frame that performed the switch
    u32   fromBench = 0;
    u32   toBench   = 0;
    float totalMs            = 0.0f; // whole switchTestBench()
    float waitIdleMs         = 0.0f;
    float setupMs            = 0.0f; // teardown + scene/bench setup (CPU)
    float textureUploadMs    = 0.0f;
    float geometryUploadMs   = 0.0f;
    float gcMs               = 0.0f; // collectGarbage + trim + residency commit
    float pipelineRequestMs  = 0.0f; // render-thread time spent requesting pipelines
    float renderThreadCompileMs = 0.0f; // compilation the render thread waited for
    u32   pipelinesRequested = 0;
};

class FrameTrace {
public:
    /// Reserve so recording never allocates inside the measured frames.
    void reserve(u32 frames, u32 switches);
    void addFrame(const FrameRecord& frame);
    void addSwitch(const SwitchRecord& sw);
    /// Fill gpuMs of the recorded frames, in order (shorter input is fine).
    void setGpuTimes(const std::vector<float>& gpuMs);

    [[nodiscard]] const std::vector<FrameRecord>&  frames()   const { return frames_; }
    [[nodiscard]] const std::vector<SwitchRecord>& switches() const { return switches_; }

private:
    std::vector<FrameRecord>  frames_;
    std::vector<SwitchRecord> switches_;
};

struct HitchConfig {
    u32   window     = 10;
    float p99Factor  = 1.5f;
    float marginMs   = 0.5f;
};

struct HitchReport {
    u32   frames         = 0;
    u32   steadyFrames   = 0;
    float steadyCpuP99   = 0.0f;
    float thresholdMs    = 0.0f;
    u32   switches       = 0;
    u32   hitchSwitches  = 0;   // switches classified as pipeline hitches
    u32   framesOverThreshold = 0; // post-switch window frames above threshold
    float worstPostSwitchCpuMs = 0.0f;
    float worstPostSwitchRatio = 0.0f; // max over window frames of cpuMs / its bench threshold
    float renderThreadCompileMs = 0.0f; // sum over switches
    float worstEventMs = 0.0f;
    u32 platformStallFrames = 0; // post-switch event pump > 16.67 ms, reported separately
    // Switch phases: mean and max over switches (ms).
    float totalMean = 0.0f, totalMax = 0.0f;
    float waitIdleMean = 0.0f, waitIdleMax = 0.0f;
    float setupMean = 0.0f, setupMax = 0.0f;
    float uploadMean = 0.0f, uploadMax = 0.0f; // texture + geometry
    float gcMean = 0.0f, gcMax = 0.0f;
    float pipelineMean = 0.0f, pipelineMax = 0.0f; // request time
};

[[nodiscard]] HitchReport analyzeHitches(const FrameTrace& trace, const HitchConfig& config = {});

/// "SWITCH n=15 hitches 0 | load ms mean/max: total a/b, waitIdle ..., setup ...,
///  upload ..., gc ..., pipelines ... | render-thread compile 0.00 ms |
///  post-switch frames over X ms: 0 (worst Y ms, steady p99 Z ms, worst ratio R)"
[[nodiscard]] std::string formatHitchReport(const HitchReport& report);

/// CSV with a header; one row per frame, then one row per switch (column
/// `kind` = frame|switch).
[[nodiscard]] std::string traceToCsv(const FrameTrace& trace);

} // namespace phosphor
