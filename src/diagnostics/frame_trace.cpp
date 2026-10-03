#include "diagnostics/frame_trace.h"

#include "diagnostics/bench_report.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace phosphor {

void FrameTrace::reserve(u32 frames, u32 switches) {
    frames_.reserve(frames);
    switches_.reserve(switches);
}

void FrameTrace::addFrame(const FrameRecord& frame) { frames_.push_back(frame); }

void FrameTrace::addSwitch(const SwitchRecord& sw) { switches_.push_back(sw); }

void FrameTrace::setGpuTimes(const std::vector<float>& gpuMs) {
    const size_t n = std::min(gpuMs.size(), frames_.size());
    for (size_t i = 0; i < n; ++i) frames_[i].gpuMs = gpuMs[i];
}

namespace {

// Flags that make a frame non-steady on their own.  FramePipelineSwap and
// FrameFallbackDraw are deliberately NOT here: a frame that swapped a
// pipeline object or drew with a fallback did ordinary CPU work, and
// excluding it would hide exactly the hitches we look for.
constexpr u32 kNonSteadyFlags = FrameBenchSwitch | FrameResize | FrameGraphCompile;

float engineCpu(const FrameRecord &f) {
    return std::max(0.0f, f.cpuMs - std::clamp(f.eventMs, 0.0f, f.cpuMs));
}

float thresholdFor(float p99, const HitchConfig& config) {
    return std::max(config.p99Factor * p99, p99 + config.marginMs);
}

struct Acc {
    double sum = 0.0;
    float  max = 0.0f;
    void add(float v) { sum += v; max = std::max(max, v); }
    [[nodiscard]] float mean(size_t n) const {
        return n ? static_cast<float>(sum / static_cast<double>(n)) : 0.0f;
    }
};

} // namespace

// Outcomes for degenerate inputs:
//   * no frames: frame statistics stay 0, switches are still summarised and a
//     switch is a hitch only through renderThreadCompileMs;
//   * a bench without steady frames is checked against the global threshold;
//   * no steady frames: steadyCpuP99 = threshold = 0 and the frame-based
//     check is skipped (nothing to compare against), so only
//     renderThreadCompileMs can flag a hitch;
//   * a switch whose frame is not in the trace has an empty window.
HitchReport analyzeHitches(const FrameTrace& trace, const HitchConfig& config) {
    HitchReport r;
    const auto& frames = trace.frames();
    const auto& switches = trace.switches();
    r.frames = static_cast<u32>(frames.size());
    r.switches = static_cast<u32>(switches.size());

    // Steady set: no non-steady flag, and not among the `window` frames that
    // follow a bench-switch or resize frame.
    std::vector<float> steady;
    std::map<u32, std::vector<float>> steadyByBench;
    steady.reserve(frames.size());
    u32 excludeLeft = 0;
    for (const FrameRecord& f : frames) {
        if (f.flags & (FrameBenchSwitch | FrameResize)) {
            excludeLeft = config.window;
            continue;
        }
        if (excludeLeft > 0) {
            --excludeLeft;
            continue;
        }
        if (f.flags & kNonSteadyFlags) continue;
        steady.push_back(engineCpu(f));
        steadyByBench[f.bench].push_back(engineCpu(f));
    }
    r.steadyFrames = static_cast<u32>(steady.size());
    const bool haveSteady = !steady.empty();
    if (haveSteady) {
        r.steadyCpuP99 = summarize(steady).p99;
        r.thresholdMs = thresholdFor(r.steadyCpuP99, config);
    }
    // Per-bench thresholds; a bench without steady frames uses the global one.
    std::map<u32, float> benchThreshold;
    for (const auto& [bench, values] : steadyByBench)
        benchThreshold[bench] = thresholdFor(summarize(values).p99, config);
    auto thresholdOf = [&](u32 bench) {
        const auto it = benchThreshold.find(bench);
        return it != benchThreshold.end() ? it->second : r.thresholdMs;
    };

    Acc total, waitIdle, setup, upload, gc, pipe;
    for (const SwitchRecord& sw : switches) {
        total.add(sw.totalMs);
        waitIdle.add(sw.waitIdleMs);
        setup.add(sw.setupMs);
        upload.add(sw.textureUploadMs + sw.geometryUploadMs);
        gc.add(sw.gcMs);
        pipe.add(sw.pipelineRequestMs);
        r.renderThreadCompileMs += sw.renderThreadCompileMs;

        bool hitch = sw.renderThreadCompileMs > 0.0f;

        if (haveSteady) {
            // Window: the `window` frames after the switch frame, ending early
            // at the next switch (its frame belongs to that switch).
            size_t pos = frames.size();
            for (size_t i = 0; i < frames.size(); ++i) {
                if (frames[i].index == sw.frame) { pos = i; break; }
            }
            u32 nextSwitch = 0;
            bool hasNext = false;
            for (const SwitchRecord& o : switches) {
                if (o.frame > sw.frame && (!hasNext || o.frame < nextSwitch)) {
                    nextSwitch = o.frame;
                    hasNext = true;
                }
            }
            for (size_t i = pos + 1, n = 0; i < frames.size() && n < config.window; ++i, ++n) {
                const FrameRecord& f = frames[i];
                if ((hasNext && f.index >= nextSwitch) || (f.flags & FrameBenchSwitch)) break;
                r.worstPostSwitchCpuMs = std::max(r.worstPostSwitchCpuMs, engineCpu(f));
                r.worstEventMs = std::max(r.worstEventMs, f.eventMs);
                if (f.eventMs > 16.67f)
                    ++r.platformStallFrames;
                const float limit = thresholdOf(f.bench);
                if (limit > 0.0f)
                    r.worstPostSwitchRatio = std::max(r.worstPostSwitchRatio, engineCpu(f) / limit);
                if (engineCpu(f) > limit) {
                    ++r.framesOverThreshold;
                    hitch = true;
                }
            }
        }
        if (hitch) ++r.hitchSwitches;
    }

    const size_t ns = switches.size();
    r.totalMean = total.mean(ns);       r.totalMax = total.max;
    r.waitIdleMean = waitIdle.mean(ns); r.waitIdleMax = waitIdle.max;
    r.setupMean = setup.mean(ns);       r.setupMax = setup.max;
    r.uploadMean = upload.mean(ns);     r.uploadMax = upload.max;
    r.gcMean = gc.mean(ns);             r.gcMax = gc.max;
    r.pipelineMean = pipe.mean(ns);     r.pipelineMax = pipe.max;
    return r;
}

std::string formatHitchReport(const HitchReport& r) {
    char buf[768];
    std::snprintf(buf, sizeof(buf),
                  "SWITCH n=%u hitches %u | load ms mean/max: total %.2f/%.2f, waitIdle %.2f/%.2f, "
                  "setup %.2f/%.2f, upload %.2f/%.2f, gc %.2f/%.2f, pipelines %.2f/%.2f | "
                  "render-thread compile %.2f ms | post-switch frames over %.2f ms: %u "
                  "(worst %.2f ms, steady p99 %.2f ms, worst ratio %.2f)",
                  r.switches, r.hitchSwitches, r.totalMean, r.totalMax, r.waitIdleMean, r.waitIdleMax,
                  r.setupMean, r.setupMax, r.uploadMean, r.uploadMax, r.gcMean, r.gcMax,
                  r.pipelineMean, r.pipelineMax, r.renderThreadCompileMs, r.thresholdMs,
                  r.framesOverThreshold, r.worstPostSwitchCpuMs, r.steadyCpuP99, r.worstPostSwitchRatio);
    std::string line = buf;
    std::snprintf(buf, sizeof(buf), " | event pump worst %.2f ms, platform stalls %u", r.worstEventMs,
                  r.platformStallFrames);
    return line + buf;
}

// Columns (every row has the same column count; unused ones are empty):
//   kind,index,bench,flags,frameMs,cpuMs,waitMs,gpuMs,
//   toBench,totalMs,waitIdleMs,setupMs,textureUploadMs,geometryUploadMs,gcMs,
//   pipelineRequestMs,renderThreadCompileMs,pipelinesRequested
// Frame rows fill index..gpuMs; switch rows use index = switch frame,
// bench = fromBench, then toBench and the phase columns.
std::string traceToCsv(const FrameTrace& trace) {
    std::string out = "kind,index,bench,flags,frameMs,cpuMs,waitMs,gpuMs,toBench,totalMs,waitIdleMs,setupMs,"
                      "textureUploadMs,geometryUploadMs,gcMs,pipelineRequestMs,renderThreadCompileMs,"
                      "pipelinesRequested,eventMs\n";
    char buf[384];
    for (const FrameRecord& f : trace.frames()) {
        std::snprintf(buf, sizeof(buf), "frame,%u,%u,%u,%.4f,%.4f,%.4f,%.4f,,,,,,,,,,,%.4f\n", f.index, f.bench,
                      f.flags, f.frameMs, f.cpuMs, f.waitMs, f.gpuMs, f.eventMs);
        out += buf;
    }
    for (const SwitchRecord& s : trace.switches()) {
        std::snprintf(buf, sizeof(buf), "switch,%u,%u,,,,,,%u,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%u,\n", s.frame,
                      s.fromBench, s.toBench, s.totalMs, s.waitIdleMs, s.setupMs, s.textureUploadMs, s.geometryUploadMs,
                      s.gcMs, s.pipelineRequestMs, s.renderThreadCompileMs, s.pipelinesRequested);
        out += buf;
    }
    return out;
}

} // namespace phosphor
