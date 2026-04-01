#pragma once

#include "core/types.h"
#include "core/timer.h"
#include <array>
#include <vector>

namespace phosphor {

class GpuProfiler;

// ---------------------------------------------------------------------------
// FrameStats -- lightweight CPU/GPU frame timing tracker.
// Accumulates per-frame data and exposes rolling averages, percentiles,
// stutter detection, and histogram data for the diagnostic UI.
// ---------------------------------------------------------------------------

class FrameStats {
public:
    static constexpr u32 HISTORY_SIZE = 300; // ~5 seconds at 60 fps

    /// Primary update path: feed CPU time from Timer and GPU time from profiler.
    void update(const Timer& timer, float gpuMs) {
        cpuMs_ = timer.getDeltaTime() * 1000.0f;
        gpuMs_ = gpuMs;
        fps_   = timer.getFPS();

        cpuHistory_[writeIndex_] = cpuMs_;
        gpuHistory_[writeIndex_] = gpuMs_;
        fpsHistory_[writeIndex_] = fps_;
        writeIndex_ = (writeIndex_ + 1) % HISTORY_SIZE;
        if (sampleCount_ < HISTORY_SIZE) ++sampleCount_;
    }

    /// Lightweight path: record a single frame time in milliseconds.
    void recordFrameTime(float ms);

    /// Return the p-th percentile of recorded CPU frame times.
    /// @param p  Percentile in [0, 1] -- e.g. 0.99 for P99.
    float getPercentile(float p) const;

    /// Average frames per second over the recorded window.
    float getAverageFPS() const;

    /// Average CPU frame time in milliseconds over the recorded window.
    float getAverageFrameTime() const;

    /// Returns true if the most recent frame time exceeds the running average
    /// by more than @p thresholdMultiplier times (default 2x).
    bool detectStutter(float thresholdMultiplier = 2.0f) const;

    /// Build a histogram of CPU frame times for overlay rendering.
    /// @param buckets  Number of histogram bins.
    /// @return Vector of @p buckets normalized counts (0-1 range).
    std::vector<float> getHistogram(u32 buckets = 30) const;

    [[nodiscard]] float getCpuMs()  const { return cpuMs_; }
    [[nodiscard]] float getGpuMs()  const { return gpuMs_; }
    [[nodiscard]] float getFPS()    const { return fps_; }

    [[nodiscard]] const std::array<float, HISTORY_SIZE>& getCpuHistory() const { return cpuHistory_; }
    [[nodiscard]] const std::array<float, HISTORY_SIZE>& getGpuHistory() const { return gpuHistory_; }
    [[nodiscard]] const std::array<float, HISTORY_SIZE>& getFpsHistory() const { return fpsHistory_; }
    [[nodiscard]] const std::array<float, HISTORY_SIZE>& getRawTimes()   const { return cpuHistory_; }
    [[nodiscard]] u32 getSampleCount() const { return sampleCount_; }

    [[nodiscard]] u64 getFrameCount() const { return frameCount_; }
    void incrementFrame() { ++frameCount_; }

private:
    float cpuMs_ = 0.0f;
    float gpuMs_ = 0.0f;
    float fps_   = 0.0f;
    u64   frameCount_ = 0;

    std::array<float, HISTORY_SIZE> cpuHistory_{};
    std::array<float, HISTORY_SIZE> gpuHistory_{};
    std::array<float, HISTORY_SIZE> fpsHistory_{};
    u32 writeIndex_  = 0;
    u32 sampleCount_ = 0;
};

} // namespace phosphor
