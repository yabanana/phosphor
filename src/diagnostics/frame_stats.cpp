#include "diagnostics/frame_stats.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace phosphor {

// ---------------------------------------------------------------------------
// recordFrameTime -- lightweight path that only updates cpuHistory_
// ---------------------------------------------------------------------------

void FrameStats::recordFrameTime(float ms) {
    cpuMs_ = ms;
    cpuHistory_[writeIndex_] = ms;
    writeIndex_ = (writeIndex_ + 1) % HISTORY_SIZE;
    if (sampleCount_ < HISTORY_SIZE) {
        ++sampleCount_;
    }
}

// ---------------------------------------------------------------------------
// getPercentile
// ---------------------------------------------------------------------------

float FrameStats::getPercentile(float p) const {
    if (sampleCount_ == 0) {
        return 0.0f;
    }

    u32 count = sampleCount_ < HISTORY_SIZE ? sampleCount_ : HISTORY_SIZE;

    // Copy the valid portion of the ring buffer and sort it
    std::vector<float> sorted(count);
    if (sampleCount_ >= HISTORY_SIZE) {
        // Buffer is full -- entire array is valid
        std::copy(cpuHistory_.begin(), cpuHistory_.end(), sorted.begin());
    } else {
        // Buffer is partially filled -- only first sampleCount_ entries
        std::copy(cpuHistory_.begin(), cpuHistory_.begin() + count, sorted.begin());
    }
    std::sort(sorted.begin(), sorted.end());

    // Clamp p to [0, 1]
    float clamped = std::clamp(p, 0.0f, 1.0f);
    u32 index = static_cast<u32>(clamped * static_cast<float>(count - 1));
    return sorted[index];
}

// ---------------------------------------------------------------------------
// getAverageFPS
// ---------------------------------------------------------------------------

float FrameStats::getAverageFPS() const {
    float avg = getAverageFrameTime();
    if (avg <= 0.0f) {
        return 0.0f;
    }
    return 1000.0f / avg;
}

// ---------------------------------------------------------------------------
// getAverageFrameTime
// ---------------------------------------------------------------------------

float FrameStats::getAverageFrameTime() const {
    if (sampleCount_ == 0) {
        return 0.0f;
    }

    u32 count = sampleCount_ < HISTORY_SIZE ? sampleCount_ : HISTORY_SIZE;
    float sum = 0.0f;
    for (u32 i = 0; i < count; ++i) {
        sum += cpuHistory_[i];
    }
    return sum / static_cast<float>(count);
}

// ---------------------------------------------------------------------------
// detectStutter
// ---------------------------------------------------------------------------

bool FrameStats::detectStutter(float thresholdMultiplier) const {
    if (sampleCount_ < 2) {
        return false;
    }

    float avg = getAverageFrameTime();
    // The most recent sample is at (writeIndex_ - 1) mod HISTORY_SIZE
    u32 lastIdx = (writeIndex_ == 0) ? (HISTORY_SIZE - 1) : (writeIndex_ - 1);
    float lastFrame = cpuHistory_[lastIdx];

    return lastFrame > avg * thresholdMultiplier;
}

// ---------------------------------------------------------------------------
// getHistogram
// ---------------------------------------------------------------------------

std::vector<float> FrameStats::getHistogram(u32 buckets) const {
    std::vector<float> histogram(buckets, 0.0f);

    if (sampleCount_ == 0 || buckets == 0) {
        return histogram;
    }

    u32 count = sampleCount_ < HISTORY_SIZE ? sampleCount_ : HISTORY_SIZE;

    // Find min/max for bin edges
    float minTime = cpuHistory_[0];
    float maxTime = cpuHistory_[0];
    for (u32 i = 1; i < count; ++i) {
        minTime = std::min(minTime, cpuHistory_[i]);
        maxTime = std::max(maxTime, cpuHistory_[i]);
    }

    float range = maxTime - minTime;
    if (range <= 0.0f) {
        // All samples identical: everything in bucket 0
        histogram[0] = 1.0f;
        return histogram;
    }

    float bucketWidth = range / static_cast<float>(buckets);
    std::vector<u32> counts(buckets, 0);

    for (u32 i = 0; i < count; ++i) {
        u32 bucket = static_cast<u32>((cpuHistory_[i] - minTime) / bucketWidth);
        if (bucket >= buckets) {
            bucket = buckets - 1; // clamp the max value into the last bucket
        }
        counts[bucket]++;
    }

    // Normalize to [0, 1] by dividing by the maximum bin count
    u32 maxCount = *std::max_element(counts.begin(), counts.end());
    if (maxCount > 0) {
        for (u32 i = 0; i < buckets; ++i) {
            histogram[i] = static_cast<float>(counts[i]) / static_cast<float>(maxCount);
        }
    }

    return histogram;
}

} // namespace phosphor
