#pragma once

#include "rhi/vk_common.h"
#include "core/types.h"
#include <string>
#include <unordered_map>
#include <vector>
#include <array>

namespace phosphor {

class VulkanDevice;

struct PassTiming {
    float gpuMs = 0.0f;
    float avgMs = 0.0f;
    float maxMs = 0.0f;
};

class GpuProfiler {
public:
    static constexpr u32 MAX_QUERIES_PER_FRAME = 256;

    GpuProfiler(VulkanDevice& device);
    ~GpuProfiler();
    GpuProfiler(const GpuProfiler&) = delete;

    void beginFrame(u32 frameIndex);
    void beginPass(VkCommandBuffer cmd, const std::string& passName);
    void endPass(VkCommandBuffer cmd, const std::string& passName);
    void resolveQueries(u32 frameIndex);

    const std::unordered_map<std::string, PassTiming>& getPassTimings() const { return timings_; }
    float getTotalGpuMs() const;

private:
    VulkanDevice& device_;
    std::array<VkQueryPool, FRAMES_IN_FLIGHT> queryPools_;
    std::array<u32, FRAMES_IN_FLIGHT> queryCount_;
    float timestampPeriod_ = 1.0f;

    // Map pass name -> query index pair (begin, end) for current frame
    std::unordered_map<std::string, std::pair<u32, u32>> queryIndices_;
    std::unordered_map<std::string, PassTiming> timings_;

    static constexpr u32 SMOOTHING_FRAMES = 60;
    std::unordered_map<std::string, std::array<float, SMOOTHING_FRAMES>> timingHistory_;
    std::unordered_map<std::string, u32> historyIndex_;
};

} // namespace phosphor
