#include "diagnostics/gpu_profiler.h"
#include "rhi/vk_device.h"
#include <algorithm>
#include <numeric>

namespace phosphor {

GpuProfiler::GpuProfiler(VulkanDevice& device) : device_(device) {
    timestampPeriod_ = device_.getProperties().limits.timestampPeriod;

    VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    info.queryCount = MAX_QUERIES_PER_FRAME * 2; // begin + end per query

    for (u32 i = 0; i < FRAMES_IN_FLIGHT; i++) {
        VK_CHECK(vkCreateQueryPool(device_.getDevice(), &info, nullptr, &queryPools_[i]));
        queryCount_[i] = 0;
    }

    LOG_INFO("GPU profiler initialized (timestamp period: %.2f ns)", timestampPeriod_);
}

GpuProfiler::~GpuProfiler() {
    for (auto& pool : queryPools_) {
        if (pool) vkDestroyQueryPool(device_.getDevice(), pool, nullptr);
    }
}

void GpuProfiler::beginFrame(u32 frameIndex) {
    vkResetQueryPool(device_.getDevice(), queryPools_[frameIndex], 0, MAX_QUERIES_PER_FRAME * 2);
    queryCount_[frameIndex] = 0;
    queryIndices_.clear();
}

void GpuProfiler::beginPass(VkCommandBuffer cmd, const std::string& passName) {
    u32 frameIndex = 0; // Caller should track, but we use the current query pool
    // Find which frame's pool to use based on which pool has been reset
    for (u32 i = 0; i < FRAMES_IN_FLIGHT; i++) {
        if (queryCount_[i] < MAX_QUERIES_PER_FRAME) {
            frameIndex = i;
            break;
        }
    }

    u32 queryIdx = queryCount_[frameIndex]++;
    queryIndices_[passName].first = queryIdx * 2;

    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                         queryPools_[frameIndex], queryIdx * 2);
}

void GpuProfiler::endPass(VkCommandBuffer cmd, const std::string& passName) {
    auto it = queryIndices_.find(passName);
    if (it == queryIndices_.end()) return;

    u32 frameIndex = 0;
    for (u32 i = 0; i < FRAMES_IN_FLIGHT; i++) {
        if (queryCount_[i] > 0) { frameIndex = i; break; }
    }

    it->second.second = it->second.first + 1;
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                         queryPools_[frameIndex], it->second.second);
}

void GpuProfiler::resolveQueries(u32 frameIndex) {
    if (queryCount_[frameIndex] == 0) return;

    u32 totalQueries = queryCount_[frameIndex] * 2;
    std::vector<u64> timestamps(totalQueries);

    VkResult result = vkGetQueryPoolResults(
        device_.getDevice(), queryPools_[frameIndex],
        0, totalQueries,
        timestamps.size() * sizeof(u64), timestamps.data(),
        sizeof(u64), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

    if (result != VK_SUCCESS) return;

    for (auto& [name, indices] : queryIndices_) {
        if (indices.first >= totalQueries || indices.second >= totalQueries) continue;

        u64 begin = timestamps[indices.first];
        u64 end = timestamps[indices.second];
        float ms = static_cast<float>(end - begin) * timestampPeriod_ / 1e6f;

        // Update smoothed timing
        auto& hist = timingHistory_[name];
        auto& idx = historyIndex_[name];
        hist[idx % SMOOTHING_FRAMES] = ms;
        idx++;

        u32 count = std::min(idx, SMOOTHING_FRAMES);
        float sum = 0;
        float maxVal = 0;
        for (u32 i = 0; i < count; i++) {
            sum += hist[i];
            maxVal = std::max(maxVal, hist[i]);
        }

        timings_[name] = {ms, sum / static_cast<float>(count), maxVal};
    }
}

float GpuProfiler::getTotalGpuMs() const {
    float total = 0;
    for (auto& [name, timing] : timings_) {
        total += timing.gpuMs;
    }
    return total;
}

} // namespace phosphor
