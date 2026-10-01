#include "core/memory/memory_budget.h"

#include <algorithm>

namespace phosphor {

const char* memoryCategoryName(MemoryCategory category) {
    switch (category) {
    case MemoryCategory::Geometry:      return "Geometry";
    case MemoryCategory::Textures:      return "Textures";
    case MemoryCategory::Upload:        return "Upload";
    case MemoryCategory::Transient:     return "Transient";
    case MemoryCategory::RenderTargets: return "Render targets";
    case MemoryCategory::Scene:         return "Scene";
    case MemoryCategory::Other:         return "Other";
    case MemoryCategory::COUNT:         break;
    }
    return "?";
}

TierInfo detectTier(std::string_view gpuName, bool apple10) {
    TierInfo info;
    // Match whole words so "Pro" does not match inside other names.
    auto hasWord = [&](std::string_view word) {
        for (size_t pos = gpuName.find(word); pos != std::string_view::npos; pos = gpuName.find(word, pos + 1)) {
            const bool startOk = pos == 0 || gpuName[pos - 1] == ' ';
            const size_t end = pos + word.size();
            const bool endOk = end == gpuName.size() || gpuName[end] == ' ';
            if (startOk && endOk) return true;
        }
        return false;
    };
    // A-series (iPhone) chips are T0 whatever their suffix: "A18 Pro" is the floor.
    const size_t a = gpuName.find("Apple A");
    const bool aSeries = a != std::string_view::npos && a + 7 < gpuName.size() && gpuName[a + 7] >= '0' &&
                         gpuName[a + 7] <= '9';
    if (aSeries) {
        info.tier = HardwareTier::T0Base;
    } else if (hasWord("Max") || hasWord("Ultra")) {
        info.tier = HardwareTier::T2Max;
    } else if (hasWord("Pro")) {
        info.tier = HardwareTier::T1Pro;
    }
    info.neural = apple10 && info.tier != HardwareTier::T0Base;
    return info;
}

const char* tierName(const TierInfo& tier) {
    if (tier.neural) return tier.tier == HardwareTier::T2Max ? "T2 Max + T3 Neural" : "T1 Pro + T3 Neural";
    switch (tier.tier) {
    case HardwareTier::T0Base: return "T0 Base";
    case HardwareTier::T1Pro:  return "T1 Pro";
    case HardwareTier::T2Max:  return "T2 Max";
    }
    return "?";
}

float MemoryBudget::share(MemoryCategory category) {
    // Hypotheses (F1.4), to recalibrate with measurements: textures dominate
    // a game's footprint, geometry follows; rings and diagnostics are small.
    switch (category) {
    case MemoryCategory::Geometry:      return 0.25f;
    case MemoryCategory::Textures:      return 0.40f;
    case MemoryCategory::Upload:        return 0.03f;
    case MemoryCategory::Transient:     return 0.12f;
    case MemoryCategory::RenderTargets: return 0.12f;
    case MemoryCategory::Scene:         return 0.05f; // F5: ~0.2 GiB per million instances
    case MemoryCategory::Other:         return 0.03f;
    case MemoryCategory::COUNT:         break;
    }
    return 0.0f;
}

MemoryBudget::MemoryBudget(u64 workingSetBytes, TierInfo tier)
    : engineLimit_(static_cast<u64>(static_cast<double>(workingSetBytes) * ENGINE_SHARE)),
      workingSet_(workingSetBytes), tier_(tier) {
    for (u32 c = 0; c < MEMORY_CATEGORY_COUNT; ++c) {
        limits_[c] = static_cast<u64>(static_cast<double>(engineLimit_) * share(static_cast<MemoryCategory>(c)));
    }
}

namespace {

constexpr u64 MiB = 1ull << 20;

u64 clampSize(u64 v, u64 lo, u64 hi) { return v < lo ? lo : (v > hi ? hi : v); }

u64 roundDownMiB(u64 v) { return v / MiB * MiB; }

u64 floorPowerOfTwo(u64 v) {
    u64 p = 1;
    while (p <= v / 2) p <<= 1;
    return p;
}

} // namespace

u64 MemoryBudget::frameUploadRingSize() const {
    return roundDownMiB(clampSize(limit(MemoryCategory::Upload) / 4, 16 * MiB, 128 * MiB));
}

u64 MemoryBudget::stagingRingSize() const {
    return roundDownMiB(clampSize(limit(MemoryCategory::Upload) / 4, 16 * MiB, 256 * MiB));
}

u64 MemoryBudget::heapPageSize() const {
    return clampSize(floorPowerOfTwo(std::max<u64>(engineLimit_ / 512, 1)), 16 * MiB, 128 * MiB);
}

MemoryBudget::Level MemoryBudget::level(MemoryCategory category, u64 bytes) const {
    const u64 lim = limit(category);
    if (bytes > lim) return Level::Over;
    if (static_cast<double>(bytes) > static_cast<double>(lim) * WARNING_THRESHOLD) return Level::Warning;
    return Level::Ok;
}

} // namespace phosphor
