#pragma once

#include "core/types.h"

#include <array>
#include <string_view>

namespace phosphor {

// ---------------------------------------------------------------------------
// GPU memory categories, tracked by the backend's single allocation point
// (GpuMemory) and budgeted per hardware tier (F1.4).
// ---------------------------------------------------------------------------

enum class MemoryCategory : u8 {
    Geometry,      // vertex / index / meshlet data
    Textures,      // material textures and texture tables
    Upload,        // CPU-written rings (per-frame data, staging)
    Transient,     // render-graph transient heap (F2.2)
    RenderTargets, // persistent attachments (depth, history buffers)
    Other,         // diagnostics, capture readback, ...
    COUNT
};

constexpr u32 MEMORY_CATEGORY_COUNT = static_cast<u32>(MemoryCategory::COUNT);

[[nodiscard]] const char* memoryCategoryName(MemoryCategory category);

// Hardware tiers of docs/ROADMAP.md ("Tier hardware").
enum class HardwareTier : u8 {
    T0Base,   // M3/M4/M5 base, A18 Pro, iPad M3+
    T1Pro,    // M3/M4/M5 Pro
    T2Max,    // M3/M4/M5 Max (and Ultra)
};

struct TierInfo {
    HardwareTier tier   = HardwareTier::T0Base;
    bool         neural = false; // T3: Apple10 (M5) Pro/Max/Ultra -- neural features, path tracing
};

/// Tier from the Metal device name ("Apple M5 Max") and GPU family.  Unknown
/// names fall back to T0: the floor must always work.
[[nodiscard]] TierInfo detectTier(std::string_view gpuName, bool apple10);
[[nodiscard]] const char* tierName(const TierInfo& tier);

// ---------------------------------------------------------------------------
// MemoryBudget -- per-category GPU memory limits.
//
// Limits are fractions of the device's recommended working set (queried at
// run time, never assumed), so the same split scales from a 16 GB T0 to a
// 128 GB M5 Max.  The fractions are project hypotheses (roadmap: budgets are
// recalibrated with measurements), identical for every tier for now; the
// tier only sets the default quality presets later (F28).
// ---------------------------------------------------------------------------

class MemoryBudget {
public:
    enum class Level : u8 { Ok, Warning, Over };

    /// Share of the working set the engine plans to use at most.
    static constexpr float ENGINE_SHARE = 0.75f;
    /// Fraction of a category's limit that triggers a warning.
    static constexpr float WARNING_THRESHOLD = 0.85f;

    MemoryBudget() = default;
    MemoryBudget(u64 workingSetBytes, TierInfo tier);

    [[nodiscard]] u64 limit(MemoryCategory category) const { return limits_[static_cast<u32>(category)]; }
    [[nodiscard]] u64 engineLimit() const { return engineLimit_; }
    [[nodiscard]] u64 workingSet() const { return workingSet_; }
    [[nodiscard]] TierInfo tier() const { return tier_; }

    [[nodiscard]] Level level(MemoryCategory category, u64 bytes) const;

    // Sizes of the backend's memory pools, derived from the budget (so a T0
    // gets smaller pools than an M5 Max).  Hypotheses like the shares.
    /// Frame upload ring: a quarter of the Upload budget, 16-128 MiB.
    [[nodiscard]] u64 frameUploadRingSize() const;
    /// Staging ring for loading-time copies: a quarter of Upload, 16-256 MiB.
    [[nodiscard]] u64 stagingRingSize() const;
    /// Size of a placement-heap page: engine budget / 512, rounded down to a
    /// power of two, 16-128 MiB.  Larger pages mean fewer heaps and residency
    /// entries; smaller ones waste less on small machines.
    [[nodiscard]] u64 heapPageSize() const;

    /// Share of the engine budget of each category (sums to 1).
    [[nodiscard]] static float share(MemoryCategory category);

private:
    std::array<u64, MEMORY_CATEGORY_COUNT> limits_{};
    u64      engineLimit_ = 0;
    u64      workingSet_  = 0;
    TierInfo tier_{};
};

} // namespace phosphor
