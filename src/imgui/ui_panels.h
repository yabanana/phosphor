#pragma once

#include "core/memory/memory_budget.h"
#include "core/types.h"

#include <array>
#include <vector>

namespace phosphor {

class FrameStats;

/// Renderer settings exposed in the debug UI.
struct RenderSettings {
    int   debugMode = 0;     // 0 = lit, 1 = normals, 2 = base color
    float exposure  = 1.0f;
    bool  vsync     = true;
};

/// Read-only information shown in the stats panel.
struct RendererInfo {
    const char* gpuName     = "";
    bool        apple9      = false;
    u32         width       = 0;
    u32         height      = 0;
    u32         instances   = 0;
    u32         drawBatches = 0;
    u32         triangles   = 0;
    u32         meshlets    = 0;
    u32         textures    = 0;
    float       gpuMs       = 0.0f;
};

/// GPU memory state shown in the Memory panel (plain data, filled by the
/// engine from the backend; reuse one instance so vectors keep capacity).
struct MemoryPanelInfo {
    struct Category {
        u64                 bytes = 0;
        u32                 count = 0;
        u64                 limit = 0;
        MemoryBudget::Level level = MemoryBudget::Level::Ok;
    };
    struct Heap {
        u64   size        = 0;
        u64   used        = 0;
        u32   allocations = 0;
        float fragmentation = 0.0f;
        bool  streaming   = false;
    };
    struct Ring {
        const char* name       = "";
        u64         capacity   = 0;
        u64         inFlight   = 0;
        u64         peakFrame  = 0;
        u32         overflows  = 0;
    };
    struct ResidencySet {
        const char* name        = "";
        u32         allocations = 0;
        u64         bytes       = 0;
        u32         commits     = 0;
    };

    const char* tier            = "";
    u64         workingSet      = 0;
    u64         engineLimit     = 0;
    u64         deviceAllocated = 0; // MTLDevice currentAllocatedSize
    u64         gpuAllocations  = 0; // monotonic GpuMemory counter
    std::array<Category, MEMORY_CATEGORY_COUNT> categories{};
    std::vector<Heap> heaps;
    std::array<Ring, 2> rings{};
    std::array<ResidencySet, 2> residency{};
    const char* pressure        = "normal"; // last memory-pressure level (F1.5)
    u32         pressureEvents  = 0;
};

// ---------------------------------------------------------------------------
// UIPanels -- stateless helpers that draw the ImGui diagnostic windows.
// ---------------------------------------------------------------------------

class UIPanels {
public:
    /// Combo box listing the test benches. Sets `changed` if the user picked
    /// a different bench.
    static void drawTestBenchSelector(int& currentBench, bool& changed);

    /// FPS / CPU / GPU timings and frame-time graphs.
    static void drawPerformancePanel(const FrameStats& stats, const RendererInfo& info);

    /// Debug view selection and renderer tweaks.
    static void drawRenderPanel(RenderSettings& settings);

    /// GPU memory: budgets per category, heaps, upload rings, residency.
    static void drawMemoryPanel(const MemoryPanelInfo& info);
};

} // namespace phosphor
