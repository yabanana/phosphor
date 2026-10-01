#pragma once

#include "core/memory/memory_budget.h"
#include "core/types.h"
#include "pipeline/pipeline_registry.h"

#include <array>
#include <vector>

namespace phosphor {

class FrameStats;

/// Renderer settings exposed in the debug UI.
struct RenderSettings {
    int   debugMode = 0;     // 0 = lit, 1 = normals, 2 = base color
    float exposure  = 1.0f;
    bool  vsync     = true;
    // F4.7 debug overlay: index of OverlayMode (none, overdraw, lights, tile
    // cost, timings); the legend fields are filled by the engine.
    int         overlay         = 0;
    const char* overlayQuantity = "";
    float       overlayMax      = 0.0f;
    bool        overlayLog      = false;
    const char* overlayNote     = "";
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

/// Pipeline cache state shown in the Pipelines panel (F3).
struct PipelinePanelInfo {
    const pipe::PipelineStats* stats = nullptr;
    const char* archive  = "";
    u32         workers  = 0;
    u32         entries  = 0;
    bool        fallback = false; // this frame drew with a fallback pipeline
};

/// Persistent GPU scene state shown in the GPU Scene panel (F5): plain data
/// filled by the engine; the per-frame GPU counters are the ones read back a
/// few frames late.
struct ScenePanelInfo {
    const char* mode = "off"; // --gpu-driven
    u32 instances = 0;
    u32 slots = 0;
    u32 buckets = 0;
    u32 materials = 0;
    u32 commands = 0;
    u32 visible = 0;
    u32 culledFrustum = 0;
    u32 culledDistance = 0;
    u32 culledSize = 0;
    u32 drawCommands = 0;
    u32 cpuCommands = 0;
    u32 deltaRecords = 0;
    u32 structureChanges = 0;
    u32 queueOverflow = 0;
    u64 uploadBytes = 0;
};

// ---------------------------------------------------------------------------
// UIPanels -- stateless helpers that draw the ImGui diagnostic windows.
// ---------------------------------------------------------------------------

class PassTimings;

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

    /// Pipeline cache: archive, compilations, fallbacks (F3).
    static void drawPipelinePanel(const PipelinePanelInfo& info);

    /// F5: persistent GPU scene (instances, buckets, culling counters, upload
    /// bytes, queue overflow).
    static void drawScenePanel(const ScenePanelInfo& info);

    /// F4.1: GPU time per timed unit of the render graph (average and maximum
    /// over the last 60 frames), estimated DRAM bytes, and the sum compared
    /// with the command buffer's GPU time.  `timings` null: timing is off.
    static void drawPassTimingsPanel(const PassTimings* timings, float commandBufferGpuMs, bool unfused);

    /// F4.7 "timings" overlay: a compact, undecorated corner overlay with the
    /// average GPU time of every timed unit and their sum.
    static void drawTimingsOverlay(const PassTimings* timings);
};

} // namespace phosphor
