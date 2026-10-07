#pragma once

#include "core/types.h"

#include <string>
#include <vector>

namespace phosphor {

// ---------------------------------------------------------------------------
// Benchmark summary for `--frames N` runs (baseline numbers in perf-log.md).
//
// Per frame four times are recorded:
//   frameMs  wall time between consecutive frames (what FPS is derived from)
//   cpuMs    CPU time spent producing the frame, excluding the waits for a
//            free frame slot and for a drawable
//   gpuMs    GPU start-to-end span of the frame's command buffer; when the
//            GPU overlaps consecutive frames this includes the overlap, so it
//            is an upper bound of the frame's GPU cost (the per-pass times
//            below are the measured one)
//   waitMs   CPU time blocked in beginFrame() (free frame slot + drawable)
//
// F4.1/F4.6: with GPU timing on, `passes` holds one entry per timed unit of
// the render graph (rg::TimingPlan: a fused render group is one unit) with
// the distribution of its GPU time over the measured frames, and
// gpuPassSumMs the per-frame sum of the units.  JSON schema version 2 adds
// "schema_version", "gpu_timing", "gpu_timing_unfused", "passes" (array of
// {"name", "queue", "fused", "passes": [..], "shaders": [..],
// "dram_bytes", "frames", "gpu_ms": {summary}}), "gpu_pass_sum_ms" and
// "gpu_frame_span_ms" (summaries); version 1 fields are unchanged.
// OPT-0.4, version 3: each pass entry may carry "work": [{"pass", "draws",
// "instances", "indices", "vertices", "pixels", "threads", "lights"}]
// (PassWork); nothing else changes.
// ---------------------------------------------------------------------------

struct FrameSample {
    float frameMs = 0.0f;
    float cpuMs   = 0.0f;
    float gpuMs   = 0.0f;
    float waitMs  = 0.0f;
};

struct TimingSummary {
    float mean = 0.0f;
    float min  = 0.0f;
    float p50  = 0.0f;
    float p99  = 0.0f;
    float max  = 0.0f;
    float p95  = 0.0f; // F6 (schema 6): the 60 fps gate is on the p95 frame time (last: positional inits unchanged)
};

// OPT-0.4: work of one render-graph pass over one frame, from data the CPU
// already has (no GPU counters): what the SoC cost model multiplies the
// shaders' op counts by (tools/soc_model).  `vertices` = unique vertices of
// the drawn meshes x instances (a lower bound of vertex shader invocations;
// `indices` is the upper bound without post-transform reuse).
struct PassWork {
    std::string pass;          // render graph pass name
    u64         draws     = 0;
    u64         instances = 0;
    u64         indices   = 0; // index count x instances
    u64         vertices  = 0; // unique vertices x instances
    u64         pixels    = 0; // render area of the pass
    u64         threads   = 0; // compute threads dispatched
    u32         lights    = 0; // lights the fragment shader loops over
};

struct PassReport {
    std::string              name;       // unit name ("A + B" for a fused group)
    std::string              queue;      // "graphics" / "async"
    bool                     fused = false;
    std::vector<std::string> passes;     // render graph passes covered
    std::vector<std::string> shaders;    // their profile shaders (PassBuilder::setProfileShaders)
    u64                      dramBytes = 0; // estimated DRAM bytes per frame
    u32                      frames    = 0; // measured frames with a valid time for this unit
    TimingSummary            gpuMs;
    std::vector<PassWork>    work;       // OPT-0.4 (schema 3): passes with known work
};

// OPT-1 (schema 4): how the frame graph was compiled and what it costs.
struct GraphReport {
    bool        present = false;
    std::string mode;          // --graph-opt: off | greedy | plan
    std::string family;        // plan family of the graph (scenarios), or empty
    std::string plan;          // "applied", "none", or "rejected: <reason>"
    std::string alias;         // alias policy used
    std::string barriers;      // barrier policy used
    u32         passes       = 0; // live passes
    u32         renderPasses = 0;
    u32         memoryless   = 0;
    u32         barrierCount = 0;
    u64         dramBytes    = 0; // estimated DRAM bytes per frame (O1)
    u64         heapBytes    = 0; // transient heap (memory peak of the transients)
    u64         heapUnaliasedBytes = 0;
    u64         maxLiveBytes = 0; // lower bound of the heap for this order
};

// F5 (schema 5): the persistent GPU scene and its submission over the
// measured frames.  Per-frame quantities are summaries over those frames;
// GPU counters (visible, culled, draw commands) come from the frame's
// GPUSceneCounters read back METAL_FRAMES_IN_FLIGHT frames later.
struct SceneReport {
    bool          present = false;
    std::string   mode;            // gpu-driven "off" / "on"
    u32           instances = 0;   // live instances (last frame)
    u32           slots     = 0;
    u32           buckets   = 0;
    u32           materials = 0;
    u32           commands  = 0;   // ICB commands (buckets + sentinels)
    u32           structureChanges = 0; // structure events during the measured frames
    u32           queueOverflow    = 0; // GPU queue entries dropped (must be 0)
    TimingSummary uploadBytes;     // bytes written by the CPU for the scene per frame
    TimingSummary deltaRecords;    // delta records per frame (all buffers)
    TimingSummary visible;
    TimingSummary culledFrustum;
    TimingSummary culledDistance;
    TimingSummary culledSize;
    TimingSummary drawCommands;    // non-empty draws (on: ICB commands written; off: CPU draws)
    TimingSummary cpuCommands;     // commands the CPU encoded for the scene per frame (O8)
};

// F5 (schema 5): CPU ms per frame phase (replaces the F5 spike S1 line).
struct CpuPhasesReport {
    bool          present = false;
    TimingSummary sim;       // bench update
    TimingSummary sceneSync; // SceneStore::sync + lights
    TimingSummary prepare;   // renderer prepareFrame (delta upload)
    TimingSummary ui;
    TimingSummary graph;     // graph execution (encoding)
    TimingSummary submit;
};

// F6 (schema 6): every summary also has "p95"; the hardware is reported as
// physical device/family/memory, EFFECTIVE capabilities (--force-family) and
// the frozen preset (docs/plans/HARDWARE_VALIDATION.md, "Manifest").
struct DeviceReport {
    bool        present = false;
    std::string physicalDevice;        // GPU name
    std::string physicalFamily;        // apple10 / apple9
    u64         memoryBytes = 0;       // physical memory of the machine
    std::string effectiveCapabilities; // apple10 / apple9 (forced)
    std::string preset;                // frozen preset name ("" = none)
    std::string validationScope = "development";
    std::vector<std::string> unverifiedDevices;
};

// F6 (schema 6): the geometry path and the meshlet culling counters
// (GPUMeshletCounters, read back METAL_FRAMES_IN_FLIGHT frames later).
struct MeshletReport {
    bool          present = false;
    std::string   path;          // indexed / mesh
    std::string   cull;          // off / frustum / two-phase
    std::string   hizRequested;  // auto / compute / sampler
    std::string   hizEffective;  // compute / sampler / none
    std::string   cook;          // meshletOptionsName
    u32           meshlets = 0;  // meshlets of the scene's meshes
    u64           candidateCapacity = 0;
    u32           overflowFrames  = 0; // frames that used the indexed fallback (must be 0)
    u32           historyResets   = 0; // during the measured frames
    u32           checks = 0, checkFailures = 0; // --debug-meshlets
    TimingSummary candidates, drawnA, frustum, cone, historyRejected, drawnB, occludedB, primitives;
    TimingSummary emitted; // triangles kept by the mesh shaders' facing cull (<= primitives)
    TimingSummary sizeCulled; // approximate size cull (--meshlet-min-pixels; 0 in the exact preset)
};

struct RenderingReport {
    bool present = false, materialBinning = false, post = false, autoExposure = false, edr = false, offscreen = false;
    std::string path, asset, upscaler, tonemap;
    u32 inputWidth = 0, inputHeight = 0, views = 1, framesInFlight = 3, gpuFailures = 0;
    u64 temporalFrames = 0, nativeFrames = 0, historyResets = 0, binnedFrames = 0, genericFrames = 0;
    u32 guideChecks = 0, guideFailures = 0;
    u32 shadedPixels = 0, reusedPixels = 0;
    u64 deviceAllocatedBytes = 0, engineResourceBytes = 0;
    u64 workerDeviceBytes = 0, workerPhysicalFootprint = 0, workerBridgeBytes = 0, parentPhysicalFootprint = 0;
    u64 workersSpawned = 0, workersReaped = 0, workersPeakLive = 0, workerFailures = 0;
    u64 commandBufferRebuilds = 0;
    u64 commandBufferRebuildsMeasured = 0;
    float exposure = 1, headroom = 1, potentialHeadroom = 1;
    float mipBias = 0;
};
// F9 (schema 9): acceleration structures, measured proxy error and ray probes.
// present controls JSON emission; enabled distinguishes requested RT from a
// report of the disabled path. Bytes/counts are exact integers, timing/error
// units are explicit. Proxy errors describe the manifest's offline corpus.
struct RtReport {
    bool present = false;
    bool enabled = false;
    std::string proxyMode, probe, alphaStrategy, effectiveFamily;
    u32 blasCount = 0, compactedCount = 0, instances = 0, capacity = 0;
    u32 checks = 0, checkFailures = 0;
    u64 blasBytes = 0, uncompactedBytes = 0, blasScratchBytes = 0;
    u64 tlasBytes = 0, tlasScratchBytes = 0;
    u64 fullTriangles = 0, proxyTriangles = 0, tlasBuilds = 0, tlasRefits = 0;
    u64 blasBuilds = 0, blasRefits = 0, compactions = 0; // executed operations, distinct from live AS counts
    u64 proxyMeshes = 0, probeRays = 0, visibilityCompared = 0, visibilityMismatches = 0;
    u64 alphaTests = 0, opaqueAlphaTests = 0;
    float blasBuildMs = 0;
    float proxyShadowErrorPct = 0, proxyPrimaryErrorPct = 0, proxyDt95 = 0, proxyAcnePct = 0; // dt95: cm
    TimingSummary tlasUpdateMs, probeMs, probeNsPerRay;
};
constexpr u32 BENCH_REPORT_SCHEMA_VERSION = 9;

struct BenchReport {
    std::string   bench;
    std::string   device;
    u32           width  = 0;
    u32           height = 0;
    bool          vsync  = true;
    bool          ui     = true;
    u32           frames = 0;
    float         fps    = 0.0f; // 1000 / mean frame time
    u64           gpuAllocations = 0; // GPU buffers/textures created while measuring (O7: must be 0)
    // CPU heap growth over the measured frames (all malloc zones): live
    // blocks and bytes at the end minus at the start.  ~0 = flat (O7).
    i64           cpuHeapBlocksDelta = 0;
    i64           cpuHeapBytesDelta  = 0;
    TimingSummary frameMs;
    TimingSummary cpuMs;
    TimingSummary gpuMs;
    TimingSummary waitMs;
    std::string   pipelinesJson; // F3.5: PipelineStats as a JSON object (empty: omitted)
    // F4.1/F4.6: per-pass GPU timing (empty `passes` when timing is off).
    bool          gpuTiming        = false;
    bool          gpuTimingUnfused = false; // graph compiled without raster fusion
    std::vector<PassReport> passes;
    TimingSummary gpuPassSumMs;
    TimingSummary gpuFrameSpanMs;
    GraphReport   graph;         // OPT-1 (schema 4)
    SceneReport   scene;         // F5 (schema 5)
    CpuPhasesReport cpuPhases;   // F5 (schema 5)
    DeviceReport  hardware;      // F6 (schema 6)
    MeshletReport meshlets;      // F6 (schema 6)
    RenderingReport rendering;   // F7/F8 (schema 7), actual paths and last-frame content extent
    RtReport rt;                 // F9 (schema 9)
};

/// Nearest-rank statistics of `values` (empty input gives all zeros).
TimingSummary summarize(std::vector<float> values);

/// Fill the timing fields of `report` from the recorded samples.
void summarizeSamples(const std::vector<FrameSample>& samples, BenchReport& report);

/// One-line human-readable summary for the log.
std::string formatReportLine(const BenchReport& report);

/// JSON object with every field of the report.
std::string reportToJson(const BenchReport& report);

} // namespace phosphor
