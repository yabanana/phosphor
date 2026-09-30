#pragma once

#include "core/launch_options.h"
#include "core/types.h"
#include "diagnostics/bench_report.h"
#include "diagnostics/frame_trace.h"
#include "diagnostics/pass_timings.h"
#include "diagnostics/tracy_gpu.h"
#include "platform/metal/gpu_memory.h"
#include "platform/metal/gpu_timestamps.h"
#include "imgui/ui_panels.h"
#include "rendergraph/render_graph.h"
#include "renderer/scene_extract.h"
#include "testbench/testbench.h"

#include <chrono>
#include <memory>
#include <optional>
#include <vector>

struct SDL_Window;

namespace phosphor {

class Camera;
class ECS;
class FrameCapture;
class FrameStats;
class GpuCapture;
class GpuScene;
class KnownCostPass;
class GraphDebugPasses;
class AsyncComputeProbe;
class DebugOverlays;
class ImGuiRenderer;
class Input;
class MetalContext;
class MetalGraphExecutor;
class MemoryPressureMonitor;
class MetalTextureManager;
class PipelineCache;
class ShaderReloader;
class SceneRenderer;
class Timer;

// ---------------------------------------------------------------------------
// Engine -- composition root for the macOS app.
// Owns the SDL window, the Metal 4 backend, the scene and the test benches.
// ---------------------------------------------------------------------------

class Engine {
public:
    Engine(int argc, char* argv[]);
    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    void run();

    /// Process exit code: non-zero when a self-test (--memory-stress) failed.
    [[nodiscard]] int exitCode() const { return exitCode_; }

private:
    void processEvents();
    void injectSyntheticInput();
    void handleShortcuts();
    void switchTestBench(TestBenchType type);
    void aimCamera(const CameraSetup& setup);
    void logMemory() const;
    void fillMemoryInfo();
    void handleMemoryPressure();
    /// Simulate and render one frame; false if nothing was presented.
    bool frame(float dt);
    /// Describe the frame as a render graph and compile it; only when the
    /// graph key changes (resize, UI or capture toggled).
    void buildFrameGraph(u32 width, u32 height);
    void drawUi();
    void recordBenchmarkFrame(float dt, float cpuMs, float waitMs);
    /// Hand a hot-reloaded shader library to the pipeline cache (frame start).
    void pollShaderReload();
    /// --debug-hot-reload: exact check of the captured frame; true on PASS.
    [[nodiscard]] bool checkHotReloadCapture() const;
    /// Benchmark frames are being measured (after the warm-up).
    [[nodiscard]] bool measuring() const;
    /// Path of the pipeline archive to load ("" = none), from the options.
    [[nodiscard]] std::string pipelineArchivePath() const;
    void finishBenchmark();
    /// F4.1: consume the GPU times of a completed frame (panel window,
    /// benchmark measurement, Tracy GPU zones, capture threshold).
    void onFrameTimes(const GpuTimestamps::Resolved& r);

    SDL_Window* window_    = nullptr;
    void*       metalView_ = nullptr; // SDL_MetalView

    std::unique_ptr<MetalContext>        context_;
    std::unique_ptr<PipelineCache>       pipelines_; // F3: every pipeline of the engine
    std::unique_ptr<ShaderReloader>      reloader_;  // F3.6 hot reload (Debug)
    std::unique_ptr<SceneRenderer>       renderer_;
    std::unique_ptr<MetalTextureManager> textures_;
    std::unique_ptr<ImGuiRenderer>       imguiRenderer_;
    std::unique_ptr<FrameCapture>        capture_;
    std::unique_ptr<MetalGraphExecutor>  graphExecutor_;
    std::unique_ptr<GraphDebugPasses>    graphDebug_; // --debug-graph-transients
    std::unique_ptr<AsyncComputeProbe>   asyncProbe_;  // --debug-async-compute
    std::unique_ptr<KnownCostPass>       knownCost_;   // --debug-gpu-cost (F4.1)
    std::unique_ptr<GpuTimestamps>       timestamps_;  // F4.1 (null with --no-gpu-timing)
    std::unique_ptr<GpuCapture>          gpuCapture_;  // F4.3 (--gpu-capture*)
    PassTimings                          passTimings_;
    TracyGpuZones                        tracyGpu_;
    // F4.1 benchmark measurement: frame indices [first, last] of the measured
    // frames; their timestamps are resolved METAL_FRAMES_IN_FLIGHT frames later.
    u64                                  measureFirstFrame_ = ~0ull;
    u64                                  measureLastFrame_  = ~0ull;
    bool                                 passMeasureStarted_ = false;
    std::vector<std::string>             passNames_;    // per graph pass (PassTimings::configure)
    std::vector<std::string>             passShaders_;
    std::unique_ptr<DebugOverlays>       overlays_;    // F4.7 heatmaps
    std::unique_ptr<MemoryPressureMonitor> pressure_;

    std::unique_ptr<ECS>        ecs_;
    std::unique_ptr<GpuScene>   gpuScene_;
    std::unique_ptr<Camera>     camera_;
    std::unique_ptr<Input>      input_;
    std::unique_ptr<Timer>      timer_;
    std::unique_ptr<FrameStats> frameStats_;

    std::unique_ptr<TestBench> activeBench_;
    TestBenchType              currentBench_ = TestBenchType::TorusDemo;
    std::optional<TestBenchType> pendingBench_;

    LaunchOptions  options_;
    // Benchmark mode (--frames): presented frames so far and measured samples.
    u32                      presentedFrames_ = 0;
    u32                      framesOnBench_   = 0; // for --switch-every
    u32                      simulatedFrames_ = 0; // for --simulate-pressure
    u32                      ignoredInputEvents_ = 0; // benchmark mode ignores input
    std::vector<FrameSample> samples_;
    u64                      allocationsAtStart_ = 0;
    u64                      heapBlocksAtStart_  = 0;
    u64                      heapBytesAtStart_   = 0;
    // CPU time spent blocked in beginFrame() (slot + drawable waits).
    std::chrono::steady_clock::duration frameWait_{};

    // F3 hitch measurement: per-frame records and bench-switch phases.
    FrameTrace                  trace_;
    u32                         frameFlags_ = 0;        // FrameFlags of the frame being produced
    std::optional<SwitchRecord> pendingSwitch_;         // completed after the switch frame
    u32                         requestsBeforeFrame_ = 0;
    double                      requestMsBeforeFrame_ = 0.0;
    double                      rtCompileMsBeforeFrame_ = 0.0;
    std::chrono::steady_clock::time_point launch_;
    bool                        firstFrameLogged_ = false;
    float                       startupPipelinesMs_ = 0.0f;
    bool                        hotReloadRequested_ = false; // --debug-hot-reload issued

    // Render graph of the frame (F2), rebuilt only when its key changes.
    struct GraphKey {
        u32  width   = 0;
        u32  height  = 0;
        bool ui      = false;
        bool capture = false;
        bool splitEncoding = false;
        bool asyncCompute  = false;
        OverlayMode overlay = OverlayMode::None;
        bool operator==(const GraphKey&) const = default;
    };
    rg::RenderGraph frameGraph_;
    GraphKey        graphKey_;
    OverlayMode     overlayMode_ = OverlayMode::None; // --overlay, changed from the Rendering panel
    rg::TextureRef  drawableRef_;
    rg::BufferRef   captureRef_;
    bool            captureThisFrame_ = false;

    FrameScene      frameScene_;
    MemoryPanelInfo memoryInfo_; // reused every frame (keeps vector capacity)
    std::vector<GpuMemory::HeapStats> heapStatsScratch_;
    RenderSettings settings_;
    bool           captured_  = false;
    bool           orbitMode_ = false;
    bool           resizeToggle_ = false; // --resize-every
    u32            resizeFrames_ = 0;
    bool           running_   = true;
    int            exitCode_  = 0;
};

} // namespace phosphor
