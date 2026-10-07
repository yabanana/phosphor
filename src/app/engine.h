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
#include "rendergraph/optimizer/plan.h"
#include "rendergraph/render_graph.h"
#include "renderer/gpu_types.h"
#include "renderer/history_registry.h"
#include "renderer/scene_extract.h"
#include "testbench/testbench.h"

#include <glm/glm.hpp>

#include <array>
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
class ScenarioPasses;
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
class VisibilityRenderer;
class PostProcessor;
class MeshRenderer;
class SceneStore;
class GpuSceneChecker;
class Timer;
class AccelerationStructures;
class RtChecker;
class RtProxyTransitionCheck;
class RtVisibilityChecker;
class ShadowPasses;

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
    /// Declare every pass of the frame graph (reset first); no compilation.
    void declareFrameGraph(u32 width, u32 height);
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
    /// F5: the GPU scene counters of the frame that last used `slot` (it is
    /// complete): report samples and the panel.
    void onSceneCounters(u32 slot);
    /// GPU must be idle; preserve pending counters/times before resource reset.
    void drainFrameReadbacks();
    /// F5: --debug-gpu-scene: wait for the frame, read the scene back and
    /// compare it with the CPU mirror and references; false on FAIL.
    bool checkGpuScene(u32 slot);
    bool checkRayTracing(u32 slot);
    /// OPT-0.4 work of the forward pass for the report (F5: from the scene store).
    [[nodiscard]] ForwardWork sceneForwardWork(u32 width, u32 height) const;

    SDL_Window* window_    = nullptr;
    void*       metalView_ = nullptr; // SDL_MetalView

    std::unique_ptr<MetalContext>        context_;
    std::unique_ptr<PipelineCache>       pipelines_; // F3: every pipeline of the engine
    std::unique_ptr<ShaderReloader>      reloader_;  // F3.6 hot reload (Debug)
    std::unique_ptr<SceneRenderer>       renderer_;
    std::unique_ptr<AccelerationStructures> rt_;
    std::unique_ptr<RtChecker> rtChecker_;
    std::unique_ptr<RtProxyTransitionCheck> rtProxyTransition_;
    std::unique_ptr<RtVisibilityChecker> rtVisibility_;
    u64 rtCheckerGeometry_ = ~u64{0};
    u32 rtChecks_ = 0, rtFailures_ = 0;
    u64 rtCheckedRays_ = 0, rtAmbiguous_ = 0, rtUnsupported_ = 0;
    u64 rtVisibilityCompared_ = 0, rtVisibilityMismatches_ = 0;
    std::vector<GPURtRay> rtCheckRays_;
    std::vector<GPURtHit> rtCheckHits_;
    std::vector<GPUVertex> rtDeformedVertices_;
    std::vector<float> rtTlasTimes_, rtProbeTimes_, rtProbeNs_;
    u64 rtProbeRays_ = 0, rtAlphaTests_ = 0, rtOpaqueAlphaTests_ = 0;
    std::array<GPURtCounters, 3> rtCounterSnapshots_{};
    std::array<u64, 3> rtCounterFrames_{~u64{0}, ~u64{0}, ~u64{0}};
    std::unique_ptr<VisibilityRenderer> visibility_;
    std::unique_ptr<ShadowPasses> shadows_;
    std::unique_ptr<PostProcessor> post_;
    u64 sceneEpoch_ = 0;
    float dynamicScale_ = 1.0f;
    u32 renderBackingWidth_ = 0, renderBackingHeight_ = 0;
    float displayHeadroom_ = 1.0f, displayPotentialHeadroom_ = 1.0f;
    bool displayEDR_ = false;
    std::unique_ptr<MeshRenderer>        mesh_;      // F6 --geometry-path mesh
    std::unique_ptr<MetalTextureManager> textures_;
    std::unique_ptr<ImGuiRenderer>       imguiRenderer_;
    std::unique_ptr<FrameCapture>        capture_;
    std::unique_ptr<MetalGraphExecutor>  graphExecutor_;
    std::unique_ptr<GraphDebugPasses>    graphDebug_; // --debug-graph-transients
    std::unique_ptr<AsyncComputeProbe>   asyncProbe_;  // --debug-async-compute
    std::unique_ptr<KnownCostPass>       knownCost_;   // --debug-gpu-cost (F4.1)
    std::unique_ptr<ScenarioPasses>      scenario_;    // --graph-scenario (OPT-1)
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
    std::unique_ptr<SceneStore> store_;       // F5.1 CPU mirror of the GPU scene
    std::vector<GPULight>       lights_;      // extracted every frame (few)
    double                      sceneTime_ = 0.0; // simulated seconds since the bench started (motion)
    std::array<float, SCENE_MOTION_CLASSES * 2> motionSinCos_{};
    GPUCullParams               cullParams_{};
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
    u64 commandRebuildsAtStart_ = 0;
    u64                      heapBlocksAtStart_  = 0;
    u64                      heapBytesAtStart_   = 0;
    // CPU time spent blocked in beginFrame() (slot + drawable waits).
    std::chrono::steady_clock::duration frameWait_{};
    // F5 (report schema 5): per measured frame CPU phases and scene samples.
    struct SceneSamples {
        std::vector<float> sim, sceneSync, prepare, ui, graph, submit;            // CPU ms
        std::vector<float> uploadBytes, deltaRecords, cpuCommands;                 // per encoded frame
        std::vector<float> visible, culledFrustum, culledDistance, culledSize, drawCommands; // GPU counters
        u32 structureChanges = 0;
        u32 queueOverflow    = 0;
        void reserve(u32 frames);
    } sceneSamples_;
    std::array<u64, 3> slotFrame_{~0ull, ~0ull, ~0ull}; // frame index last submitted per slot
    std::array<bool, 3> slotMeasured_{};                // ... and whether it was a measured frame
    GPUSceneCounters lastCounters_{};
    u32  lastCpuCommands_  = 0;
    std::unique_ptr<GpuSceneChecker> sceneChecker_; // --debug-gpu-scene
    u32  gpuSceneChecks_   = 0;
    u32  gpuSceneFailures_ = 0;
    // Camera metadata accompanies the per-view geometry history registry.
    // The registered matrix is jittered for Hi-Z; MetalFX registers its
    // independent unjittered signal in PostProcessor.
    struct HiZHistory {
        glm::vec3 cameraPosition{0.0f};
        glm::vec3 cameraFront{0.0f,0.0f,-1.0f};
        double sceneTime=0;
    };
    std::array<HiZHistory,HistoryRegistry::MaxViews> histories_{};
    HistoryRegistry hizRegistry_;
    const HistoryRegistry::View &hizHistory() const { return hizRegistry_.get(currentView_); }
    u32 currentView_ = 0;
    u32 shaderGeneration_ = ~u32{0};
    HiZHistory &history() { return histories_[currentView_]; }
    const HiZHistory &history() const { return histories_[currentView_]; }
    struct MeshletSamples {
        std::vector<float> candidates, drawnA, frustum, cone, historyRejected, drawnB, occludedB, primitives, emitted, sizeCulled;
        u32 overflowFrames = 0;
        u32 historyResets  = 0;
        void reserve(u32 frames);
    } meshletSamples_;
    GPUMeshletCounters lastMeshletCounters_{};
    u32  meshletChecks_   = 0;
    u32  meshletFailures_ = 0;
    std::unique_ptr<class MeshletChecker> meshletChecker_; // --debug-meshlets
    /// F6.7 --debug-meshlets: read the frame's meshlet data back and check it; false on FAIL.
    bool checkMeshlets(u32 slot);

    // F3 hitch measurement: per-frame records and bench-switch phases.
    FrameTrace                  trace_;
    u32                         frameFlags_ = 0;        // FrameFlags of the frame being produced
    std::optional<SwitchRecord> pendingSwitch_;         // completed after the switch frame
    u32                         requestsBeforeFrame_ = 0;
    double                      requestMsBeforeFrame_ = 0.0;
    double                      rtCompileMsBeforeFrame_ = 0.0;
    std::chrono::steady_clock::time_point launch_;
    float eventPumpMs_ = 0.0f;
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
        GpuDrivenMode gpuDriven = GpuDrivenMode::Off;
        u64  sceneBuffers  = 0; // GpuSceneBuffers::version(): capacities changed
        u64  meshletBuffers = 0; // F6: MeshRenderer::version() (+ debug view)
        u32  debugView      = 0;
        rg::Format outputFormat = rg::Format::BGRA8Srgb;
        u32 backingWidth = 0, backingHeight = 0;
        u64 rtResources = 0;
        bool rtVisibilityReady = false;
        u64 lightingResources = 0;
        bool operator==(const GraphKey&) const = default;
    };
    rg::RenderGraph frameGraph_;
    GraphKey        graphKey_;
    GraphReport     graphReport_;                 // OPT-1: how the graph was compiled
    std::vector<rg::GraphPlan> graphPlans_;       // --graph-opt plan
    OverlayMode     overlayMode_ = OverlayMode::None; // --overlay, changed from the Rendering panel
    rg::TextureRef  drawableRef_;
    rg::BufferRef   captureRef_;
    rg::TextureRef captureColorRef_;
    bool            captureThisFrame_ = false;

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
