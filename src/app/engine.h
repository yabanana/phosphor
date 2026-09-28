#pragma once

#include "core/launch_options.h"
#include "core/types.h"
#include "diagnostics/bench_report.h"
#include "platform/metal/gpu_memory.h"
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
class GpuScene;
class ImGuiRenderer;
class Input;
class MetalContext;
class MetalGraphExecutor;
class MemoryPressureMonitor;
class MetalTextureManager;
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
    void finishBenchmark();

    SDL_Window* window_    = nullptr;
    void*       metalView_ = nullptr; // SDL_MetalView

    std::unique_ptr<MetalContext>        context_;
    std::unique_ptr<SceneRenderer>       renderer_;
    std::unique_ptr<MetalTextureManager> textures_;
    std::unique_ptr<ImGuiRenderer>       imguiRenderer_;
    std::unique_ptr<FrameCapture>        capture_;
    std::unique_ptr<MetalGraphExecutor>  graphExecutor_;
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

    // Render graph of the frame (F2), rebuilt only when its key changes.
    struct GraphKey {
        u32  width   = 0;
        u32  height  = 0;
        bool ui      = false;
        bool capture = false;
        bool operator==(const GraphKey&) const = default;
    };
    rg::RenderGraph frameGraph_;
    GraphKey        graphKey_;
    rg::TextureRef  drawableRef_;
    rg::BufferRef   captureRef_;
    bool            captureThisFrame_ = false;

    FrameScene      frameScene_;
    MemoryPanelInfo memoryInfo_; // reused every frame (keeps vector capacity)
    std::vector<GpuMemory::HeapStats> heapStatsScratch_;
    RenderSettings settings_;
    bool           captured_  = false;
    bool           orbitMode_ = false;
    bool           running_   = true;
    int            exitCode_  = 0;
};

} // namespace phosphor
