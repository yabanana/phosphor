#pragma once

#include "core/types.h"
#include "diagnostics/debug_overlay.h"
#include "renderer/aa_pass.h"
#include "renderer/simple_pass.h"
#include "testbench/testbench.h"
#include <memory>

namespace phosphor {

// Forward declarations -- avoid pulling large headers into every TU
class Window;
class Input;
class Timer;
class VulkanDevice;
class GpuAllocator;
class SyncManager;
class CommandManager;
class Swapchain;
class BindlessDescriptorManager;
class PipelineManager;
class ECS;
class Camera;
class GpuScene;
class TextureManager;
class VisibilityBuffer;
class MeshPass;
class MaterialResolve;
class TonemapPass;
class CompositePass;
class SimplePass;
class GpuProfiler;
class FrameStats;
class RenderDocCapture;
class ResourceTracker;
class ImGuiLayer;
class TestBench;

// ---------------------------------------------------------------------------
// Engine -- composition root.  Owns every subsystem and drives the main loop.
//
// Lifetime:
//   1. Constructor creates the window and all Vulkan infrastructure.
//   2. run() enters the main loop: poll events, update, render, present.
//   3. Destructor tears down in reverse order.
//
// Test-bench switching is handled at runtime via the ImGui selector or the
// 1-7 keyboard shortcuts.
// ---------------------------------------------------------------------------

class Engine {
public:
    Engine(int argc, char* argv[]);
    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    Engine(Engine&&) = delete;
    Engine& operator=(Engine&&) = delete;

    /// Enter the main loop. Returns when the window is closed.
    void run();

private:
    void initSystems();
    void shutdownSystems();
    void update(float dt);
    void render();
    void switchTestBench(TestBenchType type);
    void handleInput();
    void uploadSceneData();

    // ---- Core ----
    std::unique_ptr<Window>  window_;
    std::unique_ptr<Input>   input_;
    std::unique_ptr<Timer>   timer_;

    // ---- RHI ----
    std::unique_ptr<VulkanDevice>             device_;
    std::unique_ptr<GpuAllocator>             allocator_;
    std::unique_ptr<SyncManager>              sync_;
    std::unique_ptr<CommandManager>           commands_;
    std::unique_ptr<Swapchain>                swapchain_;
    std::unique_ptr<BindlessDescriptorManager> descriptors_;
    std::unique_ptr<PipelineManager>          pipelines_;

    // ---- Scene ----
    std::unique_ptr<ECS>            ecs_;
    std::unique_ptr<Camera>         camera_;
    std::unique_ptr<GpuScene>       gpuScene_;
    std::unique_ptr<TextureManager> textures_;

    // ---- Renderer passes ----
    std::unique_ptr<VisibilityBuffer> visBuffer_;
    std::unique_ptr<MeshPass>         meshPass_;
    std::unique_ptr<MaterialResolve>  materialResolve_;
    std::unique_ptr<TonemapPass>      tonemapPass_;
    std::unique_ptr<CompositePass>    compositePass_;
    std::unique_ptr<SimplePass>       simplePass_;
    // HiZ, shadows, ReSTIR, AA, DDGI can be added incrementally

    // ---- Diagnostics ----
    std::unique_ptr<GpuProfiler>     profiler_;
    std::unique_ptr<FrameStats>      frameStats_;
    std::unique_ptr<RenderDocCapture> renderdoc_;
    std::unique_ptr<ResourceTracker> resourceTracker_;

    // ---- ImGui ----
    std::unique_ptr<ImGuiLayer> imgui_;

    // ---- Test bench ----
    std::unique_ptr<TestBench> activeBench_;
    TestBenchType currentBenchType_ = TestBenchType::TorusDemo;

    // ---- Frame state ----
    u32   currentFrame_ = 0;
    float exposure_     = 1.0f;
    OverlayMode overlayMode_ = OverlayMode::None;
    AAMode      aaMode_      = AAMode::None;
    bool  ddgiEnabled_   = false;
    bool  restirEnabled_ = true;
    bool  orbitMode_     = false;
    bool  running_       = true;
};

} // namespace phosphor
