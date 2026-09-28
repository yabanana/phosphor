#pragma once

#include "core/types.h"
#include "imgui/ui_panels.h"
#include "renderer/scene_extract.h"
#include "testbench/testbench.h"

#include <memory>
#include <optional>

struct SDL_Window;

namespace phosphor {

class Camera;
class ECS;
class FrameStats;
class GpuScene;
class Input;
class MetalContext;
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

private:
    void processEvents();
    void handleShortcuts();
    void switchTestBench(TestBenchType type);
    void aimCamera(const CameraSetup& setup);
    void frame(float dt);

    SDL_Window* window_    = nullptr;
    void*       metalView_ = nullptr; // SDL_MetalView

    std::unique_ptr<MetalContext>        context_;
    std::unique_ptr<SceneRenderer>       renderer_;
    std::unique_ptr<MetalTextureManager> textures_;

    std::unique_ptr<ECS>        ecs_;
    std::unique_ptr<GpuScene>   gpuScene_;
    std::unique_ptr<Camera>     camera_;
    std::unique_ptr<Input>      input_;
    std::unique_ptr<Timer>      timer_;
    std::unique_ptr<FrameStats> frameStats_;

    std::unique_ptr<TestBench> activeBench_;
    TestBenchType              currentBench_ = TestBenchType::TorusDemo;
    std::optional<TestBenchType> pendingBench_;

    FrameScene     frameScene_;
    RenderSettings settings_;
    bool           orbitMode_ = false;
    bool           running_   = true;
};

} // namespace phosphor
