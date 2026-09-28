#include "app/engine.h"

#include "core/input.h"
#include "core/log.h"
#include "core/timer.h"
#include "diagnostics/frame_stats.h"
#include "imgui/imgui_renderer.h"
#include "platform/metal/frame_capture.h"
#include "platform/metal/gpu_memory.h"
#include "platform/metal/metal_context.h"
#include "platform/metal/metal_texture_manager.h"
#include "platform/metal/scene_renderer.h"
#include "renderer/gpu_scene.h"
#include "scene/camera.h"
#include "scene/ecs.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_metal.h>

#include <imgui.h>
#include <imgui_impl_sdl3.h>

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

namespace phosphor {

namespace {

std::string shaderLibraryPath() {
    const char* base = SDL_GetBasePath();
    return std::string(base ? base : "") + "shaders/phosphor.metallib";
}

using Clock = std::chrono::steady_clock;

float toMs(Clock::duration d) {
    return std::chrono::duration<float, std::milli>(d).count();
}

} // namespace

Engine::Engine(int argc, char* argv[]) {
    std::string error;
    if (!parseLaunchOptions(argc, argv, testBenchCount(), options_, error)) {
        throw std::runtime_error("Invalid arguments: " + error);
    }
    settings_.vsync = options_.vsync;

    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS)) {
        throw std::runtime_error(std::string("SDL_Init failed: ") + SDL_GetError());
    }

    window_ = SDL_CreateWindow("Phosphor", 1600, 900,
                               SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY | SDL_WINDOW_METAL);
    if (!window_) {
        throw std::runtime_error(std::string("SDL_CreateWindow failed: ") + SDL_GetError());
    }
    metalView_ = SDL_Metal_CreateView(window_);
    auto* layer = static_cast<CA::MetalLayer*>(SDL_Metal_GetLayer(static_cast<SDL_MetalView>(metalView_)));
    if (!layer) {
        throw std::runtime_error("Failed to obtain CAMetalLayer from SDL");
    }

    context_ = std::make_unique<MetalContext>(layer, shaderLibraryPath());
    int w = 0, h = 0;
    SDL_GetWindowSizeInPixels(window_, &w, &h);
    context_->resize(static_cast<u32>(w), static_cast<u32>(h));

    renderer_ = std::make_unique<SceneRenderer>(*context_);

    ecs_        = std::make_unique<ECS>();
    gpuScene_   = std::make_unique<GpuScene>();
    camera_     = std::make_unique<Camera>(glm::radians(60.0f), static_cast<float>(w) / std::max(h, 1), 0.05f, 1000.0f);
    input_      = std::make_unique<Input>();
    timer_      = std::make_unique<Timer>();
    frameStats_ = std::make_unique<FrameStats>();

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::GetIO().IniFilename = nullptr; // keep the repo free of imgui.ini
    ImGui::StyleColorsDark();
    ImGui_ImplSDL3_InitForMetal(window_);
    imguiRenderer_ = std::make_unique<ImGuiRenderer>(*context_);
    if (!options_.capturePath.empty()) {
        capture_ = std::make_unique<FrameCapture>(*context_);
    }

    switchTestBench(options_.bench ? static_cast<TestBenchType>(*options_.bench) : TestBenchType::TorusDemo);
}

Engine::~Engine() {
    if (context_) context_->waitIdle();

    if (activeBench_) {
        activeBench_->teardown(*ecs_, *gpuScene_);
        activeBench_.reset();
    }

    capture_.reset();
    imguiRenderer_.reset();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();

    textures_.reset();
    renderer_.reset();
    context_.reset();

    if (metalView_) SDL_Metal_DestroyView(static_cast<SDL_MetalView>(metalView_));
    if (window_) SDL_DestroyWindow(window_);
    SDL_Quit();
}

void Engine::run() {
    LOG_INFO("Entering main loop");
    if (options_.benchmark()) {
        LOG_INFO("Benchmark: %u warm-up + %u measured frames, vsync %s, UI %s", options_.warmup, options_.frames,
                 options_.vsync ? "on" : "off", options_.ui ? "on" : "off");
        if (options_.warmup == 0) {
            context_->beginGpuTimeCapture(options_.frames);
            allocationsAtStart_ = context_->memory().allocationCount();
        }
    }
    while (running_) {
        const Clock::time_point start = Clock::now();
        processEvents();
        if (!running_) break;

        if (pendingBench_) {
            switchTestBench(*pendingBench_);
            pendingBench_.reset();
        }

        timer_->tick();
        // Fixed step: deterministic animation for captures; timings stay real.
        const float simDt = options_.fixedTimestep ? 1.0f / 60.0f : timer_->getDeltaTime();
        const bool presented = frame(simDt);
        input_->resetFrameState();
        if (presented && options_.switchEvery > 0 && ++framesOnBench_ >= options_.switchEvery && !pendingBench_) {
            // Same path as the 1-7 hotkeys.
            pendingBench_ = static_cast<TestBenchType>((static_cast<int>(currentBench_) + 1) % testBenchCount());
        }
        if (presented && options_.benchmark()) {
            recordBenchmarkFrame(timer_->getDeltaTime(), toMs(Clock::now() - start - frameWait_), toMs(frameWait_));
        }
    }
    context_->waitIdle();
    if (options_.benchmark()) {
        finishBenchmark();
    }
    if (capture_) {
        capture_->writePng(options_.capturePath);
    }
}

void Engine::recordBenchmarkFrame(float dt, float cpuMs, float waitMs) {
    ++presentedFrames_;
    if (presentedFrames_ == options_.warmup) {
        // GPU times are recorded from the next submitted frame on.
        context_->beginGpuTimeCapture(options_.frames);
        samples_.reserve(options_.frames);
        allocationsAtStart_ = context_->memory().allocationCount();
    } else if (presentedFrames_ > options_.warmup) {
        samples_.push_back({dt * 1000.0f, cpuMs, 0.0f, waitMs});
        if (samples_.size() == options_.frames) running_ = false;
    }
}

void Engine::finishBenchmark() {
    const std::vector<float> gpu = context_->endGpuTimeCapture();
    for (size_t i = 0; i < samples_.size() && i < gpu.size(); ++i) {
        samples_[i].gpuMs = gpu[i];
    }

    BenchReport report;
    report.bench  = activeBench_ ? activeBench_->getName() : "";
    report.device = context_->gpuName();
    report.width  = context_->width();
    report.height = context_->height();
    report.vsync  = settings_.vsync;
    report.ui     = options_.ui;
    report.gpuAllocations = context_->memory().allocationCount() - allocationsAtStart_;
    summarizeSamples(samples_, report);

    // stdout, not the log: scripts collect this line.
    std::printf("BENCH %s\n", formatReportLine(report).c_str());
    std::fflush(stdout);
    if (!options_.reportPath.empty()) {
        std::ofstream out(options_.reportPath);
        out << reportToJson(report);
        if (!out) LOG_ERROR("Failed to write %s", options_.reportPath.c_str());
    }
}

void Engine::processEvents() {
    const ImGuiIO& io = ImGui::GetIO();
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        ImGui_ImplSDL3_ProcessEvent(&event);

        switch (event.type) {
        case SDL_EVENT_QUIT:
        case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
            running_ = false;
            break;
        case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
            context_->resize(static_cast<u32>(event.window.data1), static_cast<u32>(event.window.data2));
            break;
        default:
            break;
        }

        const bool keyboardEvent = event.type == SDL_EVENT_KEY_DOWN || event.type == SDL_EVENT_KEY_UP;
        const bool mouseEvent = event.type == SDL_EVENT_MOUSE_MOTION || event.type == SDL_EVENT_MOUSE_WHEEL ||
                                event.type == SDL_EVENT_MOUSE_BUTTON_DOWN || event.type == SDL_EVENT_MOUSE_BUTTON_UP;
        if ((keyboardEvent && io.WantCaptureKeyboard) || (mouseEvent && io.WantCaptureMouse)) {
            continue;
        }
        input_->processEvent(event);
    }
    handleShortcuts();
}

void Engine::handleShortcuts() {
    if (input_->isKeyPressed(SDL_SCANCODE_ESCAPE)) {
        running_ = false;
    }
    for (int i = 0; i < testBenchCount(); ++i) {
        if (input_->isKeyPressed(static_cast<SDL_Scancode>(SDL_SCANCODE_1 + i))) {
            const auto type = static_cast<TestBenchType>(i);
            if (type != currentBench_) pendingBench_ = type;
        }
    }
    if (input_->isKeyPressed(SDL_SCANCODE_F1)) settings_.debugMode = 0;
    if (input_->isKeyPressed(SDL_SCANCODE_F2)) settings_.debugMode = 1;
    if (input_->isKeyPressed(SDL_SCANCODE_F3)) settings_.debugMode = 2;
}

void Engine::switchTestBench(TestBenchType type) {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    context_->waitIdle();

    if (activeBench_) {
        activeBench_->teardown(*ecs_, *gpuScene_);
        activeBench_.reset();
    }
    gpuScene_->clear();
    // Textures belong to a bench; a fresh manager drops the previous set.
    textures_.reset();
    textures_ = std::make_unique<MetalTextureManager>(*context_);

    currentBench_ = type;
    framesOnBench_ = 0;
    activeBench_ = createTestBench(type);
    LOG_INFO("Switching to test bench: %s", activeBench_->getName());
    activeBench_->setup(*ecs_, *gpuScene_, *textures_);

    textures_->flushUploads();
    renderer_->syncGeometry(*gpuScene_);
    // The previous bench's resources are unused now: free their heap ranges
    // and give back heaps that became empty.
    context_->collectGarbage();
    context_->memory().trimEmptyHeaps();
    logMemory();
    aimCamera(activeBench_->getDefaultCamera());
    pool->release();
}

void Engine::logMemory() const {
    const GpuMemory& memory = context_->memory();
    u64 heapBytes = 0, heapUsed = 0;
    float fragmentation = 0.0f;
    const auto heaps = memory.heapStats();
    for (const auto& h : heaps) {
        heapBytes += h.size;
        heapUsed += h.tlsf.usedBytes;
        fragmentation = std::max(fragmentation, h.tlsf.fragmentation());
    }
    LOG_INFO("GPU memory: %.1f MiB in use; %zu placement heaps, %.1f of %.1f MiB used, max fragmentation %.2f",
             static_cast<double>(memory.totalBytes()) / (1 << 20), heaps.size(),
             static_cast<double>(heapUsed) / (1 << 20), static_cast<double>(heapBytes) / (1 << 20),
             fragmentation);
    for (const ResidencyClass cls : {ResidencyClass::Static, ResidencyClass::Streaming}) {
        const ResidencyManager::Stats r = context_->residency().stats(cls);
        LOG_INFO("Residency %s: %u allocations, %.1f MiB, %u commits",
                 cls == ResidencyClass::Static ? "static" : "streaming", r.allocations,
                 static_cast<double>(r.bytes) / (1 << 20), r.commits);
    }
}

void Engine::aimCamera(const CameraSetup& setup) {
    const glm::vec3 toTarget = setup.target - setup.position;
    const glm::vec3 dir = glm::length(toTarget) > 1e-4f ? glm::normalize(toTarget) : glm::vec3(0, 0, -1);

    orbitMode_ = setup.orbit;
    if (setup.orbit) {
        // Orbit yaw/pitch describe the direction from the target to the camera.
        camera_->setOrbitMode(setup.target, setup.distance);
        camera_->setYawPitch(glm::degrees(std::atan2(-dir.z, -dir.x)), glm::degrees(std::asin(-dir.y)));
    } else {
        camera_->setPosition(setup.position);
        camera_->setYawPitch(glm::degrees(std::atan2(dir.z, dir.x)), glm::degrees(std::asin(dir.y)));
    }
}

bool Engine::frame(float dt) {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

    // --- Simulation -----------------------------------------------------------
    if (orbitMode_) {
        camera_->updateOrbit(*input_, dt);
    } else {
        camera_->updateFPS(*input_, dt);
    }
    camera_->setAspect(static_cast<float>(context_->width()) / static_cast<float>(std::max(context_->height(), 1u)));
    camera_->updateMatrices();

    activeBench_->update(dt, *ecs_);
    extractFrameScene(*ecs_, *gpuScene_, frameScene_);
    frameStats_->update(*timer_, context_->lastGpuMs());

    context_->layer()->setDisplaySyncEnabled(settings_.vsync);

    // --- Render ------------------------------------------------------------
    MetalContext::Frame frame;
    const Clock::time_point waitStart = Clock::now();
    const bool acquired = context_->beginFrame(frame);
    frameWait_ = Clock::now() - waitStart;
    if (!acquired) {
        pool->release();
        return false;
    }

    FrameConstants constants{};
    std::memcpy(constants.viewProjection, &camera_->getViewProjection()[0][0], sizeof(constants.viewProjection));
    std::memcpy(constants.view, &camera_->getView()[0][0], sizeof(constants.view));
    const glm::vec3 camPos = camera_->getPosition();
    constants.cameraPosition[0] = camPos.x;
    constants.cameraPosition[1] = camPos.y;
    constants.cameraPosition[2] = camPos.z;
    constants.cameraPosition[3] = static_cast<float>(timer_->getTotalTime());
    constants.lightCount = static_cast<u32>(frameScene_.lights.size());
    constants.debugMode  = static_cast<u32>(settings_.debugMode);
    constants.exposure   = settings_.exposure;
    constants.frameIndex = static_cast<u32>(frame.index);

    MTL4::RenderCommandEncoder* pass =
        renderer_->render(frame, *gpuScene_, frameScene_, constants, textures_->tableAddress());

    // --- ImGui overlay, appended to the scene pass --------------------------
    if (options_.ui) {
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();

        int bench = static_cast<int>(currentBench_);
        bool changed = false;
        UIPanels::drawTestBenchSelector(bench, changed);
        if (changed) pendingBench_ = static_cast<TestBenchType>(bench);

        RendererInfo info;
        info.gpuName     = context_->gpuName();
        info.apple9      = context_->isApple9OrLater();
        info.width       = context_->width();
        info.height      = context_->height();
        info.instances   = static_cast<u32>(frameScene_.instances.size());
        info.drawBatches = static_cast<u32>(frameScene_.batches.size());
        info.triangles   = renderer_->lastTriangleCount();
        info.meshlets    = gpuScene_->getMeshletTotalCount();
        info.textures    = textures_->textureCount();
        info.gpuMs       = context_->lastGpuMs();
        UIPanels::drawPerformancePanel(*frameStats_, info);
        UIPanels::drawRenderPanel(settings_);

        ImGui::Render();
        imguiRenderer_->render(pass, ImGui::GetDrawData());
    }
    pass->endEncoding();

    // Capture the last frame of a run (or the first frame when interactive).
    const bool lastFrame = !options_.benchmark() || presentedFrames_ + 1 == options_.warmup + options_.frames;
    if (capture_ && lastFrame && !captured_) {
        capture_->encode(frame);
        captured_ = true;
    }

    context_->submitFrame(frame);
    pool->release();
    return true;
}

} // namespace phosphor
