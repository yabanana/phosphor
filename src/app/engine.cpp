#include "app/engine.h"

#include "core/input.h"
#include "core/log.h"
#include "core/timer.h"
#include "diagnostics/frame_stats.h"
#include "platform/metal/metal_context.h"
#include "platform/metal/metal_texture_manager.h"
#include "platform/metal/scene_renderer.h"
#include "renderer/gpu_scene.h"
#include "scene/camera.h"
#include "scene/ecs.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_metal.h>

#include <imgui.h>
#include <imgui_impl_metal.h>
#include <imgui_impl_sdl3.h>

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace phosphor {

namespace {

std::string shaderLibraryPath() {
    const char* base = SDL_GetBasePath();
    return std::string(base ? base : "") + "shaders/phosphor.metallib";
}

std::optional<TestBenchType> benchFromArgs(int argc, char* argv[]) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--bench") == 0) {
            const int index = std::atoi(argv[i + 1]) - 1; // 1-based like the hotkeys
            if (index >= 0 && index < testBenchCount()) {
                return static_cast<TestBenchType>(index);
            }
        }
    }
    return std::nullopt;
}

} // namespace

Engine::Engine(int argc, char* argv[]) {
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

    context_ = std::make_unique<MetalContext>(layer);
    int w = 0, h = 0;
    SDL_GetWindowSizeInPixels(window_, &w, &h);
    context_->resize(static_cast<u32>(w), static_cast<u32>(h));

    renderer_ = std::make_unique<SceneRenderer>(*context_, shaderLibraryPath());

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
    ImGui_ImplMetal_Init(context_->device());

    switchTestBench(benchFromArgs(argc, argv).value_or(TestBenchType::TorusDemo));
}

Engine::~Engine() {
    if (context_) context_->waitIdle();

    if (activeBench_) {
        activeBench_->teardown(*ecs_, *gpuScene_);
        activeBench_.reset();
    }

    ImGui_ImplMetal_Shutdown();
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
    while (running_) {
        processEvents();
        if (!running_) break;

        if (pendingBench_) {
            switchTestBench(*pendingBench_);
            pendingBench_.reset();
        }

        timer_->tick();
        frame(timer_->getDeltaTime());
        input_->resetFrameState();
    }
    context_->waitIdle();
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
    activeBench_ = createTestBench(type);
    LOG_INFO("Switching to test bench: %s", activeBench_->getName());
    activeBench_->setup(*ecs_, *gpuScene_, *textures_);

    textures_->flushUploads();
    renderer_->syncGeometry(*gpuScene_);
    aimCamera(activeBench_->getDefaultCamera());
    pool->release();
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

void Engine::frame(float dt) {
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
    if (!context_->beginFrame(frame)) {
        pool->release();
        return;
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

    renderer_->render(frame, *gpuScene_, frameScene_, constants, textures_->tableAddress());

    MTL::CommandBuffer* overlay = context_->submitScene(frame);

    // --- ImGui overlay (Metal 3 queue, ordered after the scene) -----------
    MTL::RenderPassDescriptor* uiPass = MTL::RenderPassDescriptor::renderPassDescriptor();
    MTL::RenderPassColorAttachmentDescriptor* color = uiPass->colorAttachments()->object(0);
    color->setTexture(frame.drawable->texture());
    color->setLoadAction(MTL::LoadActionLoad);
    color->setStoreAction(MTL::StoreActionStore);

    ImGui_ImplMetal_NewFrame(uiPass);
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
    MTL::RenderCommandEncoder* uiEncoder = overlay->renderCommandEncoder(uiPass);
    uiEncoder->setLabel(NS::String::string("ImGui", NS::UTF8StringEncoding));
    ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(), overlay, uiEncoder);
    uiEncoder->endEncoding();

    context_->presentFrame(frame, overlay);
    pool->release();
}

} // namespace phosphor
