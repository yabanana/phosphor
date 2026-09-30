#include "app/engine.h"

#include "core/input.h"
#include "core/log.h"
#include "core/timer.h"
#include "diagnostics/frame_stats.h"
#include "imgui/imgui_renderer.h"
#include "core/profile.h"
#include "platform/metal/frame_capture.h"
#include "platform/metal/gpu_capture.h"
#include "platform/metal/known_cost_pass.h"
#include "platform/metal/gpu_memory.h"
#include "platform/metal/async_compute_probe.h"
#include "platform/metal/debug_overlays.h"
#include "platform/metal/graph_debug_passes.h"
#include "platform/metal/memory_pressure.h"
#include "platform/metal/memory_stress.h"
#include "platform/metal/metal_context.h"
#include "platform/metal/metal_graph_executor.h"
#include "platform/metal/metal_texture_manager.h"
#include "platform/metal/pipeline_cache.h"
#include "platform/metal/scenario_passes.h"
#include "platform/metal/shader_reloader.h"
#include "platform/metal/scene_renderer.h"
#include "renderer/gpu_scene.h"
#include "rendergraph/graph_dump.h"
#include "rendergraph/pass_context.h"
#include "scene/camera.h"
#include "scene/ecs.h"

#include <SDL3/SDL.h>
#include <mach/mach_time.h>
#include <malloc/malloc.h>
#include <SDL3/SDL_metal.h>

#include <imgui.h>
#include <imgui_impl_sdl3.h>

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

// Defined by cmake/App.cmake (F3.6 hot reload); absent in metal_syntax_check.
#ifndef PHOSPHOR_SHADER_SOURCE_DIR
#define PHOSPHOR_SHADER_SOURCE_DIR "shaders"
#endif

namespace phosphor {

namespace {

std::string shaderPath(const char* file) {
    const char* base = SDL_GetBasePath();
    return std::string(base ? base : "") + "shaders/" + file;
}

std::string shaderLibraryPath() { return shaderPath("phosphor.metallib"); }

using Clock = std::chrono::steady_clock;

// Live blocks and bytes over every malloc zone (CPU heap flatness, O7).
void heapUsage(u64& blocks, u64& bytes) {
    malloc_statistics_t stats{};
    malloc_zone_statistics(nullptr, &stats);
    blocks = stats.blocks_in_use;
    bytes  = stats.size_in_use;
}

float toMs(Clock::duration d) {
    return std::chrono::duration<float, std::milli>(d).count();
}

} // namespace

Engine::Engine(int argc, char* argv[]) : launch_(Clock::now()) {
    std::string error;
    if (!parseLaunchOptions(argc, argv, testBenchCount(), options_, error)) {
        throw std::runtime_error("Invalid arguments: " + error);
    }
    settings_.vsync = options_.vsync;
    settings_.debugMode = static_cast<int>(options_.debugMode);
    overlayMode_ = options_.overlay;
    settings_.overlay = static_cast<int>(overlayMode_);

    // F4.3: the capture layer exists only if requested before any Metal
    // device is created -- SDL creates one with the window's CAMetalLayer, so
    // before SDL_Init (measured: setting it later has no effect).
    if (options_.gpuCapture) GpuCapture::enableLayer();

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

    PipelineCache::Options pipelineOptions;
    pipelineOptions.archivePath    = pipelineArchivePath();
    pipelineOptions.harvestPath    = options_.harvestPipelinesPath;
    pipelineOptions.sync           = options_.pipelineSync;
    pipelineOptions.interactiveQos = options_.compileQosInteractive;
    pipelineOptions.fallbackOnly   = options_.debugFlexiblePipelines;
    pipelines_     = std::make_unique<PipelineCache>(*context_, pipelineOptions);
    renderer_      = std::make_unique<SceneRenderer>(*context_, *pipelines_, options_.pipelineSalt,
                                                     /*genericOnly*/ options_.debugPipelineFallback,
                                                     options_.forceVariant);
    graphExecutor_ = std::make_unique<MetalGraphExecutor>(*context_);
    if (options_.debugGraphTransients) graphDebug_ = std::make_unique<GraphDebugPasses>(*context_, *pipelines_);
    if (options_.debugAsyncCompute) asyncProbe_ = std::make_unique<AsyncComputeProbe>(*context_, *pipelines_);
    if (options_.debugGpuCost > 0) {
        knownCost_ = std::make_unique<KnownCostPass>(*context_, *pipelines_, options_.debugGpuCost);
    }
    if (options_.graphScenario) {
        rg::ScenarioParams params;
        params.width        = options_.scenarioWidth;
        params.height       = options_.scenarioHeight;
        params.work         = options_.scenarioWork;
        params.wideHdr      = options_.scenarioWide;
        params.asyncCompute = options_.scenarioAsync;
        params.remat        = options_.graphRemat;
        params.views        = options_.scenarioViews;
        params.rematIterations = options_.graphRematCost;
        scenario_ = std::make_unique<ScenarioPasses>(*context_, *pipelines_, *options_.graphScenario, params);
    }
    if (options_.gpuTiming) {
        timestamps_ = std::make_unique<GpuTimestamps>(*context_, *pipelines_);
        graphExecutor_->setTimestamps(timestamps_.get());
    }
    if (options_.gpuCapture) {
        gpuCapture_ = std::make_unique<GpuCapture>(*context_, options_.gpuCaptureDir, options_.gpuCaptureMax);
    }
    overlays_ = std::make_unique<DebugOverlays>(*context_, *pipelines_);

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
    imguiRenderer_ = std::make_unique<ImGuiRenderer>(*context_, *pipelines_);
    pressure_      = std::make_unique<MemoryPressureMonitor>();
    if (!options_.capturePath.empty()) {
        capture_ = std::make_unique<FrameCapture>(*context_);
    }

    // F3.4: the harvest records the whole forward variant table, not only the
    // variants the benches of this run happen to use.
    if (pipelines_->harvesting()) renderer_->requestAllVariants();
    // Every pipeline requested so far must be usable before the first frame
    // (flexible fallbacks count); with a complete archive nothing compiles.
    const Clock::time_point pipelinesStart = Clock::now();
    pipelines_->waitAllReady();
    startupPipelinesMs_ = toMs(Clock::now() - pipelinesStart);
    pipelines_->startupDone();
    LOG_INFO("Startup pipelines ready in %.1f ms: %s", static_cast<double>(startupPipelinesMs_),
             pipe::formatPipelineStats(pipelines_->stats()).c_str());

    // F3.6: shader hot reload in Debug builds -- the source shaders by
    // default in interactive runs, or --shader-dir.
#ifndef NDEBUG
    if (!options_.shaderDir.empty() || !options_.benchmark()) {
        reloader_ = std::make_unique<ShaderReloader>(
            context_->device(), options_.shaderDir.empty() ? std::string(PHOSPHOR_SHADER_SOURCE_DIR) : options_.shaderDir);
    }
#else
    if (!options_.shaderDir.empty()) LOG_WARN("--shader-dir: shader hot reload is available in Debug builds only");
#endif

    switchTestBench(options_.bench ? static_cast<TestBenchType>(*options_.bench) : TestBenchType::TorusDemo);
}

std::string Engine::pipelineArchivePath() const {
    if (options_.noPipelineArchive || !options_.harvestPipelinesPath.empty()) return {};
    // F4.3, measured: with an MTL4Archive loaded, startCapture fails with
    // "Capturing Metal 4 Device is not supported"; captures run without it.
    if (options_.gpuCapture) {
        LOG_INFO("Pipeline archive disabled: GPU capture (--gpu-capture*) is incompatible with MTL4Archive");
        return {};
    }
    if (!options_.pipelineArchivePath.empty()) return options_.pipelineArchivePath;
    // Default: built by the phosphor_archive target next to the metallib.
    const std::string path = shaderPath("phosphor-archive.metallib");
    std::error_code ec;
    return std::filesystem::exists(path, ec) ? path : std::string();
}

bool Engine::measuring() const {
    return options_.benchmark() && presentedFrames_ >= options_.warmup;
}

Engine::~Engine() {
    if (context_) context_->waitIdle();

    if (activeBench_) {
        activeBench_->teardown(*ecs_, *gpuScene_);
        activeBench_.reset();
    }

    reloader_.reset();
    pressure_.reset();
    graphExecutor_.reset();
    timestamps_.reset();
    gpuCapture_.reset();
    graphDebug_.reset();
    asyncProbe_.reset();
    knownCost_.reset();
    scenario_.reset();
    overlays_.reset();
    capture_.reset();
    imguiRenderer_.reset();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();

    textures_.reset();
    renderer_.reset();
    pipelines_.reset();
    context_.reset();

    if (metalView_) SDL_Metal_DestroyView(static_cast<SDL_MetalView>(metalView_));
    if (window_) SDL_DestroyWindow(window_);
    SDL_Quit();
}

void Engine::run() {
    if (options_.transientTest) {
        const TransientAliasResult r = runTransientAliasTest(*context_);
        std::printf("TRANSIENT heap %.2f MiB, textures at +%llu | aliased buffers %s | aliased textures %s | "
                    "memory shared %s | %s\n",
                    static_cast<double>(r.heapSize) / (1 << 20), static_cast<unsigned long long>(r.textureOffset),
                    r.buffersOk ? "ok" : "WRONG", r.texturesOk ? "ok" : "WRONG", r.memoryShared ? "yes" : "NO",
                    r.passed ? "PASS" : "FAIL");
        std::fflush(stdout);
        exitCode_ = r.passed ? 0 : 1;
        return;
    }
    if (options_.memoryStress > 0) {
        const u32 warmup = std::min(1000u, options_.memoryStress / 2);
        const MemoryStressResult r = runMemoryStress(*context_, options_.memoryStress, warmup);
        const auto mib = [](u64 b) { return static_cast<double>(b) / (1 << 20); };
        std::printf("STRESS %u cycles | device MiB: baseline %.2f, after warm-up %.2f, final %.2f, "
                    "after trim %.2f | peak heaps %u | counts %s | %s\n",
                    options_.memoryStress, mib(r.baselineBytes), mib(r.warmBytes), mib(r.finalBytes),
                    mib(r.trimmedBytes), r.heapsPeak, r.countsRestored ? "restored" : "LEAKED",
                    r.passed ? "PASS (back to baseline)" : "FAIL (memory not returned)");
        std::fflush(stdout);
        exitCode_ = r.passed ? 0 : 1;
        return;
    }
    LOG_INFO("Entering main loop");
    if (options_.benchmark()) {
        LOG_INFO("Benchmark: %u warm-up + %u measured frames, vsync %s, UI %s", options_.warmup, options_.frames,
                 options_.vsync ? "on" : "off", options_.ui ? "on" : "off");
        if (options_.warmup == 0) {
            measureFirstFrame_ = context_->frameIndex();
            measureLastFrame_  = measureFirstFrame_ + options_.frames - 1;
            context_->beginGpuTimeCapture(options_.frames);
            allocationsAtStart_ = context_->memory().allocationCount();
            heapUsage(heapBlocksAtStart_, heapBytesAtStart_);
        }
    }
    while (running_) {
        const Clock::time_point start = Clock::now();
        frameFlags_ = 0;
        {
            const pipe::PipelineStats& ps = pipelines_->stats();
            requestsBeforeFrame_    = ps.requests;
            rtCompileMsBeforeFrame_ = ps.renderThreadCompileMs;
            requestMsBeforeFrame_   = pipelines_->requestMs();
        }
        {
            // SDL's Cocoa event pump autoreleases AppKit objects: without a
            // pool per iteration they pile up for the whole run (measured
            // with malloc_history: +4.3k live blocks in 4 minutes).
            NS::AutoreleasePool* eventPool = NS::AutoreleasePool::alloc()->init();
            PH_ZONE("Events");
            processEvents();
            eventPool->release();
        }
        if (!running_) break;

        if (pendingBench_) {
            switchTestBench(*pendingBench_);
            pendingBench_.reset();
        }
        handleMemoryPressure();

        timer_->tick();
        // Fixed step: deterministic animation for captures; timings stay real.
        const float simDt = options_.fixedTimestep ? 1.0f / 60.0f : timer_->getDeltaTime();
        const bool presented = frame(simDt);
        input_->resetFrameState();
        if (presented && options_.simulatePressure) {
            ++simulatedFrames_;
            if (simulatedFrames_ == 10) pressure_->simulate(MemoryPressureMonitor::Level::Warning);
            if (simulatedFrames_ == 20) pressure_->simulate(MemoryPressureMonitor::Level::Critical);
        }
        if (presented && options_.resizeEvery > 0 && ++resizeFrames_ % options_.resizeEvery == 0) {
            // Alternate between two window sizes: the drawable size changes,
            // so the render graph is recompiled (resize test).
            resizeToggle_ = !resizeToggle_;
            SDL_SetWindowSize(window_, resizeToggle_ ? 1280 : 1600, resizeToggle_ ? 720 : 900);
        }
        if (presented && options_.switchEvery > 0 && ++framesOnBench_ >= options_.switchEvery && !pendingBench_) {
            // Same path as the 1-7 hotkeys.
            pendingBench_ = static_cast<TestBenchType>((static_cast<int>(currentBench_) + 1) % testBenchCount());
        }
        if (presented && options_.benchmark()) {
            recordBenchmarkFrame(timer_->getDeltaTime(), toMs(Clock::now() - start - frameWait_), toMs(frameWait_));
        }
        if (presented && options_.debugCompileStorm && measuring() && samples_.size() == 60) {
            // F3.1 spike: 42 background compiles while frames are measured.
            LOG_INFO("Compile storm: requesting every forward variant (salt %u)", options_.pipelineSalt);
            renderer_->requestAllVariants();
        }
    }
    context_->waitIdle();
    if (pipelines_->harvesting()) {
        pipelines_->waitAllFinal();
        if (!pipelines_->writeHarvest()) exitCode_ = 1;
    }
    if (options_.benchmark()) {
        finishBenchmark();
    }
    // stdout, not the log: scripts collect this line (F3.5 miss rate).
    std::printf("%s\n", pipe::formatPipelineStats(pipelines_->stats()).c_str());
    std::fflush(stdout);
    if (graphDebug_ && !graphDebug_->finish()) exitCode_ = 1;
    if (asyncProbe_ && !asyncProbe_->finish()) exitCode_ = 1;
    if (capture_) {
        if (captured_) {
            capture_->writePng(options_.capturePath);
        } else {
            LOG_ERROR("No frame captured");
        }
    }
    if (!options_.debugHotReloadPath.empty() && !checkHotReloadCapture()) exitCode_ = 1;
}

void Engine::pollShaderReload() {
    // --debug-hot-reload: after a few frames, the probe library goes through
    // exactly the path of a watched change.
    if (!options_.debugHotReloadPath.empty() && !hotReloadRequested_ && presentedFrames_ >= 5) {
        hotReloadRequested_ = true;
        NS::Error* error = nullptr;
        MTL::Library* probe = context_->device()->newLibrary(
            NS::String::string(options_.debugHotReloadPath.c_str(), NS::UTF8StringEncoding), &error);
        if (!probe) {
            LOG_ERROR("--debug-hot-reload: cannot load %s: %s", options_.debugHotReloadPath.c_str(),
                      error ? error->localizedDescription()->utf8String() : "unknown error");
            exitCode_ = 1;
            return;
        }
        pipelines_->reload(probe);
        probe->release();
        return;
    }
    if (!reloader_ || pipelines_->reloadPending()) return;
    if (MTL::Library* library = reloader_->takeLibrary()) {
        pipelines_->reload(library);
        library->release();
    }
}

bool Engine::checkHotReloadCapture() const {
    // The probe forward pass writes opaque magenta; with --no-ui the frame must
    // hold exactly two colours: the clear colour and magenta.  Any pixel of
    // another colour was drawn by a pipeline that was not swapped.
    const u32 reloads = pipelines_->stats().reloads;
    if (!capture_ || !captured_ || !capture_->pixels()) {
        std::printf("HOT-RELOAD FAIL: needs --capture\n");
        return false;
    }
    const u8* p = capture_->pixels();
    const size_t count = static_cast<size_t>(capture_->width()) * capture_->height();
    const u32 magenta = 0xFFFF00FFu; // BGRA bytes FF 00 FF FF read as little-endian u32
    u64 probe = 0;
    u32 others[4] = {};
    u32 otherCount = 0;
    u64 otherPixels = 0;
    for (size_t i = 0; i < count; ++i) {
        u32 c = 0;
        std::memcpy(&c, p + i * 4, 4);
        c |= 0xFF000000u;
        if (c == magenta) {
            ++probe;
            continue;
        }
        ++otherPixels;
        bool known = false;
        for (u32 k = 0; k < otherCount; ++k) known |= others[k] == c;
        if (!known && otherCount < 4) others[otherCount++] = c;
        if (!known && otherCount == 4) otherCount = 4; // saturated: > 1 anyway
    }
    const bool pass = reloads == 1 && probe > 0 && otherCount == 1;
    std::printf("HOT-RELOAD reloads %u | probe pixels %llu | other pixels %llu in %u%s colours | %s\n", reloads,
                static_cast<unsigned long long>(probe), static_cast<unsigned long long>(otherPixels), otherCount,
                otherCount == 4 ? "+" : "", pass ? "PASS" : "FAIL");
    std::fflush(stdout);
    return pass;
}

void Engine::recordBenchmarkFrame(float dt, float cpuMs, float waitMs) {
    ++presentedFrames_;
    if (presentedFrames_ == options_.warmup) {
        // GPU times are recorded from the next submitted frame on.
        measureFirstFrame_ = context_->frameIndex();
        measureLastFrame_  = measureFirstFrame_ + options_.frames - 1;
        context_->beginGpuTimeCapture(options_.frames);
        samples_.reserve(options_.frames);
        trace_.reserve(options_.frames, options_.frames / std::max(options_.switchEvery, 1u) + 2);
        allocationsAtStart_ = context_->memory().allocationCount();
        heapUsage(heapBlocksAtStart_, heapBytesAtStart_);
    } else if (presentedFrames_ > options_.warmup) {
        FrameRecord record;
        record.index   = static_cast<u32>(samples_.size());
        record.bench   = static_cast<u32>(currentBench_);
        record.flags   = frameFlags_;
        record.frameMs = dt * 1000.0f;
        record.cpuMs   = cpuMs;
        record.waitMs  = waitMs;
        trace_.addFrame(record);
        if (pendingSwitch_) {
            // The switch frame is done: add what it cost on the pipeline side.
            const pipe::PipelineStats& ps = pipelines_->stats();
            pendingSwitch_->pipelinesRequested    = ps.requests - requestsBeforeFrame_;
            pendingSwitch_->pipelineRequestMs     = static_cast<float>(pipelines_->requestMs() - requestMsBeforeFrame_);
            pendingSwitch_->renderThreadCompileMs = static_cast<float>(ps.renderThreadCompileMs - rtCompileMsBeforeFrame_);
            trace_.addSwitch(*pendingSwitch_);
            pendingSwitch_.reset();
        }
        samples_.push_back({dt * 1000.0f, cpuMs, 0.0f, waitMs});
        if (samples_.size() == options_.frames) running_ = false;
    }
}

void Engine::finishBenchmark() {
    // Sample the CPU heap before the report's own allocations.
    u64 heapBlocks = 0, heapBytes = 0;
    heapUsage(heapBlocks, heapBytes);
    // F4.1: the last frames' timestamps (the GPU is idle): resolve the slots
    // in frame order.
    if (timestamps_) {
        for (;;) {
            u32 next = ~0u;
            for (u32 s = 0; s < METAL_FRAMES_IN_FLIGHT; ++s) {
                const u64 f = timestamps_->pendingFrame(s);
                if (f != ~0ull && (next == ~0u || f < timestamps_->pendingFrame(next))) next = s;
            }
            if (next == ~0u) break;
            onFrameTimes(timestamps_->drain(next));
        }
        if (passMeasureStarted_) passTimings_.endMeasure();
    }
    const std::vector<float> gpu = context_->endGpuTimeCapture();
    for (size_t i = 0; i < samples_.size() && i < gpu.size(); ++i) {
        samples_[i].gpuMs = gpu[i];
    }
    trace_.setGpuTimes(gpu);

    BenchReport report;
    report.bench  = activeBench_ ? activeBench_->getName() : "";
    report.device = context_->gpuName();
    report.width  = context_->width();
    report.height = context_->height();
    report.vsync  = settings_.vsync;
    report.ui     = options_.ui;
    report.gpuAllocations = context_->memory().allocationCount() - allocationsAtStart_;
    report.cpuHeapBlocksDelta = static_cast<i64>(heapBlocks) - static_cast<i64>(heapBlocksAtStart_);
    report.cpuHeapBytesDelta  = static_cast<i64>(heapBytes) - static_cast<i64>(heapBytesAtStart_);
    summarizeSamples(samples_, report);
    report.pipelinesJson = pipe::pipelineStatsJson(pipelines_->stats());
    report.gpuTiming        = timestamps_ != nullptr && timestamps_->enabled();
    report.gpuTimingUnfused = options_.gpuTimingUnfused;
    if (report.gpuTiming) passTimings_.summarize(report.passes, report.gpuPassSumMs, report.gpuFrameSpanMs);
    // OPT-0.4: work of the passes the cost model can price (CPU data of the
    // last frame; the benches' scenes do not change their draw lists).
    for (PassReport& unit : report.passes) {
        for (const std::string& pass : unit.passes) {
            PassWork w;
            w.pass = pass;
            if (pass == "Forward") {
                const ForwardWork f = forwardPassWork(*gpuScene_, frameScene_, report.width, report.height);
                w.draws     = f.draws;
                w.instances = f.instances;
                w.indices   = f.indices;
                w.vertices  = f.vertices;
                w.pixels    = f.pixels;
                w.lights    = f.lights;
            } else if (pass == "ImGui overlay" && ImGui::GetDrawData() != nullptr) {
                const ImDrawData* d = ImGui::GetDrawData();
                for (int i = 0; i < d->CmdListsCount; ++i) w.draws += static_cast<u64>(d->CmdLists[i]->CmdBuffer.Size);
                w.instances = w.draws;
                w.indices   = static_cast<u64>(d->TotalIdxCount);
                w.vertices  = static_cast<u64>(d->TotalVtxCount);
                w.pixels    = u64(report.width) * report.height;
            } else {
                continue;
            }
            unit.work.push_back(w);
        }
    }

    if (ignoredInputEvents_ > 0) {
        LOG_INFO("Benchmark: ignored %u keyboard/mouse events", ignoredInputEvents_);
    }
    // stdout, not the log: scripts collect this line.
    std::printf("BENCH %s\n", formatReportLine(report).c_str());
    if (!trace_.switches().empty() || !options_.frameTracePath.empty()) {
        std::printf("%s\n", formatHitchReport(analyzeHitches(trace_)).c_str());
    }
    std::fflush(stdout);
    if (!options_.frameTracePath.empty()) {
        std::ofstream out(options_.frameTracePath);
        out << traceToCsv(trace_);
        if (!out) LOG_ERROR("Failed to write %s", options_.frameTracePath.c_str());
    }
    if (!options_.reportPath.empty()) {
        std::ofstream out(options_.reportPath);
        out << reportToJson(report);
        if (!out) LOG_ERROR("Failed to write %s", options_.reportPath.c_str());
    }
}

void Engine::injectSyntheticInput() {
    // As if someone typed W/D/F12 and dragged the mouse over the focused window.
    // F12 exercises the GPU capture key (F4.3) when --gpu-capture is on.
    for (const SDL_Scancode key : {SDL_SCANCODE_W, SDL_SCANCODE_D, SDL_SCANCODE_F12}) {
        SDL_Event e{};
        e.type = SDL_EVENT_KEY_DOWN;
        e.key.windowID = SDL_GetWindowID(window_);
        e.key.scancode = key;
        e.key.down = true;
        SDL_PushEvent(&e);
    }
    SDL_Event motion{};
    motion.type = SDL_EVENT_MOUSE_MOTION;
    motion.motion.windowID = SDL_GetWindowID(window_);
    motion.motion.state = SDL_BUTTON_LMASK | SDL_BUTTON_RMASK;
    motion.motion.xrel = 12.0f;
    motion.motion.yrel = 4.0f;
    SDL_PushEvent(&motion);
    SDL_Event wheel{};
    wheel.type = SDL_EVENT_MOUSE_WHEEL;
    wheel.wheel.windowID = SDL_GetWindowID(window_);
    wheel.wheel.y = 1.0f;
    SDL_PushEvent(&wheel);
}

void Engine::processEvents() {
    if (options_.injectInput) injectSyntheticInput();
    const ImGuiIO& io = ImGui::GetIO();
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        const bool keyboardEvent = event.type == SDL_EVENT_KEY_DOWN || event.type == SDL_EVENT_KEY_UP ||
                                   event.type == SDL_EVENT_TEXT_INPUT;
        const bool mouseEvent = event.type == SDL_EVENT_MOUSE_MOTION || event.type == SDL_EVENT_MOUSE_WHEEL ||
                                event.type == SDL_EVENT_MOUSE_BUTTON_DOWN || event.type == SDL_EVENT_MOUSE_BUTTON_UP;
        // Benchmarks and captures must not depend on whoever types or moves
        // the mouse while the window has focus (it takes focus at launch).
        if (options_.benchmark() && (keyboardEvent || mouseEvent)) {
            ++ignoredInputEvents_;
            continue;
        }
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
    if (input_->isKeyPressed(SDL_SCANCODE_F12)) {
        if (gpuCapture_) {
            gpuCapture_->request("key");
        } else {
            LOG_WARN("F12: GPU capture needs --gpu-capture (the capture layer is inserted at launch)");
        }
    }
}

void Engine::switchTestBench(TestBenchType type) {
    PH_ZONE("Bench switch");
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    // F3: phases of the switch, recorded while measuring (hitch analysis).
    const Clock::time_point t0 = Clock::now();
    const TestBenchType from = currentBench_;
    const bool isSwitch = activeBench_ != nullptr; // not the initial load
    context_->waitIdle();
    const Clock::time_point t1 = Clock::now();

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
    const Clock::time_point t2 = Clock::now();

    textures_->flushUploads();
    const Clock::time_point t3 = Clock::now();
    renderer_->syncGeometry(*gpuScene_);
    const Clock::time_point t4 = Clock::now();
    // The previous bench's resources are unused now: free their heap ranges
    // and give back heaps that became empty.
    context_->collectGarbage();
    context_->memory().trimEmptyHeaps();
    context_->commitResidency();
    const Clock::time_point t5 = Clock::now();
    logMemory();
    aimCamera(activeBench_->getDefaultCamera());
    frameFlags_ |= FrameBenchSwitch;
    if (isSwitch && measuring()) {
        SwitchRecord sw;
        sw.frame            = static_cast<u32>(samples_.size()); // the frame about to be produced
        sw.fromBench        = static_cast<u32>(from);
        sw.toBench          = static_cast<u32>(type);
        sw.waitIdleMs       = toMs(t1 - t0);
        sw.setupMs          = toMs(t2 - t1);
        sw.textureUploadMs  = toMs(t3 - t2);
        sw.geometryUploadMs = toMs(t4 - t3);
        sw.gcMs             = toMs(t5 - t4);
        sw.totalMs          = toMs(Clock::now() - t0);
        pendingSwitch_      = sw;
    }
    pool->release();
}

void Engine::logMemory() const {
    const GpuMemory& memory = context_->memory();
    u64 heapBytes = 0, heapUsed = 0;
    float fragmentation = 0.0f;
    std::vector<GpuMemory::HeapStats> heaps;
    memory.heapStats(heaps);
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

void Engine::handleMemoryPressure() {
    MemoryPressureMonitor::Level level;
    if (!pressure_->poll(level)) return;
    if (level == MemoryPressureMonitor::Level::Normal) {
        LOG_INFO("Memory pressure back to normal");
        return;
    }
    // Between frames: nothing recorded references pending releases.  A full
    // GPU wait is acceptable here; the system is short of memory.
    const u64 before = context_->device()->currentAllocatedSize();
    context_->collectGarbage();
    const bool critical = level == MemoryPressureMonitor::Level::Critical;
    const u64 trimmed = context_->memory().trimEmptyHeaps(/*keepSpare*/ !critical);
    context_->commitResidency(); // give the memory back now, not at the next frame
    const u64 after = context_->device()->currentAllocatedSize();
    LOG_WARN("Memory pressure %s: trimmed %.1f MiB of heaps, device allocation %.1f -> %.1f MiB",
             MemoryPressureMonitor::name(level), static_cast<double>(trimmed) / (1 << 20),
             static_cast<double>(before) / (1 << 20), static_cast<double>(after) / (1 << 20));
}

void Engine::fillMemoryInfo() {
    const GpuMemory& memory = context_->memory();
    const MemoryBudget& budget = context_->budget();
    MemoryPanelInfo& m = memoryInfo_;
    m.tier            = tierName(budget.tier());
    m.workingSet      = budget.workingSet();
    m.engineLimit     = budget.engineLimit();
    m.deviceAllocated = context_->device()->currentAllocatedSize();
    m.gpuAllocations  = memory.allocationCount();
    for (u32 c = 0; c < MEMORY_CATEGORY_COUNT; ++c) {
        const auto category = static_cast<MemoryCategory>(c);
        const GpuMemory::CategoryStats& s = memory.stats(category);
        m.categories[c] = {s.bytes, s.count, budget.limit(category), budget.level(category, s.bytes)};
    }
    memory.heapStats(heapStatsScratch_);
    m.heaps.clear();
    for (const auto& h : heapStatsScratch_) {
        m.heaps.push_back({h.size, h.tlsf.usedBytes, h.tlsf.allocationCount, h.tlsf.fragmentation(),
                           h.cls == ResidencyClass::Streaming});
    }
    const auto ring = [](const char* name, const UploadRing& r) {
        const LinearRing::Stats s = r.stats();
        return MemoryPanelInfo::Ring{name, s.capacity, s.inFlightBytes, s.peakFrameBytes, s.overflows};
    };
    m.rings = {ring("Frame uploads", context_->frameUploads()), ring("Staging", context_->staging())};
    const auto set = [&](const char* name, ResidencyClass cls) {
        const ResidencyManager::Stats s = context_->residency().stats(cls);
        return MemoryPanelInfo::ResidencySet{name, s.allocations, s.bytes, s.commits};
    };
    m.residency = {set("static", ResidencyClass::Static), set("streaming", ResidencyClass::Streaming)};
    m.pressure       = MemoryPressureMonitor::name(pressure_->level());
    m.pressureEvents = pressure_->eventCount();
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

    // F3: compilations finished since the last frame become visible now,
    // never while a frame is being encoded.
    pollShaderReload();
    if (pipelines_->beginFrame() > 0) frameFlags_ |= FramePipelineSwap;

    // --- Simulation -----------------------------------------------------------
    PH_ZONE("Frame");
    if (orbitMode_) {
        camera_->updateOrbit(*input_, dt);
    } else {
        camera_->updateFPS(*input_, dt);
    }
    camera_->setAspect(static_cast<float>(context_->width()) / static_cast<float>(std::max(context_->height(), 1u)));
    camera_->updateMatrices();

    {
        PH_ZONE("Simulation");
        activeBench_->update(dt, *ecs_);
    }
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

    MTL::Texture* target = frame.drawable->texture();
    const u32 width  = static_cast<u32>(target->width());
    const u32 height = static_cast<u32>(target->height());

    // Capture the last frame of a run (or the first frame when interactive).
    const bool lastFrame = !options_.benchmark() || presentedFrames_ + 1 == options_.warmup + options_.frames;
    captureThisFrame_ = capture_ && lastFrame && !captured_;

    const GraphKey key{width, height, options_.ui, capture_ != nullptr, options_.debugSplitEncoding,
                       options_.debugAsyncCompute, overlayMode_};
    if (!(key == graphKey_) || !graphExecutor_->valid()) {
        frameFlags_ |= FrameGraphCompile;
        if (key.width != graphKey_.width || key.height != graphKey_.height) frameFlags_ |= FrameResize;
        graphKey_ = key;
        buildFrameGraph(width, height);
    }

    // Captures are deterministic: the captured frame draws with the final
    // pipelines (fallbacks: --debug-pipeline-fallback, --debug-flexible-pipelines).
    if (captureThisFrame_ && !options_.debugFlexiblePipelines) pipelines_->waitAllFinal();

    // F4.1: the slot's previous frame has completed: its GPU times.
    if (timestamps_) onFrameTimes(timestamps_->beginFrame(frame.slot, frame.index));
    // F4.3: captures cover whole frames, from before the encoding to the commit.
    if (gpuCapture_) {
        if (options_.gpuCaptureFrame && presentedFrames_ == *options_.gpuCaptureFrame) gpuCapture_->request("frame");
        gpuCapture_->beginFrame(frame.index);
    }

    if (graphDebug_) graphDebug_->beginFrame(frame.slot);
    if (asyncProbe_) asyncProbe_->beginFrame(frame.slot);
    renderer_->prepareFrame(*gpuScene_, frameScene_, constants, textures_->tableAddress(), width, height);
    overlays_->prepareFrame(overlayMode_, constants.lightCount, width, height);
    if (renderer_->usingFallback()) frameFlags_ |= FrameFallbackDraw;
    if (options_.ui) {
        PH_ZONE("UI");
        drawUi();
    }

    graphExecutor_->bindTexture(drawableRef_, target);
    if (capture_) graphExecutor_->bindBuffer(captureRef_, capture_->readback());
    if (graphDebug_) graphDebug_->bind(*graphExecutor_, frame.slot);
    if (scenario_) scenario_->bind(*graphExecutor_, frame.index);
    if (asyncProbe_) asyncProbe_->bind(*graphExecutor_, frame.slot);
    {
        PH_ZONE("Graph execute");
        graphExecutor_->execute(frame);
    }
    if (graphDebug_) graphDebug_->frameEncoded(frame.slot, frame.index);
    if (asyncProbe_) asyncProbe_->frameEncoded(frame.slot, frame.index);
    if (captureThisFrame_) captured_ = true;

    {
        PH_ZONE("Submit");
        context_->submitFrame(frame);
    }
    if (gpuCapture_) gpuCapture_->endFrame();
    // F4.1: no overlap between consecutive frames on the GPU.
    if (options_.gpuTimingSerial) context_->waitIdle();
    PH_FRAME_MARK;
    if (!firstFrameLogged_) {
        firstFrameLogged_ = true;
        // stdout: startup measurements (F3 cold start with/without archive).
        std::printf("STARTUP first frame submitted %.1f ms after launch | startup pipelines %.1f ms | %s\n",
                    static_cast<double>(toMs(Clock::now() - launch_)), static_cast<double>(startupPipelinesMs_),
                    pipe::formatPipelineStats(pipelines_->stats()).c_str());
        std::fflush(stdout);
    }
    pool->release();
    return true;
}

void Engine::onFrameTimes(const GpuTimestamps::Resolved& r) {
    if (!r.valid) return;
    tracyGpu_.emitFrame(r.startTicks, r.endTicks, r.units);
    // Benchmark: only the measured frames enter the report; the window and
    // Tracy see every frame.
    const bool measured = measureFirstFrame_ != ~0ull && r.frame >= measureFirstFrame_ && r.frame <= measureLastFrame_;
    if (measured && !passMeasureStarted_) {
        passTimings_.beginMeasure(options_.frames);
        passMeasureStarted_ = true;
    }
    passTimings_.addFrame(r.frame, r.ms, r.unitValid, r.spanMs);
    if (passMeasureStarted_ && r.frame == measureLastFrame_) passTimings_.endMeasure();
    if (gpuCapture_ && options_.gpuCaptureOverMs > 0.0f && r.sumMs > options_.gpuCaptureOverMs &&
        gpuCapture_->available()) {
        // The slow frame is 3 frames old: the capture takes the next frame.
        LOG_INFO("GPU capture: frame %llu took %.3f ms of GPU (> %.3f): capturing the next frame",
                 static_cast<unsigned long long>(r.frame), static_cast<double>(r.sumMs),
                 static_cast<double>(options_.gpuCaptureOverMs));
        gpuCapture_->request("over");
    }
}

void Engine::drawUi() {
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
    {
        const DebugOverlays::Legend legend = DebugOverlays::legend(overlayMode_);
        settings_.overlayQuantity = legend.quantity;
        settings_.overlayMax      = legend.maxValue;
        settings_.overlayLog      = legend.logScale;
        settings_.overlayNote     = legend.note;
    }
    UIPanels::drawRenderPanel(settings_);
    overlayMode_ = static_cast<OverlayMode>(settings_.overlay);
    fillMemoryInfo();
    UIPanels::drawMemoryPanel(memoryInfo_);
    PipelinePanelInfo pipelineInfo;
    pipelineInfo.stats    = &pipelines_->stats();
    pipelineInfo.archive  = pipelines_->archiveStatus().c_str();
    pipelineInfo.workers  = pipelines_->workerCount();
    pipelineInfo.entries  = pipelines_->entryCount();
    pipelineInfo.fallback = renderer_->usingFallback();
    UIPanels::drawPipelinePanel(pipelineInfo);
    UIPanels::drawPassTimingsPanel(timestamps_ ? &passTimings_ : nullptr, context_->lastGpuMs(),
                                   options_.gpuTimingUnfused);
    if (overlayMode_ == OverlayMode::Timings) UIPanels::drawTimingsOverlay(timestamps_ ? &passTimings_ : nullptr);

    ImGui::Render();
}

void Engine::buildFrameGraph(u32 width, u32 height) {
    PH_ZONE("Render graph build");
    using namespace rg;
    frameGraph_.reset();

    const TextureDesc screen{Format::BGRA8Srgb, width, height};
    // The drawable: undefined at frame start, presented after the graph; a
    // different texture every frame (the drawable wait orders its reuse).
    drawableRef_ = frameGraph_.importTexture("Drawable", screen, ImportOutput | ImportPerFrame);
    TextureRef color = drawableRef_;

    if (scenario_) {
        // OPT-1: a graph scenario of synthetic passes replaces the scene; its
        // present pass writes the drawable.
        scenario_->build(frameGraph_, drawableRef_);
        color = {drawableRef_.resource, frameGraph_.resources()[drawableRef_.resource].versions - 1};
    }

    // F2.6: seed -> reduce (async queue) -> consume, declared before Forward so
    // the async pass can overlap it.
    if (asyncProbe_) asyncProbe_->addProducers(frameGraph_);
    if (knownCost_) knownCost_->addToGraph(frameGraph_);

    if (!scenario_) frameGraph_.addPass(
        "Forward", PassType::Raster,
        [&](PassBuilder& b) {
            ClearValue clear;
            clear.color[0] = 0.02f;
            clear.color[1] = 0.025f;
            clear.color[2] = 0.035f;
            clear.color[3] = 1.0f;
            clear.depth    = 0.0f; // reverse-Z: far = 0
            TextureRef depth = b.createTexture("Depth", {Format::Depth32Float, width, height});
            color = b.writeColor(color, 0, LoadIntent::Clear, clear);
            b.writeDepth(depth, LoadIntent::Clear, clear);
            b.setHints(HintGeometryHeavy);
            b.setProfileShaders("forward_vs,forward_fs");
            // F4.1 negative control: the known-cost pass runs before the
            // forward pass, never beside it.
            if (knownCost_) b.read(knownCost_->output(), Usage::ShaderRead, StageVertex);
            // F2.5 check: the draws are recorded by 4 threads into a render
            // pass suspended/resumed across command buffers.
            if (options_.debugSplitEncoding) b.setParallelChunks(4);
        },
        [this](PassContext& ctx) {
            renderer_->encode(static_cast<MTL4::RenderCommandEncoder*>(ctx.encoder()), ctx.chunk(), ctx.chunkCount());
        });

    if (!scenario_) color = overlays_->addToGraph(frameGraph_, color, width, height, overlayMode_, *renderer_);

    if (options_.ui) {
        frameGraph_.addPass(
            "ImGui overlay", PassType::Raster,
            [&](PassBuilder& b) {
                color = b.writeColor(color, 0, LoadIntent::Preserve);
                b.setProfileShaders("imgui_vs,imgui_fs");
            },
            [this](PassContext& ctx) {
                imguiRenderer_->render(static_cast<MTL4::RenderCommandEncoder*>(ctx.encoder()), ImGui::GetDrawData());
            });
    }

    if (asyncProbe_) asyncProbe_->addConsumer(frameGraph_);
    if (graphDebug_) graphDebug_->addToGraph(frameGraph_);

    if (capture_) {
        capture_->prepare(width, height);
        captureRef_ = frameGraph_.importBuffer("Capture readback", {capture_->readbackSize()}, ImportOutput);
        frameGraph_.addPass(
            "Frame capture", PassType::Blit,
            [&](PassBuilder& b) {
                b.read(color, Usage::CopySrc, StageBlit);
                b.write(captureRef_, Usage::CopyDst, StageBlit);
                b.setSideEffect();
            },
            [this](PassContext& ctx) {
                // In the graph for the whole run (no recompilation on the
                // capture frame); copies only on the frame being captured.
                if (!captureThisFrame_) return;
                capture_->encode(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),
                                 static_cast<MTL::Texture*>(ctx.texture(drawableRef_)));
            });
    }

    // F4.1 attribution mode: every raster pass in its own render pass.
    CompileOptions compileOptions;
    compileOptions.fuseRasterPasses = !options_.gpuTimingUnfused;
    compileOptions.alias            = !options_.graphNoAlias;
    // OPT-1 spike: an execution order given by pass names.
    for (const std::string& name : options_.graphOrder) {
        const auto& passes = frameGraph_.passes();
        const auto it = std::find_if(passes.begin(), passes.end(), [&](const PassNode& p) { return p.name == name; });
        if (it == passes.end()) throw std::runtime_error("--graph-order: no pass named '" + name + "'");
        compileOptions.order.push_back(static_cast<u32>(it - passes.begin()));
    }
    if (!compileOptions.order.empty()) {
        // Passes the list does not name (the engine's UI and capture passes)
        // follow in declaration order.
        for (u32 p = 0; p < frameGraph_.passes().size(); ++p) {
            if (std::find(compileOptions.order.begin(), compileOptions.order.end(), p) == compileOptions.order.end()) {
                compileOptions.order.push_back(p);
            }
        }
    }
    if (!graphExecutor_->compile(frameGraph_, compileOptions)) {
        throw std::runtime_error("Failed to compile the frame graph");
    }
    if (timestamps_) {
        passNames_.clear();
        passShaders_.clear();
        for (const PassNode& pass : frameGraph_.passes()) {
            passNames_.push_back(pass.name);
            std::string shaders;
            for (const std::string& f : pass.profileShaders) shaders += (shaders.empty() ? "" : ",") + f;
            passShaders_.push_back(std::move(shaders));
        }
        passTimings_.configure(timestamps_->plan(), passNames_, passShaders_);
        tracyGpu_.configure(timestamps_->plan(), timestamps_->tickNs(), mach_absolute_time());
        // The measurement restarts with the new plan (resize/switch in a benchmark).
        passMeasureStarted_ = false;
    }
    if (graphDebug_) graphDebug_->onCompiled(frameGraph_, graphExecutor_->compiled());
    if (scenario_) scenario_->onCompiled(frameGraph_, graphExecutor_->compiled());
    if (asyncProbe_) asyncProbe_->onCompiled(frameGraph_, graphExecutor_->compiled());
    const BandwidthReport traffic = estimateBandwidth(frameGraph_, graphExecutor_->compiled());
    LOG_INFO("Render graph %ux%u: estimated DRAM traffic %.2f MiB/frame (read %.2f, write %.2f)", width, height,
             static_cast<double>(traffic.totalBytes()) / (1 << 20),
             static_cast<double>(traffic.totalReadBytes) / (1 << 20),
             static_cast<double>(traffic.totalWriteBytes) / (1 << 20));
    if (!options_.dumpGraphPath.empty()) {
        std::ofstream out(options_.dumpGraphPath);
        out << dumpGraphviz(frameGraph_, graphExecutor_->compiled());
        if (!out) {
            LOG_ERROR("Failed to write %s", options_.dumpGraphPath.c_str());
        } else {
            LOG_INFO("Render graph written to %s", options_.dumpGraphPath.c_str());
        }
    }
}

} // namespace phosphor
