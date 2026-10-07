#include <thread>
#include "app/engine.h"
#include <glm/gtc/type_ptr.hpp>
#include "platform/metal/temporal_worker.h"

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
#include "platform/metal/gpu_scene_check.h"
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
#include "platform/metal/mesh_renderer.h"
#include "platform/metal/visibility_renderer.h"
#include "platform/metal/post_processor.h"
#include "platform/metal/display_output.h"
#include "platform/metal/meshlet_check.h"
#include "platform/metal/acceleration_structures.h"
#include "platform/metal/rt_visibility_check.h"
#include "platform/metal/shadow_passes.h"
#include "platform/metal/direct_lighting_passes.h"
#include "platform/metal/gi_passes.h"
#include "platform/metal/reference_snapshot.h"
#include "platform/metal/linear_capture.h"
#include "platform/metal/reflection_passes.h"
#include "platform/metal/metalfx_denoise.h"
#include "platform/metal/atmosphere_passes.h"
#include "platform/metal/lighting_dispatch.h"
#include "renderer/rt_check.h"
#include "renderer/cull_reference.h"
#include "renderer/gpu_scene.h"
#include "renderer/scene_check.h"
#include "renderer/scene_store.h"
#include "renderer/transform_math.h"
#include "rendergraph/graph_dump.h"
#include "rendergraph/optimizer/plan.h"
#include "rendergraph/pass_context.h"
#include "scene/camera.h"
#include "scene/components.h"
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
#include <sstream>
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

// F5 self-check negative control (--debug-gpu-scene-corrupt touch): change a
// transform WITHOUT the ECS noticing (const access, then a cast), so the
// scene store misses it and SceneStore::verifyAgainstEcs must report it.
void corruptUntrackedTransform(const ECS& ecs) {
    const auto& meshes = ecs.getArray<MeshInstanceComponent>();
    const auto& transforms = ecs.getArray<TransformComponent>();
    for (const EntityID e : meshes.entities()) {
        if (!transforms.has(e)) continue;
        auto& t = const_cast<TransformComponent&>(transforms.get(e));
        t.worldMatrix[3][0] += 1.0f;
        return;
    }
}

std::string formatSceneLine(const SceneReport& s) {
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "SCENE gpu-driven %s | instances %u slots %u buckets %u materials %u commands %u | upload %.1f KiB/frame "
                  "(p99 %.1f) | records %.0f | visible %.0f | culled frustum %.0f distance %.0f size %.0f | draws %.0f | "
                  "cpu commands %.0f (min %.0f max %.0f) | structure changes %u | queue overflow %u",
                  s.mode.c_str(), s.instances, s.slots, s.buckets, s.materials, s.commands,
                  static_cast<double>(s.uploadBytes.mean) / 1024.0, static_cast<double>(s.uploadBytes.p99) / 1024.0,
                  static_cast<double>(s.deltaRecords.mean), static_cast<double>(s.visible.mean),
                  static_cast<double>(s.culledFrustum.mean), static_cast<double>(s.culledDistance.mean),
                  static_cast<double>(s.culledSize.mean), static_cast<double>(s.drawCommands.mean),
                  static_cast<double>(s.cpuCommands.mean), static_cast<double>(s.cpuCommands.min),
                  static_cast<double>(s.cpuCommands.max), s.structureChanges, s.queueOverflow);
    return buf;
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
                               SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY | SDL_WINDOW_METAL |
                                   (options_.offscreen ? SDL_WINDOW_HIDDEN : 0));
    if (!window_) {
        throw std::runtime_error(std::string("SDL_CreateWindow failed: ") + SDL_GetError());
    }
    metalView_ = SDL_Metal_CreateView(window_);
    auto* layer = static_cast<CA::MetalLayer*>(SDL_Metal_GetLayer(static_cast<SDL_MetalView>(metalView_)));
    if (!layer) {
        throw std::runtime_error("Failed to obtain CAMetalLayer from SDL");
    }

    // F6: --resolution sizes the window so its DRAWABLE has exactly WxH
    // pixels (the F6 gate preset is 1920x1080 at scale 1).
    if (options_.resolutionWidth > 0) {
        const float density = std::max(SDL_GetWindowPixelDensity(window_), 1.0f);
        SDL_SetWindowSize(window_, static_cast<int>(std::lround(options_.resolutionWidth / density)),
                          static_cast<int>(std::lround(options_.resolutionHeight / density)));
        SDL_SyncWindow(window_);
    }
    context_ = std::make_unique<MetalContext>(layer, shaderLibraryPath());
    // F6: --force-family apple9 restricts the EFFECTIVE capabilities before
    // any pipeline is requested; the physical device is reported unchanged.
    if (options_.forceApple9) context_->forceApple9();
    LOG_INFO("GPU family: physical %s, effective %s%s", context_->physicalFamilyName(), context_->effectiveFamilyName(),
             options_.forceApple9 ? " (--force-family apple9: Apple10 specialisations off, not an Apple9 emulation)" : "");
    int w = 0, h = 0;
    SDL_GetWindowSizeInPixels(window_, &w, &h);
    if (options_.offscreen && options_.resolutionWidth) {
        w = static_cast<int>(options_.resolutionWidth);
        h = static_cast<int>(options_.resolutionHeight);
    }
    if (options_.resolutionWidth > 0 &&
        (static_cast<u32>(w) != options_.resolutionWidth || static_cast<u32>(h) != options_.resolutionHeight)) {
        throw std::runtime_error("--resolution " + std::to_string(options_.resolutionWidth) + "x" +
                                 std::to_string(options_.resolutionHeight) + ": the drawable is " + std::to_string(w) +
                                 "x" + std::to_string(h) + " (window pixel density mismatch)");
    }
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
    if (options_.geometryPath == GeometryPath::Mesh) {
        // F6: Hi-Z backend.  auto = compute (SIMD-group reduction): spike S3
        // measured it bit-exact and faster than the Apple10 sampler path on
        // this M5 Max (docs/opt-log.md, "F6 — Spike"), and it is Apple9-legal.
        HiZBuilder::Backend hiz = HiZBuilder::Backend::Compute;
        if (options_.hizPath == HiZPath::Sampler) {
            if (!context_->effectiveApple10()) {
                throw std::runtime_error(std::string("--hiz-path sampler: the effective family is ") +
                                         context_->effectiveFamilyName() + " (Apple10 sampler min reduction missing)");
            }
            hiz = HiZBuilder::Backend::Sampler;
        }
        if (options_.overlay == OverlayMode::Overdraw || options_.overlay == OverlayMode::LightCount ||
            options_.overlay == OverlayMode::TileCost) {
            throw std::runtime_error("--overlay overdraw/lights/tilecost redraw the scene with the indexed path: not "
                                     "available with --geometry-path mesh (use --debug-view)");
        }
        MeshRenderer::Options mo;
        mo.visibility = options_.visibility;
        mo.cull        = options_.meshletCull;
        mo.hiz         = hiz;
        mo.debugView   = options_.debugView == MeshletDebugView::RT ? MeshletDebugView::None : options_.debugView;
        mo.debugHiZLevel = options_.debugHiZLevel;
        mo.checks      = options_.debugMeshlets > 0;
        mo.objectStage = options_.meshletObjectStage;
        mo.minPixels   = options_.meshletMinPixels;
        mo.triangleCull = options_.meshletTriangleCull;
        mo.salt        = options_.pipelineSalt;
        mo.genericOnly = options_.debugPipelineFallback;
        mo.forceVariant = options_.forceVariant;
        mesh_ = std::make_unique<MeshRenderer>(*context_, *pipelines_, *renderer_, mo);
    }
    if (options_.visibility)
        visibility_ = std::make_unique<VisibilityRenderer>(*context_, *pipelines_, *renderer_, *mesh_,
                                                           options_.materialBinning, options_.debugVisibility,
                                                           options_.tileResolve, options_.adaptiveShading,
                                                           options_.shadows != ShadowMode::Off || options_.directLighting != DirectLightingMode::Legacy || options_.gi != GiMode::Off || options_.reflections!=ReflectionMode::Off || options_.ao!=AoMode::Off || options_.lightingDenoise!=LightingDenoiseMode::Off || options_.atmosphere || options_.fog || options_.clouds);
    if (options_.post) {
        PostProcessor::Options po;
        po.physicalFloat32=options_.atmosphere || options_.fog || options_.clouds;
        po.forceReset = options_.debugUpscalerReset;
        po.corruptExposure = options_.debugExposureCorrupt;
        po.jitterVariant = options_.jitterVariant;
        po.debugMotionScale = options_.debugMotionScale;
        po.checkCurves = options_.debugPostCurves;
        po.corruptCurves = options_.debugPostCurvesCorrupt;
        po.neutralMipBias = options_.debugNeutralMipBias;
        po.temporal = options_.temporalUpscale;
        po.isolatedMetalFX = options_.isolatedMetalFX;
        po.resizeSettleFrames = options_.metalfxResizeSettleFrames;
        po.debugWorkerCrash = options_.debugMetalFXWorkerCrash;
        po.debugWorkerDelayMs = options_.debugMetalFXWorkerDelayMs;
        po.autoExposure = options_.autoExposure;
        po.tonemap = options_.tonemap;
        po.sharpening = options_.sharpening;
        po.whitePoint = options_.toneWhite;
        po.views = options_.temporalViews;
        post_ = std::make_unique<PostProcessor>(*context_, *pipelines_, po);
        post_->prewarm(context_->width(), context_->height());
        dynamicScale_ = options_.renderScale;
    }
    context_->setOffscreen(options_.offscreen);
    context_->setFramesInFlight(options_.framesInFlight);
    context_->setFeedbackDiagnostics(options_.feedbackDelayMs, options_.feedbackFailFrame);
    graphExecutor_ = std::make_unique<MetalGraphExecutor>(*context_, pipelines_.get());
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
        if (!options_.scenarioAsync) params.async = std::vector<std::string>{};
        params.remat        = options_.graphRemat;
        params.views        = options_.scenarioViews;
        params.rematIterations = options_.graphRematCost;
        scenario_ = std::make_unique<ScenarioPasses>(*context_, *pipelines_, *options_.graphScenario, params);
    }
    if (options_.graphOpt == GraphOptMode::Plan) {
        const std::string path = options_.graphPlanPath.empty() ? shaderPath("graph-plans.json") : options_.graphPlanPath;
        std::ifstream in(path);
        std::stringstream text;
        text << in.rdbuf();
        std::string error;
        if (!in || !rg::fromJson(text.str(), graphPlans_, &error)) {
            throw std::runtime_error("--graph-opt plan: cannot read " + path + (error.empty() ? "" : ": " + error));
        }
        LOG_INFO("Graph plans: %zu loaded from %s", graphPlans_.size(), path.c_str());
    }
    if (options_.gpuTiming) {
        timestamps_ = std::make_unique<GpuTimestamps>(*context_, *pipelines_);
        graphExecutor_->setTimestamps(timestamps_.get());
    }
    if (options_.gpuCapture) {
        gpuCapture_ = std::make_unique<GpuCapture>(*context_, options_.gpuCaptureDir, options_.gpuCaptureMax);
    }
    overlays_ = std::make_unique<DebugOverlays>(*context_, *pipelines_);

    if (options_.rtEnabled) {
        rt_ = std::make_unique<AccelerationStructures>(*context_, *pipelines_, *renderer_, options_);
        if (options_.debugRt) rtChecker_ = std::make_unique<RtChecker>();
        if (options_.debugRt && visibility_)
            rtVisibility_ = std::make_unique<RtVisibilityChecker>(*context_, *pipelines_, *renderer_, *mesh_, *rt_);
        rtCheckRays_.reserve(512);
        rtCheckHits_.reserve(512);
        rtTlasTimes_.reserve(options_.frames);
        rtProbeTimes_.reserve(options_.frames);
        rtProbeNs_.reserve(options_.frames);
    }
    if (options_.shadows != ShadowMode::Off || options_.directLighting != DirectLightingMode::Legacy || options_.gi != GiMode::Off || options_.reflections!=ReflectionMode::Off || options_.ao!=AoMode::Off || options_.lightingDenoise!=LightingDenoiseMode::Off || options_.atmosphere || options_.fog || options_.clouds)
        shadows_ = std::make_unique<ShadowPasses>(*context_, *pipelines_, *renderer_, *mesh_, *visibility_, rt_.get(), options_);
    if (options_.directLighting != DirectLightingMode::Legacy || options_.gi != GiMode::Off || options_.reflections!=ReflectionMode::Off || options_.ao!=AoMode::Off || options_.lightingDenoise!=LightingDenoiseMode::Off || options_.fog)
        directLighting_=std::make_unique<DirectLightingPasses>(*context_,*pipelines_,*renderer_,*mesh_,*visibility_,rt_.get(),*shadows_,options_);
    if(options_.gi!=GiMode::Off)gi_=std::make_unique<GiPasses>(*context_,*pipelines_,*renderer_,*rt_,*shadows_,*directLighting_,options_);
    if(options_.reflections!=ReflectionMode::Off || options_.ao!=AoMode::Off || options_.lightingDenoise!=LightingDenoiseMode::Off)
        reflections_=std::make_unique<ReflectionPasses>(*context_,*pipelines_,*renderer_,*directLighting_,rt_.get(),gi_.get(),options_);
    if(options_.atmosphere || options_.fog || options_.clouds)
        atmosphere_=std::make_unique<AtmospherePasses>(*context_,*pipelines_,*renderer_,directLighting_.get(),rt_.get(),gi_.get(),shadows_.get(),options_);
    if(options_.lightingDenoise==LightingDenoiseMode::MetalFX) {
        MetalfxDenoise::Options config;config.enabled=true;config.views=options_.temporalViews;
        config.specularHitDistance=options_.reflections!=ReflectionMode::Off;
        MetalfxDenoise::Factory factory;
#ifdef PHOSPHOR_METALFX_DENOISED_GATEWAY_AVAILABLE
        factory.request=[this](auto* desc){return pipelines_->requestTemporalDenoisedScaler(desc);};
        factory.retire=[this](auto scaler){pipelines_->retireTemporalDenoisedScaler(std::move(scaler));};
#endif
        denoised_=std::make_unique<MetalfxDenoise>(*context_,*pipelines_,config,std::move(factory));
        roughnessSplit_=pipelines_->request(lighting::kernel("denoise_split_roughness"));roughnessTable_=lighting::table(*context_,2,2);
    }

    if(!options_.exportReference.empty())referenceSnapshot_=std::make_unique<ReferenceSnapshot>(*context_,*renderer_,options_.exportReference,options_.exportReferenceFrame,rt_.get());
    if(!options_.captureLinear.empty()||!options_.captureLinearSequence.empty()) {
        LinearCapture::Config capture;capture.path=options_.captureLinear;capture.sequence=options_.captureLinearSequence;
        capture.frame=options_.captureLinearFrame;capture.every=options_.captureEvery;capture.scalar=options_.captureLinearSignal==3||options_.captureLinearSignal==7;
        linearCapture_=std::make_unique<LinearCapture>(*context_,*pipelines_,std::move(capture));
    }
    ecs_        = std::make_unique<ECS>();
    gpuScene_   = std::make_unique<GpuScene>();
    {
        // F6.1: meshlet cook options (validated: a meshlet the mesh shader
        // cannot emit is never built).
        MeshletBuildOptions cook;
        cook.algorithm = options_.meshletSpatial ? MeshletAlgorithm::Spatial : MeshletAlgorithm::Standard;
        if (options_.meshletMaxVertices) cook.maxVertices = options_.meshletMaxVertices;
        if (options_.meshletMaxTriangles) cook.maxTriangles = options_.meshletMaxTriangles;
        const std::string err = validateMeshletOptions(cook);
        if (!err.empty()) throw std::runtime_error("meshlet cook options: " + err);
        gpuScene_->setMeshletOptions(cook);
    }
    store_      = std::make_unique<SceneStore>();
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
    if (!options_.captureSequence.empty())
        std::filesystem::create_directories(options_.captureSequence);
    if (!options_.capturePath.empty() || !options_.captureSequence.empty()) {
        capture_ = std::make_unique<FrameCapture>(*context_);
    }

    // F3.4: the harvest records the whole forward variant table, not only the
    // variants the benches of this run happen to use.
    if (pipelines_->harvesting()) {
        renderer_->requestAllVariants();
        if (mesh_) mesh_->requestAllVariants();
    }
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
    rtChecker_.reset();
    rtVisibility_.reset();
    atmosphere_.reset();
    reflections_.reset();
    denoised_.reset();
    if(roughnessTable_)roughnessTable_->release();
    referenceSnapshot_.reset();
    linearCapture_.reset();
    gi_.reset();
    directLighting_.reset();
    shadows_.reset();
    rt_.reset();

    if (activeBench_) {
        activeBench_->teardown(*ecs_, *gpuScene_);
        activeBench_.reset();
    }

    reloader_.reset();
    sceneChecker_.reset();
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
    meshletChecker_.reset();
    post_.reset();
    visibility_.reset();
    mesh_.reset();
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
            sceneSamples_.reserve(options_.frames);
            meshletSamples_.reserve(options_.frames);
            allocationsAtStart_ = context_->memory().allocationCount() + TemporalWorker::totalGpuAllocations();
            commandRebuildsAtStart_ = context_->commandBufferRebuilds();
            heapUsage(heapBlocksAtStart_, heapBytesAtStart_);
        }
    }
    while (running_) {
        if (context_->gpuFailureCount() != 0 || TemporalWorker::failureCount() != 0) {
            exitCode_ = 1;
            break;
        }
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
            const auto eventsStart = Clock::now();
            processEvents();
            eventPool->release();
            eventPumpMs_ = toMs(Clock::now() - eventsStart);
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
        if (presented) ++framesOnBench_;
        if (presented && options_.switchEvery > 0 && framesOnBench_ >= options_.switchEvery && !pendingBench_) {
            // Same path as the 1-7 hotkeys.
            pendingBench_ = static_cast<TestBenchType>((static_cast<int>(currentBench_) + 1) % testBenchCount());
        }
        if (presented && options_.benchmark()) {
            recordBenchmarkFrame(timer_->getDeltaTime(), toMs(Clock::now() - start - frameWait_), toMs(frameWait_));
        } else if (presented) {
            ++presentedFrames_;
        }
        if (presented && options_.debugFrameDelayMs)
            std::this_thread::sleep_for(std::chrono::milliseconds(options_.debugFrameDelayMs));
        if (presented && options_.debugCompileStorm && measuring() && samples_.size() == 60) {
            // F3.1 spike: 42 background compiles while frames are measured.
            LOG_INFO("Compile storm: requesting every forward variant (salt %u)", options_.pipelineSalt);
            renderer_->requestAllVariants();
        }
    }
    context_->waitIdle();
    if (context_->gpuFailureCount() != 0 || TemporalWorker::failureCount() != 0)
        exitCode_ = 1;
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
            if (!options_.capturePath.empty() && !capture_->writePng(options_.capturePath))
                exitCode_ = 1;
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
    const bool pass = reloads == 1 && probe > 0 && otherCount <= 1;
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
        sceneSamples_.reserve(options_.frames);
        meshletSamples_.reserve(options_.frames);
        trace_.reserve(options_.frames, options_.frames / std::max(options_.switchEvery, 1u) + 2);
        allocationsAtStart_ = context_->memory().allocationCount() + TemporalWorker::totalGpuAllocations();
        commandRebuildsAtStart_ = context_->commandBufferRebuilds();
        heapUsage(heapBlocksAtStart_, heapBytesAtStart_);
    } else if (presentedFrames_ > options_.warmup) {
        FrameRecord record;
        record.index   = static_cast<u32>(samples_.size());
        record.bench   = static_cast<u32>(currentBench_);
        record.flags   = frameFlags_;
        record.frameMs = dt * 1000.0f;
        record.cpuMs   = cpuMs;
        record.eventMs = eventPumpMs_;
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
    report.gpuAllocations =
        context_->memory().allocationCount() + TemporalWorker::totalGpuAllocations() - allocationsAtStart_;
    report.cpuHeapBlocksDelta = static_cast<i64>(heapBlocks) - static_cast<i64>(heapBlocksAtStart_);
    report.cpuHeapBytesDelta  = static_cast<i64>(heapBytes) - static_cast<i64>(heapBytesAtStart_);
    summarizeSamples(samples_, report);
    report.pipelinesJson = pipe::pipelineStatsJson(pipelines_->stats());
    report.gpuTiming        = timestamps_ != nullptr && timestamps_->enabled();
    report.gpuTimingUnfused = options_.gpuTimingUnfused;
    report.graph            = graphReport_;
    // F5 (schema 5): the counters of the last frames (the GPU is idle).
    for (u32 k = 0; k < 3; ++k) {
        u32 next = ~0u;
        for (u32 s = 0; s < 3; ++s) {
            if (slotFrame_[s] != ~0ull && (next == ~0u || slotFrame_[s] < slotFrame_[next])) next = s;
        }
        if (next == ~0u) break;
        onSceneCounters(next);
    }
    {
        const SceneSamples& ss = sceneSamples_;
        SceneReport& sr = report.scene;
        sr.present   = true;
        sr.mode      = options_.gpuDriven == GpuDrivenMode::On ? "on" : "off";
        const SceneSyncStats& st = store_->stats();
        sr.instances = store_->instanceCount();
        sr.slots     = store_->slotCapacity();
        sr.buckets   = static_cast<u32>(store_->buckets().size());
        sr.materials = static_cast<u32>(store_->materials().size());
        sr.commands  = store_->commandCount();
        (void)st;
        sr.structureChanges = ss.structureChanges;
        sr.queueOverflow    = ss.queueOverflow;
        sr.uploadBytes    = summarize(ss.uploadBytes);
        sr.deltaRecords   = summarize(ss.deltaRecords);
        sr.visible        = summarize(ss.visible);
        sr.culledFrustum  = summarize(ss.culledFrustum);
        sr.culledDistance = summarize(ss.culledDistance);
        sr.culledSize     = summarize(ss.culledSize);
        sr.drawCommands   = summarize(ss.drawCommands);
        sr.cpuCommands    = summarize(ss.cpuCommands);
        CpuPhasesReport& cp = report.cpuPhases;
        cp.present   = true;
        cp.sim       = summarize(ss.sim);
        cp.sceneSync = summarize(ss.sceneSync);
        cp.prepare   = summarize(ss.prepare);
        cp.ui        = summarize(ss.ui);
        cp.graph     = summarize(ss.graph);
        cp.submit    = summarize(ss.submit);
    }
    {
        // F6 (schema 6): physical device vs effective capabilities, preset.
        DeviceReport& h = report.hardware;
        h.present               = true;
        h.physicalDevice        = context_->gpuName();
        h.physicalFamily        = context_->physicalFamilyName();
        h.memoryBytes           = context_->physicalMemoryBytes();
        h.effectiveCapabilities = context_->effectiveFamilyName();
        if (options_.cullingScript && currentBench_ == TestBenchType::CullingViz)
            h.preset = "culling-viz-f6-" + std::to_string(report.width) + "x" + std::to_string(report.height);
        h.validationScope   = "development";
        h.unverifiedDevices = {"physical Apple9 (M3)", "T0 base"};
    }
    {
        auto &r = report.rendering;
        r.present = true;
        r.offscreen = options_.offscreen;
        r.deviceAllocatedBytes = context_->device()->currentAllocatedSize();
        r.engineResourceBytes = context_->memory().totalBytes();
        r.commandBufferRebuilds = context_->commandBufferRebuilds();
        r.commandBufferRebuildsMeasured = r.commandBufferRebuilds - commandRebuildsAtStart_;
        r.path = options_.adaptiveShading                      ? "visibility-adaptive"
                 : options_.tileResolve                        ? "visibility-tile"
                 : options_.visibility                         ? "visibility"
                 : options_.geometryPath == GeometryPath::Mesh ? "forward-mesh"
                                                               : "forward-indexed";
        r.asset = activeBench_->assetSource();
        r.materialBinning = options_.visibility && options_.materialBinning;
        if (visibility_) {
            r.binnedFrames = visibility_->binnedFrames();
            r.genericFrames = visibility_->genericFrames();
            r.guideChecks = visibility_->checkCount();
            r.guideFailures = visibility_->checkFailures();
            const auto adaptive = visibility_->adaptiveStats();
            r.shadedPixels = adaptive[0];
            r.reusedPixels = adaptive[1];
        }
        r.post = bool(post_);
        r.upscaler = post_ ? post_->effectiveUpscaler() : "none";
        r.tonemap = options_.tonemap == 1 ? "agx-fit" : options_.tonemap == 2 ? "custom-reinhard" : "aces-fit";
        r.mipBias = post_ ? post_->mipBias() : 0;
        r.inputWidth = post_ ? post_->inputWidth() : report.width;
        r.inputHeight = post_ ? post_->inputHeight() : report.height;
        r.views = post_ ? options_.temporalViews : 1;
        r.framesInFlight = options_.framesInFlight;
        r.gpuFailures = context_->gpuFailureCount();
        r.workerDeviceBytes = post_ ? post_->workerDeviceBytes() : 0;
        r.workerPhysicalFootprint = post_ ? post_->workerPhysicalFootprint() : 0;
        r.workerBridgeBytes = post_ ? post_->workerBridgeBytes() : 0;
        r.parentPhysicalFootprint = TemporalWorker::localPhysicalFootprint();
        r.workersSpawned = TemporalWorker::spawnedCount();
        r.workersReaped = TemporalWorker::reapedCount();
        r.workersPeakLive = TemporalWorker::peakLiveCount();
        r.workerFailures = TemporalWorker::failureCount();
        r.autoExposure = bool(post_) && options_.autoExposure;
        r.exposure = post_ ? post_->lastExposure() * post_->manualExposure() : settings_.exposure;
        r.edr = displayEDR_;
        r.headroom = displayHeadroom_;
        r.potentialHeadroom = displayPotentialHeadroom_;
        if (post_) {
            r.temporalFrames = post_->temporalFrames();
            r.nativeFrames = post_->fallbackFrames();
            r.historyResets = post_->resetCount();
        }
    }
    {
        MeshletReport& m = report.meshlets;
        m.present      = true;
        m.path         = geometryPathName(options_.geometryPath);
        m.cull         = mesh_ ? meshletCullName(options_.meshletCull) : "none";
        m.hizRequested = hizPathName(options_.hizPath);
        m.hizEffective = mesh_ && mesh_->twoPhase() ? mesh_->hiz()->backendName() : "none";
        m.cook         = meshletOptionsName(gpuScene_->meshletOptions());
        m.meshlets     = gpuScene_->getMeshletTotalCount();
        if (mesh_) {
            const MeshletSamples& ms = meshletSamples_;
            m.candidateCapacity = mesh_->capacity();
            m.overflowFrames    = ms.overflowFrames;
            m.historyResets     = ms.historyResets;
            m.checks            = meshletChecks_;
            m.checkFailures     = meshletFailures_;
            m.candidates        = summarize(ms.candidates);
            m.drawnA            = summarize(ms.drawnA);
            m.frustum           = summarize(ms.frustum);
            m.cone              = summarize(ms.cone);
            m.historyRejected   = summarize(ms.historyRejected);
            m.drawnB            = summarize(ms.drawnB);
            m.occludedB         = summarize(ms.occludedB);
            m.primitives        = summarize(ms.primitives);
            m.emitted           = summarize(ms.emitted);
            m.sizeCulled        = summarize(ms.sizeCulled);
        }
    }
    if (report.gpuTiming) passTimings_.summarize(report.passes, report.gpuPassSumMs, report.gpuFrameSpanMs);
    if (shadows_) {
        report.lighting.present=true; report.lighting.shadows=shadowModeName(options_.shadows);
        report.lighting.direct=directLightingModeName(options_.directLighting);report.lighting.gi=giModeName(options_.gi);
        report.lighting.reduced=options_.reducedLighting || options_.forceApple9;report.lighting.contact=options_.contactShadows;
        report.lighting.reflections=options_.reflections==ReflectionMode::RT?"rt":options_.reflections==ReflectionMode::SSR?"ssr":options_.reflections==ReflectionMode::Probes?"probes":"off";
        report.lighting.ao=options_.ao==AoMode::RTAO?"rtao":options_.ao==AoMode::GTAO?"gtao":"off";
        report.lighting.denoiseRequested=options_.lightingDenoise==LightingDenoiseMode::MetalFX?"metalfx":options_.lightingDenoise==LightingDenoiseMode::Custom?"custom":"off";
        report.lighting.denoiseEffective=denoised_&&denoised_->ready()?"metalfx":options_.lightingDenoise!=LightingDenoiseMode::Off?"custom":"off";
        if(denoised_)report.lighting.denoiseFallback=denoised_->fallbackReason();
        if(reflections_)report.lighting.probeSource=reflections_->probeSource();
        report.lighting.reflectionCorrupt=options_.debugReflectionCorrupt;report.lighting.volumeCorrupt=options_.debugVolumeCorrupt;report.lighting.volumeOracle=options_.volumeOracle;report.lighting.fogHomogeneous=options_.fogHomogeneous;
        report.lighting.atmosphere=options_.atmosphere;report.lighting.fog=options_.fog;report.lighting.clouds=options_.clouds;report.lighting.cloudFullRate=options_.cloudFullRate;
        report.lighting.cache=options_.shadowCache;report.lighting.seed=options_.lightingSeed;report.lighting.sunIndex=shadows_->sunIndex();
        report.lighting.candidates=options_.lightingCandidates;report.lighting.spatialSamples=options_.lightingSpatialSamples;report.lighting.giRays=options_.giRays;report.lighting.checks=lightingChecks_;report.lighting.failures=lightingFailures_;
    }
    if (rt_) {
        report.rt = rt_->report();
        report.rt.checks = rtChecks_;
        report.rt.checkFailures = rtFailures_;
        report.rt.tlasUpdateMs = summarize(rtTlasTimes_);
        report.rt.probeMs = summarize(rtProbeTimes_);
        report.rt.probeNsPerRay = summarize(rtProbeNs_);
        report.rt.probeRays = rtProbeRays_;
        report.rt.alphaTests = rtAlphaTests_;
        report.rt.opaqueAlphaTests = rtOpaqueAlphaTests_;
        report.rt.visibilityCompared = rtVisibilityCompared_;
        report.rt.visibilityMismatches = rtVisibilityMismatches_;
    }
    // OPT-0.4: work of the passes the cost model can price (CPU data of the
    // last frame; the benches' scenes do not change their draw lists).
    for (PassReport& unit : report.passes) {
        for (const std::string& pass : unit.passes) {
            PassWork w;
            w.pass = pass;
            if (pass == "Forward") {
                // F5: the draws of the last frame: per bucket in off, the
                // ICB's non-empty commands (GPU draw arguments) in on.
                const ForwardWork f = sceneForwardWork(report.width, report.height);
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
    if (rt_) {
        const auto& r = report.rt;
        std::printf("RT %u BLAS (%llu bytes, %u compacted) | %u instances/%u slots | TLAS %llu builds %llu refits | "
                    "update %.4f ms | probe %s %llu rays %.4f ns/ray | proxy %s %llu/%llu triangles\n",
                    r.blasCount, static_cast<unsigned long long>(r.blasBytes), r.compactedCount, r.instances,
                    r.capacity, static_cast<unsigned long long>(r.tlasBuilds), static_cast<unsigned long long>(r.tlasRefits),
                    static_cast<double>(r.tlasUpdateMs.mean), r.probe.c_str(),
                    static_cast<unsigned long long>(r.probeRays), static_cast<double>(r.probeNsPerRay.mean),
                    r.proxyMode.c_str(), static_cast<unsigned long long>(r.proxyTriangles),
                    static_cast<unsigned long long>(r.fullTriangles));
        if (options_.debugRt)
            std::printf("RT checks %u failures %u | %s\n", rtChecks_, rtFailures_,
                        rtChecks_ && !rtFailures_ ? "PASS" : "FAIL");
    }
    if (report.scene.present) std::printf("%s\n", formatSceneLine(report.scene).c_str());
    if (gpuSceneChecks_ > 0) {
        std::printf("GPU-SCENE checks %u | failures %u | %s\n", gpuSceneChecks_, gpuSceneFailures_,
                    gpuSceneFailures_ == 0 ? "PASS" : "FAIL");
    }
    if (report.meshlets.present && mesh_) {
        const MeshletReport& m = report.meshlets;
        std::printf("MESHLET path %s cull %s hiz %s cook %s | meshlets %u capacity %llu | candidates %.0f | A drawn %.0f "
                    "frustum %.0f cone %.0f size %.0f history %.0f | B drawn %.0f occluded %.0f | primitives %.0f emitted %.0f | overflow frames %u "
                    "| history resets %u | device %s (%s) effective %s\n",
                    m.path.c_str(), m.cull.c_str(), m.hizEffective.c_str(), m.cook.c_str(), m.meshlets,
                    static_cast<unsigned long long>(m.candidateCapacity), static_cast<double>(m.candidates.mean),
                    static_cast<double>(m.drawnA.mean), static_cast<double>(m.frustum.mean),
                    static_cast<double>(m.cone.mean), static_cast<double>(m.sizeCulled.mean),
                    static_cast<double>(m.historyRejected.mean),
                    static_cast<double>(m.drawnB.mean), static_cast<double>(m.occludedB.mean),
                    static_cast<double>(m.primitives.mean), static_cast<double>(m.emitted.mean), m.overflowFrames,
                    m.historyResets,
                    report.hardware.physicalDevice.c_str(), report.hardware.physicalFamily.c_str(),
                    report.hardware.effectiveCapabilities.c_str());
    }
    if (meshletChecks_ > 0) {
        std::printf("MESHLETS checks %u | failures %u | %s\n", meshletChecks_, meshletFailures_,
                    meshletFailures_ == 0 ? "PASS" : "FAIL");
    }
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
            if (!options_.offscreen || !options_.resolutionWidth ||
                (options_.resizeEvery && resizeFrames_ >= options_.resizeEvery))
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
    if (rt_) rt_->clear();
    rtCheckerGeometry_ = ~u64{0};
    const Clock::time_point t1 = Clock::now();

    if (activeBench_) {
        activeBench_->teardown(*ecs_, *gpuScene_);
        activeBench_.reset();
    }
    gpuScene_->clear();
    // Textures belong to a bench; a fresh manager drops the previous set.
    textures_.reset();
    textures_ = std::make_unique<MetalTextureManager>(*context_, options_.debugRt > 0 || !options_.exportReference.empty());

    currentBench_ = type;
    framesOnBench_ = 0;
    TestBenchParams benchParams;
    benchParams.scenePath = options_.scenePath;
    benchParams.lightingScenario=options_.lightingScene;benchParams.reflectionScenario=options_.reflectionScene;
    benchParams.instances         = options_.sceneInstances;
    benchParams.localLightCount=options_.localLightCount;benchParams.areaLights=options_.areaLights;benchParams.stationaryLights=options_.stationaryLights;
    benchParams.meshes            = options_.sceneMeshes;
    benchParams.dynamicCpuPercent = options_.dynamicCpuPercent;
    benchParams.churn             = options_.churn;
    benchParams.cullingScript     = options_.cullingScript;
    activeBench_ = createTestBench(type, benchParams);
    LOG_INFO("Switching to test bench: %s", activeBench_->getName());
    activeBench_->setup(*ecs_, *gpuScene_, *textures_);
    const Clock::time_point t2 = Clock::now();

    textures_->flushUploads();
    const Clock::time_point t3 = Clock::now();
    renderer_->syncGeometry(*gpuScene_);
    if (mesh_) mesh_->syncGeometry(*gpuScene_);
    // F6.5: a new scene has no history.
    ++sceneEpoch_;
    for (u32 view=0;view<HistoryRegistry::MaxViews;++view)
        hizRegistry_.invalidate(view,"bench switch");
    histories_ = {};
    // F5.1: full build of the persistent scene, uploaded through staging.
    store_->clear();
    store_->sync(*ecs_, *gpuScene_);
    ecs_->endFrame();
    renderer_->loadScene(*store_);
    if (mesh_) mesh_->loadScene(*store_, *gpuScene_);
    if (rt_) rt_->loadScene(*gpuScene_, *store_, *textures_);
    if (shadows_) shadows_->loadScene(*gpuScene_, *store_);
    if (directLighting_)directLighting_->loadScene(*gpuScene_,*store_);
    if(gi_)gi_->loadScene(*gpuScene_,*store_);
    if(reflections_)reflections_->loadScene(*gpuScene_,*store_);
    if(referenceSnapshot_)referenceSnapshot_->loadScene(*gpuScene_,*store_,*textures_);
    if (options_.debugRtDeform && gpuScene_->getMeshCount()) {
        const u32 first = gpuScene_->meshInfos()[0].vertexOffset;
        const u32 end = gpuScene_->getMeshCount() > 1 ? gpuScene_->meshInfos()[1].vertexOffset
                                                     : static_cast<u32>(gpuScene_->vertices().size());
        rtDeformedVertices_.assign(gpuScene_->vertices().begin() + first, gpuScene_->vertices().begin() + end);
    }
    sceneTime_ = 0.0;
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
    // Quality reference: reconstruct one fixed scene for a complete jitter
    // cycle, with no history from a different logical scene frame.
    const bool referenceStart = presentedFrames_ % options_.settledReference == 0;
    const bool referenceEnd = (presentedFrames_ + 1) % options_.settledReference == 0;
    if (!referenceStart)
        dt = 0;

    // F3: compilations finished since the last frame become visible now,
    // never while a frame is being encoded.
    pollShaderReload();
    if (pipelines_->beginFrame() > 0) frameFlags_ |= FramePipelineSwap;
    if (shaderGeneration_ != pipelines_->generation()) {
        if (shaderGeneration_ != ~u32{0})
            ++sceneEpoch_; // shading history belongs to the served shader generation
        shaderGeneration_ = pipelines_->generation();
    }

    // --- Simulation -----------------------------------------------------------
    PH_ZONE("Frame");
    // F6: a bench may script the camera (bench 7 --culling-script); a cut
    // (teleport) resets the temporal history.
    bool scriptedCut = false;
    {
        glm::vec3 pos{0.0f}, target{0.0f};
        bool scripted = false;
        if (options_.temporalScript) {
            const double time = std::fmod(sceneTime_ + dt, 8.0);
            const u32 phase = static_cast<u32>(time / 2.0);
            scriptedCut = phase != static_cast<u32>(std::fmod(sceneTime_, 8.0) / 2.0);
            auto base = activeBench_->getDefaultCamera();
            if (currentBench_ == TestBenchType::SceneViewer) {
                base.position = glm::vec3(-8.0f, 2.2f, 0.0f);
                base.target = glm::vec3(8.0f, 2.2f, 0.0f);
                base.orbit = false;
            } else if (base.orbit)
                base.position = base.target + glm::vec3(0, 2, base.distance);
            pos = base.position;
            target = base.target;
            if (phase == 0)
                pos.x += float(time) * 1.2f;
            else if (phase == 1) {
                pos += glm::vec3(3, 0.8f, 2);
                pos.z += float(time - 2) * 0.7f;
            } else if (phase == 2) {
                pos += glm::vec3(-2, 0, 1);
                target += glm::vec3(std::sin(float(time) * 5) * 2, 0, 0);
            } else
                pos.x += float(8 - time) * 0.8f;
            scripted = true;
        } else
            scripted = activeBench_->scriptedCamera(sceneTime_ + dt, pos, target, scriptedCut);
        if (scripted) {
            CameraSetup setup;
            setup.position = pos;
            setup.target   = target;
            setup.orbit    = false;
            aimCamera(setup);
        } else if (orbitMode_) {
            camera_->updateOrbit(*input_, dt);
        } else {
            camera_->updateFPS(*input_, dt);
        }
    }
    camera_->setAspect(static_cast<float>(context_->width()) / static_cast<float>(std::max(context_->height(), 1u)));
    camera_->updateMatrices();

    const Clock::time_point s0 = Clock::now();
    {
        PH_ZONE("Simulation");
        activeBench_->update(dt, *ecs_);
    }
    sceneTime_ += dt;
    const Clock::time_point s1 = Clock::now();
    // F5.1: O(changed) sync of the persistent scene, then the ECS change
    // lists are cleared (every component change of this frame is consumed).
    const bool checkScene = options_.debugGpuScene > 0 && (presentedFrames_ + 1) % options_.debugGpuScene == 0;
    if (checkScene && options_.debugGpuSceneCorrupt == SceneCorruption::Touch) corruptUntrackedTransform(*ecs_);
    {
        PH_ZONE("Scene sync");
        store_->sync(*ecs_, *gpuScene_);
        extractLights(*ecs_, lights_);
        if(atmosphere_) {
            const double clock=sceneTime_+(options_.timeJumpEveryN?double(presentedFrames_/options_.timeJumpEveryN)*options_.atmoDayLength*0.5:0);
            const auto physical=atmosphere_->prepareLighting(clock,glm::dvec3(camera_->getPosition()));
            std::erase_if(lights_,[](const auto& l){return l.type==LIGHT_DIRECTIONAL;});
            lights_.insert(lights_.begin(),physical.begin(),physical.end());
            if(gi_)gi_->setEnvironment(atmosphere_->environment());
        }
        ecs_->endFrame();
    }
    // Material reassignment can invalidate an offline proxy's protection.
    // Rebuild before beginning a GPU frame, with affected meshes promoted to
    // full geometry; never keep a simplified MASK/emissive mesh silently.
    if (rt_ && rt_->geometry().manifestApplied &&
        (store_->stats().fullMaterials || store_->stats().fullInstances ||
         !store_->materialDeltas().empty() || !store_->instanceDeltas().empty())) {
        bool reload = false;
        for (u32 mesh : rtProxyProtectedMeshes(gpuScene_->getMeshCount(), store_->instances(), store_->materials()))
            reload |= rt_->geometry().selections[mesh].level != RtProxyLevel::Full;
        if (reload) {
            LOG_INFO("RT: material assignment promotes protected meshes to full geometry");
            rt_->loadScene(*gpuScene_, *store_, *textures_);
            rtCheckerGeometry_ = ~u64{0};
        }
    }
    const Clock::time_point s2 = Clock::now();
    frameStats_->update(*timer_, context_->lastGpuMs());

    context_->layer()->setDisplaySyncEnabled(settings_.vsync);
    if (post_ && (presentedFrames_ % 30 == 0 || displayHeadroom_ <= 1.0f)) {
        const auto display = configureDisplayOutput(window_, context_->layer(), options_.displayOutput);
        displayHeadroom_ = display.headroom;
        displayPotentialHeadroom_ = display.potentialHeadroom;
        displayEDR_ = display.edr;
    }

    // --- Render ------------------------------------------------------------
    MetalContext::Frame frame;
    const Clock::time_point waitStart = Clock::now();
    const bool acquired = context_->beginFrame(frame);
    frameWait_ = Clock::now() - waitStart;
    if (!acquired) {
        pool->release();
        return false;
    }
    // F5: the slot's previous frame has completed: its scene counters.
    onSceneCounters(frame.slot);

    MTL::Texture *target = frame.target;
    const u32 width = static_cast<u32>(target->width()), height = static_cast<u32>(target->height());
    currentView_ = post_ ? static_cast<u32>(presentedFrames_ % options_.temporalViews) : 0;
    const glm::vec3 offset(float(currentView_) * 2.0f, 0, 0);
    if(options_.planetCameraHeight>=0){auto position=camera_->getPosition();position.y=options_.planetCameraHeight;camera_->setPosition(position);camera_->updateMatrices();}
    const glm::mat4 view = camera_->getView() * glm::translate(glm::mat4(1), -offset);
    const glm::mat4 unjittered = camera_->getProjection() * view;
    const glm::vec3 camPos = camera_->getPosition() + offset;
    float scale = post_ ? options_.renderScale : 1.0f;
    if (post_ && options_.dynamicResolution) {
        if (presentedFrames_ > 0 && presentedFrames_ % 30 == 0) {
            const float gpu = context_->lastGpuMs();
            if (gpu > options_.drsBudget * 1.1f)
                dynamicScale_ = std::max(0.5f, dynamicScale_ - 0.0625f);
            else if (gpu > 0 && gpu < options_.drsBudget * 0.8f)
                dynamicScale_ = std::min(1.0f, dynamicScale_ + 0.0625f);
        }
        scale = dynamicScale_;
    }
    if (post_ && options_.resolutionScript) {
        constexpr float scales[] = {1.0f, 0.75f, 0.5f, 0.875f};
        scale = scales[(presentedFrames_ / options_.resolutionScript) % 4];
    }
    if (options_.referenceScale > 1)
        scale = float(options_.referenceScale);
    // Round up so odd output dimensions never exceed the declared 2x
    // reconstruction limit at a 0.5 render scale.
    const u32 renderWidth = std::max(32u, static_cast<u32>(std::ceil(float(width) * scale)));
    const u32 renderHeight = std::max(32u, static_cast<u32>(std::ceil(float(height) * scale)));
    if (renderWidth > 8192 || renderHeight > 8192)
        throw std::runtime_error("Internal render dimensions exceed the 8192-pixel Hi-Z limit");
    float effectiveExposure = settings_.exposure;
    // Approximate clock metering is a display-only multiplier. Histogram auto
    // exposure uses the actual physical HDR instead; never expose DI/GI/LUTs.
    if(atmosphere_&&!options_.autoExposure)effectiveExposure*=std::clamp(std::exp2(-atmosphere_->exposureEv100()),1.0f/1024.0f,1024.0f);
    if (options_.exposureScript) {
        constexpr float factors[] = {0.25f, 1.0f, 4.0f, 1.0f};
        effectiveExposure *= factors[static_cast<u32>(sceneTime_ * 2) % 4];
    }
    renderBackingWidth_ = std::max(width, renderWidth);
    renderBackingHeight_ = std::max(height, renderHeight);
    const bool viewCameraCut=scriptedCut || (hizHistory().valid &&
            (glm::length(camPos-history().cameraPosition)>4.0f ||
             glm::dot(camera_->getFront(),history().cameraFront)<0.9063f));
    if (mesh_) {
        const auto decision=hizRegistry_.begin(currentView_,{renderWidth,renderHeight,width,height},sceneEpoch_,viewCameraCut,
            options_.historyResetEvery && presentedFrames_%options_.historyResetEvery==0);
        if(decision.reset && measuring())++meshletSamples_.historyResets;
    }
    GPUTemporalParams temporal{};
    std::memcpy(temporal.currentViewProjection, &unjittered[0][0], 64);
    std::memcpy(temporal.previousViewProjection, hizHistory().previousViewProjection.data(), 64);
    temporal.renderSize[0] = float(renderWidth);
    temporal.renderSize[1] = float(renderHeight);
    temporal.deltaTime = dt;
    temporal.manualExposure = effectiveExposure;
    temporal.historyValid =
        hizHistory().valid &&
        (!visibility_ || !visibility_->needsPoseReset(currentView_, store_->instances().size_bytes()));
    temporal.viewIndex = currentView_;
    if (post_) {
        const bool cut = viewCameraCut;
        temporal =
            post_->prepareFrame(frame.slot, frame.index, currentView_, renderWidth, renderHeight, width, height,
                                &unjittered[0][0], sceneEpoch_, cut,
                                (options_.historyResetEvery && presentedFrames_ % options_.historyResetEvery == 0) ||
                                    (options_.settledReference > 1 && referenceStart) ||
                                    visibility_->needsPoseReset(currentView_, store_->instances().size_bytes()),
                                history().sceneTime > 0 ? static_cast<float>(sceneTime_ - history().sceneTime) : dt,
                                effectiveExposure, displayHeadroom_);
        if (!temporal.historyValid && hizHistory().valid) {
            hizRegistry_.invalidate(currentView_,post_->histories().get(currentView_).resetReason);
        }
    }
    temporal.debugFlags = (options_.debugMotionCorrupt ? 1u : 0u) | (options_.debugGuideCorrupt ? 2u : 0u) |
                          (options_.debugAdaptiveNoHistory ? 4u : 0u) | (options_.debugHistoryCorrupt ? 8u : 0u);
    glm::mat4 projected = unjittered;
    applyRasterJitter(&projected[0][0], &unjittered[0][0], temporal.jitter[0], temporal.jitter[1], renderWidth,
                      renderHeight);
    FrameConstants constants{};
    std::memcpy(constants.viewProjection, &projected[0][0], sizeof(constants.viewProjection));
    std::memcpy(constants.view, &view[0][0], sizeof(constants.view));
    constants.cameraPosition[0] = camPos.x;
    constants.cameraPosition[1] = camPos.y;
    constants.cameraPosition[2] = camPos.z;
    constants.cameraPosition[3] = static_cast<float>(timer_->getTotalTime());
    constants.lightCount = static_cast<u32>(lights_.size());
    constants.debugMode  = static_cast<u32>(settings_.debugMode);
    constants.exposure = effectiveExposure;
    constants.frameIndex = static_cast<u32>(frame.index);


    // Capture the last frame of a run (or the first frame when interactive).
    const bool lastFrame = !options_.benchmark() || presentedFrames_ + 1 == options_.warmup + options_.frames;
    captureThisFrame_ = capture_ && lastFrame && !captured_;

    // F5.2/F5.5: the motion table of this frame and the cull parameters.
    motionSinCosTable(sceneTime_, motionSinCos_.data());
    {
        u32 flags = CULL_FLAG_FRUSTUM;
        if (options_.cullDistance > 0.0f) flags |= CULL_FLAG_DISTANCE;
        if (options_.cullMinPixels > 0.0f) flags |= CULL_FLAG_SIZE;
        cullParams_ = makeCullParams(projected, camera_->getProjection()[1][1], renderHeight, camPos,
                                     camera_->getFront(), camera_->getNear(), flags, options_.cullDistance,
                                     options_.cullMinPixels, store_->slotCapacity());
    }
    SceneRenderer::FrameParams sceneParams;
    sceneParams.slot         = frame.slot;
    sceneParams.width = renderWidth;
    sceneParams.height = renderHeight;
    sceneParams.mode         = options_.gpuDriven;
    sceneParams.cull         = cullParams_;
    sceneParams.motionSinCos = motionSinCos_.data();
    sceneParams.corrupt      = checkScene ? options_.debugGpuSceneCorrupt : SceneCorruption::None;
    // Bench switch or capacity growth (prepareFrame may reallocate): staged
    // before the graph so a recompilation sees the new capacities.
    const Clock::time_point s3 = Clock::now();
    const u64 sceneBytes = renderer_->prepareFrame(*store_, lights_, constants, textures_->tableAddress(), sceneParams);
    const bool meshletCheckFrame = mesh_ && options_.debugMeshlets > 0 && (presentedFrames_ + 1) % options_.debugMeshlets == 0;
    if (mesh_) {
        MeshRenderer::FrameParams mp;
        mp.slot   = frame.slot;
        mp.width = renderWidth;
        mp.height = renderHeight;
        mp.allocationWidth = renderBackingWidth_;
        mp.allocationHeight = renderBackingHeight_;
        mp.viewIndex = currentView_;
        std::memcpy(mp.cull.viewProj, &projected[0][0], sizeof(mp.cull.viewProj));
        std::memcpy(mp.cull.planes, cullParams_.planes, sizeof(mp.cull.planes));
        mp.cull.cameraPosition[0] = camPos.x;
        mp.cull.cameraPosition[1] = camPos.y;
        mp.cull.cameraPosition[2] = camPos.z;
        mp.cull.nearPlane         = camera_->getNear();
        std::memcpy(mp.prevViewProj, hizHistory().previousViewProjection.data(), sizeof(mp.prevViewProj));
        mp.historyValid = hizHistory().valid;
        mp.corrupt      = meshletCheckFrame ? options_.debugMeshletsCorrupt : MeshletCorruption::None;
        mesh_->requestCheckReadback(meshletCheckFrame);
        mesh_->prepareFrame(*store_, *gpuScene_, mp);
        if (mesh_->usingFallbackPipeline()) frameFlags_ |= FrameFallbackDraw;
    }
    if (visibility_)
        visibility_->prepareFrame(
            frame.slot, renderWidth, renderHeight, renderBackingWidth_, renderBackingHeight_, *store_,
            {textures_->getDefaultWhite(), textures_->getDefaultNormal(), textures_->getDefaultMR()},
            constants.exposure, constants.debugMode, temporal, constants);
    const bool rtCheckFrame = rt_ && options_.debugRt > 0 && (presentedFrames_ + 1) % options_.debugRt == 0;
    if (rt_) {
        if (options_.debugRtDeform && framesOnBench_ >= 8 && !rtDeformedVertices_.empty()) {
            const auto first = gpuScene_->meshInfos()[0].vertexOffset;
            for (size_t i = 0; i < rtDeformedVertices_.size(); ++i) {
                auto vertex = gpuScene_->vertices()[first + i];
                vertex.py += 0.08f * std::sin(vertex.px * 3.0f + static_cast<float>(frame.index) * 0.13f);
                rtDeformedVertices_[i] = vertex;
            }
            rt_->updateVertices(0, rtDeformedVertices_, frame.index + 2, (framesOnBench_ % 16) == 0);
        }
        AccelerationStructures::FrameParams rp;
        rp.slot = frame.slot;
        rp.index = frame.index;
        rp.width = renderWidth;
        rp.height = renderHeight;
        rp.allocationWidth = renderBackingWidth_;
        rp.allocationHeight = renderBackingHeight_;
        rp.constants = constants;
        rp.check = rtCheckFrame;
        rt_->prepareFrame(*store_, rp);
        if (rtVisibility_ && rt_->active())
            rtVisibility_->prepareFrame(frame.slot, renderWidth, renderHeight, constants);
    }
    overlays_->prepareFrame(overlayMode_, constants.lightCount, width, height);
    if(linearCapture_&&options_.captureLinearSignal>=6)pipelines_->waitAllFinal();
    if (shadows_) {
        ShadowPasses::Frame sf;sf.slot=frame.slot;sf.index=frame.index;sf.view=currentView_;sf.scene=sceneEpoch_;
        sf.width=renderWidth;sf.height=renderHeight;sf.backingWidth=renderBackingWidth_;sf.backingHeight=renderBackingHeight_;
        sf.cut=viewCameraCut;sf.reset=(options_.historyResetEvery && presentedFrames_%options_.historyResetEvery==0) || (atmosphere_&&atmosphere_->clockReset());
        sf.motionSinCos=motionSinCos_;sf.motionSinCosValid=true;
        sf.constants=constants;sf.nearPlane=camera_->getNear();std::memcpy(sf.unjitteredVP,&unjittered[0][0],64);
        shadows_->prepareFrame(*store_,lights_,sf);
        if(directLighting_)directLighting_->prepareFrame(*gpuScene_,*store_,lights_,sf);
        if(gi_)gi_->prepareFrame(*gpuScene_,*store_,lights_,sf);
        if(store_->stats().structure||store_->stats().fullInstances||store_->stats().fullNodes||!store_->instanceDeltas().empty()||!store_->nodeDeltas().empty()||!store_->motionSlots().empty()||!store_->dirtyRoots().empty())++surfaceGeometryEpoch_;
        if(store_->stats().fullMaterials||!store_->materialDeltas().empty())++surfaceMaterialEpoch_;
        const GiEnvironment environment=atmosphere_?atmosphere_->environment():GiEnvironment{};
        const auto lightEpoch=surfaceLightingEpoch_.update(lights_,environment,directLighting_?directLighting_->lightRevision():0);
        const SurfaceSignal signal{sceneEpoch_,surfaceGeometryEpoch_,surfaceMaterialEpoch_,rt_?rt_->geometryRevision():gpuScene_->geometryVersion(),lightEpoch,pipelines_->generation()};
        if(!(signal==previousSurfaceSignal_)){++surfaceSignalEpoch_;previousSurfaceSignal_=signal;}
        if(denoised_) {
            MetalfxDenoise::Frame df;df.slot=sf.slot;df.view=sf.view;df.index=sf.index;df.signalEpoch=surfaceSignalEpoch_;
            df.extent={renderWidth,renderHeight,width,height};std::memcpy(df.worldToView.data(),constants.view,64);
            const auto projection=unjittered*glm::inverse(view);std::memcpy(df.viewToClip.data(),glm::value_ptr(projection),64);
            df.jitterPixels={temporal.jitter[0],temporal.jitter[1]};df.cut=sf.cut;df.reset=sf.reset;
            df.preExposure=(options_.atmosphere||options_.fog||options_.clouds)?1.0f/64.0f:1.0f;
            denoised_->prepareFrame(df);
        }
        const bool custom=options_.lightingDenoise==LightingDenoiseMode::Custom || (denoised_&&!denoised_->ready());
        if(reflections_&&reflections_->ready())reflections_->prepareFrame(*store_,sf,custom,surfaceSignalEpoch_);
        if(atmosphere_)atmosphere_->prepareFrame(sf,surfaceGeometryEpoch_,surfaceMaterialEpoch_);
        const u32 flags=(options_.shadows!=ShadowMode::Off?1u:0u) | (options_.directLighting!=DirectLightingMode::Legacy?2u:0u) | (options_.gi!=GiMode::Off?4u:0u) |
                        (options_.reflections!=ReflectionMode::Off && reflections_&&reflections_->ready()?8u:0u) | (reflections_&&reflections_->ready()?16u:0u);
        visibility_->prepareLighting(flags,shadows_->sunIndex());
    }
    if(referenceSnapshot_) {
        ShadowPasses::Frame ref;ref.slot=frame.slot;ref.index=frame.index;ref.width=renderWidth;ref.height=renderHeight;ref.constants=constants;
        ReferenceCamera camera;std::memcpy(camera.position,glm::value_ptr(camPos),12);const auto front=camera_->getFront(),up=camera_->getUp();
        std::memcpy(camera.direction,glm::value_ptr(front),12);std::memcpy(camera.up,glm::value_ptr(up),12);camera.width=renderWidth;camera.height=renderHeight;
        camera.fovYRadians=camera_->getFovY();camera.nearPlane=camera_->getNear();camera.farPlane=0;
        camera.jitterPixels[0]=temporal.jitter[0];camera.jitterPixels[1]=temporal.jitter[1];
        referenceSnapshot_->prepareFrame(ref,camera,lights_,*store_,directLighting_.get());
    }
    if(linearCapture_)linearCapture_->prepareFrame(frame.slot,frame.index,renderWidth,renderHeight);
    if (!visibility_ && renderer_->usingFallback())
        frameFlags_ |= FrameFallbackDraw;
    const Clock::time_point s4 = Clock::now();

    if (mesh_ && overlayMode_ != OverlayMode::None && overlayMode_ != OverlayMode::Timings) {
        // F4.7 scene overlays redraw with the indexed path: not in the mesh path.
        LOG_WARN("Overlay %s is not available with --geometry-path mesh (use --debug-view)", overlayName(overlayMode_));
        overlayMode_      = OverlayMode::None;
        settings_.overlay = static_cast<int>(overlayMode_);
    }
    const GraphKey key{width,
                       height,
                       options_.ui,
                       capture_ != nullptr,
                       options_.debugSplitEncoding,
                       options_.debugAsyncCompute,
                       overlayMode_,
                       options_.gpuDriven,
                       renderer_->buffers().version(),
                       mesh_ ? mesh_->version() : 0,
                       mesh_ ? static_cast<u32>(mesh_->options().debugView) : 0,
                       target->pixelFormat() == MTL::PixelFormatRGBA16Float ? rg::Format::RGBA16Float
                                                                            : rg::Format::BGRA8Srgb,
                       renderBackingWidth_,
                       renderBackingHeight_,
                       rt_ ? rt_->version() : 0,
                       rtVisibility_ && rtVisibility_->ready(),
                       shadows_?shadows_->version():0,directLighting_?directLighting_->version():0,
                       gi_?gi_->version():0,referenceSnapshot_?referenceSnapshot_->version():0,
                       linearCapture_?linearCapture_->version():0,renderWidth,renderHeight,
                       reflections_?reflections_->version():0,denoised_?denoised_->version():0,atmosphere_?atmosphere_->version():0,reflections_&&reflections_->ready()};
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
    const Clock::time_point s5 = Clock::now();
    if (options_.ui) {
        PH_ZONE("UI");
        drawUi();
    }
    const Clock::time_point s6 = Clock::now();

    graphExecutor_->bindTexture(drawableRef_, target);
    if (capture_) graphExecutor_->bindBuffer(captureRef_, capture_->readback());
    if (graphDebug_) graphDebug_->bind(*graphExecutor_, frame.slot);
    if (scenario_) scenario_->bind(*graphExecutor_, frame.index);
    if (asyncProbe_) asyncProbe_->bind(*graphExecutor_, frame.slot);
    if (rt_) rt_->bindResources(*graphExecutor_);
    if (shadows_) shadows_->bindFrame(*graphExecutor_);
    if (directLighting_)directLighting_->bindFrame(*graphExecutor_);
    if(gi_)gi_->bindFrame(*graphExecutor_);
    if(graphKey_.reflectionReady)reflections_->bindFrame(*graphExecutor_);
    if(atmosphere_)atmosphere_->bindFrame(*graphExecutor_);
    if(denoised_)denoised_->bindFrame(*graphExecutor_);
    if(referenceSnapshot_)referenceSnapshot_->bindFrame(*graphExecutor_);
    if(linearCapture_)linearCapture_->bindFrame(*graphExecutor_);
    if (rtVisibility_) rtVisibility_->bindFrame(*graphExecutor_);
    {
        PH_ZONE("Graph execute");
        if (post_)
            post_->bindFrame(*graphExecutor_);
        graphExecutor_->execute(frame);
    }
    if (graphDebug_) graphDebug_->frameEncoded(frame.slot, frame.index);
    if (asyncProbe_) asyncProbe_->frameEncoded(frame.slot, frame.index);
    if (captureThisFrame_) captured_ = true;
    lastCpuCommands_ = renderer_->cpuCommands();

    const Clock::time_point s7 = Clock::now();
    {
        PH_ZONE("Submit");
        context_->submitFrame(frame);
    }
    slotFrame_[frame.slot]    = frame.index;
    slotReflectionRecorded_[frame.slot]=graphKey_.reflectionReady;
    slotMeasured_[frame.slot] = measuring();
    if (measuring()) {
        const Clock::time_point s8 = Clock::now();
        SceneSamples& ss = sceneSamples_;
        ss.sim.push_back(toMs(s1 - s0));
        ss.sceneSync.push_back(toMs(s2 - s1));
        ss.prepare.push_back(toMs(s4 - s3));
        ss.ui.push_back(toMs(s6 - s5));
        ss.graph.push_back(toMs(s7 - s6));
        ss.submit.push_back(toMs(s8 - s7));
        ss.uploadBytes.push_back(static_cast<float>(sceneBytes));
        const SceneSyncStats& st = store_->stats();
        ss.deltaRecords.push_back(static_cast<float>(st.instanceRecords + st.materialRecords + st.nodeRecords +
                                                     st.motionRecords));
        ss.cpuCommands.push_back(static_cast<float>(lastCpuCommands_));
        if (st.structure) ++ss.structureChanges;
    }
    if (checkScene && !checkGpuScene(frame.slot)) exitCode_ = 1;
    if (rtCheckFrame && !checkRayTracing(frame.slot)) exitCode_ = 1;
    if (visibility_ && options_.debugVisibility) {
        context_->waitIdle();
        if (!visibility_->check(*gpuScene_))
            exitCode_ = 1;
        if (post_ && !post_->checkExposure(visibility_->checkedHistogram(), visibility_->checkedHistogramLow(),
                                           visibility_->checkedHistogramHigh()))
            exitCode_ = 1;
    }
    if (post_ && options_.debugPostCurves) {
        context_->waitIdle();
        if (!post_->checkCurves())
            exitCode_ = 1;
    }
    if (post_)
        post_->finishFrame(&unjittered[0][0]);
    if (mesh_) {
        hizRegistry_.read(currentView_,frame.index+1);
        hizRegistry_.write(currentView_,frame.index+1,&projected[0][0]);
        history().sceneTime = sceneTime_;
        history().cameraPosition = camPos;
        history().cameraFront = camera_->getFront();
    }
    if (meshletCheckFrame && !checkMeshlets(frame.slot)) exitCode_ = 1;
    const u64 captureFrame = presentedFrames_ / options_.settledReference;
    if (capture_ && !options_.captureSequence.empty() && referenceEnd && captureFrame % options_.captureEvery == 0) {
        context_->waitIdle();
        char name[64];
        std::snprintf(name, sizeof(name), "frame-%06llu.png", static_cast<unsigned long long>(captureFrame));
        if (!capture_->writePng((std::filesystem::path(options_.captureSequence) / name).string()))
            exitCode_ = 1;
        auto metadataPath = std::filesystem::path(options_.captureSequence) / name;
        metadataPath.replace_extension(".json");
        std::ofstream metadata(metadataPath);
        metadata << "{\"frame\":" << captureFrame << ",\"submitted_frame\":" << presentedFrames_
                 << ",\"reference_samples\":" << options_.settledReference << ",\"view\":" << currentView_
                 << ",\"camera_cut\":" << (viewCameraCut ? "true" : "false")
                 << ",\"history_reset\":" << (temporal.historyValid ? "false" : "true")
                 << ",\"input_width\":" << renderWidth << ",\"input_height\":" << renderHeight << ",\"jitter\":["
                 << temporal.jitter[0] << "," << temporal.jitter[1] << "]}\n";
        if (!metadata)
            exitCode_ = 1;
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
    if (measured && rt_) {
        float update = 0, probe = 0;
        u32 updateParts = 0;
        bool probeValid = false;
        for (u32 unit = 0; unit < r.units && unit < passTimings_.unitCount(); ++unit) {
            if (!r.unitValid[unit]) continue;
            const auto& name = passTimings_.unitName(unit);
            if (name == "RT instances" || name == "RT TLAS") {
                update += r.ms[unit];
                ++updateParts;
            }
            if (name.starts_with("RT probe ")) {
                probe += r.ms[unit];
                probeValid = true;
            }
        }
        if (updateParts == 2) rtTlasTimes_.push_back(update);
        // Slot storage always has three entries; --frames-in-flight controls
        // queue throttling, not the physical ring index (MetalContext).
        const u32 slot = static_cast<u32>(r.frame % METAL_FRAMES_IN_FLIGHT);
        const auto c = rtCounterFrames_[slot] == r.frame ? rtCounterSnapshots_[slot]
                     : slotFrame_[slot] == r.frame ? rt_->counters(slot) : GPURtCounters{};
        if (probeValid && c.rays) {
            rtProbeTimes_.push_back(probe);
            rtProbeNs_.push_back(probe * 1000000.0f / static_cast<float>(c.rays));
        }
    }
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
    info.instances   = store_->instanceCount();
    info.drawBatches = static_cast<u32>(store_->buckets().size());
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
    {
        // F5: the GPU scene (counters of the last completed frame).
        const SceneSyncStats& st = store_->stats();
        ScenePanelInfo sp;
        sp.mode             = options_.gpuDriven == GpuDrivenMode::On ? "on" : "off";
        sp.instances        = store_->instanceCount();
        sp.slots            = store_->slotCapacity();
        sp.buckets          = static_cast<u32>(store_->buckets().size());
        sp.materials        = static_cast<u32>(store_->materials().size());
        sp.commands         = store_->commandCount();
        sp.visible          = options_.gpuDriven == GpuDrivenMode::On ? lastCounters_.visible : store_->instanceCount();
        sp.culledFrustum    = lastCounters_.culledFrustum;
        sp.culledDistance   = lastCounters_.culledDistance;
        sp.culledSize       = lastCounters_.culledSize;
        sp.drawCommands     = lastCounters_.drawCommands;
        sp.cpuCommands      = lastCpuCommands_; // the previous encoded frame
        sp.deltaRecords     = st.instanceRecords + st.materialRecords + st.nodeRecords + st.motionRecords;
        sp.structureChanges = sceneSamples_.structureChanges;
        sp.queueOverflow    = lastCounters_.queueOverflow;
        sp.uploadBytes      = st.uploadBytes;
        UIPanels::drawScenePanel(sp);
    }
    UIPanels::drawPassTimingsPanel(timestamps_ ? &passTimings_ : nullptr, context_->lastGpuMs(),
                                   options_.gpuTimingUnfused);
    if (overlayMode_ == OverlayMode::Timings) UIPanels::drawTimingsOverlay(timestamps_ ? &passTimings_ : nullptr);

    ImGui::Render();
}

void Engine::SceneSamples::reserve(u32 frames) {
    for (std::vector<float>* v : {&sim, &sceneSync, &prepare, &ui, &graph, &submit, &uploadBytes, &deltaRecords,
                                  &cpuCommands, &visible, &culledFrustum, &culledDistance, &culledSize, &drawCommands}) {
        v->clear();
        v->reserve(frames);
    }
    structureChanges = 0;
    queueOverflow    = 0;
}

void Engine::MeshletSamples::reserve(u32 frames) {
    for (std::vector<float>* v : {&candidates, &drawnA, &frustum, &cone, &historyRejected, &drawnB, &occludedB, &primitives, &emitted, &sizeCulled}) {
        v->clear();
        v->reserve(frames);
    }
    overflowFrames = 0;
    historyResets  = 0;
}

bool Engine::checkMeshlets(u32 slot) {
    PH_ZONE("Meshlet check");
    context_->waitIdle();
    ++meshletChecks_;
    if (!meshletChecker_) meshletChecker_ = std::make_unique<MeshletChecker>(*context_);
    const MeshletCheckResult r = meshletChecker_->check(*mesh_, *renderer_, *store_, *gpuScene_, slot);
    if (!r.pass) ++meshletFailures_;
    // stdout: scripts and the negative controls read these lines.
    std::printf("MESHLETS frame %u | history %s (generation %llu, last reset: %s) | %s\n", presentedFrames_,
                hizHistory().valid ? "valid" : "invalid", static_cast<unsigned long long>(hizHistory().generation),
                hizHistory().resetReason, formatMeshletCheck(r).c_str());
    std::fflush(stdout);
    return r.pass;
}

void Engine::onSceneCounters(u32 slot) {
    if (slotFrame_[slot] == ~0ull) return;
    if (rt_) {
        rtCounterSnapshots_[slot] = rt_->counters(slot);
        rtCounterFrames_[slot] = slotFrame_[slot];
        if (slotMeasured_[slot]) {
            const auto c = rtCounterSnapshots_[slot];
            rtProbeRays_ += c.rays;
            rtAlphaTests_ += c.alphaTests;
            rtOpaqueAlphaTests_ += c.opaqueAlphaTests;
        }
    }
    if(referenceSnapshot_)referenceSnapshot_->consume(slot);
    if(linearCapture_)linearCapture_->consume(slot);
    if(denoised_)for(const auto& check:denoised_->drainPackChecks())if(check.available&&!check.ok){++lightingFailures_;exitCode_=1;}
    const bool reflectionPassed=!slotReflectionRecorded_[slot]||reflections_->check(slot);
    if(!reflectionPassed){++lightingFailures_;exitCode_=1;}
    if(shadows_ && options_.debugLighting && (slotFrame_[slot]+1)%options_.debugLighting==0) {
        ++lightingChecks_;const bool pass=shadows_->check(slot) && (!directLighting_ || directLighting_->check(slot)) && (!gi_ || gi_->check(slot)) && reflectionPassed && (!atmosphere_ || atmosphere_->check(slot));
        if(!pass){if(reflectionPassed)++lightingFailures_;exitCode_=1;}
        std::printf("LIGHTING check frame %llu | %s\n",static_cast<unsigned long long>(slotFrame_[slot]),pass?"PASS":"FAIL");
    }
    lastCounters_ = renderer_->counters(slot);
    if (mesh_) {
        lastMeshletCounters_ = mesh_->counters(slot);
        if (slotMeasured_[slot]) {
            const GPUMeshletCounters& c = lastMeshletCounters_;
            MeshletSamples& ms          = meshletSamples_;
            ms.candidates.push_back(static_cast<float>(c.candidates));
            ms.drawnA.push_back(static_cast<float>(c.drawnA));
            ms.frustum.push_back(static_cast<float>(c.frustum));
            ms.cone.push_back(static_cast<float>(c.cone));
            ms.historyRejected.push_back(static_cast<float>(c.historyRejected));
            ms.drawnB.push_back(static_cast<float>(c.drawnB));
            ms.occludedB.push_back(static_cast<float>(c.occludedB));
            ms.primitives.push_back(static_cast<float>(c.primitivesA + c.primitivesB));
            // Without the triangle cull every triangle of a drawn meshlet is emitted.
            ms.emitted.push_back(static_cast<float>(options_.meshletTriangleCull ? c.emitted : c.primitivesA + c.primitivesB));
            ms.sizeCulled.push_back(static_cast<float>(c.sizeCulled));
            if (c.overflow) ++ms.overflowFrames;
        }
    }
    if (slotMeasured_[slot]) {
        SceneSamples& ss = sceneSamples_;
        const bool on = options_.gpuDriven == GpuDrivenMode::On;
        // Off draws every live instance with one CPU draw per bucket.
        ss.visible.push_back(static_cast<float>(on ? lastCounters_.visible : store_->instanceCount()));
        ss.culledFrustum.push_back(static_cast<float>(lastCounters_.culledFrustum));
        ss.culledDistance.push_back(static_cast<float>(lastCounters_.culledDistance));
        ss.culledSize.push_back(static_cast<float>(lastCounters_.culledSize));
        u32 offDraws = 0;
        for (const SceneBucket& b : store_->buckets()) offDraws += b.count > 0 ? 1u : 0u;
        ss.drawCommands.push_back(static_cast<float>(on ? lastCounters_.drawCommands : offDraws));
        ss.queueOverflow = std::max(ss.queueOverflow, lastCounters_.queueOverflow);
    }
    slotFrame_[slot]    = ~0ull;
    slotReflectionRecorded_[slot]=false;
    slotMeasured_[slot] = false;
}

ForwardWork Engine::sceneForwardWork(u32 width, u32 height) const {
    ForwardWork w;
    const auto& infos = gpuScene_->meshInfos();
    const u64 totalVertices = gpuScene_->vertices().size();
    const auto buckets = store_->buckets();
    const bool on = options_.gpuDriven == GpuDrivenMode::On;
    // On: the instance counts the GPU wrote for the last frame's commands.
    const u32 lastSlot = static_cast<u32>((context_->frameIndex() + METAL_FRAMES_IN_FLIGHT - 1) % METAL_FRAMES_IN_FLIGHT);
    const auto* args = on ? static_cast<const u32*>(renderer_->buffers().frame(lastSlot).drawArgs->contents()) : nullptr;
    for (size_t i = 0; i < buckets.size(); ++i) {
        const SceneBucket& b = buckets[i];
        if (b.mesh >= infos.size() || infos[b.mesh].indexCount == 0) continue;
        const u64 instances = on ? args[2 * b.command] : b.count;
        if (instances == 0) continue;
        const GPUMeshInfo& info = infos[b.mesh];
        const u64 end = b.mesh + 1 < infos.size() ? infos[b.mesh + 1].vertexOffset : totalVertices;
        const u64 meshVertices = end > info.vertexOffset ? end - info.vertexOffset : 0;
        ++w.draws;
        w.instances += instances;
        w.indices += u64(info.indexCount) * instances;
        w.vertices += meshVertices * instances;
    }
    w.pixels = u64(width) * height;
    w.lights = static_cast<u32>(lights_.size());
    return w;
}

bool Engine::checkRayTracing(u32 slot) {
    context_->waitIdle();
    ++rtChecks_;
    const auto readback = rt_->readback(slot);
    if (!readback.valid || readback.rays.empty() || readback.rays.size() != readback.hits.size()) {
        ++rtFailures_;
        std::printf("RT check frame %u | FAIL: missing same-frame readback\n", presentedFrames_);
        return false;
    }
    if (rtCheckerGeometry_ != readback.geometryRevision) {
        rtChecker_->setGeometry(readback.vertices, rt_->geometry().indices, rt_->meshes());
        rtCheckerGeometry_ = readback.geometryRevision;
    }
    // The visibility comparison covers the full image on the GPU. The exact
    // CPU oracle samples the whole ray population, not just its first row.
    const size_t count = std::min<size_t>(512, readback.rays.size());
    rtCheckRays_.resize(count);
    rtCheckHits_.resize(count);
    for (size_t i = 0; i < count; ++i) {
        const size_t index = (2 * i + 1) * readback.rays.size() / (2 * count);
        rtCheckRays_[i] = readback.rays[index];
        rtCheckHits_[i] = readback.hits[index];
    }
    const auto result = rtChecker_->check(rtCheckRays_, rtCheckHits_, readback.instances, readback.materials,
                                          textures_->cpuTextures());
    const auto counters = rt_->counters(slot);
    bool passed = result.ok() && counters.opaqueAlphaTests == 0 && counters.invalidMesh == 0;
    if (rtVisibility_) {
        const auto v = rtVisibility_->result(slot);
        const bool complete = v.valid && v.frame == readback.frame && v.compared == u64(v.width) * v.height &&
                              !v.invalidVisibility && !v.invalidRt && !v.invalidDepth && !v.depthWithoutVisibility;
        passed &= complete;
        rtVisibilityCompared_ += v.compared;
        rtVisibilityMismatches_ += v.mismatches;
        std::printf("RT visibility frame %llu | %u/%u mismatches (hit/miss %u slot %u depth %u) | "
                    "interior opaque %u/%u edge opaque %u/%u interior MASK %u/%u edge MASK %u/%u | "
                    "max relative distance %.7g tolerance %.7g | %s\n",
                    static_cast<unsigned long long>(v.frame), v.mismatches, v.compared, v.hitMiss, v.slot, v.depth,
                    v.category[0].mismatches, v.category[0].compared, v.category[1].mismatches, v.category[1].compared,
                    v.category[2].mismatches, v.category[2].compared, v.category[3].mismatches, v.category[3].compared,
                    static_cast<double>(v.maxRelativeDistanceError), static_cast<double>(v.relativeDistanceTolerance),
                    complete ? "MEASURED" : "INCOMPLETE");
    }
    if (!passed) ++rtFailures_;
    rtCheckedRays_ += result.checked;
    rtAmbiguous_ += result.ambiguous;
    rtUnsupported_ += result.unsupported;
    std::printf("RT check frame %llu | checked %llu ambiguous %llu skipped %llu unsupported %llu | "
                "failures %llu opaque-alpha %u invalid-mesh %u | %s%s%s\n",
                static_cast<unsigned long long>(readback.frame), static_cast<unsigned long long>(result.checked),
                static_cast<unsigned long long>(result.ambiguous), static_cast<unsigned long long>(result.skipped),
                static_cast<unsigned long long>(result.unsupported), static_cast<unsigned long long>(result.failures),
                counters.opaqueAlphaTests, counters.invalidMesh, passed ? "PASS" : "FAIL",
                result.firstError.empty() ? "" : ": ", result.firstError.c_str());
    std::fflush(stdout);
    return passed;
}

bool Engine::checkGpuScene(u32 slot) {
    PH_ZONE("GPU scene check");
    context_->waitIdle();
    ++gpuSceneChecks_;
    if (!sceneChecker_) sceneChecker_ = std::make_unique<GpuSceneChecker>(*context_);
    // F6: in the mesh path Draw build writes draws only on an overflow frame.
    const bool gateOpen = !mesh_ || *static_cast<const u32*>(mesh_->frame(slot).gate->contents()) != 0u;
    const SceneCheckResult r = sceneChecker_->check(*renderer_, *store_, *gpuScene_, *ecs_, cullParams_,
                                                    motionSinCos_.data(), slot, mesh_ ? GpuDrivenMode::On : options_.gpuDriven,
                                                    gateOpen);
    if (!r.pass) ++gpuSceneFailures_;
    // stdout: scripts and the negative controls read these lines.
    std::printf("GPU-SCENE frame %u | %s\n", presentedFrames_, formatSceneCheck(r).c_str());
    std::fflush(stdout);
    return r.pass;
}

void Engine::declareFrameGraph(u32 width, u32 height) {
    using namespace rg;
    frameGraph_.reset();
    if (visibility_) visibility_->resetGraphRefs();

    const TextureDesc screen{graphKey_.outputFormat, width, height};
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

    // F5: the scene's passes (update, transforms, and with gpu-driven on the
    // instance cull and the ICB build) run before the forward pass.
    // F6: in the mesh path the meshlet candidate pass sits between the
    // instance cull and the draw build (which reads its overflow gate).
    if (!scenario_ && mesh_) {
        renderer_->addPassesToGraph(frameGraph_, GpuDrivenMode::On,
                                    [this](rg::RenderGraph& g) { return mesh_->addCandidatePass(g); });
    } else if (!scenario_) {
        renderer_->addPassesToGraph(frameGraph_, options_.gpuDriven);
    }

    if (rt_ && rt_->active()) rt_->addPassesToGraph(frameGraph_);

    // F6: phase A ("Forward"), Hi-Z, phase B: object + mesh shaders.
    if (!scenario_ && mesh_) {
        const auto depth = mesh_->addRasterPasses(frameGraph_, color, renderBackingWidth_, renderBackingHeight_);
        if (visibility_) {
            if (shadows_) {
                shadows_->addToGraph(frameGraph_,color,depth);
                if(directLighting_)directLighting_->addToGraph(frameGraph_,color,shadows_->depth());
                if(gi_)gi_->addToGraph(frameGraph_);
                visibility_->setLightingTextures(shadows_->mask(),directLighting_?directLighting_->direct():shadows_->zeroLighting(),gi_?gi_->irradiance():shadows_->zeroLighting());
            }
            visibility_->addResolve(frameGraph_, color, shadows_?shadows_->depth():depth);
            if(graphKey_.reflectionReady)visibility_->replaceColor(reflections_->addToGraph(frameGraph_,visibility_->color(),visibility_->depth()));
            if(atmosphere_)visibility_->replaceColor(atmosphere_->addToGraph(frameGraph_,visibility_->color(),visibility_->depth()));
            rg::TextureRef reconstructed;
            if(denoised_) {
                frameGraph_.addPass("Denoised roughness channel",PassType::Compute,[&](PassBuilder& b){b.read(visibility_->normalRoughness(),Usage::ShaderRead,StageDispatch);roughnessRef_=b.createTexture("Denoised perceptual roughness",{Format::R16Float,graphKey_.logicalWidth,graphKey_.logicalHeight});roughnessRef_=b.write(roughnessRef_,Usage::ShaderWrite,StageDispatch);},[this](PassContext& ctx){roughnessTable_->setAddress(visibility_->paramsAddress(),0);roughnessTable_->setTexture(static_cast<MTL::Texture*>(ctx.texture(visibility_->normalRoughness()))->gpuResourceID(),0);roughnessTable_->setTexture(static_cast<MTL::Texture*>(ctx.texture(roughnessRef_))->gpuResourceID(),1);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),*pipelines_,roughnessSplit_,roughnessTable_,graphKey_.logicalWidth,graphKey_.logicalHeight);});
                MetalfxDenoise::Inputs in;in.noisyColor=visibility_->color();in.customFallback=visibility_->color();in.depth=visibility_->depth();in.motion=directLighting_->motion();
                in.diffuseAlbedo=visibility_->diffuseAlbedo();in.specularAlbedo=visibility_->specularAlbedo();in.worldNormal=visibility_->normalRoughness();in.roughness=roughnessRef_;
                in.reactiveMask=visibility_->reactiveMask();if(graphKey_.reflectionReady)in.hitDistance=reflections_->hitDistance();
                const auto selected=denoised_->addToGraph(frameGraph_,in);if(denoised_->ready())reconstructed=selected;
            }
            if(linearCapture_)linearCapture_->addToGraph(frameGraph_,options_.captureLinearSignal==1?gi_->referenceDiffuse():
                                                       options_.captureLinearSignal==2?directLighting_->direct():options_.captureLinearSignal==6?reflections_->rawSpecular():
                                                       options_.captureLinearSignal==7?reflections_->rawAO():visibility_->color());
            color = post_ ? post_->addToGraph(frameGraph_, *visibility_, drawableRef_, graphKey_.outputFormat,reconstructed)
                          : visibility_->addPresent(frameGraph_, drawableRef_);
            visibility_->addChecks(frameGraph_);
            visibility_->addPoseSnapshot(frameGraph_);
            if (rtVisibility_ && rtVisibility_->ready() && rt_->active())
                rtVisibility_->addToGraph(frameGraph_, visibility_->visibility(), visibility_->depth());
        }
    }

    if (!scenario_ && !mesh_) frameGraph_.addPass(
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
            renderer_->declareDrawReads(b);
            b.setHints(HintGeometryHeavy);
            b.setProfileShaders("forward_vs,forward_fs");
            // F4.1 negative control: the known-cost pass runs before the
            // forward pass, never beside it.
            if (knownCost_) b.read(knownCost_->output(), Usage::ShaderRead, StageVertex);
            // F2.5 check: the draws are recorded by 4 threads into a render
            // pass suspended/resumed across command buffers.  Not with
            // gpu-driven on (measured, M5 Max / macOS 27.2): an
            // executeCommandsInBuffer inside a render encoder RESUMED in
            // another command buffer makes the GPU fault and recover (every
            // frame "Discarded (victim of GPU error/recovery)"), and the pass
            // encodes ~7 commands anyway; the ICB runs in one encoder.
            if (options_.debugSplitEncoding && options_.gpuDriven == GpuDrivenMode::Off) b.setParallelChunks(4);
        },
        [this](PassContext& ctx) {
            renderer_->encode(static_cast<MTL4::RenderCommandEncoder*>(ctx.encoder()), ctx.chunk(), ctx.chunkCount());
        });

    if(referenceSnapshot_)referenceSnapshot_->addToGraph(frameGraph_);

    if (!scenario_ && !mesh_) color = overlays_->addToGraph(frameGraph_, color, width, height, overlayMode_, *renderer_);

    if (rt_ && rt_->active() && options_.debugView == MeshletDebugView::RT)
        color = rt_->addDebugPresent(frameGraph_, color, graphKey_.outputFormat);

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
        captureColorRef_ =
            post_ && graphKey_.outputFormat == Format::RGBA16Float ? post_->addSDRCapture(frameGraph_, color) : color;
        capture_->prepare(width, height);
        captureRef_ = frameGraph_.importBuffer("Capture readback", {capture_->readbackSize()}, ImportOutput);
        frameGraph_.addPass(
            "Frame capture", PassType::Blit,
            [&](PassBuilder &b) {
                b.read(captureColorRef_, Usage::CopySrc, StageBlit);
                b.write(captureRef_, Usage::CopyDst, StageBlit);
                b.setSideEffect();
            },
            [this](PassContext &ctx) {
                // Always copy when capture instrumentation is enabled: an empty
                // encoder may be dropped by Metal and invalidate timestamps.
                capture_->encode(static_cast<MTL4::ComputeCommandEncoder *>(ctx.encoder()),
                                 static_cast<MTL::Texture *>(ctx.texture(captureColorRef_)),
                                 captureThisFrame_ || !options_.captureSequence.empty());
            });
    }

}

void Engine::buildFrameGraph(u32 width, u32 height) {
    PH_ZONE("Render graph build");
    using namespace rg;

    // OPT-1: the offline plan of this graph's family (scenarios only: the
    // engine's own graph has no plans).
    graphReport_ = GraphReport{};
    graphReport_.present = true;
    graphReport_.mode    = graphOptModeName(options_.graphOpt);
    graphReport_.plan    = "none";
    const GraphPlan* plan = nullptr;
    if (scenario_) {
        scenario_->resetBuildChoices();
        graphReport_.family = scenario_->family();
        if (options_.graphOpt == GraphOptMode::Plan) {
            plan = findPlan(graphPlans_, graphReport_.family);
            if (plan) {
                scenario_->setBuildChoices(plan->remat, plan->async);
            } else {
                LOG_WARN("Graph plan: none for '%s' in %zu plans: greedy", graphReport_.family.c_str(), graphPlans_.size());
            }
        }
    }
    declareFrameGraph(width, height);
    std::vector<u32> plannedOrder;
    if (plan) {
        std::string error;
        plannedOrder = planOrder(frameGraph_, *plan, &error);
        if (plannedOrder.empty()) {
            // Rejected (key or names): rebuild with the default choices, greedy.
            LOG_WARN("Graph plan rejected: %s: greedy", error.c_str());
            graphReport_.plan = "rejected: " + error;
            plan = nullptr;
            scenario_->resetBuildChoices();
            declareFrameGraph(width, height);
        } else {
            graphReport_.plan = "applied";
        }
    }

    // F4.1 attribution mode: every raster pass in its own render pass.
    CompileOptions compileOptions;
    compileOptions.fuseRasterPasses = !options_.gpuTimingUnfused;
    compileOptions.alias            = !options_.graphNoAlias;
    if (options_.graphOpt != GraphOptMode::Off) {
        compileOptions.aliasPolicy   = plan ? plan->aliasPolicy : AliasPolicy::Coloring;
        compileOptions.barrierPolicy = plan ? plan->barrierPolicy : BarrierPolicy::Minimal;
        compileOptions.lint          = LintMode::Warn;
    }
    if (plan) compileOptions.order = plannedOrder;
    graphReport_.alias    = aliasPolicyName(compileOptions.aliasPolicy);
    graphReport_.barriers = barrierPolicyName(compileOptions.barrierPolicy);
    // OPT-1 spike / debugging: an execution order given by pass names
    // (overrides a plan's order).
    if (!options_.graphOrder.empty()) compileOptions.order.clear();
    for (const std::string& name : options_.graphOrder) {
        const auto& passes = frameGraph_.passes();
        const auto it = std::find_if(passes.begin(), passes.end(), [&](const PassNode& p) { return p.name == name; });
        if (it == passes.end()) throw std::runtime_error("--graph-order: no pass named '" + name + "'");
        compileOptions.order.push_back(static_cast<u32>(it - passes.begin()));
    }
    if (!compileOptions.order.empty()) {
        // Passes the list does not name (the engine's UI and capture passes)
        // follow in declaration order.
        const CompiledGraph live = compileOrder(frameGraph_);
        for (u32 p = 0; p < frameGraph_.passes().size(); ++p) {
            const bool culled = p < live.culled.size() && live.culled[p];
            if (!culled &&
                std::find(compileOptions.order.begin(), compileOptions.order.end(), p) == compileOptions.order.end()) {
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
    {
        const CompiledGraph& c = graphExecutor_->compiled();
        graphReport_.passes       = static_cast<u32>(c.order.size());
        graphReport_.renderPasses = static_cast<u32>(c.renderGroups.size());
        graphReport_.memoryless   = static_cast<u32>(std::count(c.memoryless.begin(), c.memoryless.end(), true));
        graphReport_.barrierCount = 0;
        for (const PassBarriers& pb : c.barriers) graphReport_.barrierCount += static_cast<u32>(pb.barriers.size());
        graphReport_.dramBytes          = traffic.totalBytes();
        graphReport_.heapBytes          = c.aliasing.heapSize;
        graphReport_.heapUnaliasedBytes = c.aliasing.unaliasedSize;
        graphReport_.maxLiveBytes       = c.aliasing.maxLiveSize;
        for (const std::string& f : c.lint) LOG_INFO("Render graph lint: %s", f.c_str());
        LOG_INFO("Render graph mode %s (plan %s, alias %s, barriers %s): heap %.2f MiB, max live %.2f MiB, %u barriers",
                 graphReport_.mode.c_str(), graphReport_.plan.c_str(), graphReport_.alias.c_str(),
                 graphReport_.barriers.c_str(), static_cast<double>(graphReport_.heapBytes) / (1 << 20),
                 static_cast<double>(graphReport_.maxLiveBytes) / (1 << 20), graphReport_.barrierCount);
    }
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
