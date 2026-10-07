#pragma once

#include "core/types.h"

#include <optional>
#include <string>
#include <vector>

namespace phosphor {

// ---------------------------------------------------------------------------
// LaunchOptions -- command-line switches of the app.
//
//   --bench N          start on test bench N (1-based, like the hotkeys)
//   --frames N         benchmark mode: measure N frames, print a summary, exit;
//                      keyboard and mouse input is ignored (reproducible runs)
//   --warmup N         frames skipped before measuring (default 120)
//   --no-vsync         disable display sync (uncapped frame rate)
//   --no-ui            do not draw the ImGui overlay
//   --fixed-timestep   advance the simulation by 1/60 s per frame regardless
//                      of real time (deterministic captures)
//   --switch-every N   switch to the next test bench every N frames, through
//                      the same path as the 1-8 hotkeys (switching tests)
//   --resize-every N   resize the window every N frames, alternating between
//                      two sizes (tests the render graph recompilation)
//   --simulate-pressure  inject a memory-pressure warning at frame 10 and a
//                      critical notification at frame 20 (F1.5 tests)
//   --memory-stress N  create/destroy N GPU resources, check that device
//                      memory does not grow, exit (F1.6); exit code 1 on failure
//   --transient-test   aliasing self-test of the transient heap, exit (F1.1)
//   --debug-graph-transients  add a synthetic chain of compute/raster passes on
//                      transient resources (F2.2) whose result is read back and
//                      checked exactly on the CPU; prints a GRAPH-TRANSIENTS line
//                      and exits with code 1 on any mismatch or if nothing was
//                      aliased.  The drawable is untouched.
//   --inject-input     push synthetic key/mouse events every frame (tests
//                      that benchmark mode is immune to user input)
//   --capture FILE     write a frame to FILE (PNG): the last measured frame in
//                      benchmark mode, otherwise the first frame
//   --report FILE      write the benchmark summary to FILE (JSON)
//   --dump-graph FILE  write the frame's render graph (Graphviz dot, with the
//                      estimated DRAM bytes per resource) whenever it is compiled
//   --debug-split-encoding  encode the forward pass on several threads, in
//                      render passes suspended/resumed across command buffers (F2.5)
//   --debug-async-compute   add a synthetic compute pass on the second queue,
//                      synchronised with events (F2.6)
//   --pipeline-archive FILE  MTL4Archive to look pipelines up in (F3.4); by
//                      default shaders/phosphor-archive.metallib next to the
//                      shader library, if it exists
//   --no-pipeline-archive    ignore any archive: every pipeline is compiled
//   --harvest-pipelines FILE  record every pipeline descriptor the app creates
//                      (plus the whole variant table) into FILE (.mtl4-json)
//   --pipeline-sync    compile requested pipelines on the render thread
//                      (negative control for the async path, F3.1/F3.2)
//   --compile-qos Q    QoS of the compile threads: utility (default) or
//                      interactive (negative control, F3.1)
//   --pipeline-salt N  add a salt constant to every specialized variant so
//                      the OS shader cache cannot serve it (cold compiles)
//   --frame-trace FILE write per-frame times and bench-switch phases (CSV)
//   --debug-pipeline-fallback  the forward pass never requests its
//                      specialised variants and draws with the generic
//                      pipeline, the fallback a new variant uses while it
//                      compiles (F3.2/F3.3 check: same pixels as before F3)
//   --debug-mode N     start in debug view N (0 lit, 1 normals, 2 base colour;
//                      the F1-F3 keys, for benchmark/capture runs)
//   --force-variant N  the forward pass always draws with variant N of the
//                      generated table, even if it does not match the scene
//                      (tools/variant_check.sh: pixel check of every variant)
//   --debug-flexible-pipelines  render pipelines stay on their Metal 4
//                      flexible fallback (never the final object): measures
//                      what the flexible path changes (F3.2)
//   --debug-compile-storm  F3.1 spike: after 60 measured frames request every
//                      forward variant at once (with --pipeline-salt: real
//                      compiles) while the frames keep being measured
//   --shader-dir DIR   Debug: watch DIR's .metal files and hot-reload the
//                      pipelines when they change (F3.6)
//   --debug-hot-reload FILE  Debug self-test: after a few frames swap in the
//                      pipelines of the probe library FILE (.metallib) through
//                      the hot-reload path, then check the drawable exactly
//   --no-gpu-timing    F4.1: no GPU timestamps (profiling off: no counter
//                      heap, no timestamp commands)
//   --gpu-timing-unfused  F4.1 attribution: compile the graph without raster
//                      pass fusion, so every pass gets its own GPU time (costs
//                      the extra attachment store/load; declared in the report)
//   --gpu-timing-serial  F4.1: wait for each frame's GPU work before the next
//                      one (no overlap between frames: per-pass times without
//                      the neighbouring frame's work; use with --no-vsync)
//   --debug-gpu-cost N add a compute pass of known cost (N iterations of an
//                      LCG per thread; negative control of the pass timings)
//   --gpu-capture      F4.3: insert the Metal capture layer (F12 captures the
//                      next frame into a .gputrace document)
//   --gpu-capture-frame N  capture presented frame N (0-based; implies --gpu-capture)
//   --gpu-capture-over MS  capture the frame after one whose GPU time (sum of
//                      the timed passes) exceeds MS (implies --gpu-capture)
//   --gpu-capture-dir DIR  where .gputrace documents go (default: captures)
//   --gpu-capture-max K    at most K captures per run (default 1)
//   --overlay MODE     F4.7 debug overlay: none, overdraw, lights, tilecost,
//                      timings
//   --graph-scenario N OPT-1: the frame graph is graph scenario N
//                      (rendergraph/scenario.h: 0 deferred, 1 forward-plus,
//                      2 post-chain, 3 async-compute) of synthetic passes
//                      instead of the test bench's forward pass
//   --graph-scenario-size WxH  internal resolution of the scenario (2560x1440)
//   --graph-scenario-work F    scale of the scenario's ALU work and triangles
//   --graph-scenario-wide      RGBA16Float intermediates become RGBA32Float
//                      (2x bytes, same work: bandwidth control)
//   --graph-scenario-no-async  scenario 3's eligible passes stay on the
//                      graphics queue
//   --graph-scenario-views N   N independent copies of the scenario (split
//                      screen) composited by the present pass (1..6)
//   --graph-remat LIST comma-separated scenario signals recomputed by their
//                      consumers instead of stored (OPT-1.2)
//   --graph-remat-cost N  ALU steps a consumer spends per recomputed signal
//                      (default 16; break-even sweeps)
//   --graph-order LIST comma-separated pass names: the execution order the
//                      graph compiler must use (OPT-1 spike; validated)
//   --graph-no-alias   every transient gets its own memory (no aliasing:
//                      no aliasing barriers; OPT-1 spike)
//   --graph-opt MODE   OPT-1 graph compilation: off (the compiler of the end
//                      of F4: greedy aliasing, conservative barriers), greedy
//                      (OPT-1 heuristics without a plan: interval-colouring
//                      aliasing, minimal barriers), plan (the offline plan of
//                      the graph's family when its key matches, else greedy)
//   --graph-plan FILE  plans file (default: shaders/graph-plans.json next to
//                      the shader library)
//   --gpu-driven off|on  F5.3 how the scene's draws are submitted: off = the
//                      CPU encodes one draw per (cull class, mesh) bucket; on =
//                      GPU culling writes the indirect command buffer (default
//                      on; both draw the same image)
//   --instances N      bench 8: number of instances (default 1,000,000)
//   --scene-meshes K   bench 8: number of distinct meshes (1..1024, default 8)
//   --dynamic-cpu PCT  bench 8: percentage of the instances whose transform
//                      the CPU changes every frame (0..100, default 1)
//   --churn N          bench 8: N leaf instances destroyed and N created per frame
//   --cull-distance D  F5.5: instances farther than D from the camera are
//                      culled (0 = off)
//   --cull-min-pixels P  F5.5: instances whose projected size is below P
//                      pixels are culled (0 = off)
//   --debug-gpu-scene N  F5 self-check: every N frames the GPU scene is read
//                      back and compared exactly with the CPU model (prints a
//                      GPU-SCENE line, exit code 1 on mismatch; 0 = off)
//   --debug-gpu-scene-corrupt KIND  negative control of the check above:
//                      delta, plane, command or touch (one deliberate
//                      corruption that the check must report)
//   --geometry-path indexed|mesh  F6.3: the scene is drawn with indexed draws
//                      (the F5 reference, default) or with object + mesh
//                      shaders over meshlets (needs --gpu-driven on)
//   --meshlet-cull off|frustum|two-phase  F6.2/F6.5 (mesh path): no meshlet
//                      culling, conservative frustum + normal cone, or both
//                      plus two-phase Hi-Z occlusion (default two-phase)
//   --hiz-path compute|sampler|auto  F6.4 pyramid build: compute (Apple9,
//                      always available), sampler min reduction (Apple10
//                      only; refused when the effective family lacks it), auto
//                      = the backend chosen by spike S3 for the effective family
//   --force-family apple9  F6: restrict the EFFECTIVE GPU capabilities to
//                      Apple9 (Apple10 specialisations off, fallbacks used);
//                      the physical device, its memory and budget are
//                      unchanged and reported as such (not an M3 emulation)
//   --debug-meshlets N F6.7 self-check: every N frames the meshlet lists,
//                      decisions, counters and pyramids are read back and
//                      checked against the CPU references (MESHLETS line,
//                      exit code 1 on mismatch; 0 = off)
//   --debug-meshlets-corrupt KIND  negative control of the check above: id,
//                      depth or count (one deliberate corruption that the check
//                      must report)
//   --debug-view MODE  F6.7 mesh-path views: none, meshlets (colour per
//                      meshlet), cull (colour per decision: phase A, recovered
//                      in B, rejected by cone/occlusion), hiz (pyramid level
//                      overlay; --debug-hiz-level L, default 3)
//   --meshlet-builder standard|spatial, --meshlet-max-vertices N,
//   --meshlet-max-triangles N  F6.1 cook options (default standard 64/124)
//   --resolution WxH   drawable size in pixels (the window is sized for it;
//                      the F6 gate preset is 1920x1080)
//   --culling-script   bench 7: the F6 scripted Culling Viz (dense
//                      buildings, fly-through camera, disappearing occluder,
//                      camera cut, fast object, spawn/delete/reuse)
//   --meshlet-min-pixels P  F6.2 APPROXIMATE: meshlets whose projected bound
//                      covers less than P x P pixels are culled (0 = off,
//                      never in the exact/gate preset)
//   --meshlet-triangle-cull on|off  F6.3 option: the mesh shader also drops
//                      back/front-facing triangles and compacts the rest
//                      (default off: measured slower on M5 Max)
//   --meshlet-object off  spike S2: mesh-only pipeline without object stage
//                      (needs --meshlet-cull off; a measurement variant)
//   --history-reset-every N  F6.5: invalidate the Hi-Z history every N
//                      frames (camera-cut path; 0 = only on real cuts)
//
//   --rt off|on        F9 BLAS/TLAS infrastructure (default off)
//   --rt-tlas-rebuild-every N  periodic TLAS rebuild, 0 = structural changes only
//   --rt-proxy off|manifest    RT geometry source (default full geometry)
//   --rt-proxy-manifest PATH   explicit measured manifest (needs manifest mode)
//   --debug-view rt    RT primary-hit diagnostic, also on the indexed path
//   --debug-rt N       CPU reference comparison every N frames (0 = disabled)
//   --debug-rt-deform  deform mesh 0 for BLAS lifecycle diagnostics; requires
//                      --rt on --debug-rt N>0 --debug-view rt --rt-proxy off
//   --debug-rt-corrupt transform|mask|blas  negative control (needs --debug-rt N>0)
//   --debug-rt-proxy-transition mask|emissive|reassign|full-upload
//                      diagnostic material transition on a measured proxy;
//                      requires RT, manifest proxies, checks, and no bench switching
//   --rt-probe primary|shadow|ao|diffuse    per-ray traversal measurement
//                      RT settings require --rt on; synthetic graph scenarios,
//                      memory-stress and transient-only tests cannot use RT.
//
// Arguments not starting with "--" are ignored: macOS may add its own
// (e.g. -NSDocumentRevisionsDebugMode when launched from Xcode).
// ---------------------------------------------------------------------------

/// F5.3 submission of the scene's draws (--gpu-driven).
enum class GpuDrivenMode : u8 { Off, On };
/// F5 self-check negative controls (--debug-gpu-scene-corrupt).
enum class SceneCorruption : u8 { None, Delta, Plane, Command, Touch };

/// F6 geometry submission and meshlet culling.
enum class GeometryPath : u8 { Indexed, Mesh };
enum class MeshletCull : u8 { Off, Frustum, TwoPhase };
enum class HiZPath : u8 { Auto, Compute, Sampler };
enum class MeshletCorruption : u8 { None, Id, Depth, Count };
enum class MeshletDebugView : u8 { None, Meshlets, Cull, HiZ, RT };
[[nodiscard]] const char* geometryPathName(GeometryPath path);
[[nodiscard]] const char* meshletCullName(MeshletCull cull);
[[nodiscard]] const char* hizPathName(HiZPath path);
[[nodiscard]] const char* meshletDebugViewName(MeshletDebugView view);

/// F9 ray-tracing diagnostics (available on indexed and mesh paths).
enum class RtCorruption : u8 { None, Transform, Mask, Blas };
enum class RtProbe : u8 { None, Primary, Shadow, AO, Diffuse };
enum class RtProxyTransition : u8 { None, Mask, Emissive, Reassign, FullUpload };
[[nodiscard]] const char* rtCorruptionName(RtCorruption corruption);
[[nodiscard]] const char* rtProbeName(RtProbe probe);
[[nodiscard]] const char* rtProxyTransitionName(RtProxyTransition transition);

// F10-F12 experimental lighting paths. Defaults preserve the F9 baseline.
enum class ShadowMode : u8 { Off, CSM, RT };
enum class DirectLightingMode : u8 { Legacy, BruteForce, Clustered, ReSTIR };
enum class GiMode : u8 { Off, DDGI, Cache, ReSTIR };
[[nodiscard]] const char* shadowModeName(ShadowMode);
[[nodiscard]] const char* directLightingModeName(DirectLightingMode);
[[nodiscard]] const char* giModeName(GiMode);

/// OPT-1 graph compilation modes (--graph-opt).
enum class GraphOptMode : u8 { Off, Greedy, Plan };
[[nodiscard]] const char* graphOptModeName(GraphOptMode mode);

/// F4.7 debug overlays.
enum class OverlayMode : u8 { None, Overdraw, LightCount, TileCost, Timings };
[[nodiscard]] const char* overlayName(OverlayMode mode);

struct LaunchOptions {
    std::optional<int> bench;  // 0-based TestBenchType index
    u32         frames = 0;    // 0 = interactive
    u32         warmup = 120;
    bool        vsync  = true;
    bool        ui     = true;
    bool        fixedTimestep = false;
    u32         switchEvery   = 0; // 0 = never
    u32         resizeEvery   = 0; // 0 = never
    bool        simulatePressure = false;
    u32         memoryStress  = 0; // cycles; 0 = off
    bool        transientTest = false;
    bool        debugGraphTransients = false; // F2.2 aliasing self-test on the GPU
    bool        injectInput   = false; // test: synthetic keys/mouse every frame
    std::string capturePath;
    std::string reportPath;
    std::string dumpGraphPath;
    bool        debugSplitEncoding = false;
    bool        debugAsyncCompute  = false;
    // F3: pipelines.
    std::string pipelineArchivePath;          // empty: default location
    bool        noPipelineArchive  = false;
    std::string harvestPipelinesPath;
    bool        pipelineSync       = false;
    bool        compileQosInteractive = false; // --compile-qos interactive
    u32         pipelineSalt       = 0;        // 0 = no salt
    std::string frameTracePath;
    bool        debugCompileStorm  = false;
    bool        debugPipelineFallback = false;
    bool        debugFlexiblePipelines = false;
    u32         debugMode      = 0;
    std::optional<u32> forceVariant;
    std::string shaderDir;                     // hot reload (Debug)
    std::string debugHotReloadPath;
    // F4: observability.
    bool        gpuTiming        = true;
    bool        gpuTimingUnfused = false;
    bool        gpuTimingSerial  = false;
    u32         debugGpuCost     = 0;      // 0 = no known-cost pass
    bool        gpuCapture       = false;  // capture layer inserted
    std::optional<u32> gpuCaptureFrame;
    float       gpuCaptureOverMs = 0.0f;   // 0 = off
    std::string gpuCaptureDir    = "captures";
    u32         gpuCaptureMax    = 1;
    OverlayMode overlay          = OverlayMode::None;
    // OPT-1: graph scenarios and plans.
    std::optional<u32> graphScenario;
    u32         scenarioWidth    = 2560;
    u32         scenarioHeight   = 1440;
    float       scenarioWork     = 1.0f;
    bool        scenarioWide     = false;
    bool        scenarioAsync    = true;
    u32         scenarioViews    = 1;
    std::vector<std::string> graphRemat;
    u32         graphRematCost   = 16;
    std::vector<std::string> graphOrder;
    bool        graphNoAlias     = false;
    GraphOptMode graphOpt        = GraphOptMode::Off;
    std::string graphPlanPath;               // empty: default location

    /// True when the app runs a fixed number of frames and then exits.
    // F5: GPU scene and GPU-driven submission.
    GpuDrivenMode gpuDriven      = GpuDrivenMode::On; // --gpu-driven off|on
    u32         sceneInstances   = 0;     // --instances N (bench 8; 0 = bench default)
    u32         sceneMeshes      = 0;     // --scene-meshes K (bench 8; 0 = bench default)
    float       dynamicCpuPercent = -1.0f; // --dynamic-cpu PCT (bench 8; < 0 = bench default)
    u32         churn            = 0;     // --churn N spawn + despawn per frame (bench 8)
    float       cullDistance     = 0.0f;  // --cull-distance D (0 = off)
    float       cullMinPixels    = 0.0f;  // --cull-min-pixels P (0 = off)
    u32         debugGpuScene    = 0;     // --debug-gpu-scene N: exact readback check every N frames (0 = off)
    SceneCorruption debugGpuSceneCorrupt = SceneCorruption::None; // --debug-gpu-scene-corrupt KIND
    // F6: mesh shaders, meshlet culling, Hi-Z, capabilities.
    GeometryPath geometryPath    = GeometryPath::Indexed;
    MeshletCull  meshletCull     = MeshletCull::TwoPhase;
    HiZPath      hizPath         = HiZPath::Auto;
    bool         forceApple9     = false;
    u32          debugMeshlets   = 0;     // --debug-meshlets N (0 = off)
    MeshletCorruption debugMeshletsCorrupt = MeshletCorruption::None;
    MeshletDebugView debugView   = MeshletDebugView::None;
    u32          debugHiZLevel   = 3;
    bool         meshletSpatial  = false; // --meshlet-builder spatial
    u32          meshletMaxVertices  = 0; // 0 = default (64)
    u32          meshletMaxTriangles = 0; // 0 = default (124)
    // F9: RT infrastructure is opt-in until a lighting consumer needs it.
    bool         rtEnabled = false;           // --rt off|on
    u32          rtTlasRebuildEvery = 0;        // 0 = structural changes only; cadence needs engine measurements
    bool         rtProxyManifest = false;      // --rt-proxy off|manifest
    std::string  rtProxyManifestPath;          // empty = scene's default manifest, missing = full geometry
    u32          debugRt = 0;                  // --debug-rt N: readback/check cadence, 0 = off
    bool         debugRtDeform = false;       // diagnostic mesh-0 deformation; raster bounds are unchanged
    RtProxyTransition debugRtProxyTransition = RtProxyTransition::None;
    RtCorruption debugRtCorrupt = RtCorruption::None;
    RtProbe      rtProbe = RtProbe::None;
    // F10-F12: writing presets only; numerical adoption requires tester evidence.
    ShadowMode shadows = ShadowMode::Off;
    DirectLightingMode directLighting = DirectLightingMode::Legacy;
    GiMode gi = GiMode::Off;
    bool contactShadows = false, shadowCache = false, reducedLighting = false;
    u32 shadowMapResolution = 2048, lightingSeed = 1;
    u32 lightingCandidates = 8, lightingSpatialSamples = 4, giRays = 64;
    u32 debugLighting = 0, debugLightingCorrupt = 0;
    std::string exportReference;
    // F7/F8 renderer, temporal reconstruction, capture and bounded diagnostics.
    std::string scenePath;
    bool visibility = false, materialBinning = false; // measured baseline; specialization stays opt-in
    bool tileResolve = false;
    bool adaptiveShading = false;
    bool debugAdaptiveNoHistory = false;
    bool post = false, temporalUpscale = false, autoExposure = false, dynamicResolution = false;
    bool isolatedMetalFX = false;
    u32 metalfxResizeSettleFrames = 4; // stable output frames before a new in-process scaler is requested
    u32 debugMetalFXWorkerCrash = 0, debugMetalFXWorkerDelayMs = 0;
    u32 debugFrameDelayMs = 0;
    bool debugVisibility = false, debugMotionCorrupt = false, debugExposureCorrupt = false, debugUpscalerReset = false;
    bool debugGuideCorrupt = false;
    bool debugHistoryCorrupt = false;
    bool debugPostCurves = false, debugPostCurvesCorrupt = false;
    bool debugNeutralMipBias = false;
    bool offscreen = false, temporalScript = false, exposureScript = false;
    std::string captureSequence;
    u32 captureEvery = 1;
    u32 referenceScale = 1;
    u32 settledReference = 1;
    u32 framesInFlight = 3, feedbackDelayMs = 0, feedbackFailFrame = 0, jitterVariant = 0;
    u32 tonemap = 0, temporalViews = 1, resolutionScript = 0, displayOutput = 0;
    float renderScale = 1.0f, drsBudget = 14.0f, sharpening = 0.0f, toneWhite = 4.0f, debugMotionScale = 1.0f;
    u32          resolutionWidth  = 0;    // --resolution WxH (0 = window default)
    u32          resolutionHeight = 0;
    bool         cullingScript    = false;
    u32          historyResetEvery = 0;
    bool         meshletObjectStage = true; // --meshlet-object off: spike S2 mesh-only pipeline
    float        meshletMinPixels   = 0.0f; // --meshlet-min-pixels P (approximate size cull, 0 = off)
    bool         meshletTriangleCull = false; // --meshlet-triangle-cull on

    [[nodiscard]] bool benchmark() const { return frames > 0; }
};

/// Parse argv into `out`.  Returns false and sets `error` on malformed input.
/// `benchCount` bounds the accepted --bench values.
bool parseLaunchOptions(int argc, const char* const* argv, int benchCount,
                        LaunchOptions& out, std::string& error);

} // namespace phosphor
