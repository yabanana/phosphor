#include "core/launch_options.h"

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string_view>

namespace phosphor {

namespace {

bool parseU32(std::string_view text, u32& value) {
    const char* end = text.data() + text.size();
    auto [ptr, ec] = std::from_chars(text.data(), end, value);
    return ec == std::errc{} && ptr == end;
}

} // namespace

const char* graphOptModeName(GraphOptMode mode) {
    switch (mode) {
    case GraphOptMode::Off:    return "off";
    case GraphOptMode::Greedy: return "greedy";
    case GraphOptMode::Plan:   return "plan";
    }
    return "off";
}

const char* overlayName(OverlayMode mode) {
    switch (mode) {
    case OverlayMode::None:       return "none";
    case OverlayMode::Overdraw:   return "overdraw";
    case OverlayMode::LightCount: return "lights";
    case OverlayMode::TileCost:   return "tilecost";
    case OverlayMode::Timings:    return "timings";
    }
    return "none";
}

const char* geometryPathName(GeometryPath path) { return path == GeometryPath::Mesh ? "mesh" : "indexed"; }

const char* meshletCullName(MeshletCull cull) {
    switch (cull) {
    case MeshletCull::Off:      return "off";
    case MeshletCull::Frustum:  return "frustum";
    case MeshletCull::TwoPhase: return "two-phase";
    }
    return "off";
}

const char* hizPathName(HiZPath path) {
    switch (path) {
    case HiZPath::Auto:    return "auto";
    case HiZPath::Compute: return "compute";
    case HiZPath::Sampler: return "sampler";
    }
    return "auto";
}

const char* meshletDebugViewName(MeshletDebugView view) {
    switch (view) {
    case MeshletDebugView::None:     return "none";
    case MeshletDebugView::Meshlets: return "meshlets";
    case MeshletDebugView::Cull:     return "cull";
    case MeshletDebugView::HiZ:      return "hiz";
    }
    return "none";
}

bool parseLaunchOptions(int argc, const char* const* argv, int benchCount,
                        LaunchOptions& out, std::string& error) {
    out = LaunchOptions{};
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (!arg.starts_with("--")) continue;

        const bool hasValue = i + 1 < argc;
        auto needValue = [&]() -> std::optional<std::string_view> {
            if (!hasValue) {
                error = std::string(arg) + " requires a value";
                return std::nullopt;
            }
            return std::string_view(argv[++i]);
        };
        auto needCount = [&](u32& target) {
            const auto value = needValue();
            if (!value) return false;
            if (!parseU32(*value, target)) {
                error = std::string(arg) + ": expected a non-negative integer, got '" + std::string(*value) + "'";
                return false;
            }
            return true;
        };
        // A finite decimal number (std::strtof); NaN/inf are rejected.
        auto needFloat = [&](float& target) {
            const auto value = needValue();
            if (!value) return false;
            const std::string text(*value);
            char* end = nullptr;
            const float f = std::strtof(text.c_str(), &end);
            if (text.empty() || end != text.c_str() + text.size() || !std::isfinite(f)) {
                error = std::string(arg) + ": expected a number, got '" + text + "'";
                return false;
            }
            target = f;
            return true;
        };
        auto needString =[&](std::string& target) {
            const auto value = needValue();
            if (!value) return false;
            target = *value;
            return true;
        };

        if (arg == "--bench") {
            u32 bench = 0;
            if (!needCount(bench)) return false;
            if (bench < 1 || bench > static_cast<u32>(benchCount)) {
                error = "--bench: expected 1.." + std::to_string(benchCount);
                return false;
            }
            out.bench = static_cast<int>(bench) - 1;
        } else if (arg == "--frames") {
            if (!needCount(out.frames)) return false;
        } else if (arg == "--warmup") {
            if (!needCount(out.warmup)) return false;
        } else if (arg == "--no-vsync") {
            out.vsync = false;
        } else if (arg == "--no-ui") {
            out.ui = false;
        } else if (arg == "--fixed-timestep") {
            out.fixedTimestep = true;
        } else if (arg == "--inject-input") {
            out.injectInput = true;
        } else if (arg == "--debug-graph-transients") {
            out.debugGraphTransients = true;
        } else if (arg == "--transient-test") {
            out.transientTest = true;
        } else if (arg == "--memory-stress") {
            if (!needCount(out.memoryStress)) return false;
        } else if (arg == "--simulate-pressure") {
            out.simulatePressure = true;
        } else if (arg == "--switch-every") {
            if (!needCount(out.switchEvery)) return false;
        } else if (arg == "--resize-every") {
            if (!needCount(out.resizeEvery)) return false;
        } else if (arg == "--capture") {
            const auto value = needValue();
            if (!value) return false;
            out.capturePath = *value;
        } else if (arg == "--report") {
            const auto value = needValue();
            if (!value) return false;
            out.reportPath = *value;
        } else if (arg == "--dump-graph") {
            const auto value = needValue();
            if (!value) return false;
            out.dumpGraphPath = *value;
        } else if (arg == "--debug-split-encoding") {
            out.debugSplitEncoding = true;
        } else if (arg == "--debug-async-compute") {
            out.debugAsyncCompute = true;
        } else if (arg == "--pipeline-archive") {
            if (!needString(out.pipelineArchivePath)) return false;
        } else if (arg == "--no-pipeline-archive") {
            out.noPipelineArchive = true;
        } else if (arg == "--harvest-pipelines") {
            if (!needString(out.harvestPipelinesPath)) return false;
        } else if (arg == "--pipeline-sync") {
            out.pipelineSync = true;
        } else if (arg == "--compile-qos") {
            const auto value = needValue();
            if (!value) return false;
            if (*value == "utility") {
                out.compileQosInteractive = false;
            } else if (*value == "interactive") {
                out.compileQosInteractive = true;
            } else {
                error = "--compile-qos: expected utility or interactive, got '" + std::string(*value) + "'";
                return false;
            }
        } else if (arg == "--pipeline-salt") {
            if (!needCount(out.pipelineSalt)) return false;
        } else if (arg == "--debug-mode") {
            if (!needCount(out.debugMode)) return false;
            if (out.debugMode > 2) {
                error = "--debug-mode: expected 0..2";
                return false;
            }
        } else if (arg == "--force-variant") {
            u32 variant = 0;
            if (!needCount(variant)) return false;
            out.forceVariant = variant; // range checked by the engine (generated table)
        } else if (arg == "--debug-flexible-pipelines") {
            out.debugFlexiblePipelines = true;
        } else if (arg == "--debug-pipeline-fallback") {
            out.debugPipelineFallback = true;
        } else if (arg == "--debug-compile-storm") {
            out.debugCompileStorm = true;
        } else if (arg == "--frame-trace") {
            if (!needString(out.frameTracePath)) return false;
        } else if (arg == "--shader-dir") {
            if (!needString(out.shaderDir)) return false;
        } else if (arg == "--debug-hot-reload") {
            if (!needString(out.debugHotReloadPath)) return false;
        } else if (arg == "--no-gpu-timing") {
            out.gpuTiming = false;
        } else if (arg == "--gpu-timing-unfused") {
            out.gpuTimingUnfused = true;
        } else if (arg == "--gpu-timing-serial") {
            out.gpuTimingSerial = true;
        } else if (arg == "--debug-gpu-cost") {
            if (!needCount(out.debugGpuCost)) return false;
        } else if (arg == "--gpu-capture") {
            out.gpuCapture = true;
        } else if (arg == "--gpu-capture-frame") {
            u32 frame = 0;
            if (!needCount(frame)) return false;
            out.gpuCaptureFrame = frame;
            out.gpuCapture = true;
        } else if (arg == "--gpu-capture-over") {
            const auto value = needValue();
            if (!value) return false;
            const std::string text(*value);
            char* end = nullptr;
            const float ms = std::strtof(text.c_str(), &end);
            if (text.empty() || end != text.c_str() + text.size() || !(ms > 0.0f)) {
                error = "--gpu-capture-over: expected a positive number of ms, got '" + std::string(*value) + "'";
                return false;
            }
            out.gpuCaptureOverMs = ms;
            out.gpuCapture = true;
        } else if (arg == "--gpu-capture-dir") {
            if (!needString(out.gpuCaptureDir)) return false;
        } else if (arg == "--gpu-capture-max") {
            if (!needCount(out.gpuCaptureMax)) return false;
        } else if (arg == "--overlay") {
            const auto value = needValue();
            if (!value) return false;
            bool found = false;
            for (const OverlayMode m : {OverlayMode::None, OverlayMode::Overdraw, OverlayMode::LightCount,
                                        OverlayMode::TileCost, OverlayMode::Timings}) {
                if (*value == overlayName(m)) {
                    out.overlay = m;
                    found = true;
                }
            }
            if (!found) {
                error = "--overlay: expected none, overdraw, lights, tilecost or timings, got '" +
                        std::string(*value) + "'";
                return false;
            }
        } else if (arg == "--graph-scenario") {
            u32 n = 0;
            if (!needCount(n)) return false;
            out.graphScenario = n;
        } else if (arg == "--graph-scenario-size") {
            const auto value = needValue();
            if (!value) return false;
            const size_t x = value->find('x');
            u32 w = 0, h = 0;
            if (x == std::string_view::npos || !parseU32(value->substr(0, x), w) || !parseU32(value->substr(x + 1), h) ||
                w < 64 || h < 64 || w > 16384 || h > 16384) {
                error = "--graph-scenario-size: expected WxH (64..16384), got '" + std::string(*value) + "'";
                return false;
            }
            out.scenarioWidth  = w;
            out.scenarioHeight = h;
        } else if (arg == "--graph-scenario-work") {
            const auto value = needValue();
            if (!value) return false;
            const std::string text(*value);
            char* end = nullptr;
            const float work = std::strtof(text.c_str(), &end);
            if (text.empty() || end != text.c_str() + text.size() || !(work > 0.0f) || work > 64.0f) {
                error = "--graph-scenario-work: expected a factor in (0, 64], got '" + text + "'";
                return false;
            }
            out.scenarioWork = work;
        } else if (arg == "--graph-scenario-wide") {
            out.scenarioWide = true;
        } else if (arg == "--graph-scenario-no-async") {
            out.scenarioAsync = false;
        } else if (arg == "--graph-opt") {
            const auto value = needValue();
            if (!value) return false;
            if (*value == "off") out.graphOpt = GraphOptMode::Off;
            else if (*value == "greedy") out.graphOpt = GraphOptMode::Greedy;
            else if (*value == "plan") out.graphOpt = GraphOptMode::Plan;
            else {
                error = "--graph-opt: expected off, greedy or plan, got '" + std::string(*value) + "'";
                return false;
            }
        } else if (arg == "--graph-plan") {
            if (!needString(out.graphPlanPath)) return false;
        } else if (arg == "--gpu-driven") {
            const auto value = needValue();
            if (!value) return false;
            if (*value == "off") out.gpuDriven = GpuDrivenMode::Off;
            else if (*value == "on") out.gpuDriven = GpuDrivenMode::On;
            else {
                error = "--gpu-driven: expected off or on, got '" + std::string(*value) + "'";
                return false;
            }
        } else if (arg == "--instances") {
            if (!needCount(out.sceneInstances)) return false;
            if (out.sceneInstances < 1 || out.sceneInstances > 16777216) {
                error = "--instances: expected 1..16777216";
                return false;
            }
        } else if (arg == "--scene-meshes") {
            if (!needCount(out.sceneMeshes)) return false;
            if (out.sceneMeshes < 1 || out.sceneMeshes > 1024) {
                error = "--scene-meshes: expected 1..1024";
                return false;
            }
        } else if (arg == "--dynamic-cpu") {
            float pct = 0.0f;
            if (!needFloat(pct)) return false;
            if (pct < 0.0f || pct > 100.0f) {
                error = "--dynamic-cpu: expected a percentage in 0..100";
                return false;
            }
            out.dynamicCpuPercent = pct;
        } else if (arg == "--churn") {
            if (!needCount(out.churn)) return false;
        } else if (arg == "--cull-distance") {
            if (!needFloat(out.cullDistance)) return false;
            if (out.cullDistance < 0.0f) {
                error = "--cull-distance: expected a distance >= 0 (0 = off)";
                return false;
            }
        } else if (arg == "--cull-min-pixels") {
            if (!needFloat(out.cullMinPixels)) return false;
            if (out.cullMinPixels < 0.0f) {
                error = "--cull-min-pixels: expected a size >= 0 (0 = off)";
                return false;
            }
        } else if (arg == "--debug-gpu-scene") {
            if (!needCount(out.debugGpuScene)) return false;
        } else if (arg == "--debug-gpu-scene-corrupt") {
            const auto value = needValue();
            if (!value) return false;
            if (*value == "delta") out.debugGpuSceneCorrupt = SceneCorruption::Delta;
            else if (*value == "plane") out.debugGpuSceneCorrupt = SceneCorruption::Plane;
            else if (*value == "command") out.debugGpuSceneCorrupt = SceneCorruption::Command;
            else if (*value == "touch") out.debugGpuSceneCorrupt = SceneCorruption::Touch;
            else {
                error = "--debug-gpu-scene-corrupt: expected delta, plane, command or touch, got '" +
                        std::string(*value) + "'";
                return false;
            }
        } else if (arg == "--geometry-path") {
            const auto value = needValue();
            if (!value) return false;
            if (*value == "indexed") out.geometryPath = GeometryPath::Indexed;
            else if (*value == "mesh") out.geometryPath = GeometryPath::Mesh;
            else {
                error = "--geometry-path: expected indexed or mesh, got '" + std::string(*value) + "'";
                return false;
            }
        } else if (arg == "--meshlet-cull") {
            const auto value = needValue();
            if (!value) return false;
            if (*value == "off") out.meshletCull = MeshletCull::Off;
            else if (*value == "frustum") out.meshletCull = MeshletCull::Frustum;
            else if (*value == "two-phase") out.meshletCull = MeshletCull::TwoPhase;
            else {
                error = "--meshlet-cull: expected off, frustum or two-phase, got '" + std::string(*value) + "'";
                return false;
            }
        } else if (arg == "--hiz-path") {
            const auto value = needValue();
            if (!value) return false;
            if (*value == "auto") out.hizPath = HiZPath::Auto;
            else if (*value == "compute") out.hizPath = HiZPath::Compute;
            else if (*value == "sampler") out.hizPath = HiZPath::Sampler;
            else {
                error = "--hiz-path: expected compute, sampler or auto, got '" + std::string(*value) + "'";
                return false;
            }
        } else if (arg == "--force-family") {
            const auto value = needValue();
            if (!value) return false;
            if (*value != "apple9") {
                error = "--force-family: expected apple9 (capabilities can only be removed), got '" +
                        std::string(*value) + "'";
                return false;
            }
            out.forceApple9 = true;
        } else if (arg == "--debug-meshlets") {
            if (!needCount(out.debugMeshlets)) return false;
        } else if (arg == "--debug-meshlets-corrupt") {
            const auto value = needValue();
            if (!value) return false;
            if (*value == "id") out.debugMeshletsCorrupt = MeshletCorruption::Id;
            else if (*value == "depth") out.debugMeshletsCorrupt = MeshletCorruption::Depth;
            else if (*value == "count") out.debugMeshletsCorrupt = MeshletCorruption::Count;
            else {
                error = "--debug-meshlets-corrupt: expected id, depth or count, got '" + std::string(*value) + "'";
                return false;
            }
        } else if (arg == "--debug-view") {
            const auto value = needValue();
            if (!value) return false;
            if (*value == "none") out.debugView = MeshletDebugView::None;
            else if (*value == "meshlets") out.debugView = MeshletDebugView::Meshlets;
            else if (*value == "cull") out.debugView = MeshletDebugView::Cull;
            else if (*value == "hiz") out.debugView = MeshletDebugView::HiZ;
            else {
                error = "--debug-view: expected none, meshlets, cull or hiz, got '" + std::string(*value) + "'";
                return false;
            }
        } else if (arg == "--debug-hiz-level") {
            if (!needCount(out.debugHiZLevel)) return false;
            if (out.debugHiZLevel > 15) {
                error = "--debug-hiz-level: expected 0..15";
                return false;
            }
        } else if (arg == "--meshlet-builder") {
            const auto value = needValue();
            if (!value) return false;
            if (*value == "standard") out.meshletSpatial = false;
            else if (*value == "spatial") out.meshletSpatial = true;
            else {
                error = "--meshlet-builder: expected standard or spatial, got '" + std::string(*value) + "'";
                return false;
            }
        } else if (arg == "--meshlet-max-vertices" || arg == "--meshlet-max-triangles") {
            u32& target = arg == "--meshlet-max-vertices" ? out.meshletMaxVertices : out.meshletMaxTriangles;
            if (!needCount(target)) return false;
            const u32 lo = arg == "--meshlet-max-vertices" ? 3u : 1u;
            if (target < lo || target > 128) {
                error = std::string(arg) + ": expected " + std::to_string(lo) + "..128 (mesh shader output limit)";
                return false;
            }
        } else if (arg == "--resolution") {
            const auto value = needValue();
            if (!value) return false;
            const size_t x = value->find('x');
            if (x == std::string_view::npos || !parseU32(value->substr(0, x), out.resolutionWidth) ||
                !parseU32(value->substr(x + 1), out.resolutionHeight) || out.resolutionWidth < 64 ||
                out.resolutionHeight < 64 || out.resolutionWidth > 8192 || out.resolutionHeight > 8192) {
                error = "--resolution: expected WxH in pixels (64..8192), got '" + std::string(*value) + "'";
                return false;
            }
        } else if (arg == "--culling-script") {
            out.cullingScript = true;
        } else if (arg == "--meshlet-min-pixels") {
            if (!needFloat(out.meshletMinPixels)) return false;
            if (out.meshletMinPixels < 0.0f) {
                error = "--meshlet-min-pixels: expected a size >= 0 (0 = off)";
                return false;
            }
        } else if (arg == "--meshlet-object") {
            const auto value = needValue();
            if (!value) return false;
            if (*value == "on") out.meshletObjectStage = true;
            else if (*value == "off") out.meshletObjectStage = false;
            else {
                error = "--meshlet-object: expected on or off, got '" + std::string(*value) + "'";
                return false;
            }
        } else if (arg == "--history-reset-every") {
            if (!needCount(out.historyResetEvery)) return false;
        } else if (arg == "--graph-no-alias") {
            out.graphNoAlias = true;
        } else if (arg == "--graph-remat-cost") {
            if (!needCount(out.graphRematCost)) return false;
        } else if (arg == "--graph-scenario-views") {
            if (!needCount(out.scenarioViews)) return false;
            if (out.scenarioViews < 1 || out.scenarioViews > 6) {
                error = "--graph-scenario-views: expected 1..6";
                return false;
            }
        } else if (arg == "--graph-remat" || arg == "--graph-order") {
            const auto value = needValue();
            if (!value) return false;
            std::vector<std::string>& list = arg == "--graph-remat" ? out.graphRemat : out.graphOrder;
            list.clear();
            size_t start = 0;
            while (start <= value->size()) {
                const size_t comma = value->find(',', start);
                const size_t end   = comma == std::string_view::npos ? value->size() : comma;
                if (end > start) list.emplace_back(value->substr(start, end - start));
                if (comma == std::string_view::npos) break;
                start = comma + 1;
            }
        } else {
            error = "unknown option " + std::string(arg);
            return false;
        }
    }
    // F6: combinations the engine cannot honour are refused, never ignored.
    if (out.geometryPath == GeometryPath::Mesh && out.gpuDriven != GpuDrivenMode::On) {
        error = "--geometry-path mesh needs --gpu-driven on (meshlet candidates come from the GPU instance cull)";
        return false;
    }
    if (out.forceApple9 && out.hizPath == HiZPath::Sampler) {
        error = "--hiz-path sampler needs Apple10 sampler min reduction: refused with --force-family apple9";
        return false;
    }
    if (out.geometryPath != GeometryPath::Mesh &&
        (out.debugMeshlets > 0 || out.debugMeshletsCorrupt != MeshletCorruption::None ||
         out.debugView != MeshletDebugView::None)) {
        error = "--debug-meshlets / --debug-meshlets-corrupt / --debug-view need --geometry-path mesh";
        return false;
    }
    return true;
}

} // namespace phosphor
