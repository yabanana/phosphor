#include "core/launch_options.h"
#include "testbench/lighting_validation.h"
#include "testbench/reflection_validation.h"

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
    case MeshletDebugView::RT:       return "rt";
    }
    return "none";
}

const char* rtCorruptionName(RtCorruption corruption) {
    switch (corruption) {
    case RtCorruption::None:      return "none";
    case RtCorruption::Transform: return "transform";
    case RtCorruption::Mask:      return "mask";
    case RtCorruption::Blas:      return "blas";
    }
    return "none";
}

const char* rtProbeName(RtProbe probe) {
    switch (probe) {
    case RtProbe::None:    return "none";
    case RtProbe::Primary: return "primary";
    case RtProbe::Shadow:  return "shadow";
    case RtProbe::AO:      return "ao";
    case RtProbe::Diffuse: return "diffuse";
    }
    return "none";
}

const char* rtProxyTransitionName(RtProxyTransition transition) {
    switch (transition) {
    case RtProxyTransition::None: return "none";
    case RtProxyTransition::Mask: return "mask";
    case RtProxyTransition::Emissive: return "emissive";
    case RtProxyTransition::Reassign: return "reassign";
    case RtProxyTransition::FullUpload: return "full-upload";
    }
    return "none";
}

const char* shadowModeName(ShadowMode m) {
    switch (m) { case ShadowMode::CSM: return "csm"; case ShadowMode::RT: return "rt"; default: return "off"; }
}
const char* directLightingModeName(DirectLightingMode m) {
    switch (m) { case DirectLightingMode::BruteForce: return "brute"; case DirectLightingMode::Clustered: return "clustered";
        case DirectLightingMode::ReSTIR: return "restir"; default: return "legacy"; }
}
const char* giModeName(GiMode m) {
    switch (m) { case GiMode::DDGI: return "ddgi"; case GiMode::Cache: return "cache";
        case GiMode::ReSTIR: return "restir"; default: return "off"; }
}

bool parseLaunchOptions(int argc, const char* const* argv, int benchCount,
                        LaunchOptions& out, std::string& error) {
    out = LaunchOptions{};
    bool rtSettingsSpecified = false;
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
        } else if (arg == "--rt") {
            const auto value = needValue();
            if (!value) return false;
            if (*value == "on") out.rtEnabled = true;
            else if (*value == "off") out.rtEnabled = false;
            else { error = "--rt: expected off or on"; return false; }
        } else if (arg == "--atmosphere" || arg == "--fog" || arg == "--clouds") {
            auto v=needValue();if(!v)return false;if(*v!="on"&&*v!="off"){error=std::string(arg)+": expected on or off";return false;}
            if(arg=="--atmosphere")out.atmosphere=*v=="on";else if(arg=="--fog")out.fog=*v=="on";else out.clouds=*v=="on";
        } else if (arg == "--cloud-full-rate") {out.cloudFullRate=true;
        } else if (arg == "--day-length") {
            if(!needFloat(out.atmoDayLength)||out.atmoDayLength<=0){error="--day-length: seconds >0";return false;}
        } else if (arg == "--start-hour") {
            if(!needFloat(out.atmoStartHour)||out.atmoStartHour<0||out.atmoStartHour>=24){error="--start-hour: expected0..24";return false;}
        } else if (arg == "--time-jump-every") {if(!needCount(out.timeJumpEveryN))return false;
        } else if (arg == "--planet-camera-height") {
            if(!needFloat(out.planetCameraHeight)||out.planetCameraHeight<0){error="--planet-camera-height: metres >=0";return false;}
        } else if (arg == "--reflections") {
            auto v=needValue();if(!v)return false;
            if(*v=="off")out.reflections=ReflectionMode::Off;else if(*v=="ssr")out.reflections=ReflectionMode::SSR;
            else if(*v=="rt")out.reflections=ReflectionMode::RT;else if(*v=="probes")out.reflections=ReflectionMode::Probes;
            else{error="--reflections: expected off, ssr, rt or probes";return false;}
        } else if (arg == "--ao") {
            auto v=needValue();if(!v)return false;
            if(*v=="off")out.ao=AoMode::Off;else if(*v=="gtao")out.ao=AoMode::GTAO;else if(*v=="rtao")out.ao=AoMode::RTAO;
            else{error="--ao: expected off, gtao or rtao";return false;}
        } else if (arg == "--lighting-denoise") {
            auto v=needValue();if(!v)return false;
            if(*v=="off")out.lightingDenoise=LightingDenoiseMode::Off;else if(*v=="custom")out.lightingDenoise=LightingDenoiseMode::Custom;
            else if(*v=="metalfx")out.lightingDenoise=LightingDenoiseMode::MetalFX;
            else{error="--lighting-denoise: expected off, custom or metalfx";return false;}
        } else if (arg == "--ao-radius") {
            if(!needFloat(out.aoRadius)||out.aoRadius<=0||out.aoRadius>100){error="--ao-radius: expected world metres in (0,100]";return false;}
        } else if (arg == "--reflection-samples") {
            if(!needCount(out.reflectionSamples)||out.reflectionSamples<1||out.reflectionSamples>8){error="--reflection-samples: expected1..8";return false;}
        } else if (arg == "--reflection-probe") {
            auto v=needValue();if(!v||v->empty()||v->starts_with("--")){error="--reflection-probe: expected cooked path";return false;}out.reflectionProbePath=*v;
        } else if (arg == "--reflection-capture-probe") {out.reflectionCaptureProbe=true;
        } else if (arg == "--shadows") {
            auto v = needValue(); if (!v) return false;
            if (*v == "off") out.shadows = ShadowMode::Off;
            else if (*v == "csm") out.shadows = ShadowMode::CSM;
            else if (*v == "rt") out.shadows = ShadowMode::RT;
            else { error = "--shadows: expected off, csm or rt"; return false; }
        } else if (arg == "--contact-shadows" || arg == "--shadow-cache") {
            auto v = needValue(); if (!v) return false;
            if (*v != "off" && *v != "on") { error = std::string(arg) + ": expected off or on"; return false; }
            (arg == "--contact-shadows" ? out.contactShadows : out.shadowCache) = *v == "on";
        } else if (arg == "--lighting") {
            auto v = needValue(); if (!v) return false;
            if (*v == "legacy") out.directLighting = DirectLightingMode::Legacy;
            else if (*v == "brute") out.directLighting = DirectLightingMode::BruteForce;
            else if (*v == "clustered") out.directLighting = DirectLightingMode::Clustered;
            else if (*v == "restir") out.directLighting = DirectLightingMode::ReSTIR;
            else { error = "--lighting: expected legacy, brute, clustered or restir"; return false; }
        } else if (arg == "--gi") {
            auto v = needValue(); if (!v) return false;
            if (*v == "off") out.gi = GiMode::Off;
            else if (*v == "ddgi") out.gi = GiMode::DDGI;
            else if (*v == "cache") out.gi = GiMode::Cache;
            else if (*v == "restir") out.gi = GiMode::ReSTIR;
            else { error = "--gi: expected off, ddgi, cache or restir"; return false; }
        } else if (arg == "--local-light-count") {
            if (!needCount(out.localLightCount) || out.localLightCount > 16384) { error = "--local-light-count: max16384"; return false; }
        } else if (arg == "--area-lights") {
            out.areaLights = true;
        } else if (arg == "--stationary-lights") {
            out.stationaryLights = true;
        } else if (arg == "--lighting-preset") {
            auto v = needValue(); if (!v) return false;
            if (*v != "full" && *v != "reduced") { error = "--lighting-preset: expected full or reduced"; return false; }
            out.reducedLighting = *v == "reduced";
        } else if (arg == "--shadow-map-size") {
            if (!needCount(out.shadowMapResolution)) return false;
        } else if (arg == "--lighting-seed") {
            if (!needCount(out.lightingSeed)) return false;
        } else if (arg == "--lighting-candidates") {
            if (!needCount(out.lightingCandidates)) return false;
        } else if (arg == "--lighting-spatial-samples") {
            if (!needCount(out.lightingSpatialSamples)) return false;
        } else if (arg == "--gi-probe-anchor") {
            auto v=needValue();if(!v)return false;std::string text(*v);const char* next=text.c_str();
            for(u32 i=0;i<3;++i){char* end=nullptr;out.giAnchor[i]=std::strtof(next,&end);if(end==next||!std::isfinite(out.giAnchor[i])||(i<2?*end!=',':*end!=0)){error="--gi-probe-anchor: expected finite X,Y,Z";return false;}next=end+1;}
            out.giProbeAnchor=true;
        } else if (arg == "--gi-spacing") {
            if(!needFloat(out.giSpacing)||out.giSpacing<=0){error="--gi-spacing: expected metres >0";return false;}
        } else if (arg == "--gi-grid") {
            auto v=needValue();if(!v)return false;size_t start=0;
            for(u32 i=0;i<3;++i){const auto end=v->find('x',start);const auto text=v->substr(start,end==std::string_view::npos?v->size()-start:end-start);
                if(!parseU32(text,out.giGrid[i])||out.giGrid[i]<2||out.giGrid[i]>32||(i<2?end==std::string_view::npos:end!=std::string_view::npos)){error="--gi-grid: expected 2..32 x 2..32 x 2..32";return false;}start=end+1;}
        } else if (arg == "--gi-rays") {
            if (!needCount(out.giRays)) return false;
        } else if (arg == "--reflection-scene") {
            auto v=needValue();if(!v)return false;if(!ReflectionValidation::validScenario(*v)){error="--reflection-scene: unknown scenario";return false;}out.reflectionScene=*v;
        } else if (arg == "--lighting-scene") {
            auto v=needValue();if(!v)return false;
            if(!LightingValidation::validScenario(*v)){error="--lighting-scene: unknown analytic scenario";return false;}out.lightingScene=*v;
        } else if (arg == "--denoised-fixture") {
            auto v=needValue();if(!v)return false;
            if(*v!="constant"&&*v!="impulse"&&*v!="channels"&&*v!="lifecycle"&&*v!="wide-hdr"){error="--denoised-fixture: expected constant, impulse, channels, lifecycle or wide-hdr";return false;}out.denoisedFixture=*v;
        } else if (arg == "--denoised-fixture-output") {
            auto v=needValue();if(!v||v->empty()||v->starts_with("--")){error="--denoised-fixture-output: expected output directory";return false;}out.denoisedFixtureOutput=*v;
        } else if (arg == "--denoised-fixture-pre-exposed") {
            out.denoisedFixturePreExposed=true;
        } else if (arg == "--fog-homogeneous") {
            out.fogHomogeneous=true;
        } else if (arg == "--volume-oracle") {
            auto v=needValue();if(!v||v->empty()||v->starts_with("--")){error="--volume-oracle: expected output directory";return false;}out.volumeOracle=*v;
        } else if (arg == "--debug-volume-corrupt") {
            auto v=needValue();if(!v)return false;
            if(*v=="units")out.debugVolumeCorrupt=1;else if(*v=="history")out.debugVolumeCorrupt=2;
            else if(*v=="light")out.debugVolumeCorrupt=3;else if(*v=="lut")out.debugVolumeCorrupt=4;
            else{error="--debug-volume-corrupt: expected units, history, light or lut";return false;}
        } else if (arg == "--debug-reflection-corrupt") {
            auto v=needValue();if(!v)return false;
            if(*v=="history")out.debugReflectionCorrupt=1;else if(*v=="motion")out.debugReflectionCorrupt=2;else if(*v=="normal")out.debugReflectionCorrupt=3;
            else{error="--debug-reflection-corrupt: expected history, motion or normal";return false;}
        } else if (arg == "--capture-linear" || arg == "--capture-linear-sequence") {
            auto v=needValue();if(!v || v->empty() || v->starts_with("--")){error=std::string(arg)+": expected path";return false;}
            (arg=="--capture-linear"?out.captureLinear:out.captureLinearSequence)=*v;
        } else if (arg == "--capture-linear-signal") {
            auto v=needValue();if(!v)return false;
            if(*v=="hdr")out.captureLinearSignal=0;
            else if(*v=="indirect-diffuse")out.captureLinearSignal=1;
            else if(*v=="direct")out.captureLinearSignal=2;
            else if(*v=="shadow")out.captureLinearSignal=3;
            else if(*v=="shadow-position")out.captureLinearSignal=4;
            else if(*v=="shadow-normal")out.captureLinearSignal=5;
            else if(*v=="specular")out.captureLinearSignal=6;else if(*v=="ao")out.captureLinearSignal=7;
            else{error="--capture-linear-signal: expected hdr, indirect-diffuse, direct, shadow, shadow-position, shadow-normal, specular or ao";return false;}
        } else if (arg == "--export-reference-frame") {
            if(!needCount(out.exportReferenceFrame))return false;
        } else if (arg == "--capture-linear-frame") {
            if(!needCount(out.captureLinearFrame))return false;
        } else if (arg == "--export-reference") {
            auto v = needValue(); if (!v || v->empty() || v->starts_with("--")) {
                error = "--export-reference: expected output directory"; return false;
            }
            out.exportReference = *v;
        } else if (arg == "--debug-lighting") {
            if (!needCount(out.debugLighting)) return false;
        } else if (arg == "--debug-gi-corrupt") {
            auto v=needValue();if(!v)return false;
            if(*v=="cache")out.debugGiCorrupt=1;else if(*v=="probe")out.debugGiCorrupt=2;else if(*v=="pdf")out.debugGiCorrupt=3;
            else{error="--debug-gi-corrupt: expected cache, probe or pdf";return false;}
        } else if (arg == "--debug-lighting-corrupt") {
            auto v = needValue(); if (!v) return false;
            if (*v == "bias") out.debugLightingCorrupt = 1;
            else if (*v == "caster") out.debugLightingCorrupt = 2;
            else if (*v == "history") out.debugLightingCorrupt = 3;
            else if (*v == "cache") out.debugLightingCorrupt = 4;
            else if (*v == "pdf") out.debugLightingCorrupt = 5;
            else if (*v == "light") out.debugLightingCorrupt = 6;
            else if (*v == "overflow") out.debugLightingCorrupt = 7;
            else { error = "--debug-lighting-corrupt: expected bias, caster, history, cache, pdf, light or overflow"; return false; }
        } else if (arg == "--rt-tlas-rebuild-every") {
            if (!needCount(out.rtTlasRebuildEvery)) return false;
            rtSettingsSpecified = true;
        } else if (arg == "--rt-proxy") {
            const auto value = needValue();
            if (!value) return false;
            if (*value == "manifest") out.rtProxyManifest = true;
            else if (*value == "off") out.rtProxyManifest = false;
            else { error = "--rt-proxy: expected off or manifest"; return false; }
        } else if (arg == "--rt-proxy-manifest") {
            const auto value = needValue();
            if (!value) return false;
            if (value->empty() || value->starts_with("--")) {
                error = "--rt-proxy-manifest requires a non-empty file path";
                return false;
            }
            out.rtProxyManifestPath = *value;
            rtSettingsSpecified = true;
        } else if (arg == "--debug-rt") {
            if (!needCount(out.debugRt)) return false;
            rtSettingsSpecified = true;
        } else if (arg == "--debug-rt-proxy-transition") {
            const auto value = needValue();
            if (!value) return false;
            if (*value == "mask") out.debugRtProxyTransition = RtProxyTransition::Mask;
            else if (*value == "emissive") out.debugRtProxyTransition = RtProxyTransition::Emissive;
            else if (*value == "reassign") out.debugRtProxyTransition = RtProxyTransition::Reassign;
            else if (*value == "full-upload") out.debugRtProxyTransition = RtProxyTransition::FullUpload;
            else { error = "--debug-rt-proxy-transition: expected mask, emissive, reassign or full-upload"; return false; }
            rtSettingsSpecified = true;
        } else if (arg == "--debug-rt-deform") {
            out.debugRtDeform = true;
            rtSettingsSpecified = true;
        } else if (arg == "--debug-rt-corrupt") {
            const auto value = needValue();
            if (!value) return false;
            if (*value == "transform") out.debugRtCorrupt = RtCorruption::Transform;
            else if (*value == "mask") out.debugRtCorrupt = RtCorruption::Mask;
            else if (*value == "blas") out.debugRtCorrupt = RtCorruption::Blas;
            else { error = "--debug-rt-corrupt: expected transform, mask or blas"; return false; }
            rtSettingsSpecified = true;
        } else if (arg == "--rt-probe") {
            const auto value = needValue();
            if (!value) return false;
            if (*value == "primary") out.rtProbe = RtProbe::Primary;
            else if (*value == "shadow") out.rtProbe = RtProbe::Shadow;
            else if (*value == "ao") out.rtProbe = RtProbe::AO;
            else if (*value == "diffuse") out.rtProbe = RtProbe::Diffuse;
            else { error = "--rt-probe: expected primary, shadow, ao or diffuse"; return false; }
            rtSettingsSpecified = true;
        } else if (arg == "--debug-view") {
            const auto value = needValue();
            if (!value) return false;
            if (*value == "none") out.debugView = MeshletDebugView::None;
            else if (*value == "meshlets") out.debugView = MeshletDebugView::Meshlets;
            else if (*value == "cull") out.debugView = MeshletDebugView::Cull;
            else if (*value == "hiz") out.debugView = MeshletDebugView::HiZ;
            else if (*value == "rt") out.debugView = MeshletDebugView::RT;
            else {
                error = "--debug-view: expected none, meshlets, cull, hiz or rt, got '" + std::string(*value) + "'";
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
        } else if (arg == "--capture-sequence") {
            if (!needString(out.captureSequence))
                return false;
        } else if (arg == "--capture-every") {
            if (!needCount(out.captureEvery))
                return false;
            if (out.captureEvery == 0) {
                error = "--capture-every must be positive";
                return false;
            }
        } else if (arg == "--temporal-script") {
            out.temporalScript = true;
        } else if (arg == "--exposure-script") {
            out.exposureScript = true;
        } else if (arg == "--offscreen") {
            out.offscreen = true;
        } else if (arg == "--frames-in-flight") {
            if (!needCount(out.framesInFlight))
                return false;
            if (out.framesInFlight < 1 || out.framesInFlight > 3) {
                error = "--frames-in-flight: expected 1..3";
                return false;
            }
        } else if (arg == "--debug-motion-scale") {
            if (!needFloat(out.debugMotionScale))
                return false;
            if (std::abs(out.debugMotionScale) > 2) {
                error = "--debug-motion-scale: expected -2..2";
                return false;
            }
        } else if (arg == "--debug-jitter-variant") {
            if (!needCount(out.jitterVariant))
                return false;
            if (out.jitterVariant > 3) {
                error = "--debug-jitter-variant: expected 0..3";
                return false;
            }
        } else if (arg == "--debug-feedback-delay-ms") {
            if (!needCount(out.feedbackDelayMs))
                return false;
            if (out.feedbackDelayMs > 1000) {
                error = "Feedback delay is limited to 1000 ms";
                return false;
            }
        } else if (arg == "--debug-feedback-error") {
            if (!needCount(out.feedbackFailFrame))
                return false;
        } else if (arg == "--output") {
            const auto value = needValue();
            if (!value)
                return false;
            out.post = true;
            if (*value == "sdr")
                out.displayOutput = 0;
            else if (*value == "edr")
                out.displayOutput = 1;
            else if (*value == "auto")
                out.displayOutput = 2;
            else {
                error = "--output: expected sdr, edr or auto";
                return false;
            }
        } else if (arg == "--tile-resolve") {
            out.tileResolve = true;
            out.visibility = true;
            out.materialBinning = false;
        } else if (arg == "--adaptive-shading") {
            out.adaptiveShading = true;
            out.visibility = true;
            out.materialBinning = false;
        } else if (arg == "--debug-adaptive-no-history") {
            out.debugAdaptiveNoHistory = true;
        } else if (arg == "--debug-upscaler-reset") {
            out.debugUpscalerReset = true;
            out.post = true;
        } else if (arg == "--debug-visibility") {
            out.debugVisibility = true;
            out.visibility = true;
        } else if (arg == "--debug-exposure-corrupt") {
            out.debugExposureCorrupt = true;
            out.post = true;
        } else if (arg == "--debug-motion-corrupt") {
            out.debugMotionCorrupt = true;
        } else if (arg == "--debug-guide-corrupt") {
            out.debugGuideCorrupt = true;
        } else if (arg == "--debug-history-corrupt") {
            out.debugHistoryCorrupt = true;
        } else if (arg == "--debug-post-curves") {
            out.debugPostCurves = true;
            out.post = true;
        } else if (arg == "--debug-post-curves-corrupt") {
            out.debugPostCurvesCorrupt = true;
        } else if (arg == "--debug-neutral-mip-bias") {
            out.debugNeutralMipBias = true;
            out.post = true;
        } else if (arg == "--post") {
            out.post = true;
        } else if (arg == "--upscaler") {
            const auto value = needValue();
            if (!value)
                return false;
            out.post = true;
            if (*value == "temporal")
                out.temporalUpscale = true;
            else if (*value == "native")
                out.temporalUpscale = false;
            else {
                error = "--upscaler: expected native or temporal";
                return false;
            }
        } else if (arg == "--auto-exposure") {
            out.autoExposure = true;
            out.post = true;
        } else if (arg == "--tonemap") {
            const auto value = needValue();
            if (!value)
                return false;
            out.post = true;
            if (*value == "aces")
                out.tonemap = 0;
            else if (*value == "agx")
                out.tonemap = 1;
            else if (*value == "custom")
                out.tonemap = 2;
            else {
                error = "--tonemap: expected aces, agx or custom";
                return false;
            }
        } else if (arg == "--reference-scale") {
            if (!needCount(out.referenceScale))
                return false;
            out.post = true;
            if (out.referenceScale != 1 && out.referenceScale != 2 && out.referenceScale != 4 &&
                out.referenceScale != 8) {
                error = "--reference-scale: expected 1, 2, 4 or 8";
                return false;
            }
        } else if (arg == "--settled-reference") {
            if (!needCount(out.settledReference))
                return false;
            if (out.settledReference < 16 || out.settledReference > 64) {
                error = "--settled-reference: expected 16..64 samples";
                return false;
            }
        } else if (arg == "--metalfx-mode") {
            std::string mode;
            if (!needString(mode))
                return false;
            if (mode == "isolated")
                out.isolatedMetalFX = true;
            else if (mode == "direct")
                out.isolatedMetalFX = false;
            else {
                error = "--metalfx-mode must be isolated or direct";
                return false;
            }
        } else if (arg == "--metalfx-resize-settle") {
            if (!needCount(out.metalfxResizeSettleFrames))
                return false;
            if (out.metalfxResizeSettleFrames > 120) {
                error = "--metalfx-resize-settle: expected 0..120 frames";
                return false;
            }
        } else if (arg == "--debug-frame-delay-ms") {
            if (!needCount(out.debugFrameDelayMs))
                return false;
            if (out.debugFrameDelayMs > 1000) {
                error = "Frame diagnostic delay must be at most 1000 ms";
                return false;
            }
        } else if (arg == "--debug-metalfx-worker-delay-ms") {
            if (!needCount(out.debugMetalFXWorkerDelayMs))
                return false;
            if (out.debugMetalFXWorkerDelayMs > 2000) {
                error = "Worker delay must be at most 2000 ms";
                return false;
            }
        } else if (arg == "--debug-metalfx-worker-crash") {
            if (!needCount(out.debugMetalFXWorkerCrash))
                return false;
        } else if (arg == "--render-scale") {
            if (!needFloat(out.renderScale))
                return false;
            out.post = true;
            if (out.renderScale < 0.5f || out.renderScale > 1.0f) {
                error = "--render-scale: expected 0.5..1";
                return false;
            }
        } else if (arg == "--sharpen") {
            if (!needFloat(out.sharpening))
                return false;
            out.post = true;
            if (out.sharpening < 0 || out.sharpening > 1) {
                error = "--sharpen: expected 0..1";
                return false;
            }
        } else if (arg == "--tone-white") {
            if (!needFloat(out.toneWhite))
                return false;
            out.post = true;
            if (out.toneWhite < 0.1f || out.toneWhite > 100) {
                error = "--tone-white: expected 0.1..100";
                return false;
            }
        } else if (arg == "--dynamic-resolution") {
            out.dynamicResolution = true;
            out.post = true;
        } else if (arg == "--drs-budget") {
            if (!needFloat(out.drsBudget))
                return false;
            out.post = true;
            if (out.drsBudget < 0.1f || out.drsBudget > 100) {
                error = "--drs-budget: expected 0.1..100 ms";
                return false;
            }
        } else if (arg == "--resolution-script") {
            if (!needCount(out.resolutionScript))
                return false;
            out.post = true;
        } else if (arg == "--temporal-views") {
            if (!needCount(out.temporalViews))
                return false;
            out.post = true;
            if (out.temporalViews < 1 || out.temporalViews > 4) {
                error = "--temporal-views: expected 1..4";
                return false;
            }
        } else if (arg == "--render-path") {
            const auto value = needValue();
            if (!value)
                return false;
            if (*value == "forward")
                out.visibility = false;
            else if (*value == "visibility")
                out.visibility = true;
            else {
                error = "--render-path: expected forward or visibility";
                return false;
            }
        } else if (arg == "--material-binning") {
            const auto value = needValue();
            if (!value)
                return false;
            if (*value == "on")
                out.materialBinning = true;
            else if (*value == "off")
                out.materialBinning = false;
            else {
                error = "--material-binning: expected on or off";
                return false;
            }
        } else if (arg == "--scene") {
            if (!needString(out.scenePath))
                return false;
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
        } else if (arg == "--meshlet-triangle-cull") {
            const auto value = needValue();
            if (!value) return false;
            if (*value == "on") out.meshletTriangleCull = true;
            else if (*value == "off") out.meshletTriangleCull = false;
            else {
                error = "--meshlet-triangle-cull: expected on or off, got '" + std::string(*value) + "'";
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
    if ((out.debugMotionCorrupt || out.debugExposureCorrupt || out.debugGuideCorrupt || out.debugHistoryCorrupt) &&
        !out.debugVisibility) {
        error = "Guide/motion/exposure corruption requires --debug-visibility";
        return false;
    }
    if ((out.debugMetalFXWorkerCrash || out.debugMetalFXWorkerDelayMs) &&
        (!out.temporalUpscale || !out.isolatedMetalFX)) {
        error = "Worker crash control requires isolated temporal upscaling";
        return false;
    }
    if (!out.captureSequence.empty() && !out.capturePath.empty()) {
        error = "Choose --capture or --capture-sequence, not both";
        return false;
    }
    if (out.offscreen && !out.benchmark()) {
        error = "--offscreen requires --frames";
        return false;
    }
    if (out.referenceScale > 1 && (out.temporalUpscale || out.dynamicResolution || out.resolutionScript)) {
        error = "Supersampled reference requires fixed native reconstruction";
        return false;
    }
    if (out.settledReference > 1 &&
        (!out.temporalUpscale || !out.fixedTimestep || out.warmup != 0 || out.dynamicResolution ||
         out.resolutionScript || out.temporalViews != 1 || !out.benchmark())) {
        error =
            "Settled reference requires fixed timestep, temporal upscaling, one view, zero warmup and fixed resolution";
        return false;
    }
    if(out.lightingDenoise==LightingDenoiseMode::MetalFX || out.atmosphere || out.fog || out.clouds || !out.denoisedFixture.empty())out.post=true;
    if (out.post)
        out.visibility = true;
    if (out.debugPostCurvesCorrupt && !out.debugPostCurves) {
        error = "--debug-post-curves-corrupt requires --debug-post-curves";
        return false;
    }
    if (out.tileResolve && out.adaptiveShading) {
        error = "Tile and adaptive shading are separate experiments";
        return false;
    }
    if (out.debugAdaptiveNoHistory && !out.adaptiveShading) {
        error = "--debug-adaptive-no-history requires --adaptive-shading";
        return false;
    }
    if (out.visibility) {
        if (out.graphScenario) {
            error = "Visibility is unavailable in synthetic graph scenarios";
            return false;
        }
        out.geometryPath = GeometryPath::Mesh;
        if (out.meshletMaxTriangles > 128) {
            error = "Visibility IDs allow at most 128 triangles per meshlet";
            return false;
        }
    }
    const bool lighting = out.shadows != ShadowMode::Off || out.directLighting != DirectLightingMode::Legacy || out.gi != GiMode::Off ||
                          out.reflections!=ReflectionMode::Off || out.ao!=AoMode::Off || out.lightingDenoise!=LightingDenoiseMode::Off || out.atmosphere || out.fog || out.clouds;
    if (lighting && (!out.visibility || out.tileResolve || out.adaptiveShading || out.graphScenario || out.memoryStress || out.transientTest)) {
        error = "F10-F12 lighting requires --render-path visibility with generic/binned resolve and a scene"; return false;
    }
    if(!out.denoisedFixture.empty()&&(out.denoisedFixtureOutput.empty()||!out.benchmark()||out.temporalUpscale||out.atmosphere||out.fog||out.clouds||out.lightingDenoise!=LightingDenoiseMode::Off||out.reflections!=ReflectionMode::Off||out.ao!=AoMode::Off||!out.captureLinear.empty()||!out.captureLinearSequence.empty())){error="SDK fixture requires --frames and --denoised-fixture-output, with native post and exclusive fixture inputs";return false;}
    if(out.denoisedFixture.empty()&&(!out.denoisedFixtureOutput.empty()||out.denoisedFixturePreExposed)){error="SDK fixture controls require --denoised-fixture";return false;}
    if((!out.volumeOracle.empty()||out.debugVolumeCorrupt) && (!(out.atmosphere||out.fog||out.clouds)||!out.debugLighting)){error="Volume oracle/corruption requires F14 and --debug-lighting N";return false;}
    if(out.fogHomogeneous&&(!out.fog||out.volumeOracle.empty())){error="Homogeneous fog requires --fog on and --volume-oracle";return false;}
    if(out.debugVolumeCorrupt==2&&(!out.fog&&(!out.clouds||out.cloudFullRate))){error="Volume history control requires fog or reconstructed clouds";return false;}
    if(out.debugVolumeCorrupt==3&&!out.fog){error="Volume light control requires --fog on";return false;}
    if(out.debugReflectionCorrupt&&(!out.debugLighting||!(out.reflections!=ReflectionMode::Off||out.ao!=AoMode::Off||out.lightingDenoise!=LightingDenoiseMode::Off))){error="Reflection control requires F13 and --debug-lighting N";return false;}
    if(out.debugReflectionCorrupt==1&&(out.lightingDenoise!=LightingDenoiseMode::Custom||!(out.reflections!=ReflectionMode::Off||out.ao!=AoMode::Off||out.directLighting!=DirectLightingMode::Legacy||out.gi!=GiMode::Off))){error="Reflection history control requires custom denoise and an active signal";return false;}
    if(out.cloudFullRate&&!out.clouds){error="--cloud-full-rate requires --clouds on";return false;}
    if((out.atmosphere||out.fog||out.clouds) && out.temporalUpscale){error="Physical atmosphere HDR requires native or F13 denoised reconstruction";return false;}
    if(out.planetCameraHeight>=0&&!out.atmosphere&&!out.fog&&!out.clouds){error="Planetary camera control requires F14";return false;}
    if((out.reflections==ReflectionMode::RT || out.ao==AoMode::RTAO) && !out.rtEnabled){error="RT reflections/AO/probe capture require --rt on";return false;}
    if(out.lightingDenoise==LightingDenoiseMode::MetalFX && out.temporalUpscale){error="Denoised and standard temporal reconstruction are separate paths";return false;}
    if((!out.reflectionProbePath.empty() || out.reflectionCaptureProbe) && out.reflections==ReflectionMode::Off){error="Probe controls require --reflections";return false;}
    if ((out.shadows == ShadowMode::RT || out.directLighting != DirectLightingMode::Legacy || out.gi != GiMode::Off) && !out.rtEnabled) {
        error = "RT sun, local visibility and GI require --rt on"; return false;
    }
    if(!out.reflectionScene.empty() && (!out.bench||*out.bench!=5||!out.lightingScene.empty())){error="--reflection-scene requires --bench 6 and exclusive scene fixture";return false;}
    if(out.captureLinearSignal==6 && out.reflections==ReflectionMode::Off){error="Specular capture requires --reflections";return false;}
    if(out.captureLinearSignal==7 && out.ao==AoMode::Off){error="AO capture requires --ao";return false;}
    if(!out.lightingScene.empty() && (!out.bench || *out.bench!=5)){error="--lighting-scene requires --bench 6";return false;}
    if(!out.captureLinear.empty() && !out.captureLinearSequence.empty()){error="Choose single linear capture or linear sequence";return false;}
    if((!out.captureLinear.empty() || !out.captureLinearSequence.empty()) && (!out.visibility || !out.benchmark())){error="Linear capture requires --render-path visibility and --frames";return false;}
    if((out.giProbeAnchor||out.giSpacing||out.giGrid[0])&&out.gi==GiMode::Off){error="GI volume controls require --gi";return false;}
    if(out.debugGiCorrupt && (!out.debugLighting || out.gi==GiMode::Off || (out.debugGiCorrupt==3 && out.gi==GiMode::DDGI))){error="GI corruption requires --debug-lighting and an applicable GI mode";return false;}
    if(out.captureLinearSignal==1 && out.gi==GiMode::Off){error="Indirect linear signal requires --gi";return false;}
    if(out.captureLinearSignal==2 && out.directLighting==DirectLightingMode::Legacy){error="Direct linear signal requires --lighting";return false;}
    if(out.captureLinearSignal==3 && out.shadows==ShadowMode::Off){error="Shadow linear signal requires --shadows csm|rt";return false;}
    if((out.captureLinearSignal==4||out.captureLinearSignal==5) && !lighting){error="Receiver guide capture requires a lighting producer";return false;}
    if((!out.exportReference.empty() && out.exportReferenceFrame>=u64(out.warmup)+out.frames) ||
       (!out.captureLinear.empty() && out.captureLinearFrame>=u64(out.warmup)+out.frames)){error="Capture/export frame is outside the requested run";return false;}
    if(!out.exportReference.empty() && !out.benchmark()){error="Reference export requires --frames";return false;}
    if(lighting && out.debugRtDeform){error="F9 diagnostic deformation does not publish raster bounds for lighting";return false;}
    if(out.debugLightingCorrupt==7 && out.directLighting==DirectLightingMode::Legacy){error="Finite-overflow control requires --lighting";return false;}
    if((out.debugLightingCorrupt==5 || out.debugLightingCorrupt==6) && out.directLighting!=DirectLightingMode::ReSTIR){error="PDF/light negative controls require --lighting restir";return false;}
    if (out.shadowCache && out.shadows != ShadowMode::CSM) { error = "--shadow-cache on requires --shadows csm"; return false; }
    if ((out.contactShadows || out.shadowCache) && out.shadows == ShadowMode::Off) {
        error = "Contact/cache requires an enabled shadow signal"; return false;
    }
    if (out.shadowMapResolution < 128 || out.shadowMapResolution > 8192 ||
        (out.shadowMapResolution & (out.shadowMapResolution - 1)) || out.lightingCandidates < 1 ||
        out.lightingCandidates > 8 || out.lightingSpatialSamples > 4 || out.giRays < 8 || out.giRays > 512) {
        error = "Lighting preset exceeds bounded map/candidate/spatial/ray limits"; return false;
    }
    if (out.gi != GiMode::Off && out.directLighting == DirectLightingMode::Legacy) {
        error = "GI requires --lighting clustered, brute or restir for emissive direct transport"; return false;
    }
    if (out.gi == GiMode::ReSTIR && (out.forceApple9 || out.reducedLighting)) {
        error = "ReSTIR GI requires the full T2 preset; select DDGI on the reduced path"; return false;
    }
    if (out.debugLightingCorrupt && (!lighting || !out.debugLighting)) {
        error = "Lighting corruption requires an enabled lighting path and --debug-lighting N"; return false;
    }
    if (out.debugLighting && !lighting) { error = "--debug-lighting requires F10-F12 lighting"; return false; }
    // F9: diagnostics must never appear accepted when their graph is absent.
    if (!out.rtEnabled && (rtSettingsSpecified || out.rtProxyManifest || out.debugView == MeshletDebugView::RT)) {
        error = "RT settings / --debug-view rt require --rt on";
        return false;
    }
    if (!out.rtProxyManifestPath.empty() && !out.rtProxyManifest) {
        error = "--rt-proxy-manifest requires --rt-proxy manifest";
        return false;
    }
    if (out.debugRtProxyTransition != RtProxyTransition::None) {
        if (!out.rtProxyManifest || out.debugRt == 0 || out.switchEvery || out.debugRtDeform ||
            out.debugRtCorrupt != RtCorruption::None) {
            error = "--debug-rt-proxy-transition requires --rt-proxy manifest and --debug-rt N > 0, without switch/deform/corrupt";
            return false;
        }
        // The transition happens after eight rendered frames. Preserve an
        // interactive mode, but reject short benchmark runs including warmup.
        if (out.benchmark() && u64(out.frames) + out.warmup < 24) {
            error = "--debug-rt-proxy-transition needs at least 24 total frames including warmup";
            return false;
        }
    }
    if (out.debugRtDeform && (out.debugRt == 0 || out.debugView != MeshletDebugView::RT || out.rtProxyManifest)) {
        error = "--debug-rt-deform requires --debug-rt N with N > 0, --debug-view rt and --rt-proxy off";
        return false;
    }
    if (out.debugRtCorrupt != RtCorruption::None && out.debugRt == 0) {
        error = "--debug-rt-corrupt requires --debug-rt N with N > 0";
        return false;
    }
    if (out.rtEnabled && (out.graphScenario || out.memoryStress > 0 || out.transientTest)) {
        error = "--rt on is unavailable with --graph-scenario, --memory-stress or --transient-test";
        return false;
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
         (out.debugView != MeshletDebugView::None && out.debugView != MeshletDebugView::RT))) {
        error = "--debug-meshlets / --debug-meshlets-corrupt / --debug-view need --geometry-path mesh";
        return false;
    }
    return true;
}

} // namespace phosphor
