#include "core/launch_options.h"

#include <charconv>
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
        auto needString = [&](std::string& target) {
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
    return true;
}

} // namespace phosphor
