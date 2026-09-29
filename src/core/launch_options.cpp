#include "core/launch_options.h"

#include <charconv>
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
        } else {
            error = "unknown option " + std::string(arg);
            return false;
        }
    }
    return true;
}

} // namespace phosphor
