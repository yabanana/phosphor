#include "core/launch_options.h"
#include "diagnostics/bench_report.h"

#include <doctest/doctest.h>

#include <cctype>
#include <cmath>
#include <string>
#include <vector>

using namespace phosphor;

namespace {

bool parse(std::vector<const char*> args, LaunchOptions& out, std::string& error) {
    args.insert(args.begin(), "phosphor");
    return parseLaunchOptions(static_cast<int>(args.size()), args.data(), 8, out, error);
}

} // namespace

TEST_CASE("launch options: defaults are interactive") {
    LaunchOptions o;
    std::string err;
    REQUIRE(parse({}, o, err));
    CHECK_FALSE(o.bench.has_value());
    CHECK_FALSE(o.benchmark());
    CHECK(o.vsync);
    CHECK(o.ui);
    CHECK(o.warmup == 120);
}

TEST_CASE("launch options: benchmark switches") {
    LaunchOptions o;
    std::string err;
    REQUIRE(parse({"--bench", "3", "--frames", "600", "--warmup", "10", "--no-vsync", "--no-ui",
                   "--fixed-timestep", "--switch-every", "20", "--simulate-pressure", "--capture", "out.png",
                   "--report", "r.json", "--dump-graph", "g.dot", "--debug-split-encoding", "--debug-async-compute", "--resize-every", "7"},
                  o, err));
    CHECK(o.bench == 2); // 1-based on the command line
    CHECK(o.frames == 600);
    CHECK(o.benchmark());
    CHECK(o.warmup == 10);
    CHECK_FALSE(o.vsync);
    CHECK_FALSE(o.ui);
    CHECK(o.fixedTimestep);
    CHECK(o.switchEvery == 20);
    CHECK(o.simulatePressure);
    CHECK(o.capturePath == "out.png");
    CHECK(o.reportPath == "r.json");
    CHECK(o.dumpGraphPath == "g.dot");
    CHECK(o.debugSplitEncoding);
    CHECK(o.debugAsyncCompute);
    CHECK(o.resizeEvery == 7);
}

TEST_CASE("launch options: --debug-graph-transients") {
    LaunchOptions o;
    std::string err;
    REQUIRE(parse({}, o, err));
    CHECK_FALSE(o.debugGraphTransients);
    REQUIRE(parse({"--bench", "1", "--frames", "300", "--debug-graph-transients"}, o, err));
    CHECK(o.debugGraphTransients);
    CHECK(o.frames == 300);
}

TEST_CASE("launch options: memory stress") {
    LaunchOptions o;
    std::string err;
    REQUIRE(parse({"--memory-stress", "10000", "--transient-test", "--inject-input"}, o, err));
    CHECK(o.injectInput);
    CHECK(o.memoryStress == 10000);
    CHECK(o.transientTest);
    CHECK_FALSE(o.benchmark());
}

TEST_CASE("launch options: F3 pipeline switches") {
    LaunchOptions o;
    std::string err;
    REQUIRE(parse({}, o, err));
    CHECK(o.pipelineArchivePath.empty());
    CHECK_FALSE(o.noPipelineArchive);
    CHECK_FALSE(o.pipelineSync);
    CHECK_FALSE(o.compileQosInteractive);
    CHECK(o.pipelineSalt == 0);
    REQUIRE(parse({"--pipeline-archive", "a.metallib", "--no-pipeline-archive", "--harvest-pipelines", "p.mtl4-json",
                   "--pipeline-sync", "--debug-compile-storm", "--debug-pipeline-fallback", "--debug-flexible-pipelines", "--debug-mode", "2", "--force-variant", "41", "--compile-qos", "interactive", "--pipeline-salt", "42", "--frame-trace", "t.csv",
                   "--shader-dir", "shaders", "--debug-hot-reload", "probe.metallib"},
                  o, err));
    CHECK(o.pipelineArchivePath == "a.metallib");
    CHECK(o.noPipelineArchive);
    CHECK(o.harvestPipelinesPath == "p.mtl4-json");
    CHECK(o.pipelineSync);
    CHECK(o.debugCompileStorm);
    CHECK(o.debugPipelineFallback);
    CHECK(o.debugFlexiblePipelines);
    CHECK(o.debugMode == 2);
    CHECK(o.forceVariant == 41u);
    CHECK(o.compileQosInteractive);
    CHECK(o.pipelineSalt == 42);
    CHECK(o.frameTracePath == "t.csv");
    CHECK(o.shaderDir == "shaders");
    CHECK(o.debugHotReloadPath == "probe.metallib");
    REQUIRE(parse({"--compile-qos", "utility"}, o, err));
    CHECK_FALSE(o.compileQosInteractive);
    CHECK_FALSE(parse({"--compile-qos", "background"}, o, err));
    CHECK(err.find("--compile-qos") != std::string::npos);
    CHECK_FALSE(parse({"--pipeline-archive"}, o, err));
    CHECK_FALSE(parse({"--debug-mode", "3"}, o, err));
    REQUIRE(parse({}, o, err));
    CHECK_FALSE(o.forceVariant.has_value());
}

TEST_CASE("launch options: F4 observability switches") {
    LaunchOptions o;
    std::string err;
    REQUIRE(parse({}, o, err));
    CHECK(o.gpuTiming);
    CHECK_FALSE(o.gpuTimingUnfused);
    CHECK(o.debugGpuCost == 0);
    CHECK_FALSE(o.gpuCapture);
    CHECK_FALSE(o.gpuCaptureFrame.has_value());
    CHECK(o.gpuCaptureOverMs == 0.0f);
    CHECK(o.gpuCaptureDir == "captures");
    CHECK(o.gpuCaptureMax == 1);
    CHECK(o.overlay == OverlayMode::None);

    CHECK_FALSE(o.gpuTimingSerial);
    REQUIRE(parse({"--no-gpu-timing", "--gpu-timing-unfused", "--gpu-timing-serial", "--debug-gpu-cost", "2000"}, o,
                  err));
    CHECK_FALSE(o.gpuTiming);
    CHECK(o.gpuTimingUnfused);
    CHECK(o.gpuTimingSerial);
    CHECK(o.debugGpuCost == 2000);

    REQUIRE(parse({"--gpu-capture-frame", "30", "--gpu-capture-dir", "out", "--gpu-capture-max", "3"}, o, err));
    CHECK(o.gpuCapture);
    REQUIRE(o.gpuCaptureFrame.has_value());
    CHECK(*o.gpuCaptureFrame == 30);
    CHECK(o.gpuCaptureDir == "out");
    CHECK(o.gpuCaptureMax == 3);
    REQUIRE(parse({"--gpu-capture-over", "12.5"}, o, err));
    CHECK(o.gpuCapture);
    CHECK(o.gpuCaptureOverMs == doctest::Approx(12.5f));
    REQUIRE(parse({"--gpu-capture"}, o, err));
    CHECK(o.gpuCapture);
    CHECK_FALSE(parse({"--gpu-capture-over", "0"}, o, err));
    CHECK_FALSE(parse({"--gpu-capture-over", "fast"}, o, err));
    CHECK_FALSE(parse({"--gpu-capture-over", "3ms"}, o, err));

    for (const OverlayMode m : {OverlayMode::None, OverlayMode::Overdraw, OverlayMode::LightCount,
                                OverlayMode::TileCost, OverlayMode::Timings}) {
        REQUIRE(parse({"--overlay", overlayName(m)}, o, err));
        CHECK(o.overlay == m);
    }
    CHECK_FALSE(parse({"--overlay", "heat"}, o, err));
    CHECK(err.find("--overlay") != std::string::npos);
}

TEST_CASE("launch options: errors") {
    LaunchOptions o;
    std::string err;
    CHECK_FALSE(parse({"--bench", "0"}, o, err));
    CHECK_FALSE(parse({"--bench", "9"}, o, err));
    CHECK_FALSE(parse({"--frames", "-5"}, o, err));
    CHECK_FALSE(parse({"--frames", "12x"}, o, err));
    CHECK_FALSE(parse({"--frames"}, o, err));
    CHECK_FALSE(parse({"--bogus"}, o, err));
    CHECK(err.find("--bogus") != std::string::npos);
}

TEST_CASE("launch options: macOS-injected arguments are ignored") {
    LaunchOptions o;
    std::string err;
    CHECK(parse({"-NSDocumentRevisionsDebugMode", "YES", "--bench", "1"}, o, err));
    CHECK(o.bench == 0);
}

TEST_CASE("bench report: nearest-rank statistics") {
    std::vector<float> v;
    for (int i = 1; i <= 100; ++i) v.push_back(static_cast<float>(101 - i)); // 100..1, unsorted
    const TimingSummary s = summarize(v);
    CHECK(s.min == doctest::Approx(1.0f));
    CHECK(s.max == doctest::Approx(100.0f));
    CHECK(s.mean == doctest::Approx(50.5f));
    CHECK(s.p50 == doctest::Approx(50.0f));
    CHECK(s.p99 == doctest::Approx(99.0f));

    const TimingSummary empty = summarize({});
    CHECK(empty.mean == 0.0f);
    CHECK(empty.p99 == 0.0f);

    const TimingSummary one = summarize({4.0f});
    CHECK(one.p50 == doctest::Approx(4.0f));
    CHECK(one.p99 == doctest::Approx(4.0f));
}

TEST_CASE("bench report: samples, fps and JSON") {
    std::vector<FrameSample> samples(4, FrameSample{8.0f, 2.0f, 5.0f, 1.0f});
    BenchReport r;
    r.bench  = "Torus \"Demo\"";
    r.device = "Test GPU";
    summarizeSamples(samples, r);
    CHECK(r.frames == 4);
    CHECK(r.fps == doctest::Approx(125.0f));
    CHECK(r.cpuMs.mean == doctest::Approx(2.0f));
    CHECK(r.gpuMs.p99 == doctest::Approx(5.0f));
    CHECK(r.waitMs.mean == doctest::Approx(1.0f));

    const std::string json = reportToJson(r);
    CHECK(json.find("\"fps\": 125.00") != std::string::npos);
    CHECK(json.find("\"gpu_allocations\": 0") != std::string::npos);
    CHECK(json.find("Torus \\\"Demo\\\"") != std::string::npos);
    CHECK(json.find("\"gpu_ms\": {\"mean\": 5.0000") != std::string::npos);
    CHECK(json.find("\"wait_ms\": {\"mean\": 1.0000") != std::string::npos);
    CHECK(formatReportLine(r).find("125.0 fps") != std::string::npos);
}

namespace {

// Minimal strict JSON syntax checker (objects, arrays, strings with escapes,
// numbers, true/false/null) -- enough to prove reportToJson stays parseable
// by jq.
struct JsonChecker {
    const std::string& s;
    size_t             i = 0;

    void ws() {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\t' || s[i] == '\r')) ++i;
    }
    bool lit(const char* word) {
        const std::string w = word;
        if (s.compare(i, w.size(), w) != 0) return false;
        i += w.size();
        return true;
    }
    bool str() {
        if (i >= s.size() || s[i] != '"') return false;
        ++i;
        while (i < s.size() && s[i] != '"') {
            if (static_cast<unsigned char>(s[i]) < 0x20) return false;
            if (s[i] == '\\') {
                ++i;
                if (i >= s.size()) return false;
                if (s[i] == 'u') {
                    for (int k = 0; k < 4; ++k) {
                        ++i;
                        if (i >= s.size() || !std::isxdigit(static_cast<unsigned char>(s[i]))) return false;
                    }
                } else if (std::string("\"\\/bfnrt").find(s[i]) == std::string::npos) {
                    return false;
                }
            }
            ++i;
        }
        if (i >= s.size()) return false;
        ++i;
        return true;
    }
    bool num() {
        const size_t b = i;
        if (i < s.size() && s[i] == '-') ++i;
        while (i < s.size() && (std::isdigit(static_cast<unsigned char>(s[i])) || s[i] == '.' || s[i] == 'e' ||
                                s[i] == 'E' || s[i] == '+' || s[i] == '-')) {
            ++i;
        }
        return i > b;
    }
    bool value() {
        ws();
        if (i >= s.size()) return false;
        if (s[i] == '{') {
            ++i;
            ws();
            if (i < s.size() && s[i] == '}') { ++i; return true; }
            for (;;) {
                ws();
                if (!str()) return false;
                ws();
                if (i >= s.size() || s[i++] != ':') return false;
                if (!value()) return false;
                ws();
                if (i < s.size() && s[i] == ',') { ++i; continue; }
                if (i < s.size() && s[i] == '}') { ++i; return true; }
                return false;
            }
        }
        if (s[i] == '[') {
            ++i;
            ws();
            if (i < s.size() && s[i] == ']') { ++i; return true; }
            for (;;) {
                if (!value()) return false;
                ws();
                if (i < s.size() && s[i] == ',') { ++i; continue; }
                if (i < s.size() && s[i] == ']') { ++i; return true; }
                return false;
            }
        }
        if (s[i] == '"') return str();
        if (s[i] == 't') return lit("true");
        if (s[i] == 'f') return lit("false");
        if (s[i] == 'n') return lit("null");
        return num();
    }
    bool document() {
        if (!value()) return false;
        ws();
        return i == s.size();
    }
};

bool validJson(const std::string& s) { return JsonChecker{s}.document(); }

BenchReport timedReport() {
    BenchReport r;
    r.bench            = "Forward \"PBR\"\n\tscene\\";
    r.device           = "Apple M5 Max";
    r.width            = 1920;
    r.height           = 1080;
    r.frames           = 300;
    r.fps              = 120.0f;
    r.gpuTiming        = true;
    r.gpuTimingUnfused = true;
    PassReport a;
    a.name      = "Forward + Overlay";
    a.queue     = "graphics";
    a.fused     = true;
    a.passes    = {"Forward", "Overlay"};
    a.shaders   = {"fwd_v", "fwd_f"};
    a.dramBytes = 123456;
    a.gpuMs     = {1.5f, 1.0f, 1.4f, 2.0f, 2.5f};
    PassReport b;
    b.name  = "Async \"reduce\"";
    b.queue = "async";
    r.passes          = {a, b};
    r.gpuPassSumMs    = {2.0f, 1.0f, 2.0f, 3.0f, 3.5f};
    r.gpuFrameSpanMs  = {2.5f, 1.5f, 2.5f, 3.5f, 4.0f};
    return r;
}

} // namespace

TEST_CASE("bench report: schema v3 without GPU timing keeps v1 fields and stays valid JSON") {
    BenchReport r;
    r.bench  = "Torus";
    r.device = "GPU";
    r.pipelinesJson = "{\"hits\": 3}";
    const std::string json = reportToJson(r);
    CHECK_MESSAGE(validJson(json), json);
    CHECK(json.find("\"schema_version\": 5") != std::string::npos);
    CHECK(json.find("\"gpu_timing\": false") != std::string::npos);
    CHECK(json.find("\"gpu_timing_unfused\": false") != std::string::npos);
    CHECK(json.find("\"passes\"") == std::string::npos);
    CHECK(json.find("\"gpu_pass_sum_ms\"") == std::string::npos);
    for (const char* key : {"\"bench\"", "\"device\"", "\"width\"", "\"height\"", "\"vsync\"", "\"ui\"", "\"frames\"",
                            "\"fps\"", "\"gpu_allocations\"", "\"cpu_heap_blocks_delta\"", "\"cpu_heap_bytes_delta\"",
                            "\"frame_ms\"", "\"cpu_ms\"", "\"gpu_ms\"", "\"wait_ms\"", "\"pipelines\": {\"hits\": 3}"}) {
        CHECK_MESSAGE(json.find(key) != std::string::npos, key);
    }
    CHECK(formatReportLine(r).find("GPU passes") == std::string::npos);
}

TEST_CASE("bench report: schema v3 with GPU timing") {
    const BenchReport r = timedReport();
    const std::string json = reportToJson(r);
    CHECK_MESSAGE(validJson(json), json);
    CHECK(json.find("\"schema_version\": 5") != std::string::npos);
    CHECK(json.find("\"gpu_timing\": true") != std::string::npos);
    CHECK(json.find("\"gpu_timing_unfused\": true") != std::string::npos);
    CHECK(json.find("\"passes\": [") != std::string::npos);
    CHECK(json.find("\"name\": \"Forward + Overlay\"") != std::string::npos);
    CHECK(json.find("\"queue\": \"async\"") != std::string::npos);
    CHECK(json.find("\"fused\": true") != std::string::npos);
    CHECK(json.find("\"passes\": [\"Forward\", \"Overlay\"]") != std::string::npos);
    CHECK(json.find("\"shaders\": [\"fwd_v\", \"fwd_f\"]") != std::string::npos);
    CHECK(json.find("\"dram_bytes\": 123456") != std::string::npos);
    CHECK(json.find("\"gpu_ms\": {\"mean\": 1.5000") != std::string::npos);
    CHECK(json.find("\"gpu_pass_sum_ms\": {\"mean\": 2.0000") != std::string::npos);
    CHECK(json.find("\"gpu_frame_span_ms\": {\"mean\": 2.5000") != std::string::npos);
    // Strings are escaped: quote, backslash, newline, tab.
    CHECK(json.find("Forward \\\"PBR\\\"\\n\\tscene\\\\") != std::string::npos);
    CHECK(json.find("Async \\\"reduce\\\"") != std::string::npos);

    const std::string line   = formatReportLine(r);
    const std::string suffix = " | GPU passes 2.000 ms (p99 3.000)";
    REQUIRE(line.size() >= suffix.size());
    CHECK(line.substr(line.size() - suffix.size()) == suffix);
}

TEST_CASE("bench report: timing enabled with no passes and long names stay valid JSON") {
    BenchReport r;
    r.gpuTiming = true;
    r.bench     = std::string(2000, 'x'); // longer than any fixed formatting buffer
    r.device    = std::string(700, 'd');
    CHECK(validJson(reportToJson(r)));
    CHECK(reportToJson(r).find("\"passes\": []") != std::string::npos);
    r.gpuPassSumMs.mean = std::nanf("");
    CHECK(validJson(reportToJson(r))); // non-finite numbers never reach the JSON
}

TEST_CASE("bench report: per-pass work (schema v3, OPT-0.4)") {
    BenchReport r = timedReport();
    CHECK(reportToJson(r).find("\"work\"") == std::string::npos); // omitted when unknown
    PassWork w;
    w.pass      = "Forward";
    w.draws     = 3;
    w.instances = 100;
    w.indices   = 307200;
    w.vertices  = 56100;
    w.pixels    = 5760000;
    w.lights    = 4;
    r.passes[0].work.push_back(w);
    const std::string json = reportToJson(r);
    CHECK_MESSAGE(validJson(json), json);
    CHECK(json.find("\"work\": [{\"pass\": \"Forward\", \"draws\": 3, \"instances\": 100, \"indices\": 307200, "
                    "\"vertices\": 56100, \"pixels\": 5760000, \"threads\": 0, \"lights\": 4}]") != std::string::npos);
}

TEST_CASE("launch options: F5 scene switches") {
    LaunchOptions o;
    std::string err;
    REQUIRE(parse({}, o, err));
    CHECK(o.gpuDriven == GpuDrivenMode::Off);
    CHECK(o.sceneInstances == 0);
    CHECK(o.dynamicCpuPercent < 0.0f); // bench default
    CHECK(o.debugGpuSceneCorrupt == SceneCorruption::None);

    REQUIRE(parse({"--bench", "8", "--gpu-driven", "on", "--instances", "250000", "--scene-meshes", "64",
                   "--dynamic-cpu", "2.5", "--churn", "100", "--cull-distance", "150.5", "--cull-min-pixels", "2",
                   "--debug-gpu-scene", "30", "--debug-gpu-scene-corrupt", "plane"},
                  o, err));
    CHECK(o.bench == 7);
    CHECK(o.gpuDriven == GpuDrivenMode::On);
    CHECK(o.sceneInstances == 250000);
    CHECK(o.sceneMeshes == 64);
    CHECK(o.dynamicCpuPercent == doctest::Approx(2.5f));
    CHECK(o.churn == 100);
    CHECK(o.cullDistance == doctest::Approx(150.5f));
    CHECK(o.cullMinPixels == doctest::Approx(2.0f));
    CHECK(o.debugGpuScene == 30);
    CHECK(o.debugGpuSceneCorrupt == SceneCorruption::Plane);

    REQUIRE(parse({"--gpu-driven", "off", "--dynamic-cpu", "0"}, o, err));
    CHECK(o.gpuDriven == GpuDrivenMode::Off);
    CHECK(o.dynamicCpuPercent == 0.0f); // explicit 0 differs from the default (< 0)
    for (const auto& [text, kind] : {std::pair<const char*, SceneCorruption>{"delta", SceneCorruption::Delta},
                                     {"plane", SceneCorruption::Plane},
                                     {"command", SceneCorruption::Command},
                                     {"touch", SceneCorruption::Touch}}) {
        REQUIRE(parse({"--debug-gpu-scene-corrupt", text}, o, err));
        CHECK(o.debugGpuSceneCorrupt == kind);
    }
}

TEST_CASE("launch options: F5 scene switch errors") {
    LaunchOptions o;
    std::string err;
    CHECK_FALSE(parse({"--gpu-driven", "maybe"}, o, err));
    CHECK(err.find("--gpu-driven") != std::string::npos);
    CHECK_FALSE(parse({"--gpu-driven"}, o, err));
    CHECK_FALSE(parse({"--instances", "0"}, o, err));
    CHECK_FALSE(parse({"--instances", "-3"}, o, err));
    CHECK_FALSE(parse({"--scene-meshes", "0"}, o, err));
    CHECK_FALSE(parse({"--scene-meshes", "1025"}, o, err));
    CHECK(parse({"--scene-meshes", "1024"}, o, err));
    CHECK_FALSE(parse({"--dynamic-cpu", "101"}, o, err));
    CHECK_FALSE(parse({"--dynamic-cpu", "-1"}, o, err));
    CHECK_FALSE(parse({"--dynamic-cpu", "abc"}, o, err));
    CHECK_FALSE(parse({"--dynamic-cpu", "nan"}, o, err));
    CHECK_FALSE(parse({"--churn", "x"}, o, err));
    CHECK_FALSE(parse({"--cull-distance", "-1"}, o, err));
    CHECK_FALSE(parse({"--cull-min-pixels", "1.5px"}, o, err));
    CHECK_FALSE(parse({"--debug-gpu-scene", "-1"}, o, err));
    CHECK_FALSE(parse({"--debug-gpu-scene-corrupt", "everything"}, o, err));
    CHECK(err.find("--debug-gpu-scene-corrupt") != std::string::npos);
}

TEST_CASE("bench report: schema v5 scene and cpu_phases objects") {
    BenchReport base = timedReport();
    const std::string v4 = reportToJson(base);
    CHECK(v4.find("\"scene\"") == std::string::npos); // omitted when absent
    CHECK(v4.find("\"cpu_phases\"") == std::string::npos);

    BenchReport r = base;
    r.scene.present   = true;
    r.scene.mode      = "on";
    r.scene.instances = 1000000;
    r.scene.slots     = 1048576;
    r.scene.buckets   = 24;
    r.scene.materials = 256;
    r.scene.commands  = 40;
    r.scene.structureChanges = 7;
    r.scene.queueOverflow    = 0;
    r.scene.uploadBytes   = {1000.0f, 0.0f, 900.0f, 2000.0f, 2500.0f};
    r.scene.visible       = {350000.0f, 340000.0f, 350000.0f, 360000.0f, 361000.0f};
    r.scene.cpuCommands   = {3.0f, 3.0f, 3.0f, 3.0f, 3.0f};
    r.cpuPhases.present   = true;
    r.cpuPhases.sim       = {1.5f, 1.0f, 1.4f, 2.0f, 2.5f};
    r.cpuPhases.submit    = {0.25f, 0.2f, 0.25f, 0.3f, 0.35f};
    const std::string json = reportToJson(r);
    CHECK_MESSAGE(validJson(json), json);
    CHECK(json.find("\"schema_version\": 5") != std::string::npos);
    CHECK(json.find("\"scene\": {\"mode\": \"on\", \"instances\": 1000000, \"slots\": 1048576, \"buckets\": 24, "
                    "\"materials\": 256, \"commands\": 40, \"structure_changes\": 7, \"queue_overflow\": 0, "
                    "\"upload_bytes\": {\"mean\": 1000.0000") != std::string::npos);
    for (const char* key : {"\"delta_records\"", "\"visible\": {\"mean\": 350000.0000", "\"culled_frustum\"",
                            "\"culled_distance\"", "\"culled_size\"", "\"draw_commands\"",
                            "\"cpu_commands\": {\"mean\": 3.0000", "\"cpu_phases\": {\"sim\": {\"mean\": 1.5000",
                            "\"scene_sync\"", "\"prepare\"", "\"ui\"", "\"graph\"", "\"submit\": {\"mean\": 0.2500"}) {
        CHECK_MESSAGE(json.find(key) != std::string::npos, key);
    }
    // The v4 content is unchanged: the new objects are only appended after it.
    CHECK(json.substr(0, v4.size() - 3) == v4.substr(0, v4.size() - 3));

    r.scene.mode = "of\"f";
    CHECK(validJson(reportToJson(r))); // strings are escaped
    r.scene.visible.mean = std::nanf("");
    CHECK(validJson(reportToJson(r))); // non-finite numbers never reach the JSON
}
