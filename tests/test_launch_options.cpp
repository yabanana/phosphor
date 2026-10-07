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
    CHECK(json.find("\"schema_version\": " + std::to_string(BENCH_REPORT_SCHEMA_VERSION)) != std::string::npos);
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
    CHECK(json.find("\"schema_version\": " + std::to_string(BENCH_REPORT_SCHEMA_VERSION)) != std::string::npos);
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
    CHECK(o.gpuDriven == GpuDrivenMode::On); // F5 default
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
    CHECK(json.find("\"schema_version\": " + std::to_string(BENCH_REPORT_SCHEMA_VERSION)) != std::string::npos);
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

// ---------------------------------------------------------------------------
// F6: launch options and report schema 6
// ---------------------------------------------------------------------------

TEST_CASE("launch options: F6 defaults") {
    LaunchOptions o;
    std::string err;
    REQUIRE(parse({}, o, err));
    CHECK(o.geometryPath == GeometryPath::Indexed);
    CHECK(o.meshletCull == MeshletCull::TwoPhase);
    CHECK(o.hizPath == HiZPath::Auto);
    CHECK_FALSE(o.forceApple9);
    CHECK(o.debugMeshlets == 0);
    CHECK(o.debugMeshletsCorrupt == MeshletCorruption::None);
    CHECK(o.debugView == MeshletDebugView::None);
    CHECK(o.debugHiZLevel == 3);
    CHECK_FALSE(o.meshletSpatial);
    CHECK(o.meshletMaxVertices == 0);
    CHECK(o.meshletMaxTriangles == 0);
    CHECK(o.resolutionWidth == 0);
    CHECK(o.resolutionHeight == 0);
    CHECK_FALSE(o.cullingScript);
    CHECK(o.historyResetEvery == 0);
    CHECK(std::string(geometryPathName(o.geometryPath)) == "indexed");
    CHECK(std::string(meshletCullName(o.meshletCull)) == "two-phase");
    CHECK(std::string(hizPathName(o.hizPath)) == "auto");
    CHECK(std::string(meshletDebugViewName(o.debugView)) == "none");
}

TEST_CASE("launch options: F6 accepted values") {
    LaunchOptions o;
    std::string err;
    REQUIRE(parse({"--geometry-path", "indexed"}, o, err));
    CHECK(o.geometryPath == GeometryPath::Indexed);
    REQUIRE(parse({"--geometry-path", "mesh"}, o, err));
    CHECK(o.geometryPath == GeometryPath::Mesh);
    CHECK(std::string(geometryPathName(o.geometryPath)) == "mesh");

    for (const auto& [text, cull] : {std::pair<const char*, MeshletCull>{"off", MeshletCull::Off},
                                     {"frustum", MeshletCull::Frustum},
                                     {"two-phase", MeshletCull::TwoPhase}}) {
        REQUIRE(parse({"--meshlet-cull", text}, o, err));
        CHECK(o.meshletCull == cull);
        CHECK(std::string(meshletCullName(o.meshletCull)) == text);
    }
    for (const auto& [text, hiz] : {std::pair<const char*, HiZPath>{"auto", HiZPath::Auto},
                                    {"compute", HiZPath::Compute},
                                    {"sampler", HiZPath::Sampler}}) {
        REQUIRE(parse({"--hiz-path", text}, o, err));
        CHECK(o.hizPath == hiz);
        CHECK(std::string(hizPathName(o.hizPath)) == text);
    }
    REQUIRE(parse({"--force-family", "apple9"}, o, err));
    CHECK(o.forceApple9);

    REQUIRE(parse({"--geometry-path", "mesh", "--debug-meshlets", "30"}, o, err));
    CHECK(o.debugMeshlets == 30);
    for (const auto& [text, kind] : {std::pair<const char*, MeshletCorruption>{"id", MeshletCorruption::Id},
                                     {"depth", MeshletCorruption::Depth},
                                     {"count", MeshletCorruption::Count}}) {
        REQUIRE(parse({"--geometry-path", "mesh", "--debug-meshlets-corrupt", text}, o, err));
        CHECK(o.debugMeshletsCorrupt == kind);
    }
    for (const auto& [text, view] : {std::pair<const char*, MeshletDebugView>{"none", MeshletDebugView::None},
                                     {"meshlets", MeshletDebugView::Meshlets},
                                     {"cull", MeshletDebugView::Cull},
                                     {"hiz", MeshletDebugView::HiZ}}) {
        REQUIRE(parse({"--geometry-path", "mesh", "--debug-view", text}, o, err));
        CHECK(o.debugView == view);
        CHECK(std::string(meshletDebugViewName(o.debugView)) == text);
    }
    REQUIRE(parse({"--debug-hiz-level", "0"}, o, err));
    CHECK(o.debugHiZLevel == 0);
    REQUIRE(parse({"--debug-hiz-level", "15"}, o, err));
    CHECK(o.debugHiZLevel == 15);

    REQUIRE(parse({"--meshlet-builder", "spatial", "--meshlet-max-vertices", "96", "--meshlet-max-triangles", "128"},
                  o, err));
    CHECK(o.meshletSpatial);
    CHECK(o.meshletMaxVertices == 96);
    CHECK(o.meshletMaxTriangles == 128);
    REQUIRE(parse({"--meshlet-builder", "standard"}, o, err));
    CHECK_FALSE(o.meshletSpatial);

    REQUIRE(parse({"--culling-script", "--history-reset-every", "17"}, o, err));
    CHECK(o.cullingScript);
    CHECK(o.historyResetEvery == 17);
}

TEST_CASE("launch options: F6 malformed values name the option") {
    LaunchOptions o;
    std::string err;
    for (const auto& [opt, bad] : {std::pair<const char*, const char*>{"--geometry-path", "meshes"},
                                   {"--meshlet-cull", "full"},
                                   {"--hiz-path", "gpu"},
                                   {"--force-family", "apple10"},
                                   {"--force-family", "m3"},
                                   {"--debug-meshlets", "x"},
                                   {"--debug-meshlets", "-1"},
                                   {"--debug-meshlets-corrupt", "all"},
                                   {"--debug-view", "wireframe"},
                                   {"--debug-hiz-level", "16"},
                                   {"--debug-hiz-level", "abc"},
                                   {"--meshlet-builder", "greedy"},
                                   {"--meshlet-max-vertices", "many"},
                                   {"--meshlet-max-triangles", "1.5"},
                                   {"--resolution", "1920"},
                                   {"--resolution", "axb"},
                                   {"--history-reset-every", "soon"}}) {
        err.clear();
        CHECK_FALSE_MESSAGE(parse({opt, bad}, o, err), opt << " " << bad);
        CHECK_MESSAGE(err.find(opt) != std::string::npos, opt << ": " << err);
    }
    // Missing value.
    for (const char* opt : {"--geometry-path", "--meshlet-cull", "--hiz-path", "--force-family", "--debug-view",
                            "--meshlet-builder", "--resolution", "--history-reset-every"}) {
        err.clear();
        CHECK_FALSE(parse({opt}, o, err));
        CHECK_MESSAGE(err.find(opt) != std::string::npos, opt << ": " << err);
    }
}

TEST_CASE("launch options: F6 --resolution parsing and bounds") {
    LaunchOptions o;
    std::string err;
    REQUIRE(parse({"--resolution", "1920x1080"}, o, err));
    CHECK(o.resolutionWidth == 1920);
    CHECK(o.resolutionHeight == 1080);
    REQUIRE(parse({"--resolution", "64x64"}, o, err));
    CHECK(o.resolutionWidth == 64);
    CHECK(o.resolutionHeight == 64);
    REQUIRE(parse({"--resolution", "8192x8192"}, o, err));
    CHECK(o.resolutionWidth == 8192);
    CHECK(o.resolutionHeight == 8192);
    for (const char* bad : {"63x64", "64x63", "8193x64", "64x8193", "0x0", "x1080", "1920x", "1920x1080x2",
                            "1920X1080", "-1920x1080", "", "1920 x 1080"}) {
        err.clear();
        CHECK_FALSE_MESSAGE(parse({"--resolution", bad}, o, err), bad);
        CHECK_MESSAGE(err.find("--resolution") != std::string::npos, err);
    }
}

TEST_CASE("launch options: F6 meshlet cook bounds") {
    LaunchOptions o;
    std::string err;
    REQUIRE(parse({"--meshlet-max-vertices", "3"}, o, err));
    CHECK(o.meshletMaxVertices == 3);
    REQUIRE(parse({"--meshlet-max-vertices", "128"}, o, err));
    CHECK(o.meshletMaxVertices == 128);
    REQUIRE(parse({"--meshlet-max-triangles", "1"}, o, err));
    CHECK(o.meshletMaxTriangles == 1);
    REQUIRE(parse({"--meshlet-max-triangles", "128"}, o, err));
    CHECK(o.meshletMaxTriangles == 128);
    for (const char* v : {"0", "2", "129", "256"}) {
        err.clear();
        CHECK_FALSE_MESSAGE(parse({"--meshlet-max-vertices", v}, o, err), v);
        CHECK(err.find("--meshlet-max-vertices") != std::string::npos);
    }
    for (const char* v : {"0", "129", "512"}) {
        err.clear();
        CHECK_FALSE_MESSAGE(parse({"--meshlet-max-triangles", v}, o, err), v);
        CHECK(err.find("--meshlet-max-triangles") != std::string::npos);
    }
}

TEST_CASE("launch options: F6 refused combinations") {
    LaunchOptions o;
    std::string err;

    SUBCASE("mesh path needs the GPU scene") {
        CHECK_FALSE(parse({"--geometry-path", "mesh", "--gpu-driven", "off"}, o, err));
        CHECK(err.find("--geometry-path") != std::string::npos);
        CHECK(err.find("--gpu-driven") != std::string::npos);
        CHECK_FALSE(parse({"--gpu-driven", "off", "--geometry-path", "mesh"}, o, err)); // order independent
        CHECK(parse({"--geometry-path", "mesh", "--gpu-driven", "on"}, o, err));
        CHECK(parse({"--geometry-path", "mesh"}, o, err)); // gpu-driven defaults to on
        CHECK(parse({"--geometry-path", "indexed", "--gpu-driven", "off"}, o, err));
    }
    SUBCASE("sampler Hi-Z needs Apple10") {
        CHECK_FALSE(parse({"--force-family", "apple9", "--hiz-path", "sampler"}, o, err));
        CHECK(err.find("--hiz-path") != std::string::npos);
        CHECK(err.find("--force-family") != std::string::npos);
        CHECK_FALSE(parse({"--hiz-path", "sampler", "--force-family", "apple9"}, o, err));
        CHECK(parse({"--force-family", "apple9", "--hiz-path", "compute"}, o, err));
        CHECK(parse({"--force-family", "apple9", "--hiz-path", "auto"}, o, err));
        CHECK(parse({"--hiz-path", "sampler"}, o, err));
    }
    SUBCASE("mesh path debug options need the mesh path") {
        CHECK_FALSE(parse({"--debug-meshlets", "10"}, o, err));
        CHECK(err.find("--geometry-path mesh") != std::string::npos);
        CHECK_FALSE(parse({"--debug-view", "meshlets"}, o, err));
        CHECK(err.find("--debug-view") != std::string::npos);
        CHECK_FALSE(parse({"--debug-meshlets-corrupt", "id"}, o, err));
        CHECK(err.find("--debug-meshlets-corrupt") != std::string::npos);
        CHECK_FALSE(parse({"--geometry-path", "indexed", "--debug-view", "cull"}, o, err));
        CHECK(parse({"--debug-view", "none"}, o, err));
        CHECK(parse({"--debug-meshlets", "0"}, o, err));
        CHECK(parse({"--geometry-path", "mesh", "--debug-view", "hiz", "--debug-meshlets", "5",
                     "--debug-meshlets-corrupt", "count"},
                    o, err));
    }
    SUBCASE("only apple9 can be forced") {
        CHECK_FALSE(parse({"--force-family", "apple10"}, o, err));
        CHECK(err.find("--force-family") != std::string::npos);
        CHECK_FALSE(parse({"--force-family", "apple8"}, o, err));
        CHECK_FALSE(parse({"--force-family", ""}, o, err));
    }
}

TEST_CASE("bench report: schema 6 nearest-rank p95") {
    std::vector<float> v;
    for (int i = 20; i >= 1; --i) v.push_back(static_cast<float>(i));
    const TimingSummary s = summarize(v);
    CHECK(s.p50 == doctest::Approx(10.0f));
    CHECK(s.p95 == doctest::Approx(19.0f));
    CHECK(s.p99 == doctest::Approx(20.0f));
    CHECK(s.max == doctest::Approx(20.0f));

    const TimingSummary one = summarize({7.0f});
    CHECK(one.p50 == doctest::Approx(7.0f));
    CHECK(one.p95 == doctest::Approx(7.0f));
    CHECK(one.p99 == doctest::Approx(7.0f));

    std::vector<float> h;
    for (int i = 1; i <= 100; ++i) h.push_back(static_cast<float>(i));
    const TimingSummary s100 = summarize(h);
    CHECK(s100.p50 == doctest::Approx(50.0f));
    CHECK(s100.p95 == doctest::Approx(95.0f));
    CHECK(s100.p99 == doctest::Approx(99.0f));

    const TimingSummary empty = summarize({});
    CHECK(empty.p95 == 0.0f);

    for (const TimingSummary& t : {s, one, s100}) {
        CHECK(t.p50 <= t.p95);
        CHECK(t.p95 <= t.p99);
        CHECK(t.p99 <= t.max);
    }
}

TEST_CASE("bench report: positional TimingSummary initialisers keep their meaning (p95 is last)") {
    const TimingSummary t{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    CHECK(t.mean == 1.0f);
    CHECK(t.min == 2.0f);
    CHECK(t.p50 == 3.0f);
    CHECK(t.p99 == 4.0f);
    CHECK(t.max == 5.0f);
    CHECK(t.p95 == 0.0f);
    const TimingSummary u{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 3.5f};
    CHECK(u.p95 == 3.5f);
}

TEST_CASE("bench report: schema 6 JSON has p95 between p50 and p99 in every summary") {
    std::vector<FrameSample> samples;
    for (int i = 1; i <= 20; ++i) {
        const float f = static_cast<float>(i);
        samples.push_back(FrameSample{f, f / 2.0f, f / 4.0f, 0.1f});
    }
    BenchReport r = timedReport();
    r.bench  = "Torus";
    r.device = "GPU";
    summarizeSamples(samples, r);
    r.scene.present = true;
    r.scene.visible = summarize({1.0f, 2.0f, 3.0f});
    r.cpuPhases.present = true;
    r.cpuPhases.sim = summarize({1.0f, 2.0f, 3.0f});
    r.meshlets.present = true;
    r.meshlets.candidates = summarize({10.0f, 20.0f});
    const std::string json = reportToJson(r);
    CHECK_MESSAGE(validJson(json), json);
    CHECK(json.find("\"schema_version\": " + std::to_string(BENCH_REPORT_SCHEMA_VERSION)) != std::string::npos);
    CHECK(r.frameMs.p95 == doctest::Approx(19.0f));
    CHECK(json.find("\"frame_ms\": {\"mean\": 10.5000, \"min\": 1.0000, \"p50\": 10.0000, \"p95\": 19.0000, "
                    "\"p99\": 20.0000, \"max\": 20.0000}") != std::string::npos);
    size_t summaries = 0;
    for (size_t pos = json.find("\"mean\":"); pos != std::string::npos; pos = json.find("\"mean\":", pos + 1)) {
        const size_t end = json.find('}', pos);
        const std::string obj = json.substr(pos, end - pos);
        const size_t p50 = obj.find("\"p50\"");
        const size_t p95 = obj.find("\"p95\"");
        const size_t p99 = obj.find("\"p99\"");
        CHECK_MESSAGE(p50 != std::string::npos, obj);
        CHECK_MESSAGE((p50 < p95 && p95 < p99), obj);
        ++summaries;
    }
    CHECK(summaries > 10);

    const std::string line = formatReportLine(r);
    CHECK(line.find("frame 10.500 ms (p95 19.000 p99 20.000)") != std::string::npos);
}

TEST_CASE("bench report: schema 6 hardware object only when present") {
    BenchReport r;
    r.bench  = "Torus";
    r.device = "GPU";
    CHECK(reportToJson(r).find("\"hardware\"") == std::string::npos);
    CHECK(reportToJson(r).find("\"meshlets\"") == std::string::npos);

    r.hardware.present = true;
    r.hardware.physicalDevice = "Apple M5 Max";
    r.hardware.physicalFamily = "apple10";
    r.hardware.memoryBytes = 137438953472ull;
    r.hardware.effectiveCapabilities = "apple9";
    r.hardware.preset = "t0-apple9";
    r.hardware.unverifiedDevices = {"M3 Base", "M4 \"Pro\""};
    std::string json = reportToJson(r);
    CHECK_MESSAGE(validJson(json), json);
    CHECK(json.find("\"hardware\": {\"physical_device\": \"Apple M5 Max\", \"physical_family\": \"apple10\", "
                    "\"memory_bytes\": 137438953472, \"effective_capabilities\": \"apple9\", \"preset\": "
                    "\"t0-apple9\", \"validation_scope\": \"development\", \"unverified_devices\": [\"M3 Base\", "
                    "\"M4 \\\"Pro\\\"\"]}") != std::string::npos);
    CHECK(json.find("\"meshlets\"") == std::string::npos);

    r.hardware.unverifiedDevices.clear();
    r.hardware.physicalDevice = "GPU \"x\"\\y\n";
    r.hardware.validationScope = "multi\"device";
    json = reportToJson(r);
    CHECK_MESSAGE(validJson(json), json);
    CHECK(json.find("\"unverified_devices\": []") != std::string::npos);
    CHECK(json.find("\"physical_device\": \"GPU \\\"x\\\"\\\\y\\n\"") != std::string::npos);
    CHECK(json.find("\"validation_scope\": \"multi\\\"device\"") != std::string::npos);
}

TEST_CASE("bench report: schema 6 meshlets object only when present") {
    BenchReport r;
    r.bench  = "Culling";
    r.device = "GPU";
    r.meshlets.present = true;
    r.meshlets.path = "mesh";
    r.meshlets.cull = "two-phase";
    r.meshlets.hizRequested = "auto";
    r.meshlets.hizEffective = "compute";
    r.meshlets.cook = "standard-64v124t";
    r.meshlets.meshlets = 41210;
    r.meshlets.candidateCapacity = 1048576;
    r.meshlets.overflowFrames = 0;
    r.meshlets.historyResets = 2;
    r.meshlets.checks = 12;
    r.meshlets.checkFailures = 1;
    r.meshlets.candidates = summarize({350000.0f});
    r.meshlets.primitives = summarize({1.0e7f, 1.1e7f});
    std::string json = reportToJson(r);
    CHECK_MESSAGE(validJson(json), json);
    CHECK(json.find("\"hardware\"") == std::string::npos);
    CHECK(json.find("\"meshlets\": {\"path\": \"mesh\", \"cull\": \"two-phase\", \"hiz_requested\": \"auto\", "
                    "\"hiz_effective\": \"compute\", \"cook\": \"standard-64v124t\", \"meshlets\": 41210, "
                    "\"candidate_capacity\": 1048576, \"overflow_frames\": 0, \"history_resets\": 2, "
                    "\"checks\": 12, \"check_failures\": 1, \"candidates\": {\"mean\": 350000.0000") !=
          std::string::npos);
    for (const char* key : {"\"drawn_a\"", "\"frustum\"", "\"cone\"", "\"history_rejected\"", "\"drawn_b\"",
                            "\"occluded_b\"", "\"primitives\": {\"mean\": 10500000.0000"}) {
        CHECK_MESSAGE(json.find(key) != std::string::npos, key);
    }
    r.meshlets.cook = "spatial-\"64\"";
    CHECK(validJson(reportToJson(r)));
    r.meshlets.drawnA.mean = std::nanf("");
    CHECK(validJson(reportToJson(r)));
    r.meshlets.present = false;
    CHECK(reportToJson(r).find("\"meshlets\"") == std::string::npos);
}

TEST_CASE("F7 F8 options select explicit rendering contracts and bounded quality controls") {
    LaunchOptions o;
    std::string error;
    REQUIRE(parse({"--post", "--upscaler", "temporal", "--render-scale", "0.5", "--tonemap", "agx", "--temporal-views",
                   "2", "--frames-in-flight", "1", "--scene", "assets/sponza/Sponza.gltf"},
                  o, error));
    CHECK(o.visibility);
    CHECK(o.geometryPath == GeometryPath::Mesh);
    CHECK(o.temporalUpscale);
    CHECK(o.renderScale == 0.5f);
    CHECK(o.tonemap == 1);
    CHECK(o.temporalViews == 2);
    CHECK(o.framesInFlight == 1);
    for (const char *option : {"--render-scale", "--sharpen", "--tone-white", "--drs-budget"}) {
        CHECK_FALSE(parse({option, "nan"}, o, error));
    }
    CHECK_FALSE(parse({"--render-scale", "0.4"}, o, error));
    CHECK_FALSE(parse({"--temporal-views", "5"}, o, error));
    CHECK_FALSE(parse({"--frames-in-flight", "0"}, o, error));
    CHECK_FALSE(parse({"--debug-motion-corrupt"}, o, error));
    CHECK_FALSE(parse({"--render-path", "visibility", "--gpu-driven", "off"}, o, error));
}

TEST_CASE("MetalFX runs in process by default; isolated workers stay opt-in") {
    LaunchOptions o;
    std::string error;
    REQUIRE(parse({"--post", "--upscaler", "temporal"}, o, error));
    CHECK_FALSE(o.isolatedMetalFX);
    REQUIRE(parse({"--post", "--upscaler", "temporal", "--metalfx-mode", "isolated"}, o, error));
    CHECK(o.isolatedMetalFX);
    REQUIRE(parse({"--post", "--upscaler", "temporal", "--metalfx-mode", "direct"}, o, error));
    CHECK_FALSE(o.isolatedMetalFX);
    CHECK_FALSE(parse({"--metalfx-mode", "unknown"}, o, error));
    CHECK_FALSE(parse({"--debug-frame-delay-ms", "1001"}, o, error));
    CHECK_FALSE(parse({"--debug-metalfx-worker-crash", "2"}, o, error));
    CHECK_FALSE(parse({"--upscaler", "temporal", "--debug-metalfx-worker-crash", "2"}, o, error));
    CHECK_FALSE(parse({"--upscaler", "temporal", "--metalfx-mode", "isolated", "--debug-metalfx-worker-delay-ms",
                       "2001"},
                      o, error));
    REQUIRE(parse({"--upscaler", "temporal", "--metalfx-mode", "isolated", "--debug-metalfx-worker-crash", "2"}, o,
                  error));
    CHECK(o.debugMetalFXWorkerCrash == 2);
}

TEST_CASE("MetalFX resize settle frames are bounded") {
    LaunchOptions o;
    std::string error;
    REQUIRE(parse({"--post", "--upscaler", "temporal"}, o, error));
    CHECK(o.metalfxResizeSettleFrames == 4);
    REQUIRE(parse({"--metalfx-resize-settle", "0"}, o, error));
    CHECK(o.metalfxResizeSettleFrames == 0);
    REQUIRE(parse({"--metalfx-resize-settle", "120"}, o, error));
    CHECK(o.metalfxResizeSettleFrames == 120);
    CHECK_FALSE(parse({"--metalfx-resize-settle", "121"}, o, error));
    CHECK_FALSE(parse({"--metalfx-resize-settle"}, o, error));
}

TEST_CASE("F9 launch options keep RT off by default and accept indexed diagnostics") {
    LaunchOptions o;
    std::string error;
    REQUIRE(parse({}, o, error));
    CHECK_FALSE(o.rtEnabled);
    CHECK_FALSE(o.rtProxyManifest);
    CHECK(o.rtProxyManifestPath.empty());
    CHECK(o.rtTlasRebuildEvery == 0);
    CHECK(o.debugRt == 0);
    CHECK_FALSE(o.debugRtDeform);
    CHECK(o.debugRtCorrupt == RtCorruption::None);
    CHECK(o.rtProbe == RtProbe::None);
    REQUIRE(parse({"--rt", "off", "--rt-proxy", "off"}, o, error));
    CHECK_FALSE(o.rtEnabled);

    REQUIRE(parse({"--rt", "on", "--rt-proxy", "manifest", "--rt-proxy-manifest", "path with spaces/proxy.json",
                   "--rt-tlas-rebuild-every", "100", "--debug-rt", "5", "--debug-rt-corrupt", "mask",
                   "--rt-probe", "shadow", "--debug-view", "rt"}, o, error));
    CHECK(o.rtEnabled);
    CHECK(o.rtProxyManifest);
    CHECK(o.rtProxyManifestPath == "path with spaces/proxy.json");
    CHECK(o.rtTlasRebuildEvery == 100);
    CHECK(o.debugRt == 5);
    CHECK(o.debugRtCorrupt == RtCorruption::Mask);
    CHECK(o.rtProbe == RtProbe::Shadow);
    CHECK(o.debugView == MeshletDebugView::RT);
    CHECK(o.geometryPath == GeometryPath::Indexed);
    CHECK(std::string(meshletDebugViewName(o.debugView)) == "rt");
    CHECK(std::string(rtCorruptionName(o.debugRtCorrupt)) == "mask");
    CHECK(std::string(rtProbeName(o.rtProbe)) == "shadow");
    REQUIRE(parse({"--rt", "on", "--debug-view", "rt", "--geometry-path", "mesh"}, o, error));
    CHECK(o.geometryPath == GeometryPath::Mesh);
    REQUIRE(parse({"--rt-probe", "ao", "--rt", "on"}, o, error)); // order-independent dependencies
    CHECK(o.rtProbe == RtProbe::AO);
}

TEST_CASE("F9 launch options validate enum alternatives and complete u32 counts") {
    LaunchOptions o;
    std::string error;
    for (const auto& [name, expected] : std::vector<std::pair<const char*, RtProbe>>{
        {"primary", RtProbe::Primary}, {"shadow", RtProbe::Shadow}, {"ao", RtProbe::AO}, {"diffuse", RtProbe::Diffuse}}) {
        REQUIRE(parse({"--rt", "on", "--rt-probe", name}, o, error));
        CHECK(o.rtProbe == expected);
        CHECK(std::string(rtProbeName(o.rtProbe)) == name);
    }
    for (const auto& [name, expected] : std::vector<std::pair<const char*, RtCorruption>>{
        {"transform", RtCorruption::Transform}, {"mask", RtCorruption::Mask}, {"blas", RtCorruption::Blas}}) {
        REQUIRE(parse({"--rt", "on", "--debug-rt", "1", "--debug-rt-corrupt", name}, o, error));
        CHECK(o.debugRtCorrupt == expected);
        CHECK(std::string(rtCorruptionName(o.debugRtCorrupt)) == name);
    }
    CHECK(std::string(rtProbeName(RtProbe::None)) == "none");
    CHECK(std::string(rtCorruptionName(RtCorruption::None)) == "none");
    for (const char* option : {"--rt-tlas-rebuild-every", "--debug-rt"}) {
        REQUIRE(parse({"--rt", "on", option, "0"}, o, error));
        REQUIRE(parse({"--rt", "on", option, "4294967295"}, o, error));
        if (std::string(option) == "--debug-rt") CHECK(o.debugRt == 0xffffffffu);
        else CHECK(o.rtTlasRebuildEvery == 0xffffffffu);
        for (const char* bad : {"4294967296", "-1", "+1", "1.5", "1x", "nan", ""}) {
            CHECK_FALSE(parse({"--rt", "on", option, bad}, o, error));
            CHECK_FALSE(error.empty());
        }
    }
}

TEST_CASE("F9 launch options reject missing values and unsupported combinations") {
    LaunchOptions o;
    std::string error;
    for (const char* option : {"--rt", "--rt-proxy", "--rt-proxy-manifest", "--rt-tlas-rebuild-every",
                               "--debug-rt", "--debug-rt-corrupt", "--rt-probe"}) {
        CHECK_FALSE(parse({"--rt", "on", option}, o, error));
        CHECK_FALSE(error.empty());
    }
    for (const auto& args : std::vector<std::vector<const char*>>{
        {"--rt", "yes"}, {"--rt", "on", "--rt-proxy", "ratio"}, {"--rt", "on", "--rt-probe", "none"},
        {"--rt", "on", "--debug-rt", "1", "--debug-rt-corrupt", "unknown"},
        {"--debug-view", "rt"}, {"--rt-proxy", "manifest"}, {"--rt-probe", "primary"},
        {"--debug-rt", "0"}, {"--rt-tlas-rebuild-every", "0"}, {"--debug-rt", "1"},
        {"--rt", "on", "--rt-proxy-manifest", "proxy.json"},
        {"--rt", "on", "--rt-proxy", "manifest", "--rt-proxy-manifest", ""},
        {"--rt", "on", "--rt-proxy", "manifest", "--rt-proxy-manifest", "--frames", "10"},
        {"--rt", "on", "--debug-rt-corrupt", "blas"},
        {"--rt", "on", "--debug-rt", "0", "--debug-rt-corrupt", "transform"},
        {"--rt", "on", "--graph-scenario", "0"}, {"--rt", "on", "--memory-stress", "1"},
        {"--rt", "on", "--transient-test"},
        {"--rt", "on", "--debug-meshlets", "1"}, {"--rt", "on", "--debug-view", "meshlets"},
        {"--rt", "on", "--debug-view", "rt", "--rt", "off"}}) {
        CHECK_FALSE(parse(args, o, error));
        CHECK_FALSE(error.empty());
    }
}


TEST_CASE("F9 BLAS deformation requires the checked full-geometry RT debug view") {
    LaunchOptions o;
    std::string error;
    REQUIRE(parse({"--debug-rt-deform", "--rt", "on", "--debug-rt", "1", "--debug-view", "rt"}, o, error));
    CHECK(o.debugRtDeform);
    CHECK(o.debugRt == 1);
    CHECK_FALSE(o.rtProxyManifest);
    REQUIRE(parse({"--rt", "on", "--debug-rt", "5", "--debug-view", "rt", "--rt-proxy", "off",
                   "--debug-rt-deform", "--frames", "30"}, o, error));
    CHECK(o.debugRtDeform);
    CHECK(o.frames == 30); // The flag consumes no argument.
    for (const auto& args : std::vector<std::vector<const char*>>{
        {"--debug-rt-deform"},
        {"--debug-rt-deform", "--debug-rt", "1", "--debug-view", "rt"},
        {"--rt", "off", "--debug-rt-deform", "--debug-rt", "1", "--debug-view", "rt"},
        {"--rt", "on", "--debug-rt-deform", "--debug-view", "rt"},
        {"--rt", "on", "--debug-rt-deform", "--debug-rt", "0", "--debug-view", "rt"},
        {"--rt", "on", "--debug-rt-deform", "--debug-rt", "1"},
        {"--rt", "on", "--debug-rt-deform", "--debug-rt", "1", "--debug-view", "none"},
        {"--rt", "on", "--debug-rt-deform", "--debug-rt", "1", "--debug-view", "rt", "--rt-proxy", "manifest"},
        {"--rt", "on", "--debug-rt-deform", "--debug-rt", "1", "--debug-view", "rt", "--rt-proxy", "manifest",
         "--rt-proxy-manifest", "proxy.json"}}) {
        CHECK_FALSE(parse(args, o, error));
        CHECK_FALSE(error.empty());
    }
}

TEST_CASE("F9 proxy transitions require measured proxies checks and enough total frames") {
    LaunchOptions o;
    std::string error;
    for (const char* mode : {"mask","emissive","reassign","full-upload"}) {
        REQUIRE(parse({"--rt","on","--rt-proxy","manifest","--debug-rt","1","--debug-rt-proxy-transition",mode},o,error));
        CHECK(std::string(rtProxyTransitionName(o.debugRtProxyTransition)) == mode); // interactive permitted
    }
    REQUIRE(parse({"--rt","on","--rt-proxy","manifest","--debug-rt","1","--debug-rt-proxy-transition","mask",
                   "--frames","1","--warmup","23"},o,error));
    for (const auto& args : std::vector<std::vector<const char*>>{
        {"--debug-rt-proxy-transition","mask"},
        {"--rt","on","--rt-proxy","manifest","--debug-rt-proxy-transition","mask"},
        {"--rt","on","--debug-rt","1","--debug-rt-proxy-transition","mask"},
        {"--rt","on","--rt-proxy","manifest","--debug-rt","1","--debug-rt-proxy-transition","unknown"},
        {"--rt","on","--rt-proxy","manifest","--debug-rt","1","--debug-rt-proxy-transition","mask","--switch-every","1"},
        {"--rt","on","--rt-proxy","manifest","--debug-rt","1","--debug-rt-proxy-transition","mask","--debug-rt-deform"},
        {"--rt","on","--rt-proxy","manifest","--debug-rt","1","--debug-rt-proxy-transition","mask","--debug-rt-corrupt","mask"},
        {"--rt","on","--rt-proxy","manifest","--debug-rt","1","--debug-rt-proxy-transition","mask","--frames","23","--warmup","0"},
        {"--rt","on","--rt-proxy","manifest","--debug-rt","1","--debug-rt-proxy-transition"}}) {
        CHECK_FALSE(parse(args,o,error));
        CHECK_FALSE(error.empty());
    }
}

TEST_CASE("F10-F12 launch contract refuses unavailable receiver paths and tiers") {
    LaunchOptions o; std::string e;
    REQUIRE(parse({"--render-path","visibility","--shadows","csm"},o,e));
    CHECK(o.shadows==ShadowMode::CSM); CHECK_FALSE(o.rtEnabled);
    CHECK_FALSE(parse({"--shadows","rt","--rt","on"},o,e));
    CHECK_FALSE(parse({"--render-path","visibility","--shadows","rt"},o,e));
    REQUIRE(parse({"--render-path","visibility","--shadows","rt","--rt","on"},o,e));
    CHECK_FALSE(parse({"--render-path","visibility","--rt","on","--gi","restir","--force-family","apple9"},o,e));
    CHECK_FALSE(parse({"--render-path","visibility","--rt","on","--lighting","restir","--adaptive-shading"},o,e));
    CHECK_FALSE(parse({"--render-path","visibility","--shadows","csm","--shadow-map-size","1000"},o,e));
    CHECK_FALSE(parse({"--contact-shadows","on"},o,e));
    CHECK_FALSE(parse({"--debug-lighting-corrupt","caster"},o,e));
    REQUIRE(parse({"--render-path","visibility","--shadows","csm","--debug-lighting","1","--debug-lighting-corrupt","caster"},o,e));
    CHECK(o.debugLightingCorrupt==2);
}

TEST_CASE("F12 linear capture and exact-frame export validate their signal") {
    LaunchOptions o;std::string e;
    CHECK_FALSE(parse({"--capture-linear","x.pfm","--frames","1"},o,e));
    REQUIRE(parse({"--capture-linear","x.pfm","--render-path","visibility","--frames","1"},o,e));
    CHECK_FALSE(parse({"--export-reference","snap","--frames","1","--warmup","0","--export-reference-frame","2"},o,e));
    REQUIRE(parse({"--bench","6","--lighting-scene","thin-walls","--render-path","visibility","--rt","on","--lighting","restir","--gi","ddgi","--capture-linear-signal","indirect-diffuse"},o,e));
    CHECK(o.captureLinearSignal==1);CHECK(o.lightingScene=="thin-walls");
    CHECK_FALSE(parse({"--bench","5","--lighting-scene","thin-walls"},o,e));
}

TEST_CASE("F10 raw shadow capture requires an enabled shadow producer") {
    LaunchOptions o;std::string error;
    CHECK_FALSE(parse({"--render-path","visibility","--capture-linear-signal","shadow"},o,error));
    REQUIRE(parse({"--render-path","visibility","--shadows","csm","--capture-linear-signal","shadow"},o,error));
    CHECK(o.captureLinearSignal==3);
    REQUIRE(parse({"--render-path","visibility","--shadows","rt","--rt","on","--capture-linear-signal","shadow"},o,error));
    CHECK(o.captureLinearSignal==3);
    REQUIRE(parse({"--render-path","visibility","--shadows","csm","--capture-linear-signal","shadow-position"},o,error));
    CHECK(o.captureLinearSignal==4);
    REQUIRE(parse({"--render-path","visibility","--shadows","csm","--capture-linear-signal","shadow-normal"},o,error));
    CHECK(o.captureLinearSignal==5);
    REQUIRE(parse({"--render-path","visibility","--rt","on","--gi","ddgi","--lighting","brute","--capture-linear-signal","shadow-position"},o,error));
    CHECK(o.captureLinearSignal==4);
    CHECK_FALSE(parse({"--render-path","visibility","--capture-linear-signal","shadow-normal"},o,error));
}

TEST_CASE("F13 capture IDs preserve tester shadow diagnostics and named scalar AO") {
    LaunchOptions o;std::string error;
    REQUIRE(parse({"--render-path","visibility","--reflections","ssr","--capture-linear-signal","specular"},o,error));
    CHECK(o.captureLinearSignal==6);
    REQUIRE(parse({"--render-path","visibility","--ao","gtao","--capture-linear-signal","ao"},o,error));
    CHECK(o.captureLinearSignal==7);
    CHECK_FALSE(parse({"--render-path","visibility","--capture-linear-signal","ao"},o,error));
}

TEST_CASE("F13/F14 diagnostic hooks require actual active consumers and checks") {
    LaunchOptions o;std::string error;
    REQUIRE(parse({"--render-path","visibility","--reflections","ssr","--lighting-denoise","custom","--debug-lighting","1","--debug-reflection-corrupt","history"},o,error));
    CHECK(o.debugReflectionCorrupt==1);
    CHECK_FALSE(parse({"--render-path","visibility","--reflections","ssr","--debug-lighting","1","--debug-reflection-corrupt","history"},o,error));
    REQUIRE(parse({"--fog","on","--volume-oracle","oracle","--fog-homogeneous","--debug-lighting","1"},o,error));
    CHECK(o.fogHomogeneous);CHECK(o.volumeOracle=="oracle");
    CHECK_FALSE(parse({"--fog","on","--fog-homogeneous"},o,error));
    REQUIRE(parse({"--atmosphere","on","--debug-volume-corrupt","lut","--debug-lighting","1"},o,error));
    CHECK(o.debugVolumeCorrupt==4);
    CHECK_FALSE(parse({"--atmosphere","on","--debug-volume-corrupt","history","--debug-lighting","1"},o,error));
    CHECK_FALSE(parse({"--fog","on","--debug-volume-corrupt","history","--debug-lighting","1"},o,error));
    REQUIRE(parse({"--fog","on","--atmo-freeze-clock","--debug-volume-corrupt","history","--debug-lighting","1"},o,error));
    CHECK(o.atmoFreezeClock);CHECK(o.debugVolumeCorrupt==2);
    CHECK_FALSE(parse({"--atmo-freeze-clock"},o,error));
    CHECK_FALSE(parse({"--fog","on","--volume-oracle","oracle","--fog-homogeneous","--atmo-freeze-clock","--debug-volume-corrupt","history","--debug-lighting","1"},o,error));
}

TEST_CASE("F13 SDK fixture is an explicit source experiment with isolated inputs") {
    LaunchOptions o;std::string error;
    REQUIRE(parse({"--frames","64","--denoised-fixture","wide-hdr","--denoised-fixture-output","sdk","--denoised-fixture-pre-exposed"},o,error));
    CHECK(o.denoisedFixture=="wide-hdr");CHECK(o.denoisedFixturePreExposed);CHECK(o.post);CHECK(o.visibility);
    CHECK_FALSE(parse({"--denoised-fixture","constant","--denoised-fixture-output","sdk"},o,error));
    CHECK_FALSE(parse({"--frames","64","--denoised-fixture","constant","--denoised-fixture-output","sdk","--atmosphere","on"},o,error));
    CHECK_FALSE(parse({"--frames","64","--denoised-fixture-pre-exposed"},o,error));
}

TEST_CASE("F13 filtered indirect capture exports selected irradiance as diffuse radiance once") {
    LaunchOptions o;std::string error;
    REQUIRE(parse({"--render-path","visibility","--rt","on","--gi","cache","--lighting","restir","--lighting-denoise","custom","--capture-linear-signal","indirect-diffuse-filtered"},o,error));
    CHECK(o.captureLinearSignal==8);
    CHECK_FALSE(parse({"--render-path","visibility","--rt","on","--gi","cache","--lighting","restir","--capture-linear-signal","indirect-diffuse-filtered"},o,error));
    REQUIRE(parse({"--render-path","visibility","--rt","on","--gi","cache","--lighting","restir","--capture-linear-signal","indirect-diffuse"},o,error));
    CHECK(o.captureLinearSignal==1);
}
