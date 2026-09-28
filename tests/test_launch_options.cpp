#include "core/launch_options.h"
#include "diagnostics/bench_report.h"

#include <doctest/doctest.h>

#include <string>
#include <vector>

using namespace phosphor;

namespace {

bool parse(std::vector<const char*> args, LaunchOptions& out, std::string& error) {
    args.insert(args.begin(), "phosphor");
    return parseLaunchOptions(static_cast<int>(args.size()), args.data(), 7, out, error);
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
                   "--report", "r.json", "--dump-graph", "g.dot", "--debug-split-encoding", "--debug-async-compute"},
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

TEST_CASE("launch options: errors") {
    LaunchOptions o;
    std::string err;
    CHECK_FALSE(parse({"--bench", "0"}, o, err));
    CHECK_FALSE(parse({"--bench", "8"}, o, err));
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
