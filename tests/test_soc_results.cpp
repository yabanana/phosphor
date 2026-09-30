#include "diagnostics/soc_results.h"

#include <doctest/doctest.h>

#include <cmath>
#include <string>
#include <vector>

using namespace phosphor;
using namespace phosphor::soc;

namespace {

Results oneRun(double alu, double bw, Control neg, Status st) {
    Results r;
    r.machine.chip = "Apple M5 Max";
    r.machine.slug = slugify(r.machine.chip);
    r.machine.gpuFamily = "Apple10";
    r.machine.gpuCores = 40;
    r.machine.os = "macOS 27.2";
    r.machine.osSlug = slugify(r.machine.os);
    r.machine.gpuPStateMHz = {338, 1620};
    r.machine.thermalEnd = "nominal";
    Benchmark b;
    b.id = "B-01";
    b.name = "alu.throughput";
    b.negative = neg;
    b.status = st;
    Metric m;
    m.name = "fp32.fma.independent";
    m.unit = "TFLOPS";
    m.value = alu;
    m.within = computeStats({alu, alu * 1.01, alu * 0.99});
    m.params["chains"] = 4;
    b.metrics.push_back(m);
    Benchmark c;
    c.id = "B-08";
    c.name = "mem.hierarchy";
    Metric d;
    d.name = "stream_bw.ws_1024MiB";
    d.unit = "GB/s";
    d.value = bw;
    c.metrics.push_back(d);
    r.benchmarks = {b, c};
    return r;
}

} // namespace

TEST_CASE("soc results: stats are nearest rank with sample CV") {
    const Stats s = computeStats({5, 1, 4, 2, 3});
    CHECK(s.n == 5);
    CHECK(s.min == 1);
    CHECK(s.max == 5);
    CHECK(s.median == 3);
    CHECK(s.p10 == 1);
    CHECK(s.p90 == 5);
    CHECK(s.mean == doctest::Approx(3.0));
    CHECK(s.cv == doctest::Approx(std::sqrt(2.5) / 3.0));
    CHECK(computeStats({}).n == 0);
    CHECK(computeStats({7}).cv == 0);
}

TEST_CASE("soc results: slugs") {
    CHECK(slugify("Apple M5 Max") == "m5max");
    CHECK(slugify("Apple M3") == "m3");
    CHECK(slugify("macOS 27.2") == "macos27.2");
}

TEST_CASE("soc results: JSON round trip") {
    const Results r = oneRun(10.5, 540, Control::Pass, Status::Ok);
    Results back;
    std::string err;
    REQUIRE(fromJson(toJson(r), back, &err));
    CHECK(back.machine.slug == "m5max");
    CHECK(back.machine.gpuPStateMHz.size() == 2);
    REQUIRE(back.benchmarks.size() == 2);
    const Metric* m = back.find("B-01", "fp32.fma.independent");
    REQUIRE(m != nullptr);
    CHECK(m->value == doctest::Approx(10.5));
    CHECK(m->params.at("chains") == 4);
    CHECK(m->within.n == 3);
    CHECK(back.benchmarks[0].negative == Control::Pass);
    CHECK(back.find("B-99") == nullptr);
}

TEST_CASE("soc results: malformed or foreign JSON is rejected") {
    Results out;
    std::string err;
    CHECK_FALSE(fromJson("{", out, &err));
    CHECK_FALSE(err.empty());
    CHECK_FALSE(fromJson(R"({"schema":"other"})", out, &err));
    std::string text = toJson(oneRun(1, 1, Control::Pass, Status::Ok));
    const auto pos = text.find("\"schema_version\": 1");
    REQUIRE(pos != std::string::npos);
    text.replace(pos, 19, "\"schema_version\": 9");
    CHECK_FALSE(fromJson(text, out, &err));
    CHECK(err.find("schema_version 9") != std::string::npos);
}

TEST_CASE("soc results: merging runs") {
    Results a = oneRun(10.0, 500, Control::Pass, Status::Ok);
    Results b = oneRun(11.0, 540, Control::Fail, Status::Ok);
    Results c = oneRun(12.0, 520, Control::Pass, Status::Partial);
    b.benchmarks[0].negativeDetail = "2x work took 1.2x";
    c.machine.thermalEnd = "fair";
    const Results m = mergeRuns({a, b, c});
    CHECK(m.run.runs == 3);
    CHECK(m.machine.thermalEnd == "fair");
    const Metric* alu = m.find("B-01", "fp32.fma.independent");
    REQUIRE(alu != nullptr);
    CHECK(alu->runs.size() == 3);
    CHECK(alu->value == doctest::Approx(11.0));
    CHECK(alu->runCv == doctest::Approx(1.0 / 11.0));
    CHECK(m.find("B-08", "stream_bw.ws_1024MiB")->value == doctest::Approx(520));
    const Benchmark* b01 = m.find("B-01");
    CHECK(b01->negative == Control::Fail);       // any failing run fails
    CHECK(b01->status == Status::Partial);       // worst status
    CHECK(b01->negativeDetail.find("1.2x") != std::string::npos);
}
