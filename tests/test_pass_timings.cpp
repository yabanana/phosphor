#include "diagnostics/pass_timings.h"
#include "rendergraph/timing_plan.h"

#include <doctest/doctest.h>

#include <cmath>
#include <string>
#include <vector>

using namespace phosphor;

namespace {

rg::TimingPlan twoUnitPlan() {
    rg::TimingPlan plan;
    rg::TimedUnit a;
    a.name      = "Forward + Overlay";
    a.fused     = true;
    a.passes    = {0, 1};
    a.dramBytes = 1000;
    rg::TimedUnit b;
    b.name      = "Async reduce";
    b.queue     = rg::Queue::AsyncCompute;
    b.kind      = rg::TimestampKind::ComputeEnd;
    b.passes    = {2};
    b.dramBytes = 64;
    plan.units  = {a, b};
    plan.unitOfPosition = {0, 0, 1};
    plan.commitStartQueries = 2;
    return plan;
}

const std::vector<std::string> kNames   = {"Forward", "Overlay", "Async reduce"};
const std::vector<std::string> kShaders = {"fwd_v, fwd_f", "imgui_v,fwd_f", "reduce"};

PassTimings makeTimings() {
    PassTimings t;
    t.configure(twoUnitPlan(), kNames, kShaders);
    return t;
}

} // namespace

TEST_CASE("computeUnitTimes: ticks to ms and invalid cases") {
    const u64 ticks[]  = {1000, 3000, 0, 5000, 2000, 500};
    const u32 start[]  = {0, 1, 2, 0, 5, 0};
    const u32 end[]    = {1, 3, 3, 4, 4, 9};
    float ms[6];
    bool  valid[6];
    computeUnitTimes(ticks, 6, start, end, 6, 1000.0, ms, valid); // 1 tick = 1000 ns = 0.001 ms
    CHECK(valid[0]);
    CHECK(ms[0] == doctest::Approx(2.0f));
    CHECK(valid[1]);
    CHECK(ms[1] == doctest::Approx(2.0f));
    CHECK_FALSE(valid[2]); // start tick 0
    CHECK(ms[2] == 0.0f);
    CHECK(valid[3]);
    CHECK(ms[3] == doctest::Approx(1.0f));
    CHECK(valid[4]);
    CHECK(ms[4] == doctest::Approx(1.5f));
    CHECK_FALSE(valid[5]); // query index out of range
}

TEST_CASE("computeUnitTimes: end before start is invalid") {
    const u64 ticks[] = {5000, 4000};
    const u32 start[] = {0};
    const u32 end[]   = {1};
    float ms[1];
    bool  valid[1];
    computeUnitTimes(ticks, 2, start, end, 1, 1.0, ms, valid);
    CHECK_FALSE(valid[0]);
    CHECK(ms[0] == 0.0f);
    const u64 zero[]    = {7, 0};
    const u32 s2[]      = {0};
    const u32 e2[]      = {1};
    computeUnitTimes(zero, 2, s2, e2, 1, 1.0, ms, valid);
    CHECK_FALSE(valid[0]); // end tick 0
    computeUnitTimes(nullptr, 0, s2, e2, 0, 1.0, ms, valid); // no units: no access
}

TEST_CASE("PassTimings: configure copies unit metadata") {
    PassTimings t = makeTimings();
    REQUIRE(t.unitCount() == 2);
    CHECK(t.unitName(0) == "Forward + Overlay");
    CHECK(t.lastFrame() == ~0ull);
    CHECK(t.rollingSumMs() == 0.0f);

    t.beginMeasure(1);
    const float ms[]  = {1.0f, 0.5f};
    const bool  ok[]  = {true, true};
    t.addFrame(3, ms, ok, 2.0f);
    CHECK(t.endMeasure() == 1);
    std::vector<PassReport> reports;
    TimingSummary sum, span;
    t.summarize(reports, sum, span);
    REQUIRE(reports.size() == 2);
    CHECK(reports[0].queue == "graphics");
    CHECK(reports[0].fused);
    CHECK(reports[0].dramBytes == 1000);
    CHECK(reports[0].passes == std::vector<std::string>{"Forward", "Overlay"});
    // union of the members' shader lists, in first-seen order, trimmed
    CHECK(reports[0].shaders == std::vector<std::string>{"fwd_v", "fwd_f", "imgui_v"});
    CHECK(reports[1].queue == "async");
    CHECK_FALSE(reports[1].fused);
    CHECK(reports[1].passes == std::vector<std::string>{"Async reduce"});
    CHECK(reports[1].shaders == std::vector<std::string>{"reduce"});
}

TEST_CASE("PassTimings: rolling average, max, sum and span skip invalid samples") {
    PassTimings t = makeTimings();
    const float f0[] = {1.0f, 2.0f};
    const bool  v0[] = {true, true};
    const float f1[] = {3.0f, 99.0f};
    const bool  v1[] = {true, false};
    t.addFrame(0, f0, v0, 4.0f);
    t.addFrame(1, f1, v1, -1.0f); // unknown span
    CHECK(t.lastFrame() == 1);
    const PassTimings::UnitStats a = t.rolling(0);
    CHECK(a.samples == 2);
    CHECK(a.avgMs == doctest::Approx(2.0f));
    CHECK(a.maxMs == doctest::Approx(3.0f));
    const PassTimings::UnitStats b = t.rolling(1);
    CHECK(b.samples == 1);
    CHECK(b.avgMs == doctest::Approx(2.0f));
    CHECK(b.maxMs == doctest::Approx(2.0f));
    CHECK(t.rollingSumMs() == doctest::Approx(2.0f + 2.0f)); // sum of the unit averages
    CHECK(t.rollingSpanMs() == doctest::Approx(4.0f)); // the unknown span is skipped
    CHECK(t.rolling(7).samples == 0);
}

TEST_CASE("PassTimings: window wraps after kWindow frames") {
    PassTimings t = makeTimings();
    const bool ok[] = {true, true};
    // 100 frames: the first 40 are large and must fall out of the window.
    for (u32 i = 0; i < 100; ++i) {
        const float v = i < 40 ? 100.0f : 2.0f;
        const float ms[] = {v, v};
        t.addFrame(i, ms, ok, v);
    }
    CHECK(t.lastFrame() == 99);
    const PassTimings::UnitStats s = t.rolling(0);
    CHECK(s.samples == PassTimings::kWindow);
    CHECK(s.avgMs == doctest::Approx(2.0f));
    CHECK(s.maxMs == doctest::Approx(2.0f));
    CHECK(t.rollingSumMs() == doctest::Approx(4.0f));
    CHECK(t.rollingSpanMs() == doctest::Approx(2.0f));
}

TEST_CASE("PassTimings: partially filled window") {
    PassTimings t = makeTimings();
    const bool ok[]  = {true, true};
    const float ms[] = {4.0f, 1.0f};
    for (u32 i = 0; i < 5; ++i) t.addFrame(i, ms, ok, 5.0f);
    CHECK(t.rolling(0).samples == 5);
    CHECK(t.rolling(0).avgMs == doctest::Approx(4.0f));
    CHECK(t.rollingSumMs() == doctest::Approx(5.0f));
}

TEST_CASE("PassTimings: measured frames summarised, invalid skipped, overflow dropped") {
    PassTimings t = makeTimings();
    t.beginMeasure(10);
    for (u32 i = 1; i <= 12; ++i) { // 12 frames into a capacity of 10
        const float ms[] = {static_cast<float>(i), 1.0f};
        const bool  ok[] = {true, i % 2 == 0}; // unit 1 valid on even frames only
        t.addFrame(i, ms, ok, static_cast<float>(i) * 2.0f);
    }
    CHECK(t.endMeasure() == 10);
    std::vector<PassReport> reports;
    TimingSummary sum, span;
    t.summarize(reports, sum, span);
    REQUIRE(reports.size() == 2);
    // unit 0: frames 1..10
    CHECK(reports[0].gpuMs.min == doctest::Approx(1.0f));
    CHECK(reports[0].gpuMs.max == doctest::Approx(10.0f));
    CHECK(reports[0].gpuMs.mean == doctest::Approx(5.5f));
    CHECK(reports[0].gpuMs.p50 == doctest::Approx(5.0f));
    // unit 1: 5 valid samples of 1.0
    CHECK(reports[1].gpuMs.mean == doctest::Approx(1.0f));
    CHECK(reports[1].gpuMs.max == doctest::Approx(1.0f));
    // per-frame sum of valid units: odd i -> i, even i -> i + 1
    CHECK(sum.max == doctest::Approx(11.0f));
    CHECK(sum.min == doctest::Approx(1.0f));
    CHECK(span.max == doctest::Approx(20.0f));
    CHECK(span.min == doctest::Approx(2.0f));
}

TEST_CASE("PassTimings: unmeasured or unknown values summarise to zero") {
    PassTimings t = makeTimings();
    t.beginMeasure(4);
    const float ms[] = {0.0f, 0.0f};
    const bool  bad[] = {false, false};
    t.addFrame(0, ms, bad, -1.0f);
    t.endMeasure();
    std::vector<PassReport> reports;
    TimingSummary sum, span;
    t.summarize(reports, sum, span);
    REQUIRE(reports.size() == 2);
    CHECK(reports[0].gpuMs.max == 0.0f);
    CHECK(sum.max == 0.0f);
    CHECK(span.max == 0.0f);

    PassTimings empty; // never configured
    empty.summarize(reports, sum, span);
    CHECK(reports.empty());
    empty.addFrame(1, nullptr, nullptr, 1.0f); // no units: no access
    CHECK(empty.lastFrame() == 1);
}

TEST_CASE("PassTimings: measure capacity is fixed, overflow frames are dropped") {
    PassTimings t = makeTimings();
    t.beginMeasure(50);
    // beginMeasure() reserves frames x units; addFrame() only appends inside
    // that capacity and drops the rest, the window uses fixed arrays.  The
    // capacity itself is private, so this checks the observable contract:
    // exactly `capacity` frames kept, the earliest ones.
    const bool ok[] = {true, true};
    for (u32 i = 0; i < 80; ++i) {
        const float ms[] = {1.0f + static_cast<float>(i), 2.0f};
        t.addFrame(i, ms, ok, 3.0f);
    }
    CHECK(t.endMeasure() == 50);
    std::vector<PassReport> reports;
    TimingSummary sum, span;
    t.summarize(reports, sum, span);
    CHECK(reports[0].gpuMs.max == doctest::Approx(50.0f));
    CHECK(reports[0].gpuMs.min == doctest::Approx(1.0f));

    // Restarting a measurement discards the previous one.
    t.beginMeasure(2);
    const float ms[] = {9.0f, 9.0f};
    t.addFrame(100, ms, ok, 1.0f);
    CHECK(t.endMeasure() == 1);
    t.summarize(reports, sum, span);
    CHECK(reports[0].gpuMs.max == doctest::Approx(9.0f));
}

TEST_CASE("PassTimings: configure resets windows and measurements") {
    PassTimings t = makeTimings();
    const bool ok[]  = {true, true};
    const float ms[] = {1.0f, 1.0f};
    t.beginMeasure(3);
    t.addFrame(0, ms, ok, 1.0f);
    t.configure(twoUnitPlan(), kNames, kShaders);
    CHECK(t.lastFrame() == ~0ull);
    CHECK(t.rolling(0).samples == 0);
    CHECK(t.endMeasure() == 0);
}
