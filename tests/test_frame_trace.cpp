#include <doctest/doctest.h>
#include "diagnostics/frame_trace.h"

#include <algorithm>
#include <string>
#include <vector>

using namespace phosphor;

namespace {

FrameRecord frame(u32 index, float cpuMs, u32 flags = 0) {
    FrameRecord f;
    f.index = index;
    f.cpuMs = cpuMs;
    f.flags = flags;
    return f;
}

SwitchRecord sw(u32 frameIndex, float compileMs = 0.0f) {
    SwitchRecord s;
    s.frame = frameIndex;
    s.renderThreadCompileMs = compileMs;
    return s;
}

// n steady frames at 1 ms (p99 = 1, threshold = max(1.5, 1.5) = 1.5).
std::vector<FrameRecord> steadyFrames(u32 n = 100) {
    std::vector<FrameRecord> v;
    for (u32 i = 0; i < n; ++i) v.push_back(frame(i, 1.0f));
    return v;
}

FrameTrace build(const std::vector<FrameRecord>& fr, const std::vector<SwitchRecord>& sws = {}) {
    FrameTrace t;
    t.reserve(static_cast<u32>(fr.size()), static_cast<u32>(sws.size()));
    for (const auto& f : fr) t.addFrame(f);
    for (const auto& s : sws) t.addSwitch(s);
    return t;
}

} // namespace

TEST_CASE("empty trace does not crash") {
    FrameTrace t;
    const HitchReport r = analyzeHitches(t);
    CHECK(r.frames == 0);
    CHECK(r.steadyFrames == 0);
    CHECK(r.switches == 0);
    CHECK(r.hitchSwitches == 0);
    CHECK(r.thresholdMs == 0.0f);
    CHECK(formatHitchReport(r).rfind("SWITCH ", 0) == 0);
    CHECK(traceToCsv(t).rfind("kind,", 0) == 0);
}

TEST_CASE("steady set excludes switch/resize/compile frames and the window after") {
    std::vector<FrameRecord> fr = steadyFrames(30);
    fr[5].flags = FrameBenchSwitch;   // excludes 5..8 (window 3)
    fr[15].flags = FrameResize;       // excludes 15..18
    fr[25].flags = FrameGraphCompile; // excludes only 25
    HitchConfig cfg;
    cfg.window = 3;
    const HitchReport r = analyzeHitches(build(fr), cfg);
    CHECK(r.frames == 30);
    CHECK(r.steadyFrames == 30 - 4 - 4 - 1);
}

TEST_CASE("pipeline swap and fallback draw frames stay steady") {
    std::vector<FrameRecord> fr = steadyFrames(10);
    fr[2].flags = FramePipelineSwap;
    fr[3].flags = FrameFallbackDraw;
    CHECK(analyzeHitches(build(fr)).steadyFrames == 10);
}

TEST_CASE("threshold is the max of the factor and margin forms") {
    auto flat = [](float ms) {
        std::vector<FrameRecord> v;
        for (u32 i = 0; i < 10; ++i) v.push_back(frame(i, ms));
        return analyzeHitches(build(v));
    };
    CHECK(flat(1.0f).steadyCpuP99 == doctest::Approx(1.0f));
    CHECK(flat(1.0f).thresholdMs == doctest::Approx(1.5f));
    CHECK(flat(10.0f).thresholdMs == doctest::Approx(15.0f));  // factor wins
    CHECK(flat(0.1f).thresholdMs == doctest::Approx(0.6f));    // margin wins
}

TEST_CASE("p99 uses the steady set only") {
    std::vector<FrameRecord> fr = steadyFrames(100);
    fr[50].flags = FrameBenchSwitch;
    fr[50].cpuMs = 500.0f;
    HitchConfig cfg;
    cfg.window = 0;
    const HitchReport r = analyzeHitches(build(fr), cfg);
    CHECK(r.steadyFrames == 99);
    CHECK(r.steadyCpuP99 == doctest::Approx(1.0f));
}

TEST_CASE("slow post-switch frame is a hitch") {
    std::vector<FrameRecord> fr = steadyFrames();
    fr[40].flags = FrameBenchSwitch;
    fr[40].cpuMs = 200.0f; // loading cost, not a hitch by itself
    fr[42].cpuMs = 9.0f;
    HitchConfig cfg;
    cfg.window = 5;
    const HitchReport r = analyzeHitches(build(fr, {sw(40)}), cfg);
    CHECK(r.switches == 1);
    CHECK(r.hitchSwitches == 1);
    CHECK(r.framesOverThreshold == 1);
    CHECK(r.worstPostSwitchCpuMs == doctest::Approx(9.0f));
}

TEST_CASE("clean switch is not a hitch") {
    std::vector<FrameRecord> fr = steadyFrames();
    fr[40].flags = FrameBenchSwitch;
    fr[40].cpuMs = 200.0f;
    const HitchReport r = analyzeHitches(build(fr, {sw(40)}));
    CHECK(r.hitchSwitches == 0);
    CHECK(r.framesOverThreshold == 0);
    CHECK(r.worstPostSwitchCpuMs == doctest::Approx(1.0f));
}

TEST_CASE("render-thread compile is a hitch on its own") {
    std::vector<FrameRecord> fr = steadyFrames();
    fr[40].flags = FrameBenchSwitch;
    const HitchReport r = analyzeHitches(build(fr, {sw(40, 3.5f)}));
    CHECK(r.hitchSwitches == 1);
    CHECK(r.framesOverThreshold == 0);
    CHECK(r.renderThreadCompileMs == doctest::Approx(3.5f));
}

TEST_CASE("window stops at the next switch and at the trace end") {
    std::vector<FrameRecord> fr = steadyFrames();
    fr[10].flags = FrameBenchSwitch;
    fr[13].flags = FrameBenchSwitch;
    fr[14].cpuMs = 50.0f; // in switch 2's window, outside switch 1's (stops at 13)
    HitchConfig cfg;
    cfg.window = 10;
    const HitchReport r = analyzeHitches(build(fr, {sw(10), sw(13)}), cfg);
    CHECK(r.hitchSwitches == 1);
    CHECK(r.framesOverThreshold == 1);

    // Window running off the end of the trace, and a switch frame not in the trace.
    const HitchReport e = analyzeHitches(build(steadyFrames(5), {sw(4), sw(999)}), cfg);
    CHECK(e.hitchSwitches == 0);
}

TEST_CASE("no steady frames: only compile time can flag a hitch") {
    std::vector<FrameRecord> fr = {frame(0, 5.0f, FrameBenchSwitch), frame(1, 90.0f)};
    HitchConfig cfg;
    cfg.window = 5; // frame 1 falls in the exclusion window: no steady frames
    CHECK(analyzeHitches(build(fr, {sw(0)}), cfg).steadyFrames == 0);
    CHECK(analyzeHitches(build(fr, {sw(0)}), cfg).hitchSwitches == 0);
    CHECK(analyzeHitches(build(fr, {sw(0, 1.0f)}), cfg).hitchSwitches == 1);
}

TEST_CASE("phase mean and max over switches") {
    SwitchRecord a = sw(1), b = sw(2);
    a.totalMs = 10; a.waitIdleMs = 1; a.setupMs = 2; a.textureUploadMs = 3; a.geometryUploadMs = 1;
    a.gcMs = 1; a.pipelineRequestMs = 0.5f;
    b.totalMs = 30; b.waitIdleMs = 3; b.setupMs = 6; b.textureUploadMs = 5; b.geometryUploadMs = 3;
    b.gcMs = 3; b.pipelineRequestMs = 1.5f;
    const HitchReport r = analyzeHitches(build(steadyFrames(5), {a, b}));
    CHECK(r.totalMean == doctest::Approx(20.0f));
    CHECK(r.totalMax == doctest::Approx(30.0f));
    CHECK(r.waitIdleMean == doctest::Approx(2.0f));
    CHECK(r.setupMax == doctest::Approx(6.0f));
    CHECK(r.uploadMean == doctest::Approx(6.0f));
    CHECK(r.uploadMax == doctest::Approx(8.0f));
    CHECK(r.gcMean == doctest::Approx(2.0f));
    CHECK(r.pipelineMean == doctest::Approx(1.0f));
    CHECK(r.pipelineMax == doctest::Approx(1.5f));
}

TEST_CASE("report line starts with SWITCH and uses two decimals") {
    HitchReport r;
    r.switches = 15;
    r.hitchSwitches = 2;
    r.totalMean = 1.0f;
    const std::string s = formatHitchReport(r);
    CHECK(s.rfind("SWITCH n=15 hitches 2 |", 0) == 0);
    CHECK(s.find("total 1.00/0.00") != std::string::npos);
    CHECK(s.find('\n') == std::string::npos);
}

TEST_CASE("csv has a header, frame rows then switch rows") {
    FrameTrace t;
    t.addFrame(frame(7, 2.5f, FrameResize));
    SwitchRecord s = sw(7);
    s.fromBench = 1; s.toBench = 2; s.totalMs = 12.0f; s.pipelinesRequested = 4;
    t.addSwitch(s);
    const std::string csv = traceToCsv(t);
    const size_t l1 = csv.find('\n');
    const size_t l2 = csv.find('\n', l1 + 1);
    const size_t l3 = csv.find('\n', l2 + 1);
    REQUIRE(l3 != std::string::npos);
    CHECK(csv.rfind("kind,index,", 0) == 0);
    CHECK(csv.substr(l1 + 1, 8) == "frame,7,");
    CHECK(csv.substr(l2 + 1, 9) == "switch,7,");
    CHECK(csv.substr(l3 - 3, 3) == ",4,");
    auto commas = [](const std::string& x) { return std::count(x.begin(), x.end(), ','); };
    CHECK(commas(csv.substr(0, l1)) == commas(csv.substr(l1 + 1, l2 - l1)));
    CHECK(commas(csv.substr(0, l1)) == commas(csv.substr(l2 + 1, l3 - l2)));
}

TEST_CASE("setGpuTimes accepts shorter and longer input") {
    FrameTrace t = build(steadyFrames(4));
    t.setGpuTimes({1.0f, 2.0f});
    CHECK(t.frames()[0].gpuMs == 1.0f);
    CHECK(t.frames()[1].gpuMs == 2.0f);
    CHECK(t.frames()[2].gpuMs == 0.0f);
    t.setGpuTimes({9, 9, 9, 9, 9, 9});
    CHECK(t.frames()[3].gpuMs == 9.0f);
    CHECK(t.frames().size() == 4);
}

TEST_CASE("addFrame within reserved capacity does not reallocate") {
    FrameTrace t;
    t.reserve(8, 2);
    t.addFrame(frame(0, 1.0f));
    const FrameRecord* p = t.frames().data();
    for (u32 i = 1; i < 8; ++i) t.addFrame(frame(i, 1.0f));
    CHECK(t.frames().data() == p);
}

TEST_CASE("threshold is per bench: a hitch on a light bench a global threshold would miss") {
    std::vector<FrameRecord> fr;
    for (u32 i = 0; i < 40; ++i) { fr.push_back(frame(i, 2.0f)); fr.back().bench = 0; }
    for (u32 i = 40; i < 80; ++i) { fr.push_back(frame(i, 0.1f)); fr.back().bench = 1; }
    fr.push_back(frame(80, 30.0f, FrameBenchSwitch)); fr.back().bench = 1;
    for (u32 i = 81; i < 91; ++i) { fr.push_back(frame(i, 0.1f)); fr.back().bench = 1; }
    fr[83 + 0].cpuMs = 0.1f;
    fr[84].cpuMs = 1.0f; // 1 ms: far below the global threshold (3 ms), above bench 1's (0.6 ms)
    const HitchReport r = analyzeHitches(build(fr, {sw(80)}));
    CHECK(r.thresholdMs == doctest::Approx(3.0f)); // global stays in the report
    CHECK(r.hitchSwitches == 1);
    CHECK(r.framesOverThreshold == 1);
    CHECK(r.worstPostSwitchRatio == doctest::Approx(1.0f / 0.6f));
    CHECK(formatHitchReport(r).find("worst ratio 1.67") != std::string::npos);
}

TEST_CASE("bench without steady frames falls back to the global threshold") {
    std::vector<FrameRecord> fr = steadyFrames(60); // bench 0, threshold 1.5
    fr.push_back(frame(60, 20.0f, FrameBenchSwitch));
    fr.back().bench = 3;
    for (u32 i = 61; i < 66; ++i) { fr.push_back(frame(i, 1.0f)); fr.back().bench = 3; }
    fr[63].cpuMs = 3.0f;
    const HitchReport r = analyzeHitches(build(fr, {sw(60)}));
    CHECK(r.hitchSwitches == 1);
    CHECK(r.worstPostSwitchRatio == doctest::Approx(2.0f));
}

TEST_CASE("pipeline hitches distinguish event-pump stalls without hiding renderer regressions") {
    FrameTrace t;
    for (u32 i = 0; i < 40; ++i) {
        FrameRecord f;
        f.index = i;
        f.bench = 0;
        f.cpuMs = 0.1f;
        if (i == 10)
            f.flags = FrameBenchSwitch;
        if (i == 11) {
            f.cpuMs = 20.1f;
            f.eventMs = 20.0f;
        }
        t.addFrame(f);
    }
    SwitchRecord s;
    s.frame = 10;
    t.addSwitch(s);
    const auto r = analyzeHitches(t);
    CHECK(r.hitchSwitches == 0);
    CHECK(r.platformStallFrames == 1);
    CHECK(r.worstEventMs == 20.0f);
    FrameTrace slow;
    for (auto f : t.frames()) {
        if (f.index == 12) {
            f.cpuMs = 4.0f;
            f.eventMs = 0.1f;
        }
        slow.addFrame(f);
    }
    slow.addSwitch(s);
    CHECK(analyzeHitches(slow).hitchSwitches == 1);
}
