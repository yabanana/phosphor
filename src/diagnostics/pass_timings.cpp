#include "diagnostics/pass_timings.h"
#include "rendergraph/timing_plan.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace phosphor {

namespace {

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

// Split "a, b,c" into trimmed non-empty names.
std::vector<std::string> splitList(const std::string& list) {
    std::vector<std::string> out;
    size_t pos = 0;
    while (pos <= list.size()) {
        size_t end = list.find(',', pos);
        if (end == std::string::npos) end = list.size();
        size_t b = pos, e = end;
        while (b < e && (list[b] == ' ' || list[b] == '\t')) ++b;
        while (e > b && (list[e - 1] == ' ' || list[e - 1] == '\t')) --e;
        if (e > b) out.emplace_back(list, b, e - b);
        pos = end + 1;
    }
    return out;
}

} // namespace

void computeUnitTimes(const u64* ticks, u32 tickCount, const u32* startQuery, const u32* endQuery, u32 unitCount,
                      double tickNs, float* outMs, bool* outValid) {
    for (u32 u = 0; u < unitCount; ++u) {
        outMs[u]    = 0.0f;
        outValid[u] = false;
        const u32 s = startQuery[u];
        const u32 e = endQuery[u];
        if (s >= tickCount || e >= tickCount) continue;
        const u64 t0 = ticks[s];
        const u64 t1 = ticks[e];
        if (t0 == 0 || t1 == 0 || t1 < t0) continue;
        outMs[u]    = static_cast<float>(static_cast<double>(t1 - t0) * tickNs * 1e-6);
        outValid[u] = true;
    }
}

void PassTimings::configure(const rg::TimingPlan& plan, const std::vector<std::string>& passNames,
                            const std::vector<std::string>& passShaders) {
    units_.clear();
    units_.resize(plan.units.size());
    for (size_t i = 0; i < plan.units.size(); ++i) {
        const rg::TimedUnit& src = plan.units[i];
        Unit& dst                = units_[i];
        dst.name      = src.name;
        dst.queue     = src.queue == rg::Queue::AsyncCompute ? "async" : "graphics";
        dst.fused     = src.fused;
        dst.dramBytes = src.dramBytes;
        for (const u32 pass : src.passes) {
            if (pass < passNames.size()) dst.passes.push_back(passNames[pass]);
            if (pass >= passShaders.size()) continue;
            // passShaders[pass] is the comma-separated list of that pass; the
            // unit's list is the ordered union of the members' lists.
            for (std::string& name : splitList(passShaders[pass])) {
                if (std::find(dst.shaders.begin(), dst.shaders.end(), name) == dst.shaders.end()) {
                    dst.shaders.push_back(std::move(name));
                }
            }
        }
    }
    std::fill(std::begin(sumWindow_), std::end(sumWindow_), 0.0f);
    std::fill(std::begin(spanWindow_), std::end(spanWindow_), 0.0f);
    windowCount_ = 0;
    windowHead_  = 0;
    lastFrame_   = ~0ull;
    measuring_       = false;
    measureCapacity_ = 0;
    measured_        = 0;
    measuredMs_.clear();
    measuredSum_.clear();
    measuredSpan_.clear();
}

void PassTimings::addFrame(u64 frameIndex, const float* ms, const bool* valid, float frameSpanMs) {
    float sum      = 0.0f;
    bool  anyValid = false;
    for (u32 u = 0; u < units_.size(); ++u) {
        Unit& unit = units_[u];
        unit.window[windowHead_]      = valid[u] ? ms[u] : 0.0f;
        unit.windowValid[windowHead_] = valid[u];
        if (valid[u]) {
            sum += ms[u];
            anyValid = true;
        }
    }
    sumWindow_[windowHead_]  = sum;
    spanWindow_[windowHead_] = frameSpanMs;
    windowHead_              = (windowHead_ + 1) % kWindow;
    windowCount_             = std::min(windowCount_ + 1, kWindow);
    lastFrame_               = frameIndex;

    // Never allocates: the vectors were reserved by beginMeasure(); frames
    // beyond the reserved capacity are dropped.
    if (measuring_ && measured_ < measureCapacity_) {
        for (u32 u = 0; u < units_.size(); ++u) measuredMs_.push_back(valid[u] ? ms[u] : kNaN);
        measuredSum_.push_back(anyValid ? sum : kNaN);
        measuredSpan_.push_back(frameSpanMs >= 0.0f ? frameSpanMs : kNaN);
        ++measured_;
    }
}

PassTimings::UnitStats PassTimings::rolling(u32 unit) const {
    UnitStats s;
    if (unit >= units_.size()) return s;
    const Unit& u = units_[unit];
    double total  = 0.0;
    for (u32 i = 0; i < windowCount_; ++i) { // slots [0, count) hold valid data until the ring wraps, then all
        if (!u.windowValid[i]) continue;
        total += u.window[i];
        s.maxMs = std::max(s.maxMs, u.window[i]);
        ++s.samples;
    }
    if (s.samples > 0) s.avgMs = static_cast<float>(total / s.samples);
    return s;
}

float PassTimings::rollingSumMs() const {
    if (windowCount_ == 0) return 0.0f;
    double total = 0.0;
    for (u32 i = 0; i < windowCount_; ++i) total += sumWindow_[i];
    return static_cast<float>(total / windowCount_);
}

float PassTimings::rollingSpanMs() const {
    double total = 0.0;
    u32    n     = 0;
    for (u32 i = 0; i < windowCount_; ++i) {
        if (spanWindow_[i] < 0.0f) continue; // unknown
        total += spanWindow_[i];
        ++n;
    }
    return n ? static_cast<float>(total / n) : 0.0f;
}

void PassTimings::beginMeasure(u32 frames) {
    measuring_       = true;
    measureCapacity_ = frames;
    measured_        = 0;
    measuredMs_.clear();
    measuredSum_.clear();
    measuredSpan_.clear();
    measuredMs_.reserve(static_cast<size_t>(frames) * units_.size());
    measuredSum_.reserve(frames);
    measuredSpan_.reserve(frames);
}

u32 PassTimings::endMeasure() {
    measuring_ = false;
    return measured_;
}

void PassTimings::summarize(std::vector<PassReport>& out, TimingSummary& sum, TimingSummary& span) const {
    auto finite = [](const std::vector<float>& v) {
        std::vector<float> r;
        r.reserve(v.size());
        for (const float x : v) {
            if (!std::isnan(x)) r.push_back(x);
        }
        return r;
    };

    out.clear();
    const size_t n = units_.size();
    for (size_t u = 0; u < n; ++u) {
        PassReport report;
        report.name      = units_[u].name;
        report.queue     = units_[u].queue;
        report.fused     = units_[u].fused;
        report.passes    = units_[u].passes;
        report.shaders   = units_[u].shaders;
        report.dramBytes = units_[u].dramBytes;
        std::vector<float> samples;
        samples.reserve(measured_);
        for (u32 f = 0; f < measured_; ++f) {
            const float x = measuredMs_[static_cast<size_t>(f) * n + u];
            if (!std::isnan(x)) samples.push_back(x);
        }
        report.gpuMs = phosphor::summarize(std::move(samples));
        out.push_back(std::move(report));
    }
    sum  = phosphor::summarize(finite(measuredSum_));
    span = phosphor::summarize(finite(measuredSpan_));
}

} // namespace phosphor
