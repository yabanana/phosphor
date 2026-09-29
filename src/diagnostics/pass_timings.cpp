#include "diagnostics/pass_timings.h"
#include "rendergraph/timing_plan.h"

namespace phosphor {

// Contract stub (F4 contract commit): replaced by the implementation.
void computeUnitTimes(const u64*, u32, const u32*, const u32*, u32 unitCount, double, float* outMs, bool* outValid) {
    for (u32 u = 0; u < unitCount; ++u) {
        outMs[u]    = 0.0f;
        outValid[u] = false;
    }
}

void PassTimings::configure(const rg::TimingPlan&, const std::vector<std::string>&, const std::vector<std::string>&) {
    units_.clear();
}
void PassTimings::addFrame(u64 frameIndex, const float*, const bool*, float) { lastFrame_ = frameIndex; }
PassTimings::UnitStats PassTimings::rolling(u32) const { return {}; }
float PassTimings::rollingSumMs() const { return 0.0f; }
float PassTimings::rollingSpanMs() const { return 0.0f; }
void PassTimings::beginMeasure(u32 frames) { measuring_ = true; measureCapacity_ = frames; }
u32 PassTimings::endMeasure() { measuring_ = false; return measured_; }
void PassTimings::summarize(std::vector<PassReport>& out, TimingSummary&, TimingSummary&) const { out.clear(); }

} // namespace phosphor
