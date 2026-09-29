#include "rendergraph/timing_plan.h"

namespace phosphor::rg {

// Contract stub (F4 contract commit): replaced by the implementation.
TimingPlan buildTimingPlan(const RenderGraph&, const CompiledGraph&, u32 maxCommits) {
    TimingPlan plan;
    plan.commitStartQueries = maxCommits;
    return plan;
}

std::vector<PassTraffic> estimatePassBandwidth(const RenderGraph& graph, const CompiledGraph&) {
    return std::vector<PassTraffic>(graph.passes().size());
}

} // namespace phosphor::rg
