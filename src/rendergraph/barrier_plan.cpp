#include "rendergraph/barrier_plan.h"

namespace phosphor::rg {

// STUB (F2.3): to be implemented.
BarrierRules defaultBarrierRules() { return {}; }
void buildBarrierPlan(const RenderGraph&, CompiledGraph&, const BarrierRules&) {}
void buildQueueSyncs(const RenderGraph&, CompiledGraph&) {}
std::string stagesName(Stages) { return {}; }

} // namespace phosphor::rg
