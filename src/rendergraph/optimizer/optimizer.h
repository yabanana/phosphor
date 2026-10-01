#pragma once

#include "rendergraph/optimizer/cost_model.h"
#include "rendergraph/optimizer/plan.h"
#include "rendergraph/render_graph.h"

#include <functional>
#include <string>
#include <vector>

namespace phosphor::rg {

// ---------------------------------------------------------------------------
// OPT-1.1 -- offline graph optimiser (decision from OPT-1 spike 2).
//
// For every combination of build choices (rematerialised signals, passes on
// the async queue) the graph is rebuilt and its order searched:
//   1. greedy: the F2 compiler's order (always a candidate: a plan is never
//      worse than greedy by the model);
//   2. exact dynamic programming over downsets (state = passes done + the
//      open render group, fusion exactly as the compiler does it; Pareto
//      front of group costs and peak live bytes per state) while the number
//      of states stays under `dpStateCap`;
//   3. simulated annealing over topological orders judged by the real
//      compiler + evaluateGraph(), from the greedy and the DP orders.
// Each order is compiled with the end-of-F4 policies and, when `tryPolicies`,
// with the OPT-1 ones (Coloring or ColoringStageClass aliasing, Minimal
// barriers); the lowest J wins.
// ---------------------------------------------------------------------------

struct BuildChoices {
    std::vector<std::string> remat;
    std::vector<std::string> async;
};

/// Builds the graph for `choices` into an empty `graph`; false (and *error)
/// if the choices are invalid.
using GraphBuilder = std::function<bool(const BuildChoices& choices, RenderGraph& graph, std::string* error)>;

/// Heap footprints without a GPU: TextureDesc::estimatedBytes() plus the
/// lossless-compression metadata measured in OPT-1 spike 3 (+1/128, 2 KiB
/// alignment), buffers 256-byte aligned.
class EstimatedSizer final : public ResourceSizer {
public:
    [[nodiscard]] SizeAlign textureSize(u32 resource, const TextureDesc& desc) const override;
    [[nodiscard]] SizeAlign bufferSize(u32 resource, const BufferDesc& desc) const override;
};

struct OptimizeOptions {
    std::vector<std::string> rematCandidates;
    std::vector<std::string> asyncCandidates;
    BuildChoices             baselineChoices;  // what "off" builds (end of F4)
    GraphCostParams          cost;
    size_t dpStateCap       = 200000;  // states per DP layer before giving up exactness
    u32    annealIterations = 20000;
    u32    seed             = 1;
    bool   tryPolicies      = true;    // also compile with the OPT-1 policies
};

/// One evaluated candidate (for reports).
struct PlanCandidate {
    BuildChoices choices;
    std::string  method;   // "greedy", "dp", "dp-beam", "anneal"
    AliasPolicy   aliasPolicy   = AliasPolicy::Greedy;
    BarrierPolicy barrierPolicy = BarrierPolicy::Conservative;
    GraphCost    cost;
    std::vector<std::string> order;
};

struct OptimizeResult {
    bool        ok = false;
    std::string error;
    GraphPlan   plan;                      // best candidate
    GraphCost   baseline;                  // "off": baselineChoices, greedy order, F4 policies
    std::vector<PlanCandidate> candidates; // best per (choices, method)
};

[[nodiscard]] OptimizeResult optimize(const std::string& family, const GraphBuilder& builder,
                                      const OptimizeOptions& options);

/// Compile `graph` with `order` (empty: greedy) and the given policies using
/// the estimated sizer, then evaluate it.  `ok` false if compilation fails.
struct Evaluation {
    bool ok = false;
    CompiledGraph compiled;
    GraphCost     cost;
};
[[nodiscard]] Evaluation evaluateOrder(const RenderGraph& graph, const std::vector<u32>& order, AliasPolicy alias,
                                       BarrierPolicy barriers, const GraphCostParams& params);

} // namespace phosphor::rg
