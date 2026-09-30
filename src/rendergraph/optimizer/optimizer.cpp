#include "rendergraph/optimizer/optimizer.h"

#include <algorithm>

namespace phosphor::rg {

namespace {

u64 alignUp(u64 v, u64 a) { return (v + a - 1) / a * a; }

std::vector<std::string> orderNames(const RenderGraph& graph, const CompiledGraph& c) {
    std::vector<std::string> names;
    for (const u32 p : c.order) names.push_back(graph.passes()[p].name);
    return names;
}

} // namespace

SizeAlign EstimatedSizer::textureSize(u32, const TextureDesc& desc) const {
    // OPT-1 spike 3: compressible private textures carry +1/128 of metadata
    // and a 2048-byte alignment (heapTextureSizeAndAlign, M5 Max).
    const u64 bytes = desc.estimatedBytes();
    return {alignUp(bytes + bytes / 128, 2048), 2048};
}

SizeAlign EstimatedSizer::bufferSize(u32, const BufferDesc& desc) const { return {alignUp(desc.size, 256), 256}; }

Evaluation evaluateOrder(const RenderGraph& graph, const std::vector<u32>& order, AliasPolicy alias,
                         BarrierPolicy barriers, const GraphCostParams& params) {
    static const EstimatedSizer sizer;
    CompileOptions opt;
    opt.sizer         = &sizer;
    opt.order         = order;
    opt.aliasPolicy   = alias;
    opt.barrierPolicy = barriers;
    Evaluation e;
    e.compiled = compile(graph, opt);
    e.ok       = e.compiled.ok;
    if (e.ok) e.cost = evaluateGraph(graph, e.compiled, params);
    return e;
}

OptimizeResult optimize(const std::string& family, const GraphBuilder& builder, const OptimizeOptions& options) {
    OptimizeResult result;
    {
        RenderGraph g;
        if (!builder(options.baselineChoices, g, &result.error)) return result;
        const Evaluation base = evaluateOrder(g, {}, AliasPolicy::Greedy, BarrierPolicy::Conservative, options.cost);
        if (!base.ok) {
            result.error = base.compiled.errors.empty() ? "baseline does not compile" : base.compiled.errors.front();
            return result;
        }
        result.baseline = base.cost;
        PlanCandidate cand;
        cand.choices = options.baselineChoices;
        cand.method  = "greedy";
        cand.cost    = base.cost;
        cand.order   = orderNames(g, base.compiled);
        result.candidates.push_back(cand);

        GraphPlan& plan = result.plan;
        plan.family = family;
        plan.order  = cand.order;
        plan.remat  = cand.choices.remat;
        plan.async  = cand.choices.async;
        plan.key    = graphKey(g, plan.order);
        plan.method = cand.method;
        plan.predicted = {base.cost.dramBytes, base.cost.heapBytes, base.cost.maxLiveBytes, base.cost.frameMs};
        plan.baseline  = plan.predicted;
    }
    // OPT-1.1: the searches (DP over downsets, annealing) come next.
    result.ok = true;
    return result;
}

} // namespace phosphor::rg
