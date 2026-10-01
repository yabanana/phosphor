#pragma once

#include "rendergraph/render_graph.h"

#include <string>
#include <vector>

namespace phosphor::rg {

// ---------------------------------------------------------------------------
// OPT-1.1 -- offline graph plans.
//
// A plan is chosen offline (tools/graph_opt, optimizer.h) for one graph
// FAMILY -- the identity of a graph before it is built, e.g.
// "scenario:deferred:2560x1440:v1:w1" -- and records:
//   * build choices that change the graph itself (signals rematerialised by
//     their consumers, passes moved to the async compute queue);
//   * the execution order of the planned passes (by name);
//   * the compile policies (aliasing, barriers);
//   * the KEY of the graph built with those choices (graphKey over the
//     planned passes) and the predicted metrics.
// At run time the plan applies only if the built graph has the same key:
// otherwise it is rejected (logged) and the caller falls back to the greedy
// compiler.  Passes the plan does not name (UI overlay, capture, debug passes
// added by the engine after the planned ones) follow in declaration order.
//
// File format (JSON): {"schema": 1, "plans": [GraphPlan...]}.
// ---------------------------------------------------------------------------

inline constexpr u32 kGraphPlanSchema = 1;

struct PlanMetrics {
    double dramBytes = 0; // estimated DRAM bytes per frame (O1)
    double heapBytes = 0; // transient heap
    double maxLive   = 0; // lower bound of the heap for this order
    double timeMs    = 0; // cost-model frame time
};

struct GraphPlan {
    std::string family;
    u64         key = 0;
    std::vector<std::string> order;   // planned pass names, execution order
    std::vector<std::string> remat;   // build choice: rematerialised signals
    std::vector<std::string> async;   // build choice: passes on the async queue
    AliasPolicy   aliasPolicy   = AliasPolicy::Greedy;
    BarrierPolicy barrierPolicy = BarrierPolicy::Conservative;
    PlanMetrics predicted;            // with this plan
    PlanMetrics baseline;             // same family, compiler of end of F4 ("off")
    std::string method;               // "dp", "anneal", "greedy" (informational)
};

/// Structural hash (FNV-1a 64) of the passes named by `passes` (all passes
/// when empty) and of every resource they touch: pass names, types, queues,
/// hints, accesses (resource name, version, usage, stages, slot, load
/// intent), resource names, kinds, descriptors (except the size of per-frame
/// imports such as the drawable) and import flags.  Stable
/// across runs and platforms (no pointers, no indices of unnamed passes).
[[nodiscard]] u64 graphKey(const RenderGraph& graph, const std::vector<std::string>& passes = {});

/// Execution order (pass indices) for `graph` from `plan`: the planned passes
/// in plan order, then the unplanned live ones in declaration order (culled
/// passes are left out).  Empty and
/// `*error` set if a planned name is missing or duplicated, or the key does
/// not match.  Dependencies are checked by compile().
[[nodiscard]] std::vector<u32> planOrder(const RenderGraph& graph, const GraphPlan& plan, std::string* error = nullptr);

[[nodiscard]] std::string toJson(const std::vector<GraphPlan>& plans);
/// false (and *error) on malformed JSON, another schema or unknown policies.
[[nodiscard]] bool fromJson(const std::string& text, std::vector<GraphPlan>& out, std::string* error = nullptr);

/// The plan for `family`, or nullptr.
[[nodiscard]] const GraphPlan* findPlan(const std::vector<GraphPlan>& plans, const std::string& family);

[[nodiscard]] const char* aliasPolicyName(AliasPolicy p);
[[nodiscard]] const char* barrierPolicyName(BarrierPolicy p);

} // namespace phosphor::rg
