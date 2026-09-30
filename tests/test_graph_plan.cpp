#include "rendergraph/optimizer/cost_model.h"
#include "rendergraph/optimizer/optimizer.h"
#include "rendergraph/optimizer/plan.h"
#include "rendergraph/scenario.h"

#include <doctest/doctest.h>

#include <algorithm>
#include <string>
#include <vector>

using namespace phosphor;
using namespace phosphor::rg;

namespace {

void buildScenarioGraph(RenderGraph& g, u32 index, const ScenarioParams& params = {}) {
    Scenario s;
    const TextureRef drawable = g.importTexture("Drawable", {Format::BGRA8Srgb, 1920, 1080}, ImportOutput | ImportPerFrame);
    std::string error;
    REQUIRE_MESSAGE(buildScenario(index, params, g, drawable, s, {}, &error), error);
}

std::vector<std::string> names(const RenderGraph& g, const std::vector<u32>& order) {
    std::vector<std::string> out;
    for (const u32 p : order) out.push_back(g.passes()[p].name);
    return out;
}

} // namespace

TEST_CASE("graph plan: key is stable and follows the structure") {
    RenderGraph a, b, c;
    buildScenarioGraph(a, 0);
    buildScenarioGraph(b, 0);
    CHECK(graphKey(a) == graphKey(b));

    ScenarioParams wide;
    wide.wideHdr = true;
    buildScenarioGraph(c, 0, wide);
    CHECK(graphKey(a) != graphKey(c)); // formats are part of the key

    RenderGraph d;
    ScenarioParams remat;
    remat.remat = {"Velocity"};
    buildScenarioGraph(d, 0, remat);
    CHECK(graphKey(a) != graphKey(d)); // build choices change the key
}

TEST_CASE("graph plan: JSON round trip") {
    GraphPlan p;
    p.family = "scenario:test";
    p.key    = 0x0123456789abcdefull;
    p.order  = {"A", "B", "C"};
    p.remat  = {"Velocity"};
    p.async  = {"GI probe update"};
    p.aliasPolicy   = AliasPolicy::Coloring;
    p.barrierPolicy = BarrierPolicy::Minimal;
    p.method = "dp";
    p.predicted = {100.0, 200.0, 150.0, 1.5};
    p.baseline  = {120.0, 260.0, 250.0, 1.7};
    std::vector<GraphPlan> back;
    std::string error;
    REQUIRE_MESSAGE(fromJson(toJson({p}), back, &error), error);
    REQUIRE(back.size() == 1);
    CHECK(back[0].family == p.family);
    CHECK(back[0].key == p.key);
    CHECK(back[0].order == p.order);
    CHECK(back[0].remat == p.remat);
    CHECK(back[0].async == p.async);
    CHECK(back[0].aliasPolicy == AliasPolicy::Coloring);
    CHECK(back[0].barrierPolicy == BarrierPolicy::Minimal);
    CHECK(back[0].predicted.heapBytes == doctest::Approx(200.0));
    CHECK(back[0].baseline.timeMs == doctest::Approx(1.7));
    CHECK(findPlan(back, "scenario:test") != nullptr);
    CHECK(findPlan(back, "other") == nullptr);

    // Negative: malformed or unknown content is rejected.
    CHECK_FALSE(fromJson("{\"schema\": 99, \"plans\": []}", back, &error));
    CHECK_FALSE(fromJson("{\"schema\": 1, \"plans\": [{\"family\": \"x\", \"key\": \"zz\", \"order\": []}]}", back, &error));
    CHECK_FALSE(fromJson("{\"schema\": 1, \"plans\": [{\"family\": \"x\", \"key\": \"01\", \"order\": [], \"alias\": \"best\"}]}", back, &error));
    CHECK_FALSE(fromJson("not json", back, &error));
}

TEST_CASE("graph plan: order applies only to the graph it was made for") {
    RenderGraph g;
    buildScenarioGraph(g, 3);
    const CompiledGraph greedy = compileOrder(g);
    REQUIRE(greedy.ok);

    // A plan over every scenario pass except Present (unplanned -> appended).
    GraphPlan plan;
    plan.family = "t";
    for (const std::string& n : names(g, greedy.order)) {
        if (n != "Present") plan.order.push_back(n);
    }
    // Move the two compute passes first (the order the spikes found).
    std::stable_partition(plan.order.begin(), plan.order.end(),
                          [](const std::string& n) { return n == "GI probe update" || n == "Particle simulation"; });
    plan.key = graphKey(g, plan.order);
    std::string error;
    const std::vector<u32> order = planOrder(g, plan, &error);
    REQUIRE_MESSAGE(!order.empty(), error);
    CHECK(g.passes()[order.front()].name == "GI probe update");
    CHECK(g.passes()[order.back()].name == "Present");

    CompileOptions opt;
    opt.order = order;
    const CompiledGraph c = compile(g, opt);
    REQUIRE(c.ok);
    u32 memoryless = 0;
    for (const bool m : c.memoryless) memoryless += m ? 1 : 0;
    CHECK(memoryless == 3); // G-buffer + Lighting fused: albedo, normal, material

    // Negative controls: wrong key, unknown pass, duplicate pass.
    GraphPlan wrongKey = plan;
    wrongKey.key ^= 1;
    CHECK(planOrder(g, wrongKey, &error).empty());
    CHECK(error.find("key") != std::string::npos);
    GraphPlan unknown = plan;
    unknown.order.push_back("No such pass");
    unknown.key = 0;
    CHECK(planOrder(g, unknown, &error).empty());
    GraphPlan dup = plan;
    dup.order.push_back(dup.order.front());
    CHECK(planOrder(g, dup, &error).empty());

    // An order that violates a dependency is rejected by the compiler.
    std::vector<u32> bad = order;
    std::reverse(bad.begin(), bad.end());
    opt.order = bad;
    CHECK_FALSE(compile(g, opt).ok);
}

TEST_CASE("graph cost model: bytes, fixed costs and the frame") {
    RenderGraph g;
    buildScenarioGraph(g, 0);
    const Evaluation e = evaluateOrder(g, {}, AliasPolicy::Greedy, BarrierPolicy::Conservative, {});
    REQUIRE(e.ok);
    CHECK(e.cost.units.size() > 10);
    CHECK(e.cost.dramBytes > 0);
    CHECK(e.cost.heapBytes >= e.cost.maxLiveBytes);
    CHECK(e.cost.maxLiveBytes > 0);
    double sum = 0;
    for (const UnitCost& u : e.cost.units) {
        CHECK(u.ms >= u.memMs);                 // never below the bandwidth bound
        CHECK(u.ms >= u.fixedMs);
        CHECK(u.overlapMs <= u.ms);
        sum += u.ms;
    }
    CHECK(e.cost.sumMs == doctest::Approx(sum));
    CHECK(e.cost.frameMs <= e.cost.sumMs + 1e-9); // one queue: overlap only removes time

    // Twice the bandwidth: the memory part halves, the frame cannot grow.
    GraphCostParams fast;
    fast.dramGBs *= 2;
    const Evaluation f = evaluateOrder(g, {}, AliasPolicy::Greedy, BarrierPolicy::Conservative, fast);
    CHECK(f.cost.frameMs <= e.cost.frameMs);
}

TEST_CASE("graph optimiser: result is never worse than the baseline") {
    const OptimizeResult r = optimize(
        "scenario:async", [](const BuildChoices& ch, RenderGraph& g, std::string* error) {
            ScenarioParams p;
            p.remat = ch.remat;
            p.async = ch.async;
            Scenario s;
            const TextureRef d = g.importTexture("Drawable", {Format::BGRA8Srgb, 1920, 1080}, ImportOutput | ImportPerFrame);
            return buildScenario(3, p, g, d, s, {}, error);
        },
        [] {
            OptimizeOptions o;
            o.baselineChoices.async = scenarioAsyncCandidates(3);
            o.asyncCandidates       = scenarioAsyncCandidates(3);
            return o;
        }());
    REQUIRE_MESSAGE(r.ok, r.error);
    const double planJ = r.plan.predicted.timeMs + GraphCostParams{}.gammaMsPerGiB * r.plan.predicted.heapBytes / (1024.0 * 1024 * 1024);
    CHECK(planJ <= r.baseline.J + 1e-9);
    CHECK_FALSE(r.plan.order.empty());
}
