#include "rendergraph/optimizer/cost_model.h"
#include "rendergraph/timing_plan.h"
#include "diagnostics/soc_model.h"

#include <algorithm>
#include <cmath>

namespace phosphor::rg {

GraphCostParams GraphCostParams::fromSoc(const soc::SocCostModel& m) {
    GraphCostParams p;
    if (m.dramBw.has()) p.dramGBs = m.dramBw.value;
    if (m.i32Mul.has()) p.imadTops = m.i32Mul.value;
    if (m.f32Fma.has()) p.fmaTflops = m.f32Fma.value;
    return p;
}

namespace {

Stages accessStages(const Access& a) {
    if (a.stages != StageNone) return a.stages;
    return isAttachment(a.usage) ? Stages(StageFragment) : Stages(StageNone);
}

struct Interval {
    double begin = 0, end = 0;
};

double overlapLength(const Interval& a, const std::vector<Interval>& others) {
    double sum = 0;
    for (const Interval& o : others) sum += std::max(0.0, std::min(a.end, o.end) - std::max(a.begin, o.begin));
    return std::min(sum, a.end - a.begin);
}

} // namespace

GraphCost evaluateGraph(const RenderGraph& graph, const CompiledGraph& c, const GraphCostParams& P) {
    GraphCost out;
    if (!c.ok) return out;
    const auto& passes = graph.passes();
    const TimingPlan plan = buildTimingPlan(graph, c, 1024);

    // Barrier after-stages per position (queue and encoder scope alike: both
    // make the pass wait for earlier work in those stages).
    std::vector<Stages> waitAfter(c.order.size(), StageNone);
    for (const PassBarriers& pb : c.barriers) {
        for (const Barrier& b : pb.barriers) {
            if (pb.position < waitAfter.size()) waitAfter[pb.position] |= b.afterStages;
            ++out.barriers;
        }
    }

    const size_t n = plan.units.size();
    std::vector<Stages> unitStages(n, StageNone);
    std::vector<Stages> unitWait(n, StageNone);
    out.units.resize(n);
    for (size_t i = 0; i < n; ++i) {
        const TimedUnit& tu = plan.units[i];
        UnitCost& u = out.units[i];
        u.name  = tu.name;
        u.queue = tu.queue;
        u.bytes = static_cast<double>(tu.dramBytes);
        double intOps = 0, flops = 0, tris = 0;
        for (const u32 p : tu.passes) {
            intOps += passes[p].cost.intOps;
            flops  += passes[p].cost.flops;
            tris   += passes[p].cost.triangles;
            for (const auto* list : {&passes[p].reads, &passes[p].writes}) {
                for (const Access& a : *list) unitStages[i] |= accessStages(a);
            }
        }
        for (u32 pos = tu.firstPosition; pos <= tu.lastPosition && pos < waitAfter.size(); ++pos) {
            unitWait[i] |= waitAfter[pos];
        }
        u.memMs   = u.bytes / (P.dramGBs * 1e9) * 1e3;
        u.aluMs   = (intOps / (P.imadTops * 1e12) + flops / (P.fmaTflops * 1e12)) * 1e3;
        u.geoMs   = tris / P.trianglesPerSecond * 1e3;
        u.fixedMs = (tu.kind == TimestampKind::RenderEnd ? P.renderPassUs : P.computePassUs) * 1e-3;
        const double work = std::max(u.memMs, u.aluMs + u.geoMs);
        u.ms = work + u.fixedMs;
        if (work <= 0) {
            u.bound = Bound::Fixed;
        } else if (u.memMs >= u.aluMs + u.geoMs) {
            u.bound = Bound::Memory;
        } else {
            u.bound = u.geoMs > u.aluMs ? Bound::Geometry : Bound::Alu;
        }
        out.sumMs += u.ms;
        out.dramBytes += u.bytes;
    }

    // Queue simulation.
    std::vector<double> end(n, 0.0);
    std::vector<Interval> span(n);
    double t[2] = {0.0, 0.0};
    size_t prev[2] = {~size_t{0}, ~size_t{0}};
    auto unitAt = [&](u32 position) -> size_t {
        return position < plan.unitOfPosition.size() ? plan.unitOfPosition[position] : ~size_t{0};
    };
    for (size_t i = 0; i < n; ++i) {
        const TimedUnit& tu = plan.units[i];
        UnitCost& u = out.units[i];
        const int q = tu.queue == Queue::AsyncCompute ? 1 : 0;
        double start = t[q];
        bool waited = false;
        for (const QueueSync& s : c.queueSyncs) {
            if (s.waitBeforePosition < tu.firstPosition || s.waitBeforePosition > tu.lastPosition) continue;
            const size_t producer = unitAt(s.signalAfterPosition);
            if (producer == ~size_t{0} || producer >= n) continue;
            const double ready = end[producer] + P.eventLatencyMs;
            if (ready > start) {
                start = ready;
                waited = true;
            }
        }
        // Same-queue overlap with the unit just before when no barrier of
        // this unit waits for that unit's stages.
        if (!waited && prev[q] != ~size_t{0} && !(unitWait[i] & unitStages[prev[q]])) {
            const UnitCost& pu = out.units[prev[q]];
            const bool different = pu.bound != u.bound && pu.bound != Bound::Fixed && u.bound != Bound::Fixed;
            const double kappa = different ? P.overlapDifferent : P.overlapSame;
            u.overlapMs = kappa * std::min(u.ms, pu.ms);
        }
        end[i]  = start + u.ms - u.overlapMs;
        span[i] = {start, end[i]};
        t[q]    = end[i];
        prev[q] = i;
    }
    double frame = std::max(t[0], t[1]);
    // Cross-queue sharing: async work that runs beside graphics work slows
    // the pair down (B-19).
    std::vector<Interval> gfx;
    for (size_t i = 0; i < n; ++i) {
        if (out.units[i].queue == Queue::Graphics) gfx.push_back(span[i]);
    }
    for (size_t i = 0; i < n; ++i) {
        if (out.units[i].queue != Queue::AsyncCompute) continue;
        const double share = out.units[i].bound == Bound::Memory ? P.crossQueueBwShare : P.crossQueueAluShare;
        frame += share * overlapLength(span[i], gfx);
    }
    out.frameMs      = frame;
    out.heapBytes    = static_cast<double>(c.aliasing.heapSize);
    out.maxLiveBytes = static_cast<double>(c.aliasing.maxLiveSize);
    out.renderPasses = static_cast<u32>(c.renderGroups.size());
    for (const bool m : c.memoryless) out.memoryless += m ? 1 : 0;
    out.J = out.frameMs + P.gammaMsPerGiB * out.heapBytes / (1024.0 * 1024.0 * 1024.0);
    return out;
}

} // namespace phosphor::rg
