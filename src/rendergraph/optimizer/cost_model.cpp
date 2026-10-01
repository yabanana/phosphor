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

    // Queue simulation.  Units are issued in order on their queue: a unit
    // starts no earlier than the one before it started, and after the end of
    // every earlier unit of its queue whose stages one of its barriers waits
    // for (the barrier is stage-wide: it cannot tell passes apart), and after
    // the producers of its cross-queue waits (+ event latency).  Running
    // beside earlier units that did not finish costs a sharing penalty
    // proportional to the concurrent time: 1 - overlapSame for the same
    // bound (B-19), 1 - overlapDifferent otherwise (OPT-1 spike 5).
    std::vector<double> begin(n, 0.0), end(n, 0.0);
    std::vector<Interval> span(n);
    double issued[2] = {0.0, 0.0};
    double finished[2] = {0.0, 0.0};
    auto unitAt = [&](u32 position) -> size_t {
        return position < plan.unitOfPosition.size() ? plan.unitOfPosition[position] : ~size_t{0};
    };
    for (size_t i = 0; i < n; ++i) {
        const TimedUnit& tu = plan.units[i];
        UnitCost& u = out.units[i];
        const int q = tu.queue == Queue::AsyncCompute ? 1 : 0;
        double start = issued[q];
        for (size_t j = 0; j < i; ++j) {
            if (out.units[j].queue != u.queue) continue;
            if (unitWait[i] & unitStages[j]) start = std::max(start, end[j]);
        }
        for (const QueueSync& s : c.queueSyncs) {
            if (s.waitBeforePosition < tu.firstPosition || s.waitBeforePosition > tu.lastPosition) continue;
            const size_t producer = unitAt(s.signalAfterPosition);
            if (producer == ~size_t{0} || producer >= n) continue;
            start = std::max(start, end[producer] + P.eventLatencyMs);
        }
        double penalty = 0, concurrent = 0;
        for (size_t j = 0; j < i; ++j) {
            if (out.units[j].queue != u.queue || end[j] <= start) continue;
            const double len = std::min(end[j] - start, u.ms);
            const UnitCost& o = out.units[j];
            const bool different = o.bound != u.bound && o.bound != Bound::Fixed && u.bound != Bound::Fixed;
            penalty += len * (1.0 - (different ? P.overlapDifferent : P.overlapSame));
            concurrent += len;
        }
        u.overlapMs = std::min(concurrent, u.ms);
        begin[i]    = start;
        end[i]      = start + u.ms + penalty;
        span[i]     = {begin[i], end[i]};
        issued[q]   = start;
        finished[q] = std::max(finished[q], end[i]);
    }
    double t[2] = {finished[0], finished[1]};
    double frame = std::max(t[0], t[1]);
    // Cross-queue sharing: async work that runs beside graphics work slows
    // the pair down.  Per overlapping pair: the same bound -> 1 - overlapSame
    // of the concurrent time (like two passes on one queue, B-19 0.84);
    // otherwise the B-19 cross-queue share of the async unit's bound
    // (0.78 of the sum for bandwidth, 0.98 for ALU).
    for (size_t i = 0; i < n; ++i) {
        const UnitCost& a = out.units[i];
        if (a.queue != Queue::AsyncCompute) continue;
        for (size_t j = 0; j < n; ++j) {
            const UnitCost& g = out.units[j];
            if (g.queue != Queue::Graphics) continue;
            const double len = std::max(0.0, std::min(span[i].end, span[j].end) - std::max(span[i].begin, span[j].begin));
            if (len <= 0) continue;
            const bool same = a.bound == g.bound && a.bound != Bound::Fixed;
            const double share = same ? 1.0 - P.overlapSame
                                      : (a.bound == Bound::Memory ? P.crossQueueBwShare : P.crossQueueAluShare);
            frame += share * len;
        }
    }
    out.frameMs      = frame;
    out.heapBytes    = static_cast<double>(c.aliasing.heapSize);
    out.maxLiveBytes = static_cast<double>(c.aliasing.maxLiveSize);
    out.renderPasses = static_cast<u32>(c.renderGroups.size());
    for (const bool m : c.memoryless) out.memoryless += m ? 1 : 0;
    out.J = out.frameMs + P.gammaMsPerGiB * out.heapBytes / (1024.0 * 1024.0 * 1024.0) +
            P.betaMsPerGiB * out.dramBytes / (1024.0 * 1024.0 * 1024.0);
    return out;
}

} // namespace phosphor::rg
