#include "rendergraph/barrier_plan.h"

#include <algorithm>
#include <map>
#include <tuple>

namespace phosphor::rg {

namespace {

constexpr u32 kNone = ~0u;

// Stages of an access; attachments that do not name stages run in fragment.
Stages accessStages(const Access& a) {
    if (a.stages != StageNone) return a.stages;
    return isAttachment(a.usage) ? Stages(StageFragment) : Stages(StageNone);
}

// OR of the stages of `pass`'s accesses to `resource`.  `writes`/`reads`
// select which lists to look at.
Stages stagesOf(const PassNode& pass, u32 resource, bool writes, bool reads) {
    Stages s = StageNone;
    if (writes) for (const Access& a : pass.writes) if (a.resource == resource) s |= accessStages(a);
    if (reads)  for (const Access& a : pass.reads)  if (a.resource == resource) s |= accessStages(a);
    return s;
}

bool touches(const PassNode& pass, u32 resource) {
    for (const Access& a : pass.writes) if (a.resource == resource) return true;
    for (const Access& a : pass.reads)  if (a.resource == resource) return true;
    return false;
}

using Key = std::tuple<u32, u8, bool>; // position, scope, aliasing

struct Collector {
    std::map<Key, Barrier> merged;

    void add(u32 position, BarrierScope scope, bool aliasing, Stages after, Stages before, u32 resource) {
        Barrier& b = merged[Key{position, static_cast<u8>(scope), aliasing}];
        b.scope         = scope;
        b.aliasing      = aliasing;
        b.afterStages  |= after;
        b.beforeStages |= before;
        b.resources.push_back(resource);
    }
};

void legalise(Stages& before, const BarrierRules& rules) {
    if (before & rules.rasterUnsupportedBefore) {
        before &= ~rules.rasterUnsupportedBefore;
        before |= rules.rasterPromoteTo;
    }
}

void legaliseAfter(Stages& after, const BarrierRules& rules) {
    if (after & rules.unsupportedAfter) {
        after &= ~rules.unsupportedAfter;
        after |= rules.afterPromoteTo;
    }
}

} // namespace

BarrierRules defaultBarrierRules() {
    // Measured by the F2.3 spike (table in barrier_plan.h).
    BarrierRules rules;
    rules.rasterUnsupportedBefore     = StageTile;
    rules.rasterPromoteTo             = StageGeometry;
    rules.unsupportedAfter            = StageTile;
    rules.afterPromoteTo              = StageFragment;
    rules.rasterForbiddenEncoderAfter = StageFragment | StageTile;
    rules.computeEncoderStages        = StageDispatch | StageBlit | StageAccelerationStructure;
    return rules;
}

void buildBarrierPlan(const RenderGraph& graph, CompiledGraph& compiled, const BarrierRules& rules,
                      BarrierPolicy policy) {
    (void)policy; // OPT-1.4 Minimal: not implemented yet (Conservative)
    compiled.barriers.clear();
    const auto& passes = graph.passes();
    const u32   count  = static_cast<u32>(compiled.order.size());

    auto groupOf = [&](u32 pos) { return pos < compiled.groupOfPosition.size() ? compiled.groupOfPosition[pos] : kNone; };
    auto encoderOf = [&](u32 pos) { return pos < compiled.encoderOfPosition.size() ? compiled.encoderOfPosition[pos] : kNone; };
    auto isRasterAt = [&](u32 pos) {
        const u32 e = encoderOf(pos);
        if (e != kNone && e < compiled.encoders.size()) return compiled.encoders[e].type == PassType::Raster;
        return groupOf(pos) != kNone;
    };
    // Position at which a Queue barrier for a consumer at `pos` is encoded.
    auto queuePosition = [&](u32 pos) {
        const u32 g = groupOf(pos);
        return (g != kNone && g < compiled.renderGroups.size()) ? compiled.renderGroups[g].firstPosition : pos;
    };

    Collector out;

    // --- Dependencies -------------------------------------------------------
    for (const Dependency& d : compiled.dependencies) {
        const u32 pp = compiled.position(d.from);
        const u32 cp = compiled.position(d.to);
        if (pp == kNone || cp == kNone) continue;
        const PassNode& producer = passes[d.from];
        const PassNode& consumer = passes[d.to];
        if (producer.queue != consumer.queue) continue; // buildQueueSyncs

        const u32 gp = groupOf(pp), gc = groupOf(cp);
        if (gp != kNone && gp == gc) continue; // fused: tile memory keeps the order

        Stages after = (d.kind == DepKind::WAR) ? stagesOf(producer, d.resource, false, true)
                                                : stagesOf(producer, d.resource, true, false);
        if (after == StageNone) after = stagesOf(producer, d.resource, true, true);
        Stages before = stagesOf(consumer, d.resource, true, true);

        const u32 ep = encoderOf(pp), ec = encoderOf(cp);
        const bool sameEncoder = ep != kNone && ep == ec;
        const bool raster = isRasterAt(cp);

        if (raster) legalise(before, rules);
        legaliseAfter(after, rules);

        if (sameEncoder) {
            if (raster && (after & rules.rasterForbiddenEncoderAfter)) {
                compiled.errors.push_back("barrier '" + producer.name + "' -> '" + consumer.name +
                                          "' on resource " + std::to_string(d.resource) +
                                          ": encoder-scope barrier inside a render encoder after stages " +
                                          stagesName(after & rules.rasterForbiddenEncoderAfter));
                compiled.ok = false;
                continue;
            }
            if (!raster && ((after | before) & ~rules.computeEncoderStages)) {
                compiled.errors.push_back("barrier '" + producer.name + "' -> '" + consumer.name +
                                          "' on resource " + std::to_string(d.resource) +
                                          ": encoder-scope barrier inside a compute encoder with stages " +
                                          stagesName((after | before) & ~rules.computeEncoderStages));
                compiled.ok = false;
                continue;
            }
            out.add(cp, BarrierScope::Encoder, false, after, before, d.resource);
        } else {
            out.add(queuePosition(cp), BarrierScope::Queue, false, after, before, d.resource);
        }
    }

    // --- Transient first use ------------------------------------------------
    for (const Placement& p : compiled.aliasing.placements) {
        if (p.resource >= compiled.lifetimes.size()) continue;
        const Lifetime& life = compiled.lifetimes[p.resource];
        if (!life.used() || life.first >= count) continue;
        const u32 firstPos = life.first;
        const PassNode& firstPass = passes[compiled.order[firstPos]];

        Stages before         = stagesOf(firstPass, p.resource, true, true);
        Stages afterQueue     = StageNone;
        Stages afterEncoder   = StageNone;
        bool   encoderBarrier = false;
        const u32  enc = encoderOf(firstPos);
        const bool inCompute = enc != kNone && enc < compiled.encoders.size() &&
                               compiled.encoders[enc].type == PassType::Compute;

        for (const Placement& q : compiled.aliasing.placements) {
            if (q.offset >= p.offset + p.size || p.offset >= q.offset + q.size) continue;
            for (u32 pos = 0; pos < count; ++pos) {
                const PassNode& other = passes[compiled.order[pos]];
                if (!touches(other, q.resource)) continue;
                const Stages s = stagesOf(other, q.resource, true, true);
                afterQueue |= s;
                if (inCompute && q.resource != p.resource && pos < firstPos && encoderOf(pos) == enc) {
                    afterEncoder |= s;
                    encoderBarrier = true;
                }
            }
        }

        if (isRasterAt(firstPos)) legalise(before, rules);
        legaliseAfter(afterQueue, rules);
        out.add(queuePosition(firstPos), BarrierScope::Queue, p.aliased, afterQueue, before, p.resource);
        if (encoderBarrier) out.add(firstPos, BarrierScope::Encoder, p.aliased, afterEncoder, before, p.resource);
    }

    // --- Persistent imports: previous frame's accesses ------------------------
    // Same memory every frame: the first access waits for what the previous
    // frame did with it (the other queue's accesses are ordered by events).
    const auto& resources = graph.resources();
    for (u32 r = 0; r < resources.size(); ++r) {
        const ResourceNode& node = resources[r];
        if (!node.imported || (node.importFlags & ImportPerFrame)) continue;
        if (r >= compiled.lifetimes.size() || !compiled.lifetimes[r].used()) continue;
        const Lifetime& life = compiled.lifetimes[r];
        if (life.first >= count) continue;

        bool   written = false;
        Stages after   = StageNone;
        for (u32 pos = life.first; pos <= life.last && pos < count; ++pos) {
            const PassNode& other = passes[compiled.order[pos]];
            if (!touches(other, r)) continue;
            after |= stagesOf(other, r, true, true);
            written |= std::any_of(other.writes.begin(), other.writes.end(),
                                   [&](const Access& a) { return a.resource == r; });
        }
        if (!written) continue; // read-only in the graph: no hazard between frames

        const u32 firstPos = life.first;
        Stages before = stagesOf(passes[compiled.order[firstPos]], r, true, true);
        if (isRasterAt(firstPos)) legalise(before, rules);
        legaliseAfter(after, rules);
        out.add(queuePosition(firstPos), BarrierScope::Queue, false, after, before, r);
    }

    // --- Output ---------------------------------------------------------------
    for (auto& [key, barrier] : out.merged) {
        std::sort(barrier.resources.begin(), barrier.resources.end());
        barrier.resources.erase(std::unique(barrier.resources.begin(), barrier.resources.end()),
                                barrier.resources.end());
        const u32 pos = std::get<0>(key);
        if (compiled.barriers.empty() || compiled.barriers.back().position != pos) {
            compiled.barriers.push_back(PassBarriers{pos, {}});
        }
        compiled.barriers.back().barriers.push_back(std::move(barrier));
    }
}

void buildQueueSyncs(const RenderGraph& graph, CompiledGraph& compiled) {
    compiled.queueSyncs.clear();
    const auto& passes = graph.passes();

    auto groupOf = [&](u32 pos) { return pos < compiled.groupOfPosition.size() ? compiled.groupOfPosition[pos] : kNone; };
    // A render encoder can only signal after it ends and wait before it starts.
    auto signalPos = [&](u32 pos) {
        const u32 g = groupOf(pos);
        return (g != kNone && g < compiled.renderGroups.size()) ? compiled.renderGroups[g].lastPosition : pos;
    };
    auto waitPos = [&](u32 pos) {
        const u32 g = groupOf(pos);
        return (g != kNone && g < compiled.renderGroups.size()) ? compiled.renderGroups[g].firstPosition : pos;
    };

    struct Edge { u32 signal, wait; Queue producerQueue; };
    std::vector<Edge> edges;
    std::vector<u32>  signals;
    for (const Dependency& d : compiled.dependencies) {
        const u32 pp = compiled.position(d.from);
        const u32 cp = compiled.position(d.to);
        if (pp == kNone || cp == kNone) continue;
        if (passes[d.from].queue == passes[d.to].queue) continue;
        edges.push_back({signalPos(pp), waitPos(cp), passes[d.from].queue});
        signals.push_back(signalPos(pp));
    }
    std::sort(signals.begin(), signals.end());
    signals.erase(std::unique(signals.begin(), signals.end()), signals.end());

    std::map<std::pair<u32, u8>, QueueSync> best; // (wait position, producer queue) -> highest value
    for (const Edge& e : edges) {
        const u32 value = static_cast<u32>(std::lower_bound(signals.begin(), signals.end(), e.signal) - signals.begin()) + 1;
        QueueSync& s = best[{e.wait, static_cast<u8>(e.producerQueue)}];
        if (value > s.value) s = QueueSync{e.signal, e.wait, value};
    }
    for (const auto& [key, s] : best) compiled.queueSyncs.push_back(s);
    std::sort(compiled.queueSyncs.begin(), compiled.queueSyncs.end(), [](const QueueSync& a, const QueueSync& b) {
        return std::tie(a.waitBeforePosition, a.value) < std::tie(b.waitBeforePosition, b.value);
    });
}

std::string stagesName(Stages stages) {
    static const struct { Stages bit; const char* name; } kNames[] = {
        {StageVertex, "vertex"}, {StageFragment, "fragment"}, {StageTile, "tile"},
        {StageObject, "object"}, {StageMesh, "mesh"},         {StageDispatch, "dispatch"},
        {StageBlit, "blit"},     {StageAccelerationStructure, "accel"},
    };
    std::string s;
    for (const auto& n : kNames) {
        if (!(stages & n.bit)) continue;
        if (!s.empty()) s += '|';
        s += n.name;
    }
    return s.empty() ? "none" : s;
}

void splitEncodersAtQueueSyncs(CompiledGraph& compiled) {
    if (compiled.queueSyncs.empty()) return;
    const u32 count = static_cast<u32>(compiled.order.size());
    std::vector<bool> waitBefore(count, false), signalAfter(count, false);
    for (const QueueSync& q : compiled.queueSyncs) {
        if (q.waitBeforePosition < count) waitBefore[q.waitBeforePosition] = true;
        if (q.signalAfterPosition < count) signalAfter[q.signalAfterPosition] = true;
    }
    std::vector<EncoderPlan> split;
    for (const EncoderPlan& e : compiled.encoders) {
        if (e.type == PassType::Raster) {
            split.push_back(e);
            continue;
        }
        EncoderPlan run = e;
        for (u32 pos = e.firstPosition; pos <= e.lastPosition; ++pos) {
            const bool cutBefore = pos > run.firstPosition && waitBefore[pos];
            if (cutBefore) {
                run.lastPosition = pos - 1;
                split.push_back(run);
                run.firstPosition = pos;
            }
            if (signalAfter[pos] && pos < e.lastPosition) {
                run.lastPosition = pos;
                split.push_back(run);
                run.firstPosition = pos + 1;
            }
        }
        run.lastPosition = e.lastPosition;
        if (run.firstPosition <= run.lastPosition) split.push_back(run);
    }
    compiled.encoders = std::move(split);
    compiled.encoderOfPosition.assign(count, kNone);
    for (u32 i = 0; i < compiled.encoders.size(); ++i) {
        for (u32 pos = compiled.encoders[i].firstPosition; pos <= compiled.encoders[i].lastPosition; ++pos) {
            compiled.encoderOfPosition[pos] = i;
        }
    }
}

} // namespace phosphor::rg
