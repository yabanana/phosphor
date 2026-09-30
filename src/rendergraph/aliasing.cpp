#include "rendergraph/aliasing.h"

#include <algorithm>

namespace phosphor::rg {

namespace {

u64 alignUp(u64 value, u64 align) {
    if (align <= 1) return value;
    return (value + align - 1) / align * align;
}

struct Candidate {
    u32      resource = 0;
    u64      size     = 0;
    u64      align    = 1;
    Lifetime life;
};

struct Range {
    u64 begin = 0;
    u64 end   = 0;
};

} // namespace

AliasingPlan planAliasing(const RenderGraph& graph, const CompiledGraph& compiled, const ResourceSizer& sizer,
                          bool alias, AliasPolicy policy) {
    (void)policy; // OPT-1.3 Coloring: not implemented yet (Greedy)
    AliasingPlan plan;
    const auto& resources = graph.resources();
    const auto& passes    = graph.passes();

    // Resources touched by a live AsyncCompute pass live for the whole frame.
    std::vector<bool> async(resources.size(), false);
    for (u32 p = 0; p < passes.size(); ++p) {
        if (p < compiled.culled.size() && compiled.culled[p]) continue;
        if (passes[p].queue != Queue::AsyncCompute) continue;
        for (const Access& a : passes[p].reads)  if (a.resource < async.size()) async[a.resource] = true;
        for (const Access& a : passes[p].writes) if (a.resource < async.size()) async[a.resource] = true;
    }

    std::vector<Candidate> candidates;
    for (u32 r = 0; r < resources.size(); ++r) {
        const ResourceNode& node = resources[r];
        if (node.imported) continue;
        if (r >= compiled.lifetimes.size() || !compiled.lifetimes[r].used()) continue;
        if (r < compiled.memoryless.size() && compiled.memoryless[r]) continue;

        const SizeAlign sa = node.kind == ResourceKind::Texture ? sizer.textureSize(r, node.texture)
                                                                : sizer.bufferSize(r, node.buffer);
        Candidate c;
        c.resource = r;
        c.size     = sa.size;
        c.align    = std::max<u64>(sa.align, 1);
        c.life     = compiled.lifetimes[r];
        if (async[r] && !compiled.order.empty()) {
            c.life.first = 0;
            c.life.last  = static_cast<u32>(compiled.order.size() - 1);
        }
        candidates.push_back(c);
    }

    // Aliased layout: candidates in placement order, greedy first-fit.
    std::vector<Candidate> sorted = candidates;
    if (alias) {
        std::sort(sorted.begin(), sorted.end(), [](const Candidate& a, const Candidate& b) {
            return a.size != b.size ? a.size > b.size : a.resource < b.resource;
        });
    } // else: resource index order, back to back

    std::vector<Placement> placed;       // in placement order
    std::vector<Lifetime>  placedLife;   // parallel to `placed`
    std::vector<Range>     blocking;
    u64 heapEnd = 0;
    for (const Candidate& c : sorted) {
        u64 offset = 0;
        if (alias) {
            blocking.clear();
            for (size_t i = 0; i < placed.size(); ++i) {
                if (c.life.overlaps(placedLife[i]) && placed[i].size > 0)
                    blocking.push_back({placed[i].offset, placed[i].offset + placed[i].size});
            }
            std::sort(blocking.begin(), blocking.end(),
                      [](const Range& a, const Range& b) { return a.begin < b.begin; });
            offset = 0;
            for (const Range& r : blocking) {
                if (offset + c.size <= r.begin) break; // fits in the gap before r
                offset = std::max(offset, alignUp(r.end, c.align));
            }
        } else {
            offset = alignUp(heapEnd, c.align);
        }
        Placement pl;
        pl.resource = c.resource;
        pl.offset   = offset;
        pl.size     = c.size;
        placed.push_back(pl);
        placedLife.push_back(c.life);
        heapEnd = std::max(heapEnd, offset + c.size);
    }

    // What alias=false produces: every resource back to back in index order.
    u64 unaliased = 0;
    for (const Candidate& c : candidates) unaliased = alignUp(unaliased, c.align) + c.size;
    plan.unaliasedSize = unaliased;

    // Mixed alignments can make the size-sorted layout pad more than the
    // index-ordered one; never be worse than not aliasing at all.
    if (alias && heapEnd > unaliased) {
        u64 end = 0;
        for (size_t i = 0; i < candidates.size(); ++i) {
            placed[i] = {candidates[i].resource, alignUp(end, candidates[i].align), candidates[i].size, false};
            end       = placed[i].offset + placed[i].size;
        }
        heapEnd = end;
    }
    plan.heapSize = heapEnd;

    if (alias) {
        for (size_t i = 0; i < placed.size(); ++i) {
            for (size_t j = 0; j < placed.size(); ++j) {
                if (i != j && rangesIntersect(placed[i], placed[j])) { placed[i].aliased = true; break; }
            }
        }
    }

    // Lower bound of any packing: the largest sum of footprints alive at one
    // position (async resources are alive for the whole frame).
    for (u32 pos = 0; pos < compiled.order.size(); ++pos) {
        u64 live = 0;
        for (const Candidate& c : candidates) {
            if (c.life.first <= pos && pos <= c.life.last) live += c.size;
        }
        plan.maxLiveSize = std::max(plan.maxLiveSize, live);
    }

    std::sort(placed.begin(), placed.end(), [](const Placement& a, const Placement& b) {
        return a.resource < b.resource;
    });
    plan.placements = std::move(placed);
    return plan;
}

} // namespace phosphor::rg
