#include "rendergraph/aliasing.h"

#include <algorithm>
#include <functional>
#include <random>

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
    u32      classes = 0; // bit 0 raster, bit 1 compute (ColoringStageClass)
};

struct Range {
    u64 begin = 0;
    u64 end   = 0;
};

} // namespace

AliasingPlan planAliasing(const RenderGraph& graph, const CompiledGraph& compiled, const ResourceSizer& sizer,
                          bool alias, AliasPolicy policy) {
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

    // Stage classes of the live accesses of each resource.
    std::vector<u32> classes(resources.size(), 0);
    for (u32 p = 0; p < passes.size(); ++p) {
        if (p < compiled.culled.size() && compiled.culled[p]) continue;
        const u32 fallback = passes[p].type == PassType::Raster ? 1u : 2u;
        auto add = [&](const Access& a) {
            if (a.resource >= classes.size()) return;
            u32 k = 0;
            if (a.stages & StageRaster) k |= 1u;
            if (a.stages & (StageDispatch | StageBlit | StageAccelerationStructure)) k |= 2u;
            classes[a.resource] |= k ? k : fallback;
        };
        for (const Access& a : passes[p].reads)  add(a);
        for (const Access& a : passes[p].writes) add(a);
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
        c.classes  = classes[r];
        if (async[r] && !compiled.order.empty()) {
            c.life.first = 0;
            c.life.last  = static_cast<u32>(compiled.order.size() - 1);
        }
        candidates.push_back(c);
    }

    // What alias=false produces: every resource back to back in index order.
    u64 unaliased = 0;
    for (const Candidate& c : candidates) unaliased = alignUp(unaliased, c.align) + c.size;
    plan.unaliasedSize = unaliased;

    struct Layout {
        std::vector<Placement> placed; // in placement order
        u64 heapEnd = 0;
    };

    // Every resource back to back in index order (also the fallback).
    auto backToBack = [&]() {
        Layout l;
        for (const Candidate& c : candidates) {
            l.placed.push_back({c.resource, alignUp(l.heapEnd, c.align), c.size, false});
            l.heapEnd = l.placed.back().offset + c.size;
        }
        return l;
    };

    // Packs candidates in `order` (indices into `candidates`).  A placement
    // blocks the candidate when their lifetimes overlap or, with `byClass`,
    // when their stage classes differ.  First fit takes the lowest aligned
    // offset; best fit takes the gap with the least waste (ties: lowest
    // offset), the open-ended last gap only if nothing else fits.
    std::vector<Range> blocking;
    auto pack = [&](const std::vector<u32>& order, bool bestFit, bool byClass) {
        Layout l;
        std::vector<u32> placedIdx;
        for (const u32 ci : order) {
            const Candidate& c = candidates[ci];
            blocking.clear();
            for (size_t k = 0; k < placedIdx.size(); ++k) {
                const Candidate& o = candidates[placedIdx[k]];
                if (o.size == 0) continue;
                if (c.life.overlaps(o.life) || (byClass && c.classes != o.classes))
                    blocking.push_back({l.placed[k].offset, l.placed[k].offset + o.size});
            }
            std::sort(blocking.begin(), blocking.end(),
                      [](const Range& a, const Range& b) { return a.begin < b.begin; });
            u64 cursor = 0, best = 0, bestWaste = ~0ull;
            bool found = false;
            for (const Range& r : blocking) {
                const u64 off = alignUp(cursor, c.align);
                if (off + c.size <= r.begin) {
                    const u64 waste = r.begin - off - c.size;
                    if (!found || waste < bestWaste) { best = off; bestWaste = waste; found = true; }
                    if (!bestFit) break;
                }
                cursor = std::max(cursor, r.end);
            }
            if (!found) best = alignUp(cursor, c.align);
            l.placed.push_back({c.resource, best, c.size, false});
            l.heapEnd = std::max(l.heapEnd, best + c.size);
            placedIdx.push_back(ci);
        }
        return l;
    };

    auto indices = [&](auto less) {
        std::vector<u32> v(candidates.size());
        for (u32 i = 0; i < v.size(); ++i) v[i] = i;
        std::stable_sort(v.begin(), v.end(), less);
        return v;
    };

    Layout layout;
    if (!alias) {
        layout = backToBack();
    } else {
        // F2.2 Greedy: size descending (ties by resource index), first fit.
        const auto sizeDesc = indices([&](u32 x, u32 y) {
            const Candidate &a = candidates[x], &b = candidates[y];
            return a.size != b.size ? a.size > b.size : a.resource < b.resource;
        });
        layout = pack(sizeDesc, false, false);
        // Mixed alignments can make the size-sorted layout pad more than the
        // index-ordered one; never be worse than not aliasing at all.
        if (layout.heapEnd > unaliased) layout = backToBack();

        if (policy != AliasPolicy::Greedy) {
            const bool byClass = policy == AliasPolicy::ColoringStageClass;
            // Stage-class packing cannot start from Greedy (it may mix classes).
            Layout bestLayout = byClass ? backToBack() : layout;
            // Lower bound of the packing (same as maxLiveSize below).
            u64 lower = 0;
            for (u32 pos = 0; pos < compiled.order.size(); ++pos) {
                u64 live = 0;
                for (const Candidate& c : candidates)
                    if (c.life.first <= pos && pos <= c.life.last) live += c.size;
                lower = std::max(lower, live);
            }
            auto consider = [&](const std::vector<u32>& order) {
                for (const bool bestFit : {true, false}) {
                    if (bestLayout.heapEnd <= lower) return;
                    Layout l = pack(order, bestFit, byClass);
                    if (l.heapEnd < bestLayout.heapEnd) bestLayout = std::move(l);
                }
            };
            auto len = [](const Candidate& c) { return u64(c.life.last - c.life.first) + 1; };
            using Cmp = std::function<bool(const Candidate&, const Candidate&)>;
            const std::vector<Cmp> keys = {
                [](const Candidate& a, const Candidate& b) { // start, size desc
                    return a.life.first != b.life.first ? a.life.first < b.life.first : a.size > b.size; },
                [](const Candidate& a, const Candidate& b) { return a.size > b.size; },
                [](const Candidate& a, const Candidate& b) { // end, size desc
                    return a.life.last != b.life.last ? a.life.last < b.life.last : a.size > b.size; },
                [&](const Candidate& a, const Candidate& b) { // area
                    const u64 aa = a.size * len(a), ab = b.size * len(b);
                    return aa != ab ? aa > ab : a.size > b.size; },
                [&](const Candidate& a, const Candidate& b) { // length, size desc
                    return len(a) != len(b) ? len(a) > len(b) : a.size > b.size; },
                [](const Candidate& a, const Candidate& b) { // end descending, size desc
                    return a.life.last != b.life.last ? a.life.last > b.life.last : a.size > b.size; },
            };
            for (const Cmp& k : keys) {
                consider(indices([&](u32 x, u32 y) {
                    const Candidate &a = candidates[x], &b = candidates[y];
                    if (k(a, b)) return true;
                    if (k(b, a)) return false;
                    return a.resource < b.resource;
                }));
            }
            // Deterministic local search: perturb an order with a fixed seed.
            if (bestLayout.heapEnd > lower && candidates.size() > 1) {
                std::mt19937 rng(0x0A11A5u);
                std::vector<u32> cur = indices([&](u32 x, u32 y) {
                    return candidates[x].life.first != candidates[y].life.first
                               ? candidates[x].life.first < candidates[y].life.first
                               : candidates[x].size > candidates[y].size;
                });
                u64 curEnd = pack(cur, true, byClass).heapEnd;
                for (u32 it = 0; it < 400 && bestLayout.heapEnd > lower; ++it) {
                    std::vector<u32> next = cur;
                    const u32 i = rng() % next.size(), j = rng() % next.size();
                    if (rng() & 1) {
                        std::swap(next[i], next[j]);
                    } else {
                        const u32 v = next[i];
                        next.erase(next.begin() + i);
                        next.insert(next.begin() + std::min<size_t>(j, next.size()), v);
                    }
                    Layout l = pack(next, true, byClass);
                    if (l.heapEnd <= curEnd) { cur = next; curEnd = l.heapEnd; }
                    if (l.heapEnd < bestLayout.heapEnd) bestLayout = std::move(l);
                }
            }
            // Ties keep Greedy's layout (bestLayout starts as Greedy unless byClass).
            layout = std::move(bestLayout);
        }
    }
    std::vector<Placement>& placed = layout.placed;
    plan.heapSize = layout.heapEnd;

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
