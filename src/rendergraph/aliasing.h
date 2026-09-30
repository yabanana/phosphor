#pragma once

#include "rendergraph/render_graph.h"

namespace phosphor::rg {

// ---------------------------------------------------------------------------
// F2.2 -- placement of transient resources in the frame's transient heap.
//
// Candidates: resources that are not imported, are used by a live pass and
// are not memoryless.  Footprints come from the ResourceSizer (on Metal:
// heap*SizeAndAlign, never constants).  Greedy (base for the MILP of
// OPT-1.1): candidates sorted by size (descending, ties by resource index)
// take the lowest aligned offset whose range does not intersect any already
// placed resource with an overlapping lifetime.  Resources touched by an
// AsyncCompute pass are treated as alive for the whole frame (the other
// queue's timeline is not ordered with the positions).
//
// `aliased` is set when the resource's memory range intersects the range of
// another placed resource (whose lifetime therefore does not overlap).
// With `alias` false every resource gets its own range (debug/comparison).
// ---------------------------------------------------------------------------

// OPT-1.3: `policy` Coloring packs by interval colouring (A1, not yet:
// behaves as Greedy); maxLiveSize is filled by every policy.
[[nodiscard]] AliasingPlan planAliasing(const RenderGraph& graph, const CompiledGraph& compiled,
                                        const ResourceSizer& sizer, bool alias = true,
                                        AliasPolicy policy = AliasPolicy::Greedy);

/// True if two placements share at least one byte.
[[nodiscard]] inline bool rangesIntersect(const Placement& a, const Placement& b) {
    return a.offset < b.offset + b.size && b.offset < a.offset + a.size;
}

} // namespace phosphor::rg
