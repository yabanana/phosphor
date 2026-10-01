#pragma once

#include "rendergraph/graph_budget.h"
#include "rendergraph/render_graph.h"

#include <string>
#include <vector>

namespace phosphor::rg {

// ---------------------------------------------------------------------------
// F2.8 -- Graphviz dump and DRAM bandwidth estimate per resource (O1).
//
// Estimated bytes per access (one full copy = TextureDesc::estimatedBytes()
// or BufferDesc::size):
//   attachment:  read one copy if its group loads it, write one copy if the
//                group stores it; nothing for memoryless/Clear/DontCare
//                (tile memory); counted once per group, not per pass;
//   ShaderRead, CopySrc, IndirectArgs: read one copy;
//   ShaderWrite, CopyDst: write one copy.
// Culled passes contribute nothing.
// ---------------------------------------------------------------------------

struct ResourceTraffic {
    u32 resource   = 0;
    u64 readBytes  = 0;
    u64 writeBytes = 0;
};

struct BandwidthReport {
    std::vector<ResourceTraffic> resources; // one entry per resource
    u64 totalReadBytes  = 0;
    u64 totalWriteBytes = 0;
    [[nodiscard]] u64 totalBytes() const { return totalReadBytes + totalWriteBytes; }
};

[[nodiscard]] BandwidthReport estimateBandwidth(const RenderGraph& graph, const CompiledGraph& compiled);

/// Graphviz (dot) text: passes as boxes (culled ones dashed, fused ones in a
/// cluster per render group, queue shown), resource versions as ellipses
/// labelled with format/size, memoryless/aliased/imported flags, load/store
/// actions and estimated bytes; edges for every access; barriers listed on
/// the consumer pass; a graph label with the DRAM total per frame and the
/// transient heap size (aliased vs unaliased).  OPT-1.5/1.6/1.7: each live
/// pass box also shows its estimated DRAM bytes; the graph label adds the tier
/// budget lines (measured / external flagged), SLC flags and reuse candidates
/// (graph_budget.h), and the lint findings (graph_lint.h).
[[nodiscard]] std::string dumpGraphviz(const RenderGraph& graph, const CompiledGraph& compiled,
                                       const BudgetOptions& budget = {});

} // namespace phosphor::rg
