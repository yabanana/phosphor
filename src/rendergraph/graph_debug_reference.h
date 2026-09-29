#pragma once

#include "core/types.h"
#include "rendergraph/render_graph.h"

#include <utility>
#include <vector>

namespace phosphor::rg {

// ---------------------------------------------------------------------------
// --debug-graph-transients (F2.2): a synthetic chain of passes on transient
// resources that exercises the aliasing plan on the device with exact
// integer data.
//
//   1 fill    (compute)  A[x,y]  = fillValue(x, y, frame)            R32Uint 256x256
//   2 reduce  (compute)  B[y]    = sum_x A[x,y]  (mod 2^32)          uint[256]
//   3 expand  (compute)  C[x,y]  = expandValue(B[y], x, y)           R32Uint 256x256
//   4 raster  (render)   D[x,y]  = rasterValue(C[x,y])               R32Uint color attachment
//   5 checksum(compute)  out[y*256+x] = D[x,y] ^ kChecksumMask       imported readback buffer
//
// C can take A's memory (A is dead after pass 2), so the chain crosses
// compute -> compute -> render -> compute encoders and needs aliasing
// barriers.  The reference functions below are the CPU twin of
// shaders/graph_debug.metal: keep them in sync (the GPU check fails if not).
// ---------------------------------------------------------------------------

constexpr u32 kDebugSize         = 256;
constexpr u32 kDebugReadbackSize = kDebugSize * kDebugSize * sizeof(u32);
constexpr u32 kChecksumMask      = 0x5bd1e995u;

constexpr u32 fillValue(u32 x, u32 y, u32 frame) {
    return (x * 73856093u) ^ (y * 19349663u) ^ (frame * 83492791u) ^ (x * y);
}
constexpr u32 expandValue(u32 rowSum, u32 x, u32 y) {
    return (rowSum ^ (x * 2246822519u)) + y * 3266489917u;
}
constexpr u32 rasterValue(u32 c) { return (c * 1664525u + 1013904223u) ^ (c >> 16); }

/// Expected contents of the readback buffer after frame `frame` (row-major).
void expectedReadback(u32 frame, std::vector<u32>& out);
/// Number of values of `data` (kDebugSize^2 words) that differ from the
/// expected contents for `frame`.  Exact, not sampled.
[[nodiscard]] u64 countMismatches(u32 frame, const u32* data);

struct DebugChainRefs {
    TextureRef a, c, d;
    BufferRef  b;
    BufferRef  readback;
};

struct DebugChainExec {
    ExecuteFn fill, reduce, expand, raster, checksum;
};

/// Add the five passes (and the imported readback buffer) to `graph`.  `refs`
/// is filled during the call and must outlive the graph's use (the execute
/// callbacks read it).
void addDebugChain(RenderGraph& graph, DebugChainRefs& refs, const DebugChainExec& exec);

struct DebugAliasSummary {
    u32 aliasedFlags = 0;                         // placements with `aliased` set
    std::vector<std::pair<u32, u32>> sharedPairs; // resources whose ranges intersect
    u64 heapSize      = 0;
    u64 unaliasedSize = 0;
};

[[nodiscard]] DebugAliasSummary summarizeAliasing(const CompiledGraph& compiled);

} // namespace phosphor::rg
