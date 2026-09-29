#pragma once

#include "core/types.h"
#include "rendergraph/render_graph.h"

#include <vector>

namespace phosphor::rg {

// ---------------------------------------------------------------------------
// --debug-async-compute (F2.6): a synthetic chain that crosses the graphics
// and the async compute queue twice, with exact integer data.
//
//   1 "Async seed"    (compute, Graphics)     S[i] = seedValue(i, frame)         uint[64K]
//   2 "Async reduce"  (compute, AsyncCompute) R[j] = reduceGroup(S[j*256 ..])    uint[256]
//   3 "Async consume" (compute, Graphics)     out[j] = consumeValue(R[j], j, frame)  imported readback
//
// Every kernel is a loop of mixing rounds so that a missing event between the
// queues shows up as wrong values (the consumer would read stale data).  The
// functions below are the CPU twin of shaders/async_compute.metal: keep them
// in sync (the GPU check fails if not).  All arithmetic is uint32 (wraps).
// ---------------------------------------------------------------------------

constexpr u32 kAsyncSeedCount    = 64 * 1024;
constexpr u32 kAsyncGroupSize    = 256;
constexpr u32 kAsyncResultCount  = kAsyncSeedCount / kAsyncGroupSize; // 256
constexpr u32 kAsyncSeedIters    = 128;
constexpr u32 kAsyncReduceRounds = 32;
constexpr u32 kAsyncConsumeIters = 256;
constexpr u32 kAsyncSeedBytes     = kAsyncSeedCount * sizeof(u32);
constexpr u32 kAsyncResultBytes   = kAsyncResultCount * sizeof(u32);
constexpr u32 kAsyncReadbackSize  = kAsyncResultBytes;

constexpr u32 asyncMix(u32 v) {
    v ^= v >> 15;
    v *= 2246822519u;
    v ^= v >> 13;
    v *= 3266489917u;
    v ^= v >> 16;
    return v;
}

constexpr u32 seedValue(u32 i, u32 frame) {
    u32 v = (i * 73856093u) ^ (frame * 83492791u) ^ 0x9e3779b9u;
    for (u32 k = 0; k < kAsyncSeedIters; ++k) v = asyncMix(v + k);
    return v;
}

/// Reduction of kAsyncGroupSize consecutive seed values.
constexpr u32 reduceGroup(const u32* s) {
    u32 acc = 0;
    for (u32 round = 0; round < kAsyncReduceRounds; ++round) {
        for (u32 k = 0; k < kAsyncGroupSize; ++k) acc = asyncMix((acc ^ s[k]) + round);
    }
    return acc;
}

constexpr u32 consumeValue(u32 r, u32 j, u32 frame) {
    u32 v = r ^ (frame * 2654435761u) ^ (j * 40503u);
    for (u32 k = 0; k < kAsyncConsumeIters; ++k) v = asyncMix(v + k);
    return v;
}

/// Expected readback contents after frame `frame` (kAsyncResultCount words).
void expectedAsyncReadback(u32 frame, std::vector<u32>& out);
/// Number of words of `data` (kAsyncResultCount of them) that differ from the
/// expected contents for `frame`.  Exact, not sampled.
[[nodiscard]] u64 countAsyncMismatches(u32 frame, const u32* data);

struct AsyncProbeRefs {
    BufferRef s, r;
    BufferRef readback;
};

struct AsyncProbeExec {
    ExecuteFn seed, reduce, consume;
};

/// Add the three passes (and the imported readback buffer) to `graph`.  `refs`
/// is filled during the call and must outlive the graph's use.
void addAsyncProbeChain(RenderGraph& graph, AsyncProbeRefs& refs, const AsyncProbeExec& exec);

} // namespace phosphor::rg
