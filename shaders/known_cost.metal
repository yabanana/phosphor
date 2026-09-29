// known_cost.metal -- the "Known cost" pass of --debug-gpu-cost N (F4.1).
//
// Negative control of the per-pass GPU timings: every thread runs N steps of
// a 32-bit LCG, so the pass time must scale linearly with N while the other
// passes stay unchanged (opt-log, F4 spike: 0..16000 iterations on 1M threads
// scale linearly).  The result is stored so the loop cannot be removed.
//
// Bindings (Metal 4 argument table): buffer(0) uint iterations (constant),
// buffer(1) device uint[] out.

#include <metal_stdlib>

using namespace metal;

kernel void known_cost(constant uint& iterations [[buffer(0)]],
                       device uint* out [[buffer(1)]],
                       uint i [[thread_position_in_grid]]) {
    uint a = i | 1u;
    for (uint k = 0; k < iterations; ++k) a = a * 1664525u + 1013904223u;
    out[i] = a;
}
