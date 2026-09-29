// async_compute.metal -- synthetic passes of --debug-async-compute (F2.6).
//
// Integer-only chain seed (graphics queue) -> reduce (async compute queue) ->
// consume (graphics queue).  The CPU twin of the arithmetic is
// src/rendergraph/async_probe_reference.h: keep the two in sync.  All
// arithmetic is uint32 (wraps like on the CPU).
//
// Bindings (Metal 4 argument tables):
//   async_seed   : buffer(0) uint frame (constant), buffer(1) device uint[] S (out)
//   async_reduce : buffer(1) device const uint[] S, buffer(2) device uint[] R (out)
//   async_consume: buffer(0) uint frame (constant), buffer(2) device const uint[] R,
//                  buffer(3) device uint[] readback (out)

#include <metal_stdlib>

using namespace metal;

constant uint kGroupSize    = 256;
constant uint kSeedIters    = 128;
constant uint kReduceRounds = 32;
constant uint kConsumeIters = 256;

static uint asyncMix(uint v) {
    v ^= v >> 15;
    v *= 2246822519u;
    v ^= v >> 13;
    v *= 3266489917u;
    v ^= v >> 16;
    return v;
}

kernel void async_seed(constant uint& frame [[buffer(0)]],
                       device uint* s [[buffer(1)]],
                       uint i [[thread_position_in_grid]]) {
    uint v = (i * 73856093u) ^ (frame * 83492791u) ^ 0x9e3779b9u;
    for (uint k = 0; k < kSeedIters; ++k) v = asyncMix(v + k);
    s[i] = v;
}

kernel void async_reduce(device const uint* s [[buffer(1)]],
                         device uint* r [[buffer(2)]],
                         uint j [[thread_position_in_grid]]) {
    device const uint* group = s + j * kGroupSize;
    uint acc = 0;
    for (uint round = 0; round < kReduceRounds; ++round) {
        for (uint k = 0; k < kGroupSize; ++k) acc = asyncMix((acc ^ group[k]) + round);
    }
    r[j] = acc;
}

kernel void async_consume(constant uint& frame [[buffer(0)]],
                          device const uint* r [[buffer(2)]],
                          device uint* out [[buffer(3)]],
                          uint j [[thread_position_in_grid]]) {
    uint v = r[j] ^ (frame * 2654435761u) ^ (j * 40503u);
    for (uint k = 0; k < kConsumeIters; ++k) v = asyncMix(v + k);
    out[j] = v;
}
