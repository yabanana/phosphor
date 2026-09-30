// B-05: threadgroup memory.
//  * tg_bw: every thread of a 256-thread threadgroup reads 8 words per
//    iteration from a 16 KiB threadgroup array; the word index is
//    lane * stride + uniform offset(k, c) (mod 4096), so within a SIMD-group the
//    access pattern is exactly `stride` words (bank conflicts show as a time
//    increase).  stride is a runtime value.  Results are integer sums: exact.
//  * tg_lat: a single thread chases a random single-cycle permutation stored in
//    threadgroup memory (dependent loads): latency per load.
#include <metal_stdlib>
using namespace metal;

struct TgParams {
    uint iters;
    uint stride;
    uint steps;
    uint pad;
};

constant constexpr uint kWords = 4096;

kernel void tg_bw(device uint* out [[buffer(0)]], constant TgParams& p [[buffer(1)]],
                  device const uint* src [[buffer(2)]], uint lid [[thread_index_in_threadgroup]],
                  uint gid [[thread_position_in_grid]], uint tpt [[threads_per_threadgroup]]) {
    threadgroup uint buf[kWords];
    for (uint j = lid; j < kWords; j += tpt) buf[j] = src[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint a0 = gid, a1 = gid * 2u + 1u, a2 = gid * 3u + 2u, a3 = gid * 4u + 3u;
    uint a4 = gid * 5u + 4u, a5 = gid * 6u + 5u, a6 = gid * 7u + 6u, a7 = gid * 8u + 7u;
    const uint base = lid * p.stride;
    for (uint k = 0; k < p.iters; ++k) {
        const uint o = k * 1237u;
        a0 += buf[(base + o) & (kWords - 1u)];
        a1 += buf[(base + o + 97u) & (kWords - 1u)];
        a2 += buf[(base + o + 194u) & (kWords - 1u)];
        a3 += buf[(base + o + 291u) & (kWords - 1u)];
        a4 += buf[(base + o + 388u) & (kWords - 1u)];
        a5 += buf[(base + o + 485u) & (kWords - 1u)];
        a6 += buf[(base + o + 582u) & (kWords - 1u)];
        a7 += buf[(base + o + 679u) & (kWords - 1u)];
    }
    out[gid] = a0 + a1 * 3u + a2 * 5u + a3 * 7u + a4 * 11u + a5 * 13u + a6 * 17u + a7 * 19u;
}

kernel void tg_lat(device uint* out [[buffer(0)]], constant TgParams& p [[buffer(1)]],
                   device const uint* src [[buffer(2)]], uint lid [[thread_index_in_threadgroup]],
                   uint tpt [[threads_per_threadgroup]]) {
    threadgroup uint buf[kWords];
    for (uint j = lid; j < kWords; j += tpt) buf[j] = src[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        uint idx = 0;
        for (uint s = 0; s < p.steps; ++s) idx = buf[idx];
        out[0] = idx;
    }
}
