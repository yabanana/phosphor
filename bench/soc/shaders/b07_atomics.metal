// B-07: 32-bit atomics.
//  * at_dev / at_tg: atomic_fetch_add(+1) on `naddr` addresses (power of two,
//    address = gid & (naddr-1), spaced `stride` words): contention 1 / 32 / 1024 /
//    per-thread.  Every thread sums the values returned by its atomics: the returned
//    old values of one address are a permutation of 0..n-1, so the per-address sum
//    is exactly n(n-1)/2 (verified by the CPU), in addition to the final counters.
//  * compact_*: stream compaction (each thread keeps an element with probability ~1/2):
//    naive = one device atomic per kept element; simd = prefix sum inside the SIMD-group +
//    one device atomic per SIMD-group; simd_tg = one threadgroup atomic per SIMD-group and
//    one device atomic per threadgroup.  out[slot] = gid of the kept element.
#include <metal_stdlib>
using namespace metal;

struct AtParams {
    uint iters;
    uint mask;    // naddr - 1
    uint stride;  // words between addresses
    uint elems;   // compaction: elements per thread
};

kernel void at_dev(device atomic_uint* ctr [[buffer(0)]], device uint* out [[buffer(1)]],
                   constant AtParams& p [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    device atomic_uint* a = ctr + (gid & p.mask) * p.stride;
    uint acc = 0;
    for (uint k = 0; k < p.iters; ++k) acc += atomic_fetch_add_explicit(a, 1u, memory_order_relaxed);
    out[gid] = acc;
}

kernel void at_tg(device uint* ctr [[buffer(0)]], device uint* out [[buffer(1)]],
                  constant AtParams& p [[buffer(2)]], uint gid [[thread_position_in_grid]],
                  uint lid [[thread_index_in_threadgroup]], uint tgid [[threadgroup_position_in_grid]],
                  uint tpt [[threads_per_threadgroup]]) {
    threadgroup atomic_uint tc[256];
    if (lid < 256) atomic_store_explicit(&tc[lid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup atomic_uint* a = tc + (lid & p.mask);
    uint acc = 0;
    for (uint k = 0; k < p.iters; ++k) acc += atomic_fetch_add_explicit(a, 1u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    out[gid] = acc;
    // flush the threadgroup counters (per TG copy, 256 words)
    ctr[tgid * 256u + lid] = atomic_load_explicit(&tc[lid], memory_order_relaxed);
}

// --- Compaction -------------------------------------------------------------------
inline bool keep(uint gid, uint e) {
    uint h = gid * 2654435761u + e * 40503u + 12345u;
    h ^= h >> 15;
    h *= 2246822519u;
    h ^= h >> 13;
    return (h & 1u) != 0u;
}

kernel void compact_naive(device atomic_uint* counter [[buffer(0)]], device uint* slots [[buffer(1)]],
                          constant AtParams& p [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    for (uint e = 0; e < p.elems; ++e)
        if (keep(gid, e)) {
            const uint s = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
            slots[s] = gid * p.elems + e;
        }
}

kernel void compact_simd(device atomic_uint* counter [[buffer(0)]], device uint* slots [[buffer(1)]],
                         constant AtParams& p [[buffer(2)]], uint gid [[thread_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
    for (uint e = 0; e < p.elems; ++e) {
        const uint k = keep(gid, e) ? 1u : 0u;
        const uint pre = simd_prefix_exclusive_sum(k);
        const uint total = simd_sum(k);
        uint base = 0;
        if (lane == 0 && total > 0) base = atomic_fetch_add_explicit(counter, total, memory_order_relaxed);
        base = simd_broadcast_first(base);
        if (k) slots[base + pre] = gid * p.elems + e;
    }
}

kernel void compact_simd_tg(device atomic_uint* counter [[buffer(0)]], device uint* slots [[buffer(1)]],
                            constant AtParams& p [[buffer(2)]], uint gid [[thread_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]], uint lid [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint tgCount;
    threadgroup uint tgBase;
    for (uint e = 0; e < p.elems; ++e) {
        if (lid == 0) atomic_store_explicit(&tgCount, 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint k = keep(gid, e) ? 1u : 0u;
        const uint pre = simd_prefix_exclusive_sum(k);
        const uint total = simd_sum(k);
        uint off = 0;
        if (lane == 0) off = atomic_fetch_add_explicit(&tgCount, total, memory_order_relaxed);
        off = simd_broadcast_first(off);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            const uint n = atomic_load_explicit(&tgCount, memory_order_relaxed);
            tgBase = n > 0 ? atomic_fetch_add_explicit(counter, n, memory_order_relaxed) : 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k) slots[tgBase + off + pre] = gid * p.elems + e;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
