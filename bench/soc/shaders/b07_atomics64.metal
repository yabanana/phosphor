// B-07: 64-bit atomics.  MSL 4 exposes atomic_max_explicit / atomic_min_explicit
// (no returned value) on `device atomic_ulong` only (no add/exchange/fetch_*, no
// threadgroup): kept in a separate library so a compile failure on an OS/SDK that
// lacks them only marks the 64-bit part as unsupported.  Each thread performs
// `iters` operations with pseudo-random 64-bit values on address gid & mask; the
// CPU recomputes the per-address max/min.
#include <metal_stdlib>
using namespace metal;

struct At64Params {
    uint iters;
    uint mask;
    uint stride;
    uint pad;
};

inline ulong val64(uint gid, uint k) {
    ulong z = (ulong(gid) << 20) + k + 0x9E3779B97F4A7C15ul;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ul;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBul;
    return z ^ (z >> 31);
}

kernel void at64_max(device atomic_ulong* ctr [[buffer(0)]], constant At64Params& p [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
    device atomic_ulong* a = ctr + (gid & p.mask) * p.stride;
    for (uint k = 0; k < p.iters; ++k) atomic_max_explicit(a, val64(gid, k), memory_order_relaxed);
}

kernel void at64_min(device atomic_ulong* ctr [[buffer(0)]], constant At64Params& p [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
    device atomic_ulong* a = ctr + (gid & p.mask) * p.stride;
    for (uint k = 0; k < p.iters; ++k) atomic_min_explicit(a, val64(gid, k), memory_order_relaxed);
}

// 32-bit counterpart with the same values (truncated): baseline for the 64-bit cost.
kernel void at32_max(device atomic_uint* ctr [[buffer(0)]], constant At64Params& p [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
    device atomic_uint* a = ctr + (gid & p.mask) * p.stride;
    for (uint k = 0; k < p.iters; ++k) atomic_fetch_max_explicit(a, uint(val64(gid, k)), memory_order_relaxed);
}
