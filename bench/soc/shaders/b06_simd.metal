// B-06: SIMD-group operations vs their threadgroup-memory equivalents, and
// divergence cost.  4 independent uint chains per thread; each link applies
// one operation (plus one cheap ALU glue op).  "sg_*" kernels use the SIMD-group
// intrinsic, "tg_*" kernels implement the same semantics through threadgroup
// memory with simdgroup_barrier(mem_threadgroup); both produce identical
// results, checked against a CPU model of a 32-lane SIMD-group.
#include <metal_stdlib>
using namespace metal;

struct SimdParams {
    uint iters;
    uint rounds;   // divergence bodies
    uint taken;    // lanes taking the branch (0..32)
    uint pad;
    float xa, ya, xb, yb;
};

enum { SHUF, QUAD, SUM, PREFIX, BALLOT };
constant constexpr uint K = 0x9E3779B9u;

template <int OP>
inline uint sg_step(uint v, uint lane, int c) {
    if (OP == SHUF) return simd_shuffle(v, (lane + 1u + uint(c & 1)) & 31u) + K;
    if (OP == QUAD) return quad_shuffle_xor(v, ushort((c & 1) + 1)) + K;
    if (OP == SUM) return simd_sum(v) + lane;
    if (OP == PREFIX) return simd_prefix_exclusive_sum(v) + K;
    const uint b = uint(ulong(simd_ballot((v & 1u) != 0u)));
    return b ^ (v + lane);
}

template <int OP>
kernel void sg(device uint* out [[buffer(0)]], constant SimdParams& p [[buffer(1)]],
               uint lane [[thread_index_in_simdgroup]], uint gid [[thread_position_in_grid]]) {
    uint v0 = gid + 1u, v1 = gid * 3u + 7919u, v2 = gid * 5u + 15838u, v3 = gid * 7u + 23757u;
    for (uint k = 0; k < p.iters; ++k) {
        v0 = sg_step<OP>(v0, lane, 0);
        v1 = sg_step<OP>(v1, lane, 1);
        v2 = sg_step<OP>(v2, lane, 2);
        v3 = sg_step<OP>(v3, lane, 3);
    }
    out[gid] = v0 + v1 * 3u + v2 * 5u + v3 * 7u;
}

// --- threadgroup-memory equivalents ------------------------------------------------
template <int OP>
inline uint tg_step(threadgroup uint* buf, uint v, uint lid, uint lane, int c) {
    const uint base = lid & ~31u;
    threadgroup uint* b = buf + uint(c) * 256u;
    if (OP == SHUF || OP == QUAD) {
        b[lid] = v;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const uint src = (OP == SHUF) ? (base + ((lane + 1u + uint(c & 1)) & 31u)) : (lid ^ uint((c & 1) + 1));
        const uint r = b[src] + K;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        return r;
    }
    if (OP == SUM || OP == BALLOT) {
        b[lid] = (OP == SUM) ? v : (((v & 1u) != 0u ? 1u : 0u) << lane);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = 16; s > 0; s >>= 1) {
            if (lane < s) b[lid] += b[lid + s];
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        const uint r = b[base];
        simdgroup_barrier(mem_flags::mem_threadgroup);
        return (OP == SUM) ? (r + lane) : (r ^ (v + lane));
    }
    // PREFIX: Hillis-Steele inclusive scan, then exclusive = inclusive - v
    uint incl = v;
    b[lid] = incl;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = 1; off < 32; off <<= 1) {
        const uint t = (lane >= off) ? b[lid - off] : 0u;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        incl += t;
        b[lid] = incl;
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    return (incl - v) + K;
}

template <int OP>
kernel void tgk(device uint* out [[buffer(0)]], constant SimdParams& p [[buffer(1)]],
                uint lane [[thread_index_in_simdgroup]], uint gid [[thread_position_in_grid]],
                uint lid [[thread_index_in_threadgroup]]) {
    threadgroup uint buf[4 * 256];
    uint v0 = gid + 1u, v1 = gid * 3u + 7919u, v2 = gid * 5u + 15838u, v3 = gid * 7u + 23757u;
    for (uint k = 0; k < p.iters; ++k) {
        v0 = tg_step<OP>(buf, v0, lid, lane, 0);
        v1 = tg_step<OP>(buf, v1, lid, lane, 1);
        v2 = tg_step<OP>(buf, v2, lid, lane, 2);
        v3 = tg_step<OP>(buf, v3, lid, lane, 3);
    }
    out[gid] = v0 + v1 * 3u + v2 * 5u + v3 * 7u;
}

#define OPS(OPN, NAME)                                                                                            \
    template [[host_name("sg_" #NAME)]] kernel void sg<OPN>(device uint*, constant SimdParams&, uint, uint);      \
    template [[host_name("tg_" #NAME)]] kernel void tgk<OPN>(device uint*, constant SimdParams&, uint, uint, uint);
OPS(SHUF, shuffle) OPS(QUAD, quad_shuffle) OPS(SUM, sum) OPS(PREFIX, prefix) OPS(BALLOT, ballot)

// --- Divergence ---------------------------------------------------------------------
// Lanes take the branch when ((lane * 13) & 31) < taken (a scattered, runtime-defined
// subset).  Both bodies are `rounds` rounds of 4 independent FMA chains with different
// constants (A: xa/ya, B: xb/yb).  IFELSE: taken -> A, others -> B.  IFONLY: taken -> A,
// others do nothing.
template <bool ELSE>
kernel void dv(device uint* out [[buffer(0)]], constant SimdParams& p [[buffer(1)]],
               uint lane [[thread_index_in_simdgroup]], uint gid [[thread_position_in_grid]]) {
    float a = float(gid & 15u) * 0.0625f + 1.0f, b = a + 0.25f, c = a + 0.5f, d = a + 0.75f;
    const bool take = ((lane * 13u) & 31u) < p.taken;
    for (uint k = 0; k < p.iters; ++k) {
        if (take) {
            for (uint r = 0; r < p.rounds; ++r) {
                a = fma(a, p.xa, p.ya); b = fma(b, p.xa, p.ya);
                c = fma(c, p.xa, p.ya); d = fma(d, p.xa, p.ya);
            }
        } else if (ELSE) {
            for (uint r = 0; r < p.rounds; ++r) {
                a = fma(a, p.xb, p.yb); b = fma(b, p.xb, p.yb);
                c = fma(c, p.xb, p.yb); d = fma(d, p.xb, p.yb);
            }
        }
    }
    out[gid] = as_type<uint>(a) + as_type<uint>(b) * 3u + as_type<uint>(c) * 5u + as_type<uint>(d) * 7u;
}
template [[host_name("dv_ifelse")]] kernel void dv<true>(device uint*, constant SimdParams&, uint, uint);
template [[host_name("dv_ifonly")]] kernel void dv<false>(device uint*, constant SimdParams&, uint, uint);
