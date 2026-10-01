// F5-S4 / F5-S7: instance culling of 1M instances and compaction of the
// visible slots.  The cull test is written with the SAME formulas and the same
// operation order as cullCpu() in bench/f5_spike/s4_cull.cpp (the CPU
// reference); the fast-math and MathModeSafe builds differ from it only by
// FMA contraction / reciprocal division, which the host accounts for with a
// tolerance band around every decision boundary.
//
// Variants:
//   (a) s4_clear + s4_cull_atomic : atomics per bucket, one atomic per
//       SIMD-group when its lanes share the bucket (simd_prefix_exclusive_sum
//       for the lane offsets), per-lane atomic fallback otherwise.
//   (b) s4_flags -> s4_scan_groups -> s4_write_list : stable reduce-then-scan.
//   S7 raster load: s7_vs / s7_fs (full-screen triangles, seeded ALU chain).
#include <metal_stdlib>
using namespace metal;

struct Instance {
    float m[16]; // column-major model matrix
    uint  mesh;  // mesh index == bucket
    uint  pad[3];
};

struct CullParams {
    float planes[20]; // 5 x (nx, ny, nz, d): left, right, bottom, top, near; inside: dot(n, p) + d >= 0
    float cam[3];
    float maxDistance;
    float fwd[3];
    float minPixels;
    float projY;         // proj[1][1]
    float halfViewportH; // viewportHeight * 0.5
    float nearPlane;
    uint  count;
};

constant constexpr uint kGroups = 1024; // N / 1024 threads per group (one threadgroup scans them all)

// One thread's decision.  No early return: callers use SIMD-group operations afterwards.
static bool cullVisible(const device Instance& in, float4 sph, constant CullParams& p) {
    const float cx = sph.x, cy = sph.y, cz = sph.z;
    const float wx = in.m[0] * cx + in.m[4] * cy + in.m[8] * cz + in.m[12];
    const float wy = in.m[1] * cx + in.m[5] * cy + in.m[9] * cz + in.m[13];
    const float wz = in.m[2] * cx + in.m[6] * cy + in.m[10] * cz + in.m[14];
    const float s0 = in.m[0] * in.m[0] + in.m[1] * in.m[1] + in.m[2] * in.m[2];
    const float s1 = in.m[4] * in.m[4] + in.m[5] * in.m[5] + in.m[6] * in.m[6];
    const float s2 = in.m[8] * in.m[8] + in.m[9] * in.m[9] + in.m[10] * in.m[10];
    const float smax   = max(max(s0, s1), s2);
    const float radius = sph.w * sqrt(smax);
    bool vis = true;
    for (uint i = 0; i < 5; ++i) {
        const float dist = p.planes[i * 4 + 0] * wx + p.planes[i * 4 + 1] * wy + p.planes[i * 4 + 2] * wz + p.planes[i * 4 + 3];
        if (dist + radius < 0.0f) vis = false;
    }
    const float dx  = wx - p.cam[0];
    const float dy  = wy - p.cam[1];
    const float dz  = wz - p.cam[2];
    const float len = sqrt(dx * dx + dy * dy + dz * dz);
    if (len - radius > p.maxDistance) vis = false;
    float depth = dx * p.fwd[0] + dy * p.fwd[1] + dz * p.fwd[2];
    depth       = max(depth, p.nearPlane);
    const float diam = 2.0f * radius * p.projY * p.halfViewportH / depth;
    if (diam < p.minPixels) vis = false;
    return vis;
}

// ---- (a) atomics ------------------------------------------------------------------------------
kernel void s4_clear(device uint* counters [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    if (tid < 8u) counters[tid] = 0u;
}

kernel void s4_cull_atomic(const device Instance* inst [[buffer(0)]], const device float4* spheres [[buffer(1)]],
                           constant CullParams& p [[buffer(2)]], device atomic_uint* counters [[buffer(3)]],
                           device uint* list [[buffer(4)]], const device uint* bucketFirst [[buffer(5)]],
                           uint tid [[thread_position_in_grid]]) {
    const device Instance& in = inst[tid];
    const uint bucket = in.mesh;
    const bool vis    = cullVisible(in, spheres[bucket], p);
    const uint first  = bucketFirst[bucket];
    // All lanes active (the host dispatches an exact multiple of 32 threads).
    const bool uniform = simd_all(bucket == simd_broadcast_first(bucket));
    if (uniform) {
        const uint v     = vis ? 1u : 0u;
        const uint rank  = simd_prefix_exclusive_sum(v);
        const uint total = simd_sum(v);
        uint base        = 0u;
        if (simd_is_first() && total != 0u) base = atomic_fetch_add_explicit(&counters[bucket], total, memory_order_relaxed);
        base = simd_broadcast_first(base);
        if (vis) list[first + base + rank] = tid;
    } else if (vis) {
        const uint pos = atomic_fetch_add_explicit(&counters[bucket], 1u, memory_order_relaxed);
        list[first + pos] = tid;
    }
}

// ---- (b) stable reduce-then-scan ------------------------------------------------------------------
// (1) flags + per-group count.
kernel void s4_flags(const device Instance* inst [[buffer(0)]], const device float4* spheres [[buffer(1)]],
                     constant CullParams& p [[buffer(2)]], device uchar* flags [[buffer(3)]],
                     device uint* groupCounts [[buffer(4)]], uint tid [[thread_position_in_grid]],
                     uint gid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint sums[32];
    const device Instance& in = inst[tid];
    const bool vis            = cullVisible(in, spheres[in.mesh], p);
    flags[tid]                = vis ? 1 : 0;
    const uint s              = simd_sum(vis ? 1u : 0u);
    if (lane == 0) sums[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const uint t = simd_sum(sums[lane]);
        if (lane == 0) groupCounts[gid] = t;
    }
}

// (2) one threadgroup of 1024 scans the 1024 group counts: groupOffsets[0..1024] (last = total).
kernel void s4_scan_groups(const device uint* groupCounts [[buffer(0)]], device uint* groupOffsets [[buffer(1)]],
                           uint tid [[thread_position_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint sums[32];
    const uint v    = groupCounts[tid];
    const uint excl = simd_prefix_exclusive_sum(v);
    if (lane == 31) sums[sg] = excl + v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) sums[lane] = simd_prefix_exclusive_sum(sums[lane]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint base   = sums[sg];
    groupOffsets[tid] = base + excl;
    if (tid == kGroups - 1) groupOffsets[kGroups] = base + excl + v;
}

// (3) every visible slot goes to groupOffset + intra-group exclusive prefix; the prefix of every slot
// (and the total at index N) is stored: per bucket first = prefix[bucketFirst], count = difference.
kernel void s4_write_list(const device uchar* flags [[buffer(0)]], const device uint* groupOffsets [[buffer(1)]],
                          device uint* prefix [[buffer(2)]], device uint* list [[buffer(3)]],
                          constant CullParams& p [[buffer(4)]], uint tid [[thread_position_in_grid]],
                          uint gid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint sums[32];
    const uint f    = flags[tid] != 0 ? 1u : 0u;
    const uint excl = simd_prefix_exclusive_sum(f);
    if (lane == 31) sums[sg] = excl + f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) sums[lane] = simd_prefix_exclusive_sum(sums[lane]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint pos = groupOffsets[gid] + sums[sg] + excl;
    prefix[tid]    = pos;
    if (f != 0u) list[pos] = tid;
    if (tid == p.count - 1u) prefix[p.count] = pos + f;
}

// ---- S7 raster load -------------------------------------------------------------------------------
struct S7VOut {
    float4 pos [[position]];
    uint   inst [[flat]];
};
vertex S7VOut s7_vs(uint vid [[vertex_id]], uint iid [[instance_id]]) {
    const float2 p = float2(float((vid << 1) & 2u), float(vid & 2u)); // 0..2 triangle
    S7VOut o;
    o.pos  = float4(p * 2.0f - 1.0f, 0.0f, 1.0f);
    o.inst = iid;
    return o;
}
// Seeded per-pixel LCG chain (not mergeable across lanes); the last instance written wins the pixel.
fragment float4 s7_fs(S7VOut in [[stage_in]], constant uint& iters [[buffer(0)]]) {
    const uint x = uint(in.pos.x), y = uint(in.pos.y);
    uint a = (x * 73856093u) ^ (y * 19349663u) ^ (in.inst * 83492791u + 1u);
    for (uint k = 0; k < iters; ++k) a = a * 1664525u + 1013904223u;
    return float4(float(a & 255u) * (1.0f / 255.0f), float((a >> 8) & 255u) * (1.0f / 255.0f),
                  float((a >> 16) & 255u) * (1.0f / 255.0f), 1.0f);
}
