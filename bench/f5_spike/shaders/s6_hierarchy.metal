// F5-S6: transform hierarchy on the GPU.  world = parentWorld * local with
// column-major float4x4, in the same operand order as glm's mat4 product:
//   R[j] = A[0]*B[j][0] + A[1]*B[j][1] + A[2]*B[j][2] + A[3]*B[j][3]
// i.e. per element ((a0*b0 + a1*b1) + a2*b2) + a3*b3.
//   kFma = false: the plain expression (the compiler may contract it in fast
//                 math; whether it does is part of what the spike measures);
//   kFma = true : explicit fma chain fma(a3,b3,fma(a2,b2,fma(a1,b1,a0*b0))),
//                 the exact result of fusing every add of the plain order.
//   kNc         : plain expression with `#pragma METAL fp contract(off)` (no
//                 fused multiply-add: what glm computes on the CPU here);
//   kSkip       : negative control of the dirty propagation (drops one child
//                 per SIMD-group from the next queue).
#include <metal_stdlib>
using namespace metal;

constant bool kFma  [[function_constant(0)]];
constant bool kSkip [[function_constant(1)]];
constant bool kNc   [[function_constant(2)]];

constant uint kNone = 0xFFFFFFFFu;

static float4x4 hmulNc(float4x4 A, float4x4 B) {
#pragma METAL fp contract(off)
    float4x4 R;
    for (uint j = 0; j < 4; ++j) {
        const float4 b = B[j];
        R[j] = A[0] * b.x + A[1] * b.y + A[2] * b.z + A[3] * b.w;
    }
    return R;
}

static float4x4 hmul(float4x4 A, float4x4 B) {
    if (kNc) return hmulNc(A, B);
    float4x4 R;
    for (uint j = 0; j < 4; ++j) {
        const float4 b = B[j];
        if (kFma) {
            R[j] = fma(A[3], float4(b.w), fma(A[2], float4(b.z), fma(A[1], float4(b.y), A[0] * b.x)));
        } else {
            R[j] = A[0] * b.x + A[1] * b.y + A[2] * b.z + A[3] * b.w;
        }
    }
    return R;
}

struct LevelParams {
    uint start, count, pad0, pad1;
};

// (a) one dispatch per level: parents are complete (barrier between levels).
kernel void h_level(device const float4x4* local  [[buffer(0)]],
                    device const uint*     parent [[buffer(1)]],
                    device float4x4*       world  [[buffer(2)]],
                    constant LevelParams&  lp     [[buffer(3)]],
                    uint t [[thread_position_in_grid]]) {
    const uint i = lp.start + t;
    const uint p = parent[i];
    world[i] = p == kNone ? local[i] : hmul(world[p], local[i]);
}

// (b) one dispatch, each thread walks to the root and multiplies top-down:
// ((root * l1) * l2) * ... which is the association of (a).
kernel void h_walk(device const float4x4* local  [[buffer(0)]],
                   device const uint*     parent [[buffer(1)]],
                   device float4x4*       world  [[buffer(2)]],
                   uint i [[thread_position_in_grid]]) {
    uint chain[8];
    uint n = 0;
    uint j = i;
    while (j != kNone && n < 8) {
        chain[n++] = j;
        j = parent[j];
    }
    float4x4 w = local[chain[n - 1]];
    for (int k = int(n) - 2; k >= 0; --k) w = hmul(w, local[chain[k]]);
    world[i] = w;
}

// ---- dirty propagation --------------------------------------------------------
// counts[L] = {tgx, tgy, tgz, count}: the first three words are the
// MTLDispatchThreadgroupsIndirectArguments of level L, `count` the entries of
// queue L (atomic).  Queue L lives at queue[levelStart[L] ...].
struct DirtyParams {
    uint curOff, nextOff, level, last;
};

kernel void h_dirty(device const float4x4* local  [[buffer(0)]],
                    device const uint*     parent [[buffer(1)]],
                    device float4x4*       world  [[buffer(2)]],
                    device uint*           queue  [[buffer(3)]],
                    device atomic_uint*    counts [[buffer(4)]],
                    device const uint2*    cs     [[buffer(5)]], // {first, count} into clist
                    device const uint*     clist  [[buffer(6)]],
                    constant DirtyParams&  P      [[buffer(7)]],
                    uint t    [[thread_position_in_grid]],
                    uint lane [[thread_index_in_simdgroup]]) {
    const uint count = atomic_load_explicit(&counts[P.level * 4u + 3u], memory_order_relaxed);
    uint nchild = 0, first = 0;
    if (t < count) {
        const uint i = queue[P.curOff + t];
        const uint p = parent[i];
        world[i] = p == kNone ? local[i] : hmul(world[p], local[i]);
        if (P.last == 0u) {
            const uint2 c = cs[i];
            first = c.x;
            nchild = c.y;
        }
    }
    if (P.last != 0u) return; // uniform: no SIMD op is skipped by a subset of lanes
    if (kSkip && lane == 0u && nchild > 0u) nchild -= 1u; // negative control: the last child is never queued
    // One atomic per SIMD-group: aggregate the children of the 32 lanes.
    const uint prefix = simd_prefix_exclusive_sum(nchild);
    const uint total  = simd_sum(nchild);
    uint base = 0;
    if (lane == 0u && total > 0u)
        base = atomic_fetch_add_explicit(&counts[(P.level + 1u) * 4u + 3u], total, memory_order_relaxed);
    base = simd_broadcast_first(base);
    for (uint k = 0; k < nchild; ++k) queue[P.nextOff + base + prefix + k] = clist[first + k];
}

// 1 thread: queue `level`'s count -> its indirect dispatch arguments.
kernel void h_args(device uint* counts [[buffer(4)]], constant DirtyParams& P [[buffer(7)]]) {
    const uint c = counts[P.level * 4u + 3u];
    counts[P.level * 4u + 0u] = (c + 63u) / 64u;
    counts[P.level * 4u + 1u] = 1u;
    counts[P.level * 4u + 2u] = 1u;
}
