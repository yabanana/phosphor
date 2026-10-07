// F9-S2: per-frame TLAS written by compute from the engine's GPU scene.
//   s2_animate        base GPUInstance[] -> animated GPUInstance[] (rotation about Y + bob)
//   s2_desc_a         one indirect instance descriptor per slot (invalid slots: mask 0)
//   s2_reset/s2_desc_b            compaction with an atomic counter (unstable order)
//   s2_count_blocks/s2_scan_blocks/s2_write_bs   stable compaction (slot order)
//   s2_trace          nearest hit through the TLAS; reports instance_id, userID,
//                     the slot's generation/mesh (read through userID) and the front face
#include "renderer/gpu_types.h"
#include "f9_rt.h"

using namespace phosphor;

// == MTL::IndirectAccelerationStructureInstanceDescriptor (72 B packed)
struct DescOut {
    packed_float3 c0, c1, c2, c3; // column-major 4x3
    uint options, mask, iftOffset, userID;
    ulong blas;
};
static_assert(sizeof(DescOut) == 72, "indirect instance descriptor size");

// Mirrors S2Args in s2_tlas.cpp.
struct S2Args {
    device GPUInstance* inst;
    const device GPUInstance* base;
    device DescOut* desc;
    const device ulong* blas;
    device atomic_uint* count;
    device uint* blockCount;
    uint capacity;
    uint numMeshes;
};

// Per-dispatch parameters, mirrors S2Mode.
struct S2Mode {
    float time;
    uint ccwMode;       // 0 none, 1 CCW option on all, 2 on mirrored only, 3 on non-mirrored only
    uint maskInvalid;   // mask written for invalid slots (0 = masked out; 0xFF = negative control)
    uint transposeBug;  // 1: write the 4x3 transposed (negative control)
    uint mask;          // trace mask
    uint pad0, pad1, pad2;
};

constant uint kOptOpaque = 1u << 2;
constant uint kOptCCW = 1u << 1;

kernel void s2_animate(constant S2Args& a [[buffer(0)]], constant S2Mode& m [[buffer(1)]],
                       uint tid [[thread_position_in_grid]]) {
    if (tid >= a.capacity) return;
    GPUInstance b = a.base[tid];
    const float ph = float(tid & 1023u) * 0.0061359f; // 2*pi/1024
    const float ang = m.time * 0.9f + ph;
    const float bob = 0.35f * sin(m.time * 1.7f + ph * 3.0f);
    float cs, sn;
    sn = sincos(ang, cs);
    // M = Base * Ry(angle): columns 0 and 2 mix, translation gets the bob.
    const float3 b0 = float3(b.modelMatrix[0], b.modelMatrix[1], b.modelMatrix[2]);
    const float3 b2 = float3(b.modelMatrix[8], b.modelMatrix[9], b.modelMatrix[10]);
    const float3 n0 = cs * b0 - sn * b2;
    const float3 n2 = sn * b0 + cs * b2;
    b.modelMatrix[0] = n0.x; b.modelMatrix[1] = n0.y; b.modelMatrix[2] = n0.z;
    b.modelMatrix[8] = n2.x; b.modelMatrix[9] = n2.y; b.modelMatrix[10] = n2.z;
    b.modelMatrix[13] += bob;
    a.inst[tid] = b;
}

inline uint optionsFor(uint flags, uint ccwMode) {
    uint o = kOptOpaque;
    const bool mir = (flags & INSTANCE_FLAG_MIRRORED) != 0;
    if (ccwMode == 1 || (ccwMode == 2 && mir) || (ccwMode == 3 && !mir)) o |= kOptCCW;
    return o;
}

inline DescOut makeDesc(const device GPUInstance& s, uint slot, uint blasId, constant S2Mode& m, bool valid) {
    DescOut d;
    const device float* M = s.modelMatrix;
    if (m.transposeBug != 0) {
        // Row-major data written with a column-major layout: wrong on purpose.
        d.c0 = packed_float3(M[0], M[4], M[8]);
        d.c1 = packed_float3(M[1], M[5], M[9]);
        d.c2 = packed_float3(M[2], M[6], M[10]);
        d.c3 = packed_float3(M[3], M[7], M[11]);
    } else {
        d.c0 = packed_float3(M[0], M[1], M[2]);
        d.c1 = packed_float3(M[4], M[5], M[6]);
        d.c2 = packed_float3(M[8], M[9], M[10]);
        d.c3 = packed_float3(M[12], M[13], M[14]);
    }
    d.options = optionsFor(s.flags, m.ccwMode);
    d.mask = valid ? 0xFFu : m.maskInvalid;
    d.iftOffset = 0;
    d.userID = slot;
    d.blas = 0;
    return d;
}

inline ulong blasOf(constant S2Args& a, uint mesh) { return a.blas[min(mesh, a.numMeshes - 1)]; }

kernel void s2_desc_a(constant S2Args& a [[buffer(0)]], constant S2Mode& m [[buffer(1)]],
                      uint tid [[thread_position_in_grid]]) {
    if (tid >= a.capacity) return;
    const device GPUInstance& s = a.inst[tid];
    const bool valid = (s.flags & INSTANCE_FLAG_VALID) != 0;
    DescOut d = makeDesc(s, tid, 0, m, valid);
    // Invalid slots keep a valid BLAS id (mesh 0) so a masked-out instance is still well formed.
    d.blas = blasOf(a, valid ? s.meshIndex : 0u);
    a.desc[tid] = d;
}

kernel void s2_reset(constant S2Args& a [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    if (tid == 0) atomic_store_explicit(a.count, 0u, memory_order_relaxed);
}

kernel void s2_desc_b(constant S2Args& a [[buffer(0)]], constant S2Mode& m [[buffer(1)]],
                      uint tid [[thread_position_in_grid]]) {
    if (tid >= a.capacity) return;
    const device GPUInstance& s = a.inst[tid];
    if ((s.flags & INSTANCE_FLAG_VALID) == 0) return;
    const uint idx = atomic_fetch_add_explicit(a.count, 1u, memory_order_relaxed);
    DescOut d = makeDesc(s, tid, 0, m, true);
    d.blas = blasOf(a, s.meshIndex);
    a.desc[idx] = d;
}

// ---- stable compaction: 256-slot tiles ------------------------------------------------------
constant uint kTile = 256;

kernel void s2_count_blocks(constant S2Args& a [[buffer(0)]], uint tid [[thread_position_in_grid]],
                            uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                            uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint part[8];
    const uint f = (tid < a.capacity && (a.inst[tid].flags & INSTANCE_FLAG_VALID) != 0) ? 1u : 0u;
    const uint s = simd_sum(f);
    if (lane == 0) part[simd] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0 && lane == 0) {
        uint t = 0;
        for (uint i = 0; i < 8; ++i) t += part[i];
        a.blockCount[tg] = t;
    }
}

// One threadgroup of 1024 threads: exclusive scan of up to 1024 block counts; the total goes to *count.
kernel void s2_scan_blocks(constant S2Args& a [[buffer(0)]], uint tid [[thread_position_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint part[32];
    const uint nBlocks = (a.capacity + kTile - 1) / kTile;
    const uint v = tid < nBlocks ? a.blockCount[tid] : 0u;
    const uint ex = simd_prefix_exclusive_sum(v);
    const uint tot = simd_sum(v);
    if (lane == 0) part[simd] = tot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint base = 0;
    for (uint i = 0; i < simd; ++i) base += part[i]; // <= 31 iterations
    if (tid < nBlocks) a.blockCount[tid] = base + ex;
    if (tid == 1023) atomic_store_explicit(a.count, base + ex + v, memory_order_relaxed);
}

kernel void s2_write_bs(constant S2Args& a [[buffer(0)]], constant S2Mode& m [[buffer(1)]],
                        uint tid [[thread_position_in_grid]], uint tg [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint part[8];
    const bool valid = tid < a.capacity && (a.inst[tid].flags & INSTANCE_FLAG_VALID) != 0;
    const uint f = valid ? 1u : 0u;
    const uint ex = simd_prefix_exclusive_sum(f);
    const uint tot = simd_sum(f);
    if (lane == 0) part[simd] = tot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint base = a.blockCount[tg];
    for (uint i = 0; i < simd; ++i) base += part[i]; // <= 7 iterations
    if (!valid) return;
    const device GPUInstance& s = a.inst[tid];
    DescOut d = makeDesc(s, tid, 0, m, true);
    d.blas = blasOf(a, s.meshIndex);
    a.desc[base + ex] = d;
}

// ---- trace ---------------------------------------------------------------------------------
struct S2Hit {        // 32 B, mirrors S2Hit in s2_tlas.cpp
    float t;          // < 0: miss
    uint instance;    // instance_id (descriptor index)
    uint userID;      // user_instance_id (descriptor userID)
    uint primitive;
    uint generation;  // inst[userID].generation read on the GPU
    uint mesh;        // inst[userID].meshIndex
    uint front;       // triangle_front_facing
    uint flags;       // inst[userID].flags
};

kernel void s2_trace(instance_acceleration_structure as [[buffer(0)]], constant S2Args& a [[buffer(1)]],
                     device const RayIn* rays [[buffer(2)]], device S2Hit* hits [[buffer(3)]],
                     constant S2Mode& m [[buffer(4)]], constant uint& count [[buffer(5)]],
                     uint tid [[thread_position_in_grid]]) {
    if (tid >= count) return;
    const RayIn r = rays[tid];
    intersector<triangle_data, instancing> isect;
    isect.assume_geometry_type(geometry_type::triangle);
    const auto res = isect.intersect(ray(float3(r.o), float3(r.d), r.tmin, r.tmax), as, m.mask);
    S2Hit h;
    h.t = -1.0f; h.instance = ~0u; h.userID = ~0u; h.primitive = ~0u; h.generation = 0; h.mesh = ~0u; h.front = 0; h.flags = 0;
    if (res.type == intersection_type::triangle) {
        h.t = res.distance;
        h.instance = res.instance_id;
        h.userID = res.user_instance_id;
        h.primitive = res.primitive_id;
        h.front = res.triangle_front_facing ? 1u : 0u;
        if (h.userID < a.capacity) {
            h.generation = a.inst[h.userID].generation;
            h.mesh = a.inst[h.userID].meshIndex;
            h.flags = a.inst[h.userID].flags;
        }
    }
    hits[tid] = h;
}
