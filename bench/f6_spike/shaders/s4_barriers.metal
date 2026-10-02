// F6-S4: producer/consumer orderings of the F6 mesh path (method of F5-S5):
// a SLOW producer (dependent ALU spin per thread, result stored) whose LAST
// threadgroup to finish (device atomic) writes what the consumer needs, so a
// missing barrier shows as a stale read.  Consumers count what they see.
#include <metal_stdlib>
using namespace metal;

struct Params {
    uint iters;
    uint numTG;
    uint salt;
    uint k; // entries written by the last producer group
};

constant uint kMagic = 0xF6F6F6F6u;

static uint s4_spin(uint seed, uint iters) {
    uint x = seed | 1u;
    for (uint i = 0; i < iters; ++i) {
        x = x * 1664525u + 1013904223u;
        x ^= x >> 15;
    }
    return x;
}

// ---- compute producers ----------------------------------------------------------------

static bool s4_last(device uint* scratch, device atomic_uint* done, constant Params& p, uint gid, uint tid,
                    threadgroup uint& flag) {
    scratch[gid] = s4_spin(gid * 2654435761u + p.salt, p.iters);
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (tid == 0) {
        const uint old = atomic_fetch_add_explicit(done, 1u, memory_order_relaxed);
        flag = (old + 1u == p.numTG) ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return flag != 0u;
}

// Indirect mesh-draw arguments {threadgroupsPerGrid[3]} = {k, 1, 1}.
kernel void s4_prod_mesh_args(device uint* scratch [[buffer(0)]], device atomic_uint* done [[buffer(1)]],
                              device uint* out [[buffer(2)]], constant Params& p [[buffer(3)]],
                              uint gid [[thread_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint flag;
    if (s4_last(scratch, done, p, gid, tid, flag) && tid == 0) {
        out[0] = p.k;
        out[1] = 1u;
        out[2] = 1u;
    }
}

// Data the object shader reads: out[0..k) = magic (one per thread of the last group).
kernel void s4_prod_data(device uint* scratch [[buffer(0)]], device atomic_uint* done [[buffer(1)]],
                         device uint* out [[buffer(2)]], constant Params& p [[buffer(3)]],
                         uint gid [[thread_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint flag;
    if (s4_last(scratch, done, p, gid, tid, flag)) {
        for (uint i = tid; i < p.k; i += 256u) out[i] = kMagic;
    }
}

// Consumer of object-shader writes: counts out[0..k) == magic (one thread each).
kernel void s4_count_magic(device const uint* data [[buffer(2)]], constant Params& p [[buffer(3)]],
                           device atomic_uint* counter [[buffer(5)]], uint gid [[thread_position_in_grid]]) {
    if (gid < p.k && data[gid] == kMagic) atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
}

// ---- mesh-pipeline consumers / producers -----------------------------------------------

struct Payload {
    uint dummy;
};

// m1: one count per object threadgroup launched (indirect args).  m2: one
// count per object threadgroup whose data word is the magic.
[[object]] void s4_object_count(object_data Payload& payload [[payload]], mesh_grid_properties grid,
                                device atomic_uint* counter [[buffer(5)]], device const uint* data [[buffer(2)]],
                                constant Params& p [[buffer(3)]], uint tg [[threadgroup_position_in_grid]],
                                uint lane [[thread_index_in_threadgroup]]) {
    if (lane == 0u) {
        const bool count = p.salt == 0u ? true : data[tg] == kMagic; // salt 0: count launches, else check data
        if (count) atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
        payload.dummy = 0u;
        grid.set_threadgroups_per_grid(uint3(0u, 1u, 1u));
    }
}

// o1: SLOW object shader producer: every object thread spins, the last object
// threadgroup to finish writes out[0..k) = magic (the compute consumer counts).
[[object]] void s4_object_write(object_data Payload& payload [[payload]], mesh_grid_properties grid,
                                device uint* scratch [[buffer(0)]], device atomic_uint* done [[buffer(1)]],
                                device uint* out [[buffer(2)]], constant Params& p [[buffer(3)]],
                                uint gid [[thread_position_in_grid]], uint tg [[threadgroup_position_in_grid]],
                                uint lane [[thread_index_in_threadgroup]]) {
    threadgroup uint flag;
    scratch[gid] = s4_spin(gid * 2654435761u + p.salt, p.iters);
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (lane == 0u) {
        const uint old = atomic_fetch_add_explicit(done, 1u, memory_order_relaxed);
        flag = (old + 1u == p.numTG) ? 1u : 0u;
        payload.dummy = 0u;
        grid.set_threadgroups_per_grid(uint3(0u, 1u, 1u));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (flag != 0u) {
        for (uint i = lane; i < p.k; i += 32u) out[i] = kMagic;
    }
    (void)tg;
}

struct VOut {
    float4 pos [[position]];
};
using EmptyMesh = metal::mesh<VOut, void, 3, 1, topology::triangle>;
[[mesh]] void s4_mesh_empty(EmptyMesh out, const object_data Payload& payload [[payload]]) {
    (void)payload;
    out.set_primitive_count(0u);
}
fragment half4 s4_fs() { return half4(0.0h); }
