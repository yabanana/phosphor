// F5-S5: which consumer-side barrier stage makes compute-written indirect
// arguments / ICB commands visible.  Producers spin (a per-thread dependent
// ALU chain, result stored so it cannot be removed) and the LAST threadgroup
// to finish (device atomic counter) writes the consumer's arguments, so the
// write really happens after every spin.  Consumers count what they run.
#include <metal_stdlib>
using namespace metal;

struct Params {
    uint iters;
    uint numTG;
    uint salt;
    uint k; // ICB commands encoded by the GPU
};

struct ICBContainer {
    command_buffer icb [[id(0)]];
};

// Dependent chain: x -> x*a + c, xor-shifted.  Seeded per thread (the compiler
// merges uniform chains) and iterated a run-time number of times.
static uint s5_spin(uint seed, uint iters) {
    uint x = seed | 1u;
    for (uint i = 0; i < iters; ++i) {
        x = x * 1664525u + 1013904223u;
        x ^= x >> 15;
    }
    return x;
}

// Spin, then elect the last threadgroup.  Every thread of the group must have
// finished spinning before thread 0 counts the group as done.
static bool s5_run(device uint* scratch, device atomic_uint* done, constant Params& p, uint gid, uint tid,
                   threadgroup uint& flag) {
    scratch[gid] = s5_spin(gid * 2654435761u + p.salt, p.iters);
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (tid == 0) {
        const uint old = atomic_fetch_add_explicit(done, 1u, memory_order_relaxed);
        flag = (old + 1u == p.numTG) ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return flag != 0u;
}

#define S5_PROD_PARAMS                                                                                          \
    device uint* scratch [[buffer(0)]], device atomic_uint* done [[buffer(1)]], device uint* out [[buffer(2)]], \
        constant Params& p [[buffer(3)]], uint gid [[thread_position_in_grid]],                                  \
        uint tid [[thread_index_in_threadgroup]]

// C1: MTLDrawIndexedPrimitivesIndirectArguments {indexCount, instanceCount, indexStart, baseVertex, baseInstance}
kernel void s5_prod_draw(S5_PROD_PARAMS) {
    threadgroup uint flag;
    if (s5_run(scratch, done, p, gid, tid, flag) && tid == 0) {
        out[0] = 1u; out[1] = 1000u; out[2] = 0u; out[3] = 0u; out[4] = 0u;
    }
}

// C2: the last group encodes p.k draw commands into the ICB (one per thread).
kernel void s5_prod_icb(S5_PROD_PARAMS, constant ICBContainer& c [[buffer(4)]]) {
    threadgroup uint flag;
    if (s5_run(scratch, done, p, gid, tid, flag) && tid < p.k) {
        render_command cmd(c.icb, tid);
        cmd.draw_primitives(primitive_type::point, tid, 1, 1, 0);
    }
}

// C3: MTLIndirectCommandBufferExecutionRange {location, length}
kernel void s5_prod_range(S5_PROD_PARAMS) {
    threadgroup uint flag;
    if (s5_run(scratch, done, p, gid, tid, flag) && tid == 0) {
        out[0] = 32u; out[1] = 64u;
    }
}

// C4: MTLDispatchThreadgroupsIndirectArguments {threadgroupsPerGrid[3]}
kernel void s5_prod_groups(S5_PROD_PARAMS) {
    threadgroup uint flag;
    if (s5_run(scratch, done, p, gid, tid, flag) && tid == 0) {
        out[0] = 512u; out[1] = 1u; out[2] = 1u;
    }
}

// C5: MTLDispatchThreadsIndirectArguments {threadsPerGrid[3], threadsPerThreadgroup[3]}
kernel void s5_prod_threads(S5_PROD_PARAMS) {
    threadgroup uint flag;
    if (s5_run(scratch, done, p, gid, tid, flag) && tid == 0) {
        out[0] = 16384u; out[1] = 1u; out[2] = 1u;
        out[3] = 64u;    out[4] = 1u; out[5] = 1u;
    }
}

// ---- consumers: every unit of work they run leaves one count ------------------------------------------------

struct VOut {
    float4 pos [[position]];
    float  ps [[point_size]];
};
// One count per vertex invocation, the point is clipped away.
vertex VOut s5_vs(uint vid [[vertex_id]], device atomic_uint* counter [[buffer(5)]]) {
    atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    VOut o;
    o.pos = float4(2.0, 2.0, 0.0, 1.0);
    o.ps  = 1.0;
    return o;
}
fragment half4 s5_fs() { return half4(0.0h); }

// One count per threadgroup (C4).
kernel void s5_count_groups(device atomic_uint* counter [[buffer(5)]], uint tid [[thread_index_in_threadgroup]]) {
    if (tid == 0) atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
}
// One count per thread (C5).
kernel void s5_count_threads(device atomic_uint* counter [[buffer(5)]]) {
    atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
}
