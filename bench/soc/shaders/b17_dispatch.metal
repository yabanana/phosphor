// B-17: empty / indirect dispatches and ICB draws.  Every kernel leaves a
// countable side effect so the CPU can verify that all N commands ran.
#include <metal_stdlib>
using namespace metal;

// One SIMD-group per threadgroup; lane 0 counts.
kernel void b17_count(device atomic_uint* counter [[buffer(0)]], uint lane [[thread_index_in_threadgroup]]) {
    if (lane == 0) atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
}

// Chain of indirect dispatches: dispatch k (k = value of the counter it sees)
// writes the arguments {1,1,1,0} of dispatch k+1.  Needs a barrier between
// consecutive dispatches: a stale (zero) argument block dispatches nothing
// and the chain stops, so the final count is < N.
struct DispatchArgs {
    uint x, y, z, pad;
};
kernel void b17_chain(device atomic_uint* counter [[buffer(0)]], device DispatchArgs* args [[buffer(1)]],
                      constant uint& total [[buffer(2)]], uint lane [[thread_index_in_threadgroup]]) {
    if (lane != 0) return;
    const uint k = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    if (k + 1u < total) args[k + 1u] = DispatchArgs{1u, 1u, 1u, 0u};
}

// ---- ICB draws: one point per draw, clipped away (pure command cost) ----
struct VOut {
    float4 pos [[position]];
    float  ps [[point_size]];
};
vertex VOut b17_vs(uint vid [[vertex_id]], device atomic_uint* counter [[buffer(0)]], device uint* flags [[buffer(1)]]) {
    atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    flags[vid] = flags[vid] + 1u; // each draw covers a distinct vertex id
    VOut o;
    o.pos = float4(2.0, 2.0, 0.0, 1.0);
    o.ps  = 1.0;
    return o;
}
fragment half4 b17_fs() { return half4(0.0h); }

// GPU-encoded ICB: one thread per command.
struct ICBContainer {
    command_buffer icb [[id(0)]];
};
kernel void b17_encode(constant ICBContainer& c [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    render_command cmd(c.icb, i);
    cmd.draw_primitives(primitive_type::point, i, 1, 1, 0);
}
