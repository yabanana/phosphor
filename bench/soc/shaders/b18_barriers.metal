// B-18: tiny passes that read-modify-write a 1024-word buffer (buf[i] += 1) in
// every stage a barrier can connect, so a missing barrier loses an update.
// p.x = optional spin iterations (slow producer for the race probe), p.y = 0
// (a runtime zero so the spin loop is kept).
#include <metal_stdlib>
using namespace metal;

inline uint spin(uint2 p, uint i) {
    uint a = i | 1u;
    for (uint k = 0; k < p.x; ++k) a = a * 1664525u + 1013904223u;
    return a & p.y;
}

kernel void b18_inc(device uint* buf [[buffer(0)]], constant uint2& p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    buf[i] = buf[i] + 1u + spin(p, i);
}

struct VOut {
    float4 pos [[position]];
    float  ps [[point_size]];
};
vertex VOut b18_v_inc(uint vid [[vertex_id]], device uint* buf [[buffer(0)]], constant uint2& p [[buffer(1)]]) {
    buf[vid] = buf[vid] + 1u + spin(p, vid);
    VOut o;
    o.pos = float4(2.0, 2.0, 0.0, 1.0); // clipped away
    o.ps  = 1.0;
    return o;
}

struct FOut {
    float4 pos [[position]];
};
vertex FOut b18_v_full(uint vid [[vertex_id]]) {
    float2 q = float2(float((vid << 1) & 2u), float(vid & 2u));
    FOut o;
    o.pos = float4(q * 2.0 - 1.0, 0.5, 1.0);
    return o;
}
// 32x32 target: pixel (x, y) -> word y * 32 + x.
fragment half4 b18_f_inc(float4 pos [[position]], device uint* buf [[buffer(0)]], constant uint2& p [[buffer(1)]]) {
    const uint i = uint(pos.y) * 32u + uint(pos.x);
    buf[i] = buf[i] + 1u + spin(p, i);
    return half4(0.0h);
}
fragment half4 b18_f_dummy() { return half4(0.0h); }
