// B-14: load/store cost.  Full-screen triangle whose fragment shader writes a
// pixel-distinct, incompressible value (hash of the pixel coordinates), so the
// store really moves bytes to DRAM (constant colours compress to nothing) and
// the CPU can recompute every value.  Colour formats write float4 (values are
// exact in RGBA8 unorm, RGBA16F and RGBA32F); depth writes a float in [0, 1).
//
// "_full" variants cover every pixel (the driver may skip loading tiles that a
// full-coverage opaque draw overwrites: measured, see the benchmark); "_disc"
// variants discard one pixel in four and write a different hash (channel
// offset 16) on the rest, so the untouched pixels must come from the load.
#include <metal_stdlib>
using namespace metal;

struct VOut { float4 pos [[position]]; };

vertex VOut b14_vertex(uint vid [[vertex_id]]) {
    const float2 q = float2(float((vid << 1) & 2u), float(vid & 2u));
    VOut o;
    o.pos = float4(q * 2.0 - 1.0, 0.5, 1.0);
    return o;
}

inline uint b14_hash(uint x, uint y, uint c) {
    uint h = x * 73856093u ^ y * 19349663u ^ (c * 83492791u + 1u);
    h ^= h >> 15;
    h *= 2246822519u;
    h ^= h >> 13;
    return h;
}

// Discarded pixels: (x & 1) == 0 && (y & 1) == 0.
inline bool b14_skip(uint x, uint y) { return ((x | y) & 1u) == 0u; }

// RGBA8 gets k/255 (exact byte), the float formats k/1024 (exact in FP16).
inline float4 b14_rgba8(uint x, uint y, uint off) {
    return float4(float(b14_hash(x, y, off + 0) & 255u), float(b14_hash(x, y, off + 1) & 255u),
                  float(b14_hash(x, y, off + 2) & 255u), float(b14_hash(x, y, off + 3) & 255u)) * (1.0f / 255.0f);
}
inline float4 b14_float(uint x, uint y, uint off) {
    return float4(float(b14_hash(x, y, off + 0) & 1023u), float(b14_hash(x, y, off + 1) & 1023u),
                  float(b14_hash(x, y, off + 2) & 1023u), float(b14_hash(x, y, off + 3) & 1023u)) * (1.0f / 1024.0f);
}

fragment float4 b14_frag_rgba8_full(VOut in [[stage_in]]) {
    return b14_rgba8(uint(in.pos.x), uint(in.pos.y), 0);
}
fragment float4 b14_frag_float_full(VOut in [[stage_in]]) {
    return b14_float(uint(in.pos.x), uint(in.pos.y), 0);
}
fragment float4 b14_frag_rgba8_disc(VOut in [[stage_in]]) {
    const uint x = uint(in.pos.x), y = uint(in.pos.y);
    if (b14_skip(x, y)) discard_fragment();
    return b14_rgba8(x, y, 16);
}
fragment float4 b14_frag_float_disc(VOut in [[stage_in]]) {
    const uint x = uint(in.pos.x), y = uint(in.pos.y);
    if (b14_skip(x, y)) discard_fragment();
    return b14_float(x, y, 16);
}

struct DOut { float d [[depth(any)]]; };
fragment DOut b14_frag_depth_full(VOut in [[stage_in]]) {
    DOut o;
    o.d = float(b14_hash(uint(in.pos.x), uint(in.pos.y), 0) & 0xFFFFFFu) * (1.0f / 16777216.0f);
    return o;
}
fragment DOut b14_frag_depth_disc(VOut in [[stage_in]]) {
    const uint x = uint(in.pos.x), y = uint(in.pos.y);
    if (b14_skip(x, y)) discard_fragment();
    DOut o;
    o.d = float(b14_hash(x, y, 16) & 0xFFFFFFu) * (1.0f / 16777216.0f);
    return o;
}
