// B-13: hidden-surface removal efficiency.  Full-screen layers at distinct
// depths with an expensive fragment shader (dependent FMA chain, `iters` from
// the parameter block).  The chain result only feeds a term masked by a
// runtime zero (p.zero), so the compiler keeps the work but the colour stays
// exactly the hash colour the CPU recomputes.
#include <metal_stdlib>
using namespace metal;

struct LayerParams {
    float z;
    uint  layer;
    uint  iters;
    uint  zero;   // always 0 at run time
};

struct VOut { float4 pos [[position]]; };

vertex VOut b13_vertex(uint vid [[vertex_id]], constant LayerParams& p [[buffer(0)]]) {
    const float2 q = float2(float((vid << 1) & 2u), float(vid & 2u));
    VOut o;
    o.pos = float4(q * 2.0 - 1.0, p.z, 1.0);
    return o;
}

inline uint b13_hash(uint x, uint y, uint c) {
    uint h = x * 73856093u ^ y * 19349663u ^ (c * 83492791u + 1u);
    h ^= h >> 15;
    h *= 2246822519u;
    h ^= h >> 13;
    return h;
}

inline float4 b13_shade(float2 pos, constant LayerParams& p, thread uint& m) {
    const uint x = uint(pos.x), y = uint(pos.y);
    float a = float(x & 1023u) * 1e-3f + 0.5f, b = float(y & 1023u) * 1e-3f + float(p.layer);
    for (uint k = 0; k < p.iters; ++k) {
        a = fma(a, 0.999f, b * 0.001f);
        b = fma(b, 0.998f, a * 0.002f);
    }
    m = as_type<uint>(a + b) & p.zero; // 0
    const float mf = float(m);
    return float4(float(b13_hash(x, y, p.layer * 4 + 0) & 255u) * (1.0f / 255.0f) + mf,
                  float(b13_hash(x, y, p.layer * 4 + 1) & 255u) * (1.0f / 255.0f),
                  float(b13_hash(x, y, p.layer * 4 + 2) & 255u) * (1.0f / 255.0f), 1.0f - mf);
}

fragment float4 b13_frag(VOut in [[stage_in]], constant LayerParams& p [[buffer(0)]]) {
    uint m;
    return b13_shade(in.pos.xy, p, m);
}

// Same, but the shader may discard (never does: m == 0 is not known statically).
fragment float4 b13_frag_discard(VOut in [[stage_in]], constant LayerParams& p [[buffer(0)]]) {
    uint m;
    const float4 c = b13_shade(in.pos.xy, p, m);
    if (m != 0u) discard_fragment();
    return c;
}
