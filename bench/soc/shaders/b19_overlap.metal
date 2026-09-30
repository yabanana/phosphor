// B-19: overlap between passes and queues.
//  * pass A: full-screen triangle, fragment-heavy (iters of a dependent
//    integer chain) or bandwidth-heavy (iters = 0, RGBA32F target);
//  * pass B: 65536 grid triangles (8 px cells over 2048x2048), vertex-heavy
//    (iters of a dependent integer chain per vertex), trivial fragment;
//  * compute: integer chain per thread.
// The chains only feed a term masked by the runtime zero p.zero, so the
// results are exactly the hash colours / values the CPU recomputes.
#include <metal_stdlib>
using namespace metal;

struct Params {
    uint iters;
    uint zero;   // 0 at run time
    uint count;
    uint pad;
};

inline uint b19_hash(uint x, uint y, uint c) {
    uint h = x * 73856093u ^ y * 19349663u ^ (c * 83492791u + 1u);
    h ^= h >> 15;
    h *= 2246822519u;
    h ^= h >> 13;
    return h;
}

inline uint b19_chain(uint seed, uint iters) {
    uint a = seed | 1u;
    for (uint k = 0; k < iters; ++k) a = a * 1664525u + 1013904223u;
    return a;
}

struct VA { float4 pos [[position]]; };

vertex VA b19_vs_full(uint vid [[vertex_id]]) {
    const float2 q = float2(float((vid << 1) & 2u), float(vid & 2u));
    VA o;
    o.pos = float4(q * 2.0 - 1.0, 0.5, 1.0);
    return o;
}

// Pass A, RGBA8: colour bytes = hash(x, y, c) & 255.
fragment float4 b19_fs_heavy(VA in [[stage_in]], constant Params& p [[buffer(0)]]) {
    const uint x = uint(in.pos.x), y = uint(in.pos.y);
    const uint m = b19_chain(x * 4099u + y, p.iters) & p.zero;
    return float4(float(b19_hash(x, y, 0) & 255u), float(b19_hash(x, y, 1) & 255u),
                  float(b19_hash(x, y, 2) & 255u), float(b19_hash(x, y, 3) & 255u)) * (1.0f / 255.0f) + float(m);
}

// Bandwidth variant, RGBA32F: k / 1024.
fragment float4 b19_fs_float(VA in [[stage_in]], constant Params& p [[buffer(0)]]) {
    const uint x = uint(in.pos.x), y = uint(in.pos.y);
    const uint m = p.zero;
    return float4(float(b19_hash(x, y, 0) & 1023u), float(b19_hash(x, y, 1) & 1023u),
                  float(b19_hash(x, y, 2) & 1023u), float(b19_hash(x, y, 3) & 1023u)) * (1.0f / 1024.0f) + float(m);
}

struct VB { float4 pos [[position]]; uint tri [[flat]]; };

// Pass B vertex: triangle t at grid cell (t & 255, t >> 8), 3.5 px legs; y down as in B-12.
vertex VB b19_vs_heavy(uint vid [[vertex_id]], constant Params& p [[buffer(0)]]) {
    const uint tri = vid / 3, corner = vid % 3;
    const uint m = b19_chain(vid, p.iters) & p.zero;
    const float2 o = float2(float(4 + (tri & 255u) * 8), float(4 + (tri >> 8) * 8));
    const float2 c = o + float2(corner == 1 ? 3.5f : 0.0f, corner == 2 ? 3.5f : 0.0f);
    VB r;
    r.pos = float4(c.x * (1.0f / 1024.0f) - 1.0f + float(m), 1.0f - c.y * (1.0f / 1024.0f), 0.5f, 1.0f);
    r.tri = tri;
    return r;
}

fragment float4 b19_fs_tri(VB in [[stage_in]]) {
    const uint h = b19_hash(in.tri, 7u, 9u);
    return float4(float(h & 255u), float((h >> 8) & 255u), float((h >> 16) & 255u), 255.0f) * (1.0f / 255.0f);
}

kernel void b19_compute(device uint* out [[buffer(0)]], constant Params& p [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
    out[i] = b19_chain(i, p.iters) + (p.zero & i);
}
