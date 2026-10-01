// F5-S2: delta updates of a persistent GPU scene.  Records are 80-byte
// GPUInstance-like structs (20 words), read as uint4 x 5 so that every word is
// consumed and nothing can be optimised away.
#include <metal_stdlib>
using namespace metal;

// Delta record = 96 B = 6 x uint4: {slot, pad, pad, pad} + 80 B instance.
// One thread per record: P[slot] = data.
kernel void s2_scatter(const device uint4* recs [[buffer(0)]], device uint4* dst [[buffer(1)]],
                       constant uint& count [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    const device uint4* r = recs + ulong(i) * 6ul;
    const uint slot = r[0].x;
    device uint4* d = dst + ulong(slot) * 5ul;
    d[0] = r[1];
    d[1] = r[2];
    d[2] = r[3];
    d[3] = r[4];
    d[4] = r[5];
}

struct ConsumeParams {
    float4 plane; // xyz normal, w offset
    uint   count;
    uint   pad0, pad1, pad2;
};

// Reads all 20 words of instance i: FNV-1a over the words (bit 0 replaced by
// a bounding-sphere-vs-plane visibility bit of the translation column 12..14,
// radius from word 0).  One u32 out per instance.
kernel void s2_consume(const device uint4* src [[buffer(0)]], device uint* out [[buffer(1)]],
                       constant ConsumeParams& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.count) return;
    const device uint4* r = src + ulong(i) * 5ul;
    const uint4 a = r[0], b = r[1], c = r[2], d = r[3], e = r[4];
    uint h = 2166136261u;
    h = (h ^ a.x) * 16777619u; h = (h ^ a.y) * 16777619u; h = (h ^ a.z) * 16777619u; h = (h ^ a.w) * 16777619u;
    h = (h ^ b.x) * 16777619u; h = (h ^ b.y) * 16777619u; h = (h ^ b.z) * 16777619u; h = (h ^ b.w) * 16777619u;
    h = (h ^ c.x) * 16777619u; h = (h ^ c.y) * 16777619u; h = (h ^ c.z) * 16777619u; h = (h ^ c.w) * 16777619u;
    h = (h ^ d.x) * 16777619u; h = (h ^ d.y) * 16777619u; h = (h ^ d.z) * 16777619u; h = (h ^ d.w) * 16777619u;
    h = (h ^ e.x) * 16777619u; h = (h ^ e.y) * 16777619u; h = (h ^ e.z) * 16777619u; h = (h ^ e.w) * 16777619u;
    const float3 t      = as_type<float4>(d).xyz;
    const float radius  = fabs(as_type<float>(a.x)) * 0.01f + 0.5f;
    const float dist    = t.x * p.plane.x + t.y * p.plane.y + t.z * p.plane.z + p.plane.w;
    out[i] = (h & 0xFFFFFFFEu) | (dist > -radius ? 1u : 0u);
}

// ---- Render part of the pipelined frame -----------------------------------
struct RenderParams {
    float halfSize; // quad half extent in NDC
    uint  stride;   // record stride between drawn instances
    uint  iters;    // fragment ALU iterations
    uint  pad;
};

struct VOut {
    float4 pos [[position]];
    float  seed [[flat]];
};

vertex VOut s2_vs(uint vid [[vertex_id]], uint iid [[instance_id]], const device uint4* src [[buffer(0)]],
                  constant RenderParams& rp [[buffer(1)]]) {
    const device uint4* r = src + ulong(iid) * ulong(rp.stride) * 5ul;
    const float3 t = as_type<float4>(r[3]).xyz;
    const float2 c = float2(float(vid & 1u), float(vid >> 1)) * 2.0f - 1.0f;
    VOut o;
    o.pos  = float4(t.xy * 0.01f + c * rp.halfSize, 0.5f, 1.0f);
    o.seed = float(iid) * 0.37f + t.z * 0.01f;
    return o;
}

// Seeded per instance and pixel so the compiler cannot merge the chain.
fragment half4 s2_fs(VOut in [[stage_in]], constant RenderParams& rp [[buffer(1)]]) {
    float x = fract(in.pos.x * 0.013f + in.seed);
    float y = fract(in.pos.y * 0.017f + in.seed * 0.5f);
    float acc = in.seed;
    for (uint k = 0; k < rp.iters; ++k) {
        x = fract(fma(x, 1.3f, y));
        y = fract(fma(y, 1.7f, x));
        acc += x * y;
    }
    return half4(half(fract(acc)), half(x), half(y), 1.0h);
}
