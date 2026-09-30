// B-12: parameter-buffer / partial-render threshold.  N small triangles (right
// triangles with 3.5 px legs = exactly 6 covered pixels, integer origin) in one
// draw, positions from a hash of the triangle id (mode 0: spread over the
// whole 2048x2048 target, no culling) or on a regular 8 px grid (mode 1, no
// overlap: used to verify every triangle on the CPU).  0/4/8/16 float4
// varyings, all consumed by the fragment shader (masked by a runtime zero).
#include <metal_stdlib>
using namespace metal;

struct Params {
    uint count;
    uint mode;
    uint zero;   // 0 at run time
    uint pad;
};

inline uint b12_hash(uint x) {
    x ^= x >> 16;
    x *= 2246822519u;
    x ^= x >> 13;
    x *= 3266489917u;
    x ^= x >> 16;
    return x;
}

inline float2 b12_origin(uint tri, uint mode) {
    if (mode == 1) return float2(float(4 + (tri & 255u) * 8), float(4 + (tri >> 8) * 8));
    const uint h = b12_hash(tri);
    return float2(float(4 + (h & 2047u) % 2040u), float(4 + ((h >> 11) & 2047u) % 2040u));
}

inline float4 b12_pos(uint vid, uint mode) {
    const uint tri = vid / 3, corner = vid % 3;
    const float2 o = b12_origin(tri, mode);
    const float2 c = o + float2(corner == 1 ? 3.5f : 0.0f, corner == 2 ? 3.5f : 0.0f);
    return float4(c.x * (1.0f / 1024.0f) - 1.0f, 1.0f - c.y * (1.0f / 1024.0f), 0.5f, 1.0f);
}

inline float4 b12_var(uint tri, uint k) {
    return float4(float(b12_hash(tri * 16u + k) & 1023u), float(k), float(tri & 1023u), 1.0f);
}

struct VOut0 { float4 pos [[position]]; uint tri [[flat]]; };
struct VOut4 { float4 pos [[position]]; uint tri [[flat]]; float4 v0; float4 v1; float4 v2; float4 v3; };
struct VOut8 { float4 pos [[position]]; uint tri [[flat]]; float4 v0; float4 v1; float4 v2; float4 v3; float4 v4; float4 v5; float4 v6; float4 v7; };
struct VOut16 { float4 pos [[position]]; uint tri [[flat]]; float4 v0; float4 v1; float4 v2; float4 v3; float4 v4; float4 v5; float4 v6; float4 v7; float4 v8; float4 v9; float4 v10; float4 v11; float4 v12; float4 v13; float4 v14; float4 v15; };

vertex VOut0 b12_vs0(uint vid [[vertex_id]], constant Params& p [[buffer(0)]]) {
    VOut0 o;
    o.pos = b12_pos(vid, p.mode);
    o.tri = vid / 3;
    return o;
}
vertex VOut4 b12_vs4(uint vid [[vertex_id]], constant Params& p [[buffer(0)]]) {
    VOut4 o;
    o.pos = b12_pos(vid, p.mode);
    o.tri = vid / 3;
    o.v0 = b12_var(o.tri, 0u);
    o.v1 = b12_var(o.tri, 1u);
    o.v2 = b12_var(o.tri, 2u);
    o.v3 = b12_var(o.tri, 3u);
    return o;
}
vertex VOut8 b12_vs8(uint vid [[vertex_id]], constant Params& p [[buffer(0)]]) {
    VOut8 o;
    o.pos = b12_pos(vid, p.mode);
    o.tri = vid / 3;
    o.v0 = b12_var(o.tri, 0u);
    o.v1 = b12_var(o.tri, 1u);
    o.v2 = b12_var(o.tri, 2u);
    o.v3 = b12_var(o.tri, 3u);
    o.v4 = b12_var(o.tri, 4u);
    o.v5 = b12_var(o.tri, 5u);
    o.v6 = b12_var(o.tri, 6u);
    o.v7 = b12_var(o.tri, 7u);
    return o;
}
vertex VOut16 b12_vs16(uint vid [[vertex_id]], constant Params& p [[buffer(0)]]) {
    VOut16 o;
    o.pos = b12_pos(vid, p.mode);
    o.tri = vid / 3;
    o.v0 = b12_var(o.tri, 0u);
    o.v1 = b12_var(o.tri, 1u);
    o.v2 = b12_var(o.tri, 2u);
    o.v3 = b12_var(o.tri, 3u);
    o.v4 = b12_var(o.tri, 4u);
    o.v5 = b12_var(o.tri, 5u);
    o.v6 = b12_var(o.tri, 6u);
    o.v7 = b12_var(o.tri, 7u);
    o.v8 = b12_var(o.tri, 8u);
    o.v9 = b12_var(o.tri, 9u);
    o.v10 = b12_var(o.tri, 10u);
    o.v11 = b12_var(o.tri, 11u);
    o.v12 = b12_var(o.tri, 12u);
    o.v13 = b12_var(o.tri, 13u);
    o.v14 = b12_var(o.tri, 14u);
    o.v15 = b12_var(o.tri, 15u);
    return o;
}

// Colour = hash bytes of the triangle id; the varyings only add a term that is 0 at run time.
inline float4 b12_color(uint tri, float acc, constant Params& p) {
    const uint m = as_type<uint>(acc) & p.zero;
    const uint h = b12_hash(tri + 12345u);
    return float4(float(h & 255u), float((h >> 8) & 255u), float((h >> 16) & 255u), 255.0f) * (1.0f / 255.0f) + float(m);
}

fragment float4 b12_fs0(VOut0 in [[stage_in]], constant Params& p [[buffer(0)]]) {
    float a = 0;
    return b12_color(in.tri, a, p);
}
fragment float4 b12_fs4(VOut4 in [[stage_in]], constant Params& p [[buffer(0)]]) {
    float a = 0;
    a += in.v0.x + in.v0.y * in.v0.z;
    a += in.v1.x + in.v1.y * in.v1.z;
    a += in.v2.x + in.v2.y * in.v2.z;
    a += in.v3.x + in.v3.y * in.v3.z;
    return b12_color(in.tri, a, p);
}
fragment float4 b12_fs8(VOut8 in [[stage_in]], constant Params& p [[buffer(0)]]) {
    float a = 0;
    a += in.v0.x + in.v0.y * in.v0.z;
    a += in.v1.x + in.v1.y * in.v1.z;
    a += in.v2.x + in.v2.y * in.v2.z;
    a += in.v3.x + in.v3.y * in.v3.z;
    a += in.v4.x + in.v4.y * in.v4.z;
    a += in.v5.x + in.v5.y * in.v5.z;
    a += in.v6.x + in.v6.y * in.v6.z;
    a += in.v7.x + in.v7.y * in.v7.z;
    return b12_color(in.tri, a, p);
}
fragment float4 b12_fs16(VOut16 in [[stage_in]], constant Params& p [[buffer(0)]]) {
    float a = 0;
    a += in.v0.x + in.v0.y * in.v0.z;
    a += in.v1.x + in.v1.y * in.v1.z;
    a += in.v2.x + in.v2.y * in.v2.z;
    a += in.v3.x + in.v3.y * in.v3.z;
    a += in.v4.x + in.v4.y * in.v4.z;
    a += in.v5.x + in.v5.y * in.v5.z;
    a += in.v6.x + in.v6.y * in.v6.z;
    a += in.v7.x + in.v7.y * in.v7.z;
    a += in.v8.x + in.v8.y * in.v8.z;
    a += in.v9.x + in.v9.y * in.v9.z;
    a += in.v10.x + in.v10.y * in.v10.z;
    a += in.v11.x + in.v11.y * in.v11.z;
    a += in.v12.x + in.v12.y * in.v12.z;
    a += in.v13.x + in.v13.y * in.v13.z;
    a += in.v14.x + in.v14.y * in.v14.z;
    a += in.v15.x + in.v15.y * in.v15.z;
    return b12_color(in.tri, a, p);
}

// Heavy-flush variant: the same primitives with 16 varyings, four RGBA32F
// attachments.  A partial render stores and reloads every attachment
// (4 x 2048^2 x 16 B = 256 MiB), so each one becomes visible in the time
// per triangle (with one RGBA8 target a flush is too cheap to see).
struct B12Mrt4 {
    float4 c0 [[color(0)]];
    float4 c1 [[color(1)]];
    float4 c2 [[color(2)]];
    float4 c3 [[color(3)]];
};
fragment B12Mrt4 b12_fs16_mrt(VOut16 in [[stage_in]], constant Params& p [[buffer(0)]]) {
    float a = 0;
    a += in.v0.x + in.v0.y * in.v0.z;
    a += in.v1.x + in.v1.y * in.v1.z;
    a += in.v2.x + in.v2.y * in.v2.z;
    a += in.v3.x + in.v3.y * in.v3.z;
    a += in.v4.x + in.v4.y * in.v4.z;
    a += in.v5.x + in.v5.y * in.v5.z;
    a += in.v6.x + in.v6.y * in.v6.z;
    a += in.v7.x + in.v7.y * in.v7.z;
    a += in.v8.x + in.v8.y * in.v8.z;
    a += in.v9.x + in.v9.y * in.v9.z;
    a += in.v10.x + in.v10.y * in.v10.z;
    a += in.v11.x + in.v11.y * in.v11.z;
    a += in.v12.x + in.v12.y * in.v12.z;
    a += in.v13.x + in.v13.y * in.v13.z;
    a += in.v14.x + in.v14.y * in.v14.z;
    a += in.v15.x + in.v15.y * in.v15.z;
    const float4 c = b12_color(in.tri, a, p);
    B12Mrt4 o;
    o.c0 = c;
    o.c1 = c.yzwx;
    o.c2 = c.zwxy;
    o.c3 = c.wxyz;
    return o;
}
