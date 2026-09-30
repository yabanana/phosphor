// B-11: cost of writing / reading RGBA8 textures with GPU-optimized contents
// (lossless compression) on or off, for compressible (constant, smooth) and
// incompressible (random) content, full and partial block writes
// (S-TEX-2 of docs/APPLE_SOC_PLAYBOOK.md).  The content function is integer
// only: the CPU replays it exactly (b11_compression.cpp).
#include <metal_stdlib>
using namespace metal;

struct WriteParams {
    uint width, height;
    uint pass;    // content varies with the pass: repeated writes are never identical
    uint kind;    // 0 constant, 1 smooth, 2 random
    uint partial; // 1: only 2x2-texel cells with (cx + cy) even are written (50% of every 4x4 block)
    uint pad0, pad1, pad2;
};

inline uint hash3(uint x, uint y, uint p) {
    uint h = x * 0x9E3779B1u ^ (y + 0x7F4A7C15u) * 0x85EBCA6Bu ^ (p + 1u) * 0xC2B2AE35u;
    h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16;
    return h;
}

inline float4 content(uint kind, uint x, uint y, uint p) {
    uint r, g, b, a = 255u;
    if (kind == 0u) {
        r = (p * 37u + 11u) & 255u; g = (p * 91u + 5u) & 255u; b = (p * 53u + 201u) & 255u;
    } else if (kind == 1u) {
        r = ((x >> 4) + p) & 255u; g = ((y >> 4) + p * 3u) & 255u; b = (((x + y) >> 5) + p * 5u) & 255u;
    } else {
        const uint h = hash3(x, y, p);
        r = h & 255u; g = (h >> 8) & 255u; b = (h >> 16) & 255u; a = (h >> 24) & 255u;
    }
    return float4(float(r), float(g), float(b), float(a)) * (1.0f / 255.0f);
}

inline bool masked(constant WriteParams& p, uint x, uint y) {
    return p.partial != 0u && (((x >> 1) + (y >> 1)) & 1u) != 0u;
}

kernel void b11_cs_write(texture2d<float, access::write> tex [[texture(0)]], constant WriteParams& p [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.width || gid.y >= p.height || masked(p, gid.x, gid.y)) return;
    tex.write(content(p.kind, gid.x, gid.y, p.pass), gid);
}

struct VOut { float4 position [[position]]; };

vertex VOut b11_vs(uint vid [[vertex_id]]) {
    const float2 pos = float2((vid << 1) & 2, vid & 2); // fullscreen triangle
    VOut o;
    o.position = float4(pos * 2.0f - 1.0f, 0.0f, 1.0f);
    return o;
}

fragment float4 b11_fs(VOut in [[stage_in]], constant WriteParams& p [[buffer(0)]]) {
    const uint2 xy = uint2(in.position.xy);
    return content(p.kind, xy.x, xy.y, p.pass);
}

// One thread per 4x4 texel block: 16 reads, one float stored (1/16 of the read traffic).
kernel void b11_read(texture2d<float, access::read> tex [[texture(0)]], device float* out [[buffer(1)]],
                     uint2 gid [[thread_position_in_grid]], uint2 gsz [[threads_per_grid]]) {
    float4 acc = 0;
    for (uint j = 0; j < 4; ++j)
        for (uint i = 0; i < 4; ++i) acc += tex.read(uint2(gid.x * 4 + i, gid.y * 4 + j));
    out[gid.y * gsz.x + gid.x] = dot(acc, float4(1.0f, 2.0f, 3.0f, 5.0f));
}

// RGBA16Float writer with random content (the "other format" texture aliasing the measured one in the heap).
kernel void b11_cs_write16(texture2d<half, access::write> tex [[texture(0)]], constant WriteParams& p [[buffer(0)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.width || gid.y >= p.height) return;
    tex.write(half4(content(p.kind, gid.x, gid.y, p.pass)), gid);
}
