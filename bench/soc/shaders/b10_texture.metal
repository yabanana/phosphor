// B-10: texture sampling throughput per format (S-TEX-1..4 of
// docs/APPLE_SOC_PLAYBOOK.md).  One dispatch of 1024x1024 threads, each
// taking `samples` independent samples; coherent (neighbouring threads ->
// neighbouring texels, windows of 1024 texels stepping over the texture) or
// sparse (hashed coordinates), point or bilinear.  The CPU replays the exact
// same coordinates (b10_texture.cpp) and compares the per-thread sums.
#include <metal_stdlib>
using namespace metal;

constant bool kSparse   [[function_constant(0)]];
constant bool kBilinear [[function_constant(1)]];

struct SampleParams {
    uint  mask;    // size - 1 (power of two)
    uint  samples;
    float inv;     // 1 / size
    uint  seed;
};

inline uint hash32(uint x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}

kernel void b10_sample(texture2d<float, access::sample> tex [[texture(0)]], constant SampleParams& p [[buffer(0)]],
                       device float* out [[buffer(1)]], uint2 gid [[thread_position_in_grid]],
                       uint2 gsz [[threads_per_grid]]) {
    constexpr sampler sPoint(coord::normalized, filter::nearest, address::clamp_to_edge, mip_filter::none);
    constexpr sampler sLin(coord::normalized, filter::linear, address::clamp_to_edge, mip_filter::none);
    const float off = kBilinear ? 0.75f : 0.5f; // 0.75: bilinear weights exactly 0.75 / 0.25
    float4 acc = 0;
    for (uint s = 0; s < p.samples; ++s) {
        uint x, y;
        if (kSparse) {
            const uint h1 = hash32(gid.x + gid.y * 2048u + s * 0x9E3779B1u + p.seed);
            const uint h2 = hash32(h1 ^ 0x68E31DA4u);
            x = h1 & p.mask;
            y = h2 & p.mask;
        } else {
            x = (gid.x + (s & 7u) * 1024u) & p.mask;
            y = (gid.y + (s >> 3) * 1024u) & p.mask;
        }
        const float2 uv = (float2(x, y) + off) * p.inv;
        if (kBilinear) acc += tex.sample(sLin, uv);
        else acc += tex.sample(sPoint, uv);
    }
    out[gid.y * gsz.x + gid.x] = dot(acc, float4(1.0f, 2.0f, 3.0f, 5.0f));
}
