// hiz.metal -- F6.4 conservative Hi-Z pyramid (reverse-Z: every texel holds the
// MINIMUM, i.e. farthest, depth of the pixels it covers).  Contract:
// renderer/meshlet_layout.h (slots), renderer/meshlet_cull_math.h (sizes).
//
// Level 0 has power-of-two dimensions >= ceil(viewport / 2): texel (x, y)
// covers the depth pixels [2x, 2x + 2) x [2y, 2y + 2); pixels outside the
// depth texture count as 1 (the neutral value of the min).  Every further
// level halves exactly (Metal's max(1, s >> L) on power-of-two sizes; when a
// dimension is already 1 its only texel is read once).  The CPU reference is
// renderer/meshlet_cull_reference.cpp (buildHiZReference), bit for bit (min
// is exact).
//
// Backends (spike S3 chooses; Apple9 always has the compute ones):
//   hiz_level0          depth -> level 0, one thread per texel (loads)
//   hiz_reduce          level L-1 -> L, one thread per texel (4 loads)
//   hiz_reduce_simd     levels L .. L+4 from level L-1 in ONE dispatch: a
//                       16x16 group loads 32x32 source texels, SIMD shuffles
//                       give L+1 (the 2x2 quad of a 16-wide SIMD-group row
//                       pair), threadgroup memory L+2 .. L+4 (O5)
//   hiz_reduce_sampler  level L-1 -> L with ONE sample of a min-reduction
//                       sampler at the centre of the 2x2 footprint (Apple10)
// Mip levels are written with texture2d::write(value, coord, lod) and read
// with read(coord, lod) of the same texture bound twice (read / write access).

#include <metal_stdlib>

#include "renderer/gpu_types.h"
#include "renderer/meshlet_layout.h"

using namespace metal;
using namespace phosphor;

static_assert(HZ_PARAMS == 0 && HZ_TEX_SRC == 0 && HZ_TEX_DST == 1, "hi-z slots");
static_assert(HIZ_GROUP == 16, "hiz_reduce_simd is written for 16x16 groups");

kernel void hiz_level0(constant GPUHiZParams& p [[buffer(0)]], depth2d<float, access::read> depth [[texture(0)]],
                       texture2d<float, access::write> dst [[texture(1)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.dstSize[0] || gid.y >= p.dstSize[1]) return;
    float m = 1.0f;
    for (uint dy = 0; dy < 2u; ++dy) {
        for (uint dx = 0; dx < 2u; ++dx) {
            const uint2 px = gid * 2u + uint2(dx, dy);
            if (px.x < p.srcSize[0] && px.y < p.srcSize[1]) m = min(m, depth.read(px));
        }
    }
    // Self-check negative control (--debug-meshlets-corrupt depth): texel
    // (0, 0) claims the nearest depth, an UNSAFE pyramid the check must report.
    if (p.pad[0] != 0u && gid.x == 0u && gid.y == 0u) m = 1.0f;
    dst.write(float4(m), gid, 0);
}

// Minimum of the (existing) 2x2 source texels of destination texel `gid`.
static float reduce2x2(texture2d<float, access::read> src, uint2 gid, uint srcLevel, uint2 srcSize) {
    float m = 1.0f;
    for (uint dy = 0; dy < 2u; ++dy) {
        for (uint dx = 0; dx < 2u; ++dx) {
            const uint2 s = gid * 2u + uint2(dx, dy);
            if (s.x < srcSize.x && s.y < srcSize.y) m = min(m, src.read(s, srcLevel).x);
        }
    }
    return m;
}

kernel void hiz_reduce(constant GPUHiZParams& p [[buffer(0)]], texture2d<float, access::read> src [[texture(0)]],
                       texture2d<float, access::write> dst [[texture(1)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.dstSize[0] || gid.y >= p.dstSize[1]) return;
    dst.write(float4(reduce2x2(src, gid, p.dstLevel - 1u, uint2(p.srcSize[0], p.srcSize[1]))), gid, p.dstLevel);
}

// Levels dstLevel .. dstLevel + 4 (those < p.levels) of the 16x16 destination
// tile of this group (level dstLevel), from level dstLevel - 1.  Threads of
// texels outside a level keep the neutral value 1 and write nothing.
kernel void hiz_reduce_simd(constant GPUHiZParams& p [[buffer(0)]], texture2d<float, access::read> src [[texture(0)]],
                            texture2d<float, access::write> dst [[texture(1)]], uint2 gid [[thread_position_in_grid]],
                            uint2 lid [[thread_position_in_threadgroup]], uint2 tg [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float tile[8][8];
    const uint2 size0 = uint2(p.dstSize[0], p.dstSize[1]);
    const bool in0    = gid.x < size0.x && gid.y < size0.y;
    float m           = in0 ? reduce2x2(src, gid, p.dstLevel - 1u, uint2(p.srcSize[0], p.srcSize[1])) : 1.0f;
    if (in0) dst.write(float4(m), gid, p.dstLevel);
    if (p.dstLevel + 1u >= p.levels) return;
    // Level +1: a 16-wide group puts two rows in one SIMD-group, so the 2x2
    // quad of (x, y) is lanes l, l^1 (x), l^16 (y), l^17.
    m = min(m, simd_shuffle_xor(m, 1u));
    m = min(m, simd_shuffle_xor(m, 16u));
    const uint2 size1 = uint2(max(size0.x >> 1, 1u), max(size0.y >> 1, 1u));
    const uint2 g1    = gid >> 1;
    const bool lead1  = (lid.x & 1u) == 0u && (lid.y & 1u) == 0u;
    if (lead1 && g1.x < size1.x && g1.y < size1.y) dst.write(float4(m), g1, p.dstLevel + 1u);
    if (lead1) tile[lid.y >> 1][lid.x >> 1] = (g1.x < size1.x && g1.y < size1.y) ? m : 1.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Levels +2 .. +4 from the 8x8 tile of level +1, in threadgroup memory.
    uint n = 8u;
    for (uint k = 2u; k <= 4u; ++k) {
        if (p.dstLevel + k >= p.levels) break;
        const uint halfN = n >> 1;
        float v         = 1.0f;
        const bool act  = lid.x < halfN && lid.y < halfN;
        if (act) {
            v = min(min(tile[lid.y * 2u][lid.x * 2u], tile[lid.y * 2u][lid.x * 2u + 1u]),
                    min(tile[lid.y * 2u + 1u][lid.x * 2u], tile[lid.y * 2u + 1u][lid.x * 2u + 1u]));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint2 sizeK = uint2(max(size0.x >> k, 1u), max(size0.y >> k, 1u));
        const uint2 gk    = tg * (HIZ_GROUP >> k) + lid;
        if (act) {
            const bool inK = gk.x < sizeK.x && gk.y < sizeK.y;
            if (inK) dst.write(float4(v), gk, p.dstLevel + k);
            tile[lid.y][lid.x] = inK ? v : 1.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        n = halfN;
    }
}

// Apple10: ONE sample of a min-reduction bilinear sampler at the centre of the
// 2x2 footprint returns the minimum of the 4 texels (exact).  A level whose
// source dimension is 1 has a 1-texel footprint on that axis: the sample at
// the texel centre with clamp-to-edge reads it (weights 1 and 0 still enter
// the min reduction as the same texel).
constexpr sampler kHiZMin(filter::linear, mip_filter::nearest, address::clamp_to_edge, reduction::minimum);

kernel void hiz_reduce_sampler(constant GPUHiZParams& p [[buffer(0)]], texture2d<float> src [[texture(0)]],
                               texture2d<float, access::write> dst [[texture(1)]],
                               uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.dstSize[0] || gid.y >= p.dstSize[1]) return;
    // Normalised coordinates of the 2x2 footprint centre in the source level
    // (the texel centre when that dimension is 1).
    const float2 size  = float2(p.srcSize[0], p.srcSize[1]);
    const float2 coord = float2(p.srcSize[0] == 1u ? 0.5f : float(gid.x) * 2.0f + 1.0f,
                                p.srcSize[1] == 1u ? 0.5f : float(gid.y) * 2.0f + 1.0f);
    dst.write(float4(src.sample(kHiZMin, coord / size, level(float(p.dstLevel - 1u))).x), gid, p.dstLevel);
}
