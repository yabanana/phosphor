// B-02: dual issue.  One kernel template with independent chains of several
// types in the same thread: NF FP32 FMA chains, NH FP16 FMA chains, NI INT32
// multiply-add chains, NQ packed half2 FMA chains (2 lanes each).  The host
// times each type alone and mixed at several threadgroup sizes / occupancy
// limits (dynamic threadgroup memory) and compares time(mix) with the sum of
// the parts.  Inputs differ per thread and per chain (identical or uniform
// chains are merged / run once per SIMD-group by the compiler: B-01).
#include <metal_stdlib>
using namespace metal;

struct MixParams {
    uint  iters;
    uint  pad;
    float xf, yf;
    int   xi, yi;
    uint  pad2, pad3;
};

inline int sd(uint i, uint c) { return int((i + c * 7u) & 15u); }

template <int NF, int NH, int NI, int NQ>
kernel void mix(device uint* out [[buffer(0)]], constant MixParams& p [[buffer(1)]],
                threadgroup uint* scratch [[threadgroup(0)]], uint i [[thread_position_in_grid]],
                uint lid [[thread_index_in_threadgroup]]) {
    const float xf = p.xf, yf = p.yf;
    const half  xh = half(p.xf), yh = half(p.yf);
    const int   xi = p.xi, yi = p.yi;
    float  f[NF > 0 ? NF : 1];
    half   h[NH > 0 ? NH : 1];
    int    n[NI > 0 ? NI : 1];
    half2  q[NQ > 0 ? NQ : 1];
    for (int c = 0; c < NF; ++c) f[c] = float(sd(i, uint(c))) * 0.0625f + 1.0f;
    for (int c = 0; c < NH; ++c) h[c] = half(sd(i + 1u, uint(c))) * half(0.0625h) + half(1.0h);
    for (int c = 0; c < NI; ++c) n[c] = int(i * 2654435761u + uint(c) * 40503u + 1u);
    for (int c = 0; c < NQ; ++c)
        q[c] = half2(half(sd(i + 2u, uint(c))), half(sd(i + 5u, uint(c)))) * half(0.0625h) + half(1.0h);
    if (p.iters == 0xFFFFFFFFu) scratch[lid] = i; // keeps the threadgroup allocation alive (occupancy knob)
    for (uint k = 0; k < p.iters; ++k) {
        for (int c = 0; c < NF; ++c) f[c] = fma(f[c], xf, yf);
        for (int c = 0; c < NH; ++c) h[c] = fma(h[c], xh, yh);
        for (int c = 0; c < NI; ++c) n[c] = n[c] * xi + yi;
        for (int c = 0; c < NQ; ++c) q[c] = fma(q[c], half2(xh), half2(yh));
    }
    uint hs = 0;
    for (int c = 0; c < NF; ++c) hs = hs * 31u + as_type<uint>(f[c]);
    for (int c = 0; c < NH; ++c) hs = hs * 31u + uint(as_type<ushort>(h[c]));
    for (int c = 0; c < NI; ++c) hs = hs * 31u + uint(n[c]);
    for (int c = 0; c < NQ; ++c)
        hs = hs * 31u + uint(as_type<ushort>(q[c].x)) + 17u * uint(as_type<ushort>(q[c].y));
    out[i] = hs;
}

#define MIX(NF, NH, NI, NQ) \
    template [[host_name("mix_" #NF "_" #NH "_" #NI "_" #NQ)]] kernel void mix<NF, NH, NI, NQ>( \
        device uint*, constant MixParams&, threadgroup uint*, uint, uint);
// alone, ILP sweep
MIX(4, 0, 0, 0) MIX(8, 0, 0, 0) MIX(16, 0, 0, 0) MIX(32, 0, 0, 0)
MIX(0, 4, 0, 0) MIX(0, 8, 0, 0) MIX(0, 16, 0, 0) MIX(0, 32, 0, 0)
MIX(0, 0, 8, 0) MIX(0, 0, 0, 4) MIX(0, 0, 0, 8) MIX(0, 0, 0, 16)
// mixes
MIX(8, 8, 0, 0) MIX(8, 0, 8, 0) MIX(0, 8, 8, 0) MIX(8, 8, 8, 0) MIX(8, 0, 0, 8)
