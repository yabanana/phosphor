// B-04: live registers vs throughput.  N FP32 values stay live across a loop
// (each is updated by an FMA every round and all are summed at the end; N is
// a template parameter and the loops are fully unrolled, so every value is an
// SSA register: no dynamic indexing).  Structure per thread:
//     for l in 0..loads:                   (fixed count)
//         [LD mode] v0 += gather(tab[hash(v0, i, l)])   (dependent load: latency exposed
//                                                        unless other threads hide it)
//         for m in 0..rounds: v[c] = fma(v[c], x, y) for all c
// The number of FMAs per thread is loads * rounds * N with rounds ~ 1024 / N,
// i.e. about constant across N.  The N values live across each load are what
// occupies the register file (S-OCC-1..3): when the registers do not fit the
// hardware lowers occupancy (fewer threads hide the load latency) or the
// compiler spills.  ALU mode: same without the loads.  DYN mode (control):
// the same FMAs on an array indexed dynamically (lives on the stack).
#include <metal_stdlib>
using namespace metal;

struct RegParams {
    uint  loads, rounds;
    float x, y;
    uint  mask, pad;
};

enum { LD, ALU, DYN };

template <int N, int MODE>
kernel void regs(device uint* out [[buffer(0)]], constant RegParams& p [[buffer(1)]],
                 device const uint* tab [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    const float x = p.x, y = p.y;
    float v[N];
    _Pragma("clang loop unroll(full)")
    for (int c = 0; c < N; ++c) v[c] = float(int((i + uint(c) * 7u) & 15u)) * 0.0625f + 1.0f;
    for (uint l = 0; l < p.loads; ++l) {
        if (MODE == LD) {
            const uint idx = (as_type<uint>(v[0]) * 2654435761u + i * 40503u + l) & p.mask;
            v[0] = fma(float(tab[idx] & 15u), 0.001f, v[0]);
        }
        for (uint m = 0; m < p.rounds; ++m) {
            if (MODE == DYN) {
                for (int c = 0; c < N; ++c) {
                    const uint k = (uint(c) + m) & uint(N - 1);
                    v[k] = fma(v[k], x, y);
                }
            } else {
                _Pragma("clang loop unroll(full)")
                for (int c = 0; c < N; ++c) v[c] = fma(v[c], x, y);
            }
        }
    }
    float s = 0.0f;
    _Pragma("clang loop unroll(full)")
    for (int c = 0; c < N; ++c) s += v[c];
    out[i] = as_type<uint>(s);
}

#define REGS(N, MODE, NAME) \
    template [[host_name("regs_" #NAME "_" #N)]] kernel void regs<N, MODE>(device uint*, constant RegParams&, device const uint*, uint);
#define REGS_N(N) REGS(N, LD, ld) REGS(N, ALU, alu)
REGS_N(8) REGS_N(16) REGS_N(32) REGS_N(48) REGS_N(64) REGS_N(96) REGS_N(104) REGS_N(112) REGS_N(120) REGS_N(128) REGS_N(160) REGS_N(192) REGS_N(256)
REGS(32, DYN, dyn)
