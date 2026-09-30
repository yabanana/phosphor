// B-28: known GPU workload for the commit / event latency benchmark.
// 4 independent FMA chains per thread (per-thread seeds, MathModeSafe library
// so the CPU model is bit-exact), result stored and checked by the CPU.
#include <metal_stdlib>
using namespace metal;

struct WorkParams {
    uint iters;
    uint pad[3];
};

kernel void b28_work(device uint* out [[buffer(0)]], constant WorkParams& p [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {
    float a = float(i & 1023u) * 1e-3f, b = a + 1.0f, c = a + 2.0f, d = a + 3.0f;
    for (uint k = 0; k < p.iters; ++k) {
        a = fma(a, 0.999f, 0.001f);
        b = fma(b, 0.999f, 0.001f);
        c = fma(c, 0.999f, 0.001f);
        d = fma(d, 0.999f, 0.001f);
    }
    out[i] = as_type<uint>(a) ^ (as_type<uint>(b) * 3u) ^ (as_type<uint>(c) * 5u) ^ (as_type<uint>(d) * 7u);
}
