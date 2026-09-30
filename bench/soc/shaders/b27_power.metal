// B-27: fixed GPU load whose time tracks the clock (FMA chains, stored).
#include <metal_stdlib>
using namespace metal;
struct PowerParams { uint iters; uint zero; };
kernel void power_load(device uint* out [[buffer(0)]], constant PowerParams& p [[buffer(1)]],
                       uint i [[thread_position_in_grid]]) {
    float a = float(i & 1023u) * 1e-3f, b = a + 1.0f, c = a + 2.0f, d = a + 3.0f;
    float e = a + 4.0f, f = a + 5.0f, g = a + 6.0f, h = a + 7.0f;
    for (uint k = 0; k < p.iters; ++k) {
        a = fma(a, 0.999f, 0.001f); b = fma(b, 0.999f, 0.001f); c = fma(c, 0.999f, 0.001f); d = fma(d, 0.999f, 0.001f);
        e = fma(e, 0.999f, 0.001f); f = fma(f, 0.999f, 0.001f); g = fma(g, 0.999f, 0.001f); h = fma(h, 0.999f, 0.001f);
    }
    out[i] = as_type<uint>(a + b + c + d + e + f + g + h) & p.zero;
}
