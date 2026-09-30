// OPT-0.1 harness kernels: timing anchor and the warm-up load.
#include <metal_stdlib>
using namespace metal;

// 1-thread dispatch that anchors a timestamp (an encoder without a dispatch
// is dropped by the driver, measured in F4).
kernel void soc_anchor(device uint* scratch [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i == 0) scratch[0] = 0u;
}

struct BusyParams { uint iters; uint zero; };

// Warm-up / keep-warm load: 4 independent FMA chains, result stored (masked
// by a runtime zero so the loop is kept).
kernel void soc_busy(device uint* scratch [[buffer(0)]], constant BusyParams& p [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {
    float a = float(i & 1023u) * 1e-3f, b = a + 1.0f, c = a + 2.0f, d = a + 3.0f;
    for (uint k = 0; k < p.iters; ++k) {
        a = fma(a, 0.999f, 0.001f); b = fma(b, 0.999f, 0.001f);
        c = fma(c, 0.999f, 0.001f); d = fma(d, 0.999f, 0.001f);
    }
    scratch[1 + (i & 1023u)] = as_type<uint>(a + b + c + d) & p.zero;
}
