// OPT-0 spike 3: CPU matrix units.  A MEASUREMENT PROBE, not engine code.
// SGEMM through Accelerate (routed to SME/AMX by the OS) vs a NEON kernel, and
// whether the compiler accepts SME2 streaming code directly.
#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <mach/mach_time.h>

#include <algorithm>
#include <cstdio>
#include <vector>

static double nowMs() {
    static mach_timebase_info_data_t tb = [] { mach_timebase_info_data_t t; mach_timebase_info(&t); return t; }();
    return double(mach_absolute_time()) * tb.numer / tb.denom * 1e-6;
}

#if defined(__ARM_FEATURE_SME)
#include <arm_sme.h>
// Streaming-mode vector length in bytes (only callable if SME is present).
__arm_locally_streaming static unsigned svlBytes() { return unsigned(svcntsb()); }
#endif

int main() {
    for (int n : {512, 1024, 2048, 4096}) {
        std::vector<float> a(size_t(n) * n, 1.0f), b(size_t(n) * n, 0.5f), c(size_t(n) * n, 0.0f);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0f, a.data(), n, b.data(), n, 0.0f, c.data(), n);
        std::vector<double> t;
        for (int r = 0; r < 5; ++r) {
            const double t0 = nowMs();
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0f, a.data(), n, b.data(), n, 0.0f, c.data(), n);
            t.push_back(nowMs() - t0);
        }
        std::sort(t.begin(), t.end());
        std::printf("accelerate sgemm %4d: p50 %8.3f ms -> %7.1f GFLOPS  c[0]=%.1f (expect %.1f)\n", n, t[2],
                    2.0 * n * double(n) * n / (t[2] * 1e-3) * 1e-9, c[0], 0.5 * n);
    }
    // NEON FMA peak on one core: 8 independent float32x4 chains.
    float32x4_t v[8];
    for (int i = 0; i < 8; ++i) v[i] = vdupq_n_f32(float(i));
    const float32x4_t m = vdupq_n_f32(0.999f), k = vdupq_n_f32(0.001f);
    const long iters = 200000000;
    const double t0 = nowMs();
    for (long i = 0; i < iters; ++i)
        for (int j = 0; j < 8; ++j) v[j] = vfmaq_f32(k, v[j], m);
    const double ms = nowMs() - t0;
    float sum = 0;
    for (int i = 0; i < 8; ++i) sum += vaddvq_f32(v[i]);
    std::printf("neon fma 1 core: %.1f GFLOPS (sum %.3f)\n", iters * 8 * 4 * 2 / (ms * 1e-3) * 1e-9, sum);
#if defined(__ARM_FEATURE_SME)
    std::printf("SME compiled in: streaming vector length %u bytes\n", svlBytes());
#else
    std::printf("SME not compiled in (no -march with +sme)\n");
#endif
    return 0;
}
