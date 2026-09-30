// B-24: CPU compute and scheduling on the SoC: scalar vs NEON FMA per core,
// all cores (perflevel "Super" vs "Performance"), SGEMM through Accelerate
// and through direct SME2 (FMOPA outer products in streaming mode), thread
// wake-up latency per QoS class and which cores each QoS runs on.
//
// Serves S-CPU-1..4 of docs/APPLE_SOC_PLAYBOOK.md (§11-§12).
//
// macOS has no thread affinity: "per perflevel" numbers are recovered from
// the per-thread rates of an all-cores run (the fastest N0 threads = level 0),
// and are labelled as such in the notes.

#include "harness.h"

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <dispatch/dispatch.h>
#include <pthread.h>
#include <pthread/qos.h>
#include <sys/sysctl.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <thread>

#if defined(__aarch64__) && __has_include(<arm_sme.h>)
#include <arm_sme.h>
#define SOC_SME2 1
#else
#define SOC_SME2 0
#endif

namespace soc {
namespace {

// --- sysctl helpers ------------------------------------------------------------
int64_t sysctlInt(const char* name, int64_t fallback = 0) {
    int64_t v = 0;
    size_t len = sizeof(v);
    int32_t v32 = 0;
    size_t l32 = sizeof(v32);
    if (sysctlbyname(name, &v, &len, nullptr, 0) == 0 && len == 8) return v;
    if (sysctlbyname(name, &v32, &l32, nullptr, 0) == 0) return v32;
    return fallback;
}
std::string sysctlStr(const char* name) {
    char buf[64] = {};
    size_t len = sizeof(buf);
    if (sysctlbyname(name, buf, &len, nullptr, 0) != 0) return "";
    return buf;
}
std::string lower(std::string s) {
    for (char& c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

// --- FMA kernels -----------------------------------------------------------------
constexpr float kMul = 0.999f, kAdd = 0.001f; // fixed point 1.0: chains stay finite

// 16 independent float32x4 chains (4 FMA pipes x 4-cycle latency need 16).
// Returns the sum of the chains (kept alive so the loop cannot be removed).
float neonKernel(u64 iters) {
    float32x4_t v[16];
    for (int i = 0; i < 16; ++i) v[i] = vdupq_n_f32(float(i) * 0.0625f);
    const float32x4_t m = vdupq_n_f32(kMul), k = vdupq_n_f32(kAdd);
    for (u64 i = 0; i < iters; ++i)
        for (int j = 0; j < 16; ++j) v[j] = vfmaq_f32(k, v[j], m);
    float s = 0;
    for (int i = 0; i < 16; ++i) s += vaddvq_f32(v[i]);
    return s;
}
constexpr double kNeonFlopPerIter = 16.0 * 4.0 * 2.0;

// 16 independent scalar chains through inline asm (the compiler would
// otherwise SLP-vectorise them into NEON).
float scalarKernel(u64 iters) {
    float a0 = 0, a1 = 0.0625f, a2 = 0.125f, a3 = 0.1875f, a4 = 0.25f, a5 = 0.3125f, a6 = 0.375f, a7 = 0.4375f;
    float b0 = 0.5f, b1 = 0.5625f, b2 = 0.625f, b3 = 0.6875f, b4 = 0.75f, b5 = 0.8125f, b6 = 0.875f, b7 = 0.9375f;
    const float m = kMul, k = kAdd;
    for (u64 i = 0; i < iters; ++i) {
        asm volatile(
            "fmadd %s0, %s0, %s16, %s17\n fmadd %s1, %s1, %s16, %s17\n fmadd %s2, %s2, %s16, %s17\n fmadd %s3, %s3, %s16, %s17\n"
            "fmadd %s4, %s4, %s16, %s17\n fmadd %s5, %s5, %s16, %s17\n fmadd %s6, %s6, %s16, %s17\n fmadd %s7, %s7, %s16, %s17\n"
            "fmadd %s8, %s8, %s16, %s17\n fmadd %s9, %s9, %s16, %s17\n fmadd %s10, %s10, %s16, %s17\n fmadd %s11, %s11, %s16, %s17\n"
            "fmadd %s12, %s12, %s16, %s17\n fmadd %s13, %s13, %s16, %s17\n fmadd %s14, %s14, %s16, %s17\n fmadd %s15, %s15, %s16, %s17\n"
            : "+w"(a0), "+w"(a1), "+w"(a2), "+w"(a3), "+w"(a4), "+w"(a5), "+w"(a6), "+w"(a7), "+w"(b0), "+w"(b1), "+w"(b2),
              "+w"(b3), "+w"(b4), "+w"(b5), "+w"(b6), "+w"(b7)
            : "w"(m), "w"(k));
    }
    return a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + b0 + b1 + b2 + b3 + b4 + b5 + b6 + b7;
}
constexpr double kScalarFlopPerIter = 16.0 * 2.0;

// Bit-exact model (one lane of chain j) of both kernels.
float modelChain(float start, u64 iters) {
    float v = start;
    for (u64 i = 0; i < iters; ++i) v = std::fma(v, kMul, kAdd);
    return v;
}
bool kernelsExact() {
    const u64 it = 20000;
    // neon: sum over 16 chains x 4 lanes in the kernel's own order
    float ref = 0;
    for (int i = 0; i < 16; ++i) {
        const float c = modelChain(float(i) * 0.0625f, it);
        const float lane = (c + c) + (c + c); // vaddvq of four equal lanes
        ref += lane;
    }
    const bool neonOk = neonKernel(it) == ref;
    // scalar chain starts: a0..a7 = 0,1/16..7/16, b0..b7 = 8/16..15/16 (same as the NEON starts)
    float sref = 0;
    for (int i = 0; i < 16; ++i) sref += modelChain(float(i) * 0.0625f, it);
    const bool scalarOk = scalarKernel(it) == sref;
    return neonOk && scalarOk;
}

// --- threads ---------------------------------------------------------------------
enum QosKind { Interactive, Initiated, Utility, Background };
constexpr QosKind kQos[] = {Interactive, Initiated, Utility, Background};
const char* qosName(QosKind q) {
    switch (q) {
    case Interactive: return "user_interactive";
    case Initiated: return "user_initiated";
    case Utility: return "utility";
    default: return "background";
    }
}
qos_class_t qosClass(QosKind q) {
    switch (q) {
    case Interactive: return QOS_CLASS_USER_INTERACTIVE;
    case Initiated: return QOS_CLASS_USER_INITIATED;
    case Utility: return QOS_CLASS_UTILITY;
    default: return QOS_CLASS_BACKGROUND;
    }
}

struct Deadline {
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::atomic<bool> stop{false};
};

volatile float g_sink; // keeps kernel results alive

/// Runs `threads` NEON (or scalar) loops for `ms` at `qos`; returns per-thread GFLOPS.
std::vector<double> runThreads(u32 threads, QosKind qos, double ms, bool neon, const std::vector<QosKind>* perThread = nullptr) {
    Deadline d;
    std::vector<double> rate(threads, 0.0);
    std::vector<std::thread> pool;
    const u64 chunk = 1u << 16;
    for (u32 t = 0; t < threads; ++t) {
        pool.emplace_back([&, t] {
            pthread_set_qos_class_self_np(qosClass(perThread ? (*perThread)[t] : qos), 0);
            d.ready.fetch_add(1);
            while (!d.go.load(std::memory_order_acquire)) {}
            const double t0 = nowMs();
            u64 done = 0;
            float sink = 0;
            while (!d.stop.load(std::memory_order_relaxed)) {
                sink += neon ? neonKernel(chunk) : scalarKernel(chunk);
                done += chunk;
            }
            const double dt = nowMs() - t0;
            g_sink = sink;
            rate[t] = (done == 0 || dt <= 0) ? 0.0 : double(done) * (neon ? kNeonFlopPerIter : kScalarFlopPerIter) / (dt * 1e-3) * 1e-9;
        });
    }
    while (d.ready.load() < int(threads)) std::this_thread::yield();
    d.go.store(true, std::memory_order_release);
    std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(ms));
    d.stop.store(true);
    for (auto& th : pool) th.join();
    return rate;
}

/// Wall time of one fixed-size run on a fresh thread at `qos`.
double timedKernel(bool neon, u64 iters, QosKind qos = Interactive) {
    double dt = 0;
    std::thread th([&] {
        pthread_set_qos_class_self_np(qosClass(qos), 0);
        const double t0 = nowMs();
        g_sink = neon ? neonKernel(iters) : scalarKernel(iters);
        dt = nowMs() - t0;
    });
    th.join();
    return dt;
}

Stats gflopsStats(const Stats& timeMs, double flop) {
    Stats s = timeMs;
    const double k = flop * 1e-9 / 1e-3;
    s.median = k / timeMs.median;
    s.min = k / timeMs.max;
    s.max = k / timeMs.min;
    s.p10 = k / timeMs.p90;
    s.p90 = k / timeMs.p10;
    s.mean = k / timeMs.mean;
    return s;
}

// --- wake-up latency ---------------------------------------------------------------
/// Two threads at `qos` ping-pong on two semaphores; one sample = half a round trip (us).
std::vector<double> wakeupSamples(QosKind qos, u32 rounds) {
    dispatch_semaphore_t toB = dispatch_semaphore_create(0), toA = dispatch_semaphore_create(0);
    std::atomic<bool> bReady{false};
    std::thread b([&] {
        pthread_set_qos_class_self_np(qosClass(qos), 0);
        bReady.store(true);
        for (u32 i = 0; i < rounds + 16; ++i) {
            dispatch_semaphore_wait(toB, DISPATCH_TIME_FOREVER);
            dispatch_semaphore_signal(toA);
        }
    });
    std::vector<double> us;
    std::thread a([&] {
        pthread_set_qos_class_self_np(qosClass(qos), 0);
        while (!bReady.load()) std::this_thread::yield();
        for (u32 i = 0; i < rounds + 16; ++i) {
            const double t0 = nowMs();
            dispatch_semaphore_signal(toB);
            dispatch_semaphore_wait(toA, DISPATCH_TIME_FOREVER);
            const double dt = nowMs() - t0;
            if (i >= 16) us.push_back(dt * 1e3 * 0.5); // skip warm-up rounds
        }
    });
    a.join();
    b.join();
    dispatch_release(toB);
    dispatch_release(toA);
    return us;
}

// --- SGEMM ---------------------------------------------------------------------------
struct GemmData {
    u32 n;
    std::vector<float> a, b, c;
};

// Small integers: every product/sum is exact in FP32 for K <= 4096 (|sum| <= 4096*16).
void fillGemm(GemmData& g, u32 n) {
    g.n = n;
    g.a.resize(size_t(n) * n);
    g.b.resize(size_t(n) * n);
    g.c.assign(size_t(n) * n, -1.0f);
    u64 s = 0x1234567ull * n;
    for (float& x : g.a) x = float(int(xorshift64(s) % 9) - 4);
    for (float& x : g.b) x = float(int(xorshift64(s) % 9) - 4);
}
/// Number of wrong entries among 128 sampled C entries (exact integer arithmetic).
u32 gemmWrong(const GemmData& g) {
    const u32 n = g.n;
    u32 bad = 0;
    for (u32 s = 0; s < 128; ++s) {
        const u32 r = (s * 977u + 3u) % n, col = (s * 613u + 5u) % n;
        int64_t acc = 0;
        for (u32 k = 0; k < n; ++k) acc += int64_t(g.a[size_t(r) * n + k]) * int64_t(g.b[size_t(k) * n + col]);
        if (double(g.c[size_t(r) * n + col]) != double(acc)) ++bad;
    }
    return bad;
}
void accelerateGemm(GemmData& g) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, int(g.n), int(g.n), int(g.n), 1.0f, g.a.data(), int(g.n),
                g.b.data(), int(g.n), 0.0f, g.c.data(), int(g.n));
}

#if SOC_SME2
// FMOPA outer products on 4 ZA tiles (2x2 of 16x16 FP32, streaming vector 512 bit).
__arm_locally_streaming __arm_new("za") __attribute__((noinline, target("sme2")))
void smeGemm(const float* at, const float* b, float* c, u32 n) {
    const svbool_t pg = svptrue_b32();
    for (u32 i0 = 0; i0 < n; i0 += 32) {
        for (u32 j0 = 0; j0 < n; j0 += 32) {
            svzero_za();
            for (u32 k = 0; k < n; ++k) {
                const float* ap = at + size_t(k) * n + i0;
                const float* bp = b + size_t(k) * n + j0;
                const svfloat32_t a0 = svld1_f32(pg, ap), a1 = svld1_f32(pg, ap + 16);
                const svfloat32_t b0 = svld1_f32(pg, bp), b1 = svld1_f32(pg, bp + 16);
                svmopa_za32_f32_m(0, pg, pg, a0, b0);
                svmopa_za32_f32_m(1, pg, pg, a0, b1);
                svmopa_za32_f32_m(2, pg, pg, a1, b0);
                svmopa_za32_f32_m(3, pg, pg, a1, b1);
            }
            for (u32 r = 0; r < 16; ++r) {
                float* c0 = c + size_t(i0 + r) * n + j0;
                float* c1 = c + size_t(i0 + 16 + r) * n + j0;
                svst1_hor_za32(0, r, pg, c0);
                svst1_hor_za32(1, r, pg, c0 + 16);
                svst1_hor_za32(2, r, pg, c1);
                svst1_hor_za32(3, r, pg, c1 + 16);
            }
        }
    }
}
/// Pure FMOPA throughput: 4 tiles, no memory traffic in the loop; returns tile[0][0].
__arm_locally_streaming __arm_new("za") __attribute__((noinline, target("sme2")))
float smePeak(u64 iters) {
    const svbool_t pg = svptrue_b32();
    const svfloat32_t a = svdup_f32(1.0f), b = svdup_f32(0.5f);
    svzero_za();
    for (u64 i = 0; i < iters; ++i) {
        svmopa_za32_f32_m(0, pg, pg, a, b);
        svmopa_za32_f32_m(1, pg, pg, a, b);
        svmopa_za32_f32_m(2, pg, pg, a, b);
        svmopa_za32_f32_m(3, pg, pg, a, b);
    }
    float out[16];
    svst1_hor_za32(0, 0, pg, out);
    return out[0];
}
__arm_locally_streaming __attribute__((noinline, target("sme2"))) unsigned smeVectorWords() { return unsigned(svcntw()); }
#endif

void benchCpu(Context& ctx, Report& rep) {
    const bool quick = ctx.quick();
    const double ms = quick ? 50.0 : 100.0;
    const u32 logical = u32(sysctlInt("hw.logicalcpu", 8));
    const u32 n0 = u32(sysctlInt("hw.perflevel0.logicalcpu", 0)), n1 = u32(sysctlInt("hw.perflevel1.logicalcpu", 0));
    const std::string name0 = lower(sysctlStr("hw.perflevel0.name")), name1 = lower(sysctlStr("hw.perflevel1.name"));
    rep.note("perflevel0 = " + sysctlStr("hw.perflevel0.name") + " x" + std::to_string(n0) + ", perflevel1 = " +
             sysctlStr("hw.perflevel1.name") + " x" + std::to_string(n1) + ", logical CPUs " + std::to_string(logical));

    // --- correctness of the kernels ------------------------------------------------
    const bool exact = kernelsExact();
    rep.note(std::string("NEON and scalar kernels vs std::fma model (20000 iterations, bit exact): ") + (exact ? "match" : "DIFFER"));

    // --- single-core scalar vs NEON ---------------------------------------------------
    // Calibrate iterations to ~ms on one thread.
    const u64 probe = 1u << 20;
    const double tp = std::max(1e-3, timedKernel(true, probe));
    const u64 iters = std::max<u64>(1u << 16, u64(double(probe) * ms / tp));
    const double tps = std::max(1e-3, timedKernel(false, probe));
    const u64 itersScalar = std::max<u64>(1u << 16, u64(double(probe) * ms / tps));
    const Stats neonT = ctx.measure([&] { return timedKernel(true, iters); });
    const Stats neon = gflopsStats(neonT, double(iters) * kNeonFlopPerIter);
    const Stats scalarT = ctx.measure([&] { return timedKernel(false, itersScalar); });
    const Stats scalar = gflopsStats(scalarT, double(itersScalar) * kScalarFlopPerIter);
    rep.metric("cpu.neon.gflops_per_core", "GFLOPS", neon, {{"chains", 16}, {"lanes", 4}, {"iters", double(iters)}});
    rep.metric("cpu.scalar.gflops_per_core", "GFLOPS", scalar, {{"chains", 16}, {"iters", double(itersScalar)}});
    rep.value("cpu.neon_over_scalar", "ratio", neon.median / scalar.median, {});
    // Control: 2x work -> 2x time (catches a loop the compiler folded).
    const double t2n = ctx.measure([&] { return timedKernel(true, iters * 2); }, 5).median;
    const double t2s = ctx.measure([&] { return timedKernel(false, itersScalar * 2); }, 5).median;
    const double rN = t2n / neonT.median, rS = t2s / scalarT.median;
    bool linear = rN > 1.85 && rN < 2.15 && rS > 1.85 && rS < 2.15;

    // --- all cores, per perflevel (by rate ranking) ------------------------------------------
    const std::vector<u32> counts = quick ? std::vector<u32>{n0 ? n0 : 1u, logical} : std::vector<u32>{1u, n0 ? n0 : 1u, n0 + n1 / 2, logical};
    std::vector<double> lastRates;
    for (u32 c : counts) {
        if (c == 0 || c > logical) continue;
        std::vector<double> perThread;
        const Stats total = ctx.measure(
            [&] {
                std::vector<double> r = runThreads(c, Interactive, ms, true);
                double sum = 0;
                for (double x : r) sum += x;
                if (c == logical) lastRates = r;
                return sum;
            },
            quick ? 3 : 7);
        rep.metric("cpu.neon.threads_" + std::to_string(c) + ".gflops", "GFLOPS", total, {{"threads", double(c)}});
        if (c == logical) {
            std::vector<double> r = lastRates;
            std::sort(r.rbegin(), r.rend());
            auto mean = [&](u32 a, u32 b) {
                double s = 0;
                for (u32 i = a; i < b && i < r.size(); ++i) s += r[i];
                return b > a ? s / double(std::min<size_t>(b, r.size()) - a) : 0.0;
            };
            if (n0 && n1) {
                rep.value("cpu.neon.all_cores." + name0 + ".gflops_per_core", "GFLOPS", mean(0, n0), {{"cores", double(n0)}});
                rep.value("cpu.neon.all_cores." + name1 + ".gflops_per_core", "GFLOPS", mean(n0, n0 + n1), {{"cores", double(n1)}});
                rep.note("per-level rates come from the ranked per-thread rates of the all-cores run (no affinity on macOS)");
            }
            rep.value("cpu.neon.all_cores.fastest_thread.gflops", "GFLOPS", r.front(), {});
            rep.value("cpu.neon.all_cores.slowest_thread.gflops", "GFLOPS", r.back(), {});
        }
    }

    // --- SGEMM: Accelerate ---------------------------------------------------------------------
    bool gemmOk = true;
    std::string gemmDetail;
    const std::vector<u32> sizes = quick ? std::vector<u32>{512, 2048} : std::vector<u32>{512, 1024, 2048, 4096};
    double accelBest = 0;
    for (u32 n : sizes) {
        GemmData g;
        fillGemm(g, n);
        accelerateGemm(g); // warm-up
        u32 bad = gemmWrong(g);
        const Stats t = ctx.measure([&] {
            const double t0 = nowMs();
            accelerateGemm(g);
            return nowMs() - t0;
        }, quick ? 5 : 9);
        bad += gemmWrong(g);
        const Stats gf = gflopsStats(t, 2.0 * n * double(n) * n);
        const bool largest = n == sizes.back();
        rep.metric(largest ? "cpu.sgemm.accelerate.gflops" : "cpu.sgemm.accelerate.n" + std::to_string(n) + ".gflops", "GFLOPS", gf,
                   {{"n", double(n)}});
        if (largest) rep.value("cpu.sgemm.accelerate.gflops.size_n", "count", n, {});
        accelBest = std::max(accelBest, gf.median);
        if (bad) {
            gemmOk = false;
            gemmDetail += "accelerate n=" + std::to_string(n) + " " + std::to_string(bad) + "/256 wrong; ";
        }
    }

    // --- SGEMM: direct SME2 (FMOPA, streaming mode) ---------------------------------------------
#if SOC_SME2
    const bool hasSme2 = sysctlInt("hw.optional.arm.FEAT_SME2", 0) != 0;
    if (hasSme2 && smeVectorWords() == 16) {
        // peak FMOPA
        const u64 pi = 4'000'000;
        const float chk = smePeak(1000);
        const bool peakOk = chk == 500.0f; // 1000 x (1 x 0.5)
        auto peakTime = [&](u64 it) {
            double dt = 0;
            std::thread th([&] {
                pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
                const double t0 = nowMs();
                g_sink = smePeak(it);
                dt = nowMs() - t0;
            });
            th.join();
            return dt;
        };
        const double p1 = std::max(1e-3, peakTime(pi));
        const u64 pit = std::max<u64>(100000, u64(double(pi) * ms / p1));
        const Stats pt = ctx.measure([&] { return peakTime(pit); });
        rep.metric("cpu.sme2.fmopa.gflops_per_core", "GFLOPS", gflopsStats(pt, double(pit) * 4 * 512), {{"tiles", 4}, {"svl_bytes", 64}});
        const double pr = ctx.measure([&] { return peakTime(pit * 2); }, 5).median / pt.median;
        linear = linear && pr > 1.85 && pr < 2.15;
        if (!peakOk) { gemmOk = false; gemmDetail += "smePeak tile != 500; "; }
        // SGEMM
        for (u32 n : quick ? std::vector<u32>{1024} : std::vector<u32>{1024, 2048}) {
            GemmData g;
            fillGemm(g, n);
            std::vector<float> at(size_t(n) * n);
            for (u32 i = 0; i < n; ++i)
                for (u32 k = 0; k < n; ++k) at[size_t(k) * n + i] = g.a[size_t(i) * n + k];
            auto run = [&] {
                double dt = 0;
                std::thread th([&] {
                    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
                    const double t0 = nowMs();
                    smeGemm(at.data(), g.b.data(), g.c.data(), n);
                    dt = nowMs() - t0;
                });
                th.join();
                return dt;
            };
            run();
            u32 bad = gemmWrong(g);
            const Stats t = ctx.measure(run, quick ? 3 : 5);
            bad += gemmWrong(g);
            rep.metric("cpu.sgemm.sme2.n" + std::to_string(n) + ".gflops", "GFLOPS", gflopsStats(t, 2.0 * n * double(n) * n), {{"n", double(n)}, {"threads", 1}});
            if (bad) { gemmOk = false; gemmDetail += "sme2 n=" + std::to_string(n) + " " + std::to_string(bad) + "/256 wrong; "; }
        }
        rep.note("SME2: hand-written 32x32 FMOPA kernel, ONE thread (one SME unit per cluster), A transposed outside the timed region; Accelerate uses all cores/units");
    } else {
        rep.note(std::string("SME2 direct path skipped: ") + (hasSme2 ? "streaming vector length is not 64 bytes" : "FEAT_SME2 absent"));
    }
#else
    rep.note("SME2 direct path not compiled (no <arm_sme.h>): Accelerate is the measured path");
#endif

    // --- wake-up latency per QoS --------------------------------------------------------------
    const u32 rounds = quick ? 800 : 4000;
    std::map<QosKind, double> wake;
    for (QosKind q : kQos) {
        std::vector<double> us = wakeupSamples(q, rounds);
        const Stats s = phosphor::soc::computeStats(us);
        std::sort(us.begin(), us.end());
        const std::string base = std::string("cpu.wakeup.") + qosName(q);
        rep.metric(base + ".us", "us", s, {{"rounds", double(rounds)}}, false);
        rep.value(base + ".p99_us", "us", us[us.size() * 99 / 100], {{"rounds", double(rounds)}}, false);
        wake[q] = s.median;
    }
    rep.note("wake-up = half a round trip of two threads at the same QoS ping-ponging on dispatch semaphores");

    // --- which cores does each QoS run on: single-thread NEON per QoS ---------------------------
    std::map<QosKind, double> qosRate;
    double contendedBgOverUi = 0;
    for (QosKind q : kQos) {
        const Stats t = ctx.measure([&] { return timedKernel(true, iters, q); }, quick ? 5 : 9);
        const Stats gf = gflopsStats(t, double(iters) * kNeonFlopPerIter);
        rep.metric(std::string("cpu.qos.") + qosName(q) + ".neon_gflops", "GFLOPS", gf, {});
        qosRate[q] = gf.median;
    }
    // Contended: 18 user-interactive + 6 background threads on 18 CPUs.
    {
        std::vector<QosKind> mix(logical, Interactive);
        for (u32 i = 0; i < 6; ++i) mix.push_back(Background);
        std::vector<double> ui, bg;
        u32 starved = 0;
        const Stats st = ctx.measure(
            [&] {
                const std::vector<double> r = runThreads(u32(mix.size()), Interactive, ms, true, &mix);
                double su = 0, sb = 0;
                for (size_t i = 0; i < r.size(); ++i) (mix[i] == Interactive ? su : sb) += r[i];
                for (size_t i = 0; i < r.size(); ++i)
                    if (mix[i] == Background && r[i] == 0.0) ++starved;
                ui.push_back(su / double(logical));
                bg.push_back(sb / 6.0);
                return sb / 6.0;
            },
            quick ? 3 : 7);
        rep.metric("cpu.qos.contended.background.neon_gflops_per_thread", "GFLOPS", st, {{"user_interactive_threads", double(logical)}, {"background_threads", 6}});
        rep.value("cpu.qos.contended.user_interactive.neon_gflops_per_thread", "GFLOPS", phosphor::soc::computeStats(ui).median,
                  {{"user_interactive_threads", double(logical)}, {"background_threads", 6}});
        rep.value("cpu.qos.contended.background.starved_threads", "count", double(starved), {{"repetitions", double(ui.size())}}, false);
        rep.note("contended: background threads that got no CPU during a repetition count as 0 GFLOPS (starved_threads = total over repetitions)");
        contendedBgOverUi = st.median / phosphor::soc::computeStats(ui).median;
    }
    // Background QoS with the machine otherwise busy (all other cores loaded at user-interactive).
    rep.note("cpu.qos.<q>.neon_gflops: one thread at that QoS on an otherwise idle machine; a rate below user_interactive means the class is placed on slower cores or clocked down");

    // --- controls ---------------------------------------------------------------------------------
    rep.negative(exact && linear && gemmOk && qosRate[Background] <= qosRate[Interactive] * 1.05 && contendedBgOverUi <= 1.05,
                 std::string("kernels bit-exact vs std::fma: ") + (exact ? "yes" : "NO") + "; 2x work -> 2x time: neon " +
                     std::to_string(rN).substr(0, 4) + "x, scalar " + std::to_string(rS).substr(0, 4) + "x" + (linear ? "" : " (NOT linear)") +
                     "; SGEMM " + (gemmOk ? "exact vs integer reference (sampled)" : "WRONG: " + gemmDetail) + "; background/user_interactive NEON = " +
                     std::to_string(qosRate[Background] / qosRate[Interactive]).substr(0, 5) + " idle, " + std::to_string(contendedBgOverUi).substr(0, 5) + " contended (18 UI + 6 bg threads)");
    if (!exact || !gemmOk) rep.status(Status::Failed, "wrong results");
}

} // namespace

SOC_BENCH("B-24", "cpu.compute", "CPU: scalar/NEON FMA, all cores, SGEMM (Accelerate, SME2), wake-up latency per QoS", benchCpu);

} // namespace soc
