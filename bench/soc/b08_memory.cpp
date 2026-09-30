// B-08: memory hierarchy -- pointer-chase latency vs working set, streaming
// read bandwidth vs working set, DRAM write and copy bandwidth, and a
// documented model fit for the SLC size.
//
// Serves S-MEM-1..3 and S-MEM-6 of docs/APPLE_SOC_PLAYBOOK.md.
//
// SLC estimate (model, NOT a measurement): the spike (docs/opt-log.md, OPT-0
// point 5) found no sharp knee; bandwidth above DRAM up to ~384 MiB is
// compatible with a thrash-resistant last-level cache with hit ratio
// h = min(1, C / WS).  Time per byte of a streaming read is then
//   t(WS) = (1 - h) / B_dram + h / B_cache.
// For each C on a log grid (8 MiB .. 2 GiB, 1% steps) the pair (1/B_dram,
// 1/B_cache) is the least-squares solution (relative residuals) over the tail
// of the curve (WS >= 64 MiB: beyond the on-chip caches); the C with the
// smallest relative RMS residual is the estimate.  The residual is reported.

#include "harness.h"

#include <CoreFoundation/CoreFoundation.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <string>

extern "C" {
typedef struct IOReportSubscription* IOReportSubscriptionRef;
CFDictionaryRef IOReportCopyChannelsInGroup(CFStringRef, CFStringRef, uint64_t, uint64_t, uint64_t);
IOReportSubscriptionRef IOReportCreateSubscription(void*, CFMutableDictionaryRef, CFMutableDictionaryRef*, uint64_t,
                                                   CFTypeRef);
CFDictionaryRef IOReportCreateSamples(IOReportSubscriptionRef, CFMutableDictionaryRef, CFTypeRef);
CFDictionaryRef IOReportCreateSamplesDelta(CFDictionaryRef, CFDictionaryRef, CFTypeRef);
int32_t IOReportStateGetCount(CFDictionaryRef);
int64_t IOReportStateGetResidency(CFDictionaryRef, int32_t);
}

namespace soc {
namespace {

struct MemParams {
    u32 iters, zero, words, seed;
};
static_assert(sizeof(MemParams) == 16);

constexpr size_t KiB = size_t(1) << 10, MiB = size_t(1) << 20, GiB = size_t(1) << 30;
constexpr u32 kMaxThreads = 327680; // 40 cores x 8192

Stats toRate(const Stats& s, double scale) { // value = scale / time
    Stats r = s;
    r.median = scale / s.median;
    r.min    = scale / s.max;
    r.max    = scale / s.min;
    r.p10    = scale / s.p90;
    r.p90    = scale / s.p10;
    r.mean   = scale / s.mean;
    return r;
}

struct Rig {
    Context& ctx;
    MTL::Buffer* params;
    MTL::Buffer* out;
};

double timeKernel(Rig& r, MTL::ComputePipelineState* pso, u64 a0, u64 a2, const MemParams& p, u32 threads, u32 tg) {
    std::memcpy(r.params->contents(), &p, sizeof(p));
    ComputeTimer t(r.ctx);
    MTL4::ComputeCommandEncoder* e = t.begin();
    r.ctx.table()->setAddress(a0, 0);
    r.ctx.table()->setAddress(r.params->gpuAddress(), 1);
    r.ctx.table()->setAddress(a2, 2);
    e->setComputePipelineState(pso);
    e->setArgumentTable(r.ctx.table());
    e->dispatchThreads(MTL::Size::Make(threads, 1, 1), MTL::Size::Make(tg, 1, 1));
    t.lap();
    return t.finish()[0];
}

u32 threadsFor(u32 words) { return std::min<u32>(words, kMaxThreads); }

// --- CPU checks ------------------------------------------------------------------
std::vector<u32> sampleThreads(u32 n) {
    std::vector<u32> v;
    for (u32 c : {0u, 1u, 2u, 255u, 256u, n / 2, n > 2 ? n - 2 : 0, n - 1})
        if (c < n && std::find(v.begin(), v.end(), c) == v.end()) v.push_back(c);
    return v;
}

bool checkRead(const u32* src, u32 words, u32 n, u32 iters, const u32* out) {
    for (u32 i : sampleThreads(n)) {
        u32 s[4] = {0, 0, 0, 0};
        for (u32 j = i; j < words; j += n)
            for (int c = 0; c < 4; ++c) s[c] += src[size_t(j) * 4 + c];
        for (int c = 0; c < 4; ++c)
            if (out[size_t(i) * 4 + c] != s[c] * iters) return false;
    }
    return true;
}

bool checkWrite(const u32* dst, u32 words, u32 iters, u32 seed) {
    u64 st = 12345;
    for (int s = 0; s < 64; ++s) {
        const u32 j = s == 0 ? 0 : s == 1 ? words - 1 : u32(xorshift64(st) % words);
        const u32 v = j * 2654435761u + (iters - 1) * 40503u + seed;
        const u32 e[4] = {v, v ^ 0x55555555u, v * 3u, ~v};
        for (int c = 0; c < 4; ++c)
            if (dst[size_t(j) * 4 + c] != e[c]) return false;
    }
    return true;
}

bool checkCopy(const u32* src, const u32* dst, u32 words) {
    u64 st = 777;
    for (int s = 0; s < 64; ++s) {
        const u32 j = s == 0 ? 0 : s == 1 ? words - 1 : u32(xorshift64(st) % words);
        for (int c = 0; c < 4; ++c)
            if (dst[size_t(j) * 4 + c] != src[size_t(j) * 4 + c]) return false;
    }
    return true;
}

// Sattolo: a uniformly random single cycle over `lines` 128-byte lines.
// GPU fabric state ("GPU Stats" / "AFR Performance States", IOReport, no
// root): mean P-state index of the busy time over a window.  The single
// thread chase runs with the fabric at ~P5 of 13 (measured), so its
// latencies depend on the fabric governor, not only on the memory.
class AfrState {
public:
    AfrState() {
        CFDictionaryRef c = IOReportCopyChannelsInGroup(CFSTR("GPU Stats"), CFSTR("AFR Performance States"), 0, 0, 0);
        if (!c) return;
        CFMutableDictionaryRef m = CFDictionaryCreateMutableCopy(nullptr, 0, c);
        CFRelease(c);
        sub_ = IOReportCreateSubscription(nullptr, m, &subbed_, 0, nullptr);
        CFRelease(m);
    }
    ~AfrState() {
        if (start_) CFRelease(start_);
        if (subbed_) CFRelease(subbed_);
        if (sub_) CFRelease(reinterpret_cast<CFTypeRef>(sub_));
    }
    AfrState(const AfrState&) = delete;
    AfrState& operator=(const AfrState&) = delete;
    void begin() {
        if (!sub_) return;
        if (start_) CFRelease(start_);
        start_ = IOReportCreateSamples(sub_, subbed_, nullptr);
    }
    /// Mean P index of the busy (non-OFF) time since begin(), -1 if unknown.
    double end() {
        if (!sub_ || !start_) return -1;
        CFDictionaryRef now = IOReportCreateSamples(sub_, subbed_, nullptr);
        CFDictionaryRef d = IOReportCreateSamplesDelta(start_, now, nullptr);
        CFRelease(now);
        CFRelease(start_);
        start_ = nullptr;
        double busy = 0, weighted = 0;
        auto arr = static_cast<CFArrayRef>(CFDictionaryGetValue(d, CFSTR("IOReportChannels")));
        for (CFIndex i = 0; arr && i < CFArrayGetCount(arr); ++i) {
            auto c = static_cast<CFDictionaryRef>(CFArrayGetValueAtIndex(arr, i));
            for (int32_t q = 1; q < IOReportStateGetCount(c); ++q) {
                const double r = double(IOReportStateGetResidency(c, q));
                busy += r;
                weighted += r * q;
            }
        }
        CFRelease(d);
        return busy > 0 ? weighted / busy : -1;
    }

private:
    IOReportSubscriptionRef sub_ = nullptr;
    CFMutableDictionaryRef subbed_ = nullptr;
    CFDictionaryRef start_ = nullptr;
};

void buildCycle(u32* next, size_t lines, u64 seed) {
    std::vector<u32> a(lines);
    std::iota(a.begin(), a.end(), 0u);
    u64 s = seed;
    for (size_t i = lines - 1; i > 0; --i) std::swap(a[i], a[xorshift64(s) % i]);
    for (size_t i = 0; i < lines; ++i) next[i * 32] = a[i] * 32u;
}

// Page-local cycle: pages (16 KiB = 128 lines) in random order, the lines
// of each page in random order before the next page: the same DRAM lines
// as a random cycle, but one TLB / page-table miss per 128 steps.
void buildPageLocalCycle(u32* next, size_t lines, u64 seed) {
    constexpr size_t kLinesPerPage = 16384 / 128;
    const size_t pages = lines / kLinesPerPage;
    std::vector<u32> pageOrder(pages), lineOrder(kLinesPerPage), order;
    order.reserve(lines);
    std::iota(pageOrder.begin(), pageOrder.end(), 0u);
    u64 s = seed;
    for (size_t i = pages - 1; i > 0; --i) std::swap(pageOrder[i], pageOrder[xorshift64(s) % (i + 1)]);
    for (u32 p : pageOrder) {
        std::iota(lineOrder.begin(), lineOrder.end(), 0u);
        for (size_t i = kLinesPerPage - 1; i > 0; --i) std::swap(lineOrder[i], lineOrder[xorshift64(s) % (i + 1)]);
        for (u32 l : lineOrder) order.push_back(p * u32(kLinesPerPage) + l);
    }
    for (size_t k = 0; k < order.size(); ++k) next[size_t(order[k]) * 32] = order[(k + 1) % order.size()] * 32u;
}

u32 walk(const u32* next, u32 start, u32 steps) {
    u32 idx = start;
    for (u32 k = 0; k < steps; ++k) idx = next[idx];
    return idx;
}

// --- SLC model fit -----------------------------------------------------------------
struct Fit {
    double C = 0, a = 0, b = 0, rms = 1e9; // C in MiB, a/b = ns per byte
    bool ok = false, atBound = false;
    size_t points = 0;
};

Fit fitSlc(const std::vector<double>& wsMiB, const std::vector<double>& bwGBs) {
    std::vector<double> ws, y;
    for (size_t i = 0; i < wsMiB.size(); ++i)
        if (wsMiB[i] >= 64.0 && bwGBs[i] > 0) {
            ws.push_back(wsMiB[i]);
            y.push_back(1.0 / bwGBs[i]); // ns per byte
        }
    Fit best;
    best.points = ws.size();
    if (ws.size() < 4) return best;
    const double lo = 8.0, hi = 2048.0;
    for (double C = lo; C <= hi; C *= 1.01) {
        double suu = 0, suv = 0, svv = 0, syu = 0, syv = 0;
        for (size_t i = 0; i < ws.size(); ++i) {
            const double h = std::min(1.0, C / ws[i]);
            const double u = (1 - h) / y[i], v = h / y[i];
            suu += u * u; suv += u * v; svv += v * v; syu += u; syv += v;
        }
        const double det = suu * svv - suv * suv;
        if (std::fabs(det) < 1e-12 * (suu * svv + 1e-30)) continue;
        const double a = (syu * svv - syv * suv) / det;
        const double b = (suu * syv - suv * syu) / det;
        if (a <= 0 || b <= 0) continue;
        double r2 = 0;
        for (size_t i = 0; i < ws.size(); ++i) {
            const double h = std::min(1.0, C / ws[i]);
            const double m = (a * (1 - h) + b * h) / y[i] - 1.0;
            r2 += m * m;
        }
        const double rms = std::sqrt(r2 / double(ws.size()));
        if (rms < best.rms) {
            const size_t p = best.points;
            best = {C, a, b, rms, true, false, p};
        }
    }
    if (best.ok) best.atBound = best.C < lo * 1.05 || best.C > hi / 1.05;
    return best;
}

void benchMemory(Context& ctx, Report& rep) {
    const bool quick = ctx.quick();
    MTL::Library* lib = ctx.library("b08_memory.metal", /*fastMath=*/true);
    MTL::ComputePipelineState* chase = ctx.compute(lib, "b08_chase");
    MTL::ComputePipelineState* read  = ctx.compute(lib, "b08_read");
    MTL::ComputePipelineState* write = ctx.compute(lib, "b08_write");
    MTL::ComputePipelineState* copy  = ctx.compute(lib, "b08_copy");

    Rig rig{ctx, ctx.buffer(256), ctx.buffer(8 * MiB)};
    // Random content: zero-filled buffers read ~25% faster on chip (spike).
    MTL::Buffer* A = ctx.randomBuffer(2 * GiB);
    const u32* a32 = static_cast<const u32*>(A->contents());
    u32* w32       = static_cast<u32*>(A->contents());
    const u64 base = A->gpuAddress();
    const u32* out32 = static_cast<const u32*>(rig.out->contents());
    const double targetMs = quick ? 0.3 : 0.5;
    bool resultsOk = true;
    std::string wrongWhat;
    auto bad = [&](const std::string& s) { resultsOk = false; wrongWhat += s + " "; };

    // --- (b) read bandwidth curve ---------------------------------------------------
    const std::vector<size_t> readMiB = quick ? std::vector<size_t>{1, 4, 16, 32, 64, 96, 128, 192, 256, 384, 512, 1024}
                                              : std::vector<size_t>{1, 2, 4, 8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128,
                                                                    160, 192, 256, 320, 384, 512, 768, 1024};
    std::vector<double> rdWs, rdBw;
    auto readTime = [&](size_t bytes, u64 addr, const u32* host, u32 passes, const std::string& tag) {
        const u32 words = u32(bytes / 16), n = threadsFor(words);
        const Stats s = ctx.measure([&] {
            return timeKernel(rig, read, addr, rig.out->gpuAddress(), {passes, 0, words, 0}, n, 256);
        }, std::max<u32>(ctx.reps(), 15)); // cheap: 15 even with --quick
        if (!checkRead(host, words, n, passes, out32)) bad("read " + tag);
        return s;
    };
    for (size_t mib : readMiB) {
        const size_t bytes = mib * MiB;
        const u32 words = u32(bytes / 16);
        // Calibrate the pass count to ~targetMs (probe ~64 MiB of reads).
        const u32 probe = u32(std::max<size_t>(1, 64 * MiB / bytes));
        const double tp = std::max(1e-4, ctx.measure([&] {
            return timeKernel(rig, read, base, rig.out->gpuAddress(), {probe, 0, words, 0}, threadsFor(words), 256);
        }, 7).min); // min: contention only adds time
        const u32 passes = std::max<u32>(1, u32(targetMs / (tp / double(probe))));
        const Stats t = readTime(bytes, base, a32, passes, "ws" + std::to_string(mib));
        const Stats bw = toRate(t, double(bytes) * passes * 1e-6);
        rep.metric("read_bw.ws_" + std::to_string(mib * 1024) + "KiB", "GB/s", bw,
                   {{"ws_KiB", double(mib * 1024)}, {"passes", double(passes)}, {"threads", double(threadsFor(words))}, {"ms", t.median}});
        rdWs.push_back(double(mib));
        rdBw.push_back(bw.median);
        ctx.keepWarm(15);
    }

    // Control 1: 2x data -> 2x time at the DRAM size (1 GiB, 1 vs 2 passes).
    double dramRatio = 0;
    {
        const Stats t1 = readTime(GiB, base, a32, 1, "dram1");
        ctx.keepWarm(20);
        const Stats t2 = readTime(GiB, base, a32, 2, "dram2");
        dramRatio = t2.min / t1.min; // minima: contention only adds time
    }
    // Random vs zero data on chip (16 MiB): noted, not a pass criterion.
    double zeroRatio = 0;
    {
        MTL::Buffer* Z = ctx.buffer(16 * MiB);
        std::memset(Z->contents(), 0, 16 * MiB);
        const u32 passes = 64;
        const Stats tz = readTime(16 * MiB, Z->gpuAddress(), static_cast<const u32*>(Z->contents()), passes, "zero16");
        const Stats tr = readTime(16 * MiB, base, a32, passes, "rand16");
        zeroRatio = tr.median / tz.median; // >1: zeros faster
        rep.value("read_bw.zero_data_speedup.ws_16384KiB", "ratio", zeroRatio, {{"passes", double(passes)}});
    }
    ctx.keepWarm(30);

    // --- (c) write and copy bandwidth on DRAM sizes ---------------------------------------
    double writeBw = 0, copyBw = 0;
    {
        // Copy first (source must still be the random content): [0,1G) -> [1G,2G).
        const u32 words = u32(GiB / 16), n = threadsFor(words);
        const Stats t = ctx.measure([&] {
            return timeKernel(rig, copy, base, base + GiB, {1, 0, words, 0}, n, 256);
        });
        if (!checkCopy(a32, a32 + GiB / 4, words)) bad("copy");
        const Stats bw = toRate(t, double(2 * GiB) * 1e-6);
        copyBw = bw.median;
        rep.metric("copy_bw.dram", "GB/s", bw, {{"bytes_each", double(GiB)}, {"ms", t.median}, {"counts_read_plus_write", 1}});
        ctx.keepWarm(30);
        const u32 seed = 0xC0FFEEu;
        const Stats tw = ctx.measure([&] {
            return timeKernel(rig, write, base + GiB, rig.out->gpuAddress(), {1, 0, words, seed}, n, 256);
        });
        if (!checkWrite(a32 + GiB / 4, words, 1, seed)) bad("write");
        const Stats bww = toRate(tw, double(GiB) * 1e-6);
        writeBw = bww.median;
        rep.metric("write_bw.dram", "GB/s", bww, {{"bytes", double(GiB)}, {"ms", tw.median}});
        ctx.keepWarm(30);
    }

    // --- (a) latency curve ------------------------------------------------------------------
    const std::vector<size_t> latKiB = quick ? std::vector<size_t>{16, 128, 1024, 4096, 32768, 131072, 524288, 1048576}
                                             : std::vector<size_t>{16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
                                                                   65536, 131072, 262144, 524288, 1048576, 2097152};
    std::vector<double> latNs;
    double chaseRatio = 0;
    GpuState chaseState;          // own IOReport subscription (the suite's window stays intact)
    AfrState afr;
    std::string lowClockPoints;   // latency points measured below the top P-state after 3 attempts
    for (size_t kib : latKiB) {
        const size_t lines = kib * KiB / 128;
        buildCycle(w32, lines, 88172645463325252ull + kib);
        // Every run starts where the previous one ended: the chain keeps
        // touching fresh lines (re-walking the same N lines would measure a
        // cache-resident working set of N x 128 bytes, not the whole WS).
        u32 cur = 0;
        auto run = [&](u32 steps) {
            const double ms = timeKernel(rig, chase, base, rig.out->gpuAddress(), {steps, 0, 0, cur}, 1, 1);
            if (out32[0] != walk(a32, cur, steps)) bad("chase" + std::to_string(kib));
            cur = out32[0];
            return ms;
        };
        const double tp = std::max(1e-4, ctx.measure([&] { return run(512); }, 7).min); // min: contention only adds time
        // ~3 ms (4N ~12 ms): with 0.6 ms spans the 2 GiB point varied 363..465 ns between runs.
        const u32 N = std::clamp<u32>(u32((quick ? 0.6 : 3.0) / (tp / 512.0)), 256, 400000);
        const u32 latReps = std::max<u32>(ctx.reps(), 15); // cheap: 15 even with --quick
        // DVFS guard: a single-thread chase barely loads the GPU, so its
        // clock can drop in the middle of the measurement (one 2 GiB point
        // read 1224 ns instead of ~475 in a 3-run battery).  The P-state
        // residency of the window is checked; up to 2 re-measurements.
        Stats s1, s2;
        phosphor::soc::GpuWindow win;
        double afrMean = -1;
        u32 attempts = 0;
        do {
            if (attempts) ctx.keepWarm(50);
            chaseState.begin();
            afr.begin();
            s1 = ctx.measure([&] { return run(N); }, latReps);
            s2 = ctx.measure([&] { return run(4 * N); }, latReps);
            win = chaseState.end();
            afrMean = afr.end();
        } while (++attempts < 3 && chaseState.available() && win.topStateShare < 0.9);
        if (win.topStateShare >= 0 && win.topStateShare < 0.9)
            lowClockPoints += std::to_string(kib) + "KiB(" + std::to_string(int(win.topStateShare * 100)) + "%) ";
        // Slope between N and 4N steps cancels the fixed dispatch cost (large and noisy, ~50-80 us, hence the long chains).  Minima:
        // other GPU clients only ever add time to a single-thread chase.
        const double ns = std::max(0.0, (s2.min - s1.min)) * 1e6 / (3.0 * N);
        // Linearity control on the smallest set: stable (CV 0%), and a loop
        // the compiler removed still shows as ~1x.  At DRAM sizes the time
        // per step follows the GPU fabric (AFR) P-state, which drops to ~5 of
        // 13 during a single-thread chase (IOReport, measured): no stable
        // ratio exists there (4N/N read 3.9..8.8x between runs).
        if (kib == latKiB.front()) chaseRatio = s2.min / s1.min;
        latNs.push_back(ns);
        rep.value("latency.ws_" + std::to_string(kib) + "KiB", "ns", ns,
                  {{"ws_KiB", double(kib)}, {"steps", double(N)}, {"t1_min_ms", s1.min}, {"t2_min_ms", s2.min},
                   {"top_state_share", win.topStateShare}, {"attempts", double(attempts)}, {"afr_mean_pstate", afrMean},
                   {"median_slope_ns", std::max(0.0, s2.median - s1.median) * 1e6 / double(N)}}, false);
        ctx.keepWarm(15);
    }

    // --- (a2) page-local chase at the largest set: same lines, 1 page miss per 128 steps ------
    double latPageLocal = 0;
    {
        const size_t kib = latKiB.back();
        const size_t lines = kib * KiB / 128;
        buildPageLocalCycle(w32, lines, 0x5DEECE66Dull + kib);
        u32 cur = 0;
        auto run = [&](u32 steps) {
            const double ms = timeKernel(rig, chase, base, rig.out->gpuAddress(), {steps, 0, 0, cur}, 1, 1);
            if (out32[0] != walk(a32, cur, steps)) bad("chase_page_local");
            cur = out32[0];
            return ms;
        };
        const double tp = std::max(1e-4, ctx.measure([&] { return run(512); }, 7).min);
        const u32 NL = std::clamp<u32>(u32((quick ? 8.0 : 30.0) / (tp / 512.0)), 1024, 4000000);
        const Stats sp = ctx.measure([&] { return run(NL); }, quick ? 3 : 7);
        latPageLocal = sp.median * 1e6 / double(NL);
        rep.value("latency_dram.page_local", "ns", latPageLocal,
                  {{"ws_KiB", double(kib)}, {"steps", double(NL)}, {"t_min_ms", sp.min}, {"t_median_ms", sp.median}}, false);
        ctx.keepWarm(15);
    }

    // --- (d) summaries ------------------------------------------------------------------------
    const double latL1 = latNs.front(), latDram = latNs.back();
    if (!lowClockPoints.empty()) rep.note("latency points below the top P-state after 3 attempts: " + lowClockPoints);
    rep.value("latency_l1", "ns", latL1, {{"ws_KiB", double(latKiB.front())}}, false);
    rep.value("latency_dram", "ns", latDram, {{"ws_KiB", double(latKiB.back())}}, false);
    const double dramBw = rdBw.back();
    rep.value("dram_bw", "GB/s", dramBw, {{"ws_KiB", double(readMiB.back() * 1024)}});
    std::vector<double> on;
    for (size_t i = 0; i < rdWs.size(); ++i)
        if (rdWs[i] >= 4 && rdWs[i] <= 32) on.push_back(rdBw[i]);
    const double onchip = phosphor::soc::computeStats(on).median;
    rep.value("onchip_bw", "GB/s", onchip, {{"ws_min_KiB", 4096}, {"ws_max_KiB", 32768}, {"points", double(on.size())}});

    const Fit fit = fitSlc(rdWs, rdBw);
    rep.value("slc_size_estimate", "MiB", fit.ok ? fit.C : 0, {{"fit_ws_min_MiB", 64}, {"fit_points", double(fit.points)}});
    if (fit.ok) {
        rep.value("slc_fit.rms_rel_residual", "ratio", fit.rms, {}, false);
        rep.value("slc_fit.dram_bw", "GB/s", 1.0 / fit.a, {});
        rep.value("slc_fit.cache_bw", "GB/s", 1.0 / fit.b, {});
    }
    rep.value("write_bw_over_read_bw.dram", "ratio", writeBw / dramBw, {});
    rep.note("slc_size_estimate is a MODEL estimate (h = min(1, C/WS), least squares on the read curve for WS >= 64 MiB), not a measured size; see slc_fit.* for the residual");
    rep.note("latency = slope of the minimum times between N and 4N chase steps (15 reps) (128-byte lines, includes TLB effects; a single GPU thread, core unknown: no per-die latency asymmetry test, S-MEM-6 open)");
    rep.note("onchip_bw = median read bandwidth over 4..32 MiB working sets (GPU-wide, above DRAM); read passes carry a runtime-zero offset so loads cannot be reused");
    rep.note("IOReport PMP 'DCS BW' under-reads (spike): not used");
    if (!fit.ok) rep.status(Status::Partial, "SLC fit failed (too few points or degenerate)");
    else if (fit.rms > 0.15 || fit.atBound)
        rep.status(Status::Partial, "SLC model fit poor (rms " + std::to_string(fit.rms).substr(0, 5) + (fit.atBound ? ", C at grid bound" : "") + ")");

    // --- negative controls -----------------------------------------------------------------------
    const bool lin = dramRatio > 1.8 && dramRatio < 2.2 && chaseRatio > 2.5 && chaseRatio < 6.5;
    rep.negative(lin && resultsOk,
                 "1 GiB read 2 passes/1 pass = " + std::to_string(dramRatio).substr(0, 5) + "x, chase 4N/N steps at the L1 set (want 2.5..6.5) = " +
                     std::to_string(chaseRatio).substr(0, 5) + "x" + (resultsOk ? "; every kernel result matches the CPU" : "; WRONG RESULTS: " + wrongWhat));
    rep.negative(latDram > 5.0 * latL1, "latency_dram " + std::to_string(int(latDram)) + " ns vs 5 x latency_l1 " + std::to_string(int(5 * latL1)) + " ns");
    // Bandwidth monotone-ish from the peak: an up-step > 30% (plateaus vary up to ~25% between runs, spike) is a violation.
    const size_t peak = size_t(std::max_element(rdBw.begin(), rdBw.end()) - rdBw.begin());
    std::string viol;
    int nv = 0;
    for (size_t i = peak + 1; i < rdBw.size(); ++i)
        if (rdBw[i] > rdBw[i - 1] * 1.30) {
            ++nv;
            viol += std::to_string(int(rdWs[i])) + "MiB ";
        }
    rep.negative(nv <= 2, "read curve monotone-ish after the peak: " + std::to_string(nv) + " violations (>30% up-step)" + (nv ? ": " + viol : ""));
    rep.note("random vs zero data at 16 MiB: zeros " + std::to_string(zeroRatio).substr(0, 4) + "x the speed of random data; copy_bw " +
             std::to_string(int(copyBw)) + " GB/s counts read+write; M5 Max nominal 614 GB/s is an external reference");
}

} // namespace

SOC_BENCH("B-08", "memory.hierarchy", "Latency and bandwidth per level (L1, SLC, DRAM), SLC size model fit", benchMemory);

} // namespace soc
