// B-05: threadgroup memory bandwidth by access stride (bank conflicts) and
// load-to-use latency (pointer chase).
//
// Serves S-SIMD-3 (and S-OCC-3) of docs/APPLE_SOC_PLAYBOOK.md.

#include "harness.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace soc {
namespace {

struct TgParams {
    u32 iters, stride, steps, pad;
};
static_assert(sizeof(TgParams) == 16);

constexpr u32 kWords = 4096;
constexpr u32 kTg = 256;
constexpr u32 kThreads = 1u << 19; // 2048 threadgroups of 256
constexpr u32 kLoadsPerIter = 8;

u32 bwReference(const u32* tab, u32 gid, u32 lid, u32 stride, u32 iters) {
    u32 a[8];
    for (u32 c = 0; c < 8; ++c) a[c] = gid * (c + 1) + c;
    a[0] = gid;
    const u32 base = lid * stride;
    static const u32 mul[8] = {1, 3, 5, 7, 11, 13, 17, 19};
    for (u32 k = 0; k < iters; ++k)
        for (u32 c = 0; c < 8; ++c) a[c] += tab[(base + k * 1237u + c * 97u) & (kWords - 1)];
    u32 r = 0;
    for (u32 c = 0; c < 8; ++c) r += a[c] * mul[c];
    return r;
}

struct Rig {
    Context&      ctx;
    MTL::ComputePipelineState* bw;
    MTL::ComputePipelineState* lat;
    MTL::Buffer*  params;
    MTL::Buffer*  out;
    MTL::Buffer*  src;
    u32           wrong = 0;
    std::string   wrongWhat{};

    Stats timeBw(u32 stride, u32 iters, bool check = true) {
        TgParams p{iters, stride, 0, 0};
        std::memcpy(params->contents(), &p, sizeof(p));
        const Stats s = ctx.measure([&] {
            ComputeTimer t(ctx);
            MTL4::ComputeCommandEncoder* e = t.begin();
            ctx.table()->setAddress(out->gpuAddress(), 0);
            ctx.table()->setAddress(params->gpuAddress(), 1);
            ctx.table()->setAddress(src->gpuAddress(), 2);
            e->setComputePipelineState(bw);
            e->setArgumentTable(ctx.table());
            e->dispatchThreads(MTL::Size::Make(kThreads, 1, 1), MTL::Size::Make(kTg, 1, 1));
            t.lap();
            return t.finish()[0];
        });
        if (check) {
            const u32* o = static_cast<const u32*>(out->contents());
            const u32* tab = static_cast<const u32*>(src->contents());
            for (u32 gid : {0u, 1u, 31u, 33u, 255u, 256u, 100000u, kThreads - 1})
                if (o[gid] != bwReference(tab, gid, gid % kTg, stride, iters)) {
                    ++wrong;
                    wrongWhat += "stride_" + std::to_string(stride) + " ";
                    break;
                }
        }
        return s;
    }

    Stats timeLat(u32 steps) {
        TgParams p{0, 0, steps, 0};
        std::memcpy(params->contents(), &p, sizeof(p));
        const Stats s = ctx.measure([&] {
            ComputeTimer t(ctx);
            MTL4::ComputeCommandEncoder* e = t.begin();
            ctx.table()->setAddress(out->gpuAddress(), 0);
            ctx.table()->setAddress(params->gpuAddress(), 1);
            ctx.table()->setAddress(src->gpuAddress(), 2);
            e->setComputePipelineState(lat);
            e->setArgumentTable(ctx.table());
            e->dispatchThreads(MTL::Size::Make(32, 1, 1), MTL::Size::Make(32, 1, 1));
            t.lap();
            return t.finish()[0];
        });
        const u32* tab = static_cast<const u32*>(src->contents());
        u32 idx = 0;
        for (u32 i = 0; i < steps; ++i) idx = tab[idx];
        if (static_cast<const u32*>(out->contents())[0] != idx) {
            ++wrong;
            wrongWhat += "lat ";
        }
        return s;
    }
};

Stats bwStats(const Stats& ms, double bytes) { // GB/s from time stats
    Stats t = ms;
    const double s = bytes * 1e-9 * 1e3;
    t.median = s / ms.median;
    t.min = s / ms.max;
    t.max = s / ms.min;
    t.p10 = s / ms.p90;
    t.p90 = s / ms.p10;
    t.mean = s / ms.mean;
    return t;
}

void benchThreadgroup(Context& ctx, Report& rep) {
    MTL::Library* lib = ctx.library("b05_threadgroup.metal", false);
    Rig rig{ctx, ctx.compute(lib, "tg_bw"), ctx.compute(lib, "tg_lat"), ctx.buffer(256),
            ctx.buffer(size_t(kThreads) * 4), ctx.buffer(kWords * 4)};
    // Random single-cycle permutation (Sattolo): also the table the bandwidth kernel reads.
    {
        std::vector<u32> perm(kWords);
        std::iota(perm.begin(), perm.end(), 0u);
        u64 st = 0xB05B05ull;
        for (u32 i = kWords - 1; i > 0; --i) std::swap(perm[i], perm[xorshift64(st) % i]);
        // perm as a cycle: next[perm[i]] = perm[i+1]
        u32* d = static_cast<u32*>(rig.src->contents());
        for (u32 i = 0; i < kWords; ++i) d[perm[i]] = perm[(i + 1) % kWords];
    }
    const double targetMs = ctx.quick() ? 0.15 : 0.3;
    // Calibrate on stride 1.
    const u32 probe = 64;
    const double tp = std::max(1e-3, rig.timeBw(1, probe, false).min);
    const u32 iters = std::max<u32>(32, u32(double(probe) * targetMs / tp));
    ctx.log("B-05: iters=%u", iters);

    const u32 strides[] = {0, 1, 2, 4, 8, 16, 32, 33};
    double ms1 = 0, ms32 = 0;
    double bw1 = 0;
    for (u32 stride : strides) {
        const Stats s = rig.timeBw(stride, iters);
        const double bytes = double(kThreads) * double(iters) * kLoadsPerIter * 4.0;
        const Stats b = bwStats(s, bytes);
        rep.metric("tg_bw.stride_" + std::to_string(stride), "GB/s", b,
                   {{"stride", double(stride)}, {"iters", double(iters)}, {"tg", double(kTg)}, {"ms", s.median}});
        if (stride == 1) {
            ms1 = s.min;
            bw1 = b.median;
        }
        if (stride == 32) ms32 = s.min;
        ctx.keepWarm(10);
    }
    // Fill the ratios (time relative to stride 1, minimum-time based).
    // (values set below through a second pass: cheap and keeps metric order stable)
    const Stats s1 = rig.timeBw(1, iters);
    ctx.keepWarm(10);
    for (u32 stride : {2u, 4u, 8u, 16u, 32u, 33u}) {
        const Stats s = rig.timeBw(stride, iters);
        rep.value("tg_time_ratio.stride_" + std::to_string(stride), "ratio", s.min / s1.min, {{"stride", double(stride)}},
                  false);
        ctx.keepWarm(10);
    }
    (void)bw1;

    // Latency: (t(3s) - t(s)) / 2s removes the fixed cost (fill + launch).
    const u32 steps = ctx.quick() ? 4096 : 8192;
    const Stats l1 = rig.timeLat(steps), l3 = rig.timeLat(steps * 3), l2 = rig.timeLat(steps * 2);
    const double latNs = (l3.min - l1.min) * 1e6 / double(steps * 2);
    Stats lat = l3;
    const double f = 1e6 / double(steps * 2);
    lat.median = (l3.median - l1.median) * f;
    lat.min = (l3.min - l1.max) * f;
    lat.max = (l3.max - l1.min) * f;
    lat.p10 = (l3.p10 - l1.p90) * f;
    lat.p90 = (l3.p90 - l1.p10) * f;
    lat.mean = (l3.mean - l1.mean) * f;
    rep.metric("tg_latency", "ns", lat, {{"steps", double(steps)}, {"threads", 1}}, false);
    rep.value("tg_latency.min_based", "ns", latNs, {{"steps", double(steps)}}, false);
    const double lin = (l3.median - l2.median) / std::max(1e-9, l2.median - l1.median); // both differences = `steps` steps
    // (l3-l2) and (l2-l1) each cover `steps` chases: ratio ~ 1

    // Controls
    // Controls are validity checks: repeated up to 5 times under interference (minimum times).
    double ratio32 = ms32 / ms1, linBw = 0;
    int bwTries = 0;
    for (; bwTries < 5; ++bwTries) {
        ctx.keepWarm(30);
        const double a = rig.timeBw(1, iters, false).min;
        ctx.keepWarm(30);
        const double b = rig.timeBw(1, iters * 2, false).min;
        const double c32 = rig.timeBw(32, iters, false).min;
        ratio32 = c32 / a;
        linBw = b / a;
        if (ratio32 > 0.95 && linBw > 1.6 && linBw < 2.3) break;
    }
    rig.timeBw(1, iters * 2); // results of the last variant checked against the CPU
    double latLin = lin;
    int latTries = 0;
    for (; latTries < 5 && !(latLin > 0.7 && latLin < 1.4); ++latTries) {
        ctx.keepWarm(30);
        const Stats k1 = rig.timeLat(steps), k2 = rig.timeLat(steps * 2), k3 = rig.timeLat(steps * 3);
        latLin = (k3.median - k2.median) / std::max(1e-9, k2.median - k1.median);
    }
    rep.negative(ratio32 > 0.95 && linBw > 1.6 && linBw < 2.3 && rig.wrong == 0,
                 "stride 32 / stride 1 time = " + std::to_string(ratio32).substr(0, 5) +
                     " (must not be < 0.95); stride 1 2x iterations -> " + std::to_string(linBw).substr(0, 5) +
                     "x (1.6..2.3, tries " + std::to_string(bwTries + 1) + "); " + (rig.wrong ? "results differ from CPU: " + rig.wrongWhat : "all results match the CPU"));
    rep.negative(latNs > 1.0 && latNs < 500.0 && latLin > 0.7 && latLin < 1.4,
                 "latency " + std::to_string(latNs).substr(0, 6) + " ns in (1, 500); increments of 1x/2x steps consistent: ratio " +
                     std::to_string(latLin).substr(0, 5) + " (0.7..1.4)");
    if (rig.wrong) rep.status(Status::Failed, "wrong results");
    rep.note("tg_bw: 256-thread threadgroups, each thread reads 8 uint per iteration from a 16 KiB threadgroup array at index lane*stride + uniform offset (mod 4096); bank conflicts appear as time growth with stride (32 banks x 4 B assumed, stride 33 = conflict free); GB/s of loads only, whole GPU");
    rep.note("tg_latency: one thread chases a random single-cycle permutation held in threadgroup memory; (t(3n)-t(n))/2n removes fill and launch costs; min-based value in tg_latency.min_based");
    rep.note("tg_time_ratio.stride_k = min time relative to stride 1");
}

} // namespace

SOC_BENCH("B-05", "threadgroup.memory", "Threadgroup memory: bandwidth by stride (bank conflicts) and latency", benchThreadgroup);

} // namespace soc
