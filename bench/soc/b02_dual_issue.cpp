// B-02: dual issue of FP16 / FP32 / INT32 (and packed half2) work.
// Kernels with independent chains of one or several types per thread
// (shaders/b02_dual_issue.metal); the mix is timed against the parts, as a
// function of threadgroup size and of the occupancy limit imposed by dynamic
// threadgroup memory.  Also: does FP16 reach 2x FP32 with more ILP (16
// chains) or with packed half2?
//
// Serves S-ALU-2 (dual issue) and S-ALU-3 (FP16 rate) of
// docs/APPLE_SOC_PLAYBOOK.md.

#include "harness.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>

namespace soc {
namespace {

struct MixParams {
    u32   iters, pad;
    float xf, yf;
    i32   xi, yi;
    u32   pad2, pad3;
};
static_assert(sizeof(MixParams) == 32);

constexpr u32    kThreads = 1u << 20;
constexpr float  kXf = 0.9990234375f; // exact in FP16
constexpr float  kYf = 0.0009765625f;
constexpr i32    kXi = 1664525;
constexpr i32    kYi = 1013904223;
constexpr double kMaxOpsPerCoreClk = 512.0;

struct Mix {
    int f, h, i, q; // FP32, FP16, INT32 chains, half2 chains
    std::string name() const {
        return "mix_" + std::to_string(f) + "_" + std::to_string(h) + "_" + std::to_string(i) + "_" + std::to_string(q);
    }
    double lanes() const { return double(f + h + i + 2 * q); }
};

u32 floatBits(float v) {
    u32 b;
    std::memcpy(&b, &v, 4);
    return b;
}
u32 halfBits(_Float16 v) {
    u16 b;
    std::memcpy(&b, &v, 2);
    return b;
}
i32 sd(u32 i, u32 c) { return i32((i + c * 7u) & 15u); }
_Float16 fmaH(_Float16 v, _Float16 x, _Float16 y) { return _Float16(double(v) * double(x) + double(y)); }

// Bit-exact CPU model of the kernel.
u32 reference(const Mix& m, u32 i, u32 iters) {
    const _Float16 xh = _Float16(kXf), yh = _Float16(kYf);
    std::vector<float> f(m.f);
    std::vector<_Float16> h(m.h);
    std::vector<u32> n(m.i);
    std::vector<_Float16> q(2 * m.q);
    for (int c = 0; c < m.f; ++c) f[c] = float(sd(i, c)) * 0.0625f + 1.0f;
    for (int c = 0; c < m.h; ++c) h[c] = _Float16(_Float16(sd(i + 1, c)) * _Float16(0.0625) + _Float16(1));
    for (int c = 0; c < m.i; ++c) n[c] = i * 2654435761u + u32(c) * 40503u + 1u;
    for (int c = 0; c < m.q; ++c) {
        q[2 * c]     = _Float16(_Float16(sd(i + 2, c)) * _Float16(0.0625) + _Float16(1));
        q[2 * c + 1] = _Float16(_Float16(sd(i + 5, c)) * _Float16(0.0625) + _Float16(1));
    }
    for (u32 k = 0; k < iters; ++k) {
        for (auto& v : f) v = std::fma(v, kXf, kYf);
        for (auto& v : h) v = fmaH(v, xh, yh);
        for (auto& v : n) v = v * u32(kXi) + u32(kYi);
        for (auto& v : q) v = fmaH(v, xh, yh);
    }
    u32 hs = 0;
    for (float v : f) hs = hs * 31u + floatBits(v);
    for (auto v : h) hs = hs * 31u + halfBits(v);
    for (u32 v : n) hs = hs * 31u + v;
    for (int c = 0; c < m.q; ++c) hs = hs * 31u + halfBits(q[2 * c]) + 17u * halfBits(q[2 * c + 1]);
    return hs;
}

struct Rig {
    Context&      ctx;
    MTL::Library* lib;
    MTL::Buffer*  params;
    MTL::Buffer*  out;
    std::map<std::string, MTL::ComputePipelineState*> psos;
    u32         wrong = 0;
    std::string wrongWhat;

    MTL::ComputePipelineState* pso(const Mix& m) {
        auto it = psos.find(m.name());
        if (it != psos.end()) return it->second;
        return psos[m.name()] = ctx.compute(lib, m.name());
    }

    double timeOnce(const Mix& m, u32 iters, u32 tg, u32 tgMemBytes = 0) {
        MixParams p{iters, 0, kXf, kYf, kXi, kYi, 0, 0};
        std::memcpy(params->contents(), &p, sizeof(p));
        ComputeTimer t(ctx);
        MTL4::ComputeCommandEncoder* e = t.begin();
        ctx.table()->setAddress(out->gpuAddress(), 0);
        ctx.table()->setAddress(params->gpuAddress(), 1);
        e->setComputePipelineState(pso(m));
        e->setArgumentTable(ctx.table());
        e->setThreadgroupMemoryLength(std::max<u32>(16, tgMemBytes), 0);
        e->dispatchThreads(MTL::Size::Make(kThreads, 1, 1), MTL::Size::Make(tg, 1, 1));
        t.lap();
        return t.finish()[0];
    }

    /// Stats in ms; checks the stored results of the last repetition.
    Stats time(const Mix& m, u32 iters, u32 tg, u32 tgMemBytes = 0) {
        const Stats s = ctx.measure([&] { return timeOnce(m, iters, tg, tgMemBytes); });
        const u32* o = static_cast<const u32*>(out->contents());
        u32 bad = 0;
        for (u32 i : {0u, 1u, 7u, 16u, 1000u, 65535u, kThreads - 1})
            if (o[i] != reference(m, i, iters)) ++bad;
        if (bad) {
            wrong += bad;
            wrongWhat += m.name() + "@tg" + std::to_string(tg) + " ";
        }
        return s;
    }
};

Stats opsPerSec(const Stats& ms, double ops) { // ops per second (Top/s scaled by 1e-12), from time stats
    Stats t = ms;
    const double s = ops * 1e-12 * 1e3; // ms -> s
    t.median = s / ms.median;
    t.min = s / ms.max;
    t.max = s / ms.min;
    t.p10 = s / ms.p90;
    t.p90 = s / ms.p10;
    t.mean = s / ms.mean;
    return t;
}

void benchDual(Context& ctx, Report& rep) {
    MTL::Library* lib = ctx.library("b02_dual_issue.metal", /*fastMath=*/false);
    Rig rig{ctx, lib, ctx.buffer(256), ctx.buffer(size_t(kThreads) * 4), {}, 0, ""};
    const u32 cores = std::max<u32>(1, describeMachine(ctx).gpuCores);
    const auto& mhz = ctx.gpuState().pstateMHz();
    const double topMHz = mhz.empty() ? 0 : mhz.back();
    const double perCoreClk = topMHz > 0 ? double(cores) * topMHz * 1e6 : 0;
    const double targetMs = ctx.quick() ? 0.15 : 0.2;
    bool plausible = true;
    std::string implausible;

    const Mix F8{8, 0, 0, 0}, H8{0, 8, 0, 0}, I8{0, 0, 8, 0}, Q8{0, 0, 0, 8};
    // Calibrate on FP32 (8 chains, TG 256): the same iteration count for every kernel.
    const u32 probe = 256;
    const double tp = std::max(1e-3, ctx.measure([&] { return rig.timeOnce(F8, probe, 256); }, 3).median);
    const u32 iters = std::max<u32>(16, u32(double(probe) * targetMs / tp));
    ctx.log("B-02: iters=%u (%.3f ms probe)", iters, tp);

    auto opsOf = [&](const Mix& m) { return double(kThreads) * double(iters) * m.lanes(); };
    auto perClk = [&](const Mix& m, double ms) { return perCoreClk > 0 ? opsOf(m) / (ms * 1e-3) / perCoreClk : 0.0; };
    auto checkPlausible = [&](const Mix& m, double ms, const std::string& what) {
        if (perCoreClk > 0 && perClk(m, ms) > kMaxOpsPerCoreClk) {
            plausible = false;
            implausible += what + " ";
        }
    };

    // --- 1. ILP sweep: does FP16 reach 2x FP32? -------------------------------
    std::map<std::string, double> thr; // "f32.8" -> op/s
    for (int n : {4, 8, 16, 32}) {
        const Mix mf{n, 0, 0, 0}, mh{0, n, 0, 0};
        const std::string sn = std::to_string(n);
        const Stats sf = rig.time(mf, iters, 256), sh = rig.time(mh, iters, 256);
        ctx.keepWarm(20);
        const double of = opsOf(mf) / (sf.median * 1e-3), oh = opsOf(mh) / (sh.median * 1e-3);
        thr["f32." + sn] = of;
        thr["f16." + sn] = oh;
        rep.metric("f32.fma.chains_" + sn, "Top/s", opsPerSec(sf, opsOf(mf)), {{"chains", double(n)}});
        rep.metric("f16.fma.chains_" + sn, "Top/s", opsPerSec(sh, opsOf(mh)), {{"chains", double(n)}});
        rep.value("f32.fma.chains_" + sn + ".per_core_clk", "op/core/clk", perClk(mf, sf.median), {{"chains", double(n)}});
        rep.value("f16.fma.chains_" + sn + ".per_core_clk", "op/core/clk", perClk(mh, sh.median), {{"chains", double(n)}});
        rep.value("f16_vs_f32.chains_" + sn + ".ratio", "ratio", oh / of, {{"chains", double(n)}});
        checkPlausible(mf, sf.median, "f32x" + sn);
        checkPlausible(mh, sh.median, "f16x" + sn);
    }
    for (int n : {4, 8, 16}) {
        const Mix mq{0, 0, 0, n};
        const std::string sn = std::to_string(n);
        const Stats sq = rig.time(mq, iters, 256);
        ctx.keepWarm(20);
        const double oq = opsOf(mq) / (sq.median * 1e-3); // lanes/s
        rep.metric("f16x2.fma.chains_" + sn, "Top/s", opsPerSec(sq, opsOf(mq)), {{"chains", double(n)}, {"lanes", 2}});
        rep.value("f16x2.fma.chains_" + sn + ".per_core_clk", "op/core/clk", perClk(mq, sq.median), {{"chains", double(n)}});
        rep.value("f16x2_vs_f32.chains_" + sn + ".ratio", "ratio", oq / thr["f32.8"], {{"chains", double(n)}});
        checkPlausible(mq, sq.median, "f16x2x" + sn);
    }
    {
        const Stats si = rig.time(I8, iters, 256);
        rep.metric("i32.mad.chains_8", "Top/s", opsPerSec(si, opsOf(I8)), {{"chains", 8}});
        rep.value("i32.mad.chains_8.per_core_clk", "op/core/clk", perClk(I8, si.median), {{"chains", 8}});
        checkPlausible(I8, si.median, "i32x8");
    }
    ctx.keepWarm(20);

    // --- 2. Mixes vs sum of the parts, by threadgroup size -----------------------
    struct MixCase {
        std::string      label;
        Mix              mix;
        std::vector<Mix> parts;
    };
    const std::vector<MixCase> cases = {
        {"f16_f32", {8, 8, 0, 0}, {F8, H8}},
        {"f32_i32", {8, 0, 8, 0}, {F8, I8}},
        {"f16_i32", {0, 8, 8, 0}, {H8, I8}},
        {"f16_f32_i32", {8, 8, 8, 0}, {F8, H8, I8}},
        {"f16x2_f32", {8, 0, 0, 8}, {F8, Q8}},
    };
    const std::vector<u32> tgs = ctx.quick() ? std::vector<u32>{32, 128, 256}
                                             : std::vector<u32>{32, 64, 128, 256, 512, 1024};
    std::map<std::string, double> headline;
    double minRatio = 1e9;
    for (u32 tg : tgs) {
        std::map<std::string, double> partMs;
        for (const auto& c : cases)
            for (const Mix& part : c.parts)
                if (!partMs.count(part.name())) partMs[part.name()] = rig.time(part, iters, tg).median;
        for (const auto& c : cases) {
            const Stats sm = rig.time(c.mix, iters, tg);
            double sum = 0;
            for (const Mix& part : c.parts) sum += partMs[part.name()];
            const double ratio = sm.median / sum;
            minRatio = std::min(minRatio, ratio);
            rep.value("mix." + c.label + ".tg_" + std::to_string(tg) + ".ratio", "ratio", ratio,
                      {{"tg", double(tg)}, {"simdgroups", double(tg / 32)}, {"ms_mix", sm.median}, {"ms_parts", sum}},
                      /*higherIsBetter=*/false);
            checkPlausible(c.mix, sm.median, c.label + "@tg" + std::to_string(tg));
            if (tg == 256) headline[c.label] = ratio;
        }
        ctx.keepWarm(20);
    }
    for (const auto& c : cases)
        rep.value("mix." + c.label + ".ratio", "ratio", headline.count(c.label) ? headline[c.label] : 0, {{"tg", 256}},
                  /*higherIsBetter=*/false);

    // --- 3. Occupancy limited by dynamic threadgroup memory (TG = 128) -----------
    const u32 maxMem = u32(ctx.device()->maxThreadgroupMemoryLength());
    std::vector<u32> tgmems = {0, 4096, 8192, 16384, 24576, 32768};
    tgmems.erase(std::remove_if(tgmems.begin(), tgmems.end(), [&](u32 v) { return v > maxMem; }), tgmems.end());
    if (ctx.quick()) tgmems = {0, 8192, 24576};
    for (u32 mem : tgmems) {
        if (mem > maxMem) continue;
        const double tf = rig.time(F8, iters, 128, mem).median, th = rig.time(H8, iters, 128, mem).median;
        const Stats sm = rig.time(cases[0].mix, iters, 128, mem);
        rep.value("mix.f16_f32.tgmem_" + std::to_string(mem / 1024) + "KiB.ratio", "ratio", sm.median / (tf + th),
                  {{"tgmem_bytes", double(mem)}, {"tg", 128}, {"ms_mix", sm.median}, {"ms_f32", tf}, {"ms_f16", th}},
                  false);
        rep.value("f32.fma.tgmem_" + std::to_string(mem / 1024) + "KiB.ms", "ms", tf, {{"tgmem_bytes", double(mem)}},
                  false);
    }
    ctx.keepWarm(20);

    // --- Controls -----------------------------------------------------------------
    // (a) linearity of the triple mix: 2x iterations -> 2x time, and exact results.
    const Mix triple{8, 8, 8, 0};
    // Controls are validity checks of the measurement: under interference from other GPU clients
    // (clock changes between the two timings) they are repeated up to 5 times (minimum times).
    double lin = 0;
    int linTries = 0;
    for (; linTries < 5; ++linTries) {
        ctx.keepWarm(30);
        const double t1 = rig.time(triple, iters, 256).min;
        ctx.keepWarm(30);
        const double t2 = rig.time(triple, iters * 2, 256).min;
        lin = t2 / t1;
        if (lin > 1.85 && lin < 2.15) break;
    }
    rep.negative(lin > 1.85 && lin < 2.15 && rig.wrong == 0,
                 "f16+f32+i32 mix 2x iterations -> " + std::to_string(lin).substr(0, 5) + "x time (want 1.85..2.15, tries " + std::to_string(linTries + 1) + "); " +
                     (rig.wrong ? "results differ from the CPU model: " + rig.wrongWhat
                                : "all results match the CPU model"));
    // (b) same-pipe control: 32 FP32 chains take ~2x the time of 16 (both saturated) (a bigger kernel is no free
    // lunch), plus a plausibility bound on every kernel.
    double same = 0;
    int sameTries = 0;
    for (; sameTries < 5; ++sameTries) {
        ctx.keepWarm(30);
        const double t16 = rig.time(Mix{16, 0, 0, 0}, iters, 256).min, t32 = rig.time(Mix{32, 0, 0, 0}, iters, 256).min;
        same = t32 / t16;
        if (same > 1.3 && same < 2.4) break;
    }
    rep.negative(same > 1.3 && same < 2.4 && plausible,
                 "f32x32 / f32x16 time = " + std::to_string(same).substr(0, 5) + " (want 1.3..2.4: doubling the work cannot be free nor more than 2.4x; the lower bound is loose because shader validation leaves 16 chains unsaturated, tries " + std::to_string(sameTries + 1) + "); " +
                     (plausible ? "every kernel <= 512 op/core/clk" : "IMPLAUSIBLE: " + implausible));
    rep.value("control.f32_32_vs_16.ratio", "ratio", same, {}, false);
    rep.value("mix.min_ratio", "ratio", minRatio, {}, false);
    if (rig.wrong) rep.status(Status::Failed, "wrong results");
    rep.note("ratio = time(mix) / sum(time(parts)), same iterations and threads: 1 = no overlap, 0.5 = perfect overlap of two parts, <1 = concurrent issue");
    rep.note("mixes use 8 independent chains per type per thread; headline ratios at 256 threads/threadgroup; f16x2 = packed half2 FMA (2 lanes per op)");
    rep.note("occupancy knob: dynamic threadgroup memory per 128-thread threadgroup limits resident threadgroups; library in MathModeSafe");
    rep.note("per_core_clk assumes the top P-state (" + std::to_string(int(topMHz)) + " MHz): see gpu.top_state_share");
}

} // namespace

SOC_BENCH("B-02", "alu.dual_issue", "FP16/FP32/INT32 mixes vs parts by SIMD-groups per threadgroup; FP16 vs FP32 ILP",
          benchDual);

} // namespace soc
