// B-06: SIMD-group operations (simd_shuffle, quad_shuffle, simd_sum,
// simd_prefix_exclusive_sum, simd_ballot) vs the same semantics implemented
// through threadgroup memory, and divergence cost (branch taken by 0/25/50/100%
// of the lanes, if/else and if-only, expensive bodies).
//
// Serves S-SIMD-1..2 of docs/APPLE_SOC_PLAYBOOK.md.

#include "harness.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

namespace soc {
namespace {

struct SimdParams {
    u32   iters, rounds, taken, pad;
    float xa, ya, xb, yb;
};
static_assert(sizeof(SimdParams) == 32);

constexpr u32 kThreads = 1u << 20;
constexpr u32 kTg = 256;
constexpr u32 kK = 0x9E3779B9u;
constexpr float kXa = 0.999f, kYa = 0.001f, kXb = 0.9985f, kYb = 0.0015f;
constexpr u32 kRounds = 8;

enum Op { Shuf, Quad, Sum, Prefix, Ballot };
const char* const kOpName[] = {"shuffle", "quad_shuffle", "sum", "prefix", "ballot"};

using Lanes = std::array<u32, 32>;

u32 stepLane(Op op, const Lanes& v, u32 lane, int c) {
    switch (op) {
    case Shuf: return v[(lane + 1u + u32(c & 1)) & 31u] + kK;
    case Quad: return v[lane ^ u32((c & 1) + 1)] + kK;
    case Sum: {
        u32 s = 0;
        for (u32 x : v) s += x;
        return s + lane;
    }
    case Prefix: {
        u32 s = 0;
        for (u32 l = 0; l < lane; ++l) s += v[l];
        return s + kK;
    }
    default: {
        u32 b = 0;
        for (u32 l = 0; l < 32; ++l) b |= ((v[l] & 1u) ? 1u : 0u) << l;
        return b ^ (v[lane] + lane);
    }
    }
}

/// Result of every lane of the SIMD-group starting at thread `first` (multiple of 32).
Lanes simdReference(Op op, u32 first, u32 iters) {
    Lanes v[4];
    for (u32 l = 0; l < 32; ++l) {
        const u32 g = first + l;
        v[0][l] = g + 1u;
        v[1][l] = g * 3u + 7919u;
        v[2][l] = g * 5u + 15838u;
        v[3][l] = g * 7u + 23757u;
    }
    for (u32 k = 0; k < iters; ++k)
        for (int c = 0; c < 4; ++c) {
            Lanes n;
            for (u32 l = 0; l < 32; ++l) n[l] = stepLane(op, v[c], l, c);
            v[c] = n;
        }
    Lanes r;
    for (u32 l = 0; l < 32; ++l) r[l] = v[0][l] + v[1][l] * 3u + v[2][l] * 5u + v[3][l] * 7u;
    return r;
}

u32 bits(float f) {
    u32 b;
    std::memcpy(&b, &f, 4);
    return b;
}
u32 divReference(bool elseBranch, u32 gid, u32 taken, u32 iters) {
    const u32 lane = gid & 31u;
    float a = float(gid & 15u) * 0.0625f + 1.0f, b = a + 0.25f, c = a + 0.5f, d = a + 0.75f;
    const bool take = ((lane * 13u) & 31u) < taken;
    for (u32 k = 0; k < iters; ++k) {
        if (take) {
            for (u32 r = 0; r < kRounds; ++r) {
                a = std::fma(a, kXa, kYa); b = std::fma(b, kXa, kYa);
                c = std::fma(c, kXa, kYa); d = std::fma(d, kXa, kYa);
            }
        } else if (elseBranch) {
            for (u32 r = 0; r < kRounds; ++r) {
                a = std::fma(a, kXb, kYb); b = std::fma(b, kXb, kYb);
                c = std::fma(c, kXb, kYb); d = std::fma(d, kXb, kYb);
            }
        }
    }
    return bits(a) + bits(b) * 3u + bits(c) * 5u + bits(d) * 7u;
}

struct Rig {
    Context&      ctx;
    MTL::Library* lib;
    MTL::Buffer*  params;
    MTL::Buffer*  out;
    u32           wrong = 0;
    std::string   wrongWhat{};

    Stats time(const std::string& fn, const SimdParams& p) {
        MTL::ComputePipelineState* pso = ctx.compute(lib, fn);
        std::memcpy(params->contents(), &p, sizeof(p));
        return ctx.measure([&] {
            ComputeTimer t(ctx);
            MTL4::ComputeCommandEncoder* e = t.begin();
            ctx.table()->setAddress(out->gpuAddress(), 0);
            ctx.table()->setAddress(params->gpuAddress(), 1);
            e->setComputePipelineState(pso);
            e->setArgumentTable(ctx.table());
            e->dispatchThreads(MTL::Size::Make(kThreads, 1, 1), MTL::Size::Make(kTg, 1, 1));
            t.lap();
            return t.finish()[0];
        });
    }
    void checkSimd(Op op, const std::string& what, u32 iters) {
        const u32* o = static_cast<const u32*>(out->contents());
        for (u32 first : {0u, 32u, 4096u + 64u, kThreads - 32}) {
            const Lanes ref = simdReference(op, first, iters);
            for (u32 l = 0; l < 32; ++l)
                if (o[first + l] != ref[l]) {
                    ++wrong;
                    wrongWhat += what + " ";
                    return;
                }
        }
    }
    void checkDiv(bool elseBranch, const std::string& what, u32 taken, u32 iters) {
        const u32* o = static_cast<const u32*>(out->contents());
        for (u32 g : {0u, 1u, 7u, 31u, 32u, 1000u, kThreads - 1})
            if (o[g] != divReference(elseBranch, g, taken, iters)) {
                ++wrong;
                wrongWhat += what + " ";
                return;
            }
    }
    u32 calibrate(const std::string& fn, SimdParams p, double targetMs) {
        p.iters = 16;
        const double tp = std::max(1e-3, time(fn, p).min);
        return std::max<u32>(4, u32(16.0 * targetMs / tp));
    }
};

Stats gops(const Stats& ms, double ops) { // Gop/s from time stats
    Stats t = ms;
    const double s = ops * 1e-9 * 1e3;
    t.median = s / ms.median;
    t.min = s / ms.max;
    t.max = s / ms.min;
    t.p10 = s / ms.p90;
    t.p90 = s / ms.p10;
    t.mean = s / ms.mean;
    return t;
}

void benchSimd(Context& ctx, Report& rep) {
    Rig rig{ctx, ctx.library("b06_simd.metal", false), ctx.buffer(256), ctx.buffer(size_t(kThreads) * 4)};
    const double targetMs = ctx.quick() ? 0.15 : 0.3;
    const u32 cores = std::max<u32>(1, describeMachine(ctx).gpuCores);
    const auto& mhz = ctx.gpuState().pstateMHz();
    const double topMHz = mhz.empty() ? 0 : mhz.back();
    SimdParams base{0, kRounds, 0, 0, kXa, kYa, kXb, kYb};

    bool sgFaster = true; // control candidate: report only
    double linSg = 0;
    u32 shufIters = 0;
    std::string slowWhat;
    for (int o = 0; o < 5; ++o) {
        const Op op = Op(o);
        const std::string name = kOpName[o];
        SimdParams p = base;
        p.iters = rig.calibrate("sg_" + name, p, targetMs);
        const u32 iters = p.iters;
        const Stats ssg = rig.time("sg_" + name, p);
        rig.checkSimd(op, "sg_" + name, iters);
        ctx.keepWarm(10);
        // The tg version is several times slower: same iteration count (identical semantics), fewer if too slow.
        const Stats stg = rig.time("tg_" + name, p);
        rig.checkSimd(op, "tg_" + name, iters);
        ctx.keepWarm(10);
        // SIMD-group instructions: threads/32 * iters * 4 chains.
        const double sgops = double(kThreads) / 32.0 * double(iters) * 4.0;
        rep.metric("simd." + name, "Gop/s", gops(ssg, sgops), {{"iters", double(iters)}, {"chains", 4}, {"ms", ssg.median}});
        rep.metric("tg." + name, "Gop/s", gops(stg, sgops), {{"iters", double(iters)}, {"chains", 4}, {"ms", stg.median}});
        rep.value("speedup." + name, "ratio", stg.min / ssg.min, {{"ms_simd", ssg.min}, {"ms_tg", stg.min}});
        if (topMHz > 0)
            rep.value("simd." + name + ".per_core_clk", "op/core/clk",
                      sgops / (ssg.min * 1e-3) / (double(cores) * topMHz * 1e6), {{"mhz_assumed", topMHz}});
        if (stg.min < ssg.min * 0.5) {
            sgFaster = false;
            slowWhat += name + " ";
        }
        if (name == "shuffle") shufIters = iters;
    }

    // --- Divergence -----------------------------------------------------------------
    const u32 takens[] = {0, 8, 16, 32}; // 0, 25, 50, 100 %
    SimdParams dp = base;
    dp.taken = 32;
    const u32 dIters = rig.calibrate("dv_ifelse", dp, targetMs);
    double tElse[4] = {}, tOnly[4] = {};
    int divTries = 0;
    // The divergence controls are validity checks of the measurement: the whole set is repeated up to
    // 3 times when other GPU clients disturbed it (minimum times).
    for (; divTries < 5; ++divTries) {
        for (int mode = 0; mode < 2; ++mode) {
            const bool el = mode == 0;
            for (int i = 0; i < 4; ++i) {
                SimdParams p = base;
                p.iters = dIters;
                p.taken = takens[i];
                ctx.keepWarm(20);
                const Stats s = rig.time(el ? "dv_ifelse" : "dv_ifonly", p);
                rig.checkDiv(el, std::string(el ? "ifelse" : "ifonly") + "_" + std::to_string(takens[i]), takens[i], dIters);
                (el ? tElse : tOnly)[i] = s.min;
            }
        }
        const double a = tElse[2] / tElse[0], b = tElse[3] / tElse[0], c = tOnly[1] / tOnly[3], d = tOnly[0] / tOnly[3];
        if (a > 1.3 && b > 0.7 && b < 1.4 && c > 0.8 && d < 0.5) break;
    }
    for (int i = 0; i < 4; ++i) {
        const std::string pct = std::to_string(takens[i] * 100 / 32);
        rep.value("div.ifelse.pct_" + pct + ".ratio", "ratio", tElse[i] / tElse[0],
                  {{"taken_pct", double(takens[i] * 100 / 32)}, {"ms", tElse[i]}}, false); // vs 0% taken
        rep.value("div.ifonly.pct_" + pct + ".ratio", "ratio", tOnly[i] / tOnly[3],
                  {{"taken_pct", double(takens[i] * 100 / 32)}, {"ms", tOnly[i]}}, false); // vs 100% taken
    }
    rep.value("div.ifelse.ms_per_body", "ms", tElse[3], {{"iters", double(dIters)}}, false);

    // --- Controls -----------------------------------------------------------------------
    int linTries = 0;
    for (; linTries < 5; ++linTries) {
        SimdParams p1 = base, p2 = base;
        p1.iters = shufIters;
        p2.iters = shufIters * 2;
        ctx.keepWarm(30);
        const Stats a = rig.time("sg_shuffle", p1);
        rig.checkSimd(Shuf, "sg_shuffle", shufIters);
        ctx.keepWarm(30);
        const Stats b = rig.time("sg_shuffle", p2);
        rig.checkSimd(Shuf, "sg_shuffle x2", shufIters * 2);
        linSg = b.min / a.min;
        if (linSg > 1.85 && linSg < 2.15) break;
    }
    const double r50 = tElse[2] / tElse[0], r100 = tElse[3] / tElse[0], o25 = tOnly[1] / tOnly[3], o0 = tOnly[0] / tOnly[3];
    rep.negative(rig.wrong == 0 && linSg > 1.85 && linSg < 2.15,
                 "sg_shuffle 2x iterations -> " + std::to_string(linSg).substr(0, 5) + "x (1.85..2.15, tries " + std::to_string(linTries + 1) + "); " +
                     (rig.wrong ? "results differ from the CPU model: " + rig.wrongWhat : "all sg_/tg_/dv_ results match the CPU model"));
    rep.negative(r50 > 1.3 && r100 > 0.7 && r100 < 1.4 && o25 > 0.8 && o0 < 0.5,
                 "if/else 50% taken = " + std::to_string(r50).substr(0, 5) + "x of 0% (both bodies run, want > 1.3); 100% = " +
                     std::to_string(r100).substr(0, 5) + "x (0.7..1.4); if-only 25% = " + std::to_string(o25).substr(0, 5) +
                     "x of 100% (want > 0.8), 0% = " + std::to_string(o0).substr(0, 5) + "x (want < 0.5), tries " + std::to_string(divTries + 1));
    if (!sgFaster) rep.note("threadgroup-memory emulation was more than 2x FASTER than the intrinsic for: " + slowWhat);
    if (rig.wrong) rep.status(Status::Failed, "wrong results");
    rep.note("simd.<op> / tg.<op> = SIMD-group instructions per second (threads/32 x iters x 4 chains), each link = the op + one cheap ALU op (glue); speedup.<op> = time(tg) / time(simd) (>1: intrinsic faster)");
    rep.note("tg emulation uses simdgroup_barrier(mem_threadgroup): shuffle = write+read, sum = 5-step tree, prefix = Hillis-Steele scan, ballot = tree sum of pred<<lane; quad_shuffle = xor 1/2 neighbour");
    rep.note("divergence: lanes take the branch when ((lane*13)&31) < n; bodies = 8 rounds of 4 FMA chains; ifelse ratios vs 0% taken, ifonly ratios vs 100% taken; minimum-time based");
}

} // namespace

SOC_BENCH("B-06", "simd.ops", "SIMD-group ops vs threadgroup-memory equivalents; divergence cost", benchSimd);

} // namespace soc
