// B-04: live registers vs throughput; the thrashing point of the dynamic
// register file / occupancy (OPT-0.6 threshold).
//
// Kernels keep N FP32 values live across a loop with a fixed number of
// dependent gather loads per thread and ~constant FMAs per thread
// (shaders/b04_occupancy.metal).  If the N live values no longer fit, the
// hardware lowers occupancy (the load latency is no longer hidden) or the
// compiler spills: time per FMA jumps.
//
// Detection rule (documented in the notes): reference throughput = median of
// the loads-kernel throughput for N in {16, 32, 48, 64}; the thrashing point
// is the smallest N whose throughput falls below 80% of the reference
// (minimum-time based).  The same rule on the no-load kernel separates
// compiler spills from occupancy loss.
//
// Serves S-OCC-1..3 of docs/APPLE_SOC_PLAYBOOK.md.

#include "harness.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>

namespace soc {
namespace {

struct RegParams {
    u32   loads, rounds;
    float x, y;
    u32   mask, pad;
};
static_assert(sizeof(RegParams) == 24);

constexpr float kX = 0.999f, kY = 0.001f;
constexpr u32   kLoads = 16;
constexpr u32   kTabBytes = 64u << 20; // 64 MiB: beyond L2, SLC-resident
constexpr u32   kWorkPerLoad = 1024;   // FMAs per thread per load, ~constant across N
constexpr double kThreshold = 0.8;

u32 rounds(u32 n) { return std::max<u32>(1, u32(std::lround(double(kWorkPerLoad) / n))); }

// Bit-exact model of the kernel (LD, ALU or DYN order of operations is identical).
u32 reference(int mode, u32 n, u32 i, const u32* tab, u32 mask) {
    std::vector<float> v(n);
    for (u32 c = 0; c < n; ++c) v[c] = float(i32((i + c * 7u) & 15u)) * 0.0625f + 1.0f;
    const u32 r = rounds(n);
    for (u32 l = 0; l < kLoads; ++l) {
        if (mode == 0) {
            u32 b;
            std::memcpy(&b, &v[0], 4);
            const u32 idx = (b * 2654435761u + i * 40503u + l) & mask;
            v[0] = std::fma(float(tab[idx] & 15u), 0.001f, v[0]);
        }
        for (u32 m = 0; m < r; ++m) {
            if (mode == 2) {
                for (u32 c = 0; c < n; ++c) {
                    const u32 k = (c + m) & (n - 1);
                    v[k] = std::fma(v[k], kX, kY);
                }
            } else {
                for (u32 c = 0; c < n; ++c) v[c] = std::fma(v[c], kX, kY);
            }
        }
    }
    float s = 0.0f;
    for (u32 c = 0; c < n; ++c) s += v[c];
    u32 b;
    std::memcpy(&b, &s, 4);
    return b;
}

struct Rig {
    Context&      ctx;
    MTL::Library* lib;
    MTL::Buffer*  params;
    MTL::Buffer*  out;
    MTL::Buffer*  tab;
    u32           threads;
    u32           wrong = 0;
    std::string   wrongWhat{};

    /// Returns stats (ms); tgUsed set to the threadgroup size actually used.
    Stats time(const std::string& fn, u32 n, int mode, u32 loadsMul, u32& tgUsed, u32& maxTotal, u32 loads = kLoads,
               u32 roundsAbs = 0, u32 thr = 0) {
        const u32 nthreads = thr ? thr : threads;
        MTL::ComputePipelineState* pso = ctx.compute(lib, fn);
        maxTotal = u32(pso->maxTotalThreadsPerThreadgroup());
        tgUsed = std::min<u32>(256, maxTotal);
        tgUsed = std::max<u32>(32, tgUsed / 32 * 32);
        RegParams p{loads, roundsAbs ? roundsAbs : rounds(n) * loadsMul, kX, kY, kTabBytes / 4 - 1, 0};
        std::memcpy(params->contents(), &p, sizeof(p));
        const Stats s = ctx.measure([&] {
            ComputeTimer t(ctx);
            MTL4::ComputeCommandEncoder* e = t.begin();
            ctx.table()->setAddress(out->gpuAddress(), 0);
            ctx.table()->setAddress(params->gpuAddress(), 1);
            ctx.table()->setAddress(tab->gpuAddress(), 2);
            e->setComputePipelineState(pso);
            e->setArgumentTable(ctx.table());
            e->dispatchThreads(MTL::Size::Make(nthreads, 1, 1), MTL::Size::Make(tgUsed, 1, 1));
            t.lap();
            return t.finish()[0];
        });
        if (loadsMul == 1 && loads == kLoads && !roundsAbs && !thr) {
            const u32* o = static_cast<const u32*>(out->contents());
            const u32* tb = static_cast<const u32*>(tab->contents());
            for (u32 i : {0u, 3u, 17u, 1000u, threads - 1})
                if (o[i] != reference(mode, n, i, tb, kTabBytes / 4 - 1)) {
                    ++wrong;
                    wrongWhat += fn + " ";
                    break;
                }
        }
        return s;
    }
};

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0 : v[v.size() / 2];
}

/// Throughput (Top/s of FMAs) stats from time stats; the value (median field) is minimum-time based.
Stats fmaThroughput(const Stats& ms, double fmas) {
    Stats t = ms;
    const double s = fmas * 1e-12 * 1e3;
    t.median = s / ms.min;
    t.min = s / ms.max;
    t.max = s / ms.min;
    t.p10 = s / ms.p90;
    t.p90 = s / ms.p10;
    t.mean = s / ms.mean;
    return t;
}

void benchOccupancy(Context& ctx, Report& rep) {
    const u32 threads = ctx.quick() ? (1u << 17) : (1u << 18);
    Rig rig{ctx,
            ctx.library("b04_occupancy.metal", /*fastMath=*/false),
            ctx.buffer(256),
            ctx.buffer(size_t(threads) * 4),
            ctx.randomBuffer(kTabBytes, 0xB04B04B04ull),
            threads};
    const u32 ns[] = {8, 16, 32, 48, 64, 96, 104, 112, 120, 128, 160, 192, 256};
    std::map<u32, double> thrLd, thrAlu;
    std::map<u32, double> msLd, msAlu;
    for (int pass = 0; pass < 2; ++pass) {
        const int mode = pass; // 0 LD, 1 ALU
        for (u32 n : ns) {
            u32 tg = 0, maxTotal = 0;
            const std::string fn = std::string("regs_") + (mode == 0 ? "ld_" : "alu_") + std::to_string(n);
            const Stats s = rig.time(fn, n, mode, 1, tg, maxTotal);
            const double fmas = double(threads) * double(kLoads) * double(rounds(n)) * double(n);
            const Stats thr = fmaThroughput(s, fmas);
            const std::string name = std::string(mode == 0 ? "regs_" : "regs_alu_") + std::to_string(n);
            rep.metric(name, "Top/s", thr,
                       {{"live_regs", double(n)}, {"threads", double(threads)}, {"rounds", double(rounds(n))},
                        {"loads", double(mode == 0 ? kLoads : 0)}, {"ms_min", s.min}, {"tg", double(tg)}});
            (mode == 0 ? thrLd : thrAlu)[n] = thr.median;
            (mode == 0 ? msLd : msAlu)[n] = s.min;
            if (mode == 0)
                rep.value("max_threads_tg.regs_" + std::to_string(n), "count", double(maxTotal),
                          {{"live_regs", double(n)}}, true);
            ctx.keepWarm(10);
        }
    }
    // Thrashing points.
    auto detect = [&](const std::map<u32, double>& thr, double& ref) -> u32 {
        ref = median({thr.at(16), thr.at(32), thr.at(48), thr.at(64), thr.at(96)});
        for (u32 n : ns)
            if (n > 96 && thr.at(n) < kThreshold * ref) return n;
        for (u32 n : {48u, 64u})
            if (thr.at(n) < kThreshold * ref) return n;
        return 0;
    };
    double refLd = 0, refAlu = 0;
    const u32 tpLd = detect(thrLd, refLd), tpAlu = detect(thrAlu, refAlu);
    rep.value("thrashing_point.live_regs", "count", double(tpLd),
              {{"found", tpLd ? 1.0 : 0.0}, {"reference_top_s", refLd}, {"threshold", kThreshold}}, false);
    rep.value("thrashing_point_alu.live_regs", "count", double(tpAlu),
              {{"found", tpAlu ? 1.0 : 0.0}, {"reference_top_s", refAlu}, {"threshold", kThreshold}}, false);
    rep.value("regs_worst_over_best.ratio", "ratio",
              [&] {
                  double lo = 1e30, hi = 0;
                  for (u32 n : ns) {
                      lo = std::min(lo, thrLd[n]);
                      hi = std::max(hi, thrLd[n]);
                  }
                  return lo / hi;
              }(),
              {});
    ctx.log("B-04: thrashing point (loads) N=%u, (alu) N=%u; Top/s ld: 8:%.2f 64:%.2f 128:%.2f 256:%.2f", tpLd, tpAlu,
            thrLd[8], thrLd[64], thrLd[128], thrLd[256]);

    // --- Occupancy proxy: loads/s of a latency-bound kernel (1 round per load, 64 loads) ----------
    // Little's law: loads/s = resident threads / load latency while the ALU is not the limit.
    // Latency: one SIMD-group alone (32 threads), 64 dependent loads.
    {
        u32 tg2 = 0, mt2 = 0;
        const Stats l1 = rig.time("regs_ld_8", 8, 0, 1, tg2, mt2, 64, 1, 32);
        const double latNs = l1.min * 1e6 / 64.0;
        rep.value("lat_probe.load_latency_ns", "ns", latNs, {{"threads", 32}, {"loads", 64}}, false);
        const u32 cores = std::max<u32>(1, describeMachine(ctx).gpuCores);
        for (u32 n : {8u, 32u, 64u, 96u, 112u, 128u, 160u, 256u}) {
            const Stats s = rig.time("regs_ld_" + std::to_string(n), n, 0, 1, tg2, mt2, 64, 1);
            const double loadsPerSec = double(threads) * 64.0 / (s.min * 1e-3);
            const double fmaRate = loadsPerSec * n;                 // FMA/s implied
            const bool aluBound = fmaRate > 0.6 * thrAlu[n] * 1e12; // ALU close to its own limit
            rep.value("lat_regs_" + std::to_string(n), "Gload/s", loadsPerSec * 1e-9,
                      {{"live_regs", double(n)}, {"alu_bound", aluBound ? 1.0 : 0.0}, {"ms_min", s.min}});
            rep.value("inflight_threads_per_core.regs_" + std::to_string(n), "count", loadsPerSec * latNs * 1e-9 / cores,
                      {{"live_regs", double(n)}, {"alu_bound", aluBound ? 1.0 : 0.0}}, true);
            ctx.keepWarm(10);
        }
    }

    // --- Controls -------------------------------------------------------------
    // (a) linearity: 2x FMAs (rounds x2) at N=32 -> 2x time (loads unchanged, so the ALU share
    //     must be dominant: measured on the ALU kernel where nothing else scales).
    u32 tg = 0, mt = 0;
    Stats a1;
    double lin = 0;
    int linTries = 0;
    for (; linTries < 5; ++linTries) { // controls are repeated under interference (minimum times)
        ctx.keepWarm(30);
        a1 = rig.time("regs_alu_32", 32, 1, 1, tg, mt);
        ctx.keepWarm(30);
        const Stats a2 = rig.time("regs_alu_32", 32, 1, 2, tg, mt);
        lin = a2.min / a1.min;
        if (lin > 1.85 && lin < 2.15) break;
    }
    // (b) dynamic indexing (array on the stack) must be slower than the register version.
    const Stats d = rig.time("regs_dyn_32", 32, 2, 1, tg, mt);
    const double dynSlow = d.min / a1.min;
    // dyn uses ALU-only structure with loads=16 rounds: compare with the alu kernel of the same N.
    rep.value("control.dyn_vs_regs.ratio", "ratio", dynSlow, {{"live_regs", 32}}, false);
    rep.value("control.linearity_2x.ratio", "ratio", lin, {}, false);
    rep.negative(lin > 1.85 && lin < 2.15 && rig.wrong == 0,
                 "alu N=32 2x rounds -> " + std::to_string(lin).substr(0, 5) + "x time (want 1.85..2.15, tries " + std::to_string(linTries + 1) + "); " +
                     (rig.wrong ? "results differ from the CPU model: " + rig.wrongWhat : "all results match the CPU model"));
    rep.negative(dynSlow > 1.3,
                 "dynamically indexed array (stack) N=32 takes " + std::to_string(dynSlow).substr(0, 5) +
                     "x the register version (want > 1.3)");
    if (rig.wrong) rep.status(Status::Failed, "wrong results");
    rep.note("regs_<N> = FMA Top/s with 16 dependent gather loads per thread from a 64 MiB table (latency exposed unless other threads hide it) and ~16K FMAs per thread; regs_alu_<N> = same without loads (compiler spills only); values are minimum-time based");
    rep.note("thrashing_point.live_regs = smallest N > 96 (else 48 or 64) with throughput < 80% of the reference (median of N=16,32,48,64,96); 0 = none up to N=256 (found=0); compare with thrashing_point_alu to separate spills from occupancy loss");
    rep.note("lat_regs_<N> = loads/s with 1 FMA round per load (64 loads per thread, latency-bound); inflight_threads_per_core.regs_<N> = loads/s x single-SIMD-group load latency / cores = ESTIMATED resident threads per core (Little), valid only where alu_bound = 0 and once the estimate has saturated");
    rep.note("max_threads_tg.regs_<N> = pipeline maxTotalThreadsPerThreadgroup (drops when the compiler reports high register use); resident threads per core are not observable directly");
}

} // namespace

SOC_BENCH("B-04", "occupancy.registers", "Live registers vs throughput, thrashing point of the dynamic register file",
          benchOccupancy);

} // namespace soc
