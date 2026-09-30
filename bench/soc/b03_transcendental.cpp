// B-03: cost of transcendental and division operations: precise:: vs fast::
// vs half vs default calls under fast math, expressed as throughput per op
// (Top/s) and as cost relative to an FP32 FMA ("fma_equiv"); integer
// divide/modulo (runtime and compile-time divisors) vs shift/mask.
//
// Each link of a chain is v = op(fma(v, a, b)) (a contraction, see the
// shader); the cost of the op alone is t(op kernel) - t(glue kernel = fma
// only), both at the same iteration count.  Caveat: if the op unit runs
// concurrently with the FMA pipe the subtraction under-estimates the op cost;
// the raw times are kept in the params (ms_op, ms_glue).
//
// Serves S-ALU-4 of docs/APPLE_SOC_PLAYBOOK.md.

#include "harness.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>

namespace soc {
namespace {

struct TrParams {
    u32   iters, pad;
    float a, b;
    float c1, c2;
    u32   d, pad2;
};
static_assert(sizeof(TrParams) == 32);

constexpr u32 kThreads = 1u << 20;
constexpr int kChains  = 8;

struct OpInfo {
    const char* name;
    float       a, b;
    double (*f)(double y, double c1, double c2);
    bool        transcendental; // counted in the "transcendental" summary
};

const OpInfo kOps[] = {
    {"rcp", 0.5f, 0.75f, [](double y, double, double) { return 1.0 / y; }, true},
    {"rsqrt", 0.5f, 0.75f, [](double y, double, double) { return 1.0 / std::sqrt(y); }, true},
    {"sqrt", 0.5f, 0.75f, [](double y, double, double) { return std::sqrt(y); }, false},
    {"exp2", -0.5f, 0.25f, [](double y, double, double) { return std::exp2(y); }, true},
    {"log2", 0.5f, 2.0f, [](double y, double, double) { return std::log2(y); }, true},
    {"sin", 0.5f, 1.0f, [](double y, double, double) { return std::sin(y); }, true},
    {"cos", 0.5f, 1.0f, [](double y, double, double) { return std::cos(y); }, true},
    {"pow", 0.5f, 0.75f, [](double y, double, double c2) { return std::pow(y, c2); }, false},
    {"div", 0.5f, 0.75f, [](double y, double c1, double) { return c1 / y; }, false},
};
constexpr double kC1 = 1.0, kC2 = 0.75;

double fixedPoint(const OpInfo& op) {
    double x = 1.0;
    for (int k = 0; k < 400; ++k) x = op.f(double(op.a) * x + double(op.b), kC1, kC2);
    return x;
}

struct Iv {
    const char* name;
};
const char* const kIntVariants[] = {"shift", "mask", "udiv_c8", "udiv_c7", "umod_c7", "udiv_rt", "umod_rt", "sdiv_rt"};
constexpr u32 kDivisor = 7;

u32 intResult(const std::string& var, u32 i, u32 iters) {
    u32 x[kChains];
    for (int c = 0; c < kChains; ++c) x[c] = i * 2654435761u + u32(c) * 40503u + 1u;
    for (u32 k = 0; k < iters; ++k)
        for (int c = 0; c < kChains; ++c) {
            x[c] = x[c] * 1664525u + 1013904223u;
            u32 r;
            if (var == "shift") r = x[c] >> 3u;
            else if (var == "mask") r = x[c] & 7u;
            else if (var == "udiv_rt" || var == "udiv_c7") r = x[c] / kDivisor;
            else if (var == "umod_rt" || var == "umod_c7") r = x[c] % kDivisor;
            else if (var == "udiv_c8") r = x[c] / 8u;
            else if (var == "sdiv_rt") r = u32(i32(x[c]) / i32(kDivisor));
            else r = kDivisor; // base
            x[c] ^= r;
        }
    u32 h = 0;
    for (int c = 0; c < kChains; ++c) h = h * 31u + x[c];
    return h;
}

struct Rig {
    Context&      ctx;
    MTL::Library* safe;
    MTL::Library* fastLib;
    MTL::Buffer*  params;
    MTL::Buffer*  out;

    MTL::ComputePipelineState* pso(const std::string& fn, bool fastMath = false) {
        return ctx.compute(fastMath ? fastLib : safe, fn);
    }

    double timeOnce(MTL::ComputePipelineState* p, u32 iters, const OpInfo* op) {
        TrParams pr{iters, 0, op ? op->a : 0.5f, op ? op->b : 0.75f, float(kC1), float(kC2), kDivisor, 0};
        std::memcpy(params->contents(), &pr, sizeof(pr));
        ComputeTimer t(ctx);
        MTL4::ComputeCommandEncoder* e = t.begin();
        ctx.table()->setAddress(out->gpuAddress(), 0);
        ctx.table()->setAddress(params->gpuAddress(), 1);
        e->setComputePipelineState(p);
        e->setArgumentTable(ctx.table());
        e->dispatchThreads(MTL::Size::Make(kThreads, 1, 1), MTL::Size::Make(256, 1, 1));
        t.lap();
        return t.finish()[0];
    }
    Stats time(MTL::ComputePipelineState* p, u32 iters, const OpInfo* op) {
        return ctx.measure([&] { return timeOnce(p, iters, op); });
    }
    u32 calibrate(MTL::ComputePipelineState* p, const OpInfo* op, double targetMs) {
        const u32 probe = 64;
        const double tp = std::max(1e-3, ctx.measure([&] { return timeOnce(p, probe, op); }, 5).min);
        return std::max<u32>(16, u32(double(probe) * targetMs / tp));
    }
    /// Float kernel results vs the double-precision fixed point (relative tolerance).
    u32 checkFloat(const OpInfo& op, double tol) const {
        const double want = double(kChains) * fixedPoint(op);
        const u32* o = static_cast<const u32*>(out->contents());
        u32 bad = 0;
        for (u32 i : {0u, 1u, 5u, 15u, 16u, 1000u, 65535u, kThreads - 1}) {
            float f;
            std::memcpy(&f, &o[i], 4);
            if (!(std::fabs(double(f) - want) <= tol * want)) ++bad;
        }
        return bad;
    }
};

/// Throughput stats (Top/s) for `ops` operations from the time stats of the op kernel minus
/// the glue median (clamped to 2% of the op time: the op can not be free).
Stats netThroughput(const Stats& op, double glueMs, double ops) {
    // Minima, not medians: other GPU clients only ever add time (opt-log OPT-0: CV of minima <= 0.08%).
    auto net = [&](double t) { return std::max(t - glueMs, 0.02 * op.min); };
    Stats t = op;
    const double s = ops * 1e-12 * 1e3;
    t.median = s / net(op.min);
    t.min = s / net(op.max);
    t.max = s / net(op.min);
    t.p10 = s / net(op.p90);
    t.p90 = s / net(op.p10);
    t.mean = s / net(op.mean);
    return t;
}

double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

void benchTranscendental(Context& ctx, Report& rep) {
    Rig rig{ctx, ctx.library("b03_transcendental.metal", false), ctx.library("b03_transcendental.metal", true),
            ctx.buffer(256), ctx.buffer(size_t(kThreads) * 4)};
    const double targetMs = ctx.quick() ? 0.15 : 0.3;
    const u32 cores = std::max<u32>(1, describeMachine(ctx).gpuCores);
    const auto& mhz = ctx.gpuState().pstateMHz();
    const double topMHz = mhz.empty() ? 0 : mhz.back();

    u32 wrong = 0;
    std::string wrongWhat;
    bool opNotFree = true, preciseNotFaster = true;
    std::string freeWhat, fasterWhat;

    struct Mode {
        const char* label; // metric infix ("" for f16)
        const char* fn;    // kernel suffix
        bool        fastLib;
        bool        half;
        double      tol;
    };
    const Mode modes[] = {{"f32.%s.precise", "precise", false, false, 0.01},
                          {"f32.%s.fast", "fast", false, false, 0.02},
                          {"f16.%s", "half", false, true, 0.04},
                          {"f32.%s.default_fastmath", "def", true, false, 0.02}};
    std::map<std::string, double> topsOf; // "f32.rsqrt.fast" -> Top/s
    std::map<std::string, double> msOf;   // op kernel time at its own iteration count
    std::map<std::string, u32> itersOf;   // iterations of every op kernel
    double fmaRefIters = 0;
    for (const OpInfo& op : kOps) {
        // Reference FP32 FMA kernel at the same iteration count as each op kernel below.
        MTL::ComputePipelineState* refGlue = rig.pso("tr_glue_fast");
        for (const Mode& m : modes) {
            const std::string fn = std::string("tr_") + op.name + "_" + m.fn;
            MTL::ComputePipelineState* pop = rig.pso(fn, m.fastLib);
            MTL::ComputePipelineState* pglue = rig.pso(std::string("tr_glue_") + m.fn, m.fastLib);
            const u32 iters = rig.calibrate(pop, &op, targetMs);
            Stats sop = rig.time(pop, iters, &op);
            itersOf[fn] = iters;
            if (const u32 bad = rig.checkFloat(op, m.tol)) {
                wrong += bad;
                wrongWhat += fn + " ";
            }
            ctx.keepWarm(10);
            double tGlue = rig.time(pglue, iters, &op).min;
            const double tRef = rig.time(refGlue, iters, &op).min;
            ctx.keepWarm(10);
            // An op kernel faster than its glue can only be interference (clock change between the
            // two timings): re-measure both, up to 5 times, before the control flags it.
            for (int t = 0; t < 5 && sop.min < tGlue * 0.98; ++t) {
                ctx.keepWarm(30);
                sop = rig.time(pop, iters, &op);
                tGlue = std::min(tGlue, rig.time(pglue, iters, &op).min);
            }
            if (sop.min < tGlue * 0.98) {
                opNotFree = false;
                freeWhat += fn + " ";
            }
            const double ops = double(kThreads) * double(kChains) * double(iters);
            char buf[64];
            std::snprintf(buf, sizeof buf, m.label, op.name);
            const std::string base = buf;
            const double netMs = std::max(sop.min - tGlue, 0.02 * sop.min);
            const Stats thr = netThroughput(sop, tGlue, ops);
            rep.metric(base, "Top/s", thr,
                       {{"chains", double(kChains)}, {"iters", double(iters)}, {"ms_op", sop.min}, {"ms_glue", tGlue},
                        {"ms_ref_fma", tRef}});
            rep.value(base + ".fma_equiv", "ratio", netMs / tRef,
                      {{"iters", double(iters)}, {"ms_op", sop.min}, {"ms_glue", tGlue}, {"ms_ref_fma", tRef}}, false);
            if (topMHz > 0)
                rep.value(base + ".per_core_clk", "op/core/clk", thr.median * 1e12 / (double(cores) * topMHz * 1e6),
                          {{"mhz_assumed", topMHz}});
            topsOf[base] = thr.median;
            msOf[base] = sop.min;
            fmaRefIters = double(iters);
        }
    }
    (void)fmaRefIters;
    // Summaries.
    auto summary = [&](const char* label, const char* infix, const char* suffix) {
        std::vector<double> v;
        for (const OpInfo& op : kOps)
            if (op.transcendental) {
                char buf[64];
                std::snprintf(buf, sizeof buf, infix, op.name);
                v.push_back(topsOf[buf]);
            }
        (void)suffix;
        rep.value(label, "Top/s", median(v), {{"ops", double(v.size())}});
    };
    summary("f32.transcendental.fast", "f32.%s.fast", "");
    summary("f32.transcendental.precise", "f32.%s.precise", "");
    summary("f16.transcendental", "f16.%s", "");
    summary("f32.transcendental.default_fastmath", "f32.%s.default_fastmath", "");
    // precise never faster than fast (control)
    for (const OpInfo& op : kOps) {
        const double tp = msOf[std::string("f32.") + op.name + ".precise"], tf = msOf[std::string("f32.") + op.name + ".fast"];
        (void)tp;
        (void)tf;
        const double thrP = topsOf[std::string("f32.") + op.name + ".precise"], thrF = topsOf[std::string("f32.") + op.name + ".fast"];
        bool violates = thrP > thrF * 1.1;
        for (int t = 0; t < 5 && violates; ++t) { // interference: re-measure both kernels back to back
            ctx.keepWarm(30);
            double bestOp[2] = {1e30, 1e30}, bestGlue[2] = {1e30, 1e30};
            for (int mIdx = 0; mIdx < 2; ++mIdx) {
                const char* mn = mIdx == 0 ? "precise" : "fast";
                const std::string fn = std::string("tr_") + op.name + "_" + mn;
                for (int r = 0; r < 2; ++r) {
                    bestOp[mIdx] = std::min(bestOp[mIdx], rig.time(rig.pso(fn), itersOf[fn], &op).min);
                    bestGlue[mIdx] = std::min(bestGlue[mIdx], rig.time(rig.pso(std::string("tr_glue_") + mn), itersOf[fn], &op).min);
                }
            }
            const double nP = std::max(bestOp[0] - bestGlue[0], 0.02 * bestOp[0]);
            const double nF = std::max(bestOp[1] - bestGlue[1], 0.02 * bestOp[1]);
            // same op count for both kernels only if the iteration counts match: compare per-iteration net time
            const double perP = nP / itersOf[std::string("tr_") + op.name + "_precise"];
            const double perF = nF / itersOf[std::string("tr_") + op.name + "_fast"];
            violates = perF > perP * 1.1; // fast slower per op than precise by > 10%
        }
        if (violates) {
            preciseNotFaster = false;
            fasterWhat += std::string(op.name) + " ";
        }
    }

    // --- Integer divide / modulo vs shift / mask ---------------------------------
    {
        MTL::ComputePipelineState* base = rig.pso("iv_base");
        MTL::ComputePipelineState* refGlue = rig.pso("tr_glue_fast");
        for (const char* var : kIntVariants) {
            MTL::ComputePipelineState* pv = rig.pso(std::string("iv_") + var);
            const u32 iters = rig.calibrate(pv, nullptr, targetMs);
            Stats sv = rig.time(pv, iters, nullptr);
            u32 bad = 0;
            const u32* o = static_cast<const u32*>(rig.out->contents());
            for (u32 i : {0u, 1u, 7u, 16u, 1000u, 65535u, kThreads - 1})
                if (o[i] != intResult(var, i, iters)) ++bad;
            if (bad) {
                wrong += bad;
                wrongWhat += std::string("iv_") + var + " ";
            }
            ctx.keepWarm(10);
            double tBase = rig.time(base, iters, nullptr).min;
            const double tRef = rig.time(refGlue, iters, nullptr).min;
            ctx.keepWarm(10);
            for (int t = 0; t < 5 && sv.min < tBase * 0.98; ++t) { // interference: re-measure both
                ctx.keepWarm(30);
                sv = rig.time(pv, iters, nullptr);
                tBase = std::min(tBase, rig.time(base, iters, nullptr).min);
            }
            const bool cheap = std::string(var) == "shift" || std::string(var) == "mask" || std::string(var) == "udiv_c8"; // may fuse into the xor: free
            if (!cheap && sv.min < tBase * 0.98) {
                opNotFree = false;
                freeWhat += std::string("iv_") + var + " ";
            }
            const double ops = double(kThreads) * double(kChains) * double(iters);
            const std::string name = std::string("i32.") + var;
            const double netMs = std::max(sv.min - tBase, 0.02 * sv.min);
            rep.metric(name, "Top/s", netThroughput(sv, tBase, ops),
                       {{"chains", double(kChains)}, {"iters", double(iters)}, {"ms_op", sv.min}, {"ms_base", tBase}});
            rep.value(name + ".fma_equiv", "ratio", netMs / tRef, {{"ms_op", sv.min}, {"ms_base", tBase}, {"ms_ref_fma", tRef}},
                      false);
            rep.value(name + ".time_vs_base", "ratio", sv.min / tBase, {{"ms_op", sv.min}, {"ms_base", tBase}}, false);
        }
    }

    // --- Controls -----------------------------------------------------------------
    // Linearity (2x iterations -> 2x time) on the slowest FP op family (fast sin), + all results correct.
    const OpInfo* sinOp = nullptr;
    for (const OpInfo& op : kOps)
        if (std::string(op.name) == "sin") sinOp = &op;
    MTL::ComputePipelineState* psin = rig.pso("tr_sin_fast");
    const u32 itl = rig.calibrate(psin, sinOp, targetMs);
    double lin = 0;
    int linTries = 0;
    for (; linTries < 5; ++linTries) { // controls are repeated under interference (minimum times)
        ctx.keepWarm(30);
        const double l1 = rig.time(psin, itl, sinOp).min;
        ctx.keepWarm(30);
        const double l2 = rig.time(psin, itl * 2, sinOp).min;
        lin = l2 / l1;
        if (lin > 1.85 && lin < 2.15) break;
    }
    rep.negative(lin > 1.85 && lin < 2.15 && wrong == 0,
                 "sin.fast 2x iterations -> " + std::to_string(lin).substr(0, 5) + "x time (want 1.85..2.15, tries " + std::to_string(linTries + 1) + "); " +
                     (wrong ? "results out of tolerance/differ: " + wrongWhat
                            : "all float results within tolerance of the double fixed point, ints exact"));
    rep.negative(opNotFree && preciseNotFaster,
                 std::string(opNotFree ? "every op kernel >= its glue (fma only) kernel" : "op kernel FASTER than glue: " + freeWhat) +
                     "; " + (preciseNotFaster ? "no precise:: op more than 10% faster than fast::" : "precise faster than fast: " + fasterWhat));
    if (wrong) rep.status(Status::Failed, "results out of tolerance");
    rep.note("all times are MINIMA of the repetitions (other GPU clients only add time); throughput metric value = minimum-time based");
    rep.note("cost of an op = t(op kernel) - t(glue kernel = fma only) at equal iterations; fma_equiv = that time / time of one FP32 FMA per link (same iterations), i.e. the op's cost in FP32 FMAs; raw times in params");
    rep.note("chains: v = op(fma(v,a,b)), 8 per thread, contraction with a fixed point in the domain of the op; float results checked against the double fixed point (1-4% tolerance, approximations differ from CPU), ints bit-exact");
    rep.note("modes: precise:: / fast:: (library in MathModeSafe) ; f16 = half overloads ; default_fastmath = plain calls compiled with fast math ON (Xcode default); rcp = 1.0f/v, div = c/v with runtime c");
    rep.note("integer: x = x*K+C; x ^= r(x); base = mad + xor with a runtime value; udiv_c7 = constant divisor (mul-high), *_rt runtime divisor 7");
    rep.note("f32.transcendental.* = median Top/s of rcp, rsqrt, exp2, log2, sin, cos");
}

} // namespace

SOC_BENCH("B-03", "alu.transcendental", "rcp/rsqrt/sqrt/exp2/log2/sin/cos/pow/div: precise vs fast vs half; int div/mod vs shift",
          benchTranscendental);

} // namespace soc
