// B-01: ALU throughput FP32 / FP16 / INT32 (FMA-MAD, ADD, MUL), one
// dependent chain vs 8 independent chains per thread.  Reference example of
// a suite benchmark (see harness.h): calibrated dispatch length, CPU check of
// the stored results, negative control on every kernel (2x iterations must
// take 2x time: catches loops the compiler folded).
//
// Serves S-ALU-1..3 of docs/APPLE_SOC_PLAYBOOK.md.

#include "harness.h"

#include <cmath>
#include <cstring>

namespace soc {
namespace {

struct AluParams {
    u32   iters, pad;
    float xf, yf;
    i32   xi, yi;
    u32   pad2, pad3;
};
static_assert(sizeof(AluParams) == 32);

constexpr u32 kThreads = 1u << 20;
constexpr double kMaxOpsPerCoreClk = 512.0; // 4x the documented 128 ALUs per core

enum Op { Fma = 0, Add = 1, Mul = 2 };
enum Type { F32, F16, I32 };

const char* typeName(Type t) { return t == F32 ? "f32" : t == F16 ? "f16" : "i32"; }
const char* opName(Op o) { return o == Fma ? "fma" : o == Add ? "add" : "mul"; }

// Operands: |x| < 1 keeps FP chains finite (MUL decays towards 0, FMA to
// the fixed point y / (1 - x)); ints wrap.
constexpr float kXf = 0.9990234375f; // exact in FP16
constexpr float kYf = 0.0009765625f;
constexpr i32   kXi = 1664525;
constexpr i32   kYi = 1013904223;

u32 halfBits(_Float16 h) {
    u16 b;
    std::memcpy(&b, &h, 2);
    return b;
}
u32 floatBits(float f) {
    u32 b;
    std::memcpy(&b, &f, 4);
    return b;
}

// Bit-exact CPU model of the kernel (MathModeSafe: no reassociation).
u32 reference(Type t, Op op, u32 n, u32 i, u32 iters) {
    u32 h = 0;
    for (u32 c = 0; c < n; ++c) {
        const i32 seed = i32((i + c * 7u) & 15u);
        u32 vb = 0, wb = 0;
        if (t == F32) {
            float v = float(seed) * 0.0625f + 1.0f, w = kYf + float(c);
            for (u32 k = 0; k < iters; ++k) {
                if (op == Fma) v = std::fma(v, kXf, kYf);
                else if (op == Add) { v = v + w; w = w - v; }
                else v = v * kXf;
            }
            vb = floatBits(v);
            wb = floatBits(w);
        } else if (t == F16) {
            const _Float16 x = _Float16(kXf), y = _Float16(kYf);
            _Float16 v = _Float16(_Float16(seed) * _Float16(0.0625) + _Float16(1)), w = _Float16(y + _Float16(c));
            for (u32 k = 0; k < iters; ++k) {
                if (op == Fma) v = _Float16(double(v) * double(x) + double(y)); // exact in double, one rounding
                else if (op == Add) { v = v + w; w = w - v; }
                else v = v * x;
            }
            vb = halfBits(v);
            wb = halfBits(w);
        } else {
            u32 v = i * 2654435761u + c * 40503u + 1u, w = u32(kYi) + c;
            for (u32 k = 0; k < iters; ++k) {
                if (op == Fma) v = v * u32(kXi) + u32(kYi);
                else if (op == Add) { v = v + w; w = w ^ v; }
                else v = v * u32(kXi);
            }
            vb = v;
            wb = w;
        }
        h = h * 31u + vb + (op == Add ? wb * 17u : 0u);
    }
    return h;
}

struct Run {
    MTL::ComputePipelineState* pso;
    MTL::Buffer* params;
    MTL::Buffer* out;
};

double timeOnce(Context& ctx, const Run& r, u32 iters) {
    AluParams p{iters, 0, kXf, kYf, kXi, kYi, 0, 0};
    std::memcpy(r.params->contents(), &p, sizeof(p));
    ComputeTimer t(ctx);
    MTL4::ComputeCommandEncoder* e = t.begin();
    ctx.table()->setAddress(r.out->gpuAddress(), 0);
    ctx.table()->setAddress(r.params->gpuAddress(), 1);
    e->setComputePipelineState(r.pso);
    e->setArgumentTable(ctx.table());
    e->dispatchThreads(MTL::Size::Make(kThreads, 1, 1), MTL::Size::Make(256, 1, 1));
    t.lap();
    return t.finish()[0];
}

void benchAlu(Context& ctx, Report& rep) {
    MTL::Library* lib = ctx.library("b01_alu.metal", /*fastMath=*/false);
    MTL::Buffer* params = ctx.buffer(256);
    MTL::Buffer* out    = ctx.buffer(size_t(kThreads) * 4);
    const double targetMs = ctx.quick() ? 0.2 : 0.4;
    const u32 cores = std::max<u32>(1, describeMachine(ctx).gpuCores);
    const auto& mhz = ctx.gpuState().pstateMHz();
    const double topMHz = mhz.empty() ? 0 : mhz.back();

    std::string linearity;
    bool linearOk = true, wrongAny = false, plausible = true;
    std::string implausible;
    double fp32Indep = 0, fp32Dep = 0;
    for (Type t : {F32, F16, I32}) {
        for (Op op : {Fma, Add, Mul}) {
            for (u32 n : {1u, 8u}) {
                const std::string fn = std::string("alu_") + typeName(t) + "_" + opName(op) + "_" + std::to_string(n);
                const Run r{ctx.compute(lib, fn), params, out};
                // Calibrate the iteration count to ~targetMs.
                const u32 probe = 256;
                const double tp = std::max(1e-3, ctx.measure([&] { return timeOnce(ctx, r, probe); }, 3).median);
                const u32 iters = std::max<u32>(16, u32(double(probe) * targetMs / tp));
                const Stats s1 = ctx.measure([&] { return timeOnce(ctx, r, iters); });
                // Check the stored results of this exact run (iters).
                u32 wrong = 0;
                const u32* o = static_cast<const u32*>(out->contents());
                for (u32 i : {0u, 1u, 7u, 15u, 16u, 1000u, 65535u, kThreads - 1})
                    if (o[i] != reference(t, op, n, i, iters)) ++wrong;
                ctx.keepWarm(20);
                const Stats s2 = ctx.measure([&] { return timeOnce(ctx, r, iters * 2); });
                ctx.keepWarm(20);
                const double ratio = s2.median / s1.median;
                const bool lin = ratio > 1.85 && ratio < 2.15;
                linearOk &= lin;
                if (!lin) linearity += fn + " 2x->" + std::to_string(ratio).substr(0, 5) + "x ";
                if (wrong) {
                    wrongAny = true;
                    rep.note(fn + ": " + std::to_string(wrong) + "/8 results differ from the CPU model");
                }
                // Ops per thread per iteration: ADD = 2 serial ops per pair.
                const double opsPerIter = double(n) * (op == Add ? 2.0 : 1.0);
                const double ops = double(kThreads) * opsPerIter * double(iters);
                const bool fmaFloat = op == Fma && t != I32;
                const double perSec = ops / (s1.median * 1e-3);
                Stats thr = s1;
                // Throughput stats from the time stats (monotone: median maps to median).
                const double scale = ops * (fmaFloat ? 2.0 : 1.0) * 1e-12 / 1e-3;
                thr.median = scale / s1.median;
                thr.min = scale / s1.max;
                thr.max = scale / s1.min;
                thr.p10 = scale / s1.p90;
                thr.p90 = scale / s1.p10;
                thr.mean = scale / s1.mean;
                const std::string base = std::string(typeName(t)) + "." + opName(op) + (n == 1 ? ".dep" : ".indep");
                std::map<std::string, double> params = {{"chains", double(n)}, {"threads", double(kThreads)},
                                                        {"iters", double(iters)}, {"ms", s1.median}};
                rep.metric(base, fmaFloat ? "TFLOPS" : "Top/s", thr, params);
                if (topMHz > 0)
                    rep.value(base + ".per_core_clk", "op/core/clk", perSec / (double(cores) * topMHz * 1e6),
                              {{"chains", double(n)}, {"mhz_assumed", topMHz}});
                if (t == F32 && op == Fma) (n == 1 ? fp32Dep : fp32Indep) = perSec;
                // Plausibility: 128 ALUs per core (Turner) -> anything far above
                // means the compiler removed work (uniform or merged chains).
                if (topMHz > 0 && perSec / (double(cores) * topMHz * 1e6) > kMaxOpsPerCoreClk) {
                    plausible = false;
                    implausible += fn + " ";
                }
            }
        }
    }
    rep.note("per_core_clk assumes the top P-state (" + std::to_string(int(topMHz)) + " MHz): see gpu.top_state_share");
    rep.note("FMA counted as 2 FLOP (TFLOPS) and 1 op (per_core_clk); i32 add = add + xor pairs; library in MathModeSafe");
    const bool ilp = fp32Indep > fp32Dep * 1.2;
    rep.negative(plausible, plausible ? "every kernel <= 512 op/core/clk" : "IMPLAUSIBLE (> 512 op/core/clk): " + implausible);
    rep.negative(linearOk && ilp && !wrongAny,
                 std::string(linearOk ? "all 18 kernels: 2x iterations -> 1.85..2.15x time" : "NOT linear: " + linearity) +
                     "; fp32 fma indep/dep = " + std::to_string(fp32Dep > 0 ? fp32Indep / fp32Dep : 0).substr(0, 5) +
                     (wrongAny ? "; results differ from CPU" : "; results match the CPU model"));
    if (wrongAny) rep.status(Status::Failed, "wrong results");
}

} // namespace

SOC_BENCH("B-01", "alu.throughput", "FP32/FP16/INT32 FMA, ADD, MUL: dependent vs independent chains", benchAlu);

} // namespace soc
