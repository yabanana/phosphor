// B-22: Neural Accelerator GEMM through MetalPerformancePrimitives tensor ops
// (matmul2d): FP16/BF16 -> FP32, INT8 -> INT32, INT8 x INT4 -> INT32 and
// FP8 e4m3 -> FP32 (MSL 4.1 library, separate), swept over tile descriptor
// (32x32 .. 128x64) and scope (execution_simdgroups<1/2/4/8>) and matrix size;
// simdgroup_matrix FP16 baseline on the shader ALUs; a fused per-pixel MLP
// 64->64->64->3 (weights in threadgroup memory) on tensor ops with a
// cooperative destination vs simdgroup_matrix.
//
// Serves S-NA-1..3 of docs/APPLE_SOC_PLAYBOOK.md.  Results are small-integer
// exact and checked against the CPU on a sample (incl. corners); C is
// pre-filled with a sentinel so untouched tiles are caught.

#include "harness.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>

namespace soc {
namespace {

using i8_t = int8_t;
using i64_t = long long;
struct Dims { i32 m, n, k, pad; };
struct MlpParams { u32 tilesPerGroup, seed, pixels, pad; };

enum Ty { F16, BF16, I8, I4, FP8 };
const char* tyName(Ty t) {
    switch (t) { case F16: return "f16"; case BF16: return "bf16"; case I8: return "i8"; case I4: return "i4"; default: return "fp8"; }
}
bool tyInt(Ty t) { return t == I8 || t == I4; }
const char* tyUnit(Ty t) { return tyInt(t) ? "TOPS" : "TFLOPS"; }

struct Cfg { u32 tm, tn, sg; };
std::string cfgName(const Cfg& c) { return std::to_string(c.tm) + "x" + std::to_string(c.tn) + ".s" + std::to_string(c.sg); }
std::string kernelName(const char* pfx, const Cfg& c) {
    return std::string(pfx) + "_" + std::to_string(c.tm) + "x" + std::to_string(c.tn) + "_s" + std::to_string(c.sg);
}

u32 mix(u32 h) {
    h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
    return h;
}
i32 valA(u32 r, u32 c) { return i32((mix(r * 0x9E3779B1u ^ c * 0x85EBCA6Bu ^ 0x1234u) >> 8) % 5u) - 2; }
i32 valB(u32 r, u32 c) { return i32((mix(r * 0x7FEB352Du ^ c * 0x846CA68Bu ^ 0x9876u) >> 8) % 7u) - 3; }

u16 halfBits(float f) { _Float16 h = _Float16(f); u16 b; std::memcpy(&b, &h, 2); return b; }
u16 bf16Bits(float f) { u32 u; std::memcpy(&u, &f, 4); return u16(u >> 16); }
u8 fp8Bits(i32 v) { // e4m3 of a small integer
    static const u8 mag[4] = {0x00, 0x38, 0x40, 0x44};
    return u8(mag[std::abs(v)] | (v < 0 ? 0x80 : 0));
}

struct GemmData {
    Context& ctx;
    MTL::Buffer *a, *b, *c, *dims;
    std::vector<i8_t> ai, bi; // CPU copies (row-major M x K, K x N)
    i32 m = 0, n = 0, k = 0;
    Ty ty = F16;
    bool filled = false;
    explicit GemmData(Context& cx) : ctx(cx) {
        const size_t maxEl = size_t(4096) * 4096;
        a = ctx.buffer(maxEl * 2);
        b = ctx.buffer(maxEl * 2);
        c = ctx.buffer(maxEl * 4);
        dims = ctx.buffer(256);
    }
    void fill(Ty t, i32 M, i32 N, i32 K) {
        if (filled && t == ty && M == m && N == n && K == k) return;
        if (!filled || M != m || N != n || K != k) {
            ai.resize(size_t(M) * K);
            bi.resize(size_t(K) * N);
            for (i32 r = 0; r < M; ++r) for (i32 q = 0; q < K; ++q) ai[size_t(r) * K + q] = i8_t(valA(u32(r), u32(q)));
            for (i32 r = 0; r < K; ++r) for (i32 q = 0; q < N; ++q) bi[size_t(r) * N + q] = i8_t(valB(u32(r), u32(q)));
        }
        ty = t; m = M; n = N; k = K; filled = true;
        const size_t na = ai.size(), nb = bi.size();
        auto* pa = static_cast<u8*>(a->contents());
        auto* pb = static_cast<u8*>(b->contents());
        for (size_t i = 0; i < na; ++i) {
            const float v = float(ai[i]);
            if (t == F16) reinterpret_cast<u16*>(pa)[i] = halfBits(v);
            else if (t == BF16) reinterpret_cast<u16*>(pa)[i] = bf16Bits(v);
            else if (t == FP8) pa[i] = fp8Bits(ai[i]);
            else pa[i] = u8(ai[i]);
        }
        if (t == I4) {
            std::memset(pb, 0, nb / 2 + 1);
            for (size_t i = 0; i < nb; ++i) pb[i >> 1] |= u8((u32(bi[i]) & 0xFu) << ((i & 1) * 4));
        } else {
            for (size_t i = 0; i < nb; ++i) {
                const float v = float(bi[i]);
                if (t == F16) reinterpret_cast<u16*>(pb)[i] = halfBits(v);
                else if (t == BF16) reinterpret_cast<u16*>(pb)[i] = bf16Bits(v);
                else if (t == FP8) pb[i] = fp8Bits(bi[i]);
                else pb[i] = u8(bi[i]);
            }
        }
        Dims d{M, N, K, 0};
        std::memcpy(dims->contents(), &d, sizeof(d));
    }
    void sentinel() { std::memset(c->contents(), 0x7F, size_t(m) * n * 4); }
    // Number of sampled outputs that differ from the CPU (sentinel included).
    u32 check() const {
        u64 rng = 0xC0FFEEull + u64(m) * 31 + u64(n);
        std::vector<std::pair<i32, i32>> pts = {{0, 0}, {m - 1, n - 1}, {0, n - 1}, {m - 1, 0}, {127, 63}, {128, 64}, {m / 2, n / 2}, {63, 31}};
        for (int i = 0; i < 120; ++i) pts.push_back({i32(xorshift64(rng) % u64(m)), i32(xorshift64(rng) % u64(n))});
        u32 bad = 0;
        for (auto [r, q] : pts) {
            i64_t s = 0;
            for (i32 kk = 0; kk < k; ++kk) s += i64_t(ai[size_t(r) * k + kk]) * bi[size_t(kk) * n + q];
            const size_t idx = size_t(r) * n + q;
            const double got = tyInt(ty) ? double(static_cast<const i32*>(c->contents())[idx])
                                         : double(static_cast<const float*>(c->contents())[idx]);
            if (got != double(s)) ++bad;
        }
        return bad;
    }
};

double runGemm(Context& ctx, MTL::ComputePipelineState* pso, const GemmData& g, u32 tm, u32 tn, u32 threads) {
    ComputeTimer t(ctx);
    MTL4::ComputeCommandEncoder* e = t.begin();
    ctx.table()->setAddress(g.a->gpuAddress(), 0);
    ctx.table()->setAddress(g.b->gpuAddress(), 1);
    ctx.table()->setAddress(g.c->gpuAddress(), 2);
    ctx.table()->setAddress(g.dims->gpuAddress(), 3);
    e->setComputePipelineState(pso);
    e->setArgumentTable(ctx.table());
    e->dispatchThreadgroups(MTL::Size::Make(u32(g.n) / tn, u32(g.m) / tm, 1), MTL::Size::Make(threads, 1, 1));
    t.lap();
    return t.finish()[0];
}

Stats toRate(const Stats& s, double ops) { // ms stats -> T(FL)OP/s stats
    const double k = ops * 1e-9;
    Stats r = s;
    r.median = k / s.median; r.min = k / s.max; r.max = k / s.min;
    r.p10 = k / s.p90; r.p90 = k / s.p10; r.mean = k / s.mean;
    return r;
}

struct Result { bool ok = false; Stats ms; u32 bad = 0; };

// Library compiled as MSL 4.1 (raw enum value; SDK 26 has no name for it).
MTL::Library* library41(Context& ctx, const std::string& file, std::string& err) {
    const char* env = std::getenv("SOC_SHADER_DIR");
    std::ifstream in(std::string(env ? env : SOC_SHADER_DIR) + "/" + file);
    if (!in) { err = "cannot read " + file; return nullptr; }
    std::stringstream ss;
    ss << in.rdbuf();
    NS::Error* e = nullptr;
    MTL::CompileOptions* o = MTL::CompileOptions::alloc()->init();
    o->setLanguageVersion(static_cast<MTL::LanguageVersion>((4u << 16) | 1u));
    MTL::Library* lib = ctx.device()->newLibrary(NS::String::string(ss.str().c_str(), NS::UTF8StringEncoding), o, &e);
    o->release();
    if (!lib) { err = e ? e->localizedDescription()->utf8String() : "unknown"; return nullptr; }
    ctx.keep(lib);
    return lib;
}

Result runConfig(Context& ctx, MTL::Library* lib, const char* pfx, GemmData& g, const Cfg& c, std::string& why) {
    Result r;
    MTL::ComputePipelineState* pso = nullptr;
    try {
        pso = ctx.compute(lib, kernelName(pfx, c));
    } catch (const BenchError& e) { why = e.what(); return r; }
    const u32 threads = 32 * c.sg;
    if (pso->maxTotalThreadsPerThreadgroup() < threads) { why = "max threads per threadgroup " + std::to_string(pso->maxTotalThreadsPerThreadgroup()); return r; }
    g.sentinel();
    runGemm(ctx, pso, g, c.tm, c.tn, threads);
    r.bad = g.check();
    r.ms = ctx.measure([&] { return runGemm(ctx, pso, g, c.tm, c.tn, threads); });
    r.ok = true;
    return r;
}

// ---- MLP -------------------------------------------------------------------
u32 mlpInput(u32 pix, u32 k, u32 seed) {
    u32 h = (pix * 64u + k) * 0x9E3779B1u + seed;
    h ^= h >> 15; h *= 0x85EBCA6Bu; h ^= h >> 13; h *= 0xC2B2AE35u; h ^= h >> 16;
    return h % 3u;
}
void mlpReference(const std::vector<i8_t>& w, u32 pix, u32 seed, float out[3]) {
    float x[64], y[64];
    for (u32 k = 0; k < 64; ++k) x[k] = float(int(mlpInput(pix, k, seed)) - 1);
    for (int layer = 0; layer < 2; ++layer) {
        for (int n = 0; n < 64; ++n) {
            float s = 0;
            for (int k = 0; k < 64; ++k) s += x[k] * float(w[size_t(layer) * 4096 + k * 64 + n]);
            y[n] = std::min(std::max(s, 0.0f), 4.0f);
        }
        std::memcpy(x, y, sizeof(x));
    }
    for (int n = 0; n < 3; ++n) {
        float s = 0;
        for (int k = 0; k < 64; ++k) s += x[k] * float(w[8192 + k * 8 + n]);
        out[n] = s;
    }
}

void benchMlp(Context& ctx, Report& rep, bool& allOk, std::string& ctl) {
    MTL::Library* lib = ctx.library("b22_mlp.metal");
    constexpr u32 kPixels = 1u << 20, kTilesPerGroup = 16;
    constexpr size_t kWTotal = 2 * 4096 + 64 * 8;
    std::vector<i8_t> w(kWTotal, 0);
    u64 rng = 0xABCDEFull;
    for (size_t i = 0; i < kWTotal; ++i) w[i] = i8_t(i32(xorshift64(rng) % 3) - 1);
    for (int k = 0; k < 64; ++k) for (int n = 3; n < 8; ++n) w[8192 + k * 8 + n] = 0;
    MTL::Buffer* wb = ctx.buffer(kWTotal * 2);
    for (size_t i = 0; i < kWTotal; ++i) static_cast<u16*>(wb->contents())[i] = halfBits(float(w[i]));
    MTL::Buffer* out = ctx.buffer(size_t(kPixels) * 2 * 3 * 4);
    MTL::Buffer* params = ctx.buffer(256);
    const u32 seed = 0x5EEDu;

    auto run = [&](MTL::ComputePipelineState* pso, u32 pixels) {
        MlpParams p{kTilesPerGroup, seed, pixels, 0};
        std::memcpy(params->contents(), &p, sizeof(p));
        ComputeTimer t(ctx);
        MTL4::ComputeCommandEncoder* e = t.begin();
        ctx.table()->setAddress(wb->gpuAddress(), 0);
        ctx.table()->setAddress(out->gpuAddress(), 1);
        ctx.table()->setAddress(params->gpuAddress(), 2);
        e->setComputePipelineState(pso);
        e->setArgumentTable(ctx.table());
        e->dispatchThreadgroups(MTL::Size::Make(pixels / (64 * kTilesPerGroup), 1, 1), MTL::Size::Make(128, 1, 1));
        t.lap();
        return t.finish()[0];
    };
    double mp[2] = {0, 0};
    const char* names[2] = {"tensor", "simd"};
    const char* fns[2] = {"mlp_tensor", "mlp_simd"};
    for (int v = 0; v < 2; ++v) {
        MTL::ComputePipelineState* pso = nullptr;
        try { pso = ctx.compute(lib, fns[v]); } catch (const BenchError& e) {
            rep.note(std::string("mlp.") + names[v] + " unavailable: " + e.what());
            allOk = false;
            continue;
        }
        std::memset(out->contents(), 0x7F, size_t(kPixels) * 2 * 12);
        run(pso, kPixels);
        u32 bad = 0;
        u64 r2 = 77;
        for (int i = 0; i < 64; ++i) {
            const u32 pix = i < 4 ? (i == 0 ? 0u : i == 1 ? kPixels - 1 : i == 2 ? 63u : 64u) : u32(xorshift64(r2) % kPixels);
            float ref[3];
            mlpReference(w, pix, seed, ref);
            const float* o = static_cast<const float*>(out->contents()) + size_t(pix) * 3;
            for (int j = 0; j < 3; ++j) if (o[j] != ref[j]) ++bad;
        }
        Stats s1, s2;
        double ratio = 0;
        for (int attempt = 0; attempt < 3; ++attempt) { // other GPU clients add noise: retry up to 3 times
            ctx.keepWarm(30);
            s1 = ctx.measure([&] { return run(pso, kPixels); });
            ctx.keepWarm(30);
            s2 = ctx.measure([&] { return run(pso, kPixels * 2); });
            ratio = s2.median / s1.median;
            if (ratio > 1.8 && ratio < 2.2) break;
        }
        Stats rate = s1;
        const double k = double(kPixels) * 1e-3; // Mpixel / (ms * 1e-3) = kPixels/ms... (Mpix/s)
        rate.median = k / s1.median; rate.min = k / s1.max; rate.max = k / s1.min;
        rate.p10 = k / s1.p90; rate.p90 = k / s1.p10; rate.mean = k / s1.mean;
        mp[v] = rate.median;
        rep.metric(std::string("mlp.") + names[v] + ".64x64x64x3.mpix_s", "Mpix/s", rate,
                   {{"pixels", double(kPixels)}, {"ms", s1.median}, {"wrong", double(bad)}, {"ratio_2x", ratio}});
        rep.value(std::string("mlp.") + names[v] + ".64x64x64x3.tflops_eff", "TFLOPS",
                  double(kPixels) * 2.0 * (64 * 64 * 2 + 64 * 8) * 1e-9 / s1.median, {{"pixels", double(kPixels)}});
        const bool lin = ratio > 1.8 && ratio < 2.2;
        allOk &= lin && bad == 0;
        ctl += std::string("mlp.") + names[v] + " 2x pixels->" + std::to_string(ratio).substr(0, 4) + "x, wrong " + std::to_string(bad) + "; ";
    }
    if (mp[0] > 0 && mp[1] > 0) {
        rep.value("mlp.tensor_vs_simd", "ratio", mp[0] / mp[1], {{"pixels", double(kPixels)}});
        rep.note("MLP: 64-pixel tiles, 128 threads, weights (17 KB) + activation tile in threadgroup memory, " +
                 std::to_string(kTilesPerGroup) + " tiles per threadgroup; tensor path = matmul2d 64x64 (layers 1-2) and 64x8 (layer 3), "
                 "cooperative destination (clamp in registers, store to threadgroup); layer 3 padded to 8 outputs");
    }
}

void benchGemm(Context& ctx, Report& rep) {
    const bool a10 = ctx.apple10();
    const bool quick = ctx.quick();
    MTL::Library* lib = ctx.library("b22_gemm.metal");
    GemmData g(ctx);

    std::vector<Cfg> all;
    for (Cfg c : {Cfg{32, 32, 1}, {32, 32, 2}, {32, 32, 4}, {32, 32, 8}, {64, 32, 1}, {64, 32, 2}, {64, 32, 4}, {64, 32, 8},
                  {64, 64, 1}, {64, 64, 2}, {64, 64, 4}, {64, 64, 8}, {128, 64, 1}, {128, 64, 2}, {128, 64, 4}, {128, 64, 8}})
        all.push_back(c);
    const std::vector<Cfg> quickCfgs = {{64, 32, 4}, {64, 64, 4}, {128, 64, 4}};
    const std::vector<Cfg>& cfgs = quick ? quickCfgs : all;
    const std::vector<i32> sizes = quick ? std::vector<i32>{2048} : std::vector<i32>{1024, 2048, 4096};
    const i32 reqSize = sizes.back();
    auto dimsFor = [](i32 sz) { return std::array<i32, 3>{sz == 4096 ? 1024 : sz, sz, sz}; }; // M limited to 1024 at 4096 (span ~1 ms)

    bool wrongAny = false, linOk = true;
    std::string detail, notes;
    std::map<std::string, double> best; // type -> best rate at reqSize
    struct Best { double rate = 0; Cfg cfg{}; Stats ms; i32 m = 0; };
    std::map<std::string, Best> bestBy;

    auto sweepType = [&](Ty ty, MTL::Library* l, const char* pfx, const std::vector<Cfg>& cs, bool required) {
        for (i32 sz : sizes) {
            const auto d = dimsFor(sz);
            g.fill(ty, d[0], d[1], d[2]);
            const double ops = 2.0 * d[0] * double(d[1]) * d[2];
            for (const Cfg& c : cs) {
                std::string why;
                const Result r = runConfig(ctx, l, pfx, g, c, why);
                if (!r.ok) { notes += std::string(tyName(ty)) + " " + cfgName(c) + " n" + std::to_string(sz) + " unavailable (" + why + "); "; continue; }
                if (r.bad) { wrongAny = true; detail += std::string(tyName(ty)) + " " + cfgName(c) + " n" + std::to_string(sz) + ": " + std::to_string(r.bad) + " wrong; "; }
                const Stats rate = toRate(r.ms, ops);
                rep.metric(std::string("gemm.") + tyName(ty) + "." + cfgName(c) + ".n" + std::to_string(sz), tyUnit(ty), rate,
                           {{"tm", double(c.tm)}, {"tn", double(c.tn)}, {"simdgroups", double(c.sg)}, {"m", double(d[0])}, {"n", double(d[1])},
                            {"k", double(d[2])}, {"ms", r.ms.median}, {"wrong", double(r.bad)}});
                if (sz == reqSize && r.bad == 0) {
                    Best& b = bestBy[tyName(ty)];
                    if (rate.median > b.rate) b = {rate.median, c, rate, d[0]};
                }
                ctx.keepWarm(10);
            }
        }
        (void)required;
    };

    // Baseline on the ALUs: simdgroup_matrix (always).
    {
        MTL::ComputePipelineState* pso = ctx.compute(lib, "simd_f16");
        for (i32 sz : sizes) {
            const auto d = dimsFor(sz);
            g.fill(F16, d[0], d[1], d[2]);
            g.sentinel();
            runGemm(ctx, pso, g, 64, 64, 128);
            const u32 bad = g.check();
            if (bad) { wrongAny = true; detail += "simd_f16 n" + std::to_string(sz) + ": " + std::to_string(bad) + " wrong; "; }
            const Stats ms = ctx.measure([&] { return runGemm(ctx, pso, g, 64, 64, 128); });
            const Stats rate = toRate(ms, 2.0 * d[0] * double(d[1]) * d[2]);
            rep.metric("gemm.simd_f16.n" + std::to_string(sz), "TFLOPS", rate,
                       {{"m", double(d[0])}, {"n", double(d[1])}, {"k", double(d[2])}, {"ms", ms.median}, {"wrong", double(bad)}});
            if (sz == reqSize) {
                bestBy["simd_f16"] = {rate.median, {64, 64, 4}, rate, d[0]};
                rep.metric("gemm.simd_f16.tops", "TFLOPS", rate, {{"m", double(d[0])}, {"n", double(d[1])}, {"k", double(d[2])}, {"ms", ms.median}});
            }
            ctx.keepWarm(10);
        }
    }

    if (!a10) {
        rep.status(Status::Partial, "Apple9 (or --force-family apple9): tensor ops (Neural Accelerator) not run; only the simdgroup_matrix baseline "
                                    "was measured, gemm.f16/bf16/i8.tops are missing");
        rep.negative(!wrongAny, wrongAny ? "wrong results: " + detail : "simdgroup_matrix results exact (Apple9 path only)");
        return;
    }

    sweepType(F16, lib, "f16", cfgs, true);
    sweepType(I8, lib, "i8", cfgs, true);
    if (!quick) {
        sweepType(BF16, lib, "bf16", cfgs, true);
        sweepType(I4, lib, "i4", {{64, 32, 4}, {64, 64, 4}, {128, 64, 4}}, false);
    }

    // Required best-config metrics.
    for (const char* t : {"f16", "bf16", "i8"}) {
        auto it = bestBy.find(t);
        if (it == bestBy.end()) continue;
        const Best& b = it->second;
        rep.metric(std::string("gemm.") + t + ".tops", std::string(t) == "i8" ? "TOPS" : "TFLOPS", b.ms,
                   {{"tm", double(b.cfg.tm)}, {"tn", double(b.cfg.tn)}, {"simdgroups", double(b.cfg.sg)}, {"m", double(b.m)},
                    {"n", double(reqSize)}, {"k", double(reqSize)}});
    }
    if (bestBy.count("i4")) {
        const Best& b = bestBy["i4"];
        rep.metric("gemm.i4.tops", "TOPS", b.ms, {{"tm", double(b.cfg.tm)}, {"tn", double(b.cfg.tn)}, {"simdgroups", double(b.cfg.sg)}, {"m", double(b.m)}, {"n", double(reqSize)}, {"k", double(reqSize)}});
    }

    // FP8 (MSL 4.1, own library).
    if (!quick) {
        std::string err;
        MTL::Library* l8 = library41(ctx, "b22_gemm_fp8.metal", err);
        if (!l8) {
            notes += "FP8 unavailable: MSL 4.1 library did not compile (" + err + "); ";
        } else {
            sweepType(FP8, l8, "fp8", {{64, 32, 4}, {64, 64, 4}, {128, 64, 4}}, false);
            if (bestBy.count("fp8")) {
                const Best& b = bestBy["fp8"];
                rep.metric("gemm.fp8.tops", "TFLOPS", b.ms, {{"tm", double(b.cfg.tm)}, {"tn", double(b.cfg.tn)}, {"simdgroups", double(b.cfg.sg)}, {"m", double(b.m)}, {"n", double(reqSize)}, {"k", double(reqSize)}});
            }
        }
    }

    // Cube 4096^3 for the best f16 config (the sizes above use M = 1024 at 4096).
    if (!quick && bestBy.count("f16")) {
        const Cfg c = bestBy["f16"].cfg;
        g.fill(F16, 4096, 4096, 4096);
        std::string why;
        const Result r = runConfig(ctx, lib, "f16", g, c, why);
        if (r.ok) {
            if (r.bad) { wrongAny = true; detail += "f16 cube4096 wrong; "; }
            rep.metric("gemm.f16.cube4096.tops", "TFLOPS", toRate(r.ms, 2.0 * 4096.0 * 4096.0 * 4096.0),
                       {{"tm", double(c.tm)}, {"tn", double(c.tn)}, {"simdgroups", double(c.sg)}, {"ms", r.ms.median}});
        }
    }

    // Controls: 2x K -> ~2x time (best f16 and i8 configs), exact results, tensor beats simd.
    std::string linText;
    for (const char* t : {"f16", "i8"}) {
        if (!bestBy.count(t)) continue;
        const Ty ty = std::string(t) == "f16" ? F16 : I8;
        const Cfg c = bestBy[t].cfg;
        double ratio = 0;
        for (int attempt = 0; attempt < 3; ++attempt) { // other GPU clients add noise: retry up to 3 times
            double ms[2] = {0, 0};
            for (int i = 0; i < 2; ++i) {
                g.fill(ty, 2048, 2048, i == 0 ? 2048 : 4096);
                std::string why;
                const Result r = runConfig(ctx, lib, t, g, c, why);
                if (!r.ok) { linOk = false; break; }
                if (r.bad) wrongAny = true;
                ms[i] = r.ms.median;
                ctx.keepWarm(20);
            }
            ratio = ms[0] > 0 ? ms[1] / ms[0] : 0;
            if (ratio > 1.75 && ratio < 2.25) break;
        }
        const bool lin = ratio > 1.75 && ratio < 2.25;
        linOk &= lin;
        linText += std::string(t) + " 2xK->" + std::to_string(ratio).substr(0, 4) + "x; ";
        rep.value(std::string("gemm.") + t + ".k_scaling_2x", "ratio", ratio, {{"n", 2048}});
    }
    const bool beats = bestBy.count("f16") && bestBy.count("simd_f16") && bestBy["f16"].rate > bestBy["simd_f16"].rate;
    if (bestBy.count("f16") && bestBy.count("simd_f16"))
        rep.value("gemm.f16.tensor_vs_simd", "ratio", bestBy["f16"].rate / bestBy["simd_f16"].rate, {{"n", double(reqSize)}});

    bool mlpOk = true;
    std::string mlpCtl;
    ctx.keepWarm(50);
    benchMlp(ctx, rep, mlpOk, mlpCtl);

    if (!notes.empty()) rep.note(notes);
    rep.note("sizes: n = 1024/2048 cubic; n = 4096 means M = 1024, N = K = 4096 (span ~1 ms; a full 4096^3 cube is gemm.f16.cube4096.tops)");
    if (quick) rep.note("quick: n = 2048 only, f16 + i8, 3 tile configs; gemm.*.tops are at n = 2048");
    rep.negative(linOk && !wrongAny && beats && mlpOk,
                 linText + (wrongAny ? "WRONG RESULTS: " + detail : "all GEMM results exact vs CPU; ") +
                     (beats ? "tensor f16 > simdgroup_matrix f16; " : "tensor f16 NOT faster than simdgroup_matrix; ") + mlpCtl);
    if (wrongAny) rep.status(Status::Failed, "wrong results: " + detail);
}

} // namespace

SOC_BENCH("B-22", "neural_accel.gemm", "Neural Accelerator: matmul2d GEMM by tile/scope/type, MLP vs simdgroup_matrix", benchGemm);

} // namespace soc
