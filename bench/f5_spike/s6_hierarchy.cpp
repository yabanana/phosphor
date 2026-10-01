// F5-S6: transform hierarchy on the GPU (F5.4).  Question: can world matrices
// (world = parentWorld * local, glm column-major) move from the CPU to the GPU,
// how (per-level dispatches, walk to the root, dirty queues with indirect
// dispatch) and without changing a single pixel (bit-exactness vs glm)?
//
// Forest: N = 1,048,576 nodes in level order (parent index < child index);
// depth D in {1, 4, 8}: D = 1 all roots, otherwise ~10% roots and the rest
// evenly over levels 1..D-1, a random parent in the previous level.  Local
// transforms are TRS (non-uniform scale on 30%, a negative scale on ~5%) turned
// into a mat4 on the CPU exactly like TransformComponent::updateMatrix.
// Variants:
//   (a) one dispatch per level (Dispatch->Dispatch barrier between levels);
//   (b) walk: one dispatch, every thread multiplies its ancestor chain top-down
//       (same association as (a): must be bit-identical to it);
//   (c) CPU: glm mat4 product per node in level order, 1 thread and 12 threads
//       (one slice per thread per level, std::barrier between levels).
// Dirty propagation (p of the roots get a new local per frame): the CPU writes
// the dirty-root list (queue 0); kernel L processes queue L, recomputes the
// world matrix and appends ALL children (CSR) to queue L+1 with one atomic per
// SIMD-group; a 1-thread kernel turns queue L+1's count into the indirect
// dispatch arguments of the next level.  All D levels are encoded up front, no
// readback.  The result must be bit-identical to the full recompute (a).
// Bit-exactness: GPU (a) vs CPU glm / CPU plain (no contraction) / CPU std::fma
// chain, for default (fast math) and MathModeSafe libraries, and for the
// plain MSL product and the explicit-fma MSL product.
//
// Metrics: gpu.perlevel.d{D}.ms, gpu.walk.d{D}.ms, cpu.1t.d{D}.ms,
// cpu.12t.d{D}.ms, dirty.d{D}.p{permille}.ms (+ params: touched nodes, ratio
// to the full recompute), dirty.vs_full.d{D}.p{permille}.ratio,
// match.d{D}.{fast|safe}.{plain|fma|nocontract}.vs_{glm|plain|fma}.{identical_pct|
// max_ulp}, ab_identical.d{D}.{fast|safe}.{plain|fma|nocontract}.pct (a vs b),
// glm.matches_{plain|fma}_pct (does glm's product contract into FMA?).
// Timings use the explicit-fma product with the default (fast math) library.

#include "f5_common.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <atomic>
#include <barrier>
#include <cmath>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace soc {
namespace {

constexpr u32 kN        = 1u << 20;
constexpr u32 kNone     = 0xFFFFFFFFu;
constexpr u32 kTG       = 64;
constexpr u32 kThreads  = 12;
constexpr u32 kMaxDepth = 8;

// ---- CPU reference products ---------------------------------------------------------

// glm's order: R[j][r] = ((a0r*bj0 + a1r*bj1) + a2r*bj2) + a3r*bj3.
glm::mat4 mulPlain(const glm::mat4& A, const glm::mat4& B) {
#pragma clang fp contract(off)
    glm::mat4 R;
    for (int j = 0; j < 4; ++j)
        for (int r = 0; r < 4; ++r) R[j][r] = A[0][r] * B[j][0] + A[1][r] * B[j][1] + A[2][r] * B[j][2] + A[3][r] * B[j][3];
    return R;
}

// Every add fused: fma(a3,b3,fma(a2,b2,fma(a1,b1,a0*b0))).
glm::mat4 mulFma(const glm::mat4& A, const glm::mat4& B) {
    glm::mat4 R;
    for (int j = 0; j < 4; ++j)
        for (int r = 0; r < 4; ++r)
            R[j][r] = std::fma(A[3][r], B[j][3], std::fma(A[2][r], B[j][2], std::fma(A[1][r], B[j][1], A[0][r] * B[j][0])));
    return R;
}

enum Prod { Glm, Plain, Fma };

glm::mat4 mulBy(Prod p, const glm::mat4& A, const glm::mat4& B) {
    return p == Glm ? A * B : p == Plain ? mulPlain(A, B) : mulFma(A, B);
}

// ---- forest ------------------------------------------------------------------------------

float u01(u64& s) { return float(xorshift64(s) >> 40) / 16777216.0f; }

// TRS -> mat4 exactly like TransformComponent::updateMatrix (src/scene/components.h).
glm::mat4 randomLocal(u64& rng) {
    const glm::vec3 position{u01(rng) * 20.0f - 10.0f, u01(rng) * 20.0f - 10.0f, u01(rng) * 20.0f - 10.0f};
    glm::quat q{u01(rng) * 2.0f - 1.0f, u01(rng) * 2.0f - 1.0f, u01(rng) * 2.0f - 1.0f, u01(rng) * 2.0f - 1.0f};
    if (glm::dot(q, q) < 1e-6f) q = glm::quat{1.0f, 0.0f, 0.0f, 0.0f};
    const glm::quat rotation = glm::normalize(q);
    glm::vec3 scale{0.8f + u01(rng) * 0.45f};
    if (u01(rng) < 0.30f) scale = {0.8f + u01(rng) * 0.45f, 0.8f + u01(rng) * 0.45f, 0.8f + u01(rng) * 0.45f}; // non-uniform
    if (u01(rng) < 0.05f) scale.x = -scale.x;                                                                   // mirrored
    return glm::translate(glm::mat4{1.0f}, position) * glm::mat4_cast(rotation) * glm::scale(glm::mat4{1.0f}, scale);
}

struct Forest {
    u32 depth = 1;
    u32 roots = 0;
    std::vector<u32> levelStart;                 // depth + 1
    std::vector<u32> parent;                     // kNone for roots
    std::vector<u32> cs;                         // 2 per node: first, count (into clist)
    std::vector<u32> clist;                      // children sorted by parent
    u32 levelCount(u32 l) const { return levelStart[l + 1] - levelStart[l]; }
};

Forest buildForest(u32 depth, u64 seed) {
    Forest f;
    f.depth = depth;
    f.roots = depth == 1 ? kN : kN / 10;
    f.levelStart.assign(depth + 1, 0);
    f.levelStart[1] = f.roots;
    const u32 rest = kN - f.roots;
    for (u32 l = 1; l < depth; ++l) {
        const u32 c = rest / (depth - 1) + (l == depth - 1 ? rest % (depth - 1) : 0);
        f.levelStart[l + 1] = f.levelStart[l] + c;
    }
    f.levelStart[depth] = kN;
    f.parent.assign(kN, kNone);
    u64 rng = seed;
    for (u32 l = 1; l < depth; ++l) {
        const u32 pc = f.levelCount(l - 1), ps = f.levelStart[l - 1];
        for (u32 i = f.levelStart[l]; i < f.levelStart[l + 1]; ++i) f.parent[i] = ps + u32(xorshift64(rng) % pc);
    }
    std::vector<u32> cnt(kN, 0);
    for (u32 i = 0; i < kN; ++i)
        if (f.parent[i] != kNone) ++cnt[f.parent[i]];
    f.cs.assign(size_t(kN) * 2, 0);
    u32 run = 0;
    for (u32 i = 0; i < kN; ++i) {
        f.cs[2 * i]     = run;
        f.cs[2 * i + 1] = cnt[i];
        run += cnt[i];
    }
    f.clist.assign(std::max<u32>(run, 1), 0);
    std::vector<u32> fill(kN, 0);
    for (u32 i = 0; i < kN; ++i)
        if (const u32 p = f.parent[i]; p != kNone) f.clist[f.cs[2 * p] + fill[p]++] = i;
    return f;
}

// ---- CPU world matrices ------------------------------------------------------------------

void cpuRange(const Forest& f, Prod prod, const glm::mat4* local, glm::mat4* world, u32 begin, u32 end) {
    for (u32 i = begin; i < end; ++i) world[i] = f.parent[i] == kNone ? local[i] : mulBy(prod, world[f.parent[i]], local[i]);
}

void cpuFull(const Forest& f, Prod prod, const glm::mat4* local, glm::mat4* world) { cpuRange(f, prod, local, world, 0, kN); }

// 12 persistent threads, one slice of every level each, std::barrier between
// levels (nodes of a level only read the previous one).
class Pool {
public:
    Pool() : sync_(kThreads + 1), level_(kThreads) {
        for (u32 t = 0; t < kThreads; ++t) threads_.emplace_back([this, t] { worker(t); });
    }
    ~Pool() {
        stop_ = true;
        sync_.arrive_and_wait();
        for (auto& th : threads_) th.join();
    }
    void run(const Forest& f, const glm::mat4* local, glm::mat4* world) {
        f_ = &f; local_ = local; world_ = world;
        sync_.arrive_and_wait(); // start
        sync_.arrive_and_wait(); // done
    }

private:
    void worker(u32 t) {
        for (;;) {
            sync_.arrive_and_wait();
            if (stop_) return;
            for (u32 l = 0; l < f_->depth; ++l) {
                const u32 s = f_->levelStart[l], n = f_->levelCount(l);
                const u32 b = s + u32(u64(n) * t / kThreads), e = s + u32(u64(n) * (t + 1) / kThreads);
                cpuRange(*f_, Glm, local_, world_, b, e);
                level_.arrive_and_wait();
            }
            sync_.arrive_and_wait();
        }
    }
    std::barrier<> sync_, level_;
    std::vector<std::thread> threads_;
    std::atomic<bool> stop_{false};
    const Forest* f_ = nullptr;
    const glm::mat4* local_ = nullptr;
    glm::mat4* world_ = nullptr;
};

// ---- bit comparison ----------------------------------------------------------------------

u32 ulpKey(u32 b) { return (b & 0x80000000u) ? ~b : (b | 0x80000000u); }
u32 ulpDiff(u32 a, u32 b) {
    const u32 ka = ulpKey(a), kb = ulpKey(b);
    return ka > kb ? ka - kb : kb - ka;
}

struct Match {
    u64 identical = 0, total = 0;
    u32 maxUlp = 0;
    u64 matrices = 0, matricesIdentical = 0;
    [[nodiscard]] double pct() const { return total ? 100.0 * double(identical) / double(total) : 0.0; }
};

Match compare(const float* a, const float* b, u64 matrices) {
    Match m;
    m.matrices = matrices;
    m.total = matrices * 16;
    for (u64 k = 0; k < matrices; ++k) {
        bool all = true;
        for (u32 e = 0; e < 16; ++e) {
            u32 x, y;
            std::memcpy(&x, a + k * 16 + e, 4);
            std::memcpy(&y, b + k * 16 + e, 4);
            if (x == y) ++m.identical;
            else { all = false; m.maxUlp = std::max(m.maxUlp, ulpDiff(x, y)); }
        }
        if (all) ++m.matricesIdentical;
    }
    return m;
}

// ---- GPU side ----------------------------------------------------------------------------

struct Gpu {
    Context& ctx;
    MTL::Library* fastLib = nullptr;
    MTL::Library* safeLib = nullptr;
    MTL::Buffer *local = nullptr, *world = nullptr, *worldRef = nullptr;
    // per forest
    MTL::Buffer *parent = nullptr, *levelParams = nullptr, *cs = nullptr, *clist = nullptr, *queue = nullptr,
                *counts = nullptr, *dirtyParams = nullptr;
    explicit Gpu(Context& c) : ctx(c) {}

    // mode: 0 plain expression, 1 explicit fma chain, 2 plain with fp contract(off)
    MTL::ComputePipelineState* pso(MTL::Library* lib, const char* fn, u32 mode, bool skip = false) {
        MTL::FunctionConstantValues* fc = MTL::FunctionConstantValues::alloc()->init();
        const bool fma = mode == 1, nc = mode == 2;
        fc->setConstantValue(&fma, MTL::DataTypeBool, NS::UInteger(0));
        fc->setConstantValue(&skip, MTL::DataTypeBool, NS::UInteger(1));
        fc->setConstantValue(&nc, MTL::DataTypeBool, NS::UInteger(2));
        ctx.keep(fc);
        return ctx.compute(lib, fn, fc);
    }

    static void barrier(MTL4::ComputeCommandEncoder* e) {
        e->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
    }

    // (a): one dispatch per level.  Returns GPU ms.
    double perLevel(MTL::ComputePipelineState* p, const Forest& f, MTL::Buffer* out) {
        ComputeTimer t(ctx);
        MTL4::ComputeCommandEncoder* e = t.begin();
        MTL4::ArgumentTable* tb = ctx.table();
        tb->setAddress(local->gpuAddress(), 0);
        tb->setAddress(parent->gpuAddress(), 1);
        tb->setAddress(out->gpuAddress(), 2);
        e->setComputePipelineState(p);
        for (u32 l = 0; l < f.depth; ++l) {
            tb->setAddress(levelParams->gpuAddress() + u64(l) * 16, 3);
            e->setArgumentTable(tb);
            e->dispatchThreads(MTL::Size::Make(f.levelCount(l), 1, 1), MTL::Size::Make(kTG, 1, 1));
            if (l + 1 < f.depth) barrier(e);
        }
        t.lap();
        return t.finish()[0];
    }

    // (b): walk to the root.
    double walk(MTL::ComputePipelineState* p, MTL::Buffer* out) {
        ComputeTimer t(ctx);
        MTL4::ComputeCommandEncoder* e = t.begin();
        MTL4::ArgumentTable* tb = ctx.table();
        tb->setAddress(local->gpuAddress(), 0);
        tb->setAddress(parent->gpuAddress(), 1);
        tb->setAddress(out->gpuAddress(), 2);
        e->setComputePipelineState(p);
        e->setArgumentTable(tb);
        e->dispatchThreads(MTL::Size::Make(kN, 1, 1), MTL::Size::Make(kTG, 1, 1));
        t.lap();
        return t.finish()[0];
    }

    // Dirty propagation: queue 0 / counts[0] must be filled by the CPU.
    double dirty(MTL::ComputePipelineState* pd, MTL::ComputePipelineState* pa, const Forest& f) {
        ComputeTimer t(ctx);
        MTL4::ComputeCommandEncoder* e = t.begin();
        MTL4::ArgumentTable* tb = ctx.table();
        tb->setAddress(local->gpuAddress(), 0);
        tb->setAddress(parent->gpuAddress(), 1);
        tb->setAddress(world->gpuAddress(), 2);
        tb->setAddress(queue->gpuAddress(), 3);
        tb->setAddress(counts->gpuAddress(), 4);
        tb->setAddress(cs->gpuAddress(), 5);
        tb->setAddress(clist->gpuAddress(), 6);
        for (u32 l = 0; l < f.depth; ++l) {
            tb->setAddress(dirtyParams->gpuAddress() + u64(l) * 16, 7);
            if (l > 0) {
                e->setComputePipelineState(pa);
                e->setArgumentTable(tb);
                e->dispatchThreadgroups(MTL::Size::Make(1, 1, 1), MTL::Size::Make(1, 1, 1));
                barrier(e);
            }
            e->setComputePipelineState(pd);
            e->setArgumentTable(tb);
            e->dispatchThreadgroups(counts->gpuAddress() + u64(l) * 16, MTL::Size::Make(kTG, 1, 1));
            if (l + 1 < f.depth) barrier(e);
        }
        t.lap();
        return t.finish()[0];
    }

    void bindForest(const Forest& f) {
        auto fill = [&](MTL::Buffer*& b, const void* src, size_t bytes) {
            if (!b) b = ctx.buffer(size_t(kN) * 8); // sized for the largest array (cs); D = 1 has a 1-entry clist but D > 1 reuses the buffer
            std::memcpy(b->contents(), src, bytes);
        };
        fill(parent, f.parent.data(), f.parent.size() * 4);
        fill(cs, f.cs.data(), f.cs.size() * 4);
        fill(clist, f.clist.data(), f.clist.size() * 4);
        if (!queue) queue = ctx.buffer(size_t(kN) * 4);
        if (!counts) counts = ctx.buffer(kMaxDepth * 16);
        if (!levelParams) levelParams = ctx.buffer(kMaxDepth * 16);
        if (!dirtyParams) dirtyParams = ctx.buffer(kMaxDepth * 16);
        auto* lp = static_cast<u32*>(levelParams->contents());
        auto* dp = static_cast<u32*>(dirtyParams->contents());
        for (u32 l = 0; l < f.depth; ++l) {
            lp[4 * l + 0] = f.levelStart[l];
            lp[4 * l + 1] = f.levelCount(l);
            lp[4 * l + 2] = lp[4 * l + 3] = 0;
            dp[4 * l + 0] = f.levelStart[l];
            dp[4 * l + 1] = f.levelStart[l + 1 < f.depth ? l + 1 : l];
            dp[4 * l + 2] = l;
            dp[4 * l + 3] = l + 1 == f.depth ? 1 : 0;
        }
    }

    void poison(MTL::Buffer* b) { std::memset(b->contents(), 0xFF, size_t(kN) * 64); } // NaN pattern
    const float* data(MTL::Buffer* b) const { return static_cast<const float*>(b->contents()); }
};

const char* permille(double p, char* buf, size_t n) {
    std::snprintf(buf, n, "%u", u32(std::lround(p * 1000.0)));
    return buf;
}

void benchHierarchy(Context& ctx, Report& rep) {
    Gpu g(ctx);
    g.fastLib = f5::f5Library(ctx, "s6_hierarchy.metal", true);
    g.safeLib = f5::f5Library(ctx, "s6_hierarchy.metal", false);
    g.local    = ctx.buffer(size_t(kN) * 64);
    g.world    = ctx.buffer(size_t(kN) * 64);
    g.worldRef = ctx.buffer(size_t(kN) * 64);

    // Local matrices (CPU, glm): the same TRS set for every depth.
    std::vector<glm::mat4> local(kN);
    u64 rng = 0x5EEDF00D12345ull;
    for (u32 i = 0; i < kN; ++i) local[i] = randomLocal(rng);
    std::memcpy(g.local->contents(), local.data(), size_t(kN) * 64);

    bool fail = false;
    std::string failWhat;
    auto require = [&](bool ok, const std::string& what) {
        if (!ok) { fail = true; failWhat += what + "; "; }
    };

    // ---- does glm's product contract into FMA in this build? ------------------------------
    {
        u64 plainM = 0, fmaM = 0, pf = 0;
        const u32 pairs = 200000;
        for (u32 i = 0; i < pairs; ++i) {
            const glm::mat4& A = local[i];
            const glm::mat4& B = local[(u64(i) * 7919 + 13) % kN];
            const glm::mat4 g1 = A * B, p1 = mulPlain(A, B), f1 = mulFma(A, B);
            plainM += std::memcmp(&g1, &p1, 64) == 0;
            fmaM += std::memcmp(&g1, &f1, 64) == 0;
            pf += std::memcmp(&p1, &f1, 64) == 0;
        }
        rep.value("glm.matches_plain_pct", "%", 100.0 * double(plainM) / pairs, {{"pairs", double(pairs)}}, true);
        rep.value("glm.matches_fma_pct", "%", 100.0 * double(fmaM) / pairs, {{"pairs", double(pairs)}}, true);
        rep.value("plain_vs_fma_identical_pct", "%", 100.0 * double(pf) / pairs, {{"pairs", double(pairs)}}, true);
        rep.note("glm product vs explicit refs on " + std::to_string(pairs) + " random pairs: identical to the non-contracted "
                 "product in " + std::to_string(plainM) + ", to the fma chain in " + std::to_string(fmaM) +
                 " (plain vs fma differ in " + std::to_string(pairs - pf) + ")");
    }

    // Negative control of the comparison itself: one flipped bit must be seen.
    {
        std::vector<float> a(32), b(32);
        for (u32 i = 0; i < 32; ++i) a[i] = b[i] = float(i) + 0.5f;
        u32 bits;
        std::memcpy(&bits, &b[20], 4);
        bits ^= 1u;
        std::memcpy(&b[20], &bits, 4);
        const Match m = compare(a.data(), b.data(), 2);
        rep.negative(m.identical == 31 && m.maxUlp == 1 && m.matricesIdentical == 1,
                     "compare(): one flipped low bit in 32 floats -> identical " + std::to_string(m.identical) + "/32, max ulp " +
                         std::to_string(m.maxUlp) + ", matrices identical " + std::to_string(m.matricesIdentical) + "/2");
    }

    const std::vector<u32> depths = ctx.quick() ? std::vector<u32>{4} : std::vector<u32>{1, 4, 8};
    const std::vector<double> ps  = ctx.quick() ? std::vector<double>{0.01} : std::vector<double>{0.001, 0.01, 0.1};

    std::vector<glm::mat4> wGlm(kN), wPlain(kN), wFma(kN), w12(kN);
    Pool pool;
    std::string exactSummary;
    bool skipDetected = true;
    std::string skipDetail;
    bool skipRan = false;

    for (u32 D : depths) {
        const Forest f = buildForest(D, 0xF0E57ull + D);
        g.bindForest(f);
        std::memcpy(g.local->contents(), local.data(), size_t(kN) * 64); // the dirty frames of the previous depth rewrote some locals
        const std::string d = "d" + std::to_string(D);
        ctx.log("F5-S6: depth %u, roots %u, levels %u..%u", D, f.roots, f.levelCount(0), f.levelCount(D - 1));

        // ---- (c) CPU -------------------------------------------------------------------------
        cpuFull(f, Plain, local.data(), wPlain.data());
        cpuFull(f, Fma, local.data(), wFma.data());
        const Stats c1 = ctx.measure([&] {
            const double t0 = nowMs();
            cpuFull(f, Glm, local.data(), wGlm.data());
            return nowMs() - t0;
        });
        rep.metric("cpu.1t." + d + ".ms", "ms", c1, {{"depth", double(D)}, {"nodes", double(kN)}}, false);
        const Stats c12 = ctx.measure([&] {
            const double t0 = nowMs();
            pool.run(f, local.data(), w12.data());
            return nowMs() - t0;
        });
        rep.metric("cpu.12t." + d + ".ms", "ms", c12, {{"depth", double(D)}, {"threads", double(kThreads)}}, false);
        require(std::memcmp(wGlm.data(), w12.data(), size_t(kN) * 64) == 0, d + " cpu 12 threads != 1 thread");

        // ---- bit-exactness: (a) and (b) for {fast, safe} x {plain, fma} ---------------------------
        for (int lib = 0; lib < 2; ++lib)
            for (u32 fma = 0; fma < 3; ++fma) {
                MTL::Library* L = lib == 0 ? g.fastLib : g.safeLib;
                const std::string mode = std::string(lib == 0 ? "fast" : "safe") + "." + (fma == 1 ? "fma" : fma == 2 ? "nocontract" : "plain");
                g.poison(g.world);
                g.poison(g.worldRef);
                g.perLevel(g.pso(L, "h_level", fma), f, g.world);
                g.walk(g.pso(L, "h_walk", fma), g.worldRef);
                const Match ab = compare(g.data(g.world), g.data(g.worldRef), kN);
                rep.value("ab_identical." + d + "." + mode + ".pct", "%", ab.pct(), {{"max_ulp", double(ab.maxUlp)}}, true);
                if (fma == 1) require(ab.identical == ab.total, d + " " + mode + " walk != per-level (" + std::to_string(ab.total - ab.identical) + " floats)");
                const struct { const char* n; const glm::mat4* w; } refs[] = {{"glm", wGlm.data()}, {"plain", wPlain.data()}, {"fma", wFma.data()}};
                std::string line = d + " " + mode + " vs";
                for (const auto& r : refs) {
                    const Match m = compare(g.data(g.world), reinterpret_cast<const float*>(r.w), kN);
                    const std::string key = "match." + d + "." + mode + ".vs_" + r.n;
                    rep.value(key + ".identical_pct", "%", m.pct(), {{"matrices_identical", double(m.matricesIdentical)}}, true);
                    rep.value(key + ".max_ulp", "ulp", double(m.maxUlp), {}, false);
                    line += std::string(" ") + r.n + " " + std::to_string(m.pct()).substr(0, 7) + "% ulp<=" + std::to_string(m.maxUlp);
                    if (m.identical == m.total) exactSummary += d + ":" + mode + "==" + r.n + " ";
                }
                ctx.log("F5-S6: %s", line.c_str());
            }

        // ---- GPU timings (fast math, explicit fma) -------------------------------------------------
        MTL::ComputePipelineState* pLevel = g.pso(g.fastLib, "h_level", 1);
        MTL::ComputePipelineState* pWalk  = g.pso(g.fastLib, "h_walk", 1);
        ctx.log("F5-S6: %s timings", d.c_str());
        ctx.keepWarm(50);
        const Stats gl = ctx.measure([&] { return g.perLevel(pLevel, f, g.world); });
        rep.metric("gpu.perlevel." + d + ".ms", "ms", gl, {{"depth", double(D)}, {"dispatches", double(D)}}, false);
        ctx.keepWarm(20);
        const Stats gw = ctx.measure([&] { return g.walk(pWalk, g.worldRef); });
        rep.metric("gpu.walk." + d + ".ms", "ms", gw, {{"depth", double(D)}}, false);
        ctx.keepWarm(20);

        // ---- dirty propagation ------------------------------------------------------------------------
        g.perLevel(pLevel, f, g.world); // world consistent with the base locals
        MTL::ComputePipelineState* pDirty = g.pso(g.fastLib, "h_dirty", 1);
        MTL::ComputePipelineState* pArgs  = g.pso(g.fastLib, "h_args", 1);
        MTL::ComputePipelineState* pSkip  = g.pso(g.fastLib, "h_dirty", 1, true);
        for (double p : ps) {
            char pm[16];
            permille(p, pm, sizeof pm);
            const std::string key = d + ".p" + pm;
            const u32 nd = std::max<u32>(1, u32(std::lround(p * f.roots)));
            // Dirty roots: nd distinct roots (partial Fisher-Yates), sorted like a scene-order walk would emit them.
            std::vector<u32> ids(f.roots);
            for (u32 i = 0; i < f.roots; ++i) ids[i] = i;
            u64 r2 = 0xD1127ull + D * 977 + nd;
            for (u32 i = 0; i < nd; ++i) std::swap(ids[i], ids[i + xorshift64(r2) % (f.roots - i)]);
            ids.resize(nd);
            std::sort(ids.begin(), ids.end());
            std::vector<glm::mat4> alt[2];
            for (auto& s : alt) {
                s.resize(nd);
                for (auto& m : s) m = randomLocal(r2);
            }
            auto* loc = static_cast<glm::mat4*>(g.local->contents());
            u32 which = 0;
            u64 touched = 0;
            bool exact = true;
            u64 badFloats = 0;
            auto prepare = [&] {
                const auto& s = alt[which ^= 1u];
                for (u32 k = 0; k < nd; ++k) loc[ids[k]] = s[k];
                std::memcpy(static_cast<u32*>(g.queue->contents()), ids.data(), size_t(nd) * 4);
                std::memset(g.counts->contents(), 0, kMaxDepth * 16);
                auto* c = static_cast<u32*>(g.counts->contents());
                c[0] = (nd + kTG - 1) / kTG; c[1] = 1; c[2] = 1; c[3] = nd;
            };
            auto verify = [&](bool expectMismatch) {
                touched = 0;
                const auto* c = static_cast<const u32*>(g.counts->contents());
                for (u32 l = 0; l < D; ++l) touched += c[4 * l + 3];
                g.perLevel(pLevel, f, g.worldRef); // full recompute with the new locals
                const Match m = compare(g.data(g.world), g.data(g.worldRef), kN);
                badFloats = m.total - m.identical;
                if (!expectMismatch) exact &= m.identical == m.total;
                return m;
            };
            ctx.log("F5-S6: dirty %s nd=%u", key.c_str(), nd);
            const Stats sd = ctx.measure([&] {
                prepare();
                const double ms = g.dirty(pDirty, pArgs, f);
                verify(false);
                return ms;
            });
            require(exact, key + " dirty != full recompute (" + std::to_string(badFloats) + " floats)");
            const double ratio = sd.median / gl.median;
            rep.metric("dirty." + key + ".ms", "ms", sd,
                       {{"depth", double(D)}, {"p", p}, {"dirty_roots", double(nd)}, {"touched", double(touched)}, {"vs_full", ratio}}, false);
            rep.value("dirty.vs_full." + key + ".ratio", "x", ratio, {}, false);

            // Negative control: the same frame with the "skip a child" kernel must differ from the full recompute.
            if (D > 1 && !skipRan && p >= 0.0099) {
                // restore a consistent world, then run the corrupted propagation
                prepare();
                g.dirty(pSkip, pArgs, f);
                const Match m = verify(true);
                skipRan = true;
                skipDetected = m.identical != m.total;
                skipDetail = "depth " + std::to_string(D) + " p=" + pm + "/1000: corrupted propagation left " +
                             std::to_string(m.total - m.identical) + " of " + std::to_string(m.total) + " floats different from the full recompute";
                g.perLevel(pLevel, f, g.world); // back to a consistent state
            }
        }
        ctx.keepWarm(20);
    }

    rep.negative(skipRan && skipDetected, skipRan ? skipDetail : "negative control did not run");
    rep.note("bit-exact vs glm (identical_pct == 100): " + (exactSummary.empty() ? std::string("none") : exactSummary));
    rep.note("dirty propagation: kernel per level over queue L, children appended (CSR) with one atomic per SIMD-group, "
             "h_args converts count -> indirect args (counts[L] = {tgx,tgy,tgz,count}), all levels encoded up front; "
             "world after the update is compared with a full recompute (bitwise) in every repetition");
    rep.note("cpu.12t = 12 persistent std::threads + std::barrier between levels; timings use fast math + explicit fma");
    if (fail) rep.status(Status::Failed, failWhat);
}

} // namespace

SOC_BENCH("F5-S6", "scene.hierarchy", "Transform hierarchy on the GPU: per-level vs walk vs CPU, dirty queues, bit-exactness", benchHierarchy);

} // namespace soc
