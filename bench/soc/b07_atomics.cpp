// B-07: atomics.  32-bit atomic_fetch_add on device and threadgroup memory at
// several contention levels (all threads on 1 address, 32, 1024, one per
// thread; padded to 128 B or not), 64-bit atomic max/min on device memory
// (the only 64-bit atomics MSL 4 exposes: no add, no threadgroup), and
// hierarchical (SIMD-group prefix + one atomic per SIMD-group / per
// threadgroup) vs naive stream compaction.  Final counters, the sum of the
// returned old values and the compacted output are verified exactly on the CPU.
//
// Serves S-SIMD-4 of docs/APPLE_SOC_PLAYBOOK.md (and S-SIMD-1: prefix before atomics).

#include "harness.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>

namespace soc {
namespace {

struct AtParams {
    u32 iters, mask, stride, elems;
};
static_assert(sizeof(AtParams) == 16);

constexpr u32 kDevThreads = 1u << 18;
constexpr u32 kTgThreads = 1u << 20;
constexpr u32 kCompThreads = 1u << 20;

u64 splitmix(u32 gid, u32 k) {
    u64 z = (u64(gid) << 20) + k + 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

bool keepElem(u32 gid, u32 e) {
    u32 h = gid * 2654435761u + e * 40503u + 12345u;
    h ^= h >> 15;
    h *= 2246822519u;
    h ^= h >> 13;
    return (h & 1u) != 0u;
}

Stats rate(const Stats& ms, double ops, double scale) { // ops/s * scale from time stats
    Stats t = ms;
    const double s = ops * scale * 1e3;
    t.median = s / ms.median;
    t.min = s / ms.max;
    t.max = s / ms.min;
    t.p10 = s / ms.p90;
    t.p90 = s / ms.p10;
    t.mean = s / ms.mean;
    return t;
}

struct Rig {
    Context&      ctx;
    MTL::Library* lib;
    MTL::Buffer*  params;
    MTL::Buffer*  ctr;   // counters
    MTL::Buffer*  out;   // per-thread outputs / compaction slots
    u32           wrong = 0;
    std::string   wrongWhat{};

    void bad(const std::string& what) {
        ++wrong;
        wrongWhat += what + " ";
    }

    template <typename Reset, typename Encode>
    Stats run(Reset&& reset, Encode&& encode) {
        return ctx.measure([&] {
            reset();
            ComputeTimer t(ctx);
            MTL4::ComputeCommandEncoder* e = t.begin();
            encode(e);
            t.lap();
            return t.finish()[0];
        });
    }

    void bind3(MTL4::ComputeCommandEncoder* e, MTL::ComputePipelineState* pso) {
        ctx.table()->setAddress(ctr->gpuAddress(), 0);
        ctx.table()->setAddress(out->gpuAddress(), 1);
        ctx.table()->setAddress(params->gpuAddress(), 2);
        e->setComputePipelineState(pso);
        e->setArgumentTable(ctx.table());
    }

    // Device fetch_add.  Returns the time stats.
    Stats dev(u32 naddr, u32 stride, u32 iters) {
        const AtParams p{iters, naddr - 1, stride, 0};
        std::memcpy(params->contents(), &p, sizeof(p));
        MTL::ComputePipelineState* pso = ctx.compute(lib, "at_dev");
        const size_t words = size_t(naddr - 1) * stride + 1;
        const Stats s = run([&] { std::memset(ctr->contents(), 0, words * 4); },
                            [&](MTL4::ComputeCommandEncoder* e) {
                                bind3(e, pso);
                                e->dispatchThreads(MTL::Size::Make(kDevThreads, 1, 1), MTL::Size::Make(256, 1, 1));
                            });
        // Verify: final counters, sum of the old values per address.
        const u32* c = static_cast<const u32*>(ctr->contents());
        const u32* o = static_cast<const u32*>(out->contents());
        const u64 n = u64(kDevThreads / naddr) * iters;
        const u32 want = u32(n * (n - 1) / 2);
        bool ok = true;
        for (u32 a : {0u, naddr / 2, naddr - 1}) {
            if (c[size_t(a) * stride] != u32(n)) ok = false;
            u32 sum = 0;
            for (u32 t = a; t < kDevThreads; t += naddr) sum += o[t];
            if (sum != want) ok = false;
        }
        if (!ok) bad("dev_n" + std::to_string(naddr) + "_s" + std::to_string(stride));
        return s;
    }

    Stats tg(u32 naddr, u32 iters) {
        const AtParams p{iters, naddr - 1, 1, 0};
        std::memcpy(params->contents(), &p, sizeof(p));
        MTL::ComputePipelineState* pso = ctx.compute(lib, "at_tg");
        const Stats s = run([] {}, [&](MTL4::ComputeCommandEncoder* e) {
            bind3(e, pso);
            e->dispatchThreads(MTL::Size::Make(kTgThreads, 1, 1), MTL::Size::Make(256, 1, 1));
        });
        const u32* c = static_cast<const u32*>(ctr->contents()); // flushed per-TG counters (256 words per TG)
        const u32* o = static_cast<const u32*>(out->contents());
        const u64 n = u64(256 / naddr) * iters;
        const u32 want = u32(n * (n - 1) / 2);
        bool ok = true;
        for (u32 tgi : {0u, 1u, kTgThreads / 256 - 1}) {
            for (u32 a = 0; a < 256; ++a) {
                const u32 expect = a < naddr ? u32(n) : 0u;
                if (c[tgi * 256 + a] != expect) ok = false;
            }
            for (u32 a : {0u, naddr - 1}) {
                u32 sum = 0;
                for (u32 t = a; t < 256; t += naddr) sum += o[tgi * 256 + t];
                if (sum != want) ok = false;
            }
        }
        if (!ok) bad("tg_n" + std::to_string(naddr));
        return s;
    }

    // 64-bit / 32-bit max/min on device memory.
    Stats atomic64(MTL::Library* lib64, const std::string& fn, bool is64, bool isMax, u32 naddr, u32 stride, u32 iters) {
        const AtParams p{iters, naddr - 1, stride, 0};
        std::memcpy(params->contents(), &p, sizeof(p));
        MTL::ComputePipelineState* pso = ctx.compute(lib64, fn);
        const size_t words = (size_t(naddr - 1) * stride + 1) * (is64 ? 2 : 1);
        const Stats s = run(
            [&] {
                if (is64) {
                    u64* c = static_cast<u64*>(ctr->contents());
                    for (size_t i = 0; i < (words / 2); ++i) c[i] = isMax ? 0ull : ~0ull;
                } else {
                    std::memset(ctr->contents(), 0, words * 4);
                }
            },
            [&](MTL4::ComputeCommandEncoder* e) {
                ctx.table()->setAddress(ctr->gpuAddress(), 0);
                ctx.table()->setAddress(params->gpuAddress(), 1);
                e->setComputePipelineState(pso);
                e->setArgumentTable(ctx.table());
                e->dispatchThreads(MTL::Size::Make(kDevThreads, 1, 1), MTL::Size::Make(256, 1, 1));
            });
        bool ok = true;
        for (u32 a : {0u, naddr / 2, naddr - 1}) {
            u64 want = isMax ? 0 : ~0ull;
            for (u32 t = a; t < kDevThreads; t += naddr)
                for (u32 k = 0; k < iters; ++k) {
                    u64 v = splitmix(t, k);
                    if (!is64) v = u32(v);
                    want = isMax ? std::max(want, v) : std::min(want, v);
                }
            if (is64) {
                if (static_cast<const u64*>(ctr->contents())[size_t(a) * stride] != want) ok = false;
            } else if (static_cast<const u32*>(ctr->contents())[size_t(a) * stride] != u32(want)) {
                ok = false;
            }
        }
        if (!ok) bad(fn + "_n" + std::to_string(naddr));
        return s;
    }

    // Compaction; returns time stats.
    Stats compact(const std::string& fn, u32 elems) {
        const AtParams p{0, 0, 0, elems};
        std::memcpy(params->contents(), &p, sizeof(p));
        MTL::ComputePipelineState* pso = ctx.compute(lib, fn);
        const size_t slots = size_t(kCompThreads) * elems;
        const Stats s = run(
            [&] {
                std::memset(ctr->contents(), 0, 4);
                std::memset(out->contents(), 0xFF, slots * 4);
            },
            [&](MTL4::ComputeCommandEncoder* e) {
                bind3(e, pso);
                e->dispatchThreads(MTL::Size::Make(kCompThreads, 1, 1), MTL::Size::Make(256, 1, 1));
            });
        // Verify: counter == number of kept elements; slots[0..count) hold each kept element exactly once.
        u32 want = 0;
        std::vector<u8> isKept(slots, 0);
        for (u32 g = 0; g < kCompThreads; ++g)
            for (u32 e = 0; e < elems; ++e)
                if (keepElem(g, e)) {
                    ++want;
                    isKept[size_t(g) * elems + e] = 1;
                }
        const u32 count = static_cast<const u32*>(ctr->contents())[0];
        bool ok = count == want;
        if (ok) {
            const u32* sl = static_cast<const u32*>(out->contents());
            std::vector<u8> seen(slots, 0);
            for (u32 i = 0; i < count && ok; ++i) {
                const u32 v = sl[i];
                if (v >= slots || !isKept[v] || seen[v]) ok = false;
                else seen[v] = 1;
            }
            for (size_t i = count; i < std::min<size_t>(slots, count + 64) && ok; ++i)
                if (sl[i] != 0xFFFFFFFFu) ok = false; // nothing written past the end
        }
        if (!ok) bad(fn);
        return s;
    }
};

u32 calibrateDev(Rig& rig, u32 naddr, u32 stride, double targetMs) {
    const double t1 = std::max(1e-3, rig.dev(naddr, stride, 1).min);
    return std::max<u32>(1, std::min<u32>(64, u32(targetMs / t1)));
}

void benchAtomics(Context& ctx, Report& rep) {
    const u32 bufWords = kDevThreads * 32; // per-thread padded: 32 MB
    Rig rig{ctx, ctx.library("b07_atomics.metal", false), ctx.buffer(256),
            ctx.buffer(std::max<size_t>(size_t(bufWords) * 4, size_t(kTgThreads / 256) * 256 * 4)),
            ctx.buffer(size_t(kCompThreads) * 16 * 4)};
    const double targetMs = ctx.quick() ? 0.2 : 0.4;

    // --- Device 32-bit fetch_add ------------------------------------------------------
    struct DevCase {
        const char* name;
        u32         naddr, stride;
    };
    const DevCase devCases[] = {{"n1", 1, 1},        {"n32", 32, 32},        {"n32_unpadded", 32, 1},
                                {"n1024", 1024, 32}, {"nthread", kDevThreads, 1}, {"nthread_padded", kDevThreads, 32}};
    double rateN1 = 0, rateThr = 0, rate1024 = 0;
    u32 it1024 = 1;
    Stats t1024;
    for (const DevCase& c : devCases) {
        const u32 iters = calibrateDev(rig, c.naddr, c.stride, targetMs);
        const Stats s = rig.dev(c.naddr, c.stride, iters);
        const double ops = double(kDevThreads) * iters;
        rep.metric(std::string("atomic.dev.") + c.name, "Gop/s", rate(s, ops, 1e-9),
                   {{"naddr", double(c.naddr)}, {"stride_words", double(c.stride)}, {"iters", double(iters)}, {"ms", s.median}});
        const double r = ops / (s.min * 1e-3) * 1e-9;
        if (std::string(c.name) == "n1") rateN1 = r;
        if (std::string(c.name) == "nthread") rateThr = r;
        if (std::string(c.name) == "n1024") {
            rate1024 = r;
            it1024 = iters;
            t1024 = s;
        }
        ctx.keepWarm(10);
    }
    rep.value("atomic.dev.contention_ratio", "ratio", rateThr / rateN1, {}, true); // one address per thread / single address
    // --- Threadgroup 32-bit fetch_add ---------------------------------------------------
    double tgN1 = 0, tgN256 = 0;
    for (u32 naddr : {1u, 32u, 256u}) {
        const u32 iters = naddr == 1 ? 64 : 512;
        const Stats s = rig.tg(naddr, iters);
        const double ops = double(kTgThreads) * iters;
        rep.metric("atomic.tg.n" + std::to_string(naddr), "Gop/s", rate(s, ops, 1e-9),
                   {{"naddr", double(naddr)}, {"iters", double(iters)}, {"ms", s.median}});
        const double r = ops / (s.min * 1e-3) * 1e-9;
        if (naddr == 1) tgN1 = r;
        if (naddr == 256) tgN256 = r;
        ctx.keepWarm(10);
    }
    rep.value("atomic.tg.contention_ratio", "ratio", tgN256 / tgN1, {}, true);
    rep.value("atomic.tg_vs_dev.n1.ratio", "ratio", tgN1 / rateN1, {}, true);

    // --- 64-bit device atomics -----------------------------------------------------------
    MTL::Library* lib64 = nullptr;
    try {
        lib64 = ctx.library("b07_atomics64.metal", false);
    } catch (const BenchError& e) {
        rep.status(Status::Partial, std::string("64-bit atomics unavailable in MSL 4 on this toolchain: ") + e.what());
        rep.value("atomic64.supported", "count", 0, {});
    }
    if (lib64) {
        rep.value("atomic64.supported", "count", 1, {});
        struct C64 {
            const char* name;
            u32         naddr, stride;
        };
        const C64 cases64[] = {{"n1", 1, 1}, {"n32", 32, 16}, {"n1024", 1024, 16}, {"nthread", kDevThreads, 2}};
        double r64n32 = 0, r32n32 = 0;
        for (const bool isMax : {true, false}) {
            for (const C64& c : cases64) {
                const u32 iters = ctx.quick() ? 8 : 16;
                const std::string fn = isMax ? "at64_max" : "at64_min";
                const Stats s = rig.atomic64(lib64, fn, true, isMax, c.naddr, c.stride, iters);
                const double ops = double(kDevThreads) * iters;
                rep.metric(std::string("atomic64.") + (isMax ? "max." : "min.") + c.name, "Gop/s", rate(s, ops, 1e-9),
                           {{"naddr", double(c.naddr)}, {"iters", double(iters)}, {"ms", s.median}});
                if (isMax && std::string(c.name) == "n32") r64n32 = ops / (s.min * 1e-3) * 1e-9;
                ctx.keepWarm(10);
            }
        }
        for (const C64& c : cases64) {
            const u32 iters = ctx.quick() ? 8 : 16;
            const Stats s = rig.atomic64(lib64, "at32_max", false, true, c.naddr, c.stride > 1 ? 16 : 1, iters);
            const double ops = double(kDevThreads) * iters;
            rep.metric(std::string("atomic32.max.") + c.name, "Gop/s", rate(s, ops, 1e-9),
                       {{"naddr", double(c.naddr)}, {"iters", double(iters)}, {"ms", s.median}});
            if (std::string(c.name) == "n32") r32n32 = ops / (s.min * 1e-3) * 1e-9;
            ctx.keepWarm(10);
        }
        rep.value("atomic64.max.vs_32.n32.ratio", "ratio", r64n32 / std::max(1e-9, r32n32), {}, true);
    }

    // --- Compaction ----------------------------------------------------------------------
    const u32 elems = ctx.quick() ? 8 : 16;
    double tNaive = 0, tSimd = 0, tSimdTg = 0;
    for (const char* fn : {"compact_naive", "compact_simd", "compact_simd_tg"}) {
        const Stats s = rig.compact(fn, elems);
        const double n = double(kCompThreads) * elems;
        rep.metric(std::string("compact.") + (fn + 8), "Gelem/s", rate(s, n, 1e-9), {{"elems_per_thread", double(elems)}, {"ms", s.median}});
        const double t = s.min;
        if (std::string(fn) == "compact_naive") tNaive = t;
        if (std::string(fn) == "compact_simd") tSimd = t;
        if (std::string(fn) == "compact_simd_tg") tSimdTg = t;
        ctx.keepWarm(10);
    }
    rep.value("compact.speedup_simd", "ratio", tNaive / tSimd, {}, true);
    rep.value("compact.speedup_simd_tg", "ratio", tNaive / tSimdTg, {}, true);

    // --- Controls --------------------------------------------------------------------------
    // (a) linearity: 2x iterations on the 1024-address case -> 2x time (min-based), exact results.
    double lin = 0;
    int linTries = 0;
    for (; linTries < 5; ++linTries) { // controls are repeated under interference (minimum times)
        ctx.keepWarm(30);
        const Stats t1b = rig.dev(1024, 32, it1024);
        ctx.keepWarm(30);
        const Stats t2 = rig.dev(1024, 32, it1024 * 2);
        lin = t2.min / t1b.min;
        if (lin > 1.7 && lin < 2.3) break;
    }
    rep.negative(rig.wrong == 0 && lin > 1.7 && lin < 2.3,
                 "dev n1024 2x iterations -> " + std::to_string(lin).substr(0, 5) + "x time (1.7..2.3, tries " + std::to_string(linTries + 1) + "); " +
                     (rig.wrong ? "WRONG results/counters: " + rig.wrongWhat
                                : "counters, returned-value sums, 64-bit max/min and compaction outputs exact"));
    // (b) contention must cost: one address slower than one per thread; naive compaction slower than hierarchical.
    rep.negative(rateThr > rateN1 * 2.0 && rate1024 > rateN1 * 2.0,
                 "per-thread-address rate " + std::to_string(rateThr).substr(0, 6) + " vs single address " +
                     std::to_string(rateN1).substr(0, 6) + " Gop/s (want > 2x); 1024 addresses " +
                     std::to_string(rate1024).substr(0, 6) + " (want > 2x single); naive compaction " +
                     std::to_string(tNaive).substr(0, 6) + " ms vs simd " + std::to_string(tSimd).substr(0, 6) + " ms (reported, not asserted)");
    if (rig.wrong) rep.status(Status::Failed, "wrong results");
    rep.note("dev: atomic_fetch_add(+1) relaxed on address (gid & (naddr-1)) * stride words; padded = 32 words (128 B), unpadded = 1 word; 2^18 threads; tg: 2^20 threads in 256-thread groups, threadgroup atomic_uint");
    rep.note("64-bit: MSL 4 provides only atomic_max_explicit/atomic_min_explicit (no return value) on device atomic_ulong; add/exchange/fetch_* and threadgroup 64-bit atomics do not compile (metal_atomic header); atomic32.max.* is the same op on 32-bit values");
    rep.note("compaction: each thread keeps ~1/2 of its elements; naive = 1 device atomic per kept element, simd = simd_prefix_exclusive_sum + 1 atomic per SIMD-group, simd_tg = 1 tg atomic per SIMD-group + 1 device atomic per threadgroup; Gelem/s counts all input elements");
    rep.note("rates in metrics are medians; ratios use minimum times");
}

} // namespace

SOC_BENCH("B-07", "atomics", "32/64-bit atomics: device vs threadgroup, contention, hierarchical vs naive compaction", benchAtomics);

} // namespace soc
