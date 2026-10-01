// B-29: upload storage modes (S-MEM-3 of docs/APPLE_SOC_PLAYBOOK.md, OPT-1.10).
//
//  (1) CPU writes into a `shared` buffer created with CPUCacheModeWriteCombined
//      ("wc") or DefaultCache ("cached"): memcpy of a large block, loops of
//      16-byte / 64-byte NEON stores (per-object constants), sparse 16-byte
//      stores at a 64-byte stride (partial lines), 1/4/8 threads, 64 KiB..64 MiB.
//  (2) CPU reads back from each (write-combined reads are expected to be very
//      slow: also the negative control that the cache mode took effect).
//  (3) GPU streaming read (b08_read of b08_memory.metal) of the same data
//      living in a shared default-cache buffer, a shared WC buffer and a
//      private buffer, 1 MiB..1 GiB.
//
// Every pass over the destination re-writes the same values (pure function of
// the byte/slot index), so the CPU verifies samples of the final content; GPU
// results are compared with the CPU sum of the shared source.  Textures
// (shared vs private sampled read) are not covered: only buffers.

#include "harness.h"

#include <arm_neon.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

namespace soc {
namespace {

constexpr size_t KiB = size_t(1) << 10, MiB = size_t(1) << 20, GiB = size_t(1) << 30;

struct MemParams {
    u32 iters, zero, words, seed;
};

std::string sizeName(size_t b) {
    if (b >= GiB) return std::to_string(b / GiB) + "GiB";
    if (b >= MiB) return std::to_string(b / MiB) + "MiB";
    return std::to_string(b / KiB) + "KiB";
}

enum class Pat { Memcpy, Store16, Store64, Sparse16 };
const char* patName(Pat p) {
    switch (p) {
    case Pat::Memcpy: return "";
    case Pat::Store16: return "_store16";
    case Pat::Store64: return "_store64";
    case Pat::Sparse16: return "_sparse16";
    }
    return "";
}

// Runs fn(threadIndex) on n threads released together; returns the wall time
// (ms) from release to the last join.
template <class F>
double runThreads(u32 n, F fn) {
    std::atomic<u32> ready{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> th;
    for (u32 t = 0; t < n; ++t)
        th.emplace_back([&, t] {
            ready.fetch_add(1);
            while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
            fn(t);
        });
    while (ready.load() < n) std::this_thread::yield();
    const double t0 = nowMs();
    go.store(true, std::memory_order_release);
    for (auto& t : th) t.join();
    return nowMs() - t0;
}

// The slice work.  Value of 16-byte slot q (slice relative): {q, ~q}.
void writeSlice(Pat pat, u8* dst, size_t len, u32 iters, const u8* src, size_t srcLen) {
    if (pat == Pat::Memcpy) {
        const size_t chunk = std::min(len, srcLen);
        for (u32 it = 0; it < iters; ++it)
            for (size_t o = 0; o < len; o += chunk) std::memcpy(dst + o, src, chunk);
        return;
    }
    const uint64x2_t inc = {1, ~u64(0)};
    for (u32 it = 0; it < iters; ++it) {
        uint64x2_t v = {0, ~u64(0)};
        if (pat == Pat::Store16) {
            for (size_t o = 0; o < len; o += 16) {
                vst1q_u64(reinterpret_cast<u64*>(dst + o), v);
                asm volatile("" ::: "memory");
                v = vaddq_u64(v, inc);
            }
        } else if (pat == Pat::Store64) {
            for (size_t o = 0; o < len; o += 64) {
                for (int j = 0; j < 4; ++j) {
                    vst1q_u64(reinterpret_cast<u64*>(dst + o + 16 * j), v);
                    v = vaddq_u64(v, inc);
                }
                asm volatile("" ::: "memory");
            }
        } else { // Sparse16: slot 4k only
            const uint64x2_t inc4 = {4, ~u64(0) - 3};
            for (size_t o = 0; o < len; o += 64) {
                vst1q_u64(reinterpret_cast<u64*>(dst + o), v);
                asm volatile("" ::: "memory");
                v = vaddq_u64(v, inc4);
            }
        }
    }
}

bool verifySlice(Pat pat, const u8* dst, size_t len, const u8* src, size_t srcLen) {
    if (pat == Pat::Memcpy) {
        const size_t chunk = std::min(len, srcLen), n = std::min<size_t>(4 * KiB, len);
        for (size_t j = 0; j < n; ++j)
            if (dst[j] != src[j % chunk]) return false;
        for (size_t j = len - n; j < len; ++j)
            if (dst[j] != src[j % chunk]) return false;
        return true;
    }
    u64 s = 0xABCDEF12345ull ^ len;
    const size_t slots = len / 16;
    for (int k = 0; k < 64; ++k) {
        size_t q = xorshift64(s) % slots;
        if (pat == Pat::Sparse16) q &= ~size_t(3);
        u64 w[2];
        std::memcpy(w, dst + q * 16, 16);
        if (w[0] != q || w[1] != ~u64(q)) return false;
    }
    return true;
}

double storedBytes(Pat pat, size_t bytes) { return pat == Pat::Sparse16 ? double(bytes) / 4 : double(bytes); }

// Sum of the u64 words of a buffer (NEON, 4 accumulators).
u64 sumWords(const u8* p, size_t bytes) {
    uint64x2_t a0 = vdupq_n_u64(0), a1 = a0, a2 = a0, a3 = a0;
    for (size_t o = 0; o < bytes; o += 64) {
        const u64* w = reinterpret_cast<const u64*>(p + o);
        a0 = vaddq_u64(a0, vld1q_u64(w));
        a1 = vaddq_u64(a1, vld1q_u64(w + 2));
        a2 = vaddq_u64(a2, vld1q_u64(w + 4));
        a3 = vaddq_u64(a3, vld1q_u64(w + 6));
    }
    a0 = vaddq_u64(vaddq_u64(a0, a1), vaddq_u64(a2, a3));
    return vgetq_lane_u64(a0, 0) + vgetq_lane_u64(a0, 1);
}

bool checkRead(const u32* src, u32 words, u32 n, u32 iters, const u32* out) {
    for (u32 i : {0u, 1u, 255u, n / 2, n - 1}) {
        u32 s[4] = {0, 0, 0, 0};
        for (u32 j = i; j < words; j += n)
            for (int c = 0; c < 4; ++c) s[c] += src[size_t(j) * 4 + c];
        for (int c = 0; c < 4; ++c)
            if (out[size_t(i) * 4 + c] != s[c] * iters) return false;
    }
    return true;
}

void benchUpload(Context& ctx, Report& rep) {
    using phosphor::soc::computeStats;
    const bool quick = ctx.quick();
    const u32 cpuReps = quick ? 3 : 7;
    const double targetMs = quick ? 3.0 : 8.0;
    bool cpuOk = true, gpuOk = true;

    struct Mode { const char* name; MTL::ResourceOptions opt; };
    const Mode modes[2] = {{"wc", MTL::ResourceStorageModeShared | MTL::ResourceCPUCacheModeWriteCombined},
                           {"cached", MTL::ResourceStorageModeShared | MTL::ResourceCPUCacheModeDefaultCache}};

    // ---- CPU side ------------------------------------------------------------
    // Source block (ordinary heap memory, random data), 16 MiB.
    const size_t srcLen = 16 * MiB;
    std::vector<u8> srcVec(srcLen);
    {
        u64 s = 0x5DEECE66Dull;
        auto* w = reinterpret_cast<u64*>(srcVec.data());
        for (size_t i = 0; i < srcLen / 8; ++i) w[i] = xorshift64(s);
    }
    const u8* src = srcVec.data();

    const std::vector<size_t> wSizes = quick ? std::vector<size_t>{64 * KiB, 16 * MiB}
                                             : std::vector<size_t>{64 * KiB, 1 * MiB, 16 * MiB, 64 * MiB};
    const std::vector<u32> wThreads = quick ? std::vector<u32>{1, 4} : std::vector<u32>{1, 4, 8};
    const size_t maxSize = wSizes.back();
    MTL::Buffer* cpuBuf[2];
    for (int m = 0; m < 2; ++m) {
        cpuBuf[m] = ctx.buffer(maxSize, modes[m].opt);
        std::memset(cpuBuf[m]->contents(), 0x5A, maxSize); // first touch
    }
    // Check the options really reached the buffers.
    const bool modeApplied = cpuBuf[0]->cpuCacheMode() == MTL::CPUCacheModeWriteCombined &&
                             cpuBuf[1]->cpuCacheMode() == MTL::CPUCacheModeDefaultCache;

    std::map<std::string, double> wcw, cachedw; // "pattern.size.threads" -> median GB/s
    auto writePoint = [&](Pat pat, size_t size, u32 threads) {
        const std::string key = std::string(patName(pat)) + "." + sizeName(size) + "." + std::to_string(threads);
        for (int m = 0; m < 2; ++m) {
            u8* base = static_cast<u8*>(cpuBuf[m]->contents());
            const size_t slice = size / threads;
            auto once = [&](u32 it) {
                return runThreads(threads, [&](u32 t) { writeSlice(pat, base + t * slice, slice, it, src, srcLen); });
            };
            once(1); // warm
            const double t1 = std::max(once(1), 1e-3);
            const u32 iters = u32(std::clamp(std::ceil(targetMs / t1), 1.0, 200000.0));
            std::vector<double> gbps;
            for (u32 r = 0; r < cpuReps; ++r) {
                const double ms = once(iters);
                gbps.push_back(storedBytes(pat, size) * iters / (ms * 1e6));
            }
            for (u32 t = 0; t < threads; ++t)
                if (!verifySlice(pat, base + t * slice, slice, src, srcLen)) cpuOk = false;
            const Stats st = computeStats(gbps);
            const std::string name = std::string("cpu_write") + patName(pat) + "." + modes[m].name + "." + sizeName(size) + "." +
                                     std::to_string(threads) + "t.gbps";
            rep.metric(name, "GB/s", st, {{"bytes", double(size)}, {"threads", double(threads)}, {"iters", double(iters)}});
            (m == 0 ? wcw : cachedw)[key] = st.median;
        }
        ctx.log("B-29 write%s %s %ut: wc %.1f cached %.1f GB/s", patName(pat), sizeName(size).c_str(), threads, wcw[key],
                cachedw[key]);
    };
    for (size_t sz : wSizes)
        for (u32 th : wThreads) writePoint(Pat::Memcpy, sz, th);
    for (Pat p : {Pat::Store16, Pat::Store64, Pat::Sparse16})
        for (size_t sz : wSizes) {
            if (sz > 16 * MiB) continue;
            if (quick && sz > 64 * KiB) continue;
            writePoint(p, sz, 1);
        }
    for (size_t sz : wSizes)
        for (u32 th : wThreads) {
            const std::string k = "." + sizeName(sz) + "." + std::to_string(th);
            rep.value("wc_over_cached.write." + sizeName(sz) + "." + std::to_string(th) + "t", "ratio", wcw[k] / cachedw[k],
                      {{"bytes", double(sz)}, {"threads", double(th)}}, false);
        }
    const std::string hk = "." + sizeName(16 * MiB) + ".1";
    const double wcOverCachedWrite = wcw[hk] / cachedw[hk];
    rep.value("wc_over_cached.write", "ratio", wcOverCachedWrite, {{"bytes", double(16 * MiB)}, {"threads", 1}}, false);

    // ---- CPU read back -----------------------------------------------------------
    // Fill both buffers with the source data (cycled), then read and checksum.
    const std::vector<size_t> rSizes = quick ? std::vector<size_t>{1 * MiB} : std::vector<size_t>{64 * KiB, 1 * MiB, 16 * MiB};
    std::map<size_t, double> readWc, readCached;
    bool readSumOk = true;
    for (size_t size : rSizes) {
        u64 expect = 0;
        for (size_t o = 0; o < size; o += srcLen) expect += sumWords(src, std::min(size - o, srcLen));
        for (int m = 0; m < 2; ++m) {
            u8* base = static_cast<u8*>(cpuBuf[m]->contents());
            for (size_t o = 0; o < size; o += srcLen) std::memcpy(base + o, src, std::min(size - o, srcLen));
            volatile u64 sink = 0;
            auto once = [&](u32 it) {
                const double t0 = nowMs();
                u64 s = 0;
                for (u32 i = 0; i < it; ++i) s += sumWords(base, size);
                sink = s;
                const double t1 = nowMs();
                if (s != expect * it) readSumOk = false;
                return t1 - t0;
            };
            once(1);
            const double t1 = std::max(once(1), 1e-3);
            const u32 iters = u32(std::clamp(std::ceil(targetMs / t1), 1.0, 100000.0));
            std::vector<double> gbps;
            for (u32 r = 0; r < cpuReps; ++r) gbps.push_back(double(size) * iters / (once(iters) * 1e6));
            (void)sink;
            const Stats st = computeStats(gbps);
            rep.metric("cpu_read." + std::string(modes[m].name) + "." + sizeName(size) + ".gbps", "GB/s", st,
                       {{"bytes", double(size)}, {"iters", double(iters)}});
            (m == 0 ? readWc : readCached)[size] = st.median;
        }
        ctx.log("B-29 read %s: wc %.2f cached %.2f GB/s", sizeName(size).c_str(), readWc[size], readCached[size]);
        rep.value("wc_over_cached.read." + sizeName(size), "ratio", readWc[size] / readCached[size], {{"bytes", double(size)}}, false);
    }
    const size_t readKey = 1 * MiB;
    const double wcOverCachedRead = readWc[readKey] / readCached[readKey];
    rep.value("wc_over_cached.read", "ratio", wcOverCachedRead, {{"bytes", double(readKey)}}, false);

    // ---- GPU side ----------------------------------------------------------------
    MTL::Library* lib = ctx.library("b08_memory.metal", true);
    MTL::ComputePipelineState* read = ctx.compute(lib, "b08_read");
    MTL::Buffer* params = ctx.buffer(256);
    MTL::Buffer* out = ctx.buffer(8 * MiB);
    const u32* out32 = static_cast<const u32*>(out->contents());
    const u32 threads = 327680;
    const std::vector<size_t> gSizes = quick ? std::vector<size_t>{4 * MiB, 256 * MiB}
                                             : std::vector<size_t>{1 * MiB, 4 * MiB, 16 * MiB, 64 * MiB, 256 * MiB, 1 * GiB};
    ctx.warmUp(quick ? 3.0 : 8.0);

    auto gpuOnce = [&](MTL::Buffer* b, size_t size, u32 iters) {
        const MemParams p{iters, 0, u32(size / 16), 0};
        std::memcpy(params->contents(), &p, sizeof(p));
        ComputeTimer t(ctx);
        MTL4::ComputeCommandEncoder* e = t.begin();
        ctx.table()->setAddress(b->gpuAddress(), 0);
        ctx.table()->setAddress(params->gpuAddress(), 1);
        ctx.table()->setAddress(out->gpuAddress(), 2);
        e->setComputePipelineState(read);
        e->setArgumentTable(ctx.table());
        e->dispatchThreads(MTL::Size::Make(threads, 1, 1), MTL::Size::Make(256, 1, 1));
        t.lap();
        return t.finish()[0];
    };

    double ctlTime1 = 0, ctlTime2 = 0, driftRatio = 1.0;
    for (size_t size : gSizes) {
        MTL::Buffer* sh = ctx.randomBuffer(size);
        MTL::Buffer* wc = ctx.buffer(size, modes[0].opt);
        MTL::Buffer* pv = ctx.buffer(size, MTL::ResourceStorageModePrivate);
        std::memcpy(wc->contents(), sh->contents(), size);
        {
            CommandTimer ct(ctx);
            MTL4::CommandBuffer* cmd = ct.begin();
            MTL4::ComputeCommandEncoder* ce = cmd->computeCommandEncoder();
            ce->copyFromBuffer(sh, 0, pv, 0, size);
            ce->endEncoding();
            ct.finish();
        }
        const u32* s32 = static_cast<const u32*>(sh->contents());
        const u32 words = u32(size / 16);
        const u32 iters = u32(std::max<size_t>(1, GiB / size));
        const double scale = double(size) * iters * 1e-6;
        auto point = [&](MTL::Buffer* b, const char* storage) {
            std::vector<double> gbps;
            ctx.measure([&] {
                const double ms = gpuOnce(b, size, iters);
                if (!checkRead(s32, words, threads, iters, out32)) gpuOk = false;
                gbps.push_back(scale / ms);
                return ms;
            });
            const Stats st = computeStats(gbps);
            rep.metric("gpu_read." + std::string(storage) + "." + sizeName(size) + ".gbps", "GB/s", st,
                       {{"bytes", double(size)}, {"iters", double(iters)}});
            ctx.keepWarm(30);
            return st.median;
        };
        ctx.keepWarm(250);
        const double a = point(sh, "shared");
        const double b = point(pv, "private");
        const double c = point(wc, "shared_wc");
        const double a2 = point(sh, "shared_again");
        rep.value("private_over_shared.gpu_read." + sizeName(size), "ratio", b / a, {{"bytes", double(size)}});
        rep.value("wc_over_shared.gpu_read." + sizeName(size), "ratio", c / a, {{"bytes", double(size)}});
        if (size == 256 * MiB) {
            driftRatio = a2 / a;
            // 2x bytes -> ~2x time: iters 2 vs 4 on the 256 MiB shared buffer.
            // A slow first batch (clock state, other GPU clients) is measured once more; the first ratio is noted.
            for (int attempt = 0; attempt < 2; ++attempt) {
                ctx.keepWarm(200);
                ctlTime1 = ctx.measure([&] { return gpuOnce(sh, size, 2); }, std::max<u32>(ctx.reps(), 15)).median;
                ctlTime2 = ctx.measure([&] { return gpuOnce(sh, size, 4); }, std::max<u32>(ctx.reps(), 15)).median;
                const double r = ctlTime2 / ctlTime1;
                if (r > 1.7 && r < 2.3) break;
                if (attempt == 0) rep.note("linearity control re-measured once (first ratio " + std::to_string(r).substr(0, 4) + ")");
            }
            if (!checkRead(s32, words, threads, 4, out32)) gpuOk = false;
        }
        ctx.log("B-29 gpu read %s: shared %.0f private %.0f wc %.0f GB/s", sizeName(size).c_str(), a, b, c);
    }

    // ---- Controls ----------------------------------------------------------------------
    char buf[400];
    std::snprintf(buf, sizeof(buf),
                  "CPU read 1 MiB: write-combined %.2f GB/s vs default-cache %.2f GB/s (ratio %.3f, must be < 0.5: proves the cache mode took effect); cpuCacheMode() on the buffers %s",
                  readWc[readKey], readCached[readKey], wcOverCachedRead, modeApplied ? "matches" : "DOES NOT MATCH");
    rep.negative(wcOverCachedRead < 0.5 && modeApplied, buf);
    const double tr = ctlTime2 / ctlTime1;
    std::snprintf(buf, sizeof(buf), "GPU read of 256 MiB: 2 passes %.3f ms, 4 passes %.3f ms (ratio %.2f, must be 1.7..2.3)", ctlTime1,
                  ctlTime2, tr);
    rep.negative(tr > 1.7 && tr < 2.3, buf);
    std::snprintf(buf, sizeof(buf), "GPU shared read at 256 MiB repeated: ratio %.3f (must be within +-10%%); cached memcpy 16 MiB 1t %.0f GB/s (plausible 1..1000)",
                  driftRatio, cachedw[hk]);
    rep.negative(std::fabs(driftRatio - 1.0) < 0.10 && cachedw[hk] > 1 && cachedw[hk] < 1000, buf);
    rep.negative(cpuOk && readSumOk && gpuOk,
                 std::string(cpuOk ? "CPU write samples verified" : "CPU WRITE CONTENT WRONG") + (readSumOk ? "; CPU read checksums match" : "; CPU READ CHECKSUM WRONG") +
                     (gpuOk ? "; GPU results match the CPU sum" : "; GPU RESULTS WRONG"));
    if (!(cpuOk && readSumOk && gpuOk)) rep.status(Status::Failed, "wrong results");
    rep.note("cpu_write: memcpy from a 16 MiB heap block (cycled for larger sizes) or NEON 16-byte stores (store16: one per slot, compiler barrier each; store64: 4 per 64-byte object; sparse16: one per 64-byte line, GB/s counts stored bytes); threads write disjoint slices, unpinned, released together, time = release to last join");
    rep.note("wc_over_cached.write is 16 MiB memcpy 1 thread; wc_over_cached.read is 1 MiB; textures shared vs private not measured");
    rep.note("gpu_read: b08_read, 327680 threads, GiB/size passes per dispatch (a size below the SLC is read from cache after the first pass); private buffers filled with a blit copy; shared_wc = shared + write-combined");
}

} // namespace

SOC_BENCH("B-29", "upload.storage_mode", "CPU write/read bandwidth of write-combined vs cached shared buffers; GPU read of shared vs private", benchUpload);

} // namespace soc
