// B-09: CPU <-> GPU memory contention.  The GPU streams a 1 GiB working set
// (DRAM size, kernel b08_read of b08_memory.metal: same as B-08's DRAM
// read) while 0/2/6/12 CPU threads run large memcpy on private 256 MiB
// buffers (128 MiB source -> 128 MiB destination, 4 MiB chunks cycling over
// the buffer).  Both sides are measured over the same wall-clock windows.
//
// Serves S-MEM-4 of docs/APPLE_SOC_PLAYBOOK.md.
//
// Units: cpu_bw counts read + write bytes of the memcpy (DRAM traffic it
// generates), so total = gpu_bw + cpu_bw is the traffic both sides moved.
// SLC hits of either side can make the total exceed the DRAM peak.

#include "harness.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>

namespace soc {
namespace {

struct MemParams {
    u32 iters, zero, words, seed;
};

constexpr size_t MiB = size_t(1) << 20, GiB = size_t(1) << 30;
constexpr size_t kCpuBuf   = 256 * MiB;
constexpr size_t kChunk    = 4 * MiB;
constexpr double kNominalGBs = 614.0; // Apple M5 Max nominal (external reference, not a fact measured here)

struct Worker {
    std::atomic<u64> bytes{0}; // read + write
    std::atomic<bool>* stop = nullptr;
    u8* buf = nullptr;
    std::thread th;
};

class CpuLoad {
public:
    CpuLoad() = default;
    ~CpuLoad() { stop(); for (u8* b : bufs_) std::free(b); }
    void ensureBuffers(u32 n) {
        while (bufs_.size() < n) {
            void* p = nullptr;
            if (posix_memalign(&p, 16384, kCpuBuf) != 0) throw BenchError("posix_memalign failed");
            // Non-zero content: 1 MiB of xorshift, replicated.
            u64 s = 0x1234567 + bufs_.size();
            auto* w = static_cast<u64*>(p);
            for (size_t i = 0; i < MiB / 8; ++i) w[i] = xorshift64(s);
            for (size_t o = MiB; o < kCpuBuf; o += MiB) std::memcpy(static_cast<u8*>(p) + o, p, MiB);
            bufs_.push_back(static_cast<u8*>(p));
        }
    }
    void start(u32 n) {
        ensureBuffers(n);
        stopFlag_.store(false);
        workers_.clear();
        for (u32 i = 0; i < n; ++i) workers_.push_back(std::make_unique<Worker>());
        for (u32 i = 0; i < n; ++i) {
            Worker* w = workers_[i].get();
            w->buf = bufs_[i];
            w->th = std::thread([this, w] {
                const size_t half = kCpuBuf / 2;
                size_t off = 0;
                while (!stopFlag_.load(std::memory_order_relaxed)) {
                    std::memcpy(w->buf + half + off, w->buf + off, kChunk);
                    off = (off + kChunk) % half;
                    w->bytes.fetch_add(2 * kChunk, std::memory_order_relaxed);
                }
            });
        }
    }
    void stop() {
        stopFlag_.store(true);
        for (auto& w : workers_) if (w->th.joinable()) w->th.join();
        workers_.clear();
    }
    [[nodiscard]] u64 bytes() const {
        u64 t = 0;
        for (const auto& w : workers_) t += w->bytes.load(std::memory_order_relaxed);
        return t;
    }

private:
    std::vector<u8*> bufs_;
    std::vector<std::unique_ptr<Worker>> workers_;
    std::atomic<bool> stopFlag_{false};
};

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

struct Point {
    Stats gpu;              // GB/s
    Stats cpu;              // GB/s (read + write), zeros with 0 threads
    double total = 0;
};

void benchContention(Context& ctx, Report& rep) {
    const bool quick = ctx.quick();
    MTL::Library* lib = ctx.library("b08_memory.metal", true);
    MTL::ComputePipelineState* read = ctx.compute(lib, "b08_read");
    MTL::Buffer* params = ctx.buffer(256);
    MTL::Buffer* out    = ctx.buffer(8 * MiB);
    MTL::Buffer* src    = ctx.randomBuffer(GiB);
    const u32* s32   = static_cast<const u32*>(src->contents());
    const u32* out32 = static_cast<const u32*>(out->contents());
    const u32 words = u32(GiB / 16), threads = 327680;
    const std::vector<u32> counts = {0, 2, 6, 12};
    const double windowMs = quick ? 150.0 : 400.0;
    bool resultsOk = true;

    auto gpuOnce = [&]() {
        const MemParams p{1, 0, words, 0};
        std::memcpy(params->contents(), &p, sizeof(p));
        ComputeTimer t(ctx);
        MTL4::ComputeCommandEncoder* e = t.begin();
        ctx.table()->setAddress(src->gpuAddress(), 0);
        ctx.table()->setAddress(params->gpuAddress(), 1);
        ctx.table()->setAddress(out->gpuAddress(), 2);
        e->setComputePipelineState(read);
        e->setArgumentTable(ctx.table());
        e->dispatchThreads(MTL::Size::Make(threads, 1, 1), MTL::Size::Make(256, 1, 1));
        t.lap();
        return t.finish()[0];
    };
    const double gpuScale = double(GiB) * 1e-6; // GB/s = scale / ms

    CpuLoad load;
    // CPU alone (GPU idle), for the contention ratio of the CPU side.
    std::vector<double> cpuAlone(counts.size(), 0.0);
    for (size_t k = 1; k < counts.size(); ++k) {
        load.start(counts[k]);
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        const double t0 = nowMs();
        const u64 b0 = load.bytes();
        std::this_thread::sleep_for(std::chrono::milliseconds(long(windowMs)));
        const double t1 = nowMs();
        cpuAlone[k] = double(load.bytes() - b0) / ((t1 - t0) * 1e6);
        load.stop();
    }
    ctx.warmUp(5.0);

    auto runPoint = [&](u32 nThreads) {
        Point pt;
        if (nThreads) {
            load.start(nThreads);
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
        }
        std::vector<double> gpuBw, cpuBw;
        const double tStart = nowMs();
        do {
            const double t0 = nowMs();
            const u64 b0 = load.bytes();
            const Stats s = ctx.measure(gpuOnce, std::max<u32>(ctx.reps(), 15));
            const double t1 = nowMs();
            const u64 b1 = load.bytes();
            if (!checkRead(s32, words, threads, 1, out32)) resultsOk = false;
            gpuBw.push_back(gpuScale / s.median);
            if (nThreads) cpuBw.push_back(double(b1 - b0) / ((t1 - t0) * 1e6));
        } while (nowMs() - tStart < windowMs || gpuBw.size() < 3);
        load.stop();
        pt.gpu = phosphor::soc::computeStats(gpuBw);
        if (nThreads) pt.cpu = phosphor::soc::computeStats(cpuBw);
        pt.total = pt.gpu.median + pt.cpu.median;
        ctx.keepWarm(30);
        return pt;
    };

    std::vector<Point> pts;
    for (u32 n : counts) pts.push_back(runPoint(n));
    // Baseline again at the end: drift control.
    const Point base2 = runPoint(0);

    bool boundOk = true;
    std::string boundWhy;
    for (size_t k = 0; k < counts.size(); ++k) {
        const std::string tag = ".cpu_threads_" + std::to_string(counts[k]);
        const std::map<std::string, double> prm = {{"cpu_threads", double(counts[k])}, {"gpu_ws_MiB", 1024}, {"window_ms", windowMs}};
        rep.metric("gpu_bw" + tag, "GB/s", pts[k].gpu, prm);
        if (counts[k]) {
            rep.metric("cpu_bw" + tag, "GB/s", pts[k].cpu, prm);
            rep.value("cpu_bw_alone" + tag, "GB/s", cpuAlone[k], prm);
            rep.value("cpu_bw_retained" + tag, "ratio", pts[k].cpu.median / cpuAlone[k], prm);
            rep.value("gpu_bw_retained" + tag, "ratio", pts[k].gpu.median / pts[0].gpu.median, prm);
        }
        rep.value("total_bw" + tag, "GB/s", pts[k].total, prm);
        if (pts[k].total > 1.6 * kNominalGBs) {
            boundOk = false;
            boundWhy += "total(" + std::to_string(counts[k]) + ")=" + std::to_string(int(pts[k].total)) + " ";
        }
        if (counts[k] && pts[k].gpu.median > pts[0].gpu.median * 1.10) {
            boundOk = false;
            boundWhy += "gpu(" + std::to_string(counts[k]) + ") faster than alone ";
        }
        if (counts[k] && pts[k].cpu.median > cpuAlone[k] * 1.25) {
            boundOk = false;
            boundWhy += "cpu(" + std::to_string(counts[k]) + ") faster than alone ";
        }
    }
    const double drift = base2.gpu.median / pts[0].gpu.median;
    rep.value("gpu_bw.baseline_end_ratio", "ratio", drift, {{"cpu_threads", 0}});

    rep.negative(std::fabs(drift - 1.0) < 0.10 && pts[0].gpu.median > 200 && pts[0].gpu.median < 1.2 * kNominalGBs,
                 "0-thread GPU read " + std::to_string(int(pts[0].gpu.median)) + " GB/s; repeated at the end " +
                     std::to_string(int(base2.gpu.median)) + " GB/s (ratio " + std::to_string(drift).substr(0, 5) +
                     ", must be within +-10% = same measurement as B-08's dram_bw; plausible range 200..737)");
    rep.negative(boundOk && resultsOk,
                 std::string(boundOk ? "no total above 1.6 x nominal 614 GB/s, contended sides never faster than alone" : "IMPLAUSIBLE: " + boundWhy) +
                     (resultsOk ? "; GPU results match the CPU" : "; GPU RESULTS WRONG"));
    rep.note("cpu_bw = memcpy read+write bytes/s (DRAM traffic), threads default QoS, 256 MiB private buffer each (128 MiB -> 128 MiB, 4 MiB chunks); windows shared with the GPU measurements");
    rep.note("M5 Max nominal DRAM bandwidth 614 GB/s is an external reference (Apple), not a measured fact; SLC hits may push a total above it");
    rep.note("cpu threads are not pinned (macOS has no affinity API): P/E core placement is left to the scheduler");
}

} // namespace

SOC_BENCH("B-09", "memory.contention", "CPU memcpy threads vs GPU DRAM streaming: bandwidth of both sides", benchContention);

} // namespace soc
