// B-27: power and clocks under GPU, CPU and mixed load, without root
// (S-PWR-1..3).  IOReport gives the GPU energy (nJ, updated ~2x per second)
// and the GPU P-state residency; the ioreg pmgr table gives the MHz of each
// state; NSProcessInfo gives the thermal state.  The Energy Model channels of
// CPU/DRAM/ANE update only every ~5 minutes (docs/opt-log.md, OPT-0 spike 2):
// they are read only by the soak (--soak MIN) as averages between two
// updates.  powermetrics needs sudo: not used.  Partial by construction:
// no CPU/package watts per phase, a single Mac, battery power possible.

#include "harness.h"

#include <CoreFoundation/CoreFoundation.h>
#include <arm_neon.h>

#include <atomic>
#include <cstring>
#include <thread>

extern "C" {
typedef struct IOReportSubscription* IOReportSubscriptionRef;
CFDictionaryRef IOReportCopyChannelsInGroup(CFStringRef, CFStringRef, uint64_t, uint64_t, uint64_t);
IOReportSubscriptionRef IOReportCreateSubscription(void*, CFMutableDictionaryRef, CFMutableDictionaryRef*, uint64_t,
                                                   CFTypeRef);
CFDictionaryRef IOReportCreateSamples(IOReportSubscriptionRef, CFMutableDictionaryRef, CFTypeRef);
CFStringRef IOReportChannelGetChannelName(CFDictionaryRef);
int64_t IOReportSimpleGetIntegerValue(CFDictionaryRef, int32_t*);
}

namespace soc {
namespace {

struct PowerParams {
    u32 iters, zero;
};

constexpr u32 kThreads = 1u << 20;

// Absolute value (mJ) of an Energy Model channel ("CPU Energy", "DRAM0",
// "ANE0", ...): these counters move only every few minutes.
class SlowEnergy {
public:
    SlowEnergy() {
        CFDictionaryRef e = IOReportCopyChannelsInGroup(CFSTR("Energy Model"), nullptr, 0, 0, 0);
        if (!e) return;
        CFMutableDictionaryRef ch = CFDictionaryCreateMutableCopy(nullptr, 0, e);
        CFRelease(e);
        sub_ = IOReportCreateSubscription(nullptr, ch, &subbed_, 0, nullptr);
        CFRelease(ch);
    }
    ~SlowEnergy() {
        if (subbed_) CFRelease(subbed_);
        if (sub_) CFRelease(reinterpret_cast<CFTypeRef>(sub_));
    }
    SlowEnergy(const SlowEnergy&) = delete;
    SlowEnergy& operator=(const SlowEnergy&) = delete;
    /// mJ of `channel`, -1 if unavailable.
    double read(const char* channel) {
        if (!sub_) return -1;
        CFDictionaryRef s = IOReportCreateSamples(sub_, subbed_, nullptr);
        if (!s) return -1;
        double v = -1;
        auto arr = static_cast<CFArrayRef>(CFDictionaryGetValue(s, CFSTR("IOReportChannels")));
        for (CFIndex i = 0; arr && i < CFArrayGetCount(arr); ++i) {
            auto c = static_cast<CFDictionaryRef>(CFArrayGetValueAtIndex(arr, i));
            char name[128] = {};
            CFStringGetCString(IOReportChannelGetChannelName(c), name, sizeof(name), kCFStringEncodingUTF8);
            if (std::strcmp(name, channel) == 0) v = double(IOReportSimpleGetIntegerValue(c, nullptr));
        }
        CFRelease(s);
        return v;
    }

private:
    IOReportSubscriptionRef sub_ = nullptr;
    CFMutableDictionaryRef subbed_ = nullptr;
};

// All CPU cores busy with NEON FMAs until `stop`.
class CpuLoad {
public:
    void start() {
        stop_ = false;
        const unsigned n = std::max(1u, std::thread::hardware_concurrency());
        for (unsigned t = 0; t < n; ++t)
            threads_.emplace_back([this, t] {
                float32x4_t v[8];
                for (int i = 0; i < 8; ++i) v[i] = vdupq_n_f32(float(i + t));
                const float32x4_t m = vdupq_n_f32(0.999f), k = vdupq_n_f32(0.001f);
                while (!stop_.load(std::memory_order_relaxed))
                    for (int r = 0; r < 4096; ++r)
                        for (int j = 0; j < 8; ++j) v[j] = vfmaq_f32(k, v[j], m);
                float s = 0;
                for (int i = 0; i < 8; ++i) s += vaddvq_f32(v[i]);
                sink_.fetch_add(s > 0 ? 1 : 0, std::memory_order_relaxed);
            });
    }
    void stop() {
        stop_ = true;
        for (auto& t : threads_) t.join();
        threads_.clear();
    }
    ~CpuLoad() { if (!threads_.empty()) stop(); }

private:
    std::atomic<bool> stop_{false};
    std::atomic<u64> sink_{0};
    std::vector<std::thread> threads_;
};

struct Phase {
    phosphor::soc::GpuWindow gpu;
    Stats dispatchMs; // empty when the GPU was idle
};

void benchPower(Context& ctx, Report& rep) {
    MTL::Library* lib = ctx.library("b27_power.metal");
    MTL::ComputePipelineState* pso = ctx.compute(lib, "power_load");
    MTL::Buffer* params = ctx.buffer(256);
    MTL::Buffer* out = ctx.buffer(size_t(kThreads) * 4);
    auto dispatch = [&](u32 iters) {
        const PowerParams p{iters, 0};
        std::memcpy(params->contents(), &p, sizeof(p));
        ComputeTimer t(ctx);
        MTL4::ComputeCommandEncoder* e = t.begin();
        ctx.table()->setAddress(out->gpuAddress(), 0);
        ctx.table()->setAddress(params->gpuAddress(), 1);
        e->setComputePipelineState(pso);
        e->setArgumentTable(ctx.table());
        e->dispatchThreads(MTL::Size::Make(kThreads, 1, 1), MTL::Size::Make(256, 1, 1));
        t.lap();
        return t.finish()[0];
    };
    // ~1 ms per dispatch at the top clock.
    const double t256 = std::max(1e-3, ctx.measure([&] { return dispatch(256); }, 5).median);
    const u32 iters = std::max<u32>(32, u32(256.0 * 1.0 / t256));
    const double phaseSec = ctx.quick() ? 1.2 : 4.0; // IOReport GPU energy updates ~2x/s
    GpuState gs;
    if (!gs.available()) {
        rep.status(Status::Unsupported, "IOReport GPU channels unavailable");
        return;
    }
    CpuLoad cpu;
    auto runPhase = [&](bool gpuLoad, bool cpuLoad) {
        Phase ph;
        if (cpuLoad) cpu.start();
        if (!gpuLoad) std::this_thread::sleep_for(std::chrono::milliseconds(300)); // let the clock drop
        else ctx.keepWarm(200);
        gs.begin();
        const double t0 = nowMs();
        std::vector<double> ms;
        while (nowMs() - t0 < phaseSec * 1000.0) {
            if (gpuLoad) ms.push_back(dispatch(iters));
            else std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        ph.gpu = gs.end();
        if (cpuLoad) cpu.stop();
        if (!ms.empty()) ph.dispatchMs = phosphor::soc::computeStats(ms);
        // Results are masked by a runtime zero: all must be 0.
        const u32* o = static_cast<const u32*>(out->contents());
        for (u32 i : {0u, 12345u, kThreads - 1})
            if (o[i] != 0) throw BenchError("power_load wrote a non-zero masked result");
        return ph;
    };
    const std::string thermal0 = thermalStateName();
    const Phase idle  = runPhase(false, false);
    const Phase gpu   = runPhase(true, false);
    const Phase cpuPh = runPhase(false, true);
    const Phase mixed = runPhase(true, true);
    auto add = [&](const char* name, const Phase& ph) {
        const std::string n(name);
        rep.value(n + ".gpu_watts", "W", ph.gpu.watts, {}, false);
        rep.value(n + ".gpu_mhz", "MHz", ph.gpu.meanMHz);
        rep.value(n + ".gpu_active", "ratio", ph.gpu.activeShare);
        rep.value(n + ".gpu_top_state", "ratio", ph.gpu.topStateShare);
        if (ph.dispatchMs.n) rep.metric(n + ".dispatch_ms", "ms", ph.dispatchMs, {{"iters", double(iters)}}, false);
    };
    add("idle", idle);
    add("gpu", gpu);
    add("cpu", cpuPh);
    add("mixed", mixed);
    if (gpu.dispatchMs.n && mixed.dispatchMs.n)
        rep.value("mixed.gpu_slowdown", "ratio", mixed.dispatchMs.median / gpu.dispatchMs.median, {}, false);
    if (gpu.gpu.watts > 0 && gpu.dispatchMs.n) {
        // Energy per FMA at full load: W / (FMA/s).
        const double fmaPerSec = double(kThreads) * 8.0 * iters / (gpu.dispatchMs.median * 1e-3);
        rep.value("gpu.pj_per_fma", "pJ", gpu.gpu.watts / fmaPerSec * 1e12, {}, false);
    }

    // Optional soak: continuous GPU load, sampled every 30 s.
    if (ctx.options().soakMinutes > 0) {
        SlowEnergy slow;
        const double cpu0 = slow.read("CPU Energy"), dram0 = slow.read("DRAM0");
        double cpuFirstChange = -1, cpuFirstT = 0, cpuLastT = 0, cpuLast = cpu0, dramLast = dram0, dramFirstChange = -1;
        const double soakMs = ctx.options().soakMinutes * 60000.0;
        const double t0 = nowMs();
        std::vector<double> perf;
        double firstMs = 0, lastMs = 0, firstW = 0, lastW = 0;
        u32 sample = 0;
        while (nowMs() - t0 < soakMs) {
            gs.begin();
            const double s0 = nowMs();
            std::vector<double> ms;
            while (nowMs() - s0 < 30000.0) ms.push_back(dispatch(iters));
            const phosphor::soc::GpuWindow w = gs.end();
            const double med = phosphor::soc::computeStats(ms).median;
            if (sample == 0) { firstMs = med; firstW = w.watts; }
            lastMs = med;
            lastW = w.watts;
            const double c = slow.read("CPU Energy"), d = slow.read("DRAM0");
            const double tnow = nowMs() - t0;
            if (c != cpuLast) {
                if (cpuFirstChange < 0) { cpuFirstChange = c; cpuFirstT = tnow; }
                cpuLast = c;
                cpuLastT = tnow;
            }
            if (d != dramLast) {
                if (dramFirstChange < 0) dramFirstChange = d;
                dramLast = d;
            }
            ctx.log("B-27 soak %.1f min: dispatch %.3f ms, GPU %.1f W, %.0f MHz, thermal %s", tnow / 60000.0, med, w.watts,
                    w.meanMHz, thermalStateName().c_str());
            ++sample;
        }
        rep.value("soak.minutes", "min", ctx.options().soakMinutes);
        rep.value("soak.perf_end_vs_start", "ratio", firstMs / lastMs);
        rep.value("soak.gpu_watts_start", "W", firstW, {}, false);
        rep.value("soak.gpu_watts_end", "W", lastW, {}, false);
        if (cpuFirstChange >= 0 && cpuLastT > cpuFirstT)
            rep.value("soak.cpu_watts_avg", "W", (cpuLast - cpuFirstChange) * 1e-3 / ((cpuLastT - cpuFirstT) * 1e-3), {},
                      false);
        else
            rep.note("soak: CPU Energy did not update twice during the soak (period ~5 min)");
        rep.note("soak thermal end: " + thermalStateName());
    }

    rep.note("thermal " + thermal0 + " -> " + thermalStateName() + "; GPU watts = IOReport 'GPU Energy'; MHz from the "
             "pmgr voltage-states9 table weighted by P-state residency");
    rep.status(Status::Partial, "no CPU/DRAM/package watts per phase (Energy Model counters update every ~5 min, "
                                "powermetrics needs sudo); one Mac; power source recorded in machine.power_source");
    // Negative controls: an idle GPU must use far less power and sit below
    // the top P-state; the loaded GPU must be at the top state.
    const bool idleLow = idle.gpu.watts >= 0 && gpu.gpu.watts > 3.0 * std::max(idle.gpu.watts, 0.1) &&
                         idle.gpu.topStateShare < 0.5;
    const bool loadTop = gpu.gpu.topStateShare >= 0.9;
    char d[256];
    std::snprintf(d, sizeof(d), "idle %.2f W top %.2f vs load %.2f W top %.2f; mixed slowdown %.3f", idle.gpu.watts,
                  idle.gpu.topStateShare, gpu.gpu.watts, gpu.gpu.topStateShare,
                  gpu.dispatchMs.n && mixed.dispatchMs.n ? mixed.dispatchMs.median / gpu.dispatchMs.median : 0.0);
    rep.negative(idleLow && loadTop, d);
}

} // namespace

SOC_BENCH("B-27", "power.clocks", "GPU watts, P-states and MHz: idle / GPU / CPU / mixed load (soak with --soak)",
          benchPower);

} // namespace soc
