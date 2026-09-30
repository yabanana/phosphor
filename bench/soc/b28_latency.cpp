// B-28: commit and event latency of a Metal 4 command buffer as seen by the
// CPU (S-SYNC-4 of docs/APPLE_SOC_PLAYBOOK.md).
//   (a) GPU cost of an (almost) empty commit: feedback GPUEnd - GPUStart of a
//       command buffer holding one 1-thread dispatch (an encoder without a
//       dispatch is dropped by the driver);
//   (b) CPU time from queue->commit to the SharedEvent signalled after it,
//       observed by the CPU (waitUntilSignaledValue), empty work;
//   (c) the same with a known ~1 ms GPU workload: the latency must grow by the
//       workload time (negative control);
//   (d) three ways to observe the same event: spin on signaledValue(),
//       waitUntilSignaledValue(), MTLSharedEventListener notification.
// Methods and workloads are interleaved sample by sample so slow drifts hit
// all of them alike.
//
// Metrics (us): commit.gpu_us, commit.cpu_us (duration of the commit +
// signalEvent calls), commit_to_cpu.us (= wait method, empty work; p90/p99 as
// commit_to_cpu.p90_us / .p99_us), event.{spin,wait,listener}.us (+ p90/p99)
// for empty work, the same with the ~1 ms workload as
// event.<m>.work_1ms.us and the excess over the workload's GPU time as
// event.<m>.over_work.us.

#include "harness.h"

#include <dispatch/dispatch.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>

namespace soc {
namespace {

struct WorkParams {
    u32 iters, pad[3];
};
constexpr u32 kThreads = 1u << 20;

enum Method { Wait = 0, Spin = 1, Listener = 2, kMethods = 3 };
const char* methodName(int m) { return m == Wait ? "wait" : m == Spin ? "spin" : "listener"; }

float fmaChain(float v, u32 iters) {
    for (u32 k = 0; k < iters; ++k) v = std::fmaf(v, 0.999f, 0.001f);
    return v;
}
u32 bits(float f) {
    u32 b;
    std::memcpy(&b, &f, 4);
    return b;
}
// Bit-exact model of b28_work (MathModeSafe library).
u32 workReference(u32 i, u32 iters) {
    const float a = float(i & 1023u) * 1e-3f;
    return bits(fmaChain(a, iters)) ^ (bits(fmaChain(a + 1.0f, iters)) * 3u) ^ (bits(fmaChain(a + 2.0f, iters)) * 5u) ^
           (bits(fmaChain(a + 3.0f, iters)) * 7u);
}

struct Rig {
    Context& ctx;
    MTL::ComputePipelineState* work = nullptr;
    MTL::Buffer* params = nullptr;
    MTL::Buffer* out = nullptr;
    MTL::SharedEvent* event = nullptr;
    MTL::SharedEventListener* listener = nullptr;
    u64 value = 0;
    std::atomic<u64> notified{0};
    std::atomic<double> notifiedAt{0};
    explicit Rig(Context& c) : ctx(c) {}
};

MTL4::CommandBuffer* encode(Rig& r, u32 iters) {
    MTL4::CommandBuffer* c = r.ctx.beginCommands();
    MTL4::ComputeCommandEncoder* e = c->computeCommandEncoder();
    r.ctx.anchorDispatch(e);
    if (iters) {
        const WorkParams p{iters, {0, 0, 0}};
        std::memcpy(r.params->contents(), &p, sizeof(p));
        std::memset(r.out->contents(), 0, size_t(kThreads) * 4); // stale results must not pass the check
        r.ctx.table()->setAddress(r.out->gpuAddress(), 0);
        r.ctx.table()->setAddress(r.params->gpuAddress(), 1);
        e->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
        e->setComputePipelineState(r.work);
        e->setArgumentTable(r.ctx.table());
        e->dispatchThreads(MTL::Size::Make(kThreads, 1, 1), MTL::Size::Make(256, 1, 1));
    }
    e->endEncoding();
    c->endCommandBuffer();
    return c;
}

struct Sample {
    double latencyUs = 0; // start of commit() -> observed by the CPU
    double commitUs  = 0; // duration of the commit() + signalEvent() calls
};

Sample once(Rig& r, u32 iters, Method m) {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    MTL4::CommandBuffer* c = encode(r, iters);
    const u64 v = ++r.value;
    if (m == Listener) {
        Rig* rp = &r;
        r.event->notifyListener(r.listener, v, [rp](MTL::SharedEvent*, uint64_t value) {
            rp->notifiedAt.store(nowMs());
            rp->notified.store(value);
        });
    }
    const MTL4::CommandBuffer* bufs[] = {c};
    const double t0 = nowMs();
    r.ctx.queue()->commit(bufs, 1);
    r.ctx.queue()->signalEvent(r.event, v);
    const double t1 = nowMs();
    double tEnd = 0;
    if (m == Wait) {
        if (!r.event->waitUntilSignaledValue(v, 10000)) throw BenchError("B-28: event timeout");
        tEnd = nowMs();
    } else if (m == Spin) {
        while (r.event->signaledValue() < v) {
            if (nowMs() - t0 > 10000) throw BenchError("B-28: event timeout (spin)");
        }
        tEnd = nowMs();
    } else {
        while (r.notified.load(std::memory_order_acquire) < v) {
            if (nowMs() - t0 > 10000) throw BenchError("B-28: listener never fired");
        }
        tEnd = r.notifiedAt.load();
    }
    pool->release();
    return {(tEnd - t0) * 1e3, (t1 - t0) * 1e3};
}

double workSpanMs(Rig& r, u32 iters) {
    const WorkParams p{iters, {0, 0, 0}};
    std::memcpy(r.params->contents(), &p, sizeof(p));
    ComputeTimer t(r.ctx);
    MTL4::ComputeCommandEncoder* e = t.begin();
    r.ctx.table()->setAddress(r.out->gpuAddress(), 0);
    r.ctx.table()->setAddress(r.params->gpuAddress(), 1);
    e->setComputePipelineState(r.work);
    e->setArgumentTable(r.ctx.table());
    e->dispatchThreads(MTL::Size::Make(kThreads, 1, 1), MTL::Size::Make(256, 1, 1));
    t.lap();
    return t.finish()[0];
}

// Median of `v` -> Stats, p99 by nearest rank.
double quantile(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    const size_t k = std::min(v.size() - 1, size_t(std::ceil(q * double(v.size()))) - (v.empty() ? 0 : 1));
    return v[k];
}

void addLatency(Report& rep, const std::string& name, const std::vector<double>& us, std::map<std::string, double> params,
                const std::string& p90Name = "", const std::string& p99Name = "") {
    const Stats s = phosphor::soc::computeStats(us);
    params["n"] = double(us.size());
    rep.metric(name, "us", s, params, false);
    if (!p90Name.empty()) rep.value(p90Name, "us", s.p90, params, false);
    if (!p99Name.empty()) rep.value(p99Name, "us", quantile(us, 0.99), params, false);
}


// Under the validation layers (--validate) GPU times are distorted by the instrumentation: the timing controls are
// then informational (detail says so); the correctness controls stay enforced.
bool timingControlsEnforced() { return !std::getenv("MTL_SHADER_VALIDATION") && !std::getenv("MTL_DEBUG_LAYER"); }
void benchLatency(Context& ctx, Report& rep) {
    MTL::Library* lib = ctx.library("b28_latency.metal", /*fastMath=*/false);
    Rig r(ctx);
    r.work   = ctx.compute(lib, "b28_work");
    r.params = ctx.buffer(256);
    r.out    = ctx.buffer(size_t(kThreads) * 4);
    r.event  = ctx.device()->newSharedEvent();
    ctx.keep(r.event);
    dispatch_queue_attr_t attr =
        dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, 0);
    dispatch_queue_t q = dispatch_queue_create("soc.b28.listener", attr);
    r.listener = MTL::SharedEventListener::alloc()->init(q);
    ctx.keep(r.listener);
    dispatch_release(q); // the listener holds its own reference

    const u32 n = ctx.quick() ? 100 : 1000;
    ctx.keepWarm(50);

    // --- (a) GPU cost of an (almost) empty commit ---------------------------------
    const Stats gpu = ctx.measure(
        [&] {
            MTL4::CommandBuffer* c = ctx.beginCommands();
            MTL4::ComputeCommandEncoder* e = c->computeCommandEncoder();
            ctx.anchorDispatch(e);
            e->endEncoding();
            return ctx.submit() * 1e3;
        },
        n / 2);
    rep.metric("commit.gpu_us", "us", gpu, {{"n", double(gpu.n)}}, false);

    // --- known workload of ~1 ms ---------------------------------------------------
    const u32 probe = 256;
    const double tp = std::max(1e-3, ctx.measure([&] { return workSpanMs(r, probe); }, 3).median);
    const u32 iters = std::max<u32>(16, u32(double(probe) * 1.0 / tp));
    const Stats workSpan = ctx.measure([&] { return workSpanMs(r, iters); }, 15);
    const double workUs = workSpan.median * 1e3;
    std::vector<u32> checkIdx = {0, 1, 12345, kThreads - 1};
    std::vector<u32> want;
    for (u32 i : checkIdx) want.push_back(workReference(i, iters));
    u32 wrong = 0;

    // --- (b)(c)(d) interleaved samples -----------------------------------------------
    std::vector<double> lat[kMethods][2], commitUs[2];
    ctx.keepWarm(50);
    for (u32 s = 0; s < n; ++s) {
        for (int k = 0; k < 2 * kMethods; ++k) {
            const int idx = (k + int(s)) % (2 * kMethods); // rotate the order
            const Method m = Method(idx % kMethods);
            const int w = idx / kMethods;
            const Sample sm = once(r, w ? iters : 0, m);
            lat[m][w].push_back(sm.latencyUs);
            if (m == Wait) commitUs[w].push_back(sm.commitUs);
            if (w) {
                const u32* o = static_cast<const u32*>(r.out->contents());
                for (size_t q = 0; q < checkIdx.size(); ++q)
                    if (o[checkIdx[q]] != want[q]) ++wrong;
            }
        }
        if (s % 40 == 39) ctx.keepWarm(10);
    }
    addLatency(rep, "commit_to_cpu.us", lat[Wait][0], {{"work_us", 0}}, "commit_to_cpu.p90_us", "commit_to_cpu.p99_us");
    addLatency(rep, "commit.cpu_us", commitUs[0], {{"work_us", 0}});
    addLatency(rep, "commit_to_cpu.work_1ms.us", lat[Wait][1], {{"work_us", workUs}, {"iters", double(iters)}},
               "commit_to_cpu.work_1ms.p90_us", "commit_to_cpu.work_1ms.p99_us");
    for (int m = 0; m < kMethods; ++m) {
        const std::string base = std::string("event.") + methodName(m);
        addLatency(rep, base + ".us", lat[m][0], {{"work_us", 0}}, base + ".p90_us", base + ".p99_us");
        addLatency(rep, base + ".work_1ms.us", lat[m][1], {{"work_us", workUs}}, base + ".work_1ms.p90_us");
        const Stats over = phosphor::soc::computeStats(lat[m][1]);
        rep.value(base + ".over_work.us", "us", over.median - workUs, {{"work_us", workUs}}, false);
    }

    // --- negative controls ---------------------------------------------------------------
    auto deltaOf = [&] {
        return phosphor::soc::computeStats(lat[Wait][1]).median - phosphor::soc::computeStats(lat[Wait][0]).median;
    };
    double delta = deltaOf();
    const double tol = std::max(200.0, 0.3 * workUs);
    if (std::fabs(delta - workUs) > tol) {
        // Another GPU client stretched or shrank the work during the run: measure the two wait series again (reported
        // metrics keep the first batch).
        std::vector<double> a, b;
        ctx.keepWarm(50);
        for (u32 s = 0; s < 200; ++s) {
            a.push_back(once(r, 0, Wait).latencyUs);
            b.push_back(once(r, iters, Wait).latencyUs);
        }
        delta = phosphor::soc::computeStats(b).median - phosphor::soc::computeStats(a).median;
        rep.note("delta control re-measured once (first batch " + std::to_string(int(deltaOf())) + " us)");
    }
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "latency(1 ms work) - latency(empty) = %.0f us vs GPU span of the work %.0f us (tolerance %.0f us)", delta,
                  workUs, tol);
    rep.negative(std::fabs(delta - workUs) <= tol || !timingControlsEnforced(), std::string(timingControlsEnforced() ? "" : "[timing not enforced under validation] ") + buf);
    rep.negative(wrong == 0, wrong == 0 ? "workload results match the CPU model (all samples)"
                                        : std::to_string(wrong) + " workload results differ from the CPU model");
    if (wrong) rep.status(Status::Failed, "wrong results");
    rep.note("commit.gpu_us = feedback GPUEnd-GPUStart of a 1-thread dispatch (anchor kernel); latencies = commit() start to "
             "observation, event signalled by queue->signalEvent after the buffer; listener time taken inside the callback "
             "(QoS user-interactive queue) and the main thread spins on a flag");
}

} // namespace

SOC_BENCH("B-28", "latency.commit_event", "Commit and SharedEvent latency seen by the CPU, empty and 1 ms of GPU work",
          benchLatency);

} // namespace soc
