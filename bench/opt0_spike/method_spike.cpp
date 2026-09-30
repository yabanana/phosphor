// OPT-0 spike 1: how the SoC suite measures GPU time.  A MEASUREMENT TOOL,
// not engine code (creates resources with the device, compiles MSL from a
// string).
//
// Questions (docs/opt-log.md, "OPT-0 — Spike"):
//   * end-of-dispatch timestamps (Precise, compute encoder) vs the commit
//     feedback GPUStartTime/GPUEndTime for the same work;
//   * fixed cost of a commit that does (almost) nothing;
//   * DVFS: how long until the clock is stable, what intermittent load does,
//     checked in-process with IOReport ("GPU Performance States") and
//     against an xctrace recording;
//   * how many repetitions give CV < 2 %;
//   * does the compiler delete ALU work whose result is not stored
//     (anti-elimination guard);
//   * negative control: 1x/2x/4x work -> 1x/2x/4x time.
//
// Usage: method_spike <case> [args]
//   cases: linear | overhead | dvfs [seconds] | intermittent | cv | elim | all

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <CoreFoundation/CoreFoundation.h>
#include <mach/mach_time.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

extern "C" {
typedef struct IOReportSubscription* IOReportSubscriptionRef;
CFDictionaryRef IOReportCopyChannelsInGroup(CFStringRef, CFStringRef, uint64_t, uint64_t, uint64_t);
IOReportSubscriptionRef IOReportCreateSubscription(void*, CFMutableDictionaryRef, CFMutableDictionaryRef*, uint64_t, CFTypeRef);
CFDictionaryRef IOReportCreateSamples(IOReportSubscriptionRef, CFMutableDictionaryRef, CFTypeRef);
CFDictionaryRef IOReportCreateSamplesDelta(CFDictionaryRef, CFDictionaryRef, CFTypeRef);
CFStringRef IOReportChannelGetChannelName(CFDictionaryRef);
int32_t IOReportStateGetCount(CFDictionaryRef);
int64_t IOReportStateGetResidency(CFDictionaryRef, int32_t);
int64_t IOReportSimpleGetIntegerValue(CFDictionaryRef, int32_t*);
CFStringRef IOReportStateGetNameForIndex(CFDictionaryRef, int32_t);
}

namespace {

using u32 = uint32_t;
using u64 = uint64_t;

constexpr u32 kN = 1u << 20;   // threads per dispatch

NS::String* str(const char* s) { return NS::String::string(s, NS::UTF8StringEncoding); }

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "method_spike: %s\n", msg.c_str());
    std::exit(3);
}

// --- GPU state residency + GPU energy from IOReport (no root) -------------------
struct GpuState {
    IOReportSubscriptionRef sub = nullptr;
    CFMutableDictionaryRef subbed = nullptr;
    CFDictionaryRef last = nullptr;
    GpuState() {
        CFMutableDictionaryRef ch = CFDictionaryCreateMutableCopy(
            nullptr, 0, IOReportCopyChannelsInGroup(CFSTR("GPU Stats"), CFSTR("GPU Performance States"), 0, 0, 0));
        CFDictionaryRef e = IOReportCopyChannelsInGroup(CFSTR("Energy Model"), nullptr, 0, 0, 0);
        // Keep only the channel list of both groups.
        CFMutableArrayRef all = CFArrayCreateMutableCopy(
            nullptr, 0, static_cast<CFArrayRef>(CFDictionaryGetValue(ch, CFSTR("IOReportChannels"))));
        CFArrayAppendArray(all, static_cast<CFArrayRef>(CFDictionaryGetValue(e, CFSTR("IOReportChannels"))),
                           CFRangeMake(0, CFArrayGetCount(static_cast<CFArrayRef>(CFDictionaryGetValue(e, CFSTR("IOReportChannels"))))));
        CFDictionarySetValue(ch, CFSTR("IOReportChannels"), all);
        sub = IOReportCreateSubscription(nullptr, ch, &subbed, 0, nullptr);
        if (sub) last = IOReportCreateSamples(sub, subbed, nullptr);
    }
    // Fraction of time in the top state and GPU watts since the previous call.
    struct Sample { double top = -1, active = -1, watts = -1; int topIndex = -1; };
    Sample delta(double secs) {
        Sample r;
        if (!sub) return r;
        CFDictionaryRef now = IOReportCreateSamples(sub, subbed, nullptr);
        CFDictionaryRef d = IOReportCreateSamplesDelta(last, now, nullptr);
        CFRelease(last);
        last = now;
        auto arr = static_cast<CFArrayRef>(CFDictionaryGetValue(d, CFSTR("IOReportChannels")));
        for (CFIndex i = 0; arr && i < CFArrayGetCount(arr); ++i) {
            auto c = static_cast<CFDictionaryRef>(CFArrayGetValueAtIndex(arr, i));
            char name[128] = {};
            CFStringGetCString(IOReportChannelGetChannelName(c), name, sizeof(name), kCFStringEncodingUTF8);
            if (std::strcmp(name, "GPUPH") == 0) {
                const int32_t n = IOReportStateGetCount(c);
                int64_t tot = 0, off = 0;
                std::vector<int64_t> res(n);
                for (int32_t q = 0; q < n; ++q) { res[q] = IOReportStateGetResidency(c, q); tot += res[q]; }
                off = n ? res[0] : 0;
                int top = 0;
                for (int32_t q = 0; q < n; ++q) if (res[q]) top = q;
                // Highest state index with residency, and its share of the busy (non-OFF) time.
                r.topIndex = top;
                r.top = tot - off > 0 ? double(res[top]) / double(tot - off) : 0;
                r.active = tot ? double(tot - off) / double(tot) : 0;
            } else if (std::strcmp(name, "GPU Energy") == 0) {
                r.watts = double(IOReportSimpleGetIntegerValue(c, nullptr)) * 1e-9 / secs;
            }
        }
        CFRelease(d);
        return r;
    }
};

// Mean DRAM bandwidth of one agent (e.g. "AGX RD") from the PMP "DCS BW"
// histograms (bucket label = GB/s, residency = samples) since the last call.
struct DramBw {
    IOReportSubscriptionRef sub = nullptr;
    CFMutableDictionaryRef subbed = nullptr;
    CFDictionaryRef last = nullptr;
    DramBw() {
        CFMutableDictionaryRef ch = CFDictionaryCreateMutableCopy(
            nullptr, 0, IOReportCopyChannelsInGroup(CFSTR("PMP0"), CFSTR("DCS BW"), 0, 0, 0));
        sub = IOReportCreateSubscription(nullptr, ch, &subbed, 0, nullptr);
        if (sub) last = IOReportCreateSamples(sub, subbed, nullptr);
    }
    double meanGBs(const char* agent) {
        if (!sub) return -1;
        CFDictionaryRef now = IOReportCreateSamples(sub, subbed, nullptr);
        CFDictionaryRef d = IOReportCreateSamplesDelta(last, now, nullptr);
        CFRelease(last);
        last = now;
        double sum = 0, n = 0;
        auto arr = static_cast<CFArrayRef>(CFDictionaryGetValue(d, CFSTR("IOReportChannels")));
        for (CFIndex i = 0; arr && i < CFArrayGetCount(arr); ++i) {
            auto c = static_cast<CFDictionaryRef>(CFArrayGetValueAtIndex(arr, i));
            char name[128] = {};
            CFStringGetCString(IOReportChannelGetChannelName(c), name, sizeof(name), kCFStringEncodingUTF8);
            if (std::strcmp(name, agent) != 0) continue;
            for (int32_t q = 0; q < IOReportStateGetCount(c); ++q) {
                char label[64] = {};
                CFStringGetCString(IOReportStateGetNameForIndex(c, q), label, sizeof(label), kCFStringEncodingUTF8);
                const double gbs = std::atof(label);
                const double cnt = double(IOReportStateGetResidency(c, q));
                sum += gbs * cnt;
                n += cnt;
            }
        }
        CFRelease(d);
        return n > 0 ? sum / n : 0;
    }
};

const char* kShaderSource = R"MSL(
#include <metal_stdlib>
using namespace metal;
struct Params { uint iters; uint zero; float scale; uint pad; };

// 4 independent FMA chains per thread (ILP); result stored (guard: the store
// depends on every iteration and on a runtime value).
kernel void k_fma(device float* out [[buffer(0)]], constant Params& p [[buffer(1)]],
                  uint i [[thread_position_in_grid]]) {
    float a = float(i & 1023u) * 1e-3f, b = a + 1.0f, c = a + 2.0f, d = a + 3.0f;
    for (uint k = 0; k < p.iters; ++k) {
        a = fma(a, 0.999f, 0.001f); b = fma(b, 0.999f, 0.001f);
        c = fma(c, 0.999f, 0.001f); d = fma(d, 0.999f, 0.001f);
    }
    out[i] = (a + b + c + d) * p.scale;
}
// Same loop, result never stored: the compiler may delete the loop.
kernel void k_fma_dead(device float* out [[buffer(0)]], constant Params& p [[buffer(1)]],
                       uint i [[thread_position_in_grid]]) {
    float a = float(i & 1023u) * 1e-3f, b = a + 1.0f, c = a + 2.0f, d = a + 3.0f;
    for (uint k = 0; k < p.iters; ++k) {
        a = fma(a, 0.999f, 0.001f); b = fma(b, 0.999f, 0.001f);
        c = fma(c, 0.999f, 0.001f); d = fma(d, 0.999f, 0.001f);
    }
    out[i] = p.scale;
}
// Result masked by a runtime zero: kept by the compiler, costs one AND.
kernel void k_fma_masked(device uint* out [[buffer(0)]], constant Params& p [[buffer(1)]],
                         uint i [[thread_position_in_grid]]) {
    float a = float(i & 1023u) * 1e-3f, b = a + 1.0f, c = a + 2.0f, d = a + 3.0f;
    for (uint k = 0; k < p.iters; ++k) {
        a = fma(a, 0.999f, 0.001f); b = fma(b, 0.999f, 0.001f);
        c = fma(c, 0.999f, 0.001f); d = fma(d, 0.999f, 0.001f);
    }
    out[i] = as_type<uint>(a + b + c + d) & p.zero;
}
// Pointer chase: one thread follows next[] for p.iters steps (latency).
kernel void k_chase(device const uint* next [[buffer(0)]], constant Params& p [[buffer(1)]],
                    device uint* out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    uint idx = i * 32u;
    for (uint k = 0; k < p.iters; ++k) idx = next[idx];
    out[i] = idx;
}
// Streaming read of a working set of p.zero float4 (mask), p.iters passes;
// each thread reads a strided float4 per pass (coalesced).
kernel void k_stream(device const float4* src [[buffer(0)]], constant Params& p [[buffer(1)]],
                     device float* out [[buffer(2)]], uint i [[thread_position_in_grid]],
                     uint n [[threads_per_grid]]) {
    float4 acc = 0;
    const uint words = p.pad;
    for (uint k = 0; k < p.iters; ++k) {
        // Guard: each pass's base depends on the previous pass (always 0 at
        // run time), so the compiler cannot interchange the loops and reuse
        // the loads across passes.
        const uint base = as_type<uint>(acc.x) & p.zero;
        for (uint j = i; j < words; j += n) acc += src[j + base];
    }
    out[i] = acc.x + acc.y + acc.z + acc.w;
}
kernel void k_anchor(device uint* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i == 0) out[0] = 0;
}
)MSL";

struct Params { u32 iters, zero; float scale; u32 pad; };
constexpr size_t kParamStride = 256;
constexpr u32 kParamSlots = 8192;

struct Rig {
    MTL::Device* dev = nullptr;
    MTL4::CommandQueue* queue = nullptr;
    MTL4::Compiler* compiler = nullptr;
    MTL::Library* lib = nullptr;
    MTL4::CommandAllocator* alloc = nullptr;
    MTL4::CommandBuffer* cb = nullptr;
    MTL::SharedEvent* event = nullptr;
    MTL::ResidencySet* rs = nullptr;
    MTL4::CounterHeap* heap = nullptr;
    MTL4::ArgumentTable* table = nullptr;
    MTL::ComputePipelineState *fma = nullptr, *dead = nullptr, *masked = nullptr, *anchor = nullptr;
    MTL::ComputePipelineState *chase = nullptr, *stream = nullptr;
    MTL::Buffer *out = nullptr, *params = nullptr;
    u32 paramCount = 0;
    u64 eventValue = 0;
    double tickNs = 1.0;

    MTL::ComputePipelineState* pipe(const char* name) {
        NS::Error* err = nullptr;
        MTL4::LibraryFunctionDescriptor* f = MTL4::LibraryFunctionDescriptor::alloc()->init();
        f->setLibrary(lib);
        f->setName(str(name));
        MTL4::ComputePipelineDescriptor* d = MTL4::ComputePipelineDescriptor::alloc()->init();
        d->setComputeFunctionDescriptor(f);
        MTL::ComputePipelineState* p = compiler->newComputePipelineState(d, nullptr, &err);
        d->release();
        f->release();
        if (!p) die(std::string("pipeline ") + name);
        return p;
    }

    Rig() {
        dev = MTL::CreateSystemDefaultDevice();
        if (!dev || !dev->supportsFamily(MTL::GPUFamilyMetal4)) die("no Metal 4 device");
        queue = dev->newMTL4CommandQueue();
        NS::Error* err = nullptr;
        MTL4::CompilerDescriptor* cd = MTL4::CompilerDescriptor::alloc()->init();
        compiler = dev->newCompiler(cd, &err);
        cd->release();
        MTL::CompileOptions* opts = MTL::CompileOptions::alloc()->init();
        opts->setLanguageVersion(MTL::LanguageVersion4_0);
        lib = dev->newLibrary(str(kShaderSource), opts, &err);
        opts->release();
        if (!lib) die(std::string("MSL: ") + (err ? err->localizedDescription()->utf8String() : "?"));
        alloc = dev->newCommandAllocator();
        cb = dev->newCommandBuffer();
        event = dev->newSharedEvent();
        MTL4::CounterHeapDescriptor* hd = MTL4::CounterHeapDescriptor::alloc()->init();
        hd->setType(MTL4::CounterHeapTypeTimestamp);
        hd->setCount(4096);
        heap = dev->newCounterHeap(hd, &err);
        hd->release();
        if (!heap) die("counter heap");
        tickNs = 1e9 / double(dev->queryTimestampFrequency());
        fma = pipe("k_fma");
        dead = pipe("k_fma_dead");
        masked = pipe("k_fma_masked");
        anchor = pipe("k_anchor");
        chase = pipe("k_chase");
        stream = pipe("k_stream");
        out = dev->newBuffer(kN * 4, MTL::ResourceStorageModeShared);
        params = dev->newBuffer(kParamSlots * kParamStride, MTL::ResourceStorageModeShared);
        MTL::ResidencySetDescriptor* rd = MTL::ResidencySetDescriptor::alloc()->init();
        rs = dev->newResidencySet(rd, &err);
        rd->release();
        rs->addAllocation(out);
        rs->addAllocation(params);
        rs->commit();
        queue->addResidencySet(rs);
        MTL4::ArgumentTableDescriptor* ad = MTL4::ArgumentTableDescriptor::alloc()->init();
        ad->setMaxBufferBindCount(4);
        table = dev->newArgumentTable(ad, &err);
        ad->release();
    }

    u64 param(u32 iters, float scale = 1.0f) {
        Params p{iters, 0, scale, 0};
        const u32 slot = paramCount++ % kParamSlots;
        std::memcpy(static_cast<char*>(params->contents()) + slot * kParamStride, &p, sizeof(p));
        return params->gpuAddress() + slot * kParamStride;
    }
    MTL4::ComputeCommandEncoder* begin() {
        alloc->reset();
        cb->beginCommandBuffer(alloc);
        return cb->computeCommandEncoder();
    }
    void dispatch(MTL4::ComputeCommandEncoder* ce, MTL::ComputePipelineState* ps, u32 iters, u32 threads = kN) {
        table->setAddress(out->gpuAddress(), 0);
        table->setAddress(param(iters), 1);
        ce->setComputePipelineState(ps);
        ce->setArgumentTable(table);
        ce->dispatchThreads(MTL::Size::Make(threads, 1, 1), MTL::Size::Make(256, 1, 1));
    }
    // Anchor: 1-thread dispatch + Precise timestamp = start of the timed work.
    void anchorAt(MTL4::ComputeCommandEncoder* ce, u32 index) {
        table->setAddress(out->gpuAddress(), 0);
        ce->setComputePipelineState(anchor);
        ce->setArgumentTable(table);
        ce->dispatchThreads(MTL::Size::Make(1, 1, 1), MTL::Size::Make(1, 1, 1));
        ce->writeTimestamp(MTL4::TimestampGranularityPrecise, heap, index);
    }
    void barrier(MTL4::ComputeCommandEncoder* ce) {
        ce->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
    }
    // Commit, wait; returns feedback GPU ms and CPU commit->event-seen µs.
    struct Done { double fbMs; double cpuUs; };
    Done submit(MTL4::ComputeCommandEncoder* ce) {
        ce->endEncoding();
        cb->endCommandBuffer();
        std::atomic<bool> got{false};
        double gpuMs = 0;
        std::string error;
        MTL4::CommitOptions* o = MTL4::CommitOptions::alloc()->init();
        o->addFeedbackHandler([&](MTL4::CommitFeedback* fb) {
            if (fb->error()) error = fb->error()->localizedDescription()->utf8String();
            gpuMs = (fb->GPUEndTime() - fb->GPUStartTime()) * 1000.0;
            got = true;
        });
        const MTL4::CommandBuffer* bufs[] = {cb};
        const u64 t0 = mach_absolute_time();
        queue->commit(bufs, 1, o);
        o->release();
        queue->signalEvent(event, ++eventValue);
        if (!event->waitUntilSignaledValue(eventValue, 60000)) die("GPU timeout");
        const u64 t1 = mach_absolute_time();
        while (!got) std::this_thread::yield();
        if (!error.empty()) die("GPU error: " + error);
        return {gpuMs, double(t1 - t0) * tickNs * 1e-3};
    }
    std::vector<u64> read(u32 first, u32 count) {
        NS::Data* d = heap->resolveCounterRange(NS::Range::Make(first, count));
        std::vector<u64> v(count, 0);
        if (d) std::memcpy(v.data(), d->bytes(), std::min<size_t>(d->length(), count * 8));
        return v;
    }
    double ms(u64 a, u64 b) const { return (double(b) - double(a)) * tickNs * 1e-6; }

    // One timed dispatch: anchor ts, dispatch, ts.  Returns {ts ms, feedback ms}.
    std::pair<double, double> timed(MTL::ComputePipelineState* ps, u32 iters, u32 threads = kN) {
        MTL4::ComputeCommandEncoder* ce = begin();
        anchorAt(ce, 0);
        barrier(ce);
        dispatch(ce, ps, iters, threads);
        ce->writeTimestamp(MTL4::TimestampGranularityPrecise, heap, 1);
        const Done d = submit(ce);
        const auto v = read(0, 2);
        return {ms(v[0], v[1]), d.fbMs};
    }
};

struct Stats { double median = 0, p10 = 0, p90 = 0, mean = 0, cv = 0, min = 0; };
Stats stats(std::vector<double> v) {
    Stats s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    auto q = [&](double f) { return v[std::min<size_t>(v.size() - 1, size_t(f * double(v.size() - 1) + 0.5))]; };
    s.median = q(0.5); s.p10 = q(0.1); s.p90 = q(0.9); s.min = v.front();
    double sum = 0, sq = 0;
    for (double x : v) sum += x;
    s.mean = sum / double(v.size());
    for (double x : v) sq += (x - s.mean) * (x - s.mean);
    s.cv = v.size() > 1 && s.mean > 0 ? std::sqrt(sq / double(v.size() - 1)) / s.mean : 0;
    return s;
}

// Continuous load until the GPU clock is stable: dispatches of ~2 ms back to
// back; stable = top state for >= 95 % of busy time over the last window and
// the last 10 times within 1 %.  Returns seconds taken.
double warmUp(Rig& r, GpuState& g, u32 iters, double maxSecs, bool verbose) {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<double> recent;
    auto lastSample = t0;
    g.delta(1.0);
    while (true) {
        recent.push_back(r.timed(r.fma, iters).first);
        const auto now = std::chrono::steady_clock::now();
        const double el = std::chrono::duration<double>(now - t0).count();
        if (std::chrono::duration<double>(now - lastSample).count() >= 0.25) {
            const auto s = g.delta(std::chrono::duration<double>(now - lastSample).count());
            lastSample = now;
            std::vector<double> last(recent.end() - std::min<size_t>(10, recent.size()), recent.end());
            const Stats st = stats(last);
            if (verbose)
                std::printf("  t=%.2fs dispatch=%.3f ms cv10=%.2f%% top=P%d %.0f%% active=%.0f%% gpu=%.1f W\n", el,
                            st.median, st.cv * 100, s.topIndex, s.top * 100, s.active * 100, s.watts);
            if (s.top >= 0.95 && recent.size() >= 10 && st.cv < 0.01) return el;
        }
        if (el > maxSecs) return -el;
    }
}

u32 itersFor(Rig& r, double targetMs) {
    // Calibrate: time 1000 iterations at warm clocks.
    const double t = stats({r.timed(r.fma, 1000).first, r.timed(r.fma, 1000).first, r.timed(r.fma, 1000).first}).median;
    return std::max<u32>(1, u32(1000.0 * targetMs / t));
}

void caseLinear(Rig& r, GpuState& g) {
    std::printf("== linear (negative control): warm-up %.2f s\n", warmUp(r, g, 2000, 10, false));
    const u32 base = itersFor(r, 1.0);
    std::printf("| work | iters | ts ms (p50) | ts CV | feedback ms (p50) | fb-ts ms | ratio vs 1x | top state |\n|---|---|---|---|---|---|---|---|\n");
    double one = 0;
    for (u32 mult : {1u, 2u, 4u, 8u}) {
        std::vector<double> ts, fb;
        g.delta(1);
        const auto t0 = std::chrono::steady_clock::now();
        for (u32 k = 0; k < 30; ++k) {
            auto [a, b] = r.timed(r.fma, base * mult);
            ts.push_back(a);
            fb.push_back(b);
        }
        const auto s = g.delta(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        const Stats st = stats(ts), sf = stats(fb);
        if (mult == 1) one = st.median;
        std::printf("| %ux | %u | %.4f | %.2f%% | %.4f | %.4f | %.3f | P%d %.0f%% |\n", mult, base * mult, st.median,
                    st.cv * 100, sf.median, sf.median - st.median, st.median / one, s.topIndex, s.top * 100);
    }
    // Verify the stored result against the CPU (guard).
    r.timed(r.fma, 100);
    const float* o = static_cast<const float*>(r.out->contents());
    u32 bad = 0;
    for (u32 i : {0u, 1u, 511u, 1023u, 4096u, kN - 1}) {
        float a = float(i & 1023u) * 1e-3f, b = a + 1, c = a + 2, d = a + 3;
        for (u32 k = 0; k < 100; ++k) { a = std::fma(a, 0.999f, 0.001f); b = std::fma(b, 0.999f, 0.001f); c = std::fma(c, 0.999f, 0.001f); d = std::fma(d, 0.999f, 0.001f); }
        if (std::fabs(o[i] - (a + b + c + d)) > 1e-4f * std::fabs(a + b + c + d)) ++bad;
    }
    std::printf("result check (100 iters, 6 samples vs CPU): %u wrong\n", bad);
}

void caseOverhead(Rig& r, GpuState& g) {
    warmUp(r, g, 2000, 10, false);
    std::vector<double> fbAnchor, cpuAnchor, fbTiny, tsTiny, cpuTiny;
    for (u32 k = 0; k < 200; ++k) {
        MTL4::ComputeCommandEncoder* ce = r.begin();
        r.anchorAt(ce, 0);
        const auto d = r.submit(ce);
        fbAnchor.push_back(d.fbMs * 1000);
        cpuAnchor.push_back(d.cpuUs);
    }
    for (u32 k = 0; k < 200; ++k) {
        MTL4::ComputeCommandEncoder* ce = r.begin();
        r.anchorAt(ce, 0);
        r.barrier(ce);
        r.dispatch(ce, r.fma, 1, 256);   // one threadgroup, one iteration
        ce->writeTimestamp(MTL4::TimestampGranularityPrecise, r.heap, 1);
        const auto d = r.submit(ce);
        const auto v = r.read(0, 2);
        fbTiny.push_back(d.fbMs * 1000);
        cpuTiny.push_back(d.cpuUs);
        tsTiny.push_back(r.ms(v[0], v[1]) * 1000);
    }
    auto pr = [](const char* n, const Stats& s) {
        std::printf("  %-40s p50 %8.2f  p10 %8.2f  p90 %8.2f  (µs)\n", n, s.median, s.p10, s.p90);
    };
    std::printf("== overhead (200 commits each, warm clocks)\n");
    pr("anchor-only commit: feedback GPU", stats(fbAnchor));
    pr("anchor-only commit: CPU commit->event", stats(cpuAnchor));
    pr("tiny dispatch: feedback GPU", stats(fbTiny));
    pr("tiny dispatch: CPU commit->event", stats(cpuTiny));
    pr("tiny dispatch: anchor->end timestamps", stats(tsTiny));
}

// Cold start: clock ramp under continuous load; then the same kernel with
// idle gaps (intermittent load, the DVFS trap).
void caseDvfs(Rig& r, GpuState& g, double secs) {
    std::printf("== dvfs: idle 3 s, then continuous 2 ms dispatches\n");
    std::this_thread::sleep_for(std::chrono::seconds(3));
    const double t = warmUp(r, g, 2000, secs, true);
    std::printf("stable after %.2f s%s\n", std::fabs(t), t < 0 ? " (NOT reached)" : "");
}

void caseIntermittent(Rig& r, GpuState& g) {
    warmUp(r, g, 2000, 10, false);
    const u32 it = itersFor(r, 0.5);
    std::printf("== intermittent: %u iters (~0.5 ms at warm clocks), 40 dispatches per gap\n", it);
    std::printf("| gap ms | ts ms (p50) | CV | top state | busy | GPU W |\n|---|---|---|---|---|---|\n");
    for (int gap : {0, 2, 8, 16, 33}) {
        std::vector<double> ts;
        g.delta(1);
        const auto t0 = std::chrono::steady_clock::now();
        for (u32 k = 0; k < 40; ++k) {
            ts.push_back(r.timed(r.fma, it).first);
            if (gap) std::this_thread::sleep_for(std::chrono::milliseconds(gap));
        }
        const auto s = g.delta(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        const Stats st = stats(ts);
        std::printf("| %d | %.4f | %.1f%% | P%d %.0f%% | %.0f%% | %.1f |\n", gap, st.median, st.cv * 100, s.topIndex,
                    s.top * 100, s.active * 100, s.watts);
    }
}

// CV as a function of dispatch length and repetitions (warm, back to back).
void caseCv(Rig& r, GpuState& g) {
    warmUp(r, g, 2000, 10, false);
    std::printf("== cv: repetitions needed for CV < 2%% (median of 5 groups)\n");
    std::printf("| dispatch ms | CV n=5 | CV n=10 | CV n=30 | CV n=100 | min/median |\n|---|---|---|---|---|---|\n");
    for (double target : {0.05, 0.2, 1.0, 5.0}) {
        const u32 it = itersFor(r, target);
        std::vector<double> all;
        for (u32 k = 0; k < 100; ++k) all.push_back(r.timed(r.fma, it).first);
        auto cvOf = [&](size_t n) { return stats(std::vector<double>(all.begin(), all.begin() + n)).cv * 100; };
        const Stats s = stats(all);
        std::printf("| %.3f | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %.3f |\n", s.median, cvOf(5), cvOf(10), cvOf(30),
                    cvOf(100), s.min / s.median);
    }
}

void caseElim(Rig& r, GpuState& g) {
    warmUp(r, g, 2000, 10, false);
    const u32 it = itersFor(r, 2.0);
    std::printf("== elim: %u iterations, 20 reps each\n", it);
    for (auto [name, ps] : {std::pair{"stored result (k_fma)", r.fma}, std::pair{"result unused (k_fma_dead)", r.dead},
                            std::pair{"result & runtime zero (k_fma_masked)", r.masked}}) {
        std::vector<double> ts;
        for (u32 k = 0; k < 20; ++k) ts.push_back(r.timed(ps, it).first);
        std::printf("  %-40s %.4f ms\n", name, stats(ts).median);
    }
}

// CV of the MEDIAN across groups (what a suite run reports), per dispatch
// length and group size; groups are separated by 200 ms of other work so
// they sample different moments of the background load.
void caseGroups(Rig& r, GpuState& g) {
    warmUp(r, g, 2000, 10, false);
    std::printf("== groups: CV of group medians (8 groups)\n");
    std::printf("| dispatch ms | reps/group | CV of medians | CV of mins | median of medians | min/median |\n|---|---|---|---|---|---|\n");
    for (double target : {0.1, 0.5, 2.0}) {
        const u32 it = itersFor(r, target);
        for (u32 reps : {5u, 15u, 45u}) {
            std::vector<double> meds, mins;
            for (u32 grp = 0; grp < 8; ++grp) {
                std::vector<double> v;
                for (u32 k = 0; k < reps; ++k) v.push_back(r.timed(r.fma, it).first);
                const Stats s = stats(v);
                meds.push_back(s.median);
                mins.push_back(s.min);
                const auto t0 = std::chrono::steady_clock::now();   // keep clocks up between groups
                while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(200)) r.timed(r.fma, 400);
            }
            const Stats sm = stats(meds), sn = stats(mins);
            std::printf("| %.3f | %u | %.2f%% | %.2f%% | %.4f | %.3f |\n", sm.median, reps, sm.cv * 100, sn.cv * 100,
                        sm.median, sn.median / sm.median);
        }
    }
}

// Latency (single-thread pointer chase over a random cycle of 128-byte
// lines) and streaming bandwidth vs working set.  Steps in either curve are
// cache levels; the last one before DRAM is the SLC.
void caseChase(Rig& r, GpuState& g) {
    warmUp(r, g, 2000, 10, false);
    const size_t maxBytes = size_t(2) << 30;   // 2 GiB
    MTL::Buffer* buf = r.dev->newBuffer(maxBytes, MTL::ResourceStorageModeShared);
    MTL::Buffer* res = r.dev->newBuffer(1 << 20, MTL::ResourceStorageModeShared);
    r.rs->addAllocation(buf);
    r.rs->addAllocation(res);
    r.rs->commit();
    uint32_t* next = static_cast<uint32_t*>(buf->contents());
    std::printf("| working set | chase ns/load | stream GB/s |\n|---|---|---|\n");
    for (size_t ws = size_t(16) << 10; ws <= maxBytes; ws *= 2) {
        // Random cyclic permutation of the lines of the working set.
        const size_t lines = ws / 128;
        std::vector<uint32_t> perm(lines);
        for (size_t i = 0; i < lines; ++i) perm[i] = uint32_t(i);
        uint64_t seed = 88172645463325252ull;
        for (size_t i = lines - 1; i > 0; --i) {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
            std::swap(perm[i], perm[seed % (i + 1)]);
        }
        for (size_t i = 0; i < lines; ++i) next[size_t(perm[i]) * 32] = perm[(i + 1) % lines] * 32;
        const u32 steps = 1u << 16;
        std::vector<double> lat;
        for (u32 rep = 0; rep < 5; ++rep) {
            MTL4::ComputeCommandEncoder* ce = r.begin();
            r.anchorAt(ce, 0);
            r.barrier(ce);
            r.table->setAddress(buf->gpuAddress(), 0);
            r.table->setAddress(r.param(steps), 1);
            r.table->setAddress(res->gpuAddress(), 2);
            ce->setComputePipelineState(r.chase);
            ce->setArgumentTable(r.table);
            ce->dispatchThreads(MTL::Size::Make(1, 1, 1), MTL::Size::Make(1, 1, 1));
            ce->writeTimestamp(MTL4::TimestampGranularityPrecise, r.heap, 1);
            r.submit(ce);
            const auto v = r.read(0, 2);
            lat.push_back(r.ms(v[0], v[1]) * 1e6 / steps);
        }
        // Streaming: ~1 GiB of reads per measurement.
        const u32 passes = u32(std::max<size_t>(1, (size_t(1) << 30) / ws));
        std::vector<double> bw;
        for (u32 rep = 0; rep < 5; ++rep) {
            MTL4::ComputeCommandEncoder* ce = r.begin();
            r.anchorAt(ce, 0);
            r.barrier(ce);
            Params pp{passes, 0, 1.0f, u32(ws / 16)};
            const u32 slot = r.paramCount++ % kParamSlots;
            std::memcpy(static_cast<char*>(r.params->contents()) + slot * kParamStride, &pp, sizeof(pp));
            r.table->setAddress(buf->gpuAddress(), 0);
            r.table->setAddress(r.params->gpuAddress() + slot * kParamStride, 1);
            r.table->setAddress(res->gpuAddress(), 2);
            ce->setComputePipelineState(r.stream);
            ce->setArgumentTable(r.table);
            const u32 threads = u32(std::min<size_t>(ws / 16, size_t(40) * 1024 * 8));
            ce->dispatchThreads(MTL::Size::Make(threads, 1, 1), MTL::Size::Make(256, 1, 1));
            ce->writeTimestamp(MTL4::TimestampGranularityPrecise, r.heap, 1);
            r.submit(ce);
            const auto v = r.read(0, 2);
            bw.push_back(double(ws) * passes / (r.ms(v[0], v[1]) * 1e-3) * 1e-9);
        }
        std::printf("| %zu KiB | %.1f | %.1f |\n", ws >> 10, stats(lat).median, stats(bw).median);
    }
}

// Streaming bandwidth for a fine list of working sets (MiB), with the GPU's
// mean DRAM read bandwidth from IOReport over the same interval.
void caseSlc(Rig& r, GpuState& g, const std::vector<double>& mibs) {
    warmUp(r, g, 2000, 10, false);
    DramBw dram;
    const size_t maxBytes = size_t(1) << 30;
    const bool zeros = std::getenv("ZEROS") != nullptr;
    MTL::Buffer* buf = r.dev->newBuffer(maxBytes, zeros ? MTL::ResourceStorageModePrivate : MTL::ResourceStorageModeShared);
    if (!zeros) {   // incompressible content (xorshift)
        uint64_t x = 0x9E3779B97F4A7C15ull;
        auto* w = static_cast<uint64_t*>(buf->contents());
        for (size_t i = 0; i < maxBytes / 8; ++i) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; w[i] = x & 0x3F3F3F3F3F3F3F3Full; }
    }
    MTL::Buffer* res = r.dev->newBuffer(8 << 20, MTL::ResourceStorageModeShared);
    r.rs->addAllocation(buf);
    r.rs->addAllocation(res);
    r.rs->commit();
    std::printf("| working set MiB | stream GB/s | GPU DRAM read GB/s (IOReport) | DRAM/stream |\n|---|---|---|---|\n");
    for (double mib : mibs) {
        const size_t ws = size_t(mib * 1048576.0) & ~size_t(4095);
        const u32 passes = u32(std::max<size_t>(1, (size_t(4) << 30) / ws));   // ~4 GiB per measurement
        std::vector<double> bw;
        dram.meanGBs(std::getenv("AGENT") ? std::getenv("AGENT") : "AMCC RD");
        for (u32 rep = 0; rep < 5; ++rep) {
            MTL4::ComputeCommandEncoder* ce = r.begin();
            r.anchorAt(ce, 0);
            r.barrier(ce);
            Params pp{passes, 0, 1.0f, u32(ws / 16)};
            const u32 slot = r.paramCount++ % kParamSlots;
            std::memcpy(static_cast<char*>(r.params->contents()) + slot * kParamStride, &pp, sizeof(pp));
            r.table->setAddress(buf->gpuAddress(), 0);
            r.table->setAddress(r.params->gpuAddress() + slot * kParamStride, 1);
            r.table->setAddress(res->gpuAddress(), 2);
            ce->setComputePipelineState(r.stream);
            ce->setArgumentTable(r.table);
            ce->dispatchThreads(MTL::Size::Make(40 * 1024 * 8, 1, 1), MTL::Size::Make(256, 1, 1));
            ce->writeTimestamp(MTL4::TimestampGranularityPrecise, r.heap, 1);
            r.submit(ce);
            const auto v = r.read(0, 2);
            bw.push_back(double(ws) * passes / (r.ms(v[0], v[1]) * 1e-3) * 1e-9);
        }
        const double d = dram.meanGBs(std::getenv("AGENT") ? std::getenv("AGENT") : "AMCC RD");
        const double b = stats(bw).median;
        std::printf("| %.0f | %.1f | %.1f | %.3f |\n", mib, b, d, d / b);
    }
}

} // namespace

int main(int argc, char** argv) {
    const std::string c = argc > 1 ? argv[1] : "all";
    Rig r;
    GpuState g;
    std::printf("device %s, timestamp %.3f ns/tick, IOReport %s\n", r.dev->name()->utf8String(), r.tickNs,
                g.sub ? "ok" : "unavailable");
    if (c == "dvfs") caseDvfs(r, g, argc > 2 ? std::atof(argv[2]) : 20);
    if (c == "linear" || c == "all") caseLinear(r, g);
    if (c == "overhead" || c == "all") caseOverhead(r, g);
    if (c == "intermittent" || c == "all") caseIntermittent(r, g);
    if (c == "cv" || c == "all") caseCv(r, g);
    if (c == "elim" || c == "all") caseElim(r, g);
    if (c == "groups" || c == "all") caseGroups(r, g);
    if (c == "chase") caseChase(r, g);
    if (c == "slc") {
        std::vector<double> m;
        for (int i = 2; i < argc; ++i) m.push_back(std::atof(argv[i]));
        if (m.empty()) m = {8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 96, 112, 128, 160, 192, 256, 512, 1024};
        caseSlc(r, g, m);
    }
    return 0;
}
