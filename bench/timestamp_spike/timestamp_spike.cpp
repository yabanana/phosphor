// F4.1 timestamp spike: a MEASUREMENT TOOL, not engine code.
//
// Questions (docs/opt-log.md, "F4.1 — timestamp MTL4"):
//   * what do MTL4 encoder timestamps measure inside ONE encoder (fused
//     render passes, several compute dispatches), Relaxed vs Precise;
//   * what they cost (GPU time of the command buffer, CPU encode time);
//   * do they stay coherent on the async queue and across a render pass
//     suspended/resumed over several command buffers;
//   * does a synthetic workload of known cost scale linearly (negative
//     control: the measurement must follow the work, not the encoder).
//
// Like bench/barrier_spike it creates resources with the device and compiles
// MSL from a string (both forbidden in the engine).  Usage:
//   timestamp_spike <case> [reps]
// cases: info linear compute3 render2 overhead split async

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <mach/mach_time.h>

#ifdef SPIKE_TRACY
#include <tracy/TracyC.h>
using u8 = uint8_t;
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

using u32 = uint32_t;
using u64 = uint64_t;

constexpr u32 kDim = 2048;            // render target 2048 x 2048
constexpr u32 kN   = 1u << 20;        // compute threads
u32 g_reps = 7;

NS::String* str(const char* s) { return NS::String::string(s, NS::UTF8StringEncoding); }

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "timestamp_spike: %s\n", msg.c_str());
    std::exit(3);
}

const char* kShaderSource = R"MSL(
#include <metal_stdlib>
using namespace metal;
struct Params { uint iters; uint zero; uint width; uint pad; };

inline uint busy(uint seed, uint iters) {
    uint a = seed | 1u;
    for (uint k = 0; k < iters; ++k) a = a * 1664525u + 1013904223u;
    return a;
}
kernel void k_busy(device uint* out [[buffer(0)]], constant Params& p [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
    out[i] = busy(i, p.iters) & p.zero;
}
struct VOut { float4 pos [[position]]; };
vertex VOut v_fullscreen(uint vid [[vertex_id]], constant Params& p [[buffer(0)]]) {
    // p.width != 0: the vertex stage burns p.iters iterations per vertex (3 vertices).
    float2 q = float2(float((vid << 1) & 2u), float(vid & 2u));
    VOut o; o.pos = float4(q * 2.0 - 1.0, 0.5, 1.0);
    return o;
}
// Additive blending keeps every fragment alive (no hidden surface removal).
fragment half4 f_busy(float4 pos [[position]], constant Params& p [[buffer(0)]]) {
    uint a = busy(uint(pos.x) * 4099u + uint(pos.y), p.iters) & p.zero;
    return half4(half(a), 0.0h, 0.0h, 1.0h / 256.0h);
}
)MSL";

struct Params { u32 iters, zero, width, pad; };
constexpr size_t kParamStride = 256;

struct Rig {
    MTL::Device* dev          = nullptr;
    MTL4::CommandQueue* queue = nullptr;
    MTL4::CommandQueue* queue2 = nullptr;
    MTL4::Compiler* compiler  = nullptr;
    MTL::Library* lib         = nullptr;
    std::vector<MTL4::CommandAllocator*> allocators;
    std::vector<MTL4::CommandBuffer*> cbs;
    MTL::SharedEvent* event   = nullptr;
    MTL::ResidencySet* rs     = nullptr;
    MTL4::CounterHeap* heap   = nullptr;
    MTL::ComputePipelineState* kBusy = nullptr;
    MTL::RenderPipelineState* rBusy  = nullptr;
    MTL::Buffer* out    = nullptr;
    MTL::Buffer* params = nullptr;
    MTL::Buffer* resolved = nullptr;
    MTL::Texture* target = nullptr;
    u32 paramCount = 0;
    u64 eventValue = 0;
    double tickNs = 1.0;

    Rig() {
        dev = MTL::CreateSystemDefaultDevice();
        if (!dev || !dev->supportsFamily(MTL::GPUFamilyMetal4)) die("no Metal 4 device");
        queue  = dev->newMTL4CommandQueue();
        queue2 = dev->newMTL4CommandQueue();
        NS::Error* err = nullptr;
        MTL4::CompilerDescriptor* cd = MTL4::CompilerDescriptor::alloc()->init();
        compiler = dev->newCompiler(cd, &err);
        cd->release();
        MTL::CompileOptions* opts = MTL::CompileOptions::alloc()->init();
        opts->setLanguageVersion(MTL::LanguageVersion4_0);
        lib = dev->newLibrary(str(kShaderSource), opts, &err);
        opts->release();
        if (!lib) die(std::string("MSL: ") + (err ? err->localizedDescription()->utf8String() : "?"));
        for (u32 i = 0; i < 8; ++i) {
            allocators.push_back(dev->newCommandAllocator());
            cbs.push_back(dev->newCommandBuffer());
        }
        event = dev->newSharedEvent();

        MTL4::CounterHeapDescriptor* hd = MTL4::CounterHeapDescriptor::alloc()->init();
        hd->setType(MTL4::CounterHeapTypeTimestamp);
        hd->setCount(4096);
        heap = dev->newCounterHeap(hd, &err);
        hd->release();
        if (!heap) die(std::string("newCounterHeap: ") + (err ? err->localizedDescription()->utf8String() : "?"));
        tickNs = 1e9 / double(dev->queryTimestampFrequency());

        auto fn = [&](const char* name) {
            MTL4::LibraryFunctionDescriptor* f = MTL4::LibraryFunctionDescriptor::alloc()->init();
            f->setLibrary(lib);
            f->setName(str(name));
            return f;
        };
        MTL4::LibraryFunctionDescriptor* kf = fn("k_busy");
        MTL4::ComputePipelineDescriptor* cpd = MTL4::ComputePipelineDescriptor::alloc()->init();
        cpd->setComputeFunctionDescriptor(kf);
        kBusy = compiler->newComputePipelineState(cpd, nullptr, &err);
        if (!kBusy) die("k_busy");
        MTL4::LibraryFunctionDescriptor* vf = fn("v_fullscreen");
        MTL4::LibraryFunctionDescriptor* ff = fn("f_busy");
        MTL4::RenderPipelineDescriptor* rpd = MTL4::RenderPipelineDescriptor::alloc()->init();
        rpd->setVertexFunctionDescriptor(vf);
        rpd->setFragmentFunctionDescriptor(ff);
        auto* ca = rpd->colorAttachments()->object(0);
        ca->setPixelFormat(MTL::PixelFormatRGBA16Float);
        ca->setBlendingState(MTL4::BlendStateEnabled);
        ca->setSourceRGBBlendFactor(MTL::BlendFactorOne);
        ca->setDestinationRGBBlendFactor(MTL::BlendFactorOne);
        ca->setSourceAlphaBlendFactor(MTL::BlendFactorOne);
        ca->setDestinationAlphaBlendFactor(MTL::BlendFactorOne);
        rBusy = compiler->newRenderPipelineState(rpd, nullptr, &err);
        if (!rBusy) die(std::string("f_busy: ") + (err ? err->localizedDescription()->utf8String() : "?"));

        out      = dev->newBuffer(kN * 4, MTL::ResourceStorageModePrivate);
        params   = dev->newBuffer(4096 * kParamStride, MTL::ResourceStorageModeShared);
        resolved = dev->newBuffer(4096 * 8, MTL::ResourceStorageModeShared);
        MTL::TextureDescriptor* td =
            MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatRGBA16Float, kDim, kDim, false);
        td->setUsage(MTL::TextureUsageRenderTarget);
        td->setStorageMode(MTL::StorageModePrivate);
        target = dev->newTexture(td);
        MTL::ResidencySetDescriptor* rd = MTL::ResidencySetDescriptor::alloc()->init();
        rs = dev->newResidencySet(rd, &err);
        rd->release();
        rs->addAllocation(out);
        rs->addAllocation(params);
        rs->addAllocation(resolved);
        rs->addAllocation(target);
        rs->commit();
        queue->addResidencySet(rs);
        queue2->addResidencySet(rs);
    }

    u64 param(u32 iters) {
        Params p{iters, 0, 0, 0};
        const u32 slot = paramCount++ % 4096;
        std::memcpy(static_cast<char*>(params->contents()) + slot * kParamStride, &p, sizeof(p));
        return params->gpuAddress() + slot * kParamStride;
    }
    MTL4::ArgumentTable* table() {
        MTL4::ArgumentTableDescriptor* d = MTL4::ArgumentTableDescriptor::alloc()->init();
        d->setMaxBufferBindCount(4);
        NS::Error* err = nullptr;
        MTL4::ArgumentTable* t = dev->newArgumentTable(d, &err);
        d->release();
        return t;   // leaked on purpose (tool)
    }
    MTL4::CommandBuffer* begin(u32 i = 0) {
        allocators[i]->reset();
        cbs[i]->beginCommandBuffer(allocators[i]);
        return cbs[i];
    }
    void dispatch(MTL4::ComputeCommandEncoder* ce, u32 iters) {
        MTL4::ArgumentTable* t = table();
        t->setAddress(out->gpuAddress(), 0);
        t->setAddress(param(iters), 1);
        ce->setComputePipelineState(kBusy);
        ce->setArgumentTable(t);
        ce->dispatchThreads(MTL::Size::Make(kN, 1, 1), MTL::Size::Make(256, 1, 1));
    }
    MTL4::RenderPassDescriptor* passDesc() {
        MTL4::RenderPassDescriptor* pd = MTL4::RenderPassDescriptor::alloc()->init();
        auto* c = pd->colorAttachments()->object(0);
        c->setTexture(target);
        c->setLoadAction(MTL::LoadActionClear);
        c->setStoreAction(MTL::StoreActionStore);
        return pd;
    }
    MTL4::RenderCommandEncoder* renderEncoder(MTL4::CommandBuffer* cb, MTL4::RenderEncoderOptions o = 0) {
        MTL4::RenderPassDescriptor* pd = passDesc();
        MTL4::RenderCommandEncoder* re = cb->renderCommandEncoder(pd, o);
        pd->release();
        return re;
    }
    void draw(MTL4::RenderCommandEncoder* re, u32 iters, bool setState = true) {
        MTL4::ArgumentTable* t = table();
        t->setAddress(param(iters), 0);
        if (setState) {
            re->setRenderPipelineState(rBusy);
            re->setViewport(MTL::Viewport{0.0, 0.0, double(kDim), double(kDim), 0.0, 1.0});
        }
        re->setArgumentTable(t, MTL::RenderStageVertex | MTL::RenderStageFragment);
        re->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(3));
    }

    // Commits cbs [first, first+count) on `q` in one commit; waits; returns
    // the feedback GPU time (ms) of that commit.
    double run(u32 first = 0, u32 count = 1, MTL4::CommandQueue* q = nullptr) {
        q = q ? q : queue;
        for (u32 i = first; i < first + count; ++i) cbs[i]->endCommandBuffer();
        std::atomic<bool> got{false};
        double gpuMs = 0;
        std::string error;
        MTL4::CommitOptions* opts = MTL4::CommitOptions::alloc()->init();
        opts->addFeedbackHandler([&](MTL4::CommitFeedback* fb) {
            if (fb->error()) error = fb->error()->localizedDescription()->utf8String();
            gpuMs = (fb->GPUEndTime() - fb->GPUStartTime()) * 1000.0;
            got = true;
        });
        std::vector<const MTL4::CommandBuffer*> bufs(cbs.begin() + first, cbs.begin() + first + count);
        q->commit(bufs.data(), count, opts);
        opts->release();
        q->signalEvent(event, ++eventValue);
        if (!event->waitUntilSignaledValue(eventValue, 60000)) die("GPU timeout");
        for (int i = 0; i < 2000 && !got; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if (!error.empty()) die("GPU error: " + error);
        return gpuMs;
    }

    std::vector<u64> read(u32 first, u32 count) {
        NS::Data* d = heap->resolveCounterRange(NS::Range::Make(first, count));
        std::vector<u64> v(count, 0);
        if (d) std::memcpy(v.data(), d->bytes(), std::min<size_t>(d->length(), count * 8));
        return v;
    }
    double ms(u64 a, u64 b) const { return (double(b) - double(a)) * tickNs * 1e-6; }
};

double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

const char* granName(MTL4::TimestampGranularity g) {
    return g == MTL4::TimestampGranularityPrecise ? "precise" : "relaxed";
}
const MTL4::TimestampGranularity kGrans[] = {MTL4::TimestampGranularityRelaxed, MTL4::TimestampGranularityPrecise};

// A few seconds of work so the GPU clocks are up before measuring.
void warmup(Rig& r) {
    for (u32 k = 0; k < 20; ++k) {
        MTL4::CommandBuffer* cb = r.begin();
        MTL4::ComputeCommandEncoder* ce = cb->computeCommandEncoder();
        r.dispatch(ce, 8000);
        ce->endEncoding();
        r.run();
    }
}

// --- info: frequency, entry size, correlation, legacy counter sets -------------
void caseInfo(Rig& r) {
    std::printf("device: %s\n", r.dev->name()->utf8String());
    std::printf("timestamp frequency: %llu Hz (%.3f ns/tick)\n",
                static_cast<unsigned long long>(r.dev->queryTimestampFrequency()), r.tickNs);
    std::printf("sizeOfCounterHeapEntry(Timestamp): %lu\n",
                static_cast<unsigned long>(r.dev->sizeOfCounterHeapEntry(MTL4::CounterHeapTypeTimestamp)));
    MTL::Timestamp c0 = 0, g0 = 0, c1 = 0, g1 = 0;
    r.dev->sampleTimestamps(&c0, &g0);
    const u64 m0 = mach_absolute_time();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    r.dev->sampleTimestamps(&c1, &g1);
    const u64 m1 = mach_absolute_time();
    mach_timebase_info_data_t tb{};
    mach_timebase_info(&tb);
    std::printf("sampleTimestamps: cpu %llu gpu %llu | mach_absolute_time %llu (timebase %u/%u)\n",
                static_cast<unsigned long long>(c0), static_cast<unsigned long long>(g0),
                static_cast<unsigned long long>(m0), tb.numer, tb.denom);
    std::printf("over ~200 ms: cpu delta %llu, gpu delta %llu, mach delta %llu\n",
                static_cast<unsigned long long>(c1 - c0), static_cast<unsigned long long>(g1 - g0),
                static_cast<unsigned long long>(m1 - m0));

    // GPU-timeline resolve vs CPU resolve of the same entries.
    MTL4::CommandBuffer* cb = r.begin();
    cb->writeTimestampIntoHeap(r.heap, 0);
    MTL4::ComputeCommandEncoder* ce = cb->computeCommandEncoder();
    r.dispatch(ce, 2000);
    ce->endEncoding();
    cb->writeTimestampIntoHeap(r.heap, 1);
    MTL4::BufferRange br = MTL4::BufferRange::Make(r.resolved->gpuAddress(), 16);
    cb->resolveCounterHeap(r.heap, NS::Range::Make(0, 2), br, nullptr, nullptr);
    const double fbMs = r.run();
    const std::vector<u64> cpu = r.read(0, 2);
    const u64* gpu = static_cast<const u64*>(r.resolved->contents());
    std::printf("command-buffer timestamps: cpu-resolve %llu..%llu (%.3f ms), gpu-resolve %llu..%llu, feedback %.3f ms\n",
                static_cast<unsigned long long>(cpu[0]), static_cast<unsigned long long>(cpu[1]), r.ms(cpu[0], cpu[1]),
                static_cast<unsigned long long>(gpu[0]), static_cast<unsigned long long>(gpu[1]), fbMs);
    std::printf("gpu timestamp vs sampleTimestamps gpu: %+.3f ms after the first sample\n", r.ms(g0, cpu[0]));

    // Legacy (Metal 3) counter sets: MTL4 encoders cannot sample them, listed for F4.4.
    NS::Array* sets = r.dev->counterSets();
    const NS::UInteger n = sets ? sets->count() : 0;
    std::printf("legacy counterSets: %lu\n", static_cast<unsigned long>(n));
    for (NS::UInteger i = 0; i < n; ++i) {
        auto* s = sets->object<MTL::CounterSet>(i);
        std::printf("  set '%s':", s->name()->utf8String());
        NS::Array* cs = s->counters();
        for (NS::UInteger k = 0; k < cs->count(); ++k) std::printf(" %s", cs->object<MTL::Counter>(k)->name()->utf8String());
        std::printf("\n");
    }
    const struct { MTL::CounterSamplingPoint p; const char* n; } pts[] = {
        {MTL::CounterSamplingPointAtStageBoundary, "stage"}, {MTL::CounterSamplingPointAtDrawBoundary, "draw"},
        {MTL::CounterSamplingPointAtDispatchBoundary, "dispatch"}, {MTL::CounterSamplingPointAtTileDispatchBoundary, "tile"},
        {MTL::CounterSamplingPointAtBlitBoundary, "blit"}};
    std::printf("legacy supportsCounterSampling:");
    for (auto& p : pts) std::printf(" %s=%d", p.n, int(r.dev->supportsCounterSampling(p.p)));
    std::printf("\n");
}

// --- linear: one dispatch of known cost between two encoder timestamps --------
// Also command-buffer timestamps around the encoder (cb0 .. cb1).  Entries are
// invalidated before every run (an entry never written reads 0).
void caseLinear(Rig& r) {
    warmup(r);
    std::printf("| iters | granularity | enc ts ms | cb ts ms | cb0->enc0 ms | enc1->cb1 ms | feedback ms | zero entries |\n"
                "|---|---|---|---|---|---|---|---|\n");
    for (u32 iters : {0u, 500u, 1000u, 2000u, 4000u, 8000u, 16000u}) {
        for (auto g : kGrans) {
            std::vector<double> ts, cbts, pre, post, fb;
            u32 zeros = 0;
            for (u32 k = 0; k < g_reps; ++k) {
                r.heap->invalidateCounterRange(NS::Range::Make(0, 4));
                MTL4::CommandBuffer* cb = r.begin();
                cb->writeTimestampIntoHeap(r.heap, 0);
                MTL4::ComputeCommandEncoder* ce = cb->computeCommandEncoder();
                ce->writeTimestamp(g, r.heap, 1);
                r.dispatch(ce, iters);
                ce->writeTimestamp(g, r.heap, 2);
                ce->endEncoding();
                cb->writeTimestampIntoHeap(r.heap, 3);
                fb.push_back(r.run());
                const auto v = r.read(0, 4);
                for (u64 x : v) zeros += x == 0;
                ts.push_back(r.ms(v[1], v[2]));
                cbts.push_back(r.ms(v[0], v[3]));
                pre.push_back(r.ms(v[0], v[1]));
                post.push_back(r.ms(v[2], v[3]));
            }
            std::printf("| %u | %s | %.4f | %.4f | %.4f | %.4f | %.4f | %u |\n", iters, granName(g), median(ts),
                        median(cbts), median(pre), median(post), median(fb), zeros);
        }
    }
}

// --- compute3: three dispatches (costs a,b,c) in ONE compute encoder ------------
void caseCompute3(Rig& r) {
    warmup(r);
    const u32 costs[][3] = {{1000, 3000, 6000}, {6000, 3000, 1000}};
    std::printf("| costs | barriers | granularity | d1 ms | d2 ms | d3 ms | sum | feedback ms |\n|---|---|---|---|---|---|---|---|\n");
    for (auto& c : costs) {
        for (int barriers = 0; barriers < 2; ++barriers) {
            for (auto g : kGrans) {
                std::vector<double> d[3], fb, sum;
                for (u32 k = 0; k < g_reps; ++k) {
                    MTL4::CommandBuffer* cb = r.begin();
                    MTL4::ComputeCommandEncoder* ce = cb->computeCommandEncoder();
                    ce->writeTimestamp(g, r.heap, 0);
                    for (u32 i = 0; i < 3; ++i) {
                        if (barriers && i) ce->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch,
                                                                         MTL4::VisibilityOptionDevice);
                        r.dispatch(ce, c[i]);
                        ce->writeTimestamp(g, r.heap, i + 1);
                    }
                    ce->endEncoding();
                    fb.push_back(r.run());
                    const auto v = r.read(0, 4);
                    for (u32 i = 0; i < 3; ++i) d[i].push_back(r.ms(v[i], v[i + 1]));
                    sum.push_back(r.ms(v[0], v[3]));
                }
                std::printf("| %u/%u/%u | %s | %s | %.3f | %.3f | %.3f | %.3f | %.3f |\n", c[0], c[1], c[2],
                            barriers ? "yes" : "no", granName(g), median(d[0]), median(d[1]), median(d[2]),
                            median(sum), median(fb));
            }
        }
    }
}

// --- render2: two draws ("fused passes") in ONE render encoder ------------------
void caseRender2(Rig& r) {
    warmup(r);
    const u32 costs[][2] = {{100, 400}, {400, 100}, {0, 0}};
    std::printf("| costs | stage | granularity | start->A ms | A ms | B ms | sum | feedback ms | feedback no-ts ms |\n"
                "|---|---|---|---|---|---|---|---|---|\n");
    for (auto& c : costs) {
        // Reference without any timestamp.
        std::vector<double> ref;
        for (u32 k = 0; k < g_reps; ++k) {
            MTL4::CommandBuffer* cb = r.begin();
            MTL4::RenderCommandEncoder* re = r.renderEncoder(cb);
            r.draw(re, c[0]);
            r.draw(re, c[1], false);
            re->endEncoding();
            ref.push_back(r.run());
        }
        for (MTL::RenderStages stage : {MTL::RenderStageFragment, MTL::RenderStageVertex}) {
            for (auto g : kGrans) {
                std::vector<double> pre, a, b, sum, fb;
                for (u32 k = 0; k < g_reps; ++k) {
                    MTL4::CommandBuffer* cb = r.begin();
                    cb->writeTimestampIntoHeap(r.heap, 0);
                    MTL4::RenderCommandEncoder* re = r.renderEncoder(cb);
                    re->writeTimestamp(g, stage, r.heap, 1);
                    r.draw(re, c[0]);
                    re->writeTimestamp(g, stage, r.heap, 2);
                    r.draw(re, c[1], false);
                    re->writeTimestamp(g, stage, r.heap, 3);
                    re->endEncoding();
                    fb.push_back(r.run());
                    const auto v = r.read(0, 4);
                    pre.push_back(r.ms(v[0], v[1]));
                    a.push_back(r.ms(v[1], v[2]));
                    b.push_back(r.ms(v[2], v[3]));
                    sum.push_back(r.ms(v[1], v[3]));
                }
                std::printf("| %u/%u | %s | %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |\n", c[0], c[1],
                            stage == MTL::RenderStageFragment ? "fragment" : "vertex", granName(g), median(pre),
                            median(a), median(b), median(sum), median(fb), median(ref));
            }
        }
    }
}

// --- overhead: 64 cheap draws, a timestamp after each ---------------------------
void caseOverhead(Rig& r) {
    warmup(r);
    const u32 draws = 64;
    std::printf("| mode | iters/draw | feedback ms (median) | CPU encode us | first..last ts ms |\n|---|---|---|---|---|\n");
    for (u32 iters : {0u, 20u}) {
        for (int mode = 0; mode < 4; ++mode) {  // 0 none, 1 relaxed, 2 precise, 3 relaxed only at ends
            std::vector<double> fb, enc, span;
            for (u32 k = 0; k < g_reps * 3; ++k) {
                r.heap->invalidateCounterRange(NS::Range::Make(0, draws + 1));
                const auto t0 = std::chrono::steady_clock::now();
                MTL4::CommandBuffer* cb = r.begin();
                MTL4::RenderCommandEncoder* re = r.renderEncoder(cb);
                const auto g = mode == 2 ? MTL4::TimestampGranularityPrecise : MTL4::TimestampGranularityRelaxed;
                if (mode) re->writeTimestamp(g, MTL::RenderStageFragment, r.heap, 0);
                for (u32 i = 0; i < draws; ++i) {
                    r.draw(re, iters, i == 0);
                    if (mode == 1 || mode == 2) re->writeTimestamp(g, MTL::RenderStageFragment, r.heap, i + 1);
                }
                if (mode == 3) re->writeTimestamp(g, MTL::RenderStageFragment, r.heap, draws);
                re->endEncoding();
                enc.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
                fb.push_back(r.run());
                if (mode) {
                    const auto v = r.read(0, draws + 1);
                    span.push_back(r.ms(v[0], v[draws]));
                    if (k == 0) {
                        std::printf("  [mode %d] unwritten entries:", mode);
                        for (u32 i = 0; i <= draws; ++i) if (v[i] == 0) std::printf(" %u", i);
                        std::vector<u64> d(v.begin(), v.end());
                        std::sort(d.begin(), d.end());
                        std::printf(" | distinct values: %zu\n", size_t(std::unique(d.begin(), d.end()) - d.begin()));
                    }
                }
            }
            const char* names[] = {"none", "relaxed x65", "precise x65", "relaxed x2"};
            std::printf("| %s | %u | %.3f | %.1f | %.3f |\n", names[mode], iters, median(fb), median(enc), median(span));
        }
    }
}

// --- split: render pass suspended/resumed over 3 command buffers ----------------
void caseSplit(Rig& r) {
    warmup(r);
    const u32 costs[3] = {100, 300, 200};
    std::printf("| granularity | chunk1 ms | chunk2 ms | chunk3 ms | sum | feedback ms |\n|---|---|---|---|---|---|\n");
    for (auto g : kGrans) {
        std::vector<double> d[3], sum, fb;
        for (u32 k = 0; k < g_reps; ++k) {
            for (u32 i = 0; i < 3; ++i) {
                MTL4::CommandBuffer* cb = r.begin(i);
                const MTL4::RenderEncoderOptions o = (i ? MTL4::RenderEncoderOptionResuming : 0) |
                                                     (i < 2 ? MTL4::RenderEncoderOptionSuspending : 0);
                MTL4::RenderCommandEncoder* re = r.renderEncoder(cb, o);
                if (i == 0) re->writeTimestamp(g, MTL::RenderStageFragment, r.heap, 0);
                r.draw(re, costs[i]);
                re->writeTimestamp(g, MTL::RenderStageFragment, r.heap, i + 1);
                re->endEncoding();
            }
            fb.push_back(r.run(0, 3));
            const auto v = r.read(0, 4);
            for (u32 i = 0; i < 3; ++i) d[i].push_back(r.ms(v[i], v[i + 1]));
            sum.push_back(r.ms(v[0], v[3]));
        }
        std::printf("| %s | %.3f | %.3f | %.3f | %.3f | %.3f |\n", granName(g), median(d[0]), median(d[1]),
                    median(d[2]), median(sum), median(fb));
    }
}

// --- async: known-cost compute on queue 2 while queue 1 renders ----------------
void caseAsync(Rig& r) {
    warmup(r);
    std::printf("| run | q1 render start..end ms | q2 compute start..end ms | q2 start - q1 start ms | "
                "q2 alone ms |\n|---|---|---|---|---|\n");
    // q2 alone reference.
    std::vector<double> alone;
    for (u32 k = 0; k < g_reps; ++k) {
        MTL4::CommandBuffer* cb = r.begin(1);
        MTL4::ComputeCommandEncoder* ce = cb->computeCommandEncoder();
        ce->writeTimestamp(MTL4::TimestampGranularityRelaxed, r.heap, 10);
        r.dispatch(ce, 4000);
        ce->writeTimestamp(MTL4::TimestampGranularityRelaxed, r.heap, 11);
        ce->endEncoding();
        r.run(1, 1, r.queue2);
        const auto v = r.read(10, 2);
        alone.push_back(r.ms(v[0], v[1]));
    }
    for (u32 k = 0; k < g_reps; ++k) {
        MTL4::CommandBuffer* c0 = r.begin(0);
        MTL4::RenderCommandEncoder* re = r.renderEncoder(c0);
        re->writeTimestamp(MTL4::TimestampGranularityRelaxed, MTL::RenderStageFragment, r.heap, 0);
        r.draw(re, 400);
        re->writeTimestamp(MTL4::TimestampGranularityRelaxed, MTL::RenderStageFragment, r.heap, 1);
        re->endEncoding();
        MTL4::CommandBuffer* c1 = r.begin(1);
        MTL4::ComputeCommandEncoder* ce = c1->computeCommandEncoder();
        ce->writeTimestamp(MTL4::TimestampGranularityRelaxed, r.heap, 2);
        r.dispatch(ce, 4000);
        ce->writeTimestamp(MTL4::TimestampGranularityRelaxed, r.heap, 3);
        ce->endEncoding();
        c0->endCommandBuffer();
        c1->endCommandBuffer();
        const MTL4::CommandBuffer* b0[] = {c0};
        const MTL4::CommandBuffer* b1[] = {c1};
        r.queue->commit(b0, 1);
        r.queue2->commit(b1, 1);
        r.queue->signalEvent(r.event, ++r.eventValue);
        r.queue2->signalEvent(r.event, ++r.eventValue);   // max of both: waits for the later one
        r.event->waitUntilSignaledValue(r.eventValue, 60000);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        const auto v = r.read(0, 4);
        std::printf("| %u | %.3f | %.3f | %+.3f | %.3f |\n", k, r.ms(v[0], v[1]), r.ms(v[2], v[3]),
                    r.ms(v[0], v[2]), median(alone));
    }
}

// --- sequence: compute(a) render(b) compute(c) render(d), serial dependencies ---
// mode 0: no timestamps; 1: command-buffer timestamps before/after every
// encoder; 2: one encoder-end timestamp per encoder (relaxed) + cb start;
// 3: like 2, precise.  Interval of encoder i = t(i) - t(i-1).
void caseSequence(Rig& r) {
    warmup(r);
    const u32 costs[][4] = {{2000, 100, 4000, 300}, {4000, 300, 2000, 100}};
    std::printf("| costs | mode | c1 | r1 | c2 | r2 | sum | feedback ms |\n|---|---|---|---|---|---|---|---|\n");
    for (auto& c : costs) {
        for (int mode = 0; mode < 4; ++mode) {
            std::vector<double> d[4], sum, fb;
            const auto g = mode == 3 ? MTL4::TimestampGranularityPrecise : MTL4::TimestampGranularityRelaxed;
            for (u32 k = 0; k < g_reps; ++k) {
                r.heap->invalidateCounterRange(NS::Range::Make(0, 8));
                MTL4::CommandBuffer* cb = r.begin();
                if (mode) cb->writeTimestampIntoHeap(r.heap, 0);
                for (u32 i = 0; i < 4; ++i) {
                    if (i % 2 == 0) {
                        MTL4::ComputeCommandEncoder* ce = cb->computeCommandEncoder();
                        if (i) ce->barrierAfterQueueStages(MTL::StageFragment, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
                        r.dispatch(ce, c[i]);
                        if (mode >= 2) ce->writeTimestamp(g, r.heap, i + 1);
                        ce->endEncoding();
                    } else {
                        MTL4::RenderCommandEncoder* re = r.renderEncoder(cb);
                        re->barrierAfterQueueStages(MTL::StageDispatch, MTL::StageVertex | MTL::StageFragment,
                                                    MTL4::VisibilityOptionDevice);
                        r.draw(re, c[i]);
                        if (mode >= 2) re->writeTimestamp(g, MTL::RenderStageFragment, r.heap, i + 1);
                        re->endEncoding();
                    }
                    if (mode == 1) cb->writeTimestampIntoHeap(r.heap, i + 1);
                }
                fb.push_back(r.run());
                if (mode) {
                    const auto v = r.read(0, 5);
                    for (u32 i = 0; i < 4; ++i) d[i].push_back(r.ms(v[i], v[i + 1]));
                    sum.push_back(r.ms(v[0], v[4]));
                }
            }
            const char* names[] = {"none", "cb ts", "enc-end relaxed", "enc-end precise"};
            std::printf("| %u/%u/%u/%u | %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |\n", c[0], c[1], c[2], c[3],
                        names[mode], median(d[0]), median(d[1]), median(d[2]), median(d[3]), median(sum), median(fb));
        }
    }
}

// --- limit: how many timestamps per encoder are written -------------------------
void caseLimit(Rig& r) {
    std::printf("| encoder | granularity | timestamps | written | distinct |\n|---|---|---|---|---|\n");
    for (int compute = 0; compute < 2; ++compute) {
        for (auto g : kGrans) {
            for (u32 n : {1u, 2u, 3u, 4u, 5u, 6u, 8u, 16u, 64u}) {
                r.heap->invalidateCounterRange(NS::Range::Make(0, 64));
                MTL4::CommandBuffer* cb = r.begin();
                if (compute) {
                    MTL4::ComputeCommandEncoder* ce = cb->computeCommandEncoder();
                    for (u32 i = 0; i < n; ++i) {
                        r.dispatch(ce, 50);
                        ce->writeTimestamp(g, r.heap, i);
                    }
                    ce->endEncoding();
                } else {
                    MTL4::RenderCommandEncoder* re = r.renderEncoder(cb);
                    for (u32 i = 0; i < n; ++i) {
                        r.draw(re, 5, i == 0);
                        re->writeTimestamp(g, MTL::RenderStageFragment, r.heap, i);
                    }
                    re->endEncoding();
                }
                r.run();
                const auto v = r.read(0, n);
                u32 written = 0;
                for (u64 x : v) written += x != 0;
                std::vector<u64> d(v.begin(), v.end());
                std::sort(d.begin(), d.end());
                std::printf("| %s | %s | %u | %u | %zu |\n", compute ? "compute" : "render", granName(g), n, written,
                            size_t(std::unique(d.begin(), d.end()) - d.begin()));
            }
        }
    }
}

// --- capture: .gputrace of one commit from a CLI executable (F4.3) --------------
void caseCapture(Rig& r, const char* path) {
    MTL::CaptureManager* cm = MTL::CaptureManager::sharedCaptureManager();
    const bool doc = cm->supportsDestination(MTL::CaptureDestinationGPUTraceDocument);
    const bool tools = cm->supportsDestination(MTL::CaptureDestinationDeveloperTools);
    std::printf("supportsDestination: GPUTraceDocument=%d DeveloperTools=%d\n", int(doc), int(tools));
    MTL::CaptureDescriptor* d = MTL::CaptureDescriptor::alloc()->init();
    d->setCaptureObject(r.queue);
    d->setDestination(MTL::CaptureDestinationGPUTraceDocument);
    d->setOutputURL(NS::URL::fileURLWithPath(str(path)));
    NS::Error* err = nullptr;
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = cm->startCapture(d, &err);
    std::printf("startCapture: %d %s\n", int(ok), err ? err->localizedDescription()->utf8String() : "");
    MTL4::CommandBuffer* cb = r.begin();
    MTL4::RenderCommandEncoder* re = r.renderEncoder(cb);
    re->setLabel(str("Capture probe"));
    r.draw(re, 100);
    re->endEncoding();
    MTL4::ComputeCommandEncoder* ce = cb->computeCommandEncoder();
    r.dispatch(ce, 100);
    ce->endEncoding();
    r.run();
    if (ok) cm->stopCapture();
    std::printf("capture took %.1f ms, isCapturing after stop: %d\n",
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(),
                int(cm->isCapturing()));
    d->release();
}

#ifdef SPIKE_TRACY
// --- tracy: manual GPU zones fed with MTL4 timestamps (F4.2) --------------------
// Built by hand in the scratch dir with TracyClient.cpp and -DTRACY_ENABLE; the
// repo build never defines SPIKE_TRACY.  One "frame" = compute(2000)
// render(100) compute(4000) render(300); zone i spans ts[i]..ts[i+1] where
// ts[0] is a command-buffer timestamp and ts[i+1] the end of encoder i.
void caseTracy(Rig& r) {
    for (int i = 0; i < 200 && !TracyCIsConnected; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    std::printf("tracy connected: %d\n", int(TracyCIsConnected));
    warmup(r);
    const u8 ctx = 0;
    ___tracy_emit_gpu_new_context({int64_t(mach_absolute_time()), float(r.tickNs), ctx, 0, 6 /*Metal*/});
    const char ctxName[] = "MTL4 graphics";
    ___tracy_emit_gpu_context_name({ctx, ctxName, uint16_t(sizeof(ctxName) - 1)});
    static const ___tracy_source_location_data locs[4] = {
        {"Compute A", "encode", "timestamp_spike.cpp", 1, 0}, {"Render B", "encode", "timestamp_spike.cpp", 2, 0},
        {"Compute C", "encode", "timestamp_spike.cpp", 3, 0}, {"Render D", "encode", "timestamp_spike.cpp", 4, 0}};
    const u32 costs[4] = {2000, 100, 4000, 300};
    uint16_t query = 0;
    double sums[4] = {};
    const u32 frames = 200;
    for (u32 f = 0; f < frames; ++f) {
        TracyCZoneN(cpuZone, "Encode frame", 1);
        MTL4::CommandBuffer* cb = r.begin();
        cb->writeTimestampIntoHeap(r.heap, 0);
        uint16_t first = query;
        for (u32 i = 0; i < 4; ++i) {
            ___tracy_emit_gpu_zone_begin_serial({reinterpret_cast<uint64_t>(&locs[i]), query++, ctx});
            if (i % 2 == 0) {
                MTL4::ComputeCommandEncoder* ce = cb->computeCommandEncoder();
                if (i) ce->barrierAfterQueueStages(MTL::StageFragment, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
                r.dispatch(ce, costs[i]);
                ce->writeTimestamp(MTL4::TimestampGranularityRelaxed, r.heap, i + 1);
                ce->endEncoding();
            } else {
                MTL4::RenderCommandEncoder* re = r.renderEncoder(cb);
                re->barrierAfterQueueStages(MTL::StageDispatch, MTL::StageVertex | MTL::StageFragment,
                                            MTL4::VisibilityOptionDevice);
                r.draw(re, costs[i]);
                re->writeTimestamp(MTL4::TimestampGranularityRelaxed, MTL::RenderStageFragment, r.heap, i + 1);
                re->endEncoding();
            }
            ___tracy_emit_gpu_zone_end_serial({query++, ctx});
        }
        TracyCZoneEnd(cpuZone);
        r.run();
        const auto v = r.read(0, 5);
        for (u32 i = 0; i < 4; ++i) {
            ___tracy_emit_gpu_time_serial({int64_t(v[i]), uint16_t(first + 2 * i), ctx});
            ___tracy_emit_gpu_time_serial({int64_t(v[i + 1]), uint16_t(first + 2 * i + 1), ctx});
            sums[i] += r.ms(v[i], v[i + 1]);
        }
        TracyCFrameMark;
    }
    std::printf("timestamp means (ms): A %.4f B %.4f C %.4f D %.4f over %u frames\n", sums[0] / frames,
                sums[1] / frames, sums[2] / frames, sums[3] / frames, frames);
    std::this_thread::sleep_for(std::chrono::seconds(2));   // let tracy-capture drain
}
#endif

// --- async2: both queues, cb start + encoder-end timestamps -----------------------
void caseAsync2(Rig& r) {
    warmup(r);
    std::printf("| run | q1 render ms | q2 compute ms | q2 start - q1 start ms | q2 alone ms |\n|---|---|---|---|---|\n");
    std::vector<double> alone;
    for (u32 k = 0; k < g_reps; ++k) {
        MTL4::CommandBuffer* cb = r.begin(1);
        cb->writeTimestampIntoHeap(r.heap, 10);
        MTL4::ComputeCommandEncoder* ce = cb->computeCommandEncoder();
        r.dispatch(ce, 4000);
        ce->writeTimestamp(MTL4::TimestampGranularityRelaxed, r.heap, 11);
        ce->endEncoding();
        r.run(1, 1, r.queue2);
        const auto v = r.read(10, 2);
        alone.push_back(r.ms(v[0], v[1]));
    }
    for (u32 k = 0; k < g_reps; ++k) {
        r.heap->invalidateCounterRange(NS::Range::Make(0, 4));
        MTL4::CommandBuffer* c0 = r.begin(0);
        c0->writeTimestampIntoHeap(r.heap, 0);
        MTL4::RenderCommandEncoder* re = r.renderEncoder(c0);
        r.draw(re, 400);
        re->writeTimestamp(MTL4::TimestampGranularityRelaxed, MTL::RenderStageFragment, r.heap, 1);
        re->endEncoding();
        MTL4::CommandBuffer* c1 = r.begin(1);
        c1->writeTimestampIntoHeap(r.heap, 2);
        MTL4::ComputeCommandEncoder* ce = c1->computeCommandEncoder();
        r.dispatch(ce, 4000);
        ce->writeTimestamp(MTL4::TimestampGranularityRelaxed, r.heap, 3);
        ce->endEncoding();
        c0->endCommandBuffer();
        c1->endCommandBuffer();
        const MTL4::CommandBuffer* b0[] = {c0};
        const MTL4::CommandBuffer* b1[] = {c1};
        r.queue->commit(b0, 1);
        r.queue2->commit(b1, 1);
        r.queue->signalEvent(r.event, ++r.eventValue);
        r.event->waitUntilSignaledValue(r.eventValue, 60000);
        r.queue2->signalEvent(r.event, ++r.eventValue);
        r.event->waitUntilSignaledValue(r.eventValue, 60000);
        const auto v = r.read(0, 4);
        std::printf("| %u | %.3f | %.3f | %+.3f | %.3f |\n", k, r.ms(v[0], v[1]), r.ms(v[2], v[3]), r.ms(v[0], v[2]),
                    median(alone));
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) die("usage: timestamp_spike <info|linear|compute3|render2|overhead|split|async> [reps]");
    const std::string c = argv[1];
    if (argc > 2) g_reps = static_cast<u32>(std::atoi(argv[2]));
    // capture: "setenv" as 4th argument enables the capture layer in-process,
    // before the device exists.
    if (c == "capture" && argc > 4 && std::string(argv[4]) == "setenv") setenv("MTL_CAPTURE_ENABLED", "1", 1);
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    Rig r;
    if (c == "info") caseInfo(r);
    else if (c == "linear") caseLinear(r);
    else if (c == "compute3") caseCompute3(r);
    else if (c == "render2") caseRender2(r);
    else if (c == "overhead") caseOverhead(r);
    else if (c == "split") caseSplit(r);
    else if (c == "async") caseAsync(r);
    else if (c == "sequence") caseSequence(r);
    else if (c == "async2") caseAsync2(r);
    else if (c == "limit") caseLimit(r);
#ifdef SPIKE_TRACY
    else if (c == "tracy") caseTracy(r);
#endif
    else if (c == "capture") caseCapture(r, argc > 3 ? argv[3] : "spike.gputrace");
    else die("unknown case " + c);
    pool->release();
    return 0;
}
