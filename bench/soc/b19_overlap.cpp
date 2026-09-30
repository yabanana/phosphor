// B-19: overlap between passes and between queues.
// Serves S-TBDR-6 and S-SYNC-2 of docs/APPLE_SOC_PLAYBOOK.md.
//
// (a) render -> render: pass A is fragment-heavy (full-screen triangle on a
//     2048x2048 RGBA8 target), pass B vertex-heavy (65536 grid triangles, an
//     integer chain per vertex, trivial fragment, its own 2048x2048 target).
//     Measured with a CommandTimer span: A alone, B alone, A then B without any
//     dependency (the GPU may run B's vertex work during A's fragment work),
//     A then B with a forced dependency (queue barrier Fragment -> Vertex on B's
//     encoder).  overlap_ratio = time(no dep) / (time(A) + time(B)): 1 = no
//     overlap, 0.5 = perfect overlap of two equal passes.  B's cost is scaled
//     x0.5 / x1 / x2 (headline metric: x1).
// (b) compute on a second MTL4 queue against a render pass on the main queue:
//     both command buffers wait on one shared event (released by the CPU),
//     GPU timestamps (same time domain on both queues) give the span from the
//     first start to the last end; serial = the two alone; dependent = the
//     compute queue waits for the render queue's event (a forced dependency).
//     Render pass kinds: alu (heavy fragment) and bandwidth (RGBA32F 4096x2048,
//     cheap shader).
//
// Controls: the forced dependency costs the sum of the parts (+-10%, both in
// (a) and (b)); ratios cannot be below the ideal overlap; every output is
// checked on the CPU (A/B pixels, compute results) after the runs that overlap.

#include "harness.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace soc {
namespace {

constexpr u32 kDim = 2048;
constexpr u32 kTris = 65536;
constexpr u32 kThreads = 1u << 20;
constexpr size_t kSlot = 256;

struct Params {
    u32 iters, zero, count, pad;
};

u32 hash(u32 x, u32 y, u32 c) {
    u32 h = x * 73856093u ^ y * 19349663u ^ (c * 83492791u + 1u);
    h ^= h >> 15;
    h *= 2246822519u;
    h ^= h >> 13;
    return h;
}
u32 chain(u32 seed, u32 iters) {
    u32 a = seed | 1u;
    for (u32 k = 0; k < iters; ++k) a = a * 1664525u + 1013904223u;
    return a;
}

struct Rig {
    MTL::RenderPipelineState* psoA = nullptr; // RGBA8 heavy fragment
    MTL::RenderPipelineState* psoF = nullptr; // RGBA32F bandwidth
    MTL::RenderPipelineState* psoB = nullptr; // heavy vertex
    MTL::ComputePipelineState* psoC = nullptr;
    MTL::Texture* tA = nullptr;
    MTL::Texture* tF = nullptr;
    MTL::Texture* tB = nullptr;
    MTL::Buffer* params = nullptr; // slots: 0 A, 1 B, 2 C
    MTL::Buffer* out    = nullptr;
    u32 fW = 4096, fH = 2048;
};

void setParams(const Rig& r, u32 slot, u32 iters) {
    const Params p{iters, 0, 0, 0};
    std::memcpy(static_cast<u8*>(r.params->contents()) + slot * kSlot, &p, sizeof(p));
}

MTL4::RenderCommandEncoder* beginPass(MTL4::CommandBuffer* cmd, MTL::Texture* t) {
    MTL4::RenderPassDescriptor* pd = MTL4::RenderPassDescriptor::alloc()->init();
    auto* c = pd->colorAttachments()->object(0);
    c->setTexture(t);
    c->setLoadAction(MTL::LoadActionClear);
    c->setStoreAction(MTL::StoreActionStore);
    c->setClearColor(MTL::ClearColor::Make(0, 0, 0, 0));
    MTL4::RenderCommandEncoder* re = cmd->renderCommandEncoder(pd);
    pd->release();
    return re;
}

void encodeA(Context& ctx, const Rig& r, MTL4::CommandBuffer* cmd, bool bandwidth) {
    MTL4::RenderCommandEncoder* re = beginPass(cmd, bandwidth ? r.tF : r.tA);
    re->setRenderPipelineState(bandwidth ? r.psoF : r.psoA);
    ctx.table()->setAddress(r.params->gpuAddress() + 0 * kSlot, 0);
    re->setArgumentTable(ctx.table(), MTL::RenderStageVertex | MTL::RenderStageFragment);
    re->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(3));
    re->endEncoding();
}

void encodeB(Context& ctx, const Rig& r, MTL4::CommandBuffer* cmd, bool dependent) {
    MTL4::RenderCommandEncoder* re = beginPass(cmd, r.tB);
    if (dependent) re->barrierAfterQueueStages(MTL::StageFragment, MTL::StageVertex, MTL4::VisibilityOptionDevice);
    re->setRenderPipelineState(r.psoB);
    ctx.table()->setAddress(r.params->gpuAddress() + 1 * kSlot, 0);
    re->setArgumentTable(ctx.table(), MTL::RenderStageVertex | MTL::RenderStageFragment);
    re->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(3) * kTris);
    re->endEncoding();
}

enum class Mode { AOnly, BOnly, NoDep, Dep };

double timeRender(Context& ctx, const Rig& r, Mode m) {
    CommandTimer t(ctx);
    MTL4::CommandBuffer* cmd = t.begin();
    if (m == Mode::AOnly || m == Mode::NoDep || m == Mode::Dep) encodeA(ctx, r, cmd, false);
    if (m == Mode::BOnly || m == Mode::NoDep || m == Mode::Dep) encodeB(ctx, r, cmd, m == Mode::Dep);
    return t.finish() - ctx.emptySpanMs();
}

// Calibrate `iters` of slot `slot` so that `run()` takes ~target ms: two probes give slope and intercept,
// then one refinement step from the measured time of the result.
u32 calibrate(Context& ctx, const Rig& r, u32 slot, double target, u32 probe, const std::function<double()>& run) {
    auto at = [&](u32 it) {
        setParams(r, slot, it);
        ctx.keepWarm(20);
        return ctx.measure(run, 9).median;
    };
    const double t1 = at(probe), t2 = at(probe * 3);
    const double slope = (t2 - t1) / double(probe * 2);
    if (slope <= 1e-9) return probe;
    const double intercept = std::max(0.0, t1 - probe * slope);
    u32 it = std::clamp<u32>(u32(std::max(1.0, (target - intercept) / slope)), 1, 400000);
    const double t3 = at(it);
    const double slope2 = std::max(1e-9, (t3 - intercept) / double(it));
    it = std::clamp<u32>(u32(std::max(1.0, (target - intercept) / slope2)), 1, 400000);
    return it;
}

std::string num(double v, size_t n = 6) { return std::to_string(v).substr(0, n); }

// (a) correctness: A and B after an un-synchronised pair, blit back and compare.
u32 verifyRender(Context& ctx, const Rig& r) {
    MTL::Buffer* ra = ctx.buffer(64 * 64 * 4);
    MTL::Buffer* rbuf = ctx.buffer(size_t(kDim) * kDim * 4);
    std::memset(ra->contents(), 0, ra->length());
    std::memset(rbuf->contents(), 0, rbuf->length());
    MTL4::CommandBuffer* cmd = ctx.beginCommands();
    encodeA(ctx, r, cmd, false);
    encodeB(ctx, r, cmd, false);
    MTL4::ComputeCommandEncoder* ce = cmd->computeCommandEncoder();
    ce->barrierAfterQueueStages(MTL::StageFragment, MTL::StageBlit, MTL4::VisibilityOptionDevice);
    ce->copyFromTexture(r.tA, 0, 0, MTL::Origin::Make(1000, 500, 0), MTL::Size::Make(64, 64, 1), ra, 0, 64 * 4, 64 * 64 * 4);
    ce->copyFromTexture(r.tB, 0, 0, MTL::Origin::Make(0, 0, 0), MTL::Size::Make(kDim, kDim, 1), rbuf, 0, kDim * 4,
                        size_t(kDim) * kDim * 4);
    ce->endEncoding();
    ctx.submit();
    u32 wrong = 0;
    const u8* a = static_cast<const u8*>(ra->contents());
    for (u32 y = 0; y < 64; ++y)
        for (u32 x = 0; x < 64; ++x)
            for (u32 c = 0; c < 4; ++c)
                if (a[(y * 64 + x) * 4 + c] != (hash(1000 + x, 500 + y, c) & 255u)) ++wrong;
    const u8* b = static_cast<const u8*>(rbuf->contents());
    for (u32 t = 0; t < kTris; ++t) {
        const u32 x = 4 + (t & 255u) * 8, y = 4 + (t >> 8) * 8;
        const u32 h = hash(t, 7u, 9u);
        const u8* p = b + (size_t(y) * kDim + x) * 4;
        if (p[0] != (h & 255u) || p[1] != ((h >> 8) & 255u) || p[2] != ((h >> 16) & 255u) || p[3] != 255) ++wrong;
    }
    return wrong;
}

struct Async {
    Context& ctx;
    const Rig& r;
    MTL4::CommandQueue* qR;
    MTL4::CommandQueue* qC;
    MTL4::CommandBuffer* cbR;
    MTL4::CommandBuffer* cbC;
    MTL4::CommandAllocator* alR;
    MTL4::CommandAllocator* alC;
    MTL::SharedEvent* evGo;
    MTL::SharedEvent* evR;
    MTL::SharedEvent* evC;
    u64 v = 0;
};

// Timestamps 10/11 = render start/end, 12/13 = compute start/end.
void recordRender(Async& a, bool bandwidth) {
    a.alR->reset();
    a.cbR->beginCommandBuffer(a.alR);
    MTL4::ComputeCommandEncoder* e = a.cbR->computeCommandEncoder();
    a.ctx.anchorDispatch(e);
    e->writeTimestamp(MTL4::TimestampGranularityPrecise, a.ctx.counterHeap(), 10);
    e->barrierAfterStages(MTL::StageDispatch, MTL::StageVertex | MTL::StageFragment | MTL::StageDispatch, MTL4::VisibilityOptionNone);
    e->endEncoding();
    encodeA(a.ctx, a.r, a.cbR, bandwidth);
    e = a.cbR->computeCommandEncoder();
    e->barrierAfterQueueStages(MTL::StageVertex | MTL::StageFragment | MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionNone);
    a.ctx.anchorDispatch(e);
    e->writeTimestamp(MTL4::TimestampGranularityPrecise, a.ctx.counterHeap(), 11);
    e->endEncoding();
    a.cbR->endCommandBuffer();
}

void recordCompute(Async& a) {
    a.alC->reset();
    a.cbC->beginCommandBuffer(a.alC);
    MTL4::ComputeCommandEncoder* e = a.cbC->computeCommandEncoder();
    a.ctx.anchorDispatch(e);
    e->writeTimestamp(MTL4::TimestampGranularityPrecise, a.ctx.counterHeap(), 12);
    e->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
    a.ctx.table()->setAddress(a.r.out->gpuAddress(), 0);
    a.ctx.table()->setAddress(a.r.params->gpuAddress() + 2 * kSlot, 1);
    e->setComputePipelineState(a.r.psoC);
    e->setArgumentTable(a.ctx.table());
    e->dispatchThreads(MTL::Size::Make(kThreads, 1, 1), MTL::Size::Make(256, 1, 1));
    e->writeTimestamp(MTL4::TimestampGranularityPrecise, a.ctx.counterHeap(), 13);
    e->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
    e->endEncoding();
    a.cbC->endCommandBuffer();
}

enum class AMode { RenderOnly, ComputeOnly, Concurrent, Dependent };

// GPU ms: RenderOnly/ComputeOnly = that span; Concurrent/Dependent = first start .. last end.
double runAsync(Async& a, AMode m, bool bandwidth) {
    const bool doR = m != AMode::ComputeOnly, doC = m != AMode::RenderOnly;
    if (doR) recordRender(a, bandwidth);
    if (doC) recordCompute(a);
    const u64 v = ++a.v;
    if (doR) {
        a.qR->wait(a.evGo, v);
        const MTL4::CommandBuffer* b[] = {a.cbR};
        a.qR->commit(b, 1);
        a.qR->signalEvent(a.evR, v);
    }
    if (doC) {
        a.qC->wait(m == AMode::Dependent ? static_cast<const MTL::Event*>(a.evR) : static_cast<const MTL::Event*>(a.evGo), v);
        if (m == AMode::Dependent) {
            // Still start from the go event: the dependent compute waits for both.
        }
        const MTL4::CommandBuffer* b[] = {a.cbC};
        a.qC->commit(b, 1);
        a.qC->signalEvent(a.evC, v);
    }
    a.evGo->setSignaledValue(v); // release both queues
    if (doR && !a.evR->waitUntilSignaledValue(v, 60000)) throw BenchError("B-19: render queue timeout");
    if (doC && !a.evC->waitUntilSignaledValue(v, 60000)) throw BenchError("B-19: compute queue timeout");
    const std::vector<u64> t = a.ctx.readTimestamps(10, 4);
    double first = 1e30, last = 0;
    if (doR) {
        if (t[0] == 0 || t[1] < t[0]) throw BenchError("B-19: invalid render timestamps");
        first = std::min<double>(first, double(t[0]));
        last  = std::max<double>(last, double(t[1]));
    }
    if (doC) {
        if (t[2] == 0 || t[3] < t[2]) throw BenchError("B-19: invalid compute timestamps");
        first = std::min<double>(first, double(t[2]));
        last  = std::max<double>(last, double(t[3]));
    }
    return a.ctx.ticksToMs(u64(first), u64(last));
}

void benchOverlap(Context& ctx, Report& rep) {
    MTL::Library* lib = ctx.library("b19_overlap.metal");
    Rig r;
    auto rp = [&](const char* vs, const char* fs, MTL::PixelFormat pf) {
        MTL4::RenderPipelineDescriptor* d = MTL4::RenderPipelineDescriptor::alloc()->init();
        d->setVertexFunctionDescriptor(ctx.function(lib, vs));
        d->setFragmentFunctionDescriptor(ctx.function(lib, fs));
        d->colorAttachments()->object(0)->setPixelFormat(pf);
        MTL::RenderPipelineState* p = ctx.render(d);
        d->release();
        return p;
    };
    r.psoA = rp("b19_vs_full", "b19_fs_heavy", MTL::PixelFormatRGBA8Unorm);
    r.psoF = rp("b19_vs_full", "b19_fs_float", MTL::PixelFormatRGBA32Float);
    r.psoB = rp("b19_vs_heavy", "b19_fs_tri", MTL::PixelFormatRGBA8Unorm);
    r.psoC = ctx.compute(lib, "b19_compute");
    auto tex = [&](MTL::PixelFormat pf, u32 w, u32 h) {
        MTL::TextureDescriptor* td = MTL::TextureDescriptor::texture2DDescriptor(pf, w, h, false);
        td->setUsage(MTL::TextureUsageRenderTarget);
        td->setStorageMode(MTL::StorageModePrivate);
        return ctx.texture(td);
    };
    r.tA     = tex(MTL::PixelFormatRGBA8Unorm, kDim, kDim);
    r.tB     = tex(MTL::PixelFormatRGBA8Unorm, kDim, kDim);
    r.tF     = tex(MTL::PixelFormatRGBA32Float, r.fW, r.fH);
    r.params = ctx.buffer(4 * kSlot);
    r.out    = ctx.buffer(size_t(kThreads) * 4);

    const u32 reps = ctx.quick() ? 9 : 21;
    const double target = 0.45; // ms per part

    // --- Calibration ------------------------------------------------------------------
    const u32 itA = calibrate(ctx, r, 0, target, 32, [&] { return timeRender(ctx, r, Mode::AOnly); });
    const u32 itB = calibrate(ctx, r, 1, target, 3000, [&] { return timeRender(ctx, r, Mode::BOnly); });
    ctx.log("B-19 calibrated: fragment chain %u, vertex chain %u", itA, itB);
    setParams(r, 0, itA);

    // --- (a) render -> render ---------------------------------------------------------------
    struct ScaleResult {
        double f, a, b, nodep, dep;
    };
    std::vector<ScaleResult> res;
    bool depOk = true, ratioOk = true;
    std::string depText, ratioText;
    for (double f : {0.5, 1.0, 2.0}) {
        setParams(r, 1, std::max<u32>(1, u32(double(itB) * f)));
        ctx.keepWarm(30);
        // Interleaved rounds so that clock drift hits every variant.
        std::vector<double> va, vb, vn, vd;
        // A render span costs a fixed ~15 or ~60 us (bimodal, B-14): medians
        // of "A alone" and "B alone" could both land in the slow mode while
        // the pair did not (dep/(A+B) 0.87 at B x0.5, twice).  Per-round
        // minima (fast mode), rotating order, a throwaway span after the
        // warm load.
        const Mode modes[4] = {Mode::AOnly, Mode::BOnly, Mode::NoDep, Mode::Dep};
        std::vector<double>* outs[4] = {&va, &vb, &vn, &vd};
        for (u32 round = 0; round < 4; ++round) {
            ctx.keepWarm(15);
            timeRender(ctx, r, Mode::Dep);
            for (u32 k = 0; k < 4; ++k) {
                const u32 v = (round + k) % 4;
                outs[v]->push_back(ctx.measure([&] { return timeRender(ctx, r, modes[v]); }, reps).min);
            }
        }
        const Stats sa = phosphor::soc::computeStats(va), sb = phosphor::soc::computeStats(vb),
                    sn = phosphor::soc::computeStats(vn), sd = phosphor::soc::computeStats(vd);
        const std::string tag = "b_x" + num(f, 3);
        const std::map<std::string, double> params = {{"b_scale", f}, {"a_iters", double(itA)}, {"b_iters", double(itB)}};
        rep.metric("render_render." + tag + ".a_alone.ms", "ms", sa, params, false);
        rep.metric("render_render." + tag + ".b_alone.ms", "ms", sb, params, false);
        rep.metric("render_render." + tag + ".no_dep.ms", "ms", sn, params, false);
        rep.metric("render_render." + tag + ".dep.ms", "ms", sd, params, false);
        const double sum = sa.median + sb.median;
        rep.value("overlap_ratio.render_render." + tag, "ratio", sn.median / sum, params, false);
        rep.value("dep_ratio.render_render." + tag, "ratio", sd.median / sum, params, false);
        if (f == 1.0) {
            rep.value("overlap_ratio.render_render", "ratio", sn.median / sum, params, false);
            rep.value("overlap_gain.render_render.ms", "ms", sum - sn.median, params);
        }
        depOk &= std::fabs(sd.median / sum - 1.0) < 0.10;
        depText += tag + " " + num(sd.median / sum, 5) + " ";
        // Ideal overlap of two passes: max(A, B) / (A + B); anything clearly below is a measurement error.
        const double ideal = std::max(sa.median, sb.median) / sum;
        ratioOk &= sn.median / sum > 0.9 * ideal;
        ratioText += tag + " " + num(sn.median / sum, 5) + ">=" + num(0.9 * ideal, 5) + " ";
        ctx.log("B-19 (a) B x%.1f: A %.3f B %.3f | no dep %.3f dep %.3f ms | overlap %.3f dep %.3f", f, sa.median, sb.median,
                sn.median, sd.median, sn.median / sum, sd.median / sum);
    }
    const u32 wrongRender = verifyRender(ctx, r);

    // --- (b) render on the main queue, compute on a second queue -------------------------------
    Async a{ctx, r, ctx.queue(), ctx.newQueue(), ctx.newCommandBuffer(), ctx.newCommandBuffer(), ctx.newAllocator(),
            ctx.newAllocator(), ctx.device()->newSharedEvent(), ctx.device()->newSharedEvent(), ctx.device()->newSharedEvent()};
    ctx.keep(a.evGo);
    ctx.keep(a.evR);
    ctx.keep(a.evC);
    // Calibrate the compute to the same cost as the render pass A.
    const u32 itC = calibrate(ctx, r, 2, target, 2000, [&] { return runAsync(a, AMode::ComputeOnly, false); });
    ctx.log("B-19 calibrated: compute chain %u", itC);
    setParams(r, 2, itC);
    u32 wrongCompute = 0;
    bool asyncDepOk = true;
    std::string asyncText;
    for (int kind = 0; kind < 2; ++kind) {
        const bool bw = kind == 1;
        const char* name = bw ? "bandwidth" : "alu";
        if (bw) { // calibrate the compute to the bandwidth pass as well: same target ms, nothing to change
        }
        std::vector<double> vr, vc, vcc, vd;
        // Same bimodal fixed cost: per-round minima, rotating order.
        const AMode amodes[4] = {AMode::RenderOnly, AMode::ComputeOnly, AMode::Concurrent, AMode::Dependent};
        std::vector<double>* aouts[4] = {&vr, &vc, &vcc, &vd};
        for (u32 round = 0; round < 4; ++round) {
            ctx.keepWarm(15);
            runAsync(a, AMode::Dependent, bw);
            for (u32 k = 0; k < 4; ++k) {
                const u32 v = (round + k) % 4;
                aouts[v]->push_back(ctx.measure([&] { return runAsync(a, amodes[v], bw); }, reps).min);
            }
        }
        const Stats sr = phosphor::soc::computeStats(vr), sc = phosphor::soc::computeStats(vc),
                    scc = phosphor::soc::computeStats(vcc), sd = phosphor::soc::computeStats(vd);
        const std::string tag = std::string("async.") + name;
        const std::map<std::string, double> params = {{"compute_iters", double(itC)}, {"render_iters", double(itA)}};
        rep.metric(tag + ".render_alone.ms", "ms", sr, params, false);
        rep.metric(tag + ".compute_alone.ms", "ms", sc, params, false);
        rep.metric(tag + ".concurrent.ms", "ms", scc, params, false);
        rep.metric(tag + ".dependent.ms", "ms", sd, params, false);
        const double sum = sr.median + sc.median;
        rep.value("overlap_ratio." + tag, "ratio", scc.median / sum, params, false);
        rep.value("dep_ratio." + tag, "ratio", sd.median / sum, params, false);
        asyncDepOk &= sd.median >= 0.95 * sum && sd.median <= 1.10 * sum + 0.15; // cross-queue event latency
        rep.value(tag + ".dep_latency.ms", "ms", sd.median - sum, params, false);
        asyncText += std::string(name) + " " + num(sd.median / sum, 5) + " ";
        ctx.log("B-19 (b) %s: render %.3f compute %.3f | concurrent %.3f dependent %.3f ms | overlap %.3f dep %.3f", name,
                sr.median, sc.median, scc.median, sd.median, scc.median / sum, sd.median / sum);
        // Compute results of the last concurrent run.
        runAsync(a, AMode::Concurrent, bw);
        const u32* o = static_cast<const u32*>(r.out->contents());
        for (u32 i : {0u, 1u, 255u, 256u, 4097u, 65535u, 123456u, 999999u, kThreads - 1})
            if (o[i] != chain(i, itC)) ++wrongCompute;
    }
    // Render output after the concurrent run (pass A of the alu kind was the last render: kind 1 wrote tF; check tA once more).
    {
        MTL::Buffer* ra = ctx.buffer(64 * 64 * 4);
        MTL4::CommandBuffer* cmd = ctx.beginCommands();
        encodeA(ctx, r, cmd, false); // rewrite, then read back: proves the pipeline still renders correctly with two queues alive
        MTL4::ComputeCommandEncoder* ce = cmd->computeCommandEncoder();
        ce->barrierAfterQueueStages(MTL::StageFragment, MTL::StageBlit, MTL4::VisibilityOptionDevice);
        ce->copyFromTexture(r.tA, 0, 0, MTL::Origin::Make(7, 9, 0), MTL::Size::Make(64, 64, 1), ra, 0, 64 * 4, 64 * 64 * 4);
        ce->endEncoding();
        ctx.submit();
        const u8* p = static_cast<const u8*>(ra->contents());
        for (u32 y = 0; y < 64; ++y)
            for (u32 x = 0; x < 64; ++x)
                for (u32 c = 0; c < 4; ++c)
                    if (p[(y * 64 + x) * 4 + c] != (hash(7 + x, 9 + y, c) & 255u)) ++wrongCompute;
    }

    // --- Controls ------------------------------------------------------------------------------
    rep.negative(depOk, "render_render forced dependency = sum of the parts within 10%: dep/(A+B) " + depText);
    rep.negative(asyncDepOk, "async dependent (compute waits for the render queue's event) = sum within -5%..+10% plus 0.15 ms of event latency: " + asyncText);
    rep.negative(ratioOk, "render_render no-dep time not below 0.9 x the ideal overlap max(A,B)/(A+B): " + ratioText);
    rep.negative(wrongRender == 0 && wrongCompute == 0,
                 wrongRender == 0 && wrongCompute == 0
                     ? "outputs verified: pass A pixels, pass B 65536 triangle origins, compute results, pass A after two-queue runs"
                     : std::to_string(wrongRender) + " wrong render values, " + std::to_string(wrongCompute) + " wrong compute/readback values");
    if (wrongRender || wrongCompute) rep.status(Status::Failed, "output verification failed");
    rep.note("(a) span = CommandTimer; (b) GPU timestamps on both queues, cb1 and cb2 released by one shared event, span = first start .. last end; "
             "overlap_ratio = concurrent / (render alone + compute alone); each part calibrated to ~0.45 ms");
}

} // namespace

SOC_BENCH("B-19", "overlap", "Overlap: vertex work of pass B during fragment work of pass A; async compute on a second queue", benchOverlap);

} // namespace soc
