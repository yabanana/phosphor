// B-18: cost of Metal 4 barriers (S-SYNC-1 of docs/APPLE_SOC_PLAYBOOK.md);
// productises bench/barrier_spike (F2.3, docs/opt-log.md) and reuses its rules:
// only barriers the spike found LEGAL are used (queue barriers are never
// validated; encoder barriers inside a compute encoder use Dispatch->Dispatch),
// and its race-detection method (a data dependency that loses an update when
// the barrier is missing, CPU-checked).
//
//   * barrier.encoder.us: N = 1000 tiny dependent dispatches (buf[i] += 1) in
//     one compute encoder, with / without barrierAfterEncoderStages(Dispatch,
//     Dispatch) between them: (span with - span without) / (N - 1).
//   * barrier.queue.<producer>_<consumer>.us: M = 200 independent pairs of tiny
//     passes (producer encoder, consumer encoder; each pair has its own 4 KiB
//     buffer, producer makes it 1, consumer makes it 2) with / without
//     barrierAfterQueueStages(producer stage, consumer stage) at the start of
//     every consumer: (span with - span without) / M.  Pairs: dispatch,
//     blit, vertex and fragment producers into dispatch / vertex / fragment
//     consumers, the combinations found legal in F2.3.
//   * barrier.queue.us = median of the per-pair medians.
// Runs with and without barrier alternate inside every sample so drift cancels.
// The difference is the barrier plus the lost overlap of the two encoders
// (the variant without barrier is a race by construction: spike caveat).
//
// Correctness (the negative control): with the barrier every buffer must read
// back 2 (0 wrong), in the timed runs and in a race probe whose producer is
// slow (spin loop; blit producer: 24 copies).  The variant WITHOUT barrier is
// expected to lose updates in the probe: race.<pair>.wrong_bufs counts them
// (0 = never raced on this GPU: reported as such, not faked).

#include "harness.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>

namespace soc {
namespace {

constexpr u32 kPairs   = 200;
constexpr u32 kWords   = 1024;
constexpr u32 kBufSize = kWords * 4;
constexpr u32 kChain   = 1000;
constexpr u32 kSpin    = 3000; // producer spin in the race probe

enum Kind { Compute, Blit, Vertex, Fragment };

struct Pair {
    const char* name;
    Kind        prod, cons;
    MTL::Stages after, before;
};
constexpr MTL::Stages kD = MTL::StageDispatch, kB = MTL::StageBlit, kV = MTL::StageVertex, kF = MTL::StageFragment;
const Pair kPairList[] = {
    {"dispatch_dispatch", Compute, Compute, kD, kD},  {"dispatch_vertex", Compute, Vertex, kD, kV},
    {"dispatch_fragment", Compute, Fragment, kD, kF}, {"fragment_dispatch", Fragment, Compute, kF, kD},
    {"blit_fragment", Blit, Fragment, kB, kF},        {"blit_vertex", Blit, Vertex, kB, kV},
    {"fragment_vertex", Fragment, Vertex, kF, kV},    {"fragment_fragment", Fragment, Fragment, kF, kF},
    {"vertex_fragment", Vertex, Fragment, kV, kF},    {"vertex_vertex", Vertex, Vertex, kV, kV},
};

struct Rig {
    Context& ctx;
    MTL::ComputePipelineState* inc = nullptr;
    MTL::RenderPipelineState* vInc = nullptr;
    MTL::RenderPipelineState* fInc = nullptr;
    MTL::Buffer* bufs = nullptr;
    MTL::Buffer* ones = nullptr;
    MTL::Buffer* pNone = nullptr;
    MTL::Buffer* pSpin = nullptr;
    MTL::Texture* target = nullptr;
    u32 wrongWith = 0, wrongWithout = 0;
    explicit Rig(Context& c) : ctx(c) {}
};

MTL4::RenderCommandEncoder* renderEnc(Rig& r, MTL4::CommandBuffer* cmd, MTL::RenderPipelineState* pso) {
    MTL4::RenderPassDescriptor* pd = MTL4::RenderPassDescriptor::alloc()->init();
    auto* c = pd->colorAttachments()->object(0);
    c->setTexture(r.target);
    c->setLoadAction(MTL::LoadActionDontCare);
    c->setStoreAction(MTL::StoreActionDontCare);
    MTL4::RenderCommandEncoder* re = cmd->renderCommandEncoder(pd);
    pd->release();
    re->setViewport(MTL::Viewport{0.0, 0.0, 32.0, 32.0, 0.0, 1.0});
    re->setRenderPipelineState(pso);
    return re;
}

// One pass of `kind` acting on buffer `k`.  barrier: queue barrier at the
// start of this encoder.  spin: producer spin (race probe).
void pass(Rig& r, MTL4::CommandBuffer* cmd, Kind kind, u32 k, bool barrier, MTL::Stages after, MTL::Stages before,
          bool slow) {
    const u64 addr = r.bufs->gpuAddress() + u64(k) * kBufSize;
    r.ctx.table()->setAddress(addr, 0);
    r.ctx.table()->setAddress((slow ? r.pSpin : r.pNone)->gpuAddress(), 1);
    if (kind == Compute || kind == Blit) {
        MTL4::ComputeCommandEncoder* ce = cmd->computeCommandEncoder();
        if (barrier) ce->barrierAfterQueueStages(after, before, MTL4::VisibilityOptionDevice);
        if (kind == Compute) {
            ce->setComputePipelineState(r.inc);
            ce->setArgumentTable(r.ctx.table());
            ce->dispatchThreads(MTL::Size::Make(kWords, 1, 1), MTL::Size::Make(256, 1, 1));
        } else {
            for (u32 c = 0; c < (slow ? 24u : 1u); ++c) ce->copyFromBuffer(r.ones, 0, r.bufs, u64(k) * kBufSize, kBufSize);
        }
        ce->endEncoding();
    } else {
        MTL4::RenderCommandEncoder* re = renderEnc(r, cmd, kind == Vertex ? r.vInc : r.fInc);
        if (barrier) re->barrierAfterQueueStages(after, before, MTL4::VisibilityOptionDevice);
        re->setArgumentTable(r.ctx.table(), MTL::RenderStageVertex | MTL::RenderStageFragment);
        if (kind == Vertex) re->drawPrimitives(MTL::PrimitiveTypePoint, 0, kWords);
        else re->drawPrimitives(MTL::PrimitiveTypeTriangle, 0, 3);
        re->endEncoding();
    }
}

// Span (ms) of kPairs producer/consumer pairs; counts buffers that do not read 2.
double pairsSpan(Rig& r, const Pair& p, bool barrier, bool slow, u32& wrong) {
    std::memset(r.bufs->contents(), 0, size_t(kPairs) * kBufSize);
    CommandTimer t(r.ctx);
    MTL4::CommandBuffer* cmd = t.begin();
    for (u32 k = 0; k < kPairs; ++k) {
        pass(r, cmd, p.prod, k, false, 0, 0, slow);
        pass(r, cmd, p.cons, k, barrier, p.after, p.before, false);
    }
    const double ms = t.finish();
    const auto* w = static_cast<const u32*>(r.bufs->contents());
    u32 bad = 0;
    for (u32 k = 0; k < kPairs; ++k) {
        for (u32 i = 0; i < kWords; ++i)
            if (w[size_t(k) * kWords + i] != 2u) {
                ++bad;
                break;
            }
    }
    wrong = bad;
    return ms;
}

// One compute encoder: kChain dependent dispatches, optional encoder barrier.
double chainSpan(Rig& r, bool barrier, u32& wrong) {
    std::memset(r.bufs->contents(), 0, kBufSize);
    ComputeTimer t(r.ctx);
    MTL4::ComputeCommandEncoder* e = t.begin();
    r.ctx.table()->setAddress(r.bufs->gpuAddress(), 0);
    r.ctx.table()->setAddress(r.pNone->gpuAddress(), 1);
    e->setComputePipelineState(r.inc);
    e->setArgumentTable(r.ctx.table());
    for (u32 d = 0; d < kChain; ++d) {
        if (barrier && d > 0) e->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
        e->dispatchThreads(MTL::Size::Make(kWords, 1, 1), MTL::Size::Make(256, 1, 1));
    }
    t.lap();
    const double ms = t.finish()[0];
    const auto* w = static_cast<const u32*>(r.bufs->contents());
    u32 minv = ~0u;
    for (u32 i = 0; i < kWords; ++i) minv = std::min(minv, w[i]);
    wrong = minv == kChain ? 0 : 1;
    return ms;
}


// Under the validation layers (--validate) GPU times are distorted by the instrumentation: the timing controls are
// then informational (detail says so); the correctness controls stay enforced.
bool timingControlsEnforced() { return !std::getenv("MTL_SHADER_VALIDATION") && !std::getenv("MTL_DEBUG_LAYER"); }
void benchBarriers(Context& ctx, Report& rep) {
    MTL::Library* lib = ctx.library("b18_barriers.metal");
    Rig r(ctx);
    r.inc   = ctx.compute(lib, "b18_inc");
    r.bufs  = ctx.buffer(size_t(kPairs) * kBufSize);
    r.ones  = ctx.buffer(kBufSize);
    r.pNone = ctx.buffer(16);
    r.pSpin = ctx.buffer(16);
    {
        auto* o = static_cast<u32*>(r.ones->contents());
        for (u32 i = 0; i < kWords; ++i) o[i] = 1u;
        auto* pn = static_cast<u32*>(r.pNone->contents());
        pn[0] = 0; pn[1] = 0;
        auto* ps = static_cast<u32*>(r.pSpin->contents());
        ps[0] = kSpin; ps[1] = 0;
    }
    MTL::TextureDescriptor* td = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatR8Unorm, 32, 32, false);
    td->setUsage(MTL::TextureUsageRenderTarget);
    td->setStorageMode(MTL::StorageModePrivate);
    r.target = ctx.texture(td);
    {
        MTL4::RenderPipelineDescriptor* d = MTL4::RenderPipelineDescriptor::alloc()->init();
        d->setVertexFunctionDescriptor(ctx.function(lib, "b18_v_inc"));
        d->setFragmentFunctionDescriptor(ctx.function(lib, "b18_f_dummy"));
        d->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatR8Unorm);
        r.vInc = ctx.render(d);
        d->release();
        d = MTL4::RenderPipelineDescriptor::alloc()->init();
        d->setVertexFunctionDescriptor(ctx.function(lib, "b18_v_full"));
        d->setFragmentFunctionDescriptor(ctx.function(lib, "b18_f_inc"));
        d->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatR8Unorm);
        r.fInc = ctx.render(d);
        d->release();
    }
    const u32 reps = std::max<u32>(ctx.reps(), 15);
    ctx.keepWarm(50);

    bool costPositive = true;
    std::string nonPositive;
    // ---- encoder barrier ----------------------------------------------------------------
    {
        u32 wWith = 0, wWithout = 0, tot = 0;
        double spanWith = 0, spanWithout = 0;
        int flip = 0;
        const Stats s = ctx.measure(
            [&] {
                u32 a = 0, b = 0;
                double with, without;
                if ((flip++ & 1) == 0) { with = chainSpan(r, true, a); without = chainSpan(r, false, b); }
                else { without = chainSpan(r, false, b); with = chainSpan(r, true, a); }
                wWith += a; wWithout += b; ++tot;
                spanWith = with; spanWithout = without;
                return (with - without) * 1e3 / double(kChain - 1);
            },
            reps);
        r.wrongWith += wWith;
        rep.metric("barrier.encoder.us", "us", s,
                   {{"dispatches", double(kChain)}, {"span_with_ms", spanWith}, {"span_without_ms", spanWithout}}, false);
        rep.value("race.encoder.wrong_runs", "count", double(wWithout), {{"runs", double(tot)}}, false);
        if (s.median <= 0) { costPositive = false; nonPositive += "encoder "; }
    }
    ctx.keepWarm(30);

    // ---- queue barriers per pair --------------------------------------------------------
    std::vector<double> pairMedians;
    std::string raceSummary;
    u32 racedPairs = 0;
    for (const Pair& p : kPairList) {
        u32 wWithout = 0, tot = 0;
        double spanWith = 0, spanWithout = 0;
        int flip = 0;
        const Stats s = ctx.measure(
            [&] {
                u32 a = 0, b = 0;
                double with, without;
                if ((flip++ & 1) == 0) { with = pairsSpan(r, p, true, false, a); without = pairsSpan(r, p, false, false, b); }
                else { without = pairsSpan(r, p, false, false, b); with = pairsSpan(r, p, true, false, a); }
                r.wrongWith += a; wWithout += b; ++tot;
                spanWith = with; spanWithout = without;
                return (with - without) * 1e3 / double(kPairs);
            },
            reps);
        rep.metric(std::string("barrier.queue.") + p.name + ".us", "us", s,
                   {{"pairs", double(kPairs)}, {"span_with_ms", spanWith}, {"span_without_ms", spanWithout}}, false);
        pairMedians.push_back(s.median);
        // Individual pairs can cost ~0 (the consumer would wait anyway: fragment_dispatch measured slightly negative);
        // the control applies to the encoder barrier and to the median over the pairs.
        // Race probe: slow producer, 6 runs each.
        u32 raceWithout = 0, raceWith = 0;
        for (int i = 0; i < 6; ++i) {
            u32 a = 0, b = 0;
            pairsSpan(r, p, false, true, a);
            raceWithout += a;
            pairsSpan(r, p, true, true, b);
            raceWith += b;
        }
        r.wrongWith += raceWith;
        rep.value(std::string("race.") + p.name + ".wrong_bufs", "count", double(raceWithout + wWithout),
                  {{"buffers_checked", double(kPairs * (6 + tot))}}, false);
        if (raceWithout + wWithout) ++racedPairs;
        ctx.keepWarm(20);
    }
    std::sort(pairMedians.begin(), pairMedians.end());
    const size_t nm = pairMedians.size();
    const double queueMedian = nm % 2 ? pairMedians[nm / 2] : 0.5 * (pairMedians[nm / 2 - 1] + pairMedians[nm / 2]);
    rep.value("barrier.queue.us", "us", queueMedian, {{"pairs", double(nm)}}, false);
    if (queueMedian <= 0) { costPositive = false; nonPositive += "queue-median "; }

    rep.negative(r.wrongWith == 0, r.wrongWith == 0 ? "with the barrier every consumer read the producer's value (0 wrong buffers/runs, timed runs and slow-producer probes)"
                                                    : std::to_string(r.wrongWith) + " wrong reads WITH the barrier");
    rep.negative(costPositive || !timingControlsEnforced(), std::string(timingControlsEnforced() ? "" : "[timing not enforced under validation] ") + (costPositive ? "barrier variants are slower than the variants without (median cost > 0 for the encoder barrier and for the median over the 10 queue pairs)"
                                            : "barrier not slower than no barrier for: " + nonPositive));
    if (r.wrongWith) rep.status(Status::Failed, "reads wrong WITH the barrier");
    rep.note(racedPairs == 0 ? "the variants WITHOUT barrier never lost an update on this GPU in the probes (race control n/a: reported, not faked)"
                             : std::to_string(racedPairs) + "/10 queue pairs lost updates without the barrier in the slow-producer probe (race.*.wrong_bufs)");
    rep.note("cost = (span with - span without)/count: barrier plus lost overlap; only F2.3-legal barriers are used (0 validation messages expected)");
}

} // namespace

SOC_BENCH("B-18", "sync.barriers", "Cost of encoder and queue barriers per stage pair, with race control", benchBarriers);

} // namespace soc
