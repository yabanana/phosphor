// F5-S7 (F2.6 residue): the stable culling of F5-S4 (3 dispatches) on a second
// MTL4 queue beside a raster load on the main queue, against the same two
// workloads serialised on the main queue.
//
// Raster load: one render pass, 4 instances of a full-screen triangle with a
// seeded per-pixel ALU chain in the fragment shader, into a 3200x1800 RGBA8
// private texture; the chain length is calibrated to ~3 ms.
// Spans (GPU timestamps of both queues, same time domain; first start .. last
// end; both command buffers wait on one shared event released by the CPU):
//   raster alone, cull alone (on the second queue), serial (one command
//   buffer on the main queue, queue barrier Fragment -> Dispatch between the
//   raster pass and the cull), concurrent (raster on the main queue, cull on
//   the second queue), and samequeue_nobarrier (informational: raster and cull
//   in one command buffer without a barrier).
// Every concurrent / serial run verifies the cull result byte-exactly against
// the cull-alone result; the raster pixels are verified against the CPU after
// the concurrent runs.
//
// Metrics (ms): async.raster_alone.ms, async.cull_alone.ms, async.serial.ms,
// async.concurrent.ms, async.samequeue_nobarrier.ms, async.overlap_ratio
// (concurrent / (raster + cull)), async.raster.iters, async.cull_wrong_runs,
// async.raster_wrong_pixels.
// Negative control: concurrent <= serial and >= max(raster, cull), tolerance
// 5% + 0.05 ms (medians).

#include "s4_cull.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>

namespace f5::cull {
namespace {

using soc::BenchError;
using soc::Context;
using soc::Report;
using soc::Status;

constexpr u32 kW = 3200, kH = 1800;

bool timingEnforced() { return !std::getenv("MTL_SHADER_VALIDATION") && !std::getenv("MTL_DEBUG_LAYER"); }

struct Rig7 {
    Context& ctx;
    Rig& rig;
    MTL::RenderPipelineState* pso = nullptr;
    MTL::Texture* tex             = nullptr;
    MTL::Buffer* iters            = nullptr;
    MTL4::CommandQueue* qC        = nullptr;
    MTL4::CommandBuffer *cbR = nullptr, *cbC = nullptr, *cbS = nullptr;
    MTL4::CommandAllocator *alR = nullptr, *alC = nullptr, *alS = nullptr;
    MTL::SharedEvent *evGo = nullptr, *evR = nullptr, *evC = nullptr, *evS = nullptr;
    u64 v = 0;
};

void encodeRaster(Rig7& r, MTL4::CommandBuffer* cmd) {
    MTL4::RenderPassDescriptor* pd = MTL4::RenderPassDescriptor::alloc()->init();
    auto* c                        = pd->colorAttachments()->object(0);
    c->setTexture(r.tex);
    c->setLoadAction(MTL::LoadActionClear);
    c->setStoreAction(MTL::StoreActionStore);
    c->setClearColor(MTL::ClearColor::Make(0, 0, 0, 0));
    MTL4::RenderCommandEncoder* re = cmd->renderCommandEncoder(pd);
    pd->release();
    r.ctx.table()->setAddress(r.iters->gpuAddress(), 0);
    re->setRenderPipelineState(r.pso);
    re->setArgumentTable(r.ctx.table(), MTL::RenderStageVertex | MTL::RenderStageFragment);
    re->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(3), NS::UInteger(4));
    re->endEncoding();
}

void encodeCull(Rig7& r, MTL4::CommandBuffer* cmd) {
    MTL4::ComputeCommandEncoder* e = cmd->computeCommandEncoder();
    encodeStable(r.ctx, e, r.rig, r.rig.fast,
                 [&] { e->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice); });
    e->endEncoding();
}

// Head: anchor timestamp (index `t`), later work waits for it.  Tail: waits for everything, timestamp `t+1`.
void head(Rig7& r, MTL4::CommandBuffer* cmd, u32 t) {
    MTL4::ComputeCommandEncoder* e = cmd->computeCommandEncoder();
    r.ctx.anchorDispatch(e);
    e->writeTimestamp(MTL4::TimestampGranularityPrecise, r.ctx.counterHeap(), t);
    e->barrierAfterStages(MTL::StageDispatch, MTL::StageVertex | MTL::StageFragment | MTL::StageDispatch, MTL4::VisibilityOptionNone);
    e->endEncoding();
}
void tail(Rig7& r, MTL4::CommandBuffer* cmd, u32 t) {
    MTL4::ComputeCommandEncoder* e = cmd->computeCommandEncoder();
    e->barrierAfterQueueStages(MTL::StageVertex | MTL::StageFragment | MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionNone);
    r.ctx.anchorDispatch(e);
    e->writeTimestamp(MTL4::TimestampGranularityPrecise, r.ctx.counterHeap(), t);
    e->endEncoding();
}

enum class M { Raster, Cull, Serial, Concurrent, SameQueue };

// Timestamps: 10/11 raster cb, 12/13 cull cb, 14/15 serial cb.
double run(Rig7& r, M m) {
    const bool doR = m == M::Raster || m == M::Concurrent, doC = m == M::Cull || m == M::Concurrent,
               doS = m == M::Serial || m == M::SameQueue;
    if (doR) {
        r.alR->reset();
        r.cbR->beginCommandBuffer(r.alR);
        head(r, r.cbR, 10);
        encodeRaster(r, r.cbR);
        tail(r, r.cbR, 11);
        r.cbR->endCommandBuffer();
    }
    if (doC) {
        r.alC->reset();
        r.cbC->beginCommandBuffer(r.alC);
        head(r, r.cbC, 12);
        encodeCull(r, r.cbC);
        tail(r, r.cbC, 13);
        r.cbC->endCommandBuffer();
    }
    if (doS) {
        r.alS->reset();
        r.cbS->beginCommandBuffer(r.alS);
        head(r, r.cbS, 14);
        encodeRaster(r, r.cbS);
        if (m == M::Serial) {
            MTL4::ComputeCommandEncoder* e = r.cbS->computeCommandEncoder();
            e->barrierAfterQueueStages(MTL::StageVertex | MTL::StageFragment, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
            encodeStable(r.ctx, e, r.rig, r.rig.fast,
                         [&] { e->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice); });
            e->endEncoding();
        } else {
            encodeCull(r, r.cbS);
        }
        tail(r, r.cbS, 15);
        r.cbS->endCommandBuffer();
    }
    const u64 v = ++r.v;
    if (doR) {
        r.ctx.queue()->wait(r.evGo, v);
        const MTL4::CommandBuffer* b[] = {r.cbR};
        r.ctx.queue()->commit(b, 1);
        r.ctx.queue()->signalEvent(r.evR, v);
    }
    if (doC) {
        r.qC->wait(r.evGo, v);
        const MTL4::CommandBuffer* b[] = {r.cbC};
        r.qC->commit(b, 1);
        r.qC->signalEvent(r.evC, v);
    }
    if (doS) {
        r.ctx.queue()->wait(r.evGo, v);
        const MTL4::CommandBuffer* b[] = {r.cbS};
        r.ctx.queue()->commit(b, 1);
        r.ctx.queue()->signalEvent(r.evS, v);
    }
    r.evGo->setSignaledValue(v);
    if (doR && !r.evR->waitUntilSignaledValue(v, 60000)) throw BenchError("F5-S7: raster timeout");
    if (doC && !r.evC->waitUntilSignaledValue(v, 60000)) throw BenchError("F5-S7: cull timeout");
    if (doS && !r.evS->waitUntilSignaledValue(v, 60000)) throw BenchError("F5-S7: serial timeout");
    const std::vector<u64> t = r.ctx.readTimestamps(10, 6);
    double first = 1e30, last = 0;
    auto span = [&](size_t i) {
        if (t[i] == 0 || t[i + 1] < t[i]) throw BenchError("F5-S7: invalid timestamps");
        first = std::min<double>(first, double(t[i]));
        last  = std::max<double>(last, double(t[i + 1]));
    };
    if (doR) span(0);
    if (doC) span(2);
    if (doS) span(4);
    return r.ctx.ticksToMs(u64(first), u64(last));
}

// Snapshot of everything the cull wrote (exact comparison).
std::vector<u32> cullSnapshot(const Rig& rig) {
    const auto* prefix = static_cast<const u32*>(rig.prefix->contents());
    const auto* list   = static_cast<const u32*>(rig.listStable->contents());
    std::vector<u32> s(prefix, prefix + kN + 1);
    s.insert(s.end(), list, list + prefix[kN]);
    return s;
}

u32 chain(u32 x, u32 y, u32 inst, u32 iters) {
    u32 a = (x * 73856093u) ^ (y * 19349663u) ^ (inst * 83492791u + 1u);
    for (u32 k = 0; k < iters; ++k) a = a * 1664525u + 1013904223u;
    return a;
}

std::string str(double v, size_t n = 6) { return std::to_string(v).substr(0, n); }

void benchAsync(Context& ctx, Report& rep) {
    const Scene scene       = makeScene(0xF5A4);
    const CullParams params = makeParams(scene);
    Rig rig                 = makeRig(ctx, scene, params);
    MTL::Library* lib       = f5Library(ctx, "s4_cull.metal");
    Rig7 r{ctx, rig};
    MTL4::RenderPipelineDescriptor* d = MTL4::RenderPipelineDescriptor::alloc()->init();
    d->setVertexFunctionDescriptor(ctx.function(lib, "s7_vs"));
    d->setFragmentFunctionDescriptor(ctx.function(lib, "s7_fs"));
    d->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatRGBA8Unorm);
    r.pso = ctx.render(d);
    d->release();
    MTL::TextureDescriptor* td = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatRGBA8Unorm, kW, kH, false);
    td->setUsage(MTL::TextureUsageRenderTarget);
    td->setStorageMode(MTL::StorageModePrivate);
    r.tex   = ctx.texture(td);
    r.iters = ctx.buffer(256);
    r.qC    = ctx.newQueue();
    r.cbR = ctx.newCommandBuffer();
    r.cbC = ctx.newCommandBuffer();
    r.cbS = ctx.newCommandBuffer();
    r.alR = ctx.newAllocator();
    r.alC = ctx.newAllocator();
    r.alS = ctx.newAllocator();
    for (MTL::SharedEvent** e : {&r.evGo, &r.evR, &r.evC, &r.evS}) {
        *e = ctx.device()->newSharedEvent();
        ctx.keep(*e);
    }
    auto setIters = [&](u32 n) { *static_cast<u32*>(r.iters->contents()) = n; };

    // ---- calibrate the raster to ~3 ms ------------------------------------------------------------------
    ctx.warmUp(5.0);
    auto rasterMs = [&](u32 it) {
        setIters(it);
        ctx.keepWarm(20);
        return ctx.measure([&] { return run(r, M::Raster); }, 7).median;
    };
    const double t1 = rasterMs(256), t2 = rasterMs(768);
    const double slope = std::max(1e-9, (t2 - t1) / 512.0), icpt = std::max(0.0, t1 - 256 * slope);
    u32 iters = std::clamp<u32>(u32(std::max(1.0, (3.0 - icpt) / slope)), 1, 200000);
    const double t3 = rasterMs(iters);
    iters           = std::clamp<u32>(u32(std::max(1.0, (3.0 - icpt) / std::max(1e-9, (t3 - icpt) / iters))), 1, 200000);
    setIters(iters);
    ctx.log("F5-S7: raster chain %u iterations", iters);

    // ---- reference cull result (cull alone) --------------------------------------------------------------
    run(r, M::Cull);
    const std::vector<u32> refSnap = cullSnapshot(rig);
    const u32 refTotal             = refSnap[kN];
    u32 wrongRuns                  = 0;
    auto verify = [&] {
        if (cullSnapshot(rig) != refSnap) ++wrongRuns;
    };

    // ---- spans: interleaved rounds, rotating order ------------------------------------------------------
    const u32 rounds = std::max<u32>(ctx.quick() ? 20 : 30, 20);
    const M modes[5] = {M::Raster, M::Cull, M::Serial, M::Concurrent, M::SameQueue};
    std::vector<double> samples[5];
    for (u32 k = 0; k < 5; ++k) run(r, modes[k]); // warm-up
    for (u32 round = 0; round < rounds; ++round) {
        if (round % 5 == 0) ctx.keepWarm(15);
        for (u32 k = 0; k < 5; ++k) {
            const u32 v = (round + k) % 5;
            // The cull result is cleared first so that a run that did not execute the cull cannot pass.
            if (modes[v] != M::Raster) std::memset(rig.prefix->contents(), 0xFF, rig.prefix->length());
            samples[v].push_back(run(r, modes[v]));
            if (modes[v] != M::Raster) verify();
        }
    }
    const soc::Stats sR = phosphor::soc::computeStats(samples[0]), sC = phosphor::soc::computeStats(samples[1]),
                     sS = phosphor::soc::computeStats(samples[2]), sK = phosphor::soc::computeStats(samples[3]),
                     sQ = phosphor::soc::computeStats(samples[4]);
    const std::map<std::string, double> pr = {{"raster_iters", double(iters)}, {"rounds", double(rounds)}, {"visible", double(refTotal)}};
    rep.metric("async.raster_alone.ms", "ms", sR, pr, false);
    rep.metric("async.cull_alone.ms", "ms", sC, pr, false);
    rep.metric("async.serial.ms", "ms", sS, pr, false);
    rep.metric("async.concurrent.ms", "ms", sK, pr, false);
    rep.metric("async.samequeue_nobarrier.ms", "ms", sQ, pr, false);
    rep.value("async.overlap_ratio", "ratio", sK.median / (sR.median + sC.median), pr, false);
    rep.value("async.raster.iters", "iterations", iters, {}, false);
    rep.value("async.cull_wrong_runs", "runs", wrongRuns, {{"checked", double(rounds * 4)}}, false);

    // ---- raster pixels after the last concurrent run ------------------------------------------------------
    run(r, M::Concurrent);
    u32 wrongPixels = 0;
    {
        MTL::Buffer* rb = ctx.buffer(2 * 64 * 64 * 4);
        std::memset(rb->contents(), 0, rb->length());
        MTL4::CommandBuffer* cmd       = ctx.beginCommands();
        MTL4::ComputeCommandEncoder* e = cmd->computeCommandEncoder();
        e->barrierAfterQueueStages(MTL::StageFragment, MTL::StageBlit, MTL4::VisibilityOptionDevice);
        e->copyFromTexture(r.tex, 0, 0, MTL::Origin::Make(1000, 500, 0), MTL::Size::Make(64, 64, 1), rb, 0, 64 * 4, 64 * 64 * 4);
        e->copyFromTexture(r.tex, 0, 0, MTL::Origin::Make(kW - 64, kH - 64, 0), MTL::Size::Make(64, 64, 1), rb, 64 * 64 * 4, 64 * 4,
                           64 * 64 * 4);
        e->endEncoding();
        ctx.submit();
        const u8* px = static_cast<const u8*>(rb->contents());
        for (u32 blk = 0; blk < 2; ++blk)
            for (u32 y = 0; y < 64; ++y)
                for (u32 x = 0; x < 64; ++x) {
                    const u32 gx = (blk ? kW - 64 : 1000) + x, gy = (blk ? kH - 64 : 500) + y;
                    const u32 a  = chain(gx, gy, 3, iters); // the last instance drawn wins the pixel
                    const u8* p  = px + (size_t(blk) * 64 * 64 + y * 64 + x) * 4;
                    if (p[0] != (a & 255u) || p[1] != ((a >> 8) & 255u) || p[2] != ((a >> 16) & 255u) || p[3] != 255) ++wrongPixels;
                }
    }
    rep.value("async.raster_wrong_pixels", "pixels", wrongPixels, {{"checked", 2.0 * 64 * 64}}, false);

    // ---- controls ---------------------------------------------------------------------------------------------
    const double tol    = 0.05 * sS.median + 0.05;
    const bool upperOk  = sK.median <= sS.median + tol;
    const bool lowerOk  = sK.median >= std::max(sR.median, sC.median) - tol;
    const bool timingOk = upperOk && lowerOk;
    const bool exact    = wrongRuns == 0 && wrongPixels == 0;
    const std::string numbers = "raster " + str(sR.median) + " cull " + str(sC.median) + " serial " + str(sS.median) + " concurrent " +
                                str(sK.median) + " samequeue_nobarrier " + str(sQ.median) + " ms (tol " + str(tol) + "): concurrent <= serial " +
                                (upperOk ? "ok" : "VIOLATED") + ", >= max(parts) " + (lowerOk ? "ok" : "VIOLATED") + "; cull wrong runs " +
                                std::to_string(wrongRuns) + ", wrong pixels " + std::to_string(wrongPixels);
    const bool pass = exact && (timingOk || !timingEnforced());
    rep.negative(pass, numbers + (timingEnforced() ? "" : " (timing control informational under validation)"));
    if (!exact) rep.status(Status::Failed, "cull result or raster pixels wrong under concurrency");
    rep.note("spans = GPU timestamps of both queues (first start .. last end), both command buffers released by one shared event; serial = one "
             "command buffer with a Fragment->Dispatch queue barrier between the raster pass and the 3 cull dispatches; samequeue_nobarrier "
             "informational. " + std::to_string(rounds) + " interleaved rounds, rotating order, medians.");
}

} // namespace
} // namespace f5::cull

namespace soc {
SOC_BENCH("F5-S7", "scene.async", "F5-S7: culling on the second queue beside a raster load (async compute)", f5::cull::benchAsync);
} // namespace soc
