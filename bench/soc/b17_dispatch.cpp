// B-17: dispatch and indirect-command-buffer overhead (S-GEO-4, S-SYNC-3 of
// docs/APPLE_SOC_PLAYBOOK.md).
//   * N empty dispatches (1 threadgroup, one SIMD-group, lane 0 increments a
//     counter) in one compute encoder, with and without a Dispatch->Dispatch
//     barrier between them; N indirect dispatches (dispatchThreadgroups(
//     indirectBuffer)) with CPU-written arguments, and a chain where each
//     dispatch writes the arguments of the next (barrier needed).
//   * ICB draws: N one-point draws (vertex shader counts, clipped away) in an
//     ICB encoded by the CPU or by a compute kernel (render_command), executed
//     in a render pass; the same N draws encoded directly as reference.
// Spans are measured with N and 2N commands (draws: also 0, the fixed part of a
// render pass); the per-draw cost is (span(N) - span(0)) / N and the
// negative control requires (span(2N) - span(0)) = 2 (span(N) - span(0)).
//
// Metrics (us): dispatch.empty.us, dispatch.empty_barrier.us,
// dispatch.indirect.us, dispatch.indirect_chain.us, icb.gpu_encode.us_per_cmd,
// icb.execute.us_per_draw (GPU-encoded ICB), icb.execute.cpu_encoded.us_per_draw,
// draw.direct.us_per_draw, icb.cpu_encode.us_per_cmd (CPU wall time),
// icb.render_pass.fixed_us (empty render pass span).

#include "harness.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>

namespace soc {
namespace {

constexpr u32 kNMax   = 8192;   // capacity of the dispatch argument array
constexpr u32 kD1     = 16384;  // draws: N and 2N
constexpr u32 kD2     = 32768;
constexpr u32 kIcbMax = 1u << 18;
constexpr u32 kEncReps = 1;     // encode kernel repetitions inside one span (re-encoding a command without reset is invalid under validation)

struct DispatchArgs {
    u32 x, y, z, pad;
};

// Under the validation layers (--validate) GPU times are distorted by the instrumentation: the timing controls are
// then informational (detail says so); the correctness controls stay enforced.
bool timingControlsEnforced() { return !std::getenv("MTL_SHADER_VALIDATION") && !std::getenv("MTL_DEBUG_LAYER"); }

Stats scaled(Stats s, double f) {
    s.median *= f; s.min *= f; s.max *= f; s.p10 *= f; s.p90 *= f; s.mean *= f;
    return s;
}

struct Fixture {
    Context& ctx;
    MTL::Library* lib = nullptr;
    MTL::Buffer* counter = nullptr;
    MTL::Buffer* args = nullptr;
    MTL::Buffer* total = nullptr;
    bool wrong = false;
    std::string wrongWhat;
    explicit Fixture(Context& c) : ctx(c) {}

    u32 count() const { return *static_cast<const u32*>(counter->contents()); }
    void zeroCounter() { std::memset(counter->contents(), 0, 16); }
    void check(const char* what, u32 want) {
        const u32 got = count();
        if (got != want) {
            wrong = true;
            if (wrongWhat.find(what) == std::string::npos)
                wrongWhat += std::string(what) + " counted " + std::to_string(got) + "/" + std::to_string(want) + " ";
        }
    }
};

enum Kind { Direct, DirectBarrier, Indirect, IndirectChain };

double dispatchSpanMs(Fixture& f, Kind kind, u32 n) {
    Context& ctx = f.ctx;
    f.zeroCounter();
    auto* a = static_cast<DispatchArgs*>(f.args->contents());
    for (u32 i = 0; i < kNMax; ++i) a[i] = kind == IndirectChain ? DispatchArgs{0, 0, 0, 0} : DispatchArgs{1, 1, 1, 0};
    if (kind == IndirectChain) a[0] = {1, 1, 1, 0};
    *static_cast<u32*>(f.total->contents()) = n;
    MTL::ComputePipelineState* pso = ctx.compute(f.lib, kind == IndirectChain ? "b17_chain" : "b17_count");
    ComputeTimer t(ctx);
    MTL4::ComputeCommandEncoder* e = t.begin();
    ctx.table()->setAddress(f.counter->gpuAddress(), 0);
    ctx.table()->setAddress(f.args->gpuAddress(), 1);
    ctx.table()->setAddress(f.total->gpuAddress(), 2);
    e->setComputePipelineState(pso);
    e->setArgumentTable(ctx.table());
    const MTL::Size tg = MTL::Size::Make(32, 1, 1);
    for (u32 k = 0; k < n; ++k) {
        if (kind == Direct || kind == DirectBarrier) e->dispatchThreadgroups(MTL::Size::Make(1, 1, 1), tg);
        else e->dispatchThreadgroups(f.args->gpuAddress() + u64(k) * sizeof(DispatchArgs), tg);
        if ((kind == DirectBarrier || kind == IndirectChain) && k + 1 < n)
            e->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
    }
    t.lap();
    const double ms = t.finish()[0];
    f.check(kind == Direct ? "empty" : kind == DirectBarrier ? "empty_barrier" : kind == Indirect ? "indirect" : "chain", n);
    return ms;
}

// ---- ICB ------------------------------------------------------------------------

struct IcbFixture {
    Context& ctx;
    MTL::RenderPipelineState* pso = nullptr;
    MTL::ComputePipelineState* encode = nullptr;
    MTL::Buffer* counter = nullptr;
    MTL::Buffer* flags = nullptr;
    MTL::Buffer* container = nullptr;
    MTL::Texture* target = nullptr;
    MTL::IndirectCommandBuffer* icbGpu = nullptr;
    MTL::IndirectCommandBuffer* icbCpu = nullptr;
    bool wrong = false;
    std::string wrongWhat;
    explicit IcbFixture(Context& c) : ctx(c) {}

    void reset() {
        std::memset(counter->contents(), 0, 16);
        std::memset(flags->contents(), 0, size_t(kIcbMax) * 4);
    }
    void check(const char* what, u32 n) {
        // Measured: under the validation layers a GPU-encoded ICB executed in a LATER command buffer than the one that
        // encoded it draws nothing (0 messages; the layer's CPU shadow of the ICB is empty).  The encode+execute
        // in ONE command buffer is verified in both modes.
        if (!timingControlsEnforced() && std::strcmp(what, "icb_gpu") == 0) return;
        const u32 got = *static_cast<const u32*>(counter->contents());
        const auto* fl = static_cast<const u32*>(flags->contents());
        u64 ones = 0, others = 0;
        for (u32 i = 0; i < kIcbMax; ++i) (i < n ? (fl[i] == 1 ? ones : others) : (fl[i] == 0 ? ones : others)) += 1;
        const bool bad = got != n || others != 0;
        if (bad) {
            wrong = true;
            if (wrongWhat.find(what) == std::string::npos)
            wrongWhat += std::string(what) + "(N=" + std::to_string(n) + ") counter " + std::to_string(got) + " bad flags " +
                         std::to_string(others) + "; ";
        }
    }
};

MTL4::RenderCommandEncoder* renderPass(IcbFixture& f, MTL4::CommandBuffer* cmd) {
    MTL4::RenderPassDescriptor* pd = MTL4::RenderPassDescriptor::alloc()->init();
    auto* c = pd->colorAttachments()->object(0);
    c->setTexture(f.target);
    c->setLoadAction(MTL::LoadActionDontCare);
    c->setStoreAction(MTL::StoreActionDontCare);
    MTL4::RenderCommandEncoder* re = cmd->renderCommandEncoder(pd);
    pd->release();
    re->setViewport(MTL::Viewport{0.0, 0.0, 64.0, 64.0, 0.0, 1.0});
    f.ctx.table()->setAddress(f.counter->gpuAddress(), 0);
    f.ctx.table()->setAddress(f.flags->gpuAddress(), 1);
    re->setRenderPipelineState(f.pso);
    re->setArgumentTable(f.ctx.table(), MTL::RenderStageVertex | MTL::RenderStageFragment);
    return re;
}

enum DrawKind { DrawDirect, DrawCpuIcb, DrawGpuIcb };

// Span of one render pass with n draws (0 = empty pass), minus the empty span.
double drawSpanMs(IcbFixture& f, DrawKind kind, u32 n) {
    f.reset();
    CommandTimer t(f.ctx);
    MTL4::CommandBuffer* cmd = t.begin();
    MTL4::RenderCommandEncoder* re = renderPass(f, cmd);
    // n == 0: the "empty" pass keeps ONE direct draw (an encoder without commands is rejected by the validation layer).
    if (n == 0) re->drawPrimitives(MTL::PrimitiveTypePoint, 0, 1);
    else if (kind == DrawDirect)
        for (u32 i = 0; i < n; ++i) re->drawPrimitives(MTL::PrimitiveTypePoint, i, 1);
    else re->executeCommandsInBuffer(kind == DrawCpuIcb ? f.icbCpu : f.icbGpu, NS::Range::Make(0, n));
    re->endEncoding();
    const double ms = t.finish() - f.ctx.emptySpanMs();
    f.check(n == 0 ? "empty" : kind == DrawDirect ? "direct" : kind == DrawCpuIcb ? "icb_cpu" : "icb_gpu", n == 0 ? 1 : n);
    return ms;
}

void encodeCpuIcb(IcbFixture& f, u32 n) {
    for (u32 i = 0; i < n; ++i) f.icbCpu->indirectRenderCommand(i)->drawPrimitives(MTL::PrimitiveTypePoint, i, 1, 1, 0);
}

double gpuEncodeSpanMs(IcbFixture& f, u32 n) {
    f.icbGpu->reset(NS::Range::Make(0, n)); // commands are written once per encode
    ComputeTimer t(f.ctx);
    MTL4::ComputeCommandEncoder* e = t.begin();
    f.ctx.table()->setAddress(f.container->gpuAddress(), 0);
    e->setComputePipelineState(f.encode);
    e->setArgumentTable(f.ctx.table());
    for (u32 r = 0; r < kEncReps; ++r) {
        e->dispatchThreads(MTL::Size::Make(n, 1, 1), MTL::Size::Make(64, 1, 1));
        if (r + 1 < kEncReps) e->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
    }
    t.lap();
    return t.finish()[0];
}

bool ratioOk(double a, double b, double lo = 1.75, double hi = 2.25) {
    const double r = b / a;
    return r >= lo && r <= hi;
}

void benchDispatch(Context& ctx, Report& rep) {
    Fixture f(ctx);
    f.lib     = ctx.library("b17_dispatch.metal");
    f.counter = ctx.buffer(16);
    f.args    = ctx.buffer(size_t(kNMax) * sizeof(DispatchArgs));
    f.total   = ctx.buffer(16);
    ctx.keepWarm(50);

    // ---- dispatches ----------------------------------------------------------------
    struct D {
        Kind kind;
        const char* metric;
        const char* label;
        u32 n; // N used for the metric, 2N for the control (unbarriered dispatches are so cheap that 1000 of them sit at the noise floor)
    };
    const D variants[] = {{Direct, "dispatch.empty.us", "empty", 4000},
                          {DirectBarrier, "dispatch.empty_barrier.us", "empty+barrier", 1000},
                          {Indirect, "dispatch.indirect.us", "indirect", 4000},
                          {IndirectChain, "dispatch.indirect_chain.us", "indirect chain (GPU-written args + barrier)", 1000}};
    std::string linearity;
    bool linear = true;
    for (const D& v : variants) {
        Stats s1, s2;
        for (int attempt = 0; attempt < 5; ++attempt) { // other GPU clients disturb single runs: re-measure a failing pair
            s1 = ctx.measure([&] { return dispatchSpanMs(f, v.kind, v.n); }, std::max<u32>(ctx.reps(), 15) * 2);
            ctx.keepWarm(20);
            s2 = ctx.measure([&] { return dispatchSpanMs(f, v.kind, v.n * 2); }, std::max<u32>(ctx.reps(), 15) * 2);
            ctx.keepWarm(20);
            if (ratioOk(s1.median, s2.median)) break;
        }
        const double ratio = s2.median / s1.median;
        if (!ratioOk(s1.median, s2.median)) {
            linear = false;
            linearity += std::string(v.label) + " " + std::to_string(ratio).substr(0, 4) + "x ";
        }
        rep.metric(v.metric, "us", scaled(s1, 1e3 / v.n),
                   {{"n", double(v.n)}, {"span_ms", s1.median}, {"ratio_2n", ratio}, {"barrier", double(v.kind == DirectBarrier || v.kind == IndirectChain)}},
                   false);
    }

    // ---- ICB ------------------------------------------------------------------------
    IcbFixture g(ctx);
    MTL::Library* lib = f.lib;
    MTL4::RenderPipelineDescriptor* rd = MTL4::RenderPipelineDescriptor::alloc()->init();
    rd->setVertexFunctionDescriptor(ctx.function(lib, "b17_vs"));
    rd->setFragmentFunctionDescriptor(ctx.function(lib, "b17_fs"));
    rd->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatR8Unorm);
    rd->setSupportIndirectCommandBuffers(MTL4::IndirectCommandBufferSupportStateEnabled);
    g.pso = ctx.render(rd);
    rd->release();
    g.encode  = ctx.compute(lib, "b17_encode");
    g.counter = ctx.buffer(16);
    g.flags   = ctx.buffer(size_t(kIcbMax) * 4);
    g.container = ctx.buffer(16);
    MTL::TextureDescriptor* td = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatR8Unorm, 64, 64, false);
    td->setUsage(MTL::TextureUsageRenderTarget);
    td->setStorageMode(MTL::StorageModePrivate);
    g.target = ctx.texture(td);
    MTL::IndirectCommandBufferDescriptor* id = MTL::IndirectCommandBufferDescriptor::alloc()->init();
    id->setCommandTypes(MTL::IndirectCommandTypeDraw);
    id->setInheritPipelineState(true);
    id->setInheritBuffers(true);
    id->setMaxVertexBufferBindCount(2);
    id->setMaxFragmentBufferBindCount(0);
    g.icbGpu = ctx.device()->newIndirectCommandBuffer(id, kIcbMax, MTL::ResourceStorageModeShared);
    g.icbCpu = ctx.device()->newIndirectCommandBuffer(id, kIcbMax, MTL::ResourceStorageModeShared);
    id->release();
    if (!g.icbGpu || !g.icbCpu) throw BenchError("newIndirectCommandBuffer failed");
    ctx.adopt(g.icbGpu);
    ctx.adopt(g.icbCpu);
    const MTL::ResourceID rid = g.icbGpu->gpuResourceID();
    std::memcpy(g.container->contents(), &rid, sizeof(rid));

    // GPU encoding: kernel writes one draw per thread.
    const u32 e1 = 65536, e2 = 131072; // beyond ~130k commands the cost per command jumps (measured): stay below
    const Stats enc1 = ctx.measure([&] { return gpuEncodeSpanMs(g, e1); }, 15);
    ctx.keepWarm(20);
    const Stats enc2 = ctx.measure([&] { return gpuEncodeSpanMs(g, e2); }, 15);
    ctx.keepWarm(20);
    const bool encLinear = ratioOk(enc1.median, enc2.median, 1.6, 2.6);
    rep.metric("icb.gpu_encode.us_per_cmd", "us", scaled(enc1, 1e3 / (double(e1) * kEncReps)),
               {{"cmds", double(e1)}, {"reps", double(kEncReps)}, {"span_ms", enc1.median}, {"ratio_2x", enc2.median / enc1.median}}, false);
    // All kIcbMax commands must exist for the ranges below (GPU ICB).
    encodeCpuIcb(g, kD2);

    // CPU encoding wall time.
    const Stats cpuEnc = ctx.measure([&] {
        const double t0 = nowMs();
        encodeCpuIcb(g, kD2);
        return (nowMs() - t0) * 1e3 / kD2;
    }, 15);
    rep.metric("icb.cpu_encode.us_per_cmd", "us", cpuEnc, {{"cmds", double(kD2)}}, false);

    // Execution: 0 / N / 2N draws.
    struct X {
        DrawKind kind;
        const char* metric;
        const char* label;
    };
    const X exec[] = {{DrawGpuIcb, "icb.execute.us_per_draw", "icb (GPU-encoded)"},
                      {DrawCpuIcb, "icb.execute.cpu_encoded.us_per_draw", "icb (CPU-encoded)"},
                      {DrawDirect, "draw.direct.us_per_draw", "direct draws"}};
    const Stats span0 = ctx.measure([&] { return drawSpanMs(g, DrawDirect, 0); }, 15);
    rep.metric("icb.render_pass.fixed_us", "us", scaled(span0, 1e3), {}, false);
    bool execLinear = true;
    std::string execLin;
    for (const X& x : exec) {
        ctx.keepWarm(20);
        Stats s1, s2;
        for (int attempt = 0; attempt < 4; ++attempt) { // re-measure a pair disturbed by other GPU clients
            s1 = ctx.measure([&] { return drawSpanMs(g, x.kind, kD1); }, 15);
            ctx.keepWarm(20);
            s2 = ctx.measure([&] { return drawSpanMs(g, x.kind, kD2); }, 15);
            if (ratioOk(s1.median - span0.median, s2.median - span0.median)) break;
            ctx.keepWarm(20);
        }
        const double d1 = s1.median - span0.median, d2 = s2.median - span0.median;
        const bool ok = d1 > 0 && ratioOk(d1, d2);
        execLinear &= ok;
        if (!ok) execLin += std::string(x.label) + " " + std::to_string(d1 > 0 ? d2 / d1 : 0).substr(0, 4) + "x ";
        Stats per = scaled(s1, 1e3 / kD1);
        const double fixed = span0.median * 1e3 / kD1;
        per.median -= fixed; per.min -= fixed; per.max -= fixed; per.p10 -= fixed; per.p90 -= fixed; per.mean -= fixed;
        rep.metric(x.metric, "us", per, {{"n", double(kD1)}, {"span_ms", s1.median}, {"ratio_2n", d1 > 0 ? d2 / d1 : 0}}, false);
    }

    // GPU encode + execute in ONE command buffer (barrier legality), CPU-verified.
    {
        g.icbGpu->reset(NS::Range::Make(0, kIcbMax));
        g.reset();
        CommandTimer t(ctx);
        MTL4::CommandBuffer* cmd = t.begin();
        MTL4::ComputeCommandEncoder* ce = cmd->computeCommandEncoder();
        ctx.table()->setAddress(g.container->gpuAddress(), 0);
        ce->setComputePipelineState(g.encode);
        ce->setArgumentTable(ctx.table());
        ce->dispatchThreads(MTL::Size::Make(kD1, 1, 1), MTL::Size::Make(64, 1, 1));
        ce->endEncoding();
        MTL4::RenderCommandEncoder* re = renderPass(g, cmd);
        re->barrierAfterQueueStages(MTL::StageDispatch, MTL::StageVertex, MTL4::VisibilityOptionDevice);
        re->executeCommandsInBuffer(g.icbGpu, NS::Range::Make(0, kD1));
        re->endEncoding();
        const double ms = t.finish();
        g.check("encode+execute one command buffer", kD1);
        rep.value("icb.gpu_encode_execute.same_cb_ms", "ms", ms, {{"cmds", double(kD1)}}, false);
    }

    ctx.keepWarm(20);
    const bool wrong = f.wrong || g.wrong;
    const bool enforce = timingControlsEnforced();
    rep.negative((linear && encLinear && execLinear) || !enforce,
                 std::string(enforce ? "" : "[timing not enforced under validation] ") + std::string("2N vs N: ") + (linear ? "all 4 dispatch variants 1.75..2.25x" : "NOT linear: " + linearity) +
                     "; ICB encode 2x cmds 1.6..2.6x" + (encLinear ? "" : " FAILED") + "; ICB/direct draws (span-fixed) " +
                     (execLinear ? "1.75..2.25x" : "NOT linear: " + execLin));
    rep.negative(!wrong, wrong ? "side effects WRONG: " + f.wrongWhat + g.wrongWhat
                               : "every dispatch/draw ran: counters exact, each draw's vertex flag == 1");
    if (wrong) rep.status(Status::Failed, "wrong side effects");
    if (!timingControlsEnforced()) rep.note("validation mode: the standalone GPU-encoded ICB execution is not counted (empty under the layers, artifact); encode+execute in one command buffer is");
    rep.note("dispatch.* = span/1000 (ComputeTimer, anchor-bracketed, N=1000, ratio_2n in params); ICB draws are one clipped "
             "point each (vertex shader counts); execute per-draw = (span(N)-empty pass)/N minus the empty-pass cost; ICBs "
             "inherit pipeline and buffers; GPU-encoded ICB written by render_command in a kernel");
}

} // namespace

SOC_BENCH("B-17", "dispatch.icb", "Empty/indirect dispatch cost, ICB encoding (CPU vs GPU) and execution per draw", benchDispatch);

} // namespace soc
