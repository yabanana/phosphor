// F5-S5: which barrier stage on the CONSUMER side makes compute-written
// indirect arguments / ICB commands visible (extends the F2.3 spike,
// bench/barrier_spike, whose data-flow cases never read GPU-written
// *arguments*: those are fetched by the command processor, not by a shader
// stage, so the stage that gates them is not obvious).
//
// Method (F2.3): the producer is slow on purpose (64 threadgroups x 256
// threads spinning a seeded dependent ALU chain for ~6 ms, calibrated at run
// time) and the LAST threadgroup to finish (device atomic) writes the
// consumer's arguments, so the write happens after every spin.  The initial
// ("stale") arguments do nothing (0 instances / 0 threadgroups / empty range
// / no ICB commands) and the consumer counts every vertex / threadgroup /
// thread it runs, so a stale fetch shows as a short count.  20 repetitions
// (own resources) per variant are encoded in ONE command buffer, producer
// encoder then consumer encoder per repetition; "wrong" = repetitions whose
// count differs from the expected one.  Everything is reset on the CPU
// between variants (private storage: in a command buffer committed before).
//
// Consumers:
//   c1_draw_indirect      drawIndexedPrimitives(indirect args)  render encoder after the producer's compute encoder
//   c2_icb_gpu            executeCommandsInBuffer(icb, range), commands encoded by the producer (render_command)
//   c3_icb_range          executeCommandsInBuffer(icb, indirectRangeBuffer), range written by the producer
//   c4_groups_same        dispatchThreadgroups(indirect) in the SAME compute encoder as the producer
//   c4_groups_next        ... in a FOLLOWING compute encoder
//   c5_threads_same/next  dispatchThreads(indirect) (args {grid[3], threadsPerThreadgroup[3]}), same / next encoder
// Barriers (names: <kind>_<after>_<before>): none; q = barrierAfterQueueStages
// at the start of the consumer encoder; p = barrierAfterStages at the end of
// the producer encoder; e = barrierAfterEncoderStages inside the encoder.
// Every pair also runs with the argument / ICB storage private (written by
// the producer only; reset by a fill / resetCommandsInBuffer in a previous
// command buffer).  "ref" rows: arguments preset by the CPU, no producer (the
// consumer must reach the expected count: proves the counting is exact).
//
// Metrics: wrong.<consumer>.<barrier>[.private] (wrong repetitions of 20),
// span_ms.<...> (informational: command buffer GPU time, not a result),
// spin.iters, spin.producer_ms.  rep.note carries the table.
// Negative control: the "none" variants must race at least for c1 and
// c4_groups_same (detection proven; a "none" that never races is
// "inconclusive", never "safe") and every recommended barrier must give 0
// wrong (under validation the layers serialise the work and races may
// disappear: detection is then informational).

#include "f5_common.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace soc {
namespace {

constexpr u32 kReps     = 20;
constexpr u32 kTG       = 64;
constexpr u32 kTGSize   = 256;
constexpr u32 kIcbCmds  = 64;  // commands the producer encodes / the range executes
constexpr u32 kIcbMax   = 128; // ICB capacity (c3 encodes all of them on the CPU, range = {32, 64})
constexpr u32 kDrawInst = 1000;
constexpr u32 kGroups   = 512;
constexpr u32 kThreads  = 16384;
constexpr double kTargetProducerMs = 6.0;

struct Params {
    u32 iters, numTG, salt, k;
};

enum class Cons { Draw, IcbGpu, IcbRange, GroupsSame, GroupsNext, ThreadsSame, ThreadsNext };

struct ConsInfo {
    Cons        c;
    const char* name;
    const char* producer;
    u32         expect;
    bool        render; // consumer is a render encoder
    bool        same;   // consumer in the producer's compute encoder
    u32         argWords;
};
const ConsInfo kCons[] = {
    {Cons::Draw, "c1_draw_indirect", "s5_prod_draw", kDrawInst, true, false, 5},
    {Cons::IcbGpu, "c2_icb_gpu", "s5_prod_icb", kIcbCmds, true, false, 0},
    {Cons::IcbRange, "c3_icb_range", "s5_prod_range", kIcbCmds, true, false, 2},
    {Cons::GroupsSame, "c4_groups_same", "s5_prod_groups", kGroups, false, true, 3},
    {Cons::GroupsNext, "c4_groups_next", "s5_prod_groups", kGroups, false, false, 3},
    {Cons::ThreadsSame, "c5_threads_same", "s5_prod_threads", kThreads, false, true, 6},
    {Cons::ThreadsNext, "c5_threads_next", "s5_prod_threads", kThreads, false, false, 6},
};

struct Barrier {
    const char*  name;
    char         kind; // n none, q queue (consumer encoder start), p after-stages (producer encoder end), e encoder
    MTL::Stages  after, before;
};
constexpr MTL::Stages kD = MTL::StageDispatch, kV = MTL::StageVertex, kF = MTL::StageFragment,
                      kOM = MTL::StageObject | MTL::StageMesh;
const Barrier kRenderBarriers[] = {
    {"none", 'n', 0, 0},
    {"q_dispatch_vertex", 'q', kD, kV},
    {"q_dispatch_fragment", 'q', kD, kF},
    {"q_dispatch_object+mesh", 'q', kD, kOM},
    {"q_dispatch_vertex+fragment", 'q', kD, kV | kF},
    {"p_dispatch_vertex", 'p', kD, kV},
};
const Barrier kSameBarriers[] = {
    {"none", 'n', 0, 0},
    {"e_dispatch_dispatch", 'e', kD, kD},
};
const Barrier kNextBarriers[] = {
    {"none", 'n', 0, 0},
    {"q_dispatch_dispatch", 'q', kD, kD},
    {"p_dispatch_dispatch", 'p', kD, kD},
};

/// The barrier the engine should use for this consumer (the pass/fail control).
const char* recommended(const ConsInfo& ci) {
    if (ci.render) return "q_dispatch_vertex";
    return ci.same ? "e_dispatch_dispatch" : "q_dispatch_dispatch";
}

bool validationActive() { return std::getenv("MTL_SHADER_VALIDATION") || std::getenv("MTL_DEBUG_LAYER"); }

u32 spinCpu(u32 seed, u32 iters) {
    u32 x = seed | 1u;
    for (u32 i = 0; i < iters; ++i) {
        x = x * 1664525u + 1013904223u;
        x ^= x >> 15;
    }
    return x;
}

struct Rig {
    Context& ctx;
    MTL::Library* lib = nullptr;
    MTL::RenderPipelineState* renderPso = nullptr;
    MTL::Texture* target = nullptr;
    MTL::Buffer* index = nullptr;
    // per repetition
    std::vector<MTL::Buffer*> scratch, done, counter, params, args, argsPriv, container, containerPriv;
    std::vector<MTL::IndirectCommandBuffer*> icb, icbPriv, icbCpu;
    u32 iters = 1;
    explicit Rig(Context& c) : ctx(c) {}
};

struct Outcome {
    u32 wrong = 0;
    u32 minCount = ~0u, maxCount = 0;
    std::string fault; // producer misbehaved: a bug in the spike, not a race
    double ms = 0;
    std::string error; // API / GPU error text
};

void presetArgs(const ConsInfo& ci, u32* a, bool correct) {
    std::memset(a, 0, 8 * sizeof(u32));
    if (!correct) return;
    switch (ci.c) {
    case Cons::Draw: a[0] = 1; a[1] = kDrawInst; break;
    case Cons::IcbRange: a[0] = 32; a[1] = kIcbCmds; break;
    case Cons::GroupsSame: case Cons::GroupsNext: a[0] = kGroups; a[1] = 1; a[2] = 1; break;
    case Cons::ThreadsSame: case Cons::ThreadsNext: a[0] = kThreads; a[1] = 1; a[2] = 1; a[3] = 64; a[4] = 1; a[5] = 1; break;
    default: break;
    }
}

MTL4::RenderCommandEncoder* beginRender(Rig& r, MTL4::CommandBuffer* cmd) {
    MTL4::RenderPassDescriptor* pd = MTL4::RenderPassDescriptor::alloc()->init();
    auto* c = pd->colorAttachments()->object(0);
    c->setTexture(r.target);
    c->setLoadAction(MTL::LoadActionDontCare);
    c->setStoreAction(MTL::StoreActionDontCare);
    MTL4::RenderCommandEncoder* re = cmd->renderCommandEncoder(pd);
    pd->release();
    re->setViewport(MTL::Viewport{0.0, 0.0, 64.0, 64.0, 0.0, 1.0});
    return re;
}

/// One variant: 20 repetitions in one command buffer.  `reference`: arguments
/// preset by the CPU and no producer.  `iters`: spin length (0 = no spin).
Outcome runVariant(Rig& r, const ConsInfo& ci, const Barrier& b, bool priv, bool reference, u32 iters) {
    Context& ctx = r.ctx;
    Outcome out;
    const bool icbGpu = ci.c == Cons::IcbGpu;
    const bool privIcb = priv && icbGpu;
    const bool privArgs = priv && !icbGpu;
    u32 expectArgs[8];
    presetArgs(ci, expectArgs, true);

    // ---- CPU reset ---------------------------------------------------------------------------------------
    for (u32 i = 0; i < kReps; ++i) {
        std::memset(r.done[i]->contents(), 0, 16);
        std::memset(r.counter[i]->contents(), 0, 16);
        Params p{iters, kTG, i * 7u + 1u, kIcbCmds};
        std::memcpy(r.params[i]->contents(), &p, sizeof(p));
        std::memset(r.scratch[i]->contents(), 0, kTG * kTGSize * 4);
        presetArgs(ci, static_cast<u32*>(r.args[i]->contents()), reference);
        if (icbGpu && !priv) r.icb[i]->reset(NS::Range::Make(0, kIcbCmds));
    }
    // ---- private storage: stale values written by the GPU in a command buffer committed before -----------------
    if (priv) {
        MTL4::CommandBuffer* c0 = ctx.beginCommands();
        MTL4::ComputeCommandEncoder* ce = c0->computeCommandEncoder();
        ce->setLabel(NS::String::string("s5 init", NS::UTF8StringEncoding));
        for (u32 i = 0; i < kReps; ++i) {
            if (privIcb) ce->resetCommandsInBuffer(r.icbPriv[i], NS::Range::Make(0, kIcbCmds));
            else ce->fillBuffer(r.argsPriv[i], NS::Range::Make(0, 32), 0);
        }
        ce->endEncoding();
        ctx.submit();
    }

    // ---- main command buffer -------------------------------------------------------------------------------
    MTL4::ArgumentTable* t = ctx.table();
    MTL4::CommandBuffer* cmd = ctx.beginCommands();
    MTL::ComputePipelineState* prod = ctx.compute(r.lib, ci.producer);
    MTL::ComputePipelineState* cons = nullptr;
    if (!ci.render) cons = ctx.compute(r.lib, (ci.c == Cons::ThreadsSame || ci.c == Cons::ThreadsNext) ? "s5_count_threads" : "s5_count_groups");
    for (u32 i = 0; i < kReps; ++i) {
        MTL::Buffer* argBuf = privArgs ? r.argsPriv[i] : r.args[i];
        const MTL::GPUAddress argAddr = argBuf->gpuAddress();
        MTL4::ComputeCommandEncoder* ce = nullptr;
        if (!reference || ci.same) {
            ce = cmd->computeCommandEncoder();
            ce->setLabel(NS::String::string("s5 producer", NS::UTF8StringEncoding));
            if (!reference) {
                t->setAddress(r.scratch[i]->gpuAddress(), 0);
                t->setAddress(r.done[i]->gpuAddress(), 1);
                t->setAddress(argAddr, 2);
                t->setAddress(r.params[i]->gpuAddress(), 3);
                t->setAddress((privIcb ? r.containerPriv[i] : r.container[i])->gpuAddress(), 4);
                ce->setComputePipelineState(prod);
                ce->setArgumentTable(t);
                ce->dispatchThreadgroups(MTL::Size::Make(kTG, 1, 1), MTL::Size::Make(kTGSize, 1, 1));
            }
        }
        if (ci.same) {
            if (b.kind == 'e') ce->barrierAfterEncoderStages(b.after, b.before, MTL4::VisibilityOptionDevice);
            t->setAddress(r.counter[i]->gpuAddress(), 5);
            ce->setComputePipelineState(cons);
            ce->setArgumentTable(t);
            if (ci.c == Cons::GroupsSame) ce->dispatchThreadgroups(argAddr, MTL::Size::Make(64, 1, 1));
            else ce->dispatchThreads(argAddr);
            ce->endEncoding();
            continue;
        }
        if (ce) {
            if (b.kind == 'p') ce->barrierAfterStages(b.after, b.before, MTL4::VisibilityOptionDevice);
            ce->endEncoding();
        }
        if (ci.render) {
            MTL4::RenderCommandEncoder* re = beginRender(r, cmd);
            re->setLabel(NS::String::string("s5 consumer", NS::UTF8StringEncoding));
            if (b.kind == 'q') re->barrierAfterQueueStages(b.after, b.before, MTL4::VisibilityOptionDevice);
            t->setAddress(r.counter[i]->gpuAddress(), 5);
            re->setRenderPipelineState(r.renderPso);
            re->setArgumentTable(t, MTL::RenderStageVertex | MTL::RenderStageFragment);
            if (ci.c == Cons::Draw)
                re->drawIndexedPrimitives(MTL::PrimitiveTypePoint, MTL::IndexTypeUInt32, r.index->gpuAddress(), 16, argAddr);
            else if (ci.c == Cons::IcbGpu)
                re->executeCommandsInBuffer(privIcb ? r.icbPriv[i] : r.icb[i], NS::Range::Make(0, kIcbCmds));
            else
                re->executeCommandsInBuffer(r.icbCpu[i], argAddr);
            re->endEncoding();
        } else {
            MTL4::ComputeCommandEncoder* c2 = cmd->computeCommandEncoder();
            c2->setLabel(NS::String::string("s5 consumer", NS::UTF8StringEncoding));
            if (b.kind == 'q') c2->barrierAfterQueueStages(b.after, b.before, MTL4::VisibilityOptionDevice);
            t->setAddress(r.counter[i]->gpuAddress(), 5);
            c2->setComputePipelineState(cons);
            c2->setArgumentTable(t);
            if (ci.c == Cons::GroupsNext) c2->dispatchThreadgroups(argAddr, MTL::Size::Make(64, 1, 1));
            else c2->dispatchThreads(argAddr);
            c2->endEncoding();
        }
    }
    try {
        out.ms = ctx.submit();
    } catch (const BenchError& e) {
        out.error = e.what();
        return out;
    }

    // ---- check -------------------------------------------------------------------------------------------------
    for (u32 i = 0; i < kReps; ++i) {
        const u32 got = *static_cast<const u32*>(r.counter[i]->contents());
        out.minCount = std::min(out.minCount, got);
        out.maxCount = std::max(out.maxCount, got);
        if (got != ci.expect) ++out.wrong;
        if (!reference) {
            const u32 d = *static_cast<const u32*>(r.done[i]->contents());
            if (d != kTG) out.fault += "done=" + std::to_string(d) + " ";
            // every producer must have written exactly the final arguments (shared storage: readable)
            if (!priv && ci.argWords && std::memcmp(r.args[i]->contents(), expectArgs, ci.argWords * 4) != 0) out.fault += "args ";
        }
    }
    return out;
}

/// Producer alone, `iters` iterations: GPU ms (command buffer) and a CPU
/// check of a few threads' results.
double producerSpan(Rig& r, u32 iters, std::string& fault) {
    Context& ctx = r.ctx;
    Params p{iters, kTG, 1, kIcbCmds};
    std::memcpy(r.params[0]->contents(), &p, sizeof(p));
    std::memset(r.done[0]->contents(), 0, 16);
    MTL4::CommandBuffer* cmd = ctx.beginCommands();
    MTL4::ComputeCommandEncoder* ce = cmd->computeCommandEncoder();
    ce->setLabel(NS::String::string("s5 calibrate", NS::UTF8StringEncoding));
    MTL4::ArgumentTable* t = ctx.table();
    t->setAddress(r.scratch[0]->gpuAddress(), 0);
    t->setAddress(r.done[0]->gpuAddress(), 1);
    t->setAddress(r.args[0]->gpuAddress(), 2);
    t->setAddress(r.params[0]->gpuAddress(), 3);
    ce->setComputePipelineState(ctx.compute(r.lib, "s5_prod_draw"));
    ce->setArgumentTable(t);
    ce->dispatchThreadgroups(MTL::Size::Make(kTG, 1, 1), MTL::Size::Make(kTGSize, 1, 1));
    ce->endEncoding();
    const double ms = ctx.submit();
    u64 rng = 0x1234567ull + iters;
    const auto* s = static_cast<const u32*>(r.scratch[0]->contents());
    for (int k = 0; k < 8; ++k) {
        const u32 gid = static_cast<u32>(xorshift64(rng) % (kTG * kTGSize));
        if (s[gid] != spinCpu(gid * 2654435761u + 1u, iters)) fault += "scratch[" + std::to_string(gid) + "] ";
    }
    if (*static_cast<const u32*>(r.done[0]->contents()) != kTG) fault += "done ";
    return ms;
}

struct Row {
    const ConsInfo* ci;
    const Barrier* b;
    bool priv, ref;
    Outcome o;
};

std::string cell(const Row& r) {
    std::string s = r.b ? r.b->name : "ref";
    if (r.ref) s = "ref";
    s += " " + std::to_string(r.o.wrong) + "/" + std::to_string(kReps);
    if (!r.o.error.empty()) s += " ERROR";
    else if (r.o.wrong) s += " (count " + std::to_string(r.o.minCount) + ".." + std::to_string(r.o.maxCount) + " of " + std::to_string(r.ci->expect) + ")";
    return s;
}

void benchBarriers(Context& ctx, Report& rep) {
    Rig r(ctx);
    r.lib = f5::f5Library(ctx, "s5_barriers.metal");

    // ---- pipelines / resources ------------------------------------------------------------------------------
    MTL4::RenderPipelineDescriptor* rd = MTL4::RenderPipelineDescriptor::alloc()->init();
    rd->setVertexFunctionDescriptor(ctx.function(r.lib, "s5_vs"));
    rd->setFragmentFunctionDescriptor(ctx.function(r.lib, "s5_fs"));
    rd->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatR8Unorm);
    rd->setSupportIndirectCommandBuffers(MTL4::IndirectCommandBufferSupportStateEnabled);
    r.renderPso = ctx.render(rd);
    rd->release();
    MTL::TextureDescriptor* td = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatR8Unorm, 64, 64, false);
    td->setUsage(MTL::TextureUsageRenderTarget);
    td->setStorageMode(MTL::StorageModePrivate);
    r.target = ctx.texture(td);
    r.index = ctx.buffer(16);
    std::memset(r.index->contents(), 0, 16);

    MTL::IndirectCommandBufferDescriptor* id = MTL::IndirectCommandBufferDescriptor::alloc()->init();
    id->setCommandTypes(MTL::IndirectCommandTypeDraw);
    id->setInheritPipelineState(true);
    id->setInheritBuffers(true);
    id->setMaxVertexBufferBindCount(8);
    id->setMaxFragmentBufferBindCount(0);
    for (u32 i = 0; i < kReps; ++i) {
        r.scratch.push_back(ctx.buffer(size_t(kTG) * kTGSize * 4));
        r.done.push_back(ctx.buffer(16));
        r.counter.push_back(ctx.buffer(16));
        r.params.push_back(ctx.buffer(16));
        r.args.push_back(ctx.buffer(64));
        r.argsPriv.push_back(ctx.buffer(64, MTL::ResourceStorageModePrivate));
        r.container.push_back(ctx.buffer(16));
        r.containerPriv.push_back(ctx.buffer(16));
        MTL::IndirectCommandBuffer* a = ctx.device()->newIndirectCommandBuffer(id, kIcbMax, MTL::ResourceStorageModeShared);
        MTL::IndirectCommandBuffer* p = ctx.device()->newIndirectCommandBuffer(id, kIcbMax, MTL::ResourceStorageModePrivate);
        MTL::IndirectCommandBuffer* c = ctx.device()->newIndirectCommandBuffer(id, kIcbMax, MTL::ResourceStorageModeShared);
        if (!a || !p || !c) throw BenchError("newIndirectCommandBuffer failed (shared/private)");
        ctx.adopt(a); ctx.adopt(p); ctx.adopt(c);
        r.icb.push_back(a); r.icbPriv.push_back(p); r.icbCpu.push_back(c);
        const MTL::ResourceID ra = a->gpuResourceID(), rp = p->gpuResourceID();
        std::memcpy(r.container[i]->contents(), &ra, sizeof(ra));
        std::memcpy(r.containerPriv[i]->contents(), &rp, sizeof(rp));
        for (u32 k = 0; k < kIcbMax; ++k) c->indirectRenderCommand(k)->drawPrimitives(MTL::PrimitiveTypePoint, k, 1, 1, 0);
    }
    id->release();
    ctx.commitResidency();
    ctx.keepWarm(50);

    // ---- calibrate the spin so one producer takes ~kTargetProducerMs ----------------------------------------------
    std::string fault;
    u32 iters = 1u << 17;
    double ms = producerSpan(r, iters, fault);
    ms = producerSpan(r, iters, fault);
    iters = static_cast<u32>(std::clamp(double(iters) * kTargetProducerMs / std::max(ms, 0.05), 1000.0, 3.0e8));
    ms = producerSpan(r, iters, fault);
    ms = producerSpan(r, iters, fault);
    r.iters = iters;
    rep.value("spin.iters", "count", iters, {}, false);
    rep.value("spin.producer_ms", "ms", ms, {}, false);
    ctx.log("[F5-S5] spin %u iterations, producer %.2f ms%s", iters, ms, fault.empty() ? "" : (" FAULT " + fault).c_str());

    // ---- variants ---------------------------------------------------------------------------------------------------
    std::vector<Row> rows;
    std::string faults = fault.empty() ? "" : "calibration: " + fault + "; ";
    std::string errors;
    for (const ConsInfo& ci : kCons) {
        const Barrier* list = ci.render ? kRenderBarriers : ci.same ? kSameBarriers : kNextBarriers;
        const size_t n = ci.render ? std::size(kRenderBarriers) : ci.same ? std::size(kSameBarriers) : std::size(kNextBarriers);
        for (int storage = 0; storage < 2; ++storage) {
            const bool priv = storage == 1;
            // Measured artefact: under MTL_SHADER_VALIDATION any executeCommandsInBuffer(icb, indirectRangeBuffer)
            // (even CPU-written range, CPU-encoded ICB, no producer) crashes the layer while it decodes its report
            // ("NSArrayM insertObject:atIndex: object cannot be nil" in MTL4GPUDebugCommandQueue, abort 134).
            // c3 is therefore skipped under shader validation (the API-validation-only run covers it).
            if (ci.c == Cons::IcbRange && std::getenv("MTL_SHADER_VALIDATION")) {
                if (storage == 0) rep.note("c3_icb_range SKIPPED under MTL_SHADER_VALIDATION: the layer aborts decoding its report (tools artefact, also without producer)");
                continue;
            }
            if (priv && ctx.quick() && !ci.render) continue;
            // reference: preset arguments, no producer (c2 has no argument block: c3's reference covers ICB execution)
            if (!priv && ci.c != Cons::IcbGpu) {
                Row row{&ci, nullptr, false, true, runVariant(r, ci, list[0], false, true, r.iters)};
                rows.push_back(row);
            }
            for (size_t k = 0; k < n; ++k) {
                if (priv && ctx.quick() && list[k].kind != 'n' && std::strcmp(list[k].name, recommended(ci)) != 0) continue;
                Row row{&ci, &list[k], priv, false, runVariant(r, ci, list[k], priv, false, r.iters)};
                if (!row.o.fault.empty()) faults += std::string(ci.name) + "/" + list[k].name + (priv ? "/private" : "") + ": " + row.o.fault + "; ";
                if (!row.o.error.empty()) errors += std::string(ci.name) + "/" + list[k].name + (priv ? "/private" : "") + ": " + row.o.error + "; ";
                rows.push_back(row);
                ctx.keepWarm(10);
            }
        }
    }
    // informational: the same "none" races only because the producer is slow
    Outcome nospin = runVariant(r, kCons[0], kRenderBarriers[0], false, false, 0);

    // ---- report -----------------------------------------------------------------------------------------------------
    bool refOk = true, recOk = true, detC1 = false, detC4 = false;
    std::string recBad, inconclusive;
    for (const Row& row : rows) {
        const std::string suffix = row.priv ? ".private" : "";
        const std::string bn = row.ref ? "ref" : row.b->name;
        rep.value(std::string("wrong.") + row.ci->name + "." + bn + suffix, "count", row.o.wrong, {{"reps", double(kReps)}}, false);
        rep.value(std::string("span_ms.") + row.ci->name + "." + bn + suffix, "ms", row.o.ms, {}, false);
        if (row.ref) refOk &= row.o.wrong == 0 && row.o.error.empty();
        else if (row.b->kind == 'n') {
            if (!row.priv && row.ci->c == Cons::Draw) detC1 = row.o.wrong > 0;
            if (!row.priv && row.ci->c == Cons::GroupsSame) detC4 = row.o.wrong > 0;
            if (row.o.wrong == 0 && row.o.error.empty()) inconclusive += std::string(row.ci->name) + (row.priv ? "/private " : " ");
        } else if (std::strcmp(row.b->name, recommended(*row.ci)) == 0) {
            if (row.o.wrong != 0 || !row.o.error.empty()) {
                recOk = false;
                recBad += std::string(row.ci->name) + (row.priv ? "/private " : " ");
            }
        }
    }
    rep.value("wrong.c1_draw_indirect.none_nospin", "count", nospin.wrong, {{"reps", double(kReps)}}, false);
    for (const ConsInfo& ci : kCons) {
        for (int storage = 0; storage < 2; ++storage) {
            std::string line = std::string(ci.name) + (storage ? " [private]" : " [shared]") + ": ";
            bool any = false;
            for (const Row& row : rows)
                if (row.ci == &ci && row.priv == (storage == 1)) {
                    line += (any ? " | " : "") + cell(row) + (row.b && row.b->kind == 'n' && row.o.wrong == 0 && row.o.error.empty() ? " INCONCLUSIVE" : "");
                    any = true;
                }
            if (any) rep.note(line);
        }
    }
    rep.note("c1 none with spin=0: " + std::to_string(nospin.wrong) + "/" + std::to_string(kReps) + " wrong (informational: the race needs a slow producer)");
    if (!errors.empty()) rep.note("API/GPU errors: " + errors);
    if (!inconclusive.empty()) rep.note("none variants that never raced (INCONCLUSIVE, not safe): " + inconclusive);

    const bool enforceDet = !validationActive();
    const bool detOk = (detC1 && detC4) || !enforceDet;
    std::string detail = std::string(detC1 ? "c1 none raced" : "c1 none did NOT race") + ", " +
                         (detC4 ? "c4_groups_same none raced" : "c4_groups_same none did NOT race") +
                         (enforceDet ? "" : " [detection informational under validation]") + "; recommended barriers " +
                         (recOk ? "0 wrong everywhere (c1-c3 q_dispatch_vertex, c4/c5 e/q_dispatch_dispatch)" : "WRONG for " + recBad) +
                         "; references " + (refOk ? "exact" : "NOT exact") + (errors.empty() ? "" : "; errors") +
                         (faults.empty() ? "" : "; producer faults: " + faults);
    rep.negative(detOk && recOk && refOk && errors.empty() && faults.empty(), detail);
    if (!faults.empty() || !errors.empty()) rep.status(Status::Failed, "producer fault or GPU error: " + faults + errors);
}

} // namespace

SOC_BENCH("F5-S5", "scene.indirect_barriers",
          "Consumer-side barrier stage that makes compute-written indirect arguments / ICB commands visible", benchBarriers);

} // namespace soc
