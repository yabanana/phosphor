// F6-S4: barriers of the F6 mesh path (docs/plans/F6.md, spike S4), method
// of F5-S5 (bench/f5_spike/s5_barriers.cpp): the producer is slow on purpose
// (64 x 256 threads spinning a seeded ALU chain for ~6 ms, calibrated) and
// its LAST threadgroup writes what the consumer needs; the stale value makes
// the consumer count short.  20 repetitions per variant in one command buffer.
//
// Cases (consumer <- producer):
//   m1_mesh_args    drawMeshThreadgroups(indirect args) <- compute writes the
//                   args (object threadgroups counted; stale args launch 0)
//   m2_object_read  object shader reads a buffer <- compute writes it (CPU
//                   args; each object threadgroup counts a magic word)
//   o1_object_write compute reads a buffer <- the OBJECT shader of the
//                   previous render encoder wrote it (slow object shader;
//                   the B flags of phase A -> meshlet_b_count)
// Barriers: none, and q = barrierAfterQueueStages at the start of the
// consumer encoder with the listed after -> before stages.
//
// Metrics: wrong.<case>.<barrier> (repetitions of 20 with a short count),
// span_ms.* (informational).  Negative control: the "none" variant must race
// for every case (otherwise "inconclusive", never "safe") and the barrier the
// engine uses must give 0 wrong: m1/m2 q Dispatch -> Vertex|Object|Mesh, o1
// q Object|Mesh -> Dispatch (the render graph's stages for these accesses).

#include "f6_common.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace soc {
namespace {

constexpr u32 kReps   = 20;
constexpr u32 kTG     = 64;
constexpr u32 kTGSize = 256;
constexpr u32 kObjTG  = 512; // object threadgroups (m1 expected count, m2/o1 entries)
constexpr double kTargetProducerMs = 6.0;

struct Params {
    u32 iters, numTG, salt, k;
};

enum class Case { MeshArgs, ObjectRead, ObjectWrite };
struct CaseInfo {
    Case c;
    const char* name;
    u32 expect;
};
const CaseInfo kCases[] = {
    {Case::MeshArgs, "m1_mesh_args", kObjTG},
    {Case::ObjectRead, "m2_object_read", kObjTG},
    {Case::ObjectWrite, "o1_object_write", kObjTG},
};

struct Barrier {
    const char* name;
    MTL::Stages after, before; // 0/0 = none
};
constexpr MTL::Stages kD = MTL::StageDispatch, kV = MTL::StageVertex, kO = MTL::StageObject, kM = MTL::StageMesh,
                      kF = MTL::StageFragment;
const Barrier kToRender[] = {
    {"none", 0, 0},
    {"q_dispatch_vertex", kD, kV},
    {"q_dispatch_object", kD, kO},
    {"q_dispatch_mesh", kD, kM},
    {"q_dispatch_object+mesh", kD, kO | kM},
    {"q_dispatch_vertex+object+mesh", kD, kV | kO | kM},
};
const Barrier kToCompute[] = {
    {"none", 0, 0},
    {"q_object_dispatch", kO, kD},
    {"q_mesh_dispatch", kM, kD},
    {"q_object+mesh_dispatch", kO | kM, kD},
    {"q_vertex_dispatch", kV, kD},
    {"q_fragment_dispatch", kF, kD},
};
const char* recommended(const CaseInfo& ci) {
    return ci.c == Case::ObjectWrite ? "q_object+mesh_dispatch" : "q_dispatch_vertex+object+mesh";
}

bool validationActive() { return std::getenv("MTL_SHADER_VALIDATION") || std::getenv("MTL_DEBUG_LAYER"); }

struct Rig {
    Context& ctx;
    MTL::Library* lib = nullptr;
    MTL::RenderPipelineState* countPso = nullptr;
    MTL::RenderPipelineState* writePso = nullptr;
    MTL::Texture* target = nullptr;
    std::vector<MTL::Buffer*> scratch, done, counter, params, out;
    u32 iters = 1;
    explicit Rig(Context& c) : ctx(c) {}
};

MTL::RenderPipelineState* meshPipeline(Context& ctx, MTL::Library* lib, const char* object) {
    MTL4::MeshRenderPipelineDescriptor* d = MTL4::MeshRenderPipelineDescriptor::alloc()->init();
    d->setObjectFunctionDescriptor(ctx.function(lib, object));
    d->setMeshFunctionDescriptor(ctx.function(lib, "s4_mesh_empty"));
    d->setFragmentFunctionDescriptor(ctx.function(lib, "s4_fs"));
    d->setMaxTotalThreadsPerObjectThreadgroup(object == std::string("s4_object_write") ? kTGSize : 32);
    d->setMaxTotalThreadsPerMeshThreadgroup(32);
    d->setPayloadMemoryLength(16);
    d->setMaxTotalThreadgroupsPerMeshGrid(1);
    d->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatR8Unorm);
    NS::Error* err = nullptr;
    MTL::RenderPipelineState* pso = ctx.compiler()->newRenderPipelineState(d, nullptr, &err);
    d->release();
    if (!pso) throw BenchError(std::string("mesh pipeline ") + object + ": " +
                               (err ? err->localizedDescription()->utf8String() : "?"));
    ctx.keep(pso);
    return pso;
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

struct Outcome {
    u32 wrong = 0, minCount = ~0u, maxCount = 0;
    double ms = 0;
    std::string error;
};

Outcome runVariant(Rig& r, const CaseInfo& ci, const Barrier& b, u32 iters) {
    Context& ctx = r.ctx;
    Outcome o;
    for (u32 i = 0; i < kReps; ++i) {
        std::memset(r.done[i]->contents(), 0, 16);
        std::memset(r.counter[i]->contents(), 0, 16);
        std::memset(r.out[i]->contents(), 0, kObjTG * 4);
        // m2 reads data: salt != 0; m1 counts launches: salt 0 (the args are stale zeros until written).
        Params p{iters, ci.c == Case::ObjectWrite ? kObjTG : kTG, ci.c == Case::ObjectRead ? i * 7u + 1u : 0u, kObjTG};
        if (ci.c == Case::MeshArgs) p.salt = 0;
        std::memcpy(r.params[i]->contents(), &p, sizeof(p));
    }
    MTL4::ArgumentTable* t = ctx.table();
    MTL4::CommandBuffer* cmd = ctx.beginCommands();
    const MTL::Size objTG = MTL::Size::Make(32, 1, 1), meshTG = MTL::Size::Make(32, 1, 1);
    for (u32 i = 0; i < kReps; ++i) {
        t->setAddress(r.scratch[i]->gpuAddress(), 0);
        t->setAddress(r.done[i]->gpuAddress(), 1);
        t->setAddress(r.out[i]->gpuAddress(), 2);
        t->setAddress(r.params[i]->gpuAddress(), 3);
        t->setAddress(r.counter[i]->gpuAddress(), 5);
        if (ci.c == Case::ObjectWrite) {
            // producer: slow object shader; consumer: compute
            MTL4::RenderCommandEncoder* re = beginRender(r, cmd);
            re->setRenderPipelineState(r.writePso);
            re->setArgumentTable(t, MTL::RenderStageObject | MTL::RenderStageMesh | MTL::RenderStageFragment);
            re->drawMeshThreadgroups(MTL::Size::Make(kObjTG, 1, 1), MTL::Size::Make(kTGSize, 1, 1), meshTG);
            re->endEncoding();
            MTL4::ComputeCommandEncoder* ce = cmd->computeCommandEncoder();
            if (b.after) ce->barrierAfterQueueStages(b.after, b.before, MTL4::VisibilityOptionDevice);
            ce->setComputePipelineState(ctx.compute(r.lib, "s4_count_magic"));
            ce->setArgumentTable(t);
            ce->dispatchThreads(MTL::Size::Make(kObjTG, 1, 1), MTL::Size::Make(64, 1, 1));
            ce->endEncoding();
            continue;
        }
        MTL4::ComputeCommandEncoder* ce = cmd->computeCommandEncoder();
        ce->setComputePipelineState(ctx.compute(r.lib, ci.c == Case::MeshArgs ? "s4_prod_mesh_args" : "s4_prod_data"));
        ce->setArgumentTable(t);
        ce->dispatchThreadgroups(MTL::Size::Make(kTG, 1, 1), MTL::Size::Make(kTGSize, 1, 1));
        ce->endEncoding();
        MTL4::RenderCommandEncoder* re = beginRender(r, cmd);
        if (b.after) re->barrierAfterQueueStages(b.after, b.before, MTL4::VisibilityOptionDevice);
        re->setRenderPipelineState(r.countPso);
        re->setArgumentTable(t, MTL::RenderStageObject | MTL::RenderStageMesh | MTL::RenderStageFragment);
        if (ci.c == Case::MeshArgs) re->drawMeshThreadgroups(r.out[i]->gpuAddress(), objTG, meshTG);
        else re->drawMeshThreadgroups(MTL::Size::Make(kObjTG, 1, 1), objTG, meshTG);
        re->endEncoding();
    }
    try {
        o.ms = ctx.submit();
    } catch (const BenchError& e) {
        o.error = e.what();
        return o;
    }
    for (u32 i = 0; i < kReps; ++i) {
        const u32 got = *static_cast<const u32*>(r.counter[i]->contents());
        o.minCount = std::min(o.minCount, got);
        o.maxCount = std::max(o.maxCount, got);
        if (got != ci.expect) ++o.wrong;
    }
    return o;
}

double producerSpan(Rig& r, u32 iters) {
    Context& ctx = r.ctx;
    Params p{iters, kTG, 1, kObjTG};
    std::memcpy(r.params[0]->contents(), &p, sizeof(p));
    std::memset(r.done[0]->contents(), 0, 16);
    MTL4::CommandBuffer* cmd = ctx.beginCommands();
    MTL4::ComputeCommandEncoder* ce = cmd->computeCommandEncoder();
    MTL4::ArgumentTable* t = ctx.table();
    t->setAddress(r.scratch[0]->gpuAddress(), 0);
    t->setAddress(r.done[0]->gpuAddress(), 1);
    t->setAddress(r.out[0]->gpuAddress(), 2);
    t->setAddress(r.params[0]->gpuAddress(), 3);
    ce->setComputePipelineState(ctx.compute(r.lib, "s4_prod_data"));
    ce->setArgumentTable(t);
    ce->dispatchThreadgroups(MTL::Size::Make(kTG, 1, 1), MTL::Size::Make(kTGSize, 1, 1));
    ce->endEncoding();
    return ctx.submit();
}

void benchBarriers(Context& ctx, Report& rep) {
    Rig r(ctx);
    r.lib      = f6::f6Library(ctx, "s4_barriers.metal");
    r.countPso = meshPipeline(ctx, r.lib, "s4_object_count");
    r.writePso = meshPipeline(ctx, r.lib, "s4_object_write");
    MTL::TextureDescriptor* td = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatR8Unorm, 64, 64, false);
    td->setUsage(MTL::TextureUsageRenderTarget);
    td->setStorageMode(MTL::StorageModePrivate);
    r.target = ctx.texture(td);
    for (u32 i = 0; i < kReps; ++i) {
        r.scratch.push_back(ctx.buffer(size_t(kObjTG) * kTGSize * 4));
        r.done.push_back(ctx.buffer(16));
        r.counter.push_back(ctx.buffer(16));
        r.params.push_back(ctx.buffer(16));
        r.out.push_back(ctx.buffer(kObjTG * 4));
    }
    ctx.commitResidency();
    ctx.keepWarm(50);

    u32 iters = 1u << 17;
    double ms = producerSpan(r, iters);
    ms = producerSpan(r, iters);
    iters = static_cast<u32>(std::clamp(double(iters) * kTargetProducerMs / std::max(ms, 0.05), 1000.0, 3.0e8));
    ms = producerSpan(r, iters);
    r.iters = iters;
    rep.value("spin.iters", "count", iters, {}, false);
    rep.value("spin.producer_ms", "ms", ms, {}, false);
    ctx.log("[F6-S4] spin %u iterations, producer %.2f ms", iters, ms);
    // The object-shader producer has 512 groups x 256 threads (8x the compute
    // producer's threads): spin 1/8 as long so it also takes a few ms.
    const u32 objIters = std::max(iters / 8u, 1000u);

    bool recOk = true, allRaced = true;
    std::string errors, lines, inconclusive;
    for (const CaseInfo& ci : kCases) {
        const bool toCompute = ci.c == Case::ObjectWrite;
        const Barrier* list = toCompute ? kToCompute : kToRender;
        const size_t n = toCompute ? std::size(kToCompute) : std::size(kToRender);
        std::string line = std::string(ci.name) + ": ";
        for (size_t k = 0; k < n; ++k) {
            const Outcome o = runVariant(r, ci, list[k], toCompute ? objIters : r.iters);
            rep.value(std::string("wrong.") + ci.name + "." + list[k].name, "count", o.wrong, {{"reps", double(kReps)}}, false);
            rep.value(std::string("span_ms.") + ci.name + "." + list[k].name, "ms", o.ms, {}, false);
            line += std::string(k ? " | " : "") + list[k].name + " " + std::to_string(o.wrong) + "/" + std::to_string(kReps);
            if (o.wrong) line += " (" + std::to_string(o.minCount) + ".." + std::to_string(o.maxCount) + ")";
            if (!o.error.empty()) {
                errors += std::string(ci.name) + "/" + list[k].name + ": " + o.error + "; ";
                line += " ERROR";
            }
            if (k == 0 && o.wrong == 0) {
                allRaced = false;
                inconclusive += std::string(ci.name) + " ";
            }
            if (std::strcmp(list[k].name, recommended(ci)) == 0 && (o.wrong != 0 || !o.error.empty())) recOk = false;
            ctx.keepWarm(10);
        }
        rep.note(line);
        lines += line + "; ";
    }
    if (!inconclusive.empty()) rep.note("none variants that never raced (INCONCLUSIVE, not safe): " + inconclusive);
    if (!errors.empty()) rep.note("API/GPU errors: " + errors);
    const bool detOk = allRaced || validationActive();
    rep.negative(detOk && recOk && errors.empty(),
                 std::string(allRaced ? "every none variant raced" : "some none variants did not race") +
                     (validationActive() ? " [detection informational under validation]" : "") +
                     "; engine barriers (m1/m2 q dispatch->vertex+object+mesh, o1 q object+mesh->dispatch) " +
                     (recOk ? "0 wrong" : "WRONG"));
    if (!errors.empty()) rep.status(Status::Failed, errors);
}

} // namespace

SOC_BENCH("F6-S4", "meshlet.barriers",
          "Mesh path orderings: compute->indirect mesh args, compute->object reads, object writes->compute",
          benchBarriers);

} // namespace soc
