// F5-S3: draw emission -- GPU-encoded ICB vs indirect draws per bucket vs direct
// draws (docs/ROADMAP.md F5; a MEASUREMENT TOOL, not engine code).
//
// Scene: B buckets (1, 100, 1000, 10000), one small fan mesh (2-12 triangles)
// per bucket in ONE vertex buffer + ONE u32 index buffer, I = 100000 instances
// in contiguous slot ranges per bucket, cull class c = b % 3 (0 Back, 1 Back
// with front culling for mirrored instances, 2 None).  A compute "cull"
// stand-in (visible = hash(slot) % 4 != 0, buckets with b % 10 == 3 empty)
// produces the visible list in draw order (class, bucket, slot) with per-bucket
// visibleCount / firstVisible.  Render: 1024x1024 R32Uint (writes slot + 1) +
// Depth32Float reverse-Z; the vertex shader sets drawn[slot] = 1.
//
// Variants (all must give the same image and drawn[] flags):
//   a_reset    GPU-encoded ICB, one command per bucket, empty bucket -> cmd.reset()
//   a_zero     same, empty bucket -> draw with 0 instances
//   a_compact  only non-empty buckets get commands, per-class ranges written by
//              the GPU and executed with executeCommandsInBuffer(icb, rangeAddress)
//   a10        (Apple10) per-command cull mode / winding, ONE range, no CPU state
//   b_indirect kernel writes DrawIndexedPrimitivesIndirectArguments, CPU issues B indirect draws
//   c_direct   CPU knows the counts, B direct draws (empty skipped)
// Metrics (ms unless noted), per B ("s3.B<n>.<variant>."): render_ms (CommandTimer
// span - empty pass), encode_ms (encode / args kernel), reset_ms (GPU-timeline ICB
// reset), cpu_ms (wall time of the render-pass encoding), cpu_cmds (count);
// "s3.empty90.B10000.<variant>.*": the same with 90% empty buckets.
// Validation behaviour across command buffers: "s3.xcb.<v1..v4>.<variant>.drawn_ok" (1/0).
//
// Negative control (built in): a_zero with one non-empty bucket's instance count
// forced to 0 -> image and drawn[] must differ from the reference.

#include "f5_common.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace soc {
namespace {

constexpr u32 kI    = 100000;
constexpr u32 kRes  = 1024;
constexpr u32 kMaxB = 10000;
constexpr u32 kNone = 0xFFFFFFFFu;

struct Params {
    u32 buckets, emptyMode, drop, halfMode;
};
struct HBucket {
    u32 firstSlot, slotCount, indexCount, indexOffset;
    i32 vertexOffset;
    u32 cls, pos, pad;
};
static_assert(sizeof(HBucket) == 32);
struct HInst {
    float off[2];
    float depth;
    float scale;
};

bool validating() { return std::getenv("MTL_SHADER_VALIDATION") || std::getenv("MTL_DEBUG_LAYER"); }

u32 hash32(u32 x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}
bool slotVisible(u32 s) { return (hash32(s ^ 0x9E3779B9u) & 3u) != 0u; }
bool bucketEmpty(u32 b, u32 mode) { return mode == 0 ? (b % 10 == 3) : (b % 10 != 3); }

enum class V { AReset, AZero, ACompact, A10, BIndirect, CDirect };
const char* vname(V v) {
    switch (v) {
    case V::AReset: return "a_reset";
    case V::AZero: return "a_zero";
    case V::ACompact: return "a_compact";
    case V::A10: return "a10";
    case V::BIndirect: return "b_indirect";
    case V::CDirect: return "c_direct";
    }
    return "?";
}
bool isIcb(V v) { return v != V::BIndirect && v != V::CDirect; }

struct Target {
    MTL::Texture* color = nullptr;
    MTL::Texture* depth = nullptr;
    MTL::Buffer* drawn = nullptr;
    MTL::Buffer* readback = nullptr;
    MTL4::ArgumentTable* table = nullptr; // own render argument table (default: the harness table)
};

struct Verdict {
    u64 flagBad = 0;  // drawn[] != expected
    u64 pixBad = 0;   // pixels differing from the reference (or invalid slot)
    u64 pixNonZero = 0;
    u64 drawnCount = 0;
    u64 expectedCount = 0;
};

struct Rig {
    Context& ctx;
    MTL::Library* lib = nullptr;
    MTL::RenderPipelineState* pso = nullptr;
    MTL::DepthStencilState* depthState = nullptr;
    MTL::ComputePipelineState *kCount, *kScan, *kFill, *kReset, *kZero, *kCompact, *kA10, *kArgs;
    MTL::Buffer *inst, *verts, *idx, *bk, *order, *visCount, *firstVis, *cmdIdx, *vis, *ranges, *args, *containerA, *containerB;
    MTL::Buffer *pNormal, *pDrop, *pHalf;
    MTL::IndirectCommandBuffer *icbA = nullptr, *icbB = nullptr;
    Target t0;
    Target ring3[3]; // v4 targets, created with the rig: buffers created later were reported non-resident by shader validation
    // CPU copies
    std::vector<HBucket> hb;
    std::vector<u32> horder, hcount, hfirst, slotBucket;
    u32 B = 0, emptyMode = 0;
    u32 classStart[3] = {}, classLen[3] = {};
    std::vector<u32> ref, refHalf; // reference images (c_direct)
    bool bad = false;
    std::string badWhat, artifacts; // artifacts: shader-validation behaviour reported, not failures
    explicit Rig(Context& c) : ctx(c) {}

    void fail(const std::string& w) {
        bad = true;
        if (badWhat.size() < 600) badWhat += w + "; ";
    }
};

// ---- setup ------------------------------------------------------------------------

Target makeTarget(Rig& r) {
    Target t;
    MTL::TextureDescriptor* cd = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatR32Uint, kRes, kRes, false);
    cd->setUsage(MTL::TextureUsageRenderTarget);
    cd->setStorageMode(MTL::StorageModePrivate);
    t.color = r.ctx.texture(cd);
    MTL::TextureDescriptor* dd = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatDepth32Float, kRes, kRes, false);
    dd->setUsage(MTL::TextureUsageRenderTarget);
    dd->setStorageMode(MTL::StorageModePrivate);
    t.depth = r.ctx.texture(dd);
    t.drawn = r.ctx.buffer(size_t(kI) * 4);
    t.readback = r.ctx.buffer(size_t(kRes) * kRes * 4);
    return t;
}

MTL::IndirectCommandBuffer* makeIcb(Rig& r, bool perCommandCull) {
    MTL::IndirectCommandBufferDescriptor* d = MTL::IndirectCommandBufferDescriptor::alloc()->init();
    d->setCommandTypes(MTL::IndirectCommandTypeDrawIndexed);
    d->setInheritPipelineState(true);
    d->setInheritBuffers(true);
    d->setInheritDepthStencilState(true);
    d->setInheritCullMode(!perCommandCull);
    d->setInheritFrontFacingWinding(!perCommandCull);
    d->setMaxVertexBufferBindCount(4);
    d->setMaxFragmentBufferBindCount(0);
    MTL::IndirectCommandBuffer* icb = r.ctx.device()->newIndirectCommandBuffer(d, kMaxB, MTL::ResourceStorageModePrivate);
    if (!icb) icb = r.ctx.device()->newIndirectCommandBuffer(d, kMaxB, MTL::ResourceStorageModeShared);
    d->release();
    if (!icb) throw BenchError("newIndirectCommandBuffer failed");
    r.ctx.adopt(icb);
    return icb;
}

void setupRig(Rig& r) {
    Context& ctx = r.ctx;
    r.lib = f5::f5Library(ctx, "s3_draws.metal");
    MTL4::RenderPipelineDescriptor* rd = MTL4::RenderPipelineDescriptor::alloc()->init();
    rd->setVertexFunctionDescriptor(ctx.function(r.lib, "s3_vs"));
    rd->setFragmentFunctionDescriptor(ctx.function(r.lib, "s3_fs"));
    rd->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatR32Uint);
    rd->setSupportIndirectCommandBuffers(MTL4::IndirectCommandBufferSupportStateEnabled);
    r.pso = ctx.render(rd);
    rd->release();
    MTL::DepthStencilDescriptor* dsd = MTL::DepthStencilDescriptor::alloc()->init();
    dsd->setDepthCompareFunction(MTL::CompareFunctionGreater); // reverse-Z
    dsd->setDepthWriteEnabled(true);
    r.depthState = ctx.device()->newDepthStencilState(dsd);
    dsd->release();
    ctx.keep(r.depthState);
    r.kCount   = ctx.compute(r.lib, "s3_count");
    r.kScan    = ctx.compute(r.lib, "s3_scan");
    r.kFill    = ctx.compute(r.lib, "s3_fill");
    r.kReset   = ctx.compute(r.lib, "s3_encode_reset");
    r.kZero    = ctx.compute(r.lib, "s3_encode_zero");
    r.kCompact = ctx.compute(r.lib, "s3_encode_compact");
    r.kA10     = ctx.compute(r.lib, "s3_encode_a10");
    r.kArgs    = ctx.compute(r.lib, "s3_args");

    // Meshes: bucket b is a fan of nTri triangles (nTri + 1 rim vertices over 270 degrees + the centre), CCW.
    std::vector<float> vtx;
    std::vector<u32> ind;
    std::vector<u32> meshVertexOffset(kMaxB), meshIndexOffset(kMaxB), meshIndexCount(kMaxB);
    for (u32 b = 0; b < kMaxB; ++b) {
        const u32 nTri = 2 + (b * 7 + b / 3) % 11;
        meshVertexOffset[b] = static_cast<u32>(vtx.size() / 2);
        meshIndexOffset[b]  = static_cast<u32>(ind.size());
        meshIndexCount[b]   = nTri * 3;
        const float phase = float(b) * 0.37f;
        vtx.push_back(0.f);
        vtx.push_back(0.f);
        for (u32 k = 0; k <= nTri; ++k) {
            const float a = phase + 4.712389f * float(k) / float(nTri);
            vtx.push_back(std::cos(a));
            vtx.push_back(std::sin(a));
        }
        for (u32 k = 0; k < nTri; ++k) {
            ind.push_back(0);
            ind.push_back(1 + k);
            ind.push_back(2 + k);
        }
    }
    r.verts = ctx.buffer(vtx.size() * 4);
    std::memcpy(r.verts->contents(), vtx.data(), vtx.size() * 4);
    r.idx = ctx.buffer(ind.size() * 4);
    std::memcpy(r.idx->contents(), ind.data(), ind.size() * 4);

    r.hb.resize(kMaxB);
    for (u32 b = 0; b < kMaxB; ++b) {
        r.hb[b].indexCount   = meshIndexCount[b];
        r.hb[b].indexOffset  = meshIndexOffset[b];
        r.hb[b].vertexOffset = static_cast<i32>(meshVertexOffset[b]);
        r.hb[b].cls          = b % 3;
        r.hb[b].pad          = 0;
    }
    r.inst     = ctx.buffer(size_t(kI) * sizeof(HInst));
    r.bk       = ctx.buffer(size_t(kMaxB) * sizeof(HBucket));
    r.order    = ctx.buffer(size_t(kMaxB) * 4);
    r.visCount = ctx.buffer(size_t(kMaxB) * 4);
    r.firstVis = ctx.buffer(size_t(kMaxB) * 4);
    r.cmdIdx   = ctx.buffer(size_t(kMaxB) * 4);
    r.vis      = ctx.buffer(size_t(kI) * 4);
    r.ranges   = ctx.buffer(64);
    r.args     = ctx.buffer(size_t(kMaxB) * 20);
    r.pNormal  = ctx.buffer(16);
    r.pDrop    = ctx.buffer(16);
    r.pHalf    = ctx.buffer(16);
    r.containerA = ctx.buffer(16);
    r.containerB = ctx.buffer(16);
    r.icbA = makeIcb(r, false);
    const MTL::ResourceID ra = r.icbA->gpuResourceID();
    std::memcpy(r.containerA->contents(), &ra, sizeof(ra));
    if (ctx.apple10()) {
        r.icbB = makeIcb(r, true);
        const MTL::ResourceID rb = r.icbB->gpuResourceID();
        std::memcpy(r.containerB->contents(), &rb, sizeof(rb));
    }
    r.t0 = makeTarget(r);
    for (Target& t : r.ring3) t = makeTarget(r);
}

void setParams(MTL::Buffer* buf, u32 B, u32 mode, u32 drop, u32 half) {
    const Params p{B, mode, drop, half};
    std::memcpy(buf->contents(), &p, sizeof(p));
}

// ---- recording -----------------------------------------------------------------------

void setAddr(Context& ctx, std::initializer_list<const MTL::Buffer*> bufs) {
    u32 i = 0;
    for (const MTL::Buffer* b : bufs) ctx.table()->setAddress(b->gpuAddress(), i++);
}

constexpr MTL::Stages kProducers = MTL::StageDispatch | MTL::StageBlit;

// Kernels that produce the draw data of `v` (ICB reset + encode, or the indirect arguments).  `afterReset`
// runs between the GPU-timeline reset and the encode dispatch (used to take a lap).
void encodeKernels(Rig& r, MTL4::ComputeCommandEncoder* ce, V v, const MTL::Buffer* params, const std::function<void()>& afterReset = nullptr) {
    Context& ctx = r.ctx;
    if (v == V::CDirect) return;
    const MTL::Size tg = MTL::Size::Make(64, 1, 1);
    const MTL::Size grid = MTL::Size::Make(r.B, 1, 1);
    if (v == V::BIndirect) {
        setAddr(ctx, {r.bk, r.visCount, r.firstVis, r.args, params});
        ce->setComputePipelineState(r.kArgs);
        ce->setArgumentTable(ctx.table());
        ce->dispatchThreads(grid, tg);
        return;
    }
    MTL::IndirectCommandBuffer* icb = v == V::A10 ? r.icbB : r.icbA;
    ce->resetCommandsInBuffer(icb, NS::Range::Make(0, r.B)); // on the GPU timeline, never icb->reset() (may be in flight)
    ce->barrierAfterEncoderStages(kProducers, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
    if (afterReset) afterReset();
    setAddr(ctx, {v == V::A10 ? r.containerB : r.containerA, r.bk, r.visCount, r.firstVis, params, r.idx, r.cmdIdx});
    ce->setComputePipelineState(v == V::AReset ? r.kReset : v == V::AZero ? r.kZero : v == V::ACompact ? r.kCompact : r.kA10);
    ce->setArgumentTable(ctx.table());
    ce->dispatchThreads(grid, tg);
}

struct RenderStats {
    double cpuMs = 0;
    u32 cmds = 0;
};

// Records the render pass.  `barrier`: encode kernels ran earlier (this or another command buffer of the same
// queue) -> queue barrier producers -> Vertex.  `emptyPass`: a single trivial draw (fixed cost).
RenderStats encodeRender(Rig& r, MTL4::CommandBuffer* cmd, V v, Target& t, bool barrier, bool emptyPass = false, bool halfMode = false) {
    Context& ctx = r.ctx;
    RenderStats st;
    const double t0 = nowMs();
    MTL4::RenderPassDescriptor* pd = MTL4::RenderPassDescriptor::alloc()->init();
    auto* c = pd->colorAttachments()->object(0);
    c->setTexture(t.color);
    c->setLoadAction(MTL::LoadActionClear);
    c->setStoreAction(MTL::StoreActionStore);
    c->setClearColor(MTL::ClearColor::Make(0, 0, 0, 0));
    auto* d = pd->depthAttachment();
    d->setTexture(t.depth);
    d->setLoadAction(MTL::LoadActionClear);
    d->setStoreAction(MTL::StoreActionDontCare);
    d->setClearDepth(0.0); // reverse-Z
    MTL4::RenderCommandEncoder* re = cmd->renderCommandEncoder(pd);
    re->setLabel(NS::String::string("s3 render", NS::UTF8StringEncoding)); // the validation layer crashes decoding a report for an unlabelled encoder
    pd->release();
    re->setViewport(MTL::Viewport{0.0, 0.0, double(kRes), double(kRes), 0.0, 1.0});
    MTL4::ArgumentTable* tbl = t.table ? t.table : ctx.table();
    u32 slot = 0;
    for (const MTL::Buffer* b : {r.inst, r.vis, r.verts, t.drawn}) tbl->setAddress(b->gpuAddress(), slot++);
    re->setRenderPipelineState(r.pso);
    re->setDepthStencilState(r.depthState);
    re->setArgumentTable(tbl, MTL::RenderStageVertex | MTL::RenderStageFragment);
    if (barrier) re->barrierAfterQueueStages(kProducers, MTL::StageVertex, MTL4::VisibilityOptionDevice);

    const MTL::GPUAddress idxBase = r.idx->gpuAddress();
    const NS::UInteger idxLen = r.idx->length();
    // State tracked from Metal's defaults (clockwise, no culling): the validation layer rejects redundant state.
    MTL::Winding winding = MTL::WindingClockwise;
    MTL::CullMode cull = MTL::CullModeNone;
    const auto setState = [&](u32 cls) {
        const MTL::CullMode want = cls == 0 ? MTL::CullModeBack : cls == 1 ? MTL::CullModeFront : MTL::CullModeNone;
        if (winding != MTL::WindingCounterClockwise) { re->setFrontFacingWinding(winding = MTL::WindingCounterClockwise); ++st.cmds; }
        if (cull != want) { re->setCullMode(cull = want); ++st.cmds; }
    };
    if (emptyPass) {
        re->drawIndexedPrimitives(MTL::PrimitiveTypeTriangle, r.hb[0].indexCount, MTL::IndexTypeUInt32, idxBase, idxLen, 1, 0, 0);
    } else if (v == V::A10) {
        re->executeCommandsInBuffer(r.icbB, NS::Range::Make(0, r.B));
        ++st.cmds;
    } else if (v == V::AReset || v == V::AZero) {
        for (u32 cl = 0; cl < 3; ++cl) {
            if (r.classLen[cl] == 0) continue;
            setState(cl);
            re->executeCommandsInBuffer(r.icbA, NS::Range::Make(r.classStart[cl], r.classLen[cl]));
            ++st.cmds;
        }
    } else if (v == V::ACompact) {
        for (u32 cl = 0; cl < 3; ++cl) { // CPU does not know which classes are empty: always 3
            setState(cl);
            re->executeCommandsInBuffer(r.icbA, r.ranges->gpuAddress() + (3 + cl) * 8);
            ++st.cmds;
        }
    } else if (v == V::BIndirect) {
        u32 cur = kNone;
        for (u32 pos = 0; pos < r.B; ++pos) {
            const u32 cl = r.hb[r.horder[pos]].cls;
            if (cl != cur) { setState(cl); cur = cl; }
            re->drawIndexedPrimitives(MTL::PrimitiveTypeTriangle, MTL::IndexTypeUInt32, idxBase, idxLen, r.args->gpuAddress() + u64(pos) * 20);
            ++st.cmds;
        }
    } else { // direct
        u32 cur = kNone;
        for (u32 pos = 0; pos < r.B; ++pos) {
            const u32 b = r.horder[pos];
            const u32 n = r.hcount[b];
            if (n == 0 || (halfMode && (b & 1u))) continue;
            const HBucket& k = r.hb[b];
            if (k.cls != cur) { setState(k.cls); cur = k.cls; }
            re->drawIndexedPrimitives(MTL::PrimitiveTypeTriangle, k.indexCount, MTL::IndexTypeUInt32, idxBase + u64(k.indexOffset) * 4,
                                      NS::UInteger(k.indexCount) * 4, n, k.vertexOffset, r.hfirst[b]);
            ++st.cmds;
        }
    }
    re->endEncoding();
    st.cpuMs = nowMs() - t0;
    return st;
}

void encodeReadback(Rig& r, MTL4::CommandBuffer* cmd, Target& t) {
    MTL4::ComputeCommandEncoder* ce = cmd->computeCommandEncoder();
    ce->setLabel(NS::String::string("s3 compute", NS::UTF8StringEncoding));
    ce->barrierAfterQueueStages(MTL::StageFragment | MTL::StageVertex | MTL::StageDispatch, MTL::StageBlit, MTL4::VisibilityOptionDevice);
    ce->copyFromTexture(t.color, 0, 0, MTL::Origin::Make(0, 0, 0), MTL::Size::Make(kRes, kRes, 1), t.readback, 0, kRes * 4, kRes * kRes * 4);
    ce->endEncoding();
}

void encodeKernelPass(Rig& r, MTL4::CommandBuffer* cmd, V v, const MTL::Buffer* params) {
    if (v == V::CDirect) return; // an encoder without work is rejected by the validation layer
    MTL4::ComputeCommandEncoder* ce = cmd->computeCommandEncoder();
    ce->setLabel(NS::String::string("s3 compute", NS::UTF8StringEncoding));
    encodeKernels(r, ce, v, params);
    ce->endEncoding();
}

// ---- scene configuration + cull stand-in -------------------------------------------------

void configure(Rig& r, u32 B, u32 mode) {
    Context& ctx = r.ctx;
    r.B = B;
    r.emptyMode = mode;
    r.horder.clear();
    for (u32 c = 0; c < 3; ++c) {
        r.classStart[c] = static_cast<u32>(r.horder.size());
        for (u32 b = c; b < B; b += 3) r.horder.push_back(b);
        r.classLen[c] = static_cast<u32>(r.horder.size()) - r.classStart[c];
    }
    r.slotBucket.assign(kI, 0);
    for (u32 b = 0; b < B; ++b) {
        HBucket& k = r.hb[b];
        k.firstSlot = static_cast<u32>(u64(b) * kI / B);
        k.slotCount = static_cast<u32>(u64(b + 1) * kI / B) - k.firstSlot;
        for (u32 s = k.firstSlot; s < k.firstSlot + k.slotCount; ++s) r.slotBucket[s] = b;
    }
    for (u32 pos = 0; pos < B; ++pos) r.hb[r.horder[pos]].pos = pos;
    std::memcpy(r.bk->contents(), r.hb.data(), size_t(B) * sizeof(HBucket));
    std::memcpy(r.order->contents(), r.horder.data(), size_t(B) * 4);

    // Instances: random 2D offset, small radius (a few pixels), unique exactly representable depth, mirror by class.
    u64 rng = 0x1234567ull;
    auto* in = static_cast<HInst*>(r.inst->contents());
    for (u32 s = 0; s < kI; ++s) {
        const double u0 = double(xorshift64(rng) >> 11) / 9007199254740992.0, u1 = double(xorshift64(rng) >> 11) / 9007199254740992.0;
        const double u2 = double(xorshift64(rng) >> 11) / 9007199254740992.0;
        in[s].off[0] = float(u0 * 1.9 - 0.95);
        in[s].off[1] = float(u1 * 1.9 - 0.95);
        in[s].depth  = float((s * 2654435761u % (1u << 24)) + 1u) / float(1u << 24) * 0.98f;
        const u32 cls = r.hb[r.slotBucket[s]].cls;
        const bool mirrored = cls == 1 || (cls == 2 && (hash32(s) & 1u));
        const float sc = float(0.004 + u2 * 0.004);
        in[s].scale = mirrored ? -sc : sc;
    }

    setParams(r.pNormal, B, mode, kNone, 0);
    setParams(r.pHalf, B, mode, kNone, 1);

    // CPU expectation of the cull stand-in.
    r.hcount.assign(B, 0);
    r.hfirst.assign(B, 0);
    u32 run = 0;
    for (u32 pos = 0; pos < B; ++pos) {
        const u32 b = r.horder[pos];
        u32 n = 0;
        if (!bucketEmpty(b, mode))
            for (u32 s = r.hb[b].firstSlot; s < r.hb[b].firstSlot + r.hb[b].slotCount; ++s) n += slotVisible(s);
        r.hcount[b] = n;
        r.hfirst[b] = run;
        run += n;
    }

    // GPU: count -> scan -> fill, then verify against the CPU.
    MTL4::CommandBuffer* cmd = ctx.beginCommands();
    const MTL::Size tg = MTL::Size::Make(64, 1, 1), grid = MTL::Size::Make(B, 1, 1);
    const auto stage = [&](MTL4::ComputeCommandEncoder* ce) {
        ce->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
    };
    MTL4::ComputeCommandEncoder* ce = cmd->computeCommandEncoder();
    ce->setLabel(NS::String::string("s3 compute", NS::UTF8StringEncoding));
    setAddr(ctx, {r.bk, r.visCount, r.pNormal});
    ce->setComputePipelineState(r.kCount);
    ce->setArgumentTable(ctx.table());
    ce->dispatchThreads(grid, tg);
    stage(ce);
    setAddr(ctx, {r.bk, r.order, r.visCount, r.firstVis, r.cmdIdx, r.ranges, r.pNormal});
    ce->setComputePipelineState(r.kScan);
    ce->setArgumentTable(ctx.table());
    ce->dispatchThreads(MTL::Size::Make(1, 1, 1), MTL::Size::Make(1, 1, 1));
    stage(ce);
    setAddr(ctx, {r.bk, r.firstVis, r.visCount, r.vis, r.pNormal});
    ce->setComputePipelineState(r.kFill);
    ce->setArgumentTable(ctx.table());
    ce->dispatchThreads(grid, tg);
    ce->endEncoding();
    ctx.submit();

    const auto* gc = static_cast<const u32*>(r.visCount->contents());
    const auto* gf = static_cast<const u32*>(r.firstVis->contents());
    const auto* gv = static_cast<const u32*>(r.vis->contents());
    u64 badCull = 0;
    for (u32 b = 0; b < B; ++b)
        if (gc[b] != r.hcount[b] || gf[b] != r.hfirst[b]) ++badCull;
    u32 o = 0;
    for (u32 pos = 0; pos < B; ++pos) {
        const u32 b = r.horder[pos];
        for (u32 s = r.hb[b].firstSlot; s < r.hb[b].firstSlot + r.hb[b].slotCount && r.hcount[b]; ++s)
            if (slotVisible(s) && gv[o++] != s) ++badCull;
    }
    // Compacted ranges.
    const auto* rg = static_cast<const u32*>(r.ranges->contents());
    u32 ne = 0;
    for (u32 c = 0; c < 3; ++c) {
        u32 cnt = 0;
        for (u32 pos = r.classStart[c]; pos < r.classStart[c] + r.classLen[c]; ++pos) cnt += r.hcount[r.horder[pos]] > 0;
        if (rg[2 * c + 1] != r.classLen[c] || rg[2 * (3 + c) + 1] != cnt) ++badCull;
        if (r.classLen[c] && rg[2 * c] != r.classStart[c]) ++badCull;
        if (cnt && rg[2 * (3 + c)] != ne) ++badCull; // location of an empty range is irrelevant
        ne += cnt;
    }
    if (badCull) r.fail("cull stand-in mismatch B=" + std::to_string(B) + " (" + std::to_string(badCull) + ")");
    r.ref.clear();
    r.refHalf.clear();
}

bool expectedDrawn(const Rig& r, u32 slot, bool half) {
    const u32 b = r.slotBucket[slot];
    return r.hcount[b] > 0 && slotVisible(slot) && !(half && (b & 1u));
}

// ---- running + verifying ---------------------------------------------------------------

Verdict verify(Rig& r, Target& t, const std::vector<u32>* ref, bool half, std::vector<u32>* imageOut) {
    Verdict vd;
    const auto* fl = static_cast<const u32*>(t.drawn->contents());
    for (u32 s = 0; s < kI; ++s) {
        const bool e = expectedDrawn(r, s, half);
        vd.expectedCount += e;
        vd.drawnCount += fl[s] == 1;
        if ((fl[s] == 1) != e || fl[s] > 1) ++vd.flagBad;
    }
    const auto* px = static_cast<const u32*>(t.readback->contents());
    for (u32 i = 0; i < kRes * kRes; ++i) {
        if (px[i]) {
            ++vd.pixNonZero;
            if (px[i] - 1 >= kI || !expectedDrawn(r, px[i] - 1, half)) ++vd.pixBad;
        }
        if (ref && !ref->empty() && (*ref)[i] != px[i]) ++vd.pixBad;
    }
    if (imageOut) imageOut->assign(px, px + size_t(kRes) * kRes);
    return vd;
}

// v1: encode kernel + execute in the same command buffer (the engine's normal path).
Verdict runSame(Rig& r, V v, const MTL::Buffer* params, const std::vector<u32>* ref, std::vector<u32>* img, bool half = false) {
    Context& ctx = r.ctx;
    std::memset(r.t0.drawn->contents(), 0, size_t(kI) * 4);
    MTL4::CommandBuffer* cmd = ctx.beginCommands();
    encodeKernelPass(r, cmd, v, params);
    encodeRender(r, cmd, v, r.t0, true, false, half);
    encodeReadback(r, cmd, r.t0);
    ctx.submit();
    return verify(r, r.t0, ref, half, img);
}

std::string describe(const Verdict& v) {
    return "flagBad " + std::to_string(v.flagBad) + " pixBad " + std::to_string(v.pixBad) + " drawn " + std::to_string(v.drawnCount) + "/" +
           std::to_string(v.expectedCount) + " px " + std::to_string(v.pixNonZero);
}

// Reference (c_direct) + sanity: every class contributes pixels.
void makeReferences(Rig& r) {
    Verdict vd = runSame(r, V::CDirect, r.pNormal, nullptr, &r.ref);
    if (vd.flagBad || vd.pixBad || vd.pixNonZero == 0) r.fail("reference c_direct invalid: " + describe(vd));
    u64 perClass[3] = {0, 0, 0};
    for (u32 p : r.ref)
        if (p) ++perClass[r.hb[r.slotBucket[p - 1]].cls];
    for (u32 c = 0; c < 3; ++c)
        if (r.classLen[c] && perClass[c] == 0) r.fail("class " + std::to_string(c) + " has no pixels (cull state wrong?)");
    runSame(r, V::CDirect, r.pHalf, nullptr, &r.refHalf, true);
}

// ---- timing ------------------------------------------------------------------------------

Stats scaledSub(Stats s, double sub) {
    s.median -= sub; s.min -= sub; s.max -= sub; s.p10 -= sub; s.p90 -= sub; s.mean -= sub;
    return s;
}

double emptyPassMs(Rig& r) {
    const Stats s = r.ctx.measure([&] {
        CommandTimer t(r.ctx);
        MTL4::CommandBuffer* cmd = t.begin();
        encodeRender(r, cmd, V::CDirect, r.t0, false, true);
        return t.finish();
    });
    return s.median;
}

void timeVariant(Rig& r, Report& rep, V v, const std::string& prefix, double emptyMs) {
    Context& ctx = r.ctx;
    const std::map<std::string, double> params = {{"B", double(r.B)}, {"empty_mode", double(r.emptyMode)}};
    // Produce the draw data once (untimed), then time the render pass alone.
    if (v != V::CDirect) {
        MTL4::CommandBuffer* cmd = ctx.beginCommands();
        encodeKernelPass(r, cmd, v, r.pNormal);
        ctx.submit();
    }
    std::vector<double> cpu;
    RenderStats last;
    ctx.keepWarm(20);
    const Stats rs = ctx.measure([&] {
        CommandTimer t(ctx);
        MTL4::CommandBuffer* cmd = t.begin();
        last = encodeRender(r, cmd, v, r.t0, false);
        cpu.push_back(last.cpuMs);
        return t.finish() - emptyMs;
    });
    rep.metric(prefix + "render_ms", "ms", rs, params, false);
    rep.metric(prefix + "cpu_ms", "ms", phosphor::soc::computeStats(cpu), params, false);
    rep.value(prefix + "cpu_cmds", "count", double(last.cmds), params, false);
    if (v == V::CDirect) return;
    std::vector<double> resetMs;
    const Stats es = ctx.measure([&] {
        ComputeTimer t(ctx);
        MTL4::ComputeCommandEncoder* ce = t.begin();
        encodeKernels(r, ce, v, r.pNormal, [&] { t.lap(); });
        t.lap();
        const std::vector<double> laps = t.finish();
        if (isIcb(v)) {
            resetMs.push_back(laps[0]);
            return laps[1];
        }
        return laps[0];
    });
    rep.metric(prefix + "encode_ms", "ms", es, params, false);
    if (isIcb(v)) rep.metric(prefix + "reset_ms", "ms", phosphor::soc::computeStats(resetMs), params, false);
}

// ---- scene driver ----------------------------------------------------------------------------

void runScene(Rig& r, Report& rep, u32 B, u32 mode, bool control) {
    Context& ctx = r.ctx;
    configure(r, B, mode);
    makeReferences(r);
    const std::string tag = mode == 0 ? "s3.B" + std::to_string(B) + "." : "s3.empty90.B" + std::to_string(B) + ".";
    std::vector<V> vs = {V::AReset, V::AZero, V::ACompact, V::BIndirect, V::CDirect};
    if (ctx.apple10()) vs.insert(vs.begin() + 3, V::A10);
    // Under validation timings are meaningless and the timed path executes an ICB encoded in an EARLIER command
    // buffer, which shader validation reports (see the notes): only the same-command-buffer checks run.
    const bool timed = !validating();
    const double emptyMs = timed ? emptyPassMs(r) : 0.0;
    for (V v : vs) {
        if (v != V::CDirect) {
            const Verdict vd = runSame(r, v, r.pNormal, &r.ref, nullptr);
            if (vd.flagBad || vd.pixBad || vd.pixNonZero == 0) {
                const std::string what = std::string(vname(v)) + " B=" + std::to_string(B) + " mode=" + std::to_string(mode) + ": " + describe(vd);
                // Measured: under shader validation an indirect range (executeCommandsInBuffer(icb, rangeAddress)) of a
                // GPU-encoded ICB executed in the SAME command buffer draws (almost) nothing; the other variants draw exactly.
                if (validating() && v == V::ACompact) {
                    if (r.artifacts.size() < 300) r.artifacts += what + " | ";
                } else r.fail(what);
            }
        }
        if (timed) timeVariant(r, rep, v, tag + vname(v) + ".", emptyMs);
        ctx.log("F5-S3 B=%u mode=%u %-10s checked + timed", B, mode, vname(v));
    }
    if (control) {
        // Corrupt a_zero: drop the non-empty bucket with the most pixels in the reference.
        std::vector<u32> pix(B, 0);
        for (u32 p : r.ref)
            if (p) ++pix[r.slotBucket[p - 1]];
        u32 best = 0;
        for (u32 b = 0; b < B; ++b)
            if (pix[b] > pix[best]) best = b;
        setParams(r.pDrop, B, mode, std::getenv("F5S3_NO_CORRUPTION") ? kNone : best, 0);
        const Verdict vd = runSame(r, V::AZero, r.pDrop, &r.ref, nullptr);
        const bool detected = vd.flagBad > 0 && vd.pixBad > 0;
        r.ctx.log("F5-S3 negative control B=%u: dropped bucket %u (%u px): %s -> %s", B, best, pix[best], describe(vd).c_str(), detected ? "detected" : "NOT detected");
        rep.value("s3.control.B" + std::to_string(B) + ".detected", "count", detected ? 1 : 0, {{"B", double(B)}}, true);
        if (!detected) r.fail("negative control not detected at B=" + std::to_string(B));
    }
}

// ---- command-buffer crossing (v2, v3, v4) -------------------------------------------------------

struct Cross {
    bool v1 = false, v2 = false, v3 = false, v4 = false;
    std::string detail;
};

bool drawnExactly(Rig& r, Target& t, const std::vector<u32>& ref, bool half, std::string& what) {
    const Verdict vd = verify(r, t, &ref, half, nullptr);
    what = describe(vd);
    return vd.flagBad == 0 && vd.pixBad == 0 && vd.pixNonZero > 0;
}

MTL4::ArgumentTable* newTable(Context& ctx) {
    MTL4::ArgumentTableDescriptor* ad = MTL4::ArgumentTableDescriptor::alloc()->init();
    ad->setMaxBufferBindCount(8);
    NS::Error* err = nullptr;
    MTL4::ArgumentTable* t = ctx.device()->newArgumentTable(ad, &err);
    ad->release();
    if (!t) throw BenchError("newArgumentTable failed");
    ctx.keep(t);
    return t;
}

void crossingTests(Rig& r, Report& rep) {
    Context& ctx = r.ctx;
    configure(r, 1000, 0);
    makeReferences(r);
    const bool val = validating();
    std::string notes;
    std::vector<V> xv = {V::AZero, V::ACompact};
    if (std::getenv("F5S3_SWAP")) std::swap(xv[0], xv[1]);
    for (V v : xv) {
        const std::string pre = std::string("s3.xcb.") ;
        std::string what;
        // v1
        ctx.log("F5-S3 crossing v1 %s", vname(v));
        {
            const Verdict vd = runSame(r, v, r.pNormal, &r.ref, nullptr);
            const bool ok = vd.flagBad == 0 && vd.pixBad == 0 && vd.pixNonZero > 0;
            rep.value(pre + "v1." + vname(v) + ".drawn_ok", "count", ok, {}, true);
            if (!ok) {
                if (val && v == V::ACompact) notes += "v1 a_compact draws nothing under shader validation (" + describe(vd) + "); ";
                else r.fail(std::string("v1 ") + vname(v) + " does not draw exactly: " + describe(vd));
            }
        }
        // v2: encode in A, execute in B, ONE commit call
        ctx.log("F5-S3 crossing v2 %s", vname(v));
        {
            std::memset(r.t0.drawn->contents(), 0, size_t(kI) * 4);
            f5::FrameRing ring(ctx, 1, 1);
            MTL4::CommandBuffer* a = ring.begin();
            encodeKernelPass(r, a, v, r.pNormal);
            MTL4::CommandBuffer* b = ring.extra(0);
            encodeRender(r, b, v, r.t0, true);
            encodeReadback(r, b, r.t0);
            ring.commit();
            ring.drain();
            const bool ok = drawnExactly(r, r.t0, r.ref, false, what);
            rep.value(pre + "v2." + vname(v) + ".drawn_ok", "count", ok, {}, true);
            if (!ok) {
                if (val) notes += std::string("v2 ") + vname(v) + " draws nothing/differs under validation (" + what + "); ";
                else r.fail(std::string("v2 ") + vname(v) + " does not draw exactly: " + what);
            }
        }
        // v3: encode on a second queue, signal, main queue waits and executes
        ctx.log("F5-S3 crossing v3 %s", vname(v));
        {
            std::memset(r.t0.drawn->contents(), 0, size_t(kI) * 4);
            MTL4::CommandQueue* q2 = ctx.newQueue();
            MTL4::CommandBuffer* a = ctx.newCommandBuffer();
            MTL4::CommandAllocator* al = ctx.newAllocator();
            MTL::SharedEvent* ev = ctx.device()->newSharedEvent();
            ctx.keep(ev);
            a->beginCommandBuffer(al);
            encodeKernelPass(r, a, v, r.pNormal);
            a->endCommandBuffer();
            const MTL4::CommandBuffer* bufs[] = {a};
            q2->commit(bufs, 1);
            q2->signalEvent(ev, 1);
            ctx.queue()->wait(ev, 1);
            MTL4::CommandBuffer* cmd = ctx.beginCommands();
            encodeRender(r, cmd, v, r.t0, false);
            encodeReadback(r, cmd, r.t0);
            ctx.submit();
            const bool ok = drawnExactly(r, r.t0, r.ref, false, what);
            rep.value(pre + "v3." + vname(v) + ".drawn_ok", "count", ok, {}, true);
            if (!ok) {
                if (val) notes += std::string("v3 ") + vname(v) + " draws nothing/differs under validation (" + what + "); ";
                else r.fail(std::string("v3 ") + vname(v) + " does not draw exactly: " + what);
            }
        }
        // v4: the SAME ICB re-encoded on consecutive frames, 3 in flight, alternating full / half encodes
        ctx.log("F5-S3 crossing v4 %s", vname(v));
        {
            std::memset(r.t0.drawn->contents(), 0, size_t(kI) * 4); // detects flags landing in a stale binding
            Target ts[3] = {r.ring3[0], r.ring3[1], r.ring3[2]};
            for (Target& t : ts) std::memset(t.drawn->contents(), 0, size_t(kI) * 4);
            // Per-frame argument tables: a buffer bound through the (shared) table that an ICB inherits must not be
            // re-pointed by the next frame's recording (measured under validation, see the report).
            if (std::getenv("F5S3_SETTLE")) { ctx.commitResidency(); ctx.keepWarm(10); }
            if (!std::getenv("F5S3_SHARED_TABLE"))
                for (Target& t : ts) t.table = newTable(ctx);
            const u32 nf = std::getenv("F5S3_INFLIGHT") ? u32(std::atoi(std::getenv("F5S3_INFLIGHT"))) : 3u; // exploration knob (default 3 frames in flight)
            f5::FrameRing ring(ctx, nf);
            constexpr u32 kFrames = 12;
            bool ok = true;
            std::string firstBad;
            const auto check = [&](u32 frame) {
                std::string w;
                const bool half = (frame & 1u) != 0;
                const bool exact = drawnExactly(r, ts[frame % nf], half ? r.refHalf : r.ref, half, w);
                ctx.log("F5-S3 v4 %s frame %u (%s): %s %s", vname(v), frame, half ? "half" : "full", exact ? "exact" : "BAD", w.c_str());
                if (!exact) {
                    ok = false;
                    if (firstBad.empty()) firstBad = "frame " + std::to_string(frame) + ": " + w;
                }
            };
            for (u32 f = 0; f < kFrames; ++f) {
                MTL4::CommandBuffer* cmd = ring.begin(); // blocks until frame f - 3 completed
                if (f >= nf) {
                    check(f - nf);
                    std::memset(ts[(f - nf) % nf].drawn->contents(), 0, size_t(kI) * 4); // frame f - 3 is done: CPU clear (a GPU fillBuffer raced the vertex stage under validation)
                }
                Target& t = ts[f % nf];
                const bool half = (f & 1u) != 0;
                MTL4::ComputeCommandEncoder* ce = cmd->computeCommandEncoder();
    ce->setLabel(NS::String::string("s3 compute", NS::UTF8StringEncoding));
                // The previous frame may still be executing the ICB: wait for it before the GPU-timeline reset.
                ce->barrierAfterQueueStages(MTL::StageAll,
                                            MTL::StageDispatch | MTL::StageBlit, MTL4::VisibilityOptionDevice);
                encodeKernels(r, ce, v, half ? r.pHalf : r.pNormal);
                ce->endEncoding();
                encodeRender(r, cmd, v, t, true, false, false);
                encodeReadback(r, cmd, t);
                ring.commit();
            }
            ring.drain();
            {
                u64 stray = 0;
                const auto* fl0 = static_cast<const u32*>(r.t0.drawn->contents());
                for (u32 sIdx = 0; sIdx < kI; ++sIdx) stray += fl0[sIdx] != 0;
                u64 sums[3] = {0, 0, 0};
                for (u32 q = 0; q < 3; ++q) {
                    const auto* fq = static_cast<const u32*>(ts[q].drawn->contents());
                    for (u32 sIdx = 0; sIdx < kI; ++sIdx) sums[q] += fq[sIdx] != 0;
                }
                ctx.log("F5-S3 v4 %s: old t0 flags %llu, ts flags %llu %llu %llu", vname(v), (unsigned long long)stray,
                        (unsigned long long)sums[0], (unsigned long long)sums[1], (unsigned long long)sums[2]);
            }
            for (u32 f = kFrames - nf; f < kFrames; ++f) check(f);
            rep.value(pre + "v4." + vname(v) + ".drawn_ok", "count", ok, {}, true);
            if (!ok) {
                if (val) notes += std::string("v4 ") + vname(v) + " stale/missing under validation (" + firstBad + "); ";
                else r.fail(std::string("v4 ") + vname(v) + " stale or missing commands: " + firstBad);
            }
        }
    }
    rep.note(val ? "cross-command-buffer behaviour UNDER VALIDATION: " + (notes.empty() ? std::string("v1..v4 all draw exactly") : notes)
                 : "cross-command-buffer behaviour without validation: v1..v4 all draw exactly (a_zero and a_compact)");
}

void benchDraws(Context& ctx, Report& rep) {
    Rig r(ctx);
    setupRig(r);
    ctx.warmUp(5.0);
    const std::vector<u32> bs = ctx.quick() ? std::vector<u32>{1, 100, 1000, 10000} : std::vector<u32>{1, 100, 1000, 10000};
    for (u32 B : bs) runScene(r, rep, B, 0, true);
    runScene(r, rep, kMaxB, 1, false); // 90% empty buckets
    crossingTests(r, rep);
    if (!ctx.apple10()) rep.note("a10 (per-command cull mode / winding) skipped: needs Apple10 (--force-family apple9 or Apple9 device)");
    else rep.note("a10 ran: render_command::set_cull_mode / set_front_facing_winding compile and give the same image with an ICB created with inheritCullMode = inheritFrontFacingWinding = false");
    rep.note("ICB is Private; encode_ms = encode dispatch alone, reset_ms = resetCommandsInBuffer on the GPU timeline; render_ms = span - empty pass; "
             "the ICB / args are produced in an earlier command buffer before the timed render pass");
    if (!r.artifacts.empty()) rep.note("UNDER VALIDATION (artifact, not counted): a_compact (indirect execution range) " + r.artifacts);
    rep.negative(!r.bad, r.bad ? "FAILED: " + r.badWhat
                               : "all variants bit-identical (image + drawn[]) at every B, cull stand-in exact, built-in control (a_zero with the "
                                 "biggest bucket dropped) detected at every B");
    if (r.bad) rep.status(Status::Failed, r.badWhat);
}

} // namespace

SOC_BENCH("F5-S3", "scene.draws", "Draw emission: GPU-encoded ICB (ranges per cull class) vs indirect draw per bucket vs direct draws", benchDraws);

} // namespace soc
