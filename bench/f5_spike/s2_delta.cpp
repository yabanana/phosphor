// F5-S2: delta updates of a persistent GPU scene (F5.1).
//
// N = 1,048,576 instance records of 80 bytes (layout of the engine's
// GPUInstance: float modelMatrix[16]; u32 meshIndex, materialIndex, flags, pad).
// A CPU mirror holds the authoritative values.  Each frame K = f*N DISTINCT
// random slots change (partial Fisher-Yates, fixed seed); the mirror is updated
// first, then one of three strategies propagates the change to the GPU:
//   (a) scatter: one persistent PRIVATE buffer P; the CPU writes K delta records
//       {u32 slot; u32 pad[3]; Inst} (96 B) into the frame's region of a shared
//       upload buffer (3 regions = 3 frames in flight) and s2_scatter writes
//       P[slot] = data (one thread per record).
//   (b) direct: three persistent SHARED copies S[0..2], one per frame in flight;
//       frame n writes its own changes into S[n%3] and re-applies the changes of
//       frames n-1 and n-2 (values from the mirror = latest), ~3K records.
//   (c) full: the CPU memcpy's all N records into the frame's upload region (what
//       the engine does today, 80 MiB/frame at 1M).
// Consumer s2_consume reads all 20 words of every record and writes one u32 per
// instance (hash + bounding-sphere-vs-plane bit); the CPU checks every output.
//
// Exactness (the future F5.1 self-check): after the frames of each strategy the
// GPU-visible persistent data must equal the mirror byte for byte (P read back
// through a compute-encoder copy, S[last], upload region).  Negative controls
// kept in the code: (a) with ONE delta record omitted on the last frame -> the
// check must find exactly that slot; (b) with the re-application of frame n-2
// skipped on the last frame -> the check must find exactly the slots of frame
// n-2 not rewritten by frames n-1 / n.
//
// Cross-frame wait cost: pipelined frames (f5::FrameRing, 3 in flight), GPU
// bound: [update encoder: scatter of 1% records | nothing] -> cull encoder
// (s2_consume over all N reading the frame's persistent buffer) -> render
// encoder (2048x2048 RGBA8 private target; N/64 instanced quads whose vertex
// shader reads the records, seeded fragment ALU).  Variants:
//   a_barrier   : the first encoder of each frame has barrierAfterQueueStages(
//                 Dispatch | Vertex, Dispatch) (scatter n+1 must not overwrite P
//                 while frame n's cull / vertex stages read it: what the render
//                 graph emits for a persistent buffer it writes),
//   a_nobarrier : same without it (RACY, upper bound of the overlap),
//   b_copies    : per-frame shared copies, no cross-frame barrier needed.
// Intra-frame barriers: scatter -> cull Dispatch->Dispatch (start of the cull
// encoder), cull -> render Dispatch->Vertex (start of the render encoder).
//
// Metrics (the fraction tag is per mille: pm1 = 0.1%, pm10 = 1%, pm100 = 10%,
// pm1000 = 100%):
//   delta.mirror_update.cpu_ms.<tag>   CPU: new random values into the mirror
//   scatter.cpu_write.ms.<tag>         CPU: write K delta records (gather from mirror)
//   scatter.gpu.ms.<tag>               GPU: s2_scatter kernel (ComputeTimer)
//   scatter.bytes.<tag>                CPU bytes written per frame
//   direct.cpu_write.ms.<tag>, direct.bytes.<tag>   CPU: ~3K records into S[n%3]
//   full.cpu_write.ms, full.bytes      CPU: memcpy of all N records
//   consume.private.gpu.ms / consume.shared.gpu.ms / consume.upload.gpu.ms
//   wall.<variant>.ms_per_frame        pipelined wall ms per frame (min and
//                                      median over repetitions in the stats)
//   wall.overlap_lost.ms               a_barrier - b_copies (medians), also .min_ms
//   wall.nobarrier_gain.ms             a_barrier - a_nobarrier (medians)
//   exact.* values                     0/1 results of every exactness check

#include "f5_common.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace soc {
namespace {

constexpr u32 kN        = 1u << 20;
constexpr u32 kInstB    = 80;
constexpr u32 kRecB     = 96;
constexpr u32 kRing     = 3;
constexpr size_t kRegionB = size_t(kN) * kRecB; // one upload region (delta records at f = 1 fit)
constexpr size_t kParamStride = 256;            // params buffer slots: 0..2 scatter counts, 3 consume, 4 render
constexpr u32 kRenderStride = 64;
constexpr u32 kRenderInstances = kN / kRenderStride;

struct Inst {
    float m[16];
    u32 meshIndex, materialIndex, flags, pad;
};
static_assert(sizeof(Inst) == kInstB);
struct DeltaRec {
    u32 slot;
    u32 pad[3];
    Inst data;
};
static_assert(sizeof(DeltaRec) == kRecB);

struct ConsumeParams {
    float plane[4];
    u32 count, pad0, pad1, pad2;
};
struct RenderParams {
    float halfSize;
    u32 stride, iters, pad;
};

constexpr float kPlane[4] = {0.3f, 0.5f, 0.8f, -10.0f};

void randInst(u64& st, Inst& o) {
    for (float& v : o.m) v = float(double(xorshift64(st) >> 40) / 16777216.0 * 200.0 - 100.0);
    o.meshIndex     = u32(xorshift64(st) >> 20);
    o.materialIndex = u32(xorshift64(st) >> 20);
    o.flags         = u32(xorshift64(st) >> 20);
    o.pad           = u32(xorshift64(st) >> 20);
}

// CPU reference of s2_consume.  Returns the expected word and whether the
// visibility bit sits within 1e-3 of the threshold (fma contraction differences).
u32 consumeRef(const Inst& in, bool& borderline) {
    const u32* w = reinterpret_cast<const u32*>(&in);
    u32 h = 2166136261u;
    for (int i = 0; i < 20; ++i) h = (h ^ w[i]) * 16777619u;
    const double radius = std::fabs(double(in.m[0])) * 0.01 + 0.5;
    const double dist = double(in.m[12]) * kPlane[0] + double(in.m[13]) * kPlane[1] + double(in.m[14]) * kPlane[2] + kPlane[3];
    borderline = std::fabs(dist + radius) < 1e-3;
    return (h & 0xFFFFFFFEu) | (dist > -radius ? 1u : 0u);
}

enum class V { ABarrier, ANoBarrier, BCopies };
const char* variantName(V v) { return v == V::ABarrier ? "a_barrier" : v == V::ANoBarrier ? "a_nobarrier" : "b_copies"; }

std::string tagOf(double f) { return "pm" + std::to_string(int(f * 1000.0 + 0.5)); }

struct Fix {
    Context& ctx;
    MTL::Library* lib = nullptr;
    MTL::ComputePipelineState* scatterPso = nullptr;
    MTL::ComputePipelineState* consumePso = nullptr;
    MTL::RenderPipelineState* renderPso = nullptr;
    MTL::Buffer *P = nullptr, *upload = nullptr, *readback = nullptr, *params = nullptr;
    MTL::Buffer* S[kRing] = {};
    MTL::Buffer* out[kRing] = {};
    MTL::Texture* tex[kRing] = {};
    std::vector<Inst> mirror;
    std::vector<u32> perm;
    u64 rng = 0x1234567887654321ull;
    std::vector<u32> p1, p2; // slot lists of the last two frames (strategy b)
    u32 iters = 32;
    explicit Fix(Context& c) : ctx(c) {}

    Inst* region(u32 slot) const { return reinterpret_cast<Inst*>(static_cast<u8*>(upload->contents()) + size_t(slot) * kRegionB); }
    DeltaRec* recRegion(u32 slot) const { return reinterpret_cast<DeltaRec*>(static_cast<u8*>(upload->contents()) + size_t(slot) * kRegionB); }
    u64 regionAddr(u32 slot) const { return upload->gpuAddress() + size_t(slot) * kRegionB; }
    u64 paramAddr(u32 i) const { return params->gpuAddress() + i * kParamStride; }
    template <class T> T* paramPtr(u32 i) const { return reinterpret_cast<T*>(static_cast<u8*>(params->contents()) + i * kParamStride); }

    // Partial Fisher-Yates over a persistent permutation: K distinct slots, in
    // random order (random access on both sides: the worst case for gathers).
    void selectSlots(u32 K, std::vector<u32>& slots) {
        slots.resize(K);
        for (u32 i = 0; i < K; ++i) {
            const u32 j = i + u32(xorshift64(rng) % (kN - i));
            std::swap(perm[i], perm[j]);
            slots[i] = perm[i];
        }
    }
    // Distinct slots without a shuffle (odd stride mod 2^20): used by the
    // pipelined frames, where the CPU must stay far below the GPU time.
    void strideSlots(u32 K, std::vector<u32>& slots) {
        slots.resize(K);
        const u32 base = u32(xorshift64(rng)) & (kN - 1);
        const u32 stride = (u32(xorshift64(rng)) | 1u) & (kN - 1);
        for (u32 i = 0; i < K; ++i) slots[i] = (base + i * stride) & (kN - 1);
    }
    void updateMirror(const std::vector<u32>& slots) {
        for (u32 s : slots) randInst(rng, mirror[s]);
    }

    void initData() {
        mirror.resize(kN);
        perm.resize(kN);
        for (u32 i = 0; i < kN; ++i) {
            perm[i] = i;
            randInst(rng, mirror[i]);
        }
    }

    // ---- GPU helpers (sequential, one submit each) ----
    double scatterMs(u32 slot, u32 count) {
        *paramPtr<u32>(slot) = count;
        ComputeTimer t(ctx);
        MTL4::ComputeCommandEncoder* e = t.begin();
        ctx.table()->setAddress(regionAddr(slot), 0);
        ctx.table()->setAddress(P->gpuAddress(), 1);
        ctx.table()->setAddress(paramAddr(slot), 2);
        e->setComputePipelineState(scatterPso);
        e->setArgumentTable(ctx.table());
        e->dispatchThreads(MTL::Size::Make(count, 1, 1), MTL::Size::Make(64, 1, 1));
        t.lap();
        return t.finish()[0];
    }
    void setConsumeParams() {
        auto* cp = paramPtr<ConsumeParams>(3);
        std::memcpy(cp->plane, kPlane, 16);
        cp->count = kN;
    }
    double consumeMs(MTL::Buffer* src, size_t off, MTL::Buffer* dst) {
        setConsumeParams();
        ComputeTimer t(ctx);
        MTL4::ComputeCommandEncoder* e = t.begin();
        ctx.table()->setAddress(src->gpuAddress() + off, 0);
        ctx.table()->setAddress(dst->gpuAddress(), 1);
        ctx.table()->setAddress(paramAddr(3), 2);
        e->setComputePipelineState(consumePso);
        e->setArgumentTable(ctx.table());
        e->dispatchThreads(MTL::Size::Make(kN, 1, 1), MTL::Size::Make(256, 1, 1));
        t.lap();
        return t.finish()[0];
    }
    // Copy `bytes` from `src` (private) into the shared readback buffer.
    const void* readbackOf(MTL::Buffer* src, size_t bytes) {
        MTL4::CommandBuffer* cb = ctx.beginCommands();
        MTL4::ComputeCommandEncoder* e = cb->computeCommandEncoder();
        e->copyFromBuffer(src, 0, readback, 0, bytes);
        e->endEncoding();
        ctx.submit();
        return readback->contents();
    }
    // Make P equal to the mirror: memcpy into upload region 0, GPU copy.
    void resyncP() {
        std::memcpy(region(0), mirror.data(), size_t(kN) * kInstB);
        MTL4::CommandBuffer* cb = ctx.beginCommands();
        MTL4::ComputeCommandEncoder* e = cb->computeCommandEncoder();
        e->copyFromBuffer(upload, 0, P, 0, size_t(kN) * kInstB);
        e->endEncoding();
        ctx.submit();
    }
    void resyncS() {
        for (u32 i = 0; i < kRing; ++i) std::memcpy(S[i]->contents(), mirror.data(), size_t(kN) * kInstB);
        p1.clear();
        p2.clear();
    }

    // ---- verification ----
    // Slots whose 80 bytes differ from the mirror (at most `cap` listed, the
    // total count in `total`).
    std::vector<u32> diffSlots(const void* got, u64& total, size_t cap = 64) const {
        total = 0;
        std::vector<u32> d;
        if (std::memcmp(got, mirror.data(), size_t(kN) * kInstB) == 0) return d;
        const auto* g = static_cast<const Inst*>(got);
        for (u32 i = 0; i < kN; ++i) {
            if (std::memcmp(&g[i], &mirror[i], kInstB) != 0) {
                ++total;
                if (d.size() < cap) d.push_back(i);
            }
        }
        return d;
    }
    // Every output of s2_consume against the CPU reference over the mirror.
    // Returns the number of wrong outputs; `borderline` counts sphere/plane
    // decisions within 1e-3 whose visibility bit differs (tolerated).
    u64 verifyConsume(const u32* outp, u64& borderline) const {
        u64 bad = 0;
        borderline = 0;
        for (u32 i = 0; i < kN; ++i) {
            bool bl = false;
            const u32 want = consumeRef(mirror[i], bl);
            if (outp[i] == want) continue;
            if (bl && (outp[i] & ~1u) == (want & ~1u)) ++borderline;
            else ++bad;
        }
        return bad;
    }
};

// ---- Strategy frames (sequential: used by the cost measurements and the exactness checks) ----

struct FrameCost {
    double mirrorMs = 0, writeMs = 0, gpuMs = 0, bytes = 0;
    u32 omittedSlot = ~0u;
};

// (a) scatter.  omitRecord >= 0: that record index is left out (negative control).
FrameCost frameScatter(Fix& f, u32 n, u32 K, int omitRecord) {
    FrameCost c;
    std::vector<u32> slots;
    f.selectSlots(K, slots);
    double t0 = nowMs();
    f.updateMirror(slots);
    double t1 = nowMs();
    DeltaRec* dst = f.recRegion(n % kRing);
    u32 count = 0;
    for (u32 i = 0; i < K; ++i) {
        if (int(i) == omitRecord) {
            c.omittedSlot = slots[i];
            continue;
        }
        DeltaRec& r = dst[count++];
        r.slot = slots[i];
        r.pad[0] = r.pad[1] = r.pad[2] = 0;
        r.data = f.mirror[slots[i]];
    }
    double t2 = nowMs();
    c.mirrorMs = t1 - t0;
    c.writeMs = t2 - t1;
    c.bytes = double(count) * kRecB;
    c.gpuMs = f.scatterMs(n % kRing, count);
    return c;
}

// Records of `slots` (values from the mirror = latest) into copy `dstCopy`.
void applySlots(Fix& f, MTL::Buffer* dstCopy, const std::vector<u32>& slots) {
    auto* d = static_cast<Inst*>(dstCopy->contents());
    for (u32 s : slots) d[s] = f.mirror[s];
}

// (b) direct: own changes + the two frames this copy has not seen.  skipOld2:
// negative control (the re-application of frame n-2 is skipped).
FrameCost frameDirect(Fix& f, u32 n, u32 K, bool skipOld2, std::vector<u32>* expectedMissing) {
    FrameCost c;
    std::vector<u32> cur;
    f.selectSlots(K, cur);
    double t0 = nowMs();
    f.updateMirror(cur);
    double t1 = nowMs();
    MTL::Buffer* copy = f.S[n % kRing];
    double bytes = 0;
    if (!skipOld2) {
        applySlots(f, copy, f.p2);
        bytes += double(f.p2.size()) * kInstB;
    }
    applySlots(f, copy, f.p1);
    applySlots(f, copy, cur);
    bytes += double(f.p1.size() + cur.size()) * kInstB;
    double t2 = nowMs();
    if (expectedMissing && skipOld2) {
        // Slots of frame n-2 not rewritten by frames n-1 / n keep a stale value.
        std::vector<u8> mark(kN, 0);
        for (u32 s : f.p1) mark[s] = 1;
        for (u32 s : cur) mark[s] = 1;
        expectedMissing->clear();
        for (u32 s : f.p2)
            if (!mark[s]) expectedMissing->push_back(s);
        std::sort(expectedMissing->begin(), expectedMissing->end());
        expectedMissing->erase(std::unique(expectedMissing->begin(), expectedMissing->end()), expectedMissing->end());
    }
    f.p2 = std::move(f.p1);
    f.p1 = std::move(cur);
    c.mirrorMs = t1 - t0;
    c.writeMs = t2 - t1;
    c.bytes = bytes;
    return c;
}

// (c) full reload into the frame's upload region.
FrameCost frameFull(Fix& f, u32 n, u32 K) {
    FrameCost c;
    std::vector<u32> slots;
    f.selectSlots(K, slots);
    double t0 = nowMs();
    f.updateMirror(slots);
    double t1 = nowMs();
    std::memcpy(f.region(n % kRing), f.mirror.data(), size_t(kN) * kInstB);
    double t2 = nowMs();
    c.mirrorMs = t1 - t0;
    c.writeMs = t2 - t1;
    c.bytes = double(kN) * kInstB;
    return c;
}

// ---- Pipelined frame ----------------------------------------------------------

void encodeFrame(Fix& f, V v, MTL4::CommandBuffer* cb, u32 slot, u32 K) {
    Context& ctx = f.ctx;
    MTL4::ArgumentTable* tb = ctx.table();
    const bool a = v != V::BCopies;
    MTL::Buffer* src = a ? f.P : f.S[slot];
    if (a) {
        MTL4::ComputeCommandEncoder* e = cb->computeCommandEncoder();
        if (v == V::ABarrier) e->barrierAfterQueueStages(MTL::StageDispatch | MTL::StageVertex, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
        *f.paramPtr<u32>(slot) = K;
        tb->setAddress(f.regionAddr(slot), 0);
        tb->setAddress(f.P->gpuAddress(), 1);
        tb->setAddress(f.paramAddr(slot), 2);
        e->setComputePipelineState(f.scatterPso);
        e->setArgumentTable(tb);
        e->dispatchThreads(MTL::Size::Make(K, 1, 1), MTL::Size::Make(64, 1, 1));
        e->endEncoding();
    }
    {   // cull
        MTL4::ComputeCommandEncoder* e = cb->computeCommandEncoder();
        if (a) e->barrierAfterQueueStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
        tb->setAddress(src->gpuAddress(), 0);
        tb->setAddress(f.out[slot]->gpuAddress(), 1);
        tb->setAddress(f.paramAddr(3), 2);
        e->setComputePipelineState(f.consumePso);
        e->setArgumentTable(tb);
        e->dispatchThreads(MTL::Size::Make(kN, 1, 1), MTL::Size::Make(256, 1, 1));
        e->endEncoding();
    }
    {   // render
        MTL4::RenderPassDescriptor* pd = MTL4::RenderPassDescriptor::alloc()->init();
        auto* c = pd->colorAttachments()->object(0);
        c->setTexture(f.tex[slot]);
        c->setLoadAction(MTL::LoadActionClear);
        c->setClearColor(MTL::ClearColor::Make(0, 0, 0, 1));
        c->setStoreAction(MTL::StoreActionStore);
        MTL4::RenderCommandEncoder* re = cb->renderCommandEncoder(pd);
        pd->release();
        re->barrierAfterQueueStages(MTL::StageDispatch, MTL::StageVertex, MTL4::VisibilityOptionDevice);
        re->setViewport(MTL::Viewport{0.0, 0.0, 2048.0, 2048.0, 0.0, 1.0});
        auto* rp = f.paramPtr<RenderParams>(4);
        rp->halfSize = 0.04f;
        rp->stride = kRenderStride;
        rp->iters = f.iters;
        tb->setAddress(src->gpuAddress(), 0);
        tb->setAddress(f.paramAddr(4), 1);
        re->setRenderPipelineState(f.renderPso);
        re->setArgumentTable(tb, MTL::RenderStageVertex | MTL::RenderStageFragment);
        re->drawPrimitives(MTL::PrimitiveTypeTriangleStrip, 0, 4, kRenderInstances);
        re->endEncoding();
    }
}

struct PipeResult {
    double msPerFrame = 0;
    bool exactData = true;   // persistent data == mirror
    bool exactCull = true;   // last frame's cull output == reference over the final mirror
    u64 mismatches = 0, cullBad = 0;
};

// One measurement: `warm` frames, then `frames` timed frames, drain.
PipeResult runPipelined(Fix& f, f5::FrameRing& ring, V v, u32 warm, u32 frames, u32 K, bool verify) {
    Context& ctx = f.ctx;
    if (v == V::BCopies) f.resyncS();
    else f.resyncP();
    std::vector<u32> cur;
    double t0 = 0;
    for (u32 fr = 0; fr < warm + frames; ++fr) {
        if (fr == warm) t0 = nowMs();
        MTL4::CommandBuffer* cb = ring.begin(); // waits for the frame that last used this slot
        const u32 slot = ring.slot();
        f.strideSlots(K, cur);
        f.updateMirror(cur);
        if (v == V::BCopies) {
            applySlots(f, f.S[slot], f.p2);
            applySlots(f, f.S[slot], f.p1);
            applySlots(f, f.S[slot], cur);
            f.p2 = std::move(f.p1);
            f.p1 = cur;
        } else {
            DeltaRec* dst = f.recRegion(slot);
            for (u32 i = 0; i < K; ++i) {
                dst[i].slot = cur[i];
                dst[i].pad[0] = dst[i].pad[1] = dst[i].pad[2] = 0;
                dst[i].data = f.mirror[cur[i]];
            }
        }
        encodeFrame(f, v, cb, slot, K);
        ring.commit();
    }
    ring.drain();
    PipeResult r;
    r.msPerFrame = (nowMs() - t0) / frames;
    if (verify) {
        const u32 last = u32((ring.frame() - 1) % kRing);
        const void* got = v == V::BCopies ? static_cast<const void*>(f.S[last]->contents()) : f.readbackOf(f.P, size_t(kN) * kInstB);
        std::vector<u32> d = f.diffSlots(got, r.mismatches);
        r.exactData = d.empty();
        u64 bl = 0;
        r.cullBad = f.verifyConsume(static_cast<const u32*>(f.out[last]->contents()), bl);
        r.exactCull = r.cullBad == 0;
    }
    (void)ctx;
    return r;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0 : v[v.size() / 2];
}

void benchDelta(Context& ctx, Report& rep) {
    const bool quick = ctx.quick();
    Fix f(ctx);
    f.lib = f5::f5Library(ctx, "s2_delta.metal");
    f.scatterPso = ctx.compute(f.lib, "s2_scatter");
    f.consumePso = ctx.compute(f.lib, "s2_consume");

    f.P = ctx.buffer(size_t(kN) * kInstB, MTL::ResourceStorageModePrivate);
    f.upload = ctx.buffer(kRegionB * kRing);
    f.readback = ctx.buffer(size_t(kN) * kInstB);
    f.params = ctx.buffer(8 * kParamStride);
    for (u32 i = 0; i < kRing; ++i) {
        f.S[i] = ctx.buffer(size_t(kN) * kInstB);
        f.out[i] = ctx.buffer(size_t(kN) * 4);
    }
    f.initData();
    f.setConsumeParams();

    // Render pipeline + targets.
    MTL4::RenderPipelineDescriptor* rd = MTL4::RenderPipelineDescriptor::alloc()->init();
    rd->setVertexFunctionDescriptor(ctx.function(f.lib, "s2_vs"));
    rd->setFragmentFunctionDescriptor(ctx.function(f.lib, "s2_fs"));
    rd->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatRGBA8Unorm);
    f.renderPso = ctx.render(rd);
    rd->release();
    for (u32 i = 0; i < kRing; ++i) {
        MTL::TextureDescriptor* td = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatRGBA8Unorm, 2048, 2048, false);
        td->setUsage(MTL::TextureUsageRenderTarget);
        td->setStorageMode(MTL::StorageModePrivate);
        f.tex[i] = ctx.texture(td);
    }

    const std::vector<double> fractions = quick ? std::vector<double>{0.001, 0.1} : std::vector<double>{0.001, 0.01, 0.1, 1.0};
    const u32 reps = quick ? 6 : std::max<u32>(ctx.reps(), 8);
    const u32 warmFrames = kRing;
    bool allExact = true;
    std::string failures;
    auto fail = [&](const std::string& what) {
        allExact = false;
        failures += what + "; ";
    };

    // Exactness of one strategy after its frames.
    auto checkP = [&](const std::string& what) {
        u64 total = 0;
        std::vector<u32> d = f.diffSlots(f.readbackOf(f.P, size_t(kN) * kInstB), total);
        if (!d.empty()) fail(what + " P != mirror (" + std::to_string(total) + " slots)");
        return d.empty();
    };

    // Consume timing + exact output check from one source.
    auto consumeGroup = [&](const char* name, MTL::Buffer* src, size_t off) {
        ctx.keepWarm(30);
        Stats s = ctx.measure([&] { return f.consumeMs(src, off, f.out[0]); }, quick ? 5 : 0);
        u64 bl = 0;
        const u64 bad = f.verifyConsume(static_cast<const u32*>(f.out[0]->contents()), bl);
        if (bad) fail(std::string("consume from ") + name + " wrong outputs: " + std::to_string(bad));
        rep.metric(std::string("consume.") + name + ".gpu.ms", "ms", s, {{"n", double(kN)}}, false);
        ctx.log("F5-S2 consume.%s: median %.3f ms, %llu wrong, %llu borderline", name, s.median, (unsigned long long)bad,
                (unsigned long long)bl);
    };

    // ---- Part 1: cost of each strategy per change fraction ----
    u32 n = 0;
    std::vector<double> mirrorAll;
    for (size_t fi = 0; fi < fractions.size(); ++fi) {
        const double frac = fractions[fi];
        const u32 K = std::max<u32>(1, u32(frac * kN + 0.5));
        const std::string tag = tagOf(frac);
        const bool last = fi + 1 == fractions.size();
        ctx.keepWarm(30);

        // (a) scatter
        {
            f.resyncP();
            n = 0;
            for (u32 i = 0; i < warmFrames; ++i) frameScatter(f, n++, K, -1);
            std::vector<double> mir, wr, gpu;
            double bytes = 0;
            Stats s = ctx.measure(
                [&] {
                    const FrameCost c = frameScatter(f, n++, K, -1);
                    mir.push_back(c.mirrorMs);
                    wr.push_back(c.writeMs);
                    gpu.push_back(c.gpuMs);
                    bytes = c.bytes;
                    return c.writeMs;
                },
                reps);
            rep.metric("scatter.cpu_write.ms." + tag, "ms", s, {{"k", double(K)}}, false);
            rep.metric("scatter.gpu.ms." + tag, "ms", phosphor::soc::computeStats(gpu), {{"k", double(K)}}, false);
            rep.metric("delta.mirror_update.cpu_ms." + tag, "ms", phosphor::soc::computeStats(mir), {{"k", double(K)}}, false);
            rep.value("scatter.bytes." + tag, "bytes", bytes, {{"k", double(K)}}, false);
            ctx.log("F5-S2 scatter f=%g K=%u: cpu %.3f ms, gpu %.3f ms, %.1f KiB", frac, K, s.median, phosphor::soc::computeStats(gpu).median, bytes / 1024);
            if (checkP("scatter " + tag)) {}
            if (last) consumeGroup("private", f.P, 0);
        }
        // (b) direct
        {
            f.resyncS();
            n = 0;
            for (u32 i = 0; i < warmFrames; ++i) frameDirect(f, n++, K, false, nullptr);
            std::vector<double> mir;
            double bytes = 0;
            Stats s = ctx.measure(
                [&] {
                    const FrameCost c = frameDirect(f, n++, K, false, nullptr);
                    mir.push_back(c.mirrorMs);
                    bytes = c.bytes;
                    return c.writeMs;
                },
                reps);
            rep.metric("direct.cpu_write.ms." + tag, "ms", s, {{"k", double(K)}}, false);
            rep.value("direct.bytes." + tag, "bytes", bytes, {{"k", double(K)}}, false);
            ctx.log("F5-S2 direct f=%g K=%u: cpu %.3f ms, %.1f KiB", frac, K, s.median, bytes / 1024);
            const u32 lastSlot = (n - 1) % kRing;
            u64 total = 0;
            if (!f.diffSlots(f.S[lastSlot]->contents(), total).empty()) fail("direct " + tag + " S != mirror (" + std::to_string(total) + " slots)");
            if (last) consumeGroup("shared", f.S[lastSlot], 0);
        }
        // (c) full: independent of f, measured once (at the smallest f; its mirror update is irrelevant)
        if (fi == 0) {
            n = 0;
            for (u32 i = 0; i < warmFrames; ++i) frameFull(f, n++, K);
            double bytes = 0;
            Stats s = ctx.measure(
                [&] {
                    const FrameCost c = frameFull(f, n++, K);
                    bytes = c.bytes;
                    return c.writeMs;
                },
                reps);
            rep.metric("full.cpu_write.ms", "ms", s, {{"n", double(kN)}}, false);
            rep.value("full.bytes", "bytes", bytes, {{"n", double(kN)}}, false);
            ctx.log("F5-S2 full: cpu %.3f ms, %.1f MiB", s.median, bytes / 1048576.0);
            const u32 lastSlot = (n - 1) % kRing;
            u64 total = 0;
            if (!f.diffSlots(f.region(lastSlot), total).empty()) fail("full upload region != mirror (" + std::to_string(total) + " slots)");
            consumeGroup("upload", f.upload, size_t(lastSlot) * kRegionB);
        }
    }

    // ---- Negative controls (kept in the code) ----
    // (a) one record omitted on the last frame: exactly that slot must differ.
    bool negA = false, negB = false;
    std::string negDetail;
    {
        const u32 K = std::max<u32>(1, u32(0.01 * kN));
        f.resyncP();
        u32 omitted = ~0u;
        for (u32 i = 0; i < 5; ++i) frameScatter(f, i, K, -1);
        const FrameCost c = frameScatter(f, 5, K, int(K / 2));
        omitted = c.omittedSlot;
        u64 total = 0;
        std::vector<u32> d = f.diffSlots(f.readbackOf(f.P, size_t(kN) * kInstB), total);
        negA = d.size() == 1 && total == 1 && d[0] == omitted;
        negDetail += "scatter with one record omitted: " + std::to_string(total) + " slot(s) differ" +
                     (d.size() == 1 ? ", slot " + std::to_string(d[0]) + " == omitted " + std::to_string(omitted) : "") + "; ";
    }
    // (b) re-application of frame n-2 skipped on the last frame.
    {
        const u32 K = std::max<u32>(1, u32(0.01 * kN));
        f.resyncS();
        for (u32 i = 0; i < 5; ++i) frameDirect(f, i, K, false, nullptr);
        std::vector<u32> expected;
        frameDirect(f, 5, K, true, &expected);
        u64 total = 0;
        std::vector<u32> d = f.diffSlots(f.S[5 % kRing]->contents(), total, size_t(kN));
        std::sort(d.begin(), d.end());
        negB = !expected.empty() && d == expected;
        negDetail += "direct with frame n-2 re-application skipped: " + std::to_string(total) + " slot(s) differ, expected " +
                     std::to_string(expected.size()) + (negB ? " (identical sets)" : " (SETS DIFFER)") + "; ";
    }
    ctx.log("F5-S2 negative controls: %s", negDetail.c_str());

    // ---- Part 2: cross-frame wait cost with pipelined frames ----
    const u32 K2 = kN / 100;
    const u32 warm2 = quick ? 10 : 30, frames2 = quick ? 40 : 120, reps2 = quick ? 2 : 5;
    f5::FrameRing rings[3] = {f5::FrameRing(ctx, kRing), f5::FrameRing(ctx, kRing), f5::FrameRing(ctx, kRing)};
    const V variants[3] = {V::ABarrier, V::ANoBarrier, V::BCopies};

    // Make the frame GPU-bound (~4 ms): scale the fragment iterations from a
    // short b_copies run (calibration only; the result is informational).
    for (int it = 0; it < 4; ++it) {
        ctx.keepWarm(30);
        const PipeResult r = runPipelined(f, rings[2], V::BCopies, 10, 30, K2, false);
        ctx.log("F5-S2 calibration: iters=%u -> %.3f ms/frame", f.iters, r.msPerFrame);
        if (r.msPerFrame > 3.0 && r.msPerFrame < 6.0) break;
        f.iters = std::clamp<u32>(u32(f.iters * 4.0 / std::max(0.3, r.msPerFrame - 0.5)), 4, 8192);
    }

    std::vector<double> wall[3];
    PipeResult last[3];
    for (u32 r = 0; r < reps2; ++r) {
        for (u32 k = 0; k < 3; ++k) {
            const u32 vi = (r + k) % 3;
            ctx.keepWarm(30);
            const PipeResult pr = runPipelined(f, rings[vi], variants[vi], warm2, frames2, K2, true);
            wall[vi].push_back(pr.msPerFrame);
            last[vi] = pr;
            ctx.log("F5-S2 rep %u %s: %.3f ms/frame, data %s, cull %s", r, variantName(variants[vi]), pr.msPerFrame,
                    pr.exactData ? "exact" : "MISMATCH", pr.exactCull ? "exact" : "WRONG");
            // The correct variants must stay exact in every repetition; the racy one is informational.
            if (variants[vi] != V::ANoBarrier && (!pr.exactData || !pr.exactCull))
                fail(std::string("pipelined ") + variantName(variants[vi]) + " rep " + std::to_string(r) + " data " +
                     (pr.exactData ? "ok" : std::to_string(pr.mismatches) + " bad slots") + " cull " + (pr.exactCull ? "ok" : std::to_string(pr.cullBad) + " bad"));
        }
    }
    for (u32 vi = 0; vi < 3; ++vi)
        rep.metric(std::string("wall.") + variantName(vi == 0 ? V::ABarrier : vi == 1 ? V::ANoBarrier : V::BCopies) + ".ms_per_frame", "ms",
                   phosphor::soc::computeStats(wall[vi]), {{"k", double(K2)}, {"frames", double(frames2)}, {"iters", double(f.iters)}}, false);
    const double ab = median(wall[0]), nb = median(wall[1]), bc = median(wall[2]);
    const double abMin = *std::min_element(wall[0].begin(), wall[0].end()), bcMin = *std::min_element(wall[2].begin(), wall[2].end());
    rep.value("wall.overlap_lost.ms", "ms", ab - bc, {}, false);
    rep.value("wall.overlap_lost.min_ms", "ms", abMin - bcMin, {}, false);
    rep.value("wall.nobarrier_gain.ms", "ms", ab - nb, {}, false);
    if (!last[1].exactData || !last[1].exactCull)
        rep.note("a_nobarrier (racy upper bound) produced " + std::to_string(last[1].mismatches) + " data mismatches / " +
                 std::to_string(last[1].cullBad) + " wrong cull outputs in its last repetition, as expected for a race; its timing is only an upper bound of the overlap");
    rep.note("Part 2 uses distinct slots from an odd stride (no shuffle) so CPU work per frame stays far below GPU time; part 1 uses a random partial Fisher-Yates order.");
    rep.value("exact.all_normal_checks", "bool", allExact ? 1 : 0);
    rep.value("exact.negative_scatter_detected", "bool", negA ? 1 : 0);
    rep.value("exact.negative_direct_detected", "bool", negB ? 1 : 0);

    rep.negative(allExact && negA && negB, std::string(allExact ? "all exactness checks equal; " : "FAILED checks: " + failures) + negDetail);
    if (!allExact) rep.status(Status::Failed, failures);
    else if (!negA || !negB) rep.status(Status::Failed, "negative control not detected");
}

} // namespace

SOC_BENCH("F5-S2", "scene.delta", "Delta updates of a persistent GPU scene: scatter vs direct shared copies vs full, cross-frame wait", benchDelta);

} // namespace soc
