// B-12: parameter buffer / partial-render threshold (OPT-0.6).
// Serves S-TBDR-3 of docs/APPLE_SOC_PLAYBOOK.md.
//
// One render pass (2048x2048 RGBA8, clear + store, no depth, no culling), ONE
// draw of N small triangles (3.5 px legs = 6 covered pixels, positions hashed
// over the whole target) with V = 0/4/8/16 float4 varyings, N from 64Ki to
// 64Mi (--quick: 4Mi) in steps of 2^(1/2).  ns per triangle = (span - empty
// span) / N.  A partial render (the tiler's parameter buffer is full, the GPU
// flushes the tiles mid-pass) shows as a rise of ns/triangle that stays; a
// second curve (16 varyings, four RGBA32F attachments, 256 MiB stored per
// flush) makes each flush expensive.  Measured on M5 Max: knees at 23.7 M
// (16 varyings, both curves; larger jump with the heavy attachments), 47 M
// (8), 67 M (4), none up to 67 M without varyings.
//
// Knee rule (documented, applied automatically): baseline = the smallest
// ns/triangle of the curve at N >= 128Ki (steady state; smaller passes are
// dominated by the ~60 us pass overhead); the knee is the first N after the
// baseline point whose ns/triangle exceeds 1.3 x baseline and such that every
// later N also does.  Not found -> no `threshold.*` metric, Status::Partial and
// `max_n_tried.*` says how far the curve went.
//
// Verification (CPU): a pass with the triangles on a regular 8 px grid (no
// overlap, same shaders) is read back and every triangle's origin pixel must hold
// its colour; the visibility (occlusion) counter must be 6 x N.

#include "harness.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace soc {
namespace {

constexpr u32 kDim = 2048;
const u32 kVaryings[] = {0, 4, 8, 16};

struct Params {
    u32 count, mode, zero, pad;
};

u32 hash(u32 x) {
    x ^= x >> 16;
    x *= 2246822519u;
    x ^= x >> 13;
    x *= 3266489917u;
    x ^= x >> 16;
    return x;
}

struct Rig {
    MTL::RenderPipelineState* pso[4] = {};
    MTL::Texture* target = nullptr;
    MTL::Buffer* params  = nullptr;
    MTL::Buffer* visibility = nullptr;
};

void encode(Context& ctx, const Rig& r, MTL4::CommandBuffer* cmd, u32 vIndex, u32 n, bool count) {
    MTL4::RenderPassDescriptor* pd = MTL4::RenderPassDescriptor::alloc()->init();
    auto* c = pd->colorAttachments()->object(0);
    c->setTexture(r.target);
    c->setLoadAction(MTL::LoadActionClear);
    c->setStoreAction(MTL::StoreActionStore);
    c->setClearColor(MTL::ClearColor::Make(0, 0, 0, 0));
    if (count) {
        pd->setVisibilityResultBuffer(r.visibility);
        pd->setVisibilityResultType(MTL::VisibilityResultTypeAccumulate);
    }
    MTL4::RenderCommandEncoder* re = cmd->renderCommandEncoder(pd);
    pd->release();
    re->setRenderPipelineState(r.pso[vIndex]);
    if (count) re->setVisibilityResultMode(MTL::VisibilityResultModeCounting, 0);
    ctx.table()->setAddress(r.params->gpuAddress(), 0);
    re->setArgumentTable(ctx.table(), MTL::RenderStageVertex | MTL::RenderStageFragment);
    re->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(3) * n);
    re->endEncoding();
}

// Heavy-flush variant (16 varyings, 4 RGBA32F attachments): see the shader.
struct HeavyRig {
    MTL::RenderPipelineState* pso = nullptr;
    MTL::Texture* targets[4] = {};
};

double timeHeavy(Context& ctx, const Rig& r, const HeavyRig& h, u32 n) {
    CommandTimer t(ctx);
    MTL4::RenderPassDescriptor* pd = MTL4::RenderPassDescriptor::alloc()->init();
    for (u32 i = 0; i < 4; ++i) {
        auto* c = pd->colorAttachments()->object(i);
        c->setTexture(h.targets[i]);
        c->setLoadAction(MTL::LoadActionClear);
        c->setStoreAction(MTL::StoreActionStore);
        c->setClearColor(MTL::ClearColor::Make(0, 0, 0, 0));
    }
    MTL4::RenderCommandEncoder* re = t.begin()->renderCommandEncoder(pd);
    pd->release();
    re->setRenderPipelineState(h.pso);
    ctx.table()->setAddress(r.params->gpuAddress(), 0);
    re->setArgumentTable(ctx.table(), MTL::RenderStageVertex | MTL::RenderStageFragment);
    re->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(3) * n);
    re->endEncoding();
    return t.finish() - ctx.emptySpanMs();
}

void setParams(const Rig& r, u32 n, u32 mode) {
    const Params p{n, mode, 0, 0};
    std::memcpy(r.params->contents(), &p, sizeof(p));
}

double timeRun(Context& ctx, const Rig& r, u32 vIndex, u32 n) {
    CommandTimer t(ctx);
    encode(ctx, r, t.begin(), vIndex, n, false);
    return t.finish() - ctx.emptySpanMs();
}

std::string num(double v, size_t n = 6) { return std::to_string(v).substr(0, n); }

// Grid-mode pass at N = 65536: every triangle's origin pixel must hold its colour, and the
// visibility counter must be 6 x N.  Returns wrong pixels (+1 if the counter is off).
u32 verify(Context& ctx, const Rig& r, u32 vIndex, u64* counted) {
    const u32 n = 65536;
    setParams(r, n, 1);
    MTL::Buffer* rb = ctx.buffer(size_t(kDim) * kDim * 4);
    std::memset(rb->contents(), 0, rb->length());
    std::memset(r.visibility->contents(), 0, 8);
    MTL4::CommandBuffer* cmd = ctx.beginCommands();
    encode(ctx, r, cmd, vIndex, n, true);
    MTL4::ComputeCommandEncoder* ce = cmd->computeCommandEncoder();
    ce->barrierAfterQueueStages(MTL::StageFragment, MTL::StageBlit, MTL4::VisibilityOptionDevice);
    ce->copyFromTexture(r.target, 0, 0, MTL::Origin::Make(0, 0, 0), MTL::Size::Make(kDim, kDim, 1), rb, 0, kDim * 4,
                        size_t(kDim) * kDim * 4);
    ce->endEncoding();
    ctx.submit();
    const u8* px = static_cast<const u8*>(rb->contents());
    u32 wrong = 0;
    for (u32 t = 0; t < n; ++t) {
        const u32 x = 4 + (t & 255u) * 8, y = 4 + (t >> 8) * 8;
        const u32 h = hash(t + 12345u);
        const u8* p = px + (size_t(y) * kDim + x) * 4;
        if (p[0] != (h & 255u) || p[1] != ((h >> 8) & 255u) || p[2] != ((h >> 16) & 255u) || p[3] != 255) ++wrong;
    }
    *counted = *static_cast<const u64*>(r.visibility->contents());
    ctx.log("B-12 verify v%u: pixel mismatches %u, visibility %llu", vIndex, wrong, static_cast<unsigned long long>(*counted));
    if (*counted != u64(6) * n) ++wrong;
    return wrong;
}

struct Curve {
    std::vector<u32> n;
    std::vector<double> ms, ns; // ns per triangle
};

// The knee rule (see the file comment).  Returns the index of the knee or -1.
int findKnee(const Curve& c, double* baseline) {
    int base = -1;
    for (size_t i = 0; i < c.n.size(); ++i)
        if (c.n[i] >= 131072 && (base < 0 || c.ns[i] < c.ns[size_t(base)])) base = int(i);
    if (base < 0) return -1;
    *baseline = c.ns[size_t(base)];
    for (size_t i = size_t(base) + 1; i < c.n.size(); ++i) {
        bool stays = true;
        for (size_t j = i; j < c.n.size(); ++j) stays &= c.ns[j] > 1.3 * *baseline;
        if (stays) return int(i);
    }
    return -1;
}

void benchPartialRender(Context& ctx, Report& rep) {
    MTL::Library* lib = ctx.library("b12_partial_render.metal");
    Rig r;
    const char* vs[4] = {"b12_vs0", "b12_vs4", "b12_vs8", "b12_vs16"};
    const char* fs[4] = {"b12_fs0", "b12_fs4", "b12_fs8", "b12_fs16"};
    for (int k = 0; k < 4; ++k) {
        MTL4::RenderPipelineDescriptor* d = MTL4::RenderPipelineDescriptor::alloc()->init();
        d->setVertexFunctionDescriptor(ctx.function(lib, vs[k]));
        d->setFragmentFunctionDescriptor(ctx.function(lib, fs[k]));
        d->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatRGBA8Unorm);
        r.pso[k] = ctx.render(d);
        d->release();
    }
    MTL::TextureDescriptor* td = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatRGBA8Unorm, kDim, kDim, false);
    td->setUsage(MTL::TextureUsageRenderTarget);
    td->setStorageMode(MTL::StorageModePrivate);
    r.target     = ctx.texture(td);
    r.params     = ctx.buffer(256);
    r.visibility = ctx.buffer(64);

    // N grid: 2^16 .. 2^24 in half-octave steps (quick: up to 2^22, fewer repetitions).
    std::vector<u32> ns;
    // Up to 2^26: the first knees appear at 23.7 M (16 varyings), 47 M (8), 67 M (4) triangles;
    // none up to 16.7 M, where the sweep stopped at first (measured).
    const int maxK = ctx.quick() ? 22 : 26;
    for (int k = 16; k <= maxK; ++k) {
        ns.push_back(1u << k);
        if (k < maxK) ns.push_back(u32(std::llround(double(1u << k) * 1.41421356)));
    }
    u32 wrong = 0;
    std::vector<Curve> curves(4);
    std::vector<int> knees(4, -1);
    std::vector<double> baselines(4, 0);
    for (u32 vi = 0; vi < 4; ++vi) {
        u64 counted = 0;
        wrong += verify(ctx, r, vi, &counted);
        if (counted != 6ull * 65536) ctx.log("B-12 varyings %u: visibility count %llu (expected %u)", kVaryings[vi],
                                             static_cast<unsigned long long>(counted), 6 * 65536);
        Curve& cv = curves[vi];
        for (u32 n : ns) {
            setParams(r, n, 0);
            ctx.keepWarm(15);
            const u32 reps = n >= (1u << 22) ? 7 : ctx.quick() ? 9 : 15;
            timeRun(ctx, r, vi, n); // first touch of this size
            const Stats s = ctx.measure([&] { return timeRun(ctx, r, vi, n); }, reps);
            Stats per = s; // ns per triangle
            const double k = 1e6 / double(n);
            per.median = s.median * k;
            per.min    = s.min * k;
            per.max    = s.max * k;
            per.p10    = s.p10 * k;
            per.p90    = s.p90 * k;
            per.mean   = s.mean * k;
            const std::string tag = "varyings_" + std::to_string(kVaryings[vi]) + ".n_" + std::to_string(n);
            rep.metric("ns_per_tri." + tag, "ns", per, {{"varyings", double(kVaryings[vi])}, {"n", double(n)}, {"ms", s.median}}, false);
            cv.n.push_back(n);
            cv.ms.push_back(s.median);
            cv.ns.push_back(per.median);
        }
        double base = 0;
        knees[vi]      = findKnee(cv, &base);
        baselines[vi]  = base;
        const std::string v = std::to_string(kVaryings[vi]);
        rep.value("baseline_ns_per_tri.varyings_" + v, "ns", base, {{"varyings", double(kVaryings[vi])}}, false);
        rep.value("max_n_tried.varyings_" + v, "triangles", double(ns.back()), {{"varyings", double(kVaryings[vi])}});
        if (knees[vi] >= 0)
            rep.value("threshold.varyings_" + v, "triangles", double(cv.n[size_t(knees[vi])]), {{"varyings", double(kVaryings[vi])}, {"baseline_ns", base}});
        ctx.log("B-12 varyings %2u: baseline %.3f ns/tri, knee %s (%u), max N %u, ns/tri at max %.3f", kVaryings[vi], base,
                knees[vi] >= 0 ? "found" : "none", knees[vi] >= 0 ? cv.n[size_t(knees[vi])] : 0u, ns.back(), cv.ns.back());
    }

    // --- Heavy flush: 16 varyings, 4 x RGBA32F (256 MiB stored per flush) ------------------
    // With one RGBA8 target no knee appears up to 16.7 M triangles although
    // 16 float4 varyings x 3 vertices x 16.7 M = ~13 GB of vertex output
    // cannot fit any parameter buffer: partial renders must happen but a
    // flush of 16 MiB is invisible.  Here a flush costs ~0.5 ms.
    Curve heavy;
    int heavyKnee = -1;
    double heavyBase = 0;
    {
        HeavyRig h;
        MTL4::RenderPipelineDescriptor* d = MTL4::RenderPipelineDescriptor::alloc()->init();
        d->setVertexFunctionDescriptor(ctx.function(lib, "b12_vs16"));
        d->setFragmentFunctionDescriptor(ctx.function(lib, "b12_fs16_mrt"));
        for (u32 i = 0; i < 4; ++i) d->colorAttachments()->object(i)->setPixelFormat(MTL::PixelFormatRGBA32Float);
        h.pso = ctx.render(d);
        d->release();
        MTL::TextureDescriptor* hd =
            MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatRGBA32Float, kDim, kDim, false);
        hd->setUsage(MTL::TextureUsageRenderTarget);
        hd->setStorageMode(MTL::StorageModePrivate);
        for (u32 i = 0; i < 4; ++i) h.targets[i] = ctx.texture(hd);
        std::vector<u32> hn;
        for (int k = 10; k <= maxK; ++k) {
            hn.push_back(1u << k);
            if (k < maxK) hn.push_back(u32(std::llround(double(1u << k) * 1.41421356)));
        }
        for (u32 n : hn) {
            setParams(r, n, 0);
            ctx.keepWarm(15);
            timeHeavy(ctx, r, h, n);
            const Stats s = ctx.measure([&] { return timeHeavy(ctx, r, h, n); }, n >= (1u << 22) ? 7 : ctx.quick() ? 7 : 11);
            const double per = s.median * 1e6 / double(n);
            rep.value("ns_per_tri.heavy16.n_" + std::to_string(n), "ns", per, {{"n", double(n)}, {"ms", s.median}}, false);
            heavy.n.push_back(n);
            heavy.ms.push_back(s.median);
            heavy.ns.push_back(per);
        }
        heavyKnee = findKnee(heavy, &heavyBase);
        rep.value("heavy16.fixed_ms", "ms", heavy.ms.front(), {{"n", double(heavy.n.front())}}, false);
        if (heavyKnee >= 0)
            rep.value("threshold.heavy16", "triangles", double(heavy.n[size_t(heavyKnee)]), {{"baseline_ns", heavyBase}});
        std::string curve;
        for (size_t i = 0; i < heavy.n.size(); ++i)
            curve += std::to_string(heavy.n[i]) + ":" + num(heavy.ns[i], 5) + " ";
        ctx.log("B-12 heavy16 (4 x RGBA32F): baseline %.3f ns/tri, knee %s (%u); curve %s", heavyBase,
                heavyKnee >= 0 ? "found" : "none", heavyKnee >= 0 ? heavy.n[size_t(heavyKnee)] : 0u, curve.c_str());
    }

    // --- Negative controls --------------------------------------------------------------
    // 1. Below the knee (or everywhere when there is none) time is linear in N: from 256Ki on, each
    //    step of sqrt(2) in N raises the time by 1.41 x within 1.2..1.65 while ns/triangle is within
    //    30% of the baseline.
    {
        bool ok = true;
        std::string bad;
        u32 pairs = 0;
        for (u32 vi = 0; vi < 4; ++vi) {
            const Curve& c = curves[vi];
            for (size_t i = 1; i < c.n.size(); ++i) {
                if (c.n[i - 1] < 262144 || (knees[vi] >= 0 && int(i) >= knees[vi]) || c.ns[i] > 1.3 * baselines[vi]) continue;
                ++pairs;
                const double ratio = c.ms[i] / c.ms[i - 1], want = double(c.n[i]) / double(c.n[i - 1]);
                if (ratio < 0.85 * want || ratio > 1.15 * want) {
                    ok = false;
                    bad += "v" + std::to_string(kVaryings[vi]) + "@" + std::to_string(c.n[i]) + "=" + num(ratio, 4) + " ";
                }
            }
        }
        rep.negative(ok && pairs >= 4, "time linear in N below the knee: " + std::to_string(pairs) + " steps within 15% of the N ratio" +
                                           (ok ? "" : ", off: " + bad));
    }
    // 2. More varyings never raise the threshold: thr(16) <= thr(0) (missing = beyond the range),
    //    and a triangle with 16 varyings costs more than one with none at 1Mi triangles.
    {
        auto thr = [&](u32 vi) { return knees[vi] >= 0 ? double(curves[vi].n[size_t(knees[vi])]) : 1e18; };
        bool mono = thr(3) <= thr(0) * 1.0001;
        size_t i1m = 0;
        for (size_t i = 0; i < curves[0].n.size(); ++i)
            if (curves[0].n[i] == (1u << 20)) i1m = i;
        const double c0 = curves[0].ns[i1m], c16 = curves[3].ns[i1m];
        rep.negative(mono && c16 > c0, "threshold(16 varyings) <= threshold(0)" + std::string(mono ? " ok" : " VIOLATED") +
                                           "; ns/tri at 1Mi: 0 varyings " + num(c0, 5) + " < 16 varyings " + num(c16, 5));
    }
    // 3. Every triangle of the grid pass is where the CPU says, and the visibility counter counts 6 x N.
    rep.negative(wrong == 0, wrong == 0 ? "grid pass 65536 triangles x 4 varyings: all origin pixels match the CPU, visibility count = 6 x N"
                                        : std::to_string(wrong) + " wrong pixels/counters in the grid verification");
    if (wrong) rep.status(Status::Failed, "verification failed");

    std::string missing;
    for (u32 vi = 0; vi < 4; ++vi)
        if (knees[vi] < 0) missing += std::to_string(kVaryings[vi]) + " ";
    if (!missing.empty())
        rep.status(Status::Partial, "no knee (ns/triangle > 1.3 x baseline and staying) up to N = " + std::to_string(ns.back()) +
                                        " for varyings: " + missing);
    rep.note("threshold = first N of the knee rule (see b12_partial_render.cpp); resolution 2^(1/2); baseline = smallest ns/triangle at N >= 128Ki; "
             "spans above ~2 ms at large N are exposed to other GPU clients (noted deviation from the 0.1-2 ms protocol)");
}

} // namespace

SOC_BENCH("B-12", "partial_render", "Parameter buffer: ns per triangle vs triangle count with 0/4/8/16 varyings (partial-render knee)", benchPartialRender);

} // namespace soc
