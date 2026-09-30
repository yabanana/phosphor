// B-14: cost of load/store actions and of an empty render pass, per format and
// resolution.  Serves S-TBDR-4 of docs/APPLE_SOC_PLAYBOOK.md.
//
// One pass = clear|load + one full-screen triangle whose fragment shader
// writes a pixel-distinct, incompressible value + store|dontCare.  Variants
// (same draw, so the shading cost cancels):
//   A  clear, dontCare   -> the tile is written nowhere
//   B  clear, store      -> B - A = store cost
//   C  load,  store      -> C - B = load cost
//   M  clear, dontCare on a MEMORYLESS texture (RGBA8/16F/32F only)
// plus the empty pass (64x64 RGBA8 clear + dontCare, one 4096-pixel triangle), alone (minus
// the empty CommandTimer span) and 32 chained.  Time = CommandTimer span.
//
// Verification: pass B on the smallest texture, then a pass with Load and no
// draw (must keep the contents), read back to a buffer with a blit and
// compared to the CPU model of the shader hash (a 64x64 corner and the last
// 64 pixels of the last row).

#include "harness.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace soc {
namespace {

struct Fmt {
    const char* name;
    MTL::PixelFormat pf;
    u32 bpp;
    bool depth;
};
struct Res {
    const char* name;
    u32 w, h;
};

const Fmt kFormats[] = {{"rgba8", MTL::PixelFormatRGBA8Unorm, 4, false},
                        {"rgba16f", MTL::PixelFormatRGBA16Float, 8, false},
                        {"rgba32f", MTL::PixelFormatRGBA32Float, 16, false},
                        {"depth32f", MTL::PixelFormatDepth32Float, 4, true}};
const Res kResolutions[] = {{"1080p", 1920, 1080}, {"1440p", 2560, 1440}, {"1800p", 3200, 1800}, {"2160p", 3840, 2160}};

u32 hash(u32 x, u32 y, u32 c) {
    u32 h = x * 73856093u ^ y * 19349663u ^ (c * 83492791u + 1u);
    h ^= h >> 15;
    h *= 2246822519u;
    h ^= h >> 13;
    return h;
}

struct Target {
    MTL::Texture* tex = nullptr;
    const Fmt* fmt    = nullptr;
    const Res* res    = nullptr;
};

enum Draw { None = 0, Full = 1, Disc = 2 };

struct Pipes {
    MTL::RenderPipelineState* pso[3][4] = {}; // [Draw][format]
    MTL::DepthStencilState* depthState  = nullptr;
};

struct PassSpec {
    MTL::Texture* tex;
    const Fmt* fmt;
    MTL::LoadAction load;
    MTL::StoreAction store;
    Draw draw;
};

u32 fmtIndex(const Fmt* f) { return u32(f - kFormats); }

void encodePass(const Pipes& p, MTL4::CommandBuffer* cmd, const PassSpec& s) {
    MTL4::RenderPassDescriptor* pd = MTL4::RenderPassDescriptor::alloc()->init();
    if (s.fmt->depth) {
        auto* d = MTL::RenderPassDepthAttachmentDescriptor::alloc()->init();
        d->setTexture(s.tex);
        d->setLoadAction(s.load);
        d->setStoreAction(s.store);
        d->setClearDepth(0.5);
        pd->setDepthAttachment(d);
        d->release();
    } else {
        auto* c = pd->colorAttachments()->object(0);
        c->setTexture(s.tex);
        c->setLoadAction(s.load);
        c->setStoreAction(s.store);
        c->setClearColor(MTL::ClearColor::Make(0.25, 0.5, 0.75, 1.0));
    }
    MTL4::RenderCommandEncoder* re = cmd->renderCommandEncoder(pd);
    pd->release();
    if (s.draw != None) {
        re->setRenderPipelineState(p.pso[s.draw][fmtIndex(s.fmt)]);
        if (s.fmt->depth) re->setDepthStencilState(p.depthState);
        re->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(3));
    }
    re->endEncoding();
}

// GPU ms of one pass (the span minus the empty span).
double timePass(Context& ctx, const Pipes& p, const PassSpec& s) {
    CommandTimer t(ctx);
    MTL4::CommandBuffer* cmd = t.begin();
    encodePass(p, cmd, s);
    return t.finish() - ctx.emptySpanMs();
}

// The integer the shader writes into component c of pixel (x, y); `off` = 16 for the "_disc" draw.
u32 expectedComponent(const Fmt& f, u32 x, u32 y, u32 c, u32 off) {
    if (f.depth) return hash(x, y, off) & 0xFFFFFFu;
    return hash(x, y, off + c) & (f.pf == MTL::PixelFormatRGBA8Unorm ? 255u : 1023u);
}

// Clear + Full draw + store, then Load + Disc draw + store (the discarded pixels
// must come from the load); blit a corner and a row end back and compare with the
// CPU model.  Returns the number of wrong components.
u32 verifyTarget(Context& ctx, const Pipes& p, const Target& t) {
    const Fmt& f = *t.fmt;
    const u32 w = t.res->w, h = t.res->h;
    MTL::Buffer* rb = ctx.buffer(size_t(64) * 64 * f.bpp + size_t(64) * f.bpp);
    std::memset(rb->contents(), 0, rb->length());
    MTL4::CommandBuffer* cmd = ctx.beginCommands();
    encodePass(p, cmd, {t.tex, t.fmt, MTL::LoadActionClear, MTL::StoreActionStore, Full});
    encodePass(p, cmd, {t.tex, t.fmt, MTL::LoadActionLoad, MTL::StoreActionStore, Disc});
    MTL4::ComputeCommandEncoder* ce = cmd->computeCommandEncoder();
    ce->barrierAfterQueueStages(MTL::StageFragment, MTL::StageBlit, MTL4::VisibilityOptionDevice);
    const size_t half = size_t(64) * 64 * f.bpp;
    ce->copyFromTexture(t.tex, 0, 0, MTL::Origin::Make(0, 0, 0), MTL::Size::Make(64, 64, 1), rb, 0, 64 * f.bpp,
                        64 * 64 * f.bpp);
    ce->copyFromTexture(t.tex, 0, 0, MTL::Origin::Make(w - 64, h - 1, 0), MTL::Size::Make(64, 1, 1), rb, half,
                        64 * f.bpp, 64 * f.bpp);
    ce->endEncoding();
    ctx.submit();
    u32 wrong = 0;
    const u8* base = static_cast<const u8*>(rb->contents());
    auto check = [&](const u8* row, u32 x, u32 y, u32 xi) {
        const bool skipped = ((x | y) & 1u) == 0; // discarded by the second draw: still the first value
        for (u32 c = 0; c < (f.depth ? 1u : 4u); ++c) {
            u32 got = 0;
            const u32 want = expectedComponent(f, x, y, c, skipped ? 0 : 16);
            if (f.pf == MTL::PixelFormatRGBA8Unorm) {
                got = row[xi * 4 + c];
            } else if (f.pf == MTL::PixelFormatRGBA16Float) {
                _Float16 v;
                std::memcpy(&v, row + xi * 8 + c * 2, 2);
                got = u32(std::lround(double(v) * 1024.0));
            } else if (f.pf == MTL::PixelFormatRGBA32Float) {
                float v;
                std::memcpy(&v, row + xi * 16 + c * 4, 4);
                got = u32(std::lround(double(v) * 1024.0));
            } else {
                float v;
                std::memcpy(&v, row + xi * 4, 4);
                got = u32(std::lround(double(v) * 16777216.0));
            }
            if (got != want) ++wrong;
        }
    };
    for (u32 y = 0; y < 64; ++y)
        for (u32 x = 0; x < 64; ++x) check(base + size_t(y) * 64 * f.bpp, x, y, x);
    for (u32 x = 0; x < 64; ++x) check(base + half, w - 64 + x, h - 1, x);
    return wrong;
}

struct Cell {
    double a = 0, b = 0, e = 0, c = 0, m = -1; // A clear/dontCare, B clear/store, E disc clear/store, C disc load/store, M memoryless
    double bytes = 0;
};

std::string num(double v, size_t n = 6) { return std::to_string(v).substr(0, n); }

// Least-squares slope of y over x (ms per byte).
double slope(const std::vector<double>& x, const std::vector<double>& y, double* r2 = nullptr) {
    const double n = double(x.size());
    double sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        sx += x[i];
        sy += y[i];
        sxx += x[i] * x[i];
        sxy += x[i] * y[i];
        syy += y[i] * y[i];
    }
    const double den = n * sxx - sx * sx;
    const double b   = den != 0 ? (n * sxy - sx * sy) / den : 0;
    if (r2) {
        const double vy = n * syy - sy * sy;
        *r2             = (den != 0 && vy != 0) ? (n * sxy - sx * sy) * (n * sxy - sx * sy) / (den * vy) : 0;
    }
    return b;
}

void benchLoadStore(Context& ctx, Report& rep) {
    MTL::Library* lib = ctx.library("b14_load_store.metal");
    Pipes pipes;
    const char* fragNames[2][4] = {{"b14_frag_rgba8_full", "b14_frag_float_full", "b14_frag_float_full", "b14_frag_depth_full"},
                                   {"b14_frag_rgba8_disc", "b14_frag_float_disc", "b14_frag_float_disc", "b14_frag_depth_disc"}};
    for (u32 mode = 0; mode < 2; ++mode) {
        for (u32 i = 0; i < 4; ++i) {
            const Fmt& f = kFormats[i];
            MTL4::RenderPipelineDescriptor* d = MTL4::RenderPipelineDescriptor::alloc()->init();
            d->setVertexFunctionDescriptor(ctx.function(lib, "b14_vertex"));
            d->setFragmentFunctionDescriptor(ctx.function(lib, fragNames[mode][i]));
            if (!f.depth) d->colorAttachments()->object(0)->setPixelFormat(f.pf);
            pipes.pso[mode + 1][i] = ctx.render(d);
            d->release();
        }
    }
    {
        MTL::DepthStencilDescriptor* dd = MTL::DepthStencilDescriptor::alloc()->init();
        dd->setDepthCompareFunction(MTL::CompareFunctionAlways);
        dd->setDepthWriteEnabled(true);
        pipes.depthState = ctx.device()->newDepthStencilState(dd);
        dd->release();
        ctx.keep(pipes.depthState);
    }
    auto makeTex = [&](const Fmt& f, const Res& r, bool memoryless) {
        MTL::TextureDescriptor* td = MTL::TextureDescriptor::texture2DDescriptor(f.pf, r.w, r.h, false);
        td->setUsage(MTL::TextureUsageRenderTarget);
        td->setStorageMode(memoryless ? MTL::StorageModeMemoryless : MTL::StorageModePrivate);
        MTL::Texture* t = nullptr;
        if (memoryless) {
            t = ctx.device()->newTexture(td); // no backing store: nothing to make resident
            if (t) ctx.keep(t);
        } else {
            t = ctx.texture(td);
        }
        if (!t) throw BenchError(std::string("texture ") + f.name + " " + r.name);
        return t;
    };

    const u32 reps   = ctx.quick() ? 9 : 21;
    const u32 rounds = ctx.quick() ? 1 : 3;
    std::map<std::string, Cell> cells; // "fmt.res"
    bool verified = true;
    u32 wrongTotal = 0;
    const double emptyUs = ctx.emptySpanMs() * 1000.0;
    double emptyPassMs = 0;

    // --- Empty pass ---------------------------------------------------------------
    {
        MTL::TextureDescriptor* td = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatRGBA8Unorm, 64, 64, false);
        td->setUsage(MTL::TextureUsageRenderTarget);
        td->setStorageMode(MTL::StorageModePrivate);
        MTL::Texture* tiny = ctx.texture(td);
        // One tiny draw (4096 pixels): the validation layer flags a pass with no side effect at all
        // ("endEncoding called for an encoder with no side effects").
        const PassSpec s{tiny, &kFormats[0], MTL::LoadActionClear, MTL::StoreActionDontCare, Full};
        ctx.keepWarm(50);
        // Raw spans of the pass and of an empty command buffer, measured back to back
        // (the calibrated empty span drifts by a few us with the GPU clock).
        const u32 nSingle = ctx.quick() ? 31 : 101;
        const Stats raw = ctx.measure(
            [&] {
                CommandTimer t(ctx);
                encodePass(pipes, t.begin(), s);
                return t.finish() * 1000.0;
            },
            nSingle);
        const Stats emptyRaw = ctx.measure(
            [&] {
                CommandTimer t(ctx);
                t.begin();
                return t.finish() * 1000.0;
            },
            nSingle);
        Stats single = raw;
        single.median -= emptyRaw.median;
        single.min -= emptyRaw.median;
        single.max -= emptyRaw.median;
        single.p10 -= emptyRaw.median;
        single.p90 -= emptyRaw.median;
        single.mean -= emptyRaw.median;
        // Rates below subtract the fast mode (the cells are minimum-based).
        emptyPassMs = std::max(0.0, single.min) * 1e-3;
        rep.metric("empty_pass.us", "us", single, {{"w", 64}, {"h", 64}, {"empty_span_us", emptyUs}}, false);
        rep.value("empty_pass.fast_mode.us", "us", std::max(0.0, single.min), {{"w", 64}, {"h", 64}}, false);
        constexpr u32 kChain = 32;
        const Stats chain = ctx.measure(
            [&] {
                CommandTimer t(ctx);
                MTL4::CommandBuffer* cmd = t.begin();
                for (u32 i = 0; i < kChain; ++i) encodePass(pipes, cmd, s);
                return (t.finish() - ctx.emptySpanMs()) * 1000.0 / kChain;
            },
            ctx.quick() ? 15 : 31);
        rep.metric("empty_pass.chain32.us", "us", chain, {{"passes", double(kChain)}}, false);
        // A 64x64 pass must cost neither nothing (dropped pass / empty span dominates)
        // nor milliseconds.
        rep.negative(raw.median > 0.9 * emptyRaw.median && single.median < 500.0 && chain.median > 0.3,
                     "empty_pass raw " + num(raw.median) + " us vs empty span " + num(emptyRaw.median, 5) + " us (must be >= 0.9x), net " + num(single.median) + " us alone, " +
                         num(chain.median) + " us chained");
    }

    // --- Format x resolution ------------------------------------------------------------
    for (u32 fi = 0; fi < 4; ++fi) {
        const Fmt& f = kFormats[fi];
        for (const Res& r : kResolutions) {
            const Target t{makeTex(f, r, false), &f, &r};
            const double bytes = double(r.w) * r.h * f.bpp;
            auto spec = [&](MTL::LoadAction l, MTL::StoreAction s, Draw d) { return PassSpec{t.tex, &f, l, s, d}; };
            // First touch allocates and fills the texture (the load variants read real data).
            timePass(ctx, pipes, spec(MTL::LoadActionClear, MTL::StoreActionStore, Full));
            const std::string key = std::string(f.name) + "." + r.name;
            const PassSpec sa = spec(MTL::LoadActionClear, MTL::StoreActionDontCare, Full);
            const PassSpec sb = spec(MTL::LoadActionClear, MTL::StoreActionStore, Full);
            const PassSpec se = spec(MTL::LoadActionClear, MTL::StoreActionStore, Disc);
            const PassSpec sc = spec(MTL::LoadActionLoad, MTL::StoreActionStore, Disc);
            const PassSpec sd = spec(MTL::LoadActionLoad, MTL::StoreActionDontCare, Disc);
            // Interleave the variants in rounds so that clock drift hits all of them.
            std::vector<double> va, vb, ve, vc, vd;
            // The first render pass after keepWarm's compute load pays a
            // transition (measured: rgba8 1080p dontCare 0.065 ms vs store
            // 0.035 ms when dontCare always came first): one throwaway pass,
            // and the variant order rotates every round.
            const PassSpec* specs[5] = {&sa, &sb, &se, &sc, &sd};
            std::vector<double>* outs[5] = {&va, &vb, &ve, &vc, &vd};
            for (u32 round = 0; round < rounds; ++round) {
                ctx.keepWarm(15);
                timePass(ctx, pipes, sb);
                for (u32 k = 0; k < 5; ++k) {
                    const u32 v = (round + k) % 5;
                    // Minimum per round: a small pass costs either ~15 or ~60 us
                    // (bimodal, measured), and medians of different variants
                    // could land in different modes.
                    outs[v]->push_back(ctx.measure([&] { return timePass(ctx, pipes, *specs[v]); }, reps).min);
                }
            }
            const Stats sA = phosphor::soc::computeStats(va), sB = phosphor::soc::computeStats(vb),
                        sE = phosphor::soc::computeStats(ve), sC = phosphor::soc::computeStats(vc),
                        sD = phosphor::soc::computeStats(vd);
            Cell cell{sA.median, sB.median, sE.median, sC.median, -1, bytes};
            const std::map<std::string, double> params = {
                {"w", double(r.w)}, {"h", double(r.h)}, {"bpp", double(f.bpp)}, {"bytes", bytes}};
            rep.metric("dontcare." + key + ".ms", "ms", sA, params, false);
            rep.metric("store." + key + ".ms", "ms", sB, params, false);
            rep.metric("store_disc." + key + ".ms", "ms", sE, params, false);
            rep.metric("load." + key + ".ms", "ms", sC, params, false);
            rep.metric("load_dontcare." + key + ".ms", "ms", sD, params, false);
            // Effective rates.  store: bytes / (pass time - empty pass), i.e. the throughput of
            // the whole pass beyond its fixed cost; the store is overlapped with the shading of
            // later tiles, so where it is hidden this is the shading-bound rate (see the *_added
            // metrics for the time it adds over the same pass with dontCare / with clear).
            rep.value("store." + key, "GB/s", bytes / (std::max(1e-6, sB.median - emptyPassMs) * 1e-3) * 1e-9, params);
            rep.value("load." + key, "GB/s", bytes / (std::max(1e-6, sC.median - emptyPassMs) * 1e-3) * 1e-9, params);
            rep.value("store_added." + key + ".ms", "ms", sB.median - sA.median, params, false);
            rep.value("load_added." + key + ".ms", "ms", sC.median - sE.median, params, false);
            if (!f.depth) {
                MTL::Texture* m = makeTex(f, r, true);
                const PassSpec sm{m, &f, MTL::LoadActionClear, MTL::StoreActionDontCare, Full};
                ctx.keepWarm(15);
                const Stats sM = ctx.measure([&] { return timePass(ctx, pipes, sm); }, reps);
                cell.m = sM.min; // same (fast) mode as the cells above
                rep.metric("memoryless." + key + ".ms", "ms", sM, params, false);
            }
            cells[key] = cell;
            // Verify on the smallest resolution (a 64x64 corner + a row end).
            if (&r == &kResolutions[0]) {
                const u32 wrong = verifyTarget(ctx, pipes, t);
                wrongTotal += wrong;
                verified &= wrong == 0;
            }
            ctx.log("B-14 %s %s: A dontCare %.3f B store %.3f | E disc+store %.3f C load+disc+store %.3f ms | memoryless %.3f",
                    f.name, r.name, cell.a, cell.b, cell.e, cell.c, cell.m);
        }
    }
    // Compressibility (information): clear-only pass (constant colour compresses) at the smallest resolution.
    {
        const Res& r = kResolutions[0];
        for (u32 fi = 0; fi < 3; ++fi) {
            const Fmt& f = kFormats[fi];
            MTL::Texture* tex = makeTex(f, r, false);
            const PassSpec sa{tex, &f, MTL::LoadActionClear, MTL::StoreActionStore, None};
            timePass(ctx, pipes, sa);
            ctx.keepWarm(20);
            const Stats s = ctx.measure([&] { return timePass(ctx, pipes, sa); }, reps);
            rep.metric(std::string("clear_only_store.") + f.name + "." + r.name + ".ms", "ms", s,
                       {{"w", double(r.w)}, {"h", double(r.h)}, {"bpp", double(f.bpp)}}, false);
        }
    }

    // Slope of the pass time over the bytes, per format (least squares over the resolutions):
    // 1 / (slope_store - slope_dontcare) is the bandwidth the store needs once it is no longer
    // hidden behind the shading.
    std::map<std::string, double> fitGbps;
    std::string fitText;
    for (u32 fi = 0; fi < 4; ++fi) {
        std::vector<double> xs, ya, yb, ye, yc;
        for (const Res& r : kResolutions) {
            const Cell& c = cells[std::string(kFormats[fi].name) + "." + r.name];
            xs.push_back(c.bytes * 1e-6); // MB
            ya.push_back(c.a);
            yb.push_back(c.b);
            ye.push_back(c.e);
            yc.push_back(c.c);
        }
        double r2 = 0;
        const double sA = slope(xs, ya), sB = slope(xs, yb, &r2), sE = slope(xs, ye), sC = slope(xs, yc);
        const std::string fn = kFormats[fi].name;
        const double dStore = sB - sA, dLoad = sC - sE; // ms per MB
        rep.value("store_slope." + fn + ".ms_per_mb", "ms/MB", sB, {{"r2", r2}}, false);
        rep.value("dontcare_slope." + fn + ".ms_per_mb", "ms/MB", sA, {}, false);
        if (dStore > 1e-6) {
            rep.value("store_fit." + fn, "GB/s", 1.0 / dStore, {{"r2", r2}});
            fitGbps[fn] = 1.0 / dStore;
        }
        if (dLoad > 1e-6) rep.value("load_fit." + fn, "GB/s", 1.0 / dLoad, {});
        fitText += fn + " " + (dStore > 1e-6 ? num(1.0 / dStore, 5) : std::string("hidden")) + " GB/s (r2 " + num(r2, 4) + ") ";
    }

    // --- Negative controls --------------------------------------------------------------
    const std::string big   = kResolutions[3].name;
    const std::string small = kResolutions[0].name;
    // 1. dontCare < store where the store is large (rgba32f at the largest resolution); never
    //    cheaper than dontCare elsewhere (5% for noise).
    {
        const Cell& c = cells[std::string("rgba32f.") + big];
        // Only where the store must cost visibly (>= 48 MiB written, >= ~0.05 ms at 1 TB/s): in the
        // small cells the store is hidden and the bimodal pass cost can stay in its slow mode for a
        // whole cell (measured: depth32f 1080p dontCare 0.059 vs store 0.016 ms in one of 3 runs).
        bool never = true;
        u32 checked = 0;
        std::string bad;
        for (auto& [k, v] : cells) {
            if (v.bytes < 48.0 * 1048576.0) continue;
            ++checked;
            if (!(v.b > v.a * 0.95 - 0.03)) { never = false; bad += k + " "; } // 30 us of noise (other GPU clients)
        }
        rep.negative(c.b > c.a * 1.5 && never && checked >= 3,
                     "rgba32f " + big + ": dontCare " + num(c.a, 5) + " ms < store " + num(c.b, 5) + " ms (>1.5x needed), store never < 0.95 x dontCare in the " +
                         std::to_string(checked) + " cells >= 48 MiB: " + (never ? "yes" : "NO (" + bad + ")"));
    }
    // 2. Store time follows the bytes: for rgba32f the pass time is linear in bytes over the four
    //    resolutions (r2 > 0.98) and the largest texture is > 2x slower than the smallest;
    //    rgba32f (16 B/px) is > 2x slower than rgba8 (4 B/px) at the largest resolution.
    {
        std::vector<double> xs, yb;
        for (const Res& r : kResolutions) {
            const Cell& c = cells[std::string("rgba32f.") + r.name];
            xs.push_back(c.bytes);
            yb.push_back(c.b);
        }
        double r2 = 0;
        slope(xs, yb, &r2);
        const double grow = cells[std::string("rgba32f.") + big].b / cells[std::string("rgba32f.") + small].b;
        const double fmtR = cells[std::string("rgba32f.") + big].b / cells[std::string("rgba8.") + big].b;
        rep.negative(r2 > 0.95 && grow > 2.0 && fmtR > 2.0,
                     "rgba32f store time vs bytes: r2 " + num(r2, 5) + " (>0.95), " + big + "/" + small + " = " + num(grow, 4) +
                         "x (>2, bytes 4x), rgba32f/rgba8 at " + big + " = " + num(fmtR, 4) + "x (>2, bytes 4x)");
    }
    // 3. The stored/loaded data is right (CPU model of the shader hash; discarded pixels keep the loaded value).
    rep.negative(verified, verified ? "read-back after clear+draw+store then load+discard-draw+store matches the CPU model (4 formats)"
                                    : std::to_string(wrongTotal) + " wrong components in the read-back");
    if (!verified) rep.status(Status::Failed, "read-back differs from the CPU model");
    {
        u32 n = 0, slower = 0;
        for (u32 fi = 0; fi < 3; ++fi) {
            const Cell& c = cells[std::string(kFormats[fi].name) + "." + big];
            ++n;
            slower += c.m > c.a * 1.2;
        }
        if (slower)
            rep.note(std::to_string(slower) + "/" + std::to_string(n) + " memoryless passes >20% slower than private dontCare at " + big);
    }
    rep.note("fit over 4 resolutions of the added time per MB: " + fitText);
    rep.note("a small render pass between compute encoders costs either ~15 or ~60 us (bimodal: see empty_pass.us min vs "
             "median); cells = median over rounds of the per-round MINIMUM, so every variant is compared in the fast mode");
    rep.note("pass = clear|load + full-screen triangle writing an incompressible per-pixel hash + store|dontCare (A dontcare, B store, "
             "E/C: same with a discard-1-pixel-in-4 draw, clear/load); store.<k> = bytes/(B - empty pass), load.<k> = bytes/(C - empty pass); "
             "load_added = C - E, store_added = B - A; span minus empty span");
    rep.note("empty_pass.us = 64x64 RGBA8 clear+dontCare with one full-screen triangle of 4096 pixels (a pass with no side effect is flagged by the validation layer), minus the empty CommandTimer span; chain32 = 32 such passes in one command buffer / 32");
}

} // namespace

SOC_BENCH("B-14", "load_store", "Load/store cost per format and resolution, empty render pass, memoryless", benchLoadStore);

} // namespace soc
