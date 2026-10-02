// F6-S3: Hi-Z pyramid backends (docs/plans/F6.md, spike S3).  The ENGINE
// kernels of shaders/hiz.metal (compiled from source with their includes)
// build the conservative reverse-Z MIN pyramid of synthetic depth textures:
//
//   compute   hiz_level0 + one hiz_reduce dispatch per level (4 loads/texel)
//   simd      hiz_level0 + hiz_reduce_simd (5 levels per dispatch: SIMD
//             shuffles for the first, threadgroup memory for the next three)
//   sampler   hiz_level0 + one hiz_reduce_sampler dispatch per level: ONE
//             sample of a min-reduction sampler (Apple10; skipped with
//             --force-family apple9 or on Apple9)
//
// Every level of every backend is read back and compared BIT FOR BIT with a
// CPU reference (same rule as renderer/meshlet_cull_reference.cpp): level 0
// power-of-two >= ceil(size / 2), texel = min of the existing covered pixels
// (missing pixels count as 1), exact halving after.  Patterns: random depth,
// random with 1% zero holes, all zero, constant 0.8, gradient with ONE zero
// hole (the hole must survive at every level).  Sizes: 1x1, 1x17, 17x1,
// 63x65, 1920x1080, 1919x1081, 3200x1800 (NPOT edges, 1-wide levels).
// Timing (whole chain, CommandTimer span minus the empty span, median of the
// repetitions) at 1920x1080 and 3200x1800 on random depth.
//
// Negative control: a deliberately wrong reduction (bench-local
// s3_reduce_point: one texel instead of the 2x2 min) must mismatch on random
// depth, so a pass cannot come from a comparison that sees nothing.
// Metrics: mismatch.<backend> (texels differing, all patterns/sizes),
// ms.<backend>.<WxH>, levels.<WxH>.

#include "f6_common.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace soc {
namespace {

struct HiZParams { // == phosphor::GPUHiZParams
    u32 srcSize[2];
    u32 dstSize[2];
    u32 dstLevel;
    u32 levels;
    u32 pad[2];
};
static_assert(sizeof(HiZParams) == 32);

u32 level0Size(u32 v) {
    const u32 half = (v + 1) / 2;
    u32 s = 1;
    while (s < half) s <<= 1;
    return s;
}
u32 levelCount(u32 w0, u32 h0) {
    const u32 m = std::max(w0, h0);
    u32 l = 1;
    while ((1u << (l - 1)) < m) ++l;
    return l;
}

using Pyramid = std::vector<std::vector<float>>;

Pyramid cpuPyramid(const std::vector<float>& depth, u32 w, u32 h, u32 w0, u32 h0, u32 levels) {
    Pyramid p(levels);
    p[0].assign(size_t(w0) * h0, 1.0f);
    for (u32 y = 0; y < h0; ++y)
        for (u32 x = 0; x < w0; ++x) {
            float m = 1.0f;
            for (u32 dy = 0; dy < 2; ++dy)
                for (u32 dx = 0; dx < 2; ++dx) {
                    const u32 px = 2 * x + dx, py = 2 * y + dy;
                    if (px < w && py < h) m = std::min(m, depth[size_t(py) * w + px]);
                }
            p[0][size_t(y) * w0 + x] = m;
        }
    for (u32 l = 1; l < levels; ++l) {
        const u32 sw = std::max(w0 >> (l - 1), 1u), sh = std::max(h0 >> (l - 1), 1u);
        const u32 dw = std::max(w0 >> l, 1u), dh = std::max(h0 >> l, 1u);
        p[l].assign(size_t(dw) * dh, 1.0f);
        for (u32 y = 0; y < dh; ++y)
            for (u32 x = 0; x < dw; ++x) {
                float m = 1.0f;
                for (u32 dy = 0; dy < 2; ++dy)
                    for (u32 dx = 0; dx < 2; ++dx) {
                        const u32 sx = 2 * x + dx, sy = 2 * y + dy;
                        if (sx < sw && sy < sh) m = std::min(m, p[l - 1][size_t(sy) * sw + sx]);
                    }
                p[l][size_t(y) * dw + x] = m;
            }
    }
    return p;
}

enum class Backend { Compute, Simd, Sampler, Point };
const char* backendName(Backend b) {
    switch (b) {
    case Backend::Compute: return "compute";
    case Backend::Simd:    return "simd";
    case Backend::Sampler: return "sampler";
    case Backend::Point:   return "point_control";
    }
    return "?";
}

struct Rig {
    Context& ctx;
    MTL::Library* engine = nullptr;
    MTL::Library* local  = nullptr;
    explicit Rig(Context& c) : ctx(c) {}
};

struct Case {
    u32 w, h, w0, h0, levels;
    MTL::Texture* depth = nullptr;
    MTL::Texture* pyramid = nullptr;
    MTL::Buffer* params = nullptr; // HiZParams per dispatch (256-byte strides)
    std::vector<float> host;
};

Case makeCase(Rig& r, u32 w, u32 h) {
    Case c;
    c.w = w;
    c.h = h;
    c.w0 = level0Size(w);
    c.h0 = level0Size(h);
    c.levels = levelCount(c.w0, c.h0);
    MTL::TextureDescriptor* d = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatDepth32Float, w, h, false);
    d->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageRenderTarget);
    d->setStorageMode(MTL::StorageModePrivate);
    c.depth = r.ctx.texture(d);
    MTL::TextureDescriptor* p = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatR32Float, c.w0, c.h0, true);
    p->setMipmapLevelCount(c.levels);
    p->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
    p->setStorageMode(MTL::StorageModePrivate);
    c.pyramid = r.ctx.texture(p);
    c.params = r.ctx.buffer(256 * 16);
    return c;
}

void uploadDepth(Rig& r, Case& c, const std::vector<float>& depth) {
    c.host = depth;
    const size_t bytes = depth.size() * 4;
    MTL::Buffer* staging = r.ctx.buffer(bytes);
    std::memcpy(staging->contents(), depth.data(), bytes);
    MTL4::CommandBuffer* cmd = r.ctx.beginCommands();
    MTL4::ComputeCommandEncoder* ce = cmd->computeCommandEncoder();
    ce->copyFromBuffer(staging, 0, c.w * 4, bytes, MTL::Size::Make(c.w, c.h, 1), c.depth, 0, 0, MTL::Origin::Make(0, 0, 0));
    ce->endEncoding();
    r.ctx.submit();
}

/// Encode the whole chain of `b` into `ce` (dispatch barriers between levels).
void encodeChain(Rig& r, Case& c, Backend b, MTL4::ComputeCommandEncoder* ce) {
    MTL4::ArgumentTable* t = r.ctx.table();
    auto* params = static_cast<u8*>(c.params->contents());
    u32 slot = 0;
    auto put = [&](u32 sw, u32 sh, u32 dw, u32 dh, u32 level) {
        HiZParams hp{{sw, sh}, {dw, dh}, level, c.levels, {0, 0}};
        std::memcpy(params + slot * 256, &hp, sizeof(hp));
        t->setAddress(c.params->gpuAddress() + slot * 256, 0);
        ++slot;
    };
    const auto groups = [](u32 w, u32 h) { return MTL::Size::Make((w + 15) / 16, (h + 15) / 16, 1); };
    const MTL::Size tg = MTL::Size::Make(16, 16, 1);
    // level 0 from the depth
    put(c.w, c.h, c.w0, c.h0, 0);
    t->setTexture(c.depth->gpuResourceID(), 0);
    t->setTexture(c.pyramid->gpuResourceID(), 1);
    ce->setComputePipelineState(r.ctx.compute(r.engine, "hiz_level0"));
    ce->setArgumentTable(t);
    ce->dispatchThreadgroups(groups(c.w0, c.h0), tg);
    t->setTexture(c.pyramid->gpuResourceID(), 0);
    const char* fn = b == Backend::Compute ? "hiz_reduce" : b == Backend::Sampler ? "hiz_reduce_sampler" : "s3_reduce_point";
    MTL::ComputePipelineState* pso =
        b == Backend::Simd ? r.ctx.compute(r.engine, "hiz_reduce_simd")
                           : (b == Backend::Point ? r.ctx.compute(r.local, fn) : r.ctx.compute(r.engine, fn));
    const u32 step = b == Backend::Simd ? 5 : 1;
    if (c.levels > 1) ce->setComputePipelineState(pso); // once: validation rejects redundant state
    for (u32 l = 1; l < c.levels; l += step) {
        ce->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
        const u32 sw = std::max(c.w0 >> (l - 1), 1u), sh = std::max(c.h0 >> (l - 1), 1u);
        const u32 dw = std::max(c.w0 >> l, 1u), dh = std::max(c.h0 >> l, 1u);
        put(sw, sh, dw, dh, l);
        ce->setArgumentTable(t);
        ce->dispatchThreadgroups(groups(dw, dh), tg);
    }
}

/// Build with `b`, read every level back; returns the texels differing from the CPU pyramid.
u64 checkBackend(Rig& r, Case& c, Backend b, const Pyramid& ref, std::string& firstDiff) {
    std::vector<MTL::Buffer*> read(c.levels);
    for (u32 l = 0; l < c.levels; ++l) {
        const u32 dw = std::max(c.w0 >> l, 1u), dh = std::max(c.h0 >> l, 1u);
        read[l] = r.ctx.buffer(size_t(dw) * dh * 4);
        std::memset(read[l]->contents(), 0xFF, size_t(dw) * dh * 4); // NaN pattern: unwritten texels differ
    }
    MTL4::CommandBuffer* cmd = r.ctx.beginCommands();
    MTL4::ComputeCommandEncoder* ce = cmd->computeCommandEncoder();
    encodeChain(r, c, b, ce);
    ce->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageBlit, MTL4::VisibilityOptionDevice);
    for (u32 l = 0; l < c.levels; ++l) {
        const u32 dw = std::max(c.w0 >> l, 1u), dh = std::max(c.h0 >> l, 1u);
        ce->copyFromTexture(c.pyramid, 0, l, MTL::Origin::Make(0, 0, 0), MTL::Size::Make(dw, dh, 1), read[l], 0, dw * 4,
                            size_t(dw) * dh * 4);
    }
    ce->endEncoding();
    r.ctx.submit();
    u64 bad = 0;
    for (u32 l = 0; l < c.levels; ++l) {
        const auto* g = static_cast<const float*>(read[l]->contents());
        for (size_t i = 0; i < ref[l].size(); ++i) {
            if (std::memcmp(&g[i], &ref[l][i], 4) != 0) {
                if (bad == 0) {
                    const u32 dw = std::max(c.w0 >> l, 1u);
                    firstDiff = std::string(backendName(b)) + " " + std::to_string(c.w) + "x" + std::to_string(c.h) +
                                " level " + std::to_string(l) + " texel (" + std::to_string(i % dw) + "," +
                                std::to_string(i / dw) + "): gpu " + std::to_string(g[i]) + " cpu " + std::to_string(ref[l][i]);
                }
                ++bad;
            }
        }
    }
    return bad;
}

std::vector<float> pattern(int kind, u32 w, u32 h, u64 seed) {
    std::vector<float> d(size_t(w) * h);
    u64 s = seed;
    for (size_t i = 0; i < d.size(); ++i) {
        const float u = float(xorshift64(s) >> 40) / float(1u << 24);
        switch (kind) {
        case 0: d[i] = u; break;                                            // random
        case 1: d[i] = (xorshift64(s) % 100) == 0 ? 0.0f : 0.05f + u * 0.95f; break; // 1% holes
        case 2: d[i] = 0.0f; break;                                         // background only
        case 3: d[i] = 0.8f; break;                                         // full occluder
        default: d[i] = 0.3f + 0.6f * float((i % w) + (i / w)) / float(w + h); break; // gradient
        }
    }
    if (kind == 4) d[size_t(h / 2) * w + w / 2] = 0.0f; // one hole
    return d;
}
const char* patternName(int k) {
    static const char* n[] = {"random", "holes1pct", "zero", "const0.8", "gradient+hole"};
    return n[k];
}

void benchHiZ(Context& ctx, Report& rep) {
    Rig r(ctx);
    r.engine = f6::engineLibrary(ctx, "hiz.metal");
    r.local  = f6::f6Library(ctx, "s3_controls.metal");
    const bool sampler = ctx.apple10();
    std::vector<Backend> backends = {Backend::Compute, Backend::Simd};
    if (sampler) backends.push_back(Backend::Sampler);
    else rep.note("sampler backend SKIPPED: Apple10 sampler min reduction not available (Apple9 or --force-family apple9)");

    struct Size2 { u32 w, h; };
    std::vector<Size2> sizes = {{1, 1}, {1, 17}, {17, 1}, {63, 65}, {1920, 1080}, {1919, 1081}, {3200, 1800}};
    if (ctx.quick()) sizes = {{1, 1}, {17, 1}, {63, 65}, {1919, 1081}};
    std::vector<u64> mismatch(4, 0);
    std::string firstDiff, holeLost;
    u64 controlBad = 0;
    for (const Size2& sz : sizes) {
        Case c = makeCase(r, sz.w, sz.h);
        for (int k = 0; k < 5; ++k) {
            const std::vector<float> d = pattern(k, sz.w, sz.h, 0x5EED0000ull + sz.w * 131 + sz.h * 7 + k);
            uploadDepth(r, c, d);
            const Pyramid ref = cpuPyramid(d, c.w, c.h, c.w0, c.h0, c.levels);
            // The CPU reference itself must keep the hole at every level.
            if (k == 4) {
                const u32 hx = sz.w / 2, hy = sz.h / 2;
                for (u32 l = 0; l < c.levels; ++l) {
                    const u32 dw = std::max(c.w0 >> l, 1u);
                    if (ref[l][size_t(hy >> (l + 1)) * dw + (hx >> (l + 1))] != 0.0f) holeLost += std::to_string(sz.w) + "x" + std::to_string(sz.h) + "@" + std::to_string(l) + " ";
                }
            }
            for (Backend b : backends) {
                std::string diff;
                const u64 bad = checkBackend(r, c, b, ref, diff);
                mismatch[static_cast<int>(b)] += bad;
                if (bad && firstDiff.empty()) firstDiff = std::string(patternName(k)) + ": " + diff;
            }
            if (k == 0 && sz.w >= 17) {
                std::string diff;
                controlBad += checkBackend(r, c, Backend::Point, ref, diff);
            }
        }
        ctx.keepWarm(10);
    }
    for (Backend b : backends) rep.value(std::string("mismatch.") + backendName(b), "texels", double(mismatch[static_cast<int>(b)]), {}, false);
    rep.value("mismatch.point_control", "texels", double(controlBad), {}, true);

    // ---- timing ------------------------------------------------------------------------------------------
    ctx.warmUp(3.0);
    std::string timing;
    for (const Size2& sz : std::vector<Size2>{{1920, 1080}, {3200, 1800}}) {
        Case c = makeCase(r, sz.w, sz.h);
        uploadDepth(r, c, pattern(0, sz.w, sz.h, 99));
        rep.value("levels." + std::to_string(sz.w) + "x" + std::to_string(sz.h), "count", c.levels, {}, false);
        for (Backend b : backends) {
            const double empty = ctx.emptySpanMs();
            const Stats s = ctx.measure([&] {
                CommandTimer t(ctx);
                MTL4::CommandBuffer* cmd = t.begin();
                MTL4::ComputeCommandEncoder* ce = cmd->computeCommandEncoder();
                encodeChain(r, c, b, ce);
                ce->endEncoding();
                return std::max(t.finish() - empty, 0.0);
            });
            const std::string name = std::string("ms.") + backendName(b) + "." + std::to_string(sz.w) + "x" + std::to_string(sz.h);
            rep.metric(name, "ms", s, {{"width", double(sz.w)}, {"height", double(sz.h)}}, false);
            char buf[160];
            std::snprintf(buf, sizeof buf, "%s %ux%u: %.4f ms (min %.4f, p90 %.4f); ", backendName(b), sz.w, sz.h,
                          s.median, s.min, s.p90);
            timing += buf;
            ctx.keepWarm(20);
        }
    }
    rep.note("timing (chain, median): " + timing);
    if (!firstDiff.empty()) rep.note("first mismatch: " + firstDiff);
    if (!holeLost.empty()) rep.note("CPU reference lost the hole at: " + holeLost);
    bool exact = holeLost.empty();
    for (Backend b : backends) exact &= mismatch[static_cast<int>(b)] == 0;
    const bool controlOk = controlBad > 0;
    rep.negative(exact && controlOk,
                 std::string(exact ? "every backend bit-exact against the CPU pyramid on every pattern/size" : "MISMATCH") +
                     "; point-sampling control " + (controlOk ? "detected (" + std::to_string(controlBad) + " texels)" : "NOT detected"));
    if (!exact) rep.status(Status::Failed, "a backend differs from the CPU pyramid: " + firstDiff);
}

} // namespace

SOC_BENCH("F6-S3", "hiz.backends", "Hi-Z pyramid: compute / SIMD / sampler-min backends, bit-exact vs CPU, chain time",
          benchHiZ);

} // namespace soc
