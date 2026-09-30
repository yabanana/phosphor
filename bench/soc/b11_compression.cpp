// B-11: lossless texture compression cost -- writing (compute shader, full
// and partial block writes; render target) and reading a 4096x4096 RGBA8
// texture with allowGPUOptimizedContents true ("optimized") vs false
// ("plain"), for constant / smooth (compressible) and random (incompressible)
// content.
//
// Serves S-TEX-2 (M5 "universal compression": shader-written textures) of
// docs/APPLE_SOC_PLAYBOOK.md.
//
// The compression RATIO is NOT measurable: Metal exposes no compression
// counters to a program (F4.4).  Only bandwidth deltas between the two modes
// are reported, as indirect evidence; the benchmark is always Status::Partial.
// On Apple9 (or --force-family apple9) shader-written textures are not
// compressed: the same measurements run and "optimized" is expected equal to
// "plain" (render targets are compressed on every Apple GPU).

#include "harness.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

namespace soc {
namespace {

struct WriteParams {
    u32 width, height, pass, kind, partial, pad0, pad1, pad2;
};
static_assert(sizeof(WriteParams) == 32);

constexpr u32 kSize = 4096;
constexpr u32 kPasses = 4;                  // writes / reads per timed span
constexpr size_t kSlot = 256;               // params stride
constexpr double kTexBytes = double(kSize) * kSize * 4;
const char* kKinds[3] = {"constant", "smooth", "random"};

Stats toRate(const Stats& s, double scale) {
    Stats r = s;
    r.median = scale / s.median;
    r.min    = scale / s.max;
    r.max    = scale / s.min;
    r.p10    = scale / s.p90;
    r.p90    = scale / s.p10;
    r.mean   = scale / s.mean;
    return r;
}

// CPU replay of the shader's integer content function.
u32 hash3(u32 x, u32 y, u32 p) {
    u32 h = x * 0x9E3779B1u ^ (y + 0x7F4A7C15u) * 0x85EBCA6Bu ^ (p + 1u) * 0xC2B2AE35u;
    h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16;
    return h;
}
void content(u32 kind, u32 x, u32 y, u32 p, u32 c[4]) {
    c[3] = 255;
    if (kind == 0) {
        c[0] = (p * 37 + 11) & 255; c[1] = (p * 91 + 5) & 255; c[2] = (p * 53 + 201) & 255;
    } else if (kind == 1) {
        c[0] = ((x >> 4) + p) & 255; c[1] = ((y >> 4) + p * 3) & 255; c[2] = (((x + y) >> 5) + p * 5) & 255;
    } else {
        const u32 h = hash3(x, y, p);
        c[0] = h & 255; c[1] = (h >> 8) & 255; c[2] = (h >> 16) & 255; c[3] = (h >> 24) & 255;
    }
}
bool maskedOut(bool partial, u32 x, u32 y) { return partial && (((x >> 1) + (y >> 1)) & 1) != 0; }

struct Mode {
    const char* name;
    bool optimized;
    MTL::Texture* tex = nullptr;
    MTL4::RenderPassDescriptor* pass = nullptr;
};

void benchCompression(Context& ctx, Report& rep) {
    MTL::Library* lib = ctx.library("b11_compression.metal", /*fastMath=*/true);
    MTL::ComputePipelineState* csWrite = ctx.compute(lib, "b11_cs_write");
    MTL::ComputePipelineState* csRead  = ctx.compute(lib, "b11_read");
    MTL::Buffer* params = ctx.buffer(kSlot * 16);
    MTL::Buffer* out    = ctx.buffer(size_t(kSize / 4) * (kSize / 4) * 4);
    MTL::Buffer* rb     = ctx.buffer(size_t(kTexBytes));
    const float* out32  = static_cast<const float*>(out->contents());
    const u8* rb8       = static_cast<const u8*>(rb->contents());

    // Render pipeline (fullscreen triangle into RGBA8Unorm).
    MTL4::RenderPipelineDescriptor* rd = MTL4::RenderPipelineDescriptor::alloc()->init();
    rd->setVertexFunctionDescriptor(ctx.function(lib, "b11_vs"));
    rd->setFragmentFunctionDescriptor(ctx.function(lib, "b11_fs"));
    {
        MTL4::RenderPipelineColorAttachmentDescriptor* c = rd->colorAttachments()->object(0);
        c->setPixelFormat(MTL::PixelFormatRGBA8Unorm);
        c->setBlendingState(MTL4::BlendStateDisabled);
        c->setWriteMask(MTL::ColorWriteMaskAll);
        c->setRgbBlendOperation(MTL::BlendOperationAdd);
        c->setAlphaBlendOperation(MTL::BlendOperationAdd);
        c->setSourceRGBBlendFactor(MTL::BlendFactorOne);
        c->setDestinationRGBBlendFactor(MTL::BlendFactorZero);
        c->setSourceAlphaBlendFactor(MTL::BlendFactorOne);
        c->setDestinationAlphaBlendFactor(MTL::BlendFactorZero);
    }
    MTL::RenderPipelineState* rps = ctx.render(rd);
    rd->release();

    Mode modes[2] = {{"optimized", true}, {"plain", false}};
    for (Mode& m : modes) {
        MTL::TextureDescriptor* d = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatRGBA8Unorm, kSize, kSize, false);
        d->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite | MTL::TextureUsageRenderTarget);
        d->setStorageMode(MTL::StorageModePrivate);
        d->setAllowGPUOptimizedContents(m.optimized);
        m.tex = ctx.texture(d);
        m.pass = MTL4::RenderPassDescriptor::alloc()->init();
        MTL::RenderPassColorAttachmentDescriptor* c = m.pass->colorAttachments()->object(0);
        c->setTexture(m.tex);
        c->setLoadAction(MTL::LoadActionDontCare);
        c->setStoreAction(MTL::StoreActionStore);
        ctx.keep(m.pass);
    }

    bool resultsOk = true;
    std::string wrong;
    auto bad = [&](const std::string& s) { resultsOk = false; wrong += s + " "; };

    auto setParams = [&](u32 slot, u32 pass, u32 kind, u32 partial) {
        WriteParams p{kSize, kSize, pass, kind, partial, 0, 0, 0};
        std::memcpy(static_cast<u8*>(params->contents()) + slot * kSlot, &p, sizeof(p));
    };
    // Compute writes: n dispatches (passes firstPass .. firstPass+n-1), laps summed (ms).
    auto csWriteMs = [&](Mode& m, u32 kind, bool partial, u32 firstPass, u32 n) {
        for (u32 i = 0; i < n; ++i) setParams(i, firstPass + i, kind, partial);
        ComputeTimer t(ctx);
        MTL4::ComputeCommandEncoder* e = t.begin();
        for (u32 i = 0; i < n; ++i) {
            ctx.table()->setAddress(params->gpuAddress() + i * kSlot, 0);
            ctx.table()->setTexture(m.tex->gpuResourceID(), 0);
            e->setComputePipelineState(csWrite);
            e->setArgumentTable(ctx.table());
            e->dispatchThreads(MTL::Size::Make(kSize, kSize, 1), MTL::Size::Make(16, 16, 1));
            t.lap();
        }
        double s = 0;
        for (double v : t.finish()) s += v;
        return s;
    };
    auto rtWriteMs = [&](Mode& m, u32 kind, u32 firstPass, u32 n) {
        for (u32 i = 0; i < n; ++i) setParams(i, firstPass + i, kind, 0);
        CommandTimer t(ctx);
        MTL4::CommandBuffer* cmd = t.begin();
        for (u32 i = 0; i < n; ++i) {
            MTL4::RenderCommandEncoder* e = cmd->renderCommandEncoder(m.pass);
            if (i > 0) e->barrierAfterQueueStages(MTL::StageFragment, MTL::StageFragment, MTL4::VisibilityOptionDevice);
            e->setRenderPipelineState(rps);
            ctx.table()->setAddress(params->gpuAddress() + i * kSlot, 0);
            e->setArgumentTable(ctx.table(), MTL::RenderStageVertex | MTL::RenderStageFragment);
            e->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(3));
            e->endEncoding();
        }
        return std::max(1e-4, t.finish() - ctx.emptySpanMs());
    };
    auto readMs = [&](Mode& m, u32 n) {
        ComputeTimer t(ctx);
        MTL4::ComputeCommandEncoder* e = t.begin();
        for (u32 i = 0; i < n; ++i) {
            ctx.table()->setAddress(out->gpuAddress(), 1);
            ctx.table()->setTexture(m.tex->gpuResourceID(), 0);
            e->setComputePipelineState(csRead);
            e->setArgumentTable(ctx.table());
            e->dispatchThreads(MTL::Size::Make(kSize / 4, kSize / 4, 1), MTL::Size::Make(16, 16, 1));
            t.lap();
        }
        double s = 0;
        for (double v : t.finish()) s += v;
        return s;
    };
    // Texel readback through a blit (decompresses) and CPU check.
    auto verify = [&](Mode& m, u32 kind, bool partial, u32 passWritten, u32 passOther, const std::string& tag) {
        MTL4::CommandBuffer* cmd = ctx.beginCommands();
        MTL4::ComputeCommandEncoder* e = cmd->computeCommandEncoder();
        e->copyFromTexture(m.tex, 0, 0, MTL::Origin::Make(0, 0, 0), MTL::Size::Make(kSize, kSize, 1), rb, 0, kSize * 4,
                           size_t(kSize) * kSize * 4);
        e->endEncoding();
        ctx.submit();
        u64 s = 77;
        bool ok = true;
        for (int i = 0; i < 6000 && ok; ++i) {
            const u32 x = i == 0 ? 0 : i == 1 ? kSize - 1 : u32(xorshift64(s) % kSize);
            const u32 y = i == 0 ? 0 : i == 1 ? kSize - 1 : u32(xorshift64(s) % kSize);
            u32 c[4];
            content(kind, x, y, maskedOut(partial, x, y) ? passOther : passWritten, c);
            const u8* px = rb8 + (size_t(y) * kSize + x) * 4;
            for (int k = 0; k < 4; ++k)
                if (px[k] != c[k]) ok = false;
        }
        if (!ok) bad("texels " + tag);
    };
    auto verifyRead = [&](u32 kind, u32 pass, const std::string& tag) {
        for (u32 g : {0u, 1u, 513u, 1023u, 300u * 1024u + 17u, 1024u * 1024u - 1u}) {
            const u32 bx = g % (kSize / 4), by = g / (kSize / 4);
            double acc[4] = {0, 0, 0, 0};
            for (u32 j = 0; j < 4; ++j)
                for (u32 i = 0; i < 4; ++i) {
                    u32 c[4];
                    content(kind, bx * 4 + i, by * 4 + j, pass, c);
                    for (int k = 0; k < 4; ++k) acc[k] += c[k] / 255.0;
                }
            const double e = acc[0] + 2 * acc[1] + 3 * acc[2] + 5 * acc[3];
            if (std::fabs(out32[g] - e) > 1e-3 * std::max(1.0, std::fabs(e))) {
                bad("read " + tag);
                return;
            }
        }
    };

    std::map<std::string, double> gbs;
    double linRatio = 0, rtLinRatio = 0;
    for (Mode& m : modes) {
        for (u32 kind = 0; kind < 3; ++kind) {
            const std::string kn = kKinds[kind], tagBase = std::string(m.name) + "." + kn;
            const std::map<std::string, double> prm = {{"optimized", m.optimized ? 1.0 : 0.0}, {"kind", double(kind)},
                                                        {"size", double(kSize)}, {"passes", double(kPasses)}};
            // Known initial state: full write with pass 0.
            csWriteMs(m, kind, false, 0, 1);
            ctx.keepWarm(20);

            // Compute write, full blocks (passes 1..4).
            const u32 linReps = (!m.optimized && kind == 2) ? 25 : 0; // the linearity-control config: more repetitions
            Stats t = ctx.measure([&] { return csWriteMs(m, kind, false, 1, kPasses); }, linReps);
            verify(m, kind, false, kPasses, kPasses, "cs_full." + tagBase);
            Stats bw = toRate(t, kTexBytes * kPasses * 1e-6);
            rep.metric("write." + tagBase, "GB/s", bw, prm);
            gbs["write." + tagBase] = bw.median;
            if (m.optimized == false && kind == 2) {
                // Linearity control: 2x passes -> 2x time (random, plain).
                const Stats t2 = ctx.measure([&] { return csWriteMs(m, kind, false, 1, 2 * kPasses); }, linReps);
                linRatio = t2.min / t.min; // minima: contention only adds time
                verify(m, kind, false, 2 * kPasses, 2 * kPasses, "cs_full2x." + tagBase);
                csWriteMs(m, kind, false, 1, kPasses); // back to a known state (pass 4)
            }
            ctx.keepWarm(20);

            // Compute write, partial blocks (passes 5..8; the other cells keep pass 4).
            t = ctx.measure([&] { return csWriteMs(m, kind, true, 5, kPasses); });
            verify(m, kind, true, 4 + kPasses, kPasses, "cs_partial." + tagBase);
            bw = toRate(t, kTexBytes * 0.5 * kPasses * 1e-6);
            rep.metric("write_partial." + tagBase, "GB/s", bw, prm);
            gbs["write_partial." + tagBase] = bw.median;
            ctx.keepWarm(20);

            // Render target write (full-screen triangle, load DontCare, store).
            t = ctx.measure([&] { return rtWriteMs(m, kind, 1, kPasses); }, linReps);
            verify(m, kind, false, kPasses, kPasses, "rt." + tagBase);
            bw = toRate(t, kTexBytes * kPasses * 1e-6);
            rep.metric("rt_write." + tagBase, "GB/s", bw, prm);
            gbs["rt_write." + tagBase] = bw.median;
            if (m.optimized == false && kind == 2) {
                const Stats t2 = ctx.measure([&] { return rtWriteMs(m, kind, 1, 2 * kPasses); }, linReps);
                rtLinRatio = t2.min / t.min;
                rtWriteMs(m, kind, 1, kPasses); // back to a known state (pass 4)
            }
            ctx.keepWarm(20);

            // Read (texture holds pass 4 of this kind, written by the render pass above).
            t = ctx.measure([&] { return readMs(m, kPasses); });
            verifyRead(kind, kPasses, tagBase);
            bw = toRate(t, kTexBytes * kPasses * 1e-6);
            rep.metric("read." + tagBase, "GB/s", bw, prm);
            gbs["read." + tagBase] = bw.median;
            ctx.keepWarm(20);
        }
    }

    // Indirect evidence: optimized / plain bandwidth per test and content.
    std::string deltas;
    for (const char* test : {"write", "write_partial", "rt_write", "read"})
        for (const char* kn : kKinds) {
            const double o = gbs[std::string(test) + ".optimized." + kn], p = gbs[std::string(test) + ".plain." + kn];
            rep.value(std::string("opt_over_plain.") + test + "." + kn, "ratio", o / p, {{"size", double(kSize)}});
            if (std::string(kn) != "constant" || std::string(test) == "write")
                deltas += std::string(test) + "." + kn + "=" + std::to_string(o / p).substr(0, 4) + " ";
        }

    rep.negative(linRatio > 1.8 && linRatio < 2.2 && rtLinRatio > 1.7 && rtLinRatio < 2.3,
                 "2x passes -> time x" + std::to_string(linRatio).substr(0, 5) + " (compute write, plain random), x" +
                     std::to_string(rtLinRatio).substr(0, 5) + " (render target)");
    rep.negative(resultsOk, resultsOk ? "texels read back through a blit (6000 per config and write test) and 6 read-kernel sums per config match the CPU replay of the content function"
                                      : "MISMATCH: " + wrong);
    rep.status(Status::Partial, "compression ratio is not measurable (Metal exposes no compression counters, F4.4): only optimized/plain bandwidth deltas are reported as indirect evidence");
    if (!ctx.apple10())
        rep.note("Apple9 path: shader-written textures are not compressed there, 'optimized' expected equal to 'plain' for compute writes (render targets are compressed on every Apple GPU)");
    rep.note("optimized/plain ratios: " + deltas);
    rep.note("optimized = TextureDescriptor.allowGPUOptimizedContents true; plain = false; 4096x4096 RGBA8Unorm private, 4 passes per timed span, bandwidth counts logical bytes written/read (partial: 50%: 2x2 cells in a checkerboard, touching every 4x4 block)");
    rep.note("render target partial writes not measured: they need loadAction Load (a different experiment); random content = hashed texels, smooth = 16-texel-step gradients, constant = one colour per pass");
}

} // namespace

SOC_BENCH("B-11", "texture.compression", "Lossless compression: cost of writing/reading optimized vs plain textures (bandwidth deltas only)", benchCompression);

} // namespace soc
