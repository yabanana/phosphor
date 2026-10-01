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
//
// OPT-1.3 spike extension: the same measurements for a texture placed in a
// placement heap ("heap.*", like the engine's TransientHeap) and for one
// aliased with a bigger RGBA16Float texture written with random content just
// before ("heap_alias.*"), plus a "pfview" case (optimized + PixelFormatView,
// documented to disable compression) per path as the negative control, the
// heapTextureSizeAndAlign sizes, and per-path gains (opt_over_plain.*,
// adv_gain.*, gain_geomean.*).  Existing metric names are unchanged.

#include "harness.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

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

// Where the measured texture lives: device->newTexture (the original B-11),
// a placement heap like the engine's TransientHeap, or the same heap at the
// same offset as a bigger RGBA16Float texture that was written with random
// content just before (the render graph's aliasing pattern).
enum class Path : u32 { Device = 0, Heap = 1, Alias = 2 };
const char* kPathPrefix[3] = {"", "heap.", "heap_alias."};

struct Mode {
    const char* name;
    bool optimized;
    bool pfview; // TextureUsagePixelFormatView added (documented to disable lossless compression)
    Path path;
    MTL::Texture* tex = nullptr;
    MTL::Texture* alias = nullptr; // Path::Alias: the other-format texture at the same offset
    MTL4::RenderPassDescriptor* pass = nullptr;
};

constexpr u64 kHeapOffsetUnits = 4; // the texture is placed at 4 alignment units (not offset 0)

MTL::TextureDescriptor* rgba8Desc(bool optimized, bool pfview, bool heap) {
    MTL::TextureDescriptor* d = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatRGBA8Unorm, kSize, kSize, false);
    d->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite | MTL::TextureUsageRenderTarget |
                (pfview ? MTL::TextureUsagePixelFormatView : 0));
    d->setStorageMode(MTL::StorageModePrivate);
    if (heap) d->setHazardTrackingMode(MTL::HazardTrackingModeUntracked); // like the engine's TransientHeap
    d->setAllowGPUOptimizedContents(optimized);
    return d;
}
MTL::TextureDescriptor* rgba16fDesc(bool optimized, bool pfview) {
    MTL::TextureDescriptor* d = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatRGBA16Float, kSize, kSize, false);
    d->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite | (pfview ? MTL::TextureUsagePixelFormatView : 0));
    d->setStorageMode(MTL::StorageModePrivate);
    d->setHazardTrackingMode(MTL::HazardTrackingModeUntracked);
    d->setAllowGPUOptimizedContents(optimized);
    return d;
}

void benchCompression(Context& ctx, Report& rep) {
    MTL::Library* lib = ctx.library("b11_compression.metal", /*fastMath=*/true);
    MTL::ComputePipelineState* csWrite = ctx.compute(lib, "b11_cs_write");
    MTL::ComputePipelineState* csRead  = ctx.compute(lib, "b11_read");
    MTL::Buffer* params = ctx.buffer(kSlot * 16);
    MTL::Buffer* out    = ctx.buffer(size_t(kSize / 4) * (kSize / 4) * 4);
    MTL::Buffer* rb     = ctx.buffer(size_t(kTexBytes));
    const float* out32  = static_cast<const float*>(out->contents());
    const u8* rb8       = static_cast<const u8*>(rb->contents());
    MTL::Buffer* rb16   = ctx.buffer(size_t(kSize) * kSize * 8); // readback of the RGBA16F alias texture
    const _Float16* rb16h = static_cast<const _Float16*>(rb16->contents());
    MTL::ComputePipelineState* csWrite16 = ctx.compute(lib, "b11_cs_write16");

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

    std::vector<Mode> modes;
    for (Path path : {Path::Device, Path::Heap, Path::Alias}) {
        modes.push_back({"optimized", true, false, path});
        modes.push_back({"plain", false, false, path});
        modes.push_back({"pfview", true, true, path}); // optimized + PixelFormatView: the negative control
    }
    // Sizes the heap reports per descriptor: a difference is evidence of compression metadata.
    {
        struct V { const char* n; bool opt, pf; };
        for (const V v : {V{"optimized", true, false}, V{"plain", false, false}, V{"optimized_pfview", true, true}, V{"plain_pfview", false, true}}) {
            MTL::TextureDescriptor* d = rgba8Desc(v.opt, v.pf, true);
            const MTL::SizeAndAlign sa = ctx.device()->heapTextureSizeAndAlign(d);
            rep.value(std::string("heap_size.rgba8.") + v.n, "B", double(sa.size), {{"align", double(sa.align)}, {"size", double(kSize)}}, false);
            ctx.log("B-11 heapTextureSizeAndAlign RGBA8 %u^2 %s: size %llu align %llu", kSize, v.n, (unsigned long long)sa.size,
                    (unsigned long long)sa.align);
        }
        for (bool opt : {true, false}) {
            MTL::TextureDescriptor* d = rgba16fDesc(opt, false);
            const MTL::SizeAndAlign sa = ctx.device()->heapTextureSizeAndAlign(d);
            rep.value(std::string("heap_size.rgba16f.") + (opt ? "optimized" : "plain"), "B", double(sa.size), {{"align", double(sa.align)}, {"size", double(kSize)}}, false);
            ctx.log("B-11 heapTextureSizeAndAlign RGBA16F %u^2 %s: size %llu align %llu", kSize, opt ? "optimized" : "plain",
                    (unsigned long long)sa.size, (unsigned long long)sa.align);
        }
    }
    for (Mode& m : modes) {
        MTL::TextureDescriptor* d = rgba8Desc(m.optimized, m.pfview, m.path != Path::Device);
        if (m.path == Path::Device) {
            m.tex = ctx.texture(d);
        } else {
            const MTL::SizeAndAlign sa = ctx.device()->heapTextureSizeAndAlign(d);
            u64 bytes = sa.size;
            MTL::TextureDescriptor* da = nullptr;
            if (m.path == Path::Alias) {
                da = rgba16fDesc(m.optimized, m.pfview);
                bytes = std::max<u64>(bytes, ctx.device()->heapTextureSizeAndAlign(da).size);
            }
            const u64 offset = sa.align * kHeapOffsetUnits;
            MTL::HeapDescriptor* hd = MTL::HeapDescriptor::alloc()->init();
            hd->setType(MTL::HeapTypePlacement);
            hd->setStorageMode(MTL::StorageModePrivate);
            hd->setHazardTrackingMode(MTL::HazardTrackingModeUntracked);
            hd->setSize(offset + bytes);
            MTL::Heap* h = ctx.heap(hd);
            hd->release();
            m.tex = h->newTexture(d, offset);
            if (!m.tex) throw BenchError("heap->newTexture failed (RGBA8)");
            ctx.keep(m.tex);
            if (da) {
                m.alias = h->newTexture(da, offset);
                if (!m.alias) throw BenchError("heap->newTexture failed (RGBA16F alias)");
                ctx.keep(m.alias);
            }
            ctx.commitResidency();
        }
        rep.value(std::string("allocated_size.") + kPathPrefix[u32(m.path)] + m.name, "B", double(m.tex->allocatedSize()), {{"size", double(kSize)}}, false);
        m.pass = MTL4::RenderPassDescriptor::alloc()->init();
        MTL::RenderPassColorAttachmentDescriptor* c = m.pass->colorAttachments()->object(0);
        c->setTexture(m.tex);
        c->setLoadAction(MTL::LoadActionClear); // Clear: no DRAM read; DontCare after Store draws a validation performance warning
        c->setClearColor(MTL::ClearColor::Make(0, 0, 0, 0));
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
        e->setComputePipelineState(csWrite);
        for (u32 i = 0; i < n; ++i) {
            ctx.table()->setAddress(params->gpuAddress() + i * kSlot, 0);
            ctx.table()->setTexture(m.tex->gpuResourceID(), 0);
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
        e->setComputePipelineState(csRead);
        ctx.table()->setAddress(out->gpuAddress(), 1);
        ctx.table()->setTexture(m.tex->gpuResourceID(), 0);
        e->setArgumentTable(ctx.table());
        for (u32 i = 0; i < n; ++i) {
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

    // Path::Alias: write the bigger RGBA16Float texture at the same heap offset with random
    // content, then an aliasing barrier (as MetalGraphExecutor::encodeBarriers emits:
    // Device | ResourceAlias), then the first write (pass 0) of the measured RGBA8 texture.
    auto aliasPrime = [&](Mode& m, u32 kind) {
        setParams(12, 99, 2, 0);
        setParams(13, 0, kind, 0);
        MTL4::CommandBuffer* cmd = ctx.beginCommands();
        (void)cmd;
        MTL4::ComputeCommandEncoder* e = cmd->computeCommandEncoder();
        e->setComputePipelineState(csWrite16);
        ctx.table()->setAddress(params->gpuAddress() + 12 * kSlot, 0);
        ctx.table()->setTexture(m.alias->gpuResourceID(), 0);
        e->setArgumentTable(ctx.table());
        e->dispatchThreads(MTL::Size::Make(kSize, kSize, 1), MTL::Size::Make(16, 16, 1));
        e->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice | MTL4::VisibilityOptionResourceAlias);
        e->setComputePipelineState(csWrite);
        ctx.table()->setAddress(params->gpuAddress() + 13 * kSlot, 0);
        ctx.table()->setTexture(m.tex->gpuResourceID(), 0);
        e->setArgumentTable(ctx.table());
        e->dispatchThreads(MTL::Size::Make(kSize, kSize, 1), MTL::Size::Make(16, 16, 1));
        e->endEncoding();
        ctx.submit();
    };
    // The alias-writing kernel is checked once per mode: write it, read it back through a blit
    // (the later aliasing writes destroy its contents, so this is a separate dry submission).
    auto verifyAliasWrite = [&](Mode& m) {
        setParams(12, 99, 2, 0);
        MTL4::CommandBuffer* cmd = ctx.beginCommands();
        MTL4::ComputeCommandEncoder* e = cmd->computeCommandEncoder();
        e->setComputePipelineState(csWrite16);
        ctx.table()->setAddress(params->gpuAddress() + 12 * kSlot, 0);
        ctx.table()->setTexture(m.alias->gpuResourceID(), 0);
        e->setArgumentTable(ctx.table());
        e->dispatchThreads(MTL::Size::Make(kSize, kSize, 1), MTL::Size::Make(16, 16, 1));
        e->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageBlit, MTL4::VisibilityOptionDevice);
        e->copyFromTexture(m.alias, 0, 0, MTL::Origin::Make(0, 0, 0), MTL::Size::Make(kSize, kSize, 1), rb16, 0, kSize * 8,
                           size_t(kSize) * kSize * 8);
        e->endEncoding();
        ctx.submit();
        u64 st = 91;
        bool ok = true;
        for (int i = 0; i < 6000 && ok; ++i) {
            const u32 x = i == 0 ? 0 : i == 1 ? kSize - 1 : u32(xorshift64(st) % kSize);
            const u32 y = i == 0 ? 0 : i == 1 ? kSize - 1 : u32(xorshift64(st) % kSize);
            u32 c[4];
            content(2, x, y, 99, c);
            const _Float16* px = rb16h + (size_t(y) * kSize + x) * 4;
            for (int k = 0; k < 4; ++k)
                if (std::fabs(float(px[k]) - c[k] / 255.0f) > 2e-3f) ok = false;
        }
        if (!ok) bad(std::string("alias texels ") + m.name);
    };

    std::map<std::string, double> gbs, gbest;
    double linRatio = 0, rtLinRatio = 0;
    for (Mode& m : modes) {
        const std::string pre = kPathPrefix[u32(m.path)];
        const bool devPlain = m.path == Path::Device && !m.optimized; // the original linearity-control config
        if (m.path == Path::Alias) verifyAliasWrite(m);
        for (u32 kind = 0; kind < 3; ++kind) {
            const std::string kn = kKinds[kind], tagBase = std::string(m.name) + "." + kn, tagFull = pre + tagBase;
            const std::map<std::string, double> prm = {{"optimized", m.optimized ? 1.0 : 0.0}, {"kind", double(kind)},
                                                        {"size", double(kSize)}, {"passes", double(kPasses)},
                                                        {"path", double(u32(m.path))}, {"pfview", m.pfview ? 1.0 : 0.0}};
            // Known initial state: full write with pass 0 (aliasing: after a random RGBA16F write at the same offset).
            if (m.path == Path::Alias) aliasPrime(m, kind);
            else csWriteMs(m, kind, false, 0, 1);
            ctx.keepWarm(20);

            // Compute write, full blocks (passes 1..4).
            const u32 linReps = (devPlain && kind == 2) ? 25 : 0; // the linearity-control config: more repetitions
            Stats t = ctx.measure([&] { return csWriteMs(m, kind, false, 1, kPasses); }, linReps);
            verify(m, kind, false, kPasses, kPasses, "cs_full." + tagBase);
            Stats bw = toRate(t, kTexBytes * kPasses * 1e-6);
            rep.metric(pre + "write." + tagBase, "GB/s", bw, prm);
            gbs[pre + "write." + tagBase] = bw.median;
            gbest[pre + "write." + tagBase] = bw.max; // best repetition: contention only adds time
            if (devPlain && kind == 2) {
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
            rep.metric(pre + "write_partial." + tagBase, "GB/s", bw, prm);
            gbs[pre + "write_partial." + tagBase] = bw.median;
            gbest[pre + "write_partial." + tagBase] = bw.max; // best repetition: contention only adds time
            ctx.keepWarm(20);

            // Render target write (full-screen triangle, load Clear, store).
            t = ctx.measure([&] { return rtWriteMs(m, kind, 1, kPasses); }, linReps);
            verify(m, kind, false, kPasses, kPasses, "rt." + tagBase);
            bw = toRate(t, kTexBytes * kPasses * 1e-6);
            rep.metric(pre + "rt_write." + tagBase, "GB/s", bw, prm);
            gbs[pre + "rt_write." + tagBase] = bw.median;
            gbest[pre + "rt_write." + tagBase] = bw.max; // best repetition: contention only adds time
            if (devPlain && kind == 2) {
                const Stats t2 = ctx.measure([&] { return rtWriteMs(m, kind, 1, 2 * kPasses); }, linReps);
                rtLinRatio = t2.min / t.min;
                rtWriteMs(m, kind, 1, kPasses); // back to a known state (pass 4)
            }
            ctx.keepWarm(20);

            // Read (texture holds pass 4 of this kind, written by the render pass above).
            t = ctx.measure([&] { return readMs(m, kPasses); });
            verifyRead(kind, kPasses, tagBase);
            bw = toRate(t, kTexBytes * kPasses * 1e-6);
            rep.metric(pre + "read." + tagBase, "GB/s", bw, prm);
            gbs[pre + "read." + tagBase] = bw.median;
            gbest[pre + "read." + tagBase] = bw.max; // best repetition: contention only adds time
            ctx.keepWarm(20);
        }
    }

    // Indirect evidence: optimized / plain bandwidth per test and content (device: the original
    // metric names; heap paths: "opt_over_plain.heap.*", "opt_over_plain.heap_alias.*").
    const char* tests[4] = {"write", "write_partial", "rt_write", "read"};
    std::string deltas, heapDeltas;
    auto g = [&](const std::string& pre, const char* test, const char* cs, const char* kn) {
        return gbs[pre + test + "." + cs + "." + kn];
    };
    for (Path path : {Path::Device, Path::Heap, Path::Alias}) {
        const std::string pre = kPathPrefix[u32(path)];
        const std::string vpre = path == Path::Device ? "" : pre; // value-name infix
        for (const char* test : tests) {
            for (const char* kn : kKinds) {
                const double o = g(pre, test, "optimized", kn), pl = g(pre, test, "plain", kn), pf = g(pre, test, "pfview", kn);
                rep.value(std::string("opt_over_plain.") + vpre + test + "." + kn, "ratio", o / pl, {{"size", double(kSize)}, {"path", double(u32(path))}});
                rep.value(std::string("pfview_over_plain.") + vpre + test + "." + kn, "ratio", pf / pl, {{"size", double(kSize)}, {"path", double(u32(path))}});
                if (std::string(kn) != "constant" || std::string(test) == "write") {
                    std::string& dst = path == Path::Device ? deltas : heapDeltas;
                    dst += (path == Path::Device ? "" : pre) + std::string(test) + "." + kn + "=" + std::to_string(o / pl).substr(0, 4) + " ";
                }
            }
            // Advantage of compressible content over random: speed(kind) / speed(random), per case,
            // and its gain from optimized over plain (what lossless compression adds).
            for (const char* cs : {"optimized", "plain", "pfview"})
                for (const char* kn : {"constant", "smooth"})
                    rep.value(std::string("adv.") + vpre + cs + "." + test + "." + kn, "ratio",
                              g(pre, test, cs, kn) / g(pre, test, cs, "random"), {{"size", double(kSize)}, {"path", double(u32(path))}});
            for (const char* kn : {"constant", "smooth"}) {
                const double advO = g(pre, test, "optimized", kn) / g(pre, test, "optimized", "random");
                const double advP = g(pre, test, "plain", kn) / g(pre, test, "plain", "random");
                rep.value(std::string("adv_gain.") + vpre + test + "." + kn, "ratio", advO / advP, {{"size", double(kSize)}, {"path", double(u32(path))}});
            }
        }
    }

    auto gb = [&](const std::string& pre, const char* test, const char* cs, const char* kn) {
        return gbest[pre + test + "." + cs + "." + kn];
    };
    // Gain of a case over plain (BEST repetition of each cell), aggregated (geometric mean) over the compressible cells (constant and
    // smooth content x write / write_partial / rt_write / read) of one path: a single cell is noisy
    // (other GPU clients), the aggregate is what the control uses.
    auto gain = [&](Path path, const char* cs) {
        const std::string pre = kPathPrefix[u32(path)];
        double lg = 0;
        u32 n = 0;
        for (const char* test : tests)
            for (const char* kn : {"constant", "smooth"}) {
                lg += std::log(gb(pre, test, cs, kn) / gb(pre, test, "plain", kn));
                ++n;
            }
        return std::exp(lg / n);
    };
    std::string gains;
    for (Path path : {Path::Device, Path::Heap, Path::Alias})
        for (const char* cs : {"optimized", "pfview"}) {
            const double v = gain(path, cs);
            rep.value(std::string("gain_geomean.") + kPathPrefix[u32(path)] + cs, "ratio", v, {{"path", double(u32(path))}});
            gains += std::string(kPathPrefix[u32(path)]) + cs + "=" + std::to_string(v).substr(0, 4) + " ";
        }

    // Negative control (S-TEX-2 sensitivity): the optimized DEVICE texture must show a clear gain over
    // plain (>= 1.15x), and the same texture with PixelFormatView (documented to disable lossless
    // compression) must show less than half of that gain on the device path and on both heap
    // paths; if the optimized gain is absent the control proves nothing and fails.
    const double devGain = gain(Path::Device, "optimized");
    bool pfOk = true;
    std::string pfText;
    for (Path path : {Path::Device, Path::Heap, Path::Alias}) {
        const double pf = gain(path, "pfview");
        if (pf - 1.0 >= 0.5 * (devGain - 1.0)) pfOk = false;
        pfText += std::string(kPathPrefix[u32(path)]) + "pfview=" + std::to_string(pf).substr(0, 4) + " ";
    }
    rep.negative(devGain >= 1.15 && pfOk,
                 "geomean gain over plain across the compressible cells: device optimized " + std::to_string(devGain).substr(0, 4) +
                     " (needs >= 1.15), PixelFormatView " + pfText + "(each needs < 1 + half of the device gain)" +
                     (devGain >= 1.15 && pfOk ? "" : " -- CONTROL FAILED"));
    rep.note("geomean gain over plain (compressible cells): " + gains);

    rep.negative(linRatio > 1.8 && linRatio < 2.2 && rtLinRatio > 1.7 && rtLinRatio < 2.3,
                 "2x passes -> time x" + std::to_string(linRatio).substr(0, 5) + " (compute write, plain random), x" +
                     std::to_string(rtLinRatio).substr(0, 5) + " (render target)");
    rep.negative(resultsOk, resultsOk ? "texels read back through a blit (6000 per config and write test) and 6 read-kernel sums per config match the CPU replay of the content function"
                                      : "MISMATCH: " + wrong);
    rep.status(Status::Partial, "compression ratio is not measurable (Metal exposes no compression counters, F4.4): only optimized/plain bandwidth deltas are reported as indirect evidence");
    if (!ctx.apple10())
        rep.note("Apple9 path: shader-written textures are not compressed there, 'optimized' expected equal to 'plain' for compute writes (render targets are compressed on every Apple GPU)");
    rep.note("optimized/plain ratios (device): " + deltas);
    rep.note("optimized/plain ratios (heap, heap_alias): " + heapDeltas);
    rep.note("heap = placement heap (private, untracked) texture placed at 4 alignment units; heap_alias = same heap/offset, RGBA16Float 4096x4096 random-written first, aliasing barrier (Device|ResourceAlias), then the measured RGBA8 texture; pfview = optimized + TextureUsagePixelFormatView (negative control); heap_size.* = heapTextureSizeAndAlign, allocated_size.* = Resource.allocatedSize");
    rep.note("optimized = TextureDescriptor.allowGPUOptimizedContents true; plain = false; 4096x4096 RGBA8Unorm private, 4 passes per timed span, bandwidth counts logical bytes written/read (partial: 50%: 2x2 cells in a checkerboard, touching every 4x4 block)");
    rep.note("render target partial writes not measured: they need loadAction Load (a different experiment); random content = hashed texels, smooth = 16-texel-step gradients, constant = one colour per pass");
}

} // namespace

SOC_BENCH("B-11", "texture.compression", "Lossless compression: cost of writing/reading optimized vs plain textures (bandwidth deltas only)", benchCompression);

} // namespace soc
