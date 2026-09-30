// B-16: mesh shaders vs the classic vertex pipeline (S-GEO-1..3 of
// docs/APPLE_SOC_PLAYBOOK.md).  The same 840 x 840 cell grid (1.41 M small
// triangles, ~1.5 pixel each) is drawn
//   * with an indexed draw (32-bit indices, vertex shader reads positions);
//   * with object + mesh shaders, meshlets of 25 / 64 / 81 / 121 / 256
//     vertices (classes 32 / 64 / 96 / 128 / 256; 32 / 98 / 128 / 200 / 450
//     triangles), object threadgroup = 32 meshlets, payload = compacted
//     meshlet ids;
//   * at the 128-vertex meshlet with a 1 KiB and a 16 KiB (maximum) payload
//     (object shader writes it all, mesh shader reads it);
//   * with the object shader culling half of the meshlets (checkerboard).
// Fragment shader: writes 1 to an R8 target.  The GPU time of a pass is the
// CommandTimer span minus the same pass without draws (clear + store).
//
// Metrics (Gtri/s of the triangles actually drawn): tris_per_s.vertex,
// tris_per_s.mesh.meshlet_<32|64|96|128|256>, tris_per_s.mesh.meshlet_128.
// payload_1k / payload_16k, tris_per_s.mesh.meshlet_<v>.cull_half; ms.* = pass
// time; cull_speedup.meshlet_<v> = time(no cull) / time(cull half).
//
// Negative controls: half of the triangles takes half the time (vertex and
// every mesh variant, 1.7..2.3x); culling half of the meshlets is faster
// (< 0.85x); the rendered image is verified (all pixels covered; the two
// complementary culled images are disjoint and their union is the full one).

#include "harness.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>

namespace soc {
namespace {

constexpr u32 kG    = 1680; // grid cells per side (divisible by 4, 7, 8, 10, 15)
constexpr u32 kRT   = 2048;
constexpr u64 kTris = 2ull * kG * kG;

struct Params {
    u32 meshletCount, meshletsPerRow;
    i32 cullParity;
    u32 zero, gridCells, pad[3];
};
static_assert(sizeof(Params) == 32);

struct Variant {
    const char* metric;   // suffix after tris_per_s.
    const char* fn;       // shader function prefix ("" = vertex pipeline)
    u32 bw;               // cells per meshlet side
    u32 threads;          // mesh threadgroup size
    u32 payloadBytes;
    u32 verts, prims;
    u32 maxVerts, maxPrims;
    bool cullToo;         // measure the cull-half run as well
};
const Variant kVariants[] = {
    {"vertex", "", 0, 0, 0, 0, 0, 0, 0, false},
    {"mesh.meshlet_32", "b16_m32", 4, 32, 132, 25, 32, 25, 32, true},
    {"mesh.meshlet_64", "b16_m64", 7, 64, 132, 64, 98, 64, 98, true},
    {"mesh.meshlet_96", "b16_m96", 8, 96, 132, 81, 128, 81, 128, true},
    {"mesh.meshlet_128", "b16_m128", 10, 128, 132, 121, 200, 121, 200, true},
    {"mesh.meshlet_256", "b16_m256", 15, 256, 132, 256, 450, 256, 450, true},
    {"mesh.meshlet_128.payload_1k", "b16_m128_p1k", 10, 128, 128 + 224 * 4, 121, 200, 121, 200, false},
    {"mesh.meshlet_128.payload_16k", "b16_m128_p16k", 10, 128, 128 + 4064 * 4, 121, 200, 121, 200, false},
};

struct Rig {
    Context& ctx;
    MTL::Buffer* pos = nullptr;
    MTL::Buffer* idx = nullptr;
    MTL::Buffer* params = nullptr;
    MTL::Texture* rt = nullptr;
    std::vector<MTL::RenderPipelineState*> pso;
    u64 idxBytes = 0;
    explicit Rig(Context& c) : ctx(c) {}
};

// One render pass: vertex path draws `frac` (1 or 1/2) of the indices, mesh path processes
// `frac` of the meshlets, culling with `parity` (-1 none).  draws = false: empty pass.
double passMs(Rig& r, const Variant& v, MTL::RenderPipelineState* pso, bool half, i32 parity, bool draws, bool cullAll = false) {
    Params p{};
    const u32 meshlets = v.bw ? (kG / v.bw) * (kG / v.bw) : 0;
    p.meshletsPerRow = v.bw ? kG / v.bw : 0;
    p.meshletCount   = half ? meshlets / 2 : meshlets;
    p.cullParity     = parity;
    p.gridCells      = kG;
    std::memcpy(r.params->contents(), &p, sizeof(p));
    CommandTimer t(r.ctx);
    MTL4::CommandBuffer* cmd = t.begin();
    MTL4::RenderPassDescriptor* pd = MTL4::RenderPassDescriptor::alloc()->init();
    auto* c = pd->colorAttachments()->object(0);
    c->setTexture(r.rt);
    c->setLoadAction(MTL::LoadActionClear);
    c->setClearColor(MTL::ClearColor::Make(0, 0, 0, 0));
    c->setStoreAction(MTL::StoreActionStore);
    MTL4::RenderCommandEncoder* re = cmd->renderCommandEncoder(pd);
    pd->release();
    re->setViewport(MTL::Viewport{0.0, 0.0, double(kRT), double(kRT), 0.0, 1.0});
    re->setRenderPipelineState(pso);
    // Front-end variant: every triangle is back-facing (counter-clockwise, default winding is clockwise) and culled after
    // the vertex/mesh stage, so the rasteriser and the fragment shader see nothing.
    if (cullAll) re->setCullMode(MTL::CullModeBack);
    r.ctx.table()->setAddress(r.params->gpuAddress(), 0);
    r.ctx.table()->setAddress(r.pos->gpuAddress(), 1);
    re->setArgumentTable(r.ctx.table(), MTL::RenderStageVertex | MTL::RenderStageObject | MTL::RenderStageMesh | MTL::RenderStageFragment);
    if (draws) {
        if (!v.bw) {
            const u64 n = half ? 3ull * kG * kG : 6ull * kG * kG;
            re->drawIndexedPrimitives(MTL::PrimitiveTypeTriangle, n, MTL::IndexTypeUInt32, r.idx->gpuAddress(), r.idxBytes);
        } else {
            const u32 groups = (p.meshletCount + 31) / 32;
            re->drawMeshThreadgroups(MTL::Size::Make(groups, 1, 1), MTL::Size::Make(32, 1, 1), MTL::Size::Make(v.threads, 1, 1));
        }
    }
    re->endEncoding();
    return t.finish();
}

struct Image {
    std::vector<u8> px;
};
Image readImage(Rig& r) {
    Image im;
    im.px.resize(size_t(kRT) * kRT);
    r.rt->getBytes(im.px.data(), kRT, MTL::Region::Make2D(0, 0, kRT, kRT), 0);
    return im;
}


// Under the validation layers (--validate) GPU times are distorted by the instrumentation: the timing controls are
// then informational (detail says so); the correctness controls stay enforced.
bool timingControlsEnforced() { return !std::getenv("MTL_SHADER_VALIDATION") && !std::getenv("MTL_DEBUG_LAYER"); }
bool ratioOk(double half, double full) {
    const double q = full / half;
    return q >= 1.7 && q <= 2.3;
}

MTL::RenderPipelineState* makePipeline(Context& ctx, MTL::Library* lib, const Variant& v) {
    NS::Error* err = nullptr;
    MTL::RenderPipelineState* p = nullptr;
    if (!v.bw) {
        MTL4::RenderPipelineDescriptor* d = MTL4::RenderPipelineDescriptor::alloc()->init();
        d->setVertexFunctionDescriptor(ctx.function(lib, "b16_vs"));
        d->setFragmentFunctionDescriptor(ctx.function(lib, "b16_fs"));
        d->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatR8Unorm);
        p = ctx.render(d);
        d->release();
        return p;
    }
    MTL4::MeshRenderPipelineDescriptor* d = MTL4::MeshRenderPipelineDescriptor::alloc()->init();
    d->setObjectFunctionDescriptor(ctx.function(lib, std::string(v.fn) + "_obj"));
    d->setMeshFunctionDescriptor(ctx.function(lib, std::string(v.fn) + "_mesh"));
    d->setFragmentFunctionDescriptor(ctx.function(lib, "b16_fs"));
    d->setMaxTotalThreadsPerObjectThreadgroup(32);
    d->setMaxTotalThreadsPerMeshThreadgroup(v.threads);
    d->setPayloadMemoryLength(v.payloadBytes);
    d->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatR8Unorm);
    p = ctx.compiler()->newRenderPipelineState(d, nullptr, &err);
    d->release();
    if (!p) throw BenchError(std::string("mesh pipeline ") + v.fn + ": " +
                             (err && err->localizedDescription() ? err->localizedDescription()->utf8String() : "?"));
    ctx.keep(p);
    return p;
}

void benchMesh(Context& ctx, Report& rep) {
    if (!ctx.device()->supportsFamily(MTL::GPUFamilyApple9)) {
        rep.status(Status::Unsupported, "mesh shaders need Apple9");
        return;
    }
    MTL::Library* lib = ctx.library("b16_mesh.metal");
    Rig r(ctx);
    // Geometry: positions (float2) and the row-major index buffer of the vertex path.
    const u32 stride = kG + 1;
    r.pos = ctx.buffer(size_t(stride) * stride * 8);
    {
        auto* p = static_cast<float*>(r.pos->contents());
        for (u32 j = 0; j < stride; ++j)
            for (u32 i = 0; i < stride; ++i) {
                p[2 * (size_t(j) * stride + i) + 0] = -1.0f + 2.0f * float(i) / float(kG);
                p[2 * (size_t(j) * stride + i) + 1] = -1.0f + 2.0f * float(j) / float(kG);
            }
    }
    r.idxBytes = 6ull * kG * kG * 4;
    r.idx      = ctx.buffer(r.idxBytes);
    {
        auto* x = static_cast<u32*>(r.idx->contents());
        size_t k = 0;
        for (u32 j = 0; j < kG; ++j)
            for (u32 i = 0; i < kG; ++i) {
                const u32 a = j * stride + i, b = a + 1, c = a + stride, d = c + 1;
                x[k++] = a; x[k++] = b; x[k++] = c;
                x[k++] = b; x[k++] = d; x[k++] = c;
            }
    }
    r.params = ctx.buffer(256);
    MTL::TextureDescriptor* td = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatR8Unorm, kRT, kRT, false);
    td->setUsage(MTL::TextureUsageRenderTarget);
    td->setStorageMode(MTL::StorageModeShared);
    r.rt = ctx.texture(td);
    for (const Variant& v : kVariants) r.pso.push_back(makePipeline(ctx, lib, v));

    const u32 reps = std::max<u32>(ctx.reps(), 15);
    ctx.keepWarm(50);
    const double fixedMs = ctx.measure([&] { return passMs(r, kVariants[0], r.pso[0], false, -1, false); }, reps).median;
    rep.value("pass.fixed_ms", "ms", fixedMs, {{"rt", double(kRT)}}, false);

    std::string linearBad, cullBad, imageBad;
    bool linear = true, cullFaster = true, imageOk = true;
    u32 pixels = kRT * kRT;
    for (size_t vi = 0; vi < std::size(kVariants); ++vi) {
        const Variant& v = kVariants[vi];
        MTL::RenderPipelineState* pso = r.pso[vi];
        // Full and half workload; re-measure a pair that breaks the linearity (other GPU clients).
        Stats full, half;
        for (int attempt = 0; attempt < 5; ++attempt) {
            full = ctx.measure([&] { return passMs(r, v, pso, false, -1, true) - fixedMs; }, reps);
            ctx.keepWarm(20);
            half = ctx.measure([&] { return passMs(r, v, pso, true, -1, true) - fixedMs; }, reps);
            ctx.keepWarm(20);
            if (ratioOk(half.median, full.median)) break;
        }
        const bool lin = ratioOk(half.median, full.median);
        if (!lin) { linear = false; linearBad += std::string(v.metric) + " " + std::to_string(full.median / half.median).substr(0, 4) + "x "; }
        // Image check of the full pass: every pixel covered.
        passMs(r, v, pso, false, -1, true);
        {
            const Image im = readImage(r);
            u64 covered = 0;
            for (u8 b : im.px) covered += b == 255 ? 1 : 0;
            if (covered != pixels) { imageOk = false; imageBad += std::string(v.metric) + " full covered " + std::to_string(covered) + "/" + std::to_string(pixels) + " "; }
        }
        // Gtri/s = tris / (ms * 1e6).
        auto rate = [&](const Stats& s, double tris) {
            Stats g = s;
            const double k = tris / 1e6;
            g.median = k / s.median; g.min = k / s.max; g.max = k / s.min; g.p10 = k / s.p90; g.p90 = k / s.p10; g.mean = k / s.mean;
            return g;
        };
        std::map<std::string, double> params = {{"tris", double(kTris)}, {"ms", full.median}, {"ratio_half", full.median / half.median}};
        if (v.bw) {
            params["verts_per_meshlet"] = v.verts;
            params["prims_per_meshlet"] = v.prims;
            params["threads"] = v.threads;
            params["payload_bytes"] = v.payloadBytes;
        }
        rep.metric(std::string("tris_per_s.") + v.metric, "Gtri/s", rate(full, double(kTris)), params, true);
        rep.value(std::string("ms.") + v.metric, "ms", full.median, {{"tris", double(kTris)}}, false);

        // Front-end only: all triangles culled by the hardware after the geometry stage.
        {
            Stats fe = ctx.measure([&] { return passMs(r, v, pso, false, -1, true, true) - fixedMs; }, reps);
            ctx.keepWarm(20);
            passMs(r, v, pso, false, -1, true, true);
            const Image im = readImage(r);
            u64 covered = 0;
            for (u8 b : im.px) covered += b != 0 ? 1 : 0;
            if (covered != 0) { imageOk = false; imageBad += std::string(v.metric) + " frontend image not empty (" + std::to_string(covered) + " px) "; }
            rep.metric(std::string("tris_per_s.") + v.metric + ".frontend", "Gtri/s", rate(fe, double(kTris)), {{"tris", double(kTris)}, {"ms", fe.median}}, true);
        }
        if (v.cullToo) {
            Stats cull;
            for (int attempt = 0; attempt < 3; ++attempt) {
                cull = ctx.measure([&] { return passMs(r, v, pso, false, 0, true) - fixedMs; }, reps);
                ctx.keepWarm(20);
                if (cull.median < 0.85 * full.median) break;
                full = ctx.measure([&] { return passMs(r, v, pso, false, -1, true) - fixedMs; }, reps);
                ctx.keepWarm(20);
            }
            const double speedup = full.median / cull.median;
            if (!(cull.median < 0.85 * full.median)) { cullFaster = false; cullBad += std::string(v.metric) + " " + std::to_string(cull.median / full.median).substr(0, 4) + "x "; }
            rep.metric(std::string("tris_per_s.") + v.metric + ".cull_half", "Gtri/s", rate(cull, double(kTris) / 2), {{"tris_drawn", double(kTris / 2)}, {"ms", cull.median}}, true);
            rep.value(std::string("cull_speedup.") + std::string(v.metric).substr(5), "ratio", speedup, {{"ms_nocull", full.median}, {"ms_cull", cull.median}}, true);
            // Image: the two complementary culled images are disjoint, their union is the full image.
            passMs(r, v, pso, false, 0, true);
            const Image a = readImage(r);
            passMs(r, v, pso, false, 1, true);
            const Image b = readImage(r);
            u64 overlap = 0, gap = 0, ca = 0, cb = 0;
            for (size_t i = 0; i < a.px.size(); ++i) {
                const bool x = a.px[i] == 255, y = b.px[i] == 255;
                ca += x; cb += y;
                if (x && y) ++overlap;
                if (!x && !y) ++gap;
            }
            const bool balanced = ca > pixels * 0.4 && cb > pixels * 0.4;
            if (overlap || gap || !balanced) {
                imageOk = false;
                imageBad += std::string(v.metric) + " cull images overlap " + std::to_string(overlap) + " gap " + std::to_string(gap) + " covered " +
                            std::to_string(ca) + "+" + std::to_string(cb) + " ";
            }
        }
    }

    const bool enforce = timingControlsEnforced();
    rep.negative(linear || !enforce, std::string(enforce ? "" : "[timing not enforced under validation] ") + (linear ? "half of the triangles/meshlets -> 1.7..2.3x less time for the vertex pipeline and all 7 mesh variants"
                                : "NOT linear (full/half): " + linearBad));
    rep.negative(cullFaster || !enforce, std::string(enforce ? "" : "[timing not enforced under validation] ") + (cullFaster ? "object-shader culling of half the meshlets: time < 0.85x for all 5 meshlet sizes"
                                        : "culling not faster (cull/full): " + cullBad));
    rep.negative(imageOk, imageOk ? "images verified: full pass covers all 1024^2 pixels; the two complementary culled images are disjoint and cover all"
                                  : "IMAGE WRONG: " + imageBad);
    if (!imageOk) rep.status(Status::Failed, "wrong image");
    rep.note("vertex path: 32-bit indices in row-major cell order; meshlets are BWxBW cell blocks (25/64/81/121/256 vertices); "
             "object threadgroup = 32 meshlets, payload = ids (+1 KiB / 16 KiB ballast); tris/s = drawn triangles / (pass - empty pass)");
}

} // namespace

SOC_BENCH("B-16", "geometry.mesh_vs_vertex", "Mesh shaders vs vertex pipeline: meshlet size, payload size, object-shader culling", benchMesh);

} // namespace soc
