// F9-S5: alpha test in ray tracing with intersection functions.
//
// Raster rule mirrored (shaders/meshlet.metal visibility_alpha_fs and
// forward.metal): alpha = material.baseColor.a * baseColorTex.a (half), the
// fragment is discarded when alpha < material.alphaCutoff; a material is
// alpha-tested when alphaCutoff > 0.  RT has no derivatives, so the
// intersection function samples at an explicit LOD (bilinear, repeat).
//
// Strategies (BLAS geometry flags + intersection function table, rebuilt for each):
//   A      masked geometry non-opaque, opaque geometry opaque, ONE generic
//          function at table offset 0 (material lookup inside) - portable.
//   A_pd   as A, UVs from per-triangle primitive data instead of vertex fetch.
//   A_all  A with every geometry non-opaque (what opaque flags save).
//   B      per-material table slots (geometry offset = material index) with
//          specialised functions (textured cutoff / constant alpha); opaque
//          geometry stays opaque.  Needs ctx.apple10() (skipped on Apple9).
//   B_all  B with every geometry non-opaque and opaque slots filled with
//          setOpaqueTriangleIntersectionFunction.
//   C      everything opaque: negative control, must disagree with the alpha
//          reference wherever a ray crosses a transparent texel.
// Ray kinds: primary (closest hit) and shadow (accept_any_intersection).
// Corpora: Sponza (3 alpha-masked materials) and a synthetic scene with
// exactly known holes (checker, disc, constant alpha planes over an opaque
// backdrop).  CPU reference: CpuBvh with a Filter evaluating the same rule
// in double precision; texels within 2/255 of the cutoff are "ambiguous"
// (GPU sampler fixed-point weights, half conversion) and reported apart:
// a ray is ambiguous when accepting vs rejecting ambiguous texels changes
// its reference result.
//
// Also: raster/RT coherence (LOD rule study on the CPU) and a debug variant
// counting intersection function invocations per material.
#include "f9_common.h"

#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <set>

namespace f9 {
namespace {

using phosphor::GPUMaterial;
using phosphor::INVALID_TEXTURE_INDEX;

struct InstRec {
    u32 vertexOffset, indexOffset, material, pad;
};
struct AlphaParams {
    float lod;
    u32 pad0, pad1, pad2;
};
struct PrimUV {
    float uv[6];
};
struct TraceParams {
    u32 count, mask, pad0, pad1;
};
static_assert(sizeof(InstRec) == 16 && sizeof(PrimUV) == 24 && sizeof(TraceParams) == 16);

constexpr f64 kBand = 2.0 / 255.0; // ambiguity band around the cutoff

soc::Stats scaled(soc::Stats s, double k) {
    s.median *= k; s.min *= k; s.max *= k; s.p10 *= k; s.p90 *= k; s.mean *= k;
    return s;
}

// ---------------------------------------------------------------------------
// CPU textures with mips (alpha rule, same data as the GPU textures)
// ---------------------------------------------------------------------------

struct MipLevel {
    u32 w = 0, h = 0;
    std::vector<soc::u8> rgba;
};
struct Mips {
    std::vector<MipLevel> lv;
    [[nodiscard]] bool valid() const { return !lv.empty(); }
};

Mips buildMips(const CpuTexture& t) {
    Mips m;
    MipLevel l0;
    l0.w = t.width; l0.h = t.height; l0.rgba = t.rgba;
    m.lv.push_back(std::move(l0));
    while (m.lv.back().w > 1 || m.lv.back().h > 1) {
        const MipLevel& p = m.lv.back();
        MipLevel n;
        n.w = std::max(1u, p.w / 2); n.h = std::max(1u, p.h / 2);
        n.rgba.resize(size_t(n.w) * n.h * 4);
        for (u32 y = 0; y < n.h; ++y)
            for (u32 x = 0; x < n.w; ++x)
                for (u32 c = 0; c < 4; ++c) {
                    u32 sum = 0;
                    for (u32 dy = 0; dy < 2; ++dy)
                        for (u32 dx = 0; dx < 2; ++dx)
                            sum += p.rgba[(size_t(std::min(p.h - 1, 2 * y + dy)) * p.w + std::min(p.w - 1, 2 * x + dx)) * 4 + c];
                    n.rgba[(size_t(y) * n.w + x) * 4 + c] = soc::u8((sum + 2) / 4);
                }
        m.lv.push_back(std::move(n));
    }
    return m;
}

f64 texelA(const MipLevel& l, long x, long y) {
    x = ((x % long(l.w)) + long(l.w)) % long(l.w);
    y = ((y % long(l.h)) + long(l.h)) % long(l.h);
    return l.rgba[(size_t(y) * l.w + size_t(x)) * 4 + 3] / 255.0;
}

f64 bilinearA(const MipLevel& l, f64 u, f64 v) { // texel centres at (i + 0.5)
    const f64 x = u * l.w - 0.5, y = v * l.h - 0.5;
    const f64 fx = std::floor(x), fy = std::floor(y);
    const f64 ax = x - fx, ay = y - fy;
    const long ix = long(fx), iy = long(fy);
    return (texelA(l, ix, iy) * (1 - ax) + texelA(l, ix + 1, iy) * ax) * (1 - ay) +
           (texelA(l, ix, iy + 1) * (1 - ax) + texelA(l, ix + 1, iy + 1) * ax) * ay;
}

f64 sampleA(const Mips& m, f64 u, f64 v, f64 lod) { // trilinear for fractional LODs
    lod = std::clamp(lod, 0.0, f64(m.lv.size() - 1));
    const size_t l0 = size_t(std::floor(lod));
    const f64 f = lod - f64(l0);
    f64 a = bilinearA(m.lv[l0], u, v);
    if (f > 0 && l0 + 1 < m.lv.size()) a = a * (1 - f) + bilinearA(m.lv[l0 + 1], u, v) * f;
    return a;
}

// ---------------------------------------------------------------------------
// Corpus: a scene + GPU resources the intersection functions read
// ---------------------------------------------------------------------------

struct Corpus {
    std::string name;
    SceneData* s = nullptr;
    GpuGeometry g;
    std::vector<Mips> mips;               // per scene texture (empty if unused)
    std::vector<u32> meshMaterial;        // per mesh: material of its instances (~0u unused)
    std::vector<u32> meshInstances;       // per mesh: instance count
    MTL::Buffer *mats = nullptr, *texTable = nullptr, *recs = nullptr, *params = nullptr, *counters = nullptr, *pd = nullptr;
    std::vector<u64> pdOffset;            // per mesh, bytes into pd
    TriangleSoup soup;
    std::unique_ptr<CpuBvh> bvh;
    u32 maskedMaterials = 0;
    u64 maskedTriangles = 0;
    [[nodiscard]] bool masked(u32 inst) const { return s->materials[s->instances[inst].materialIndex].alphaCutoff > 0.0f; }
};

bool texturedMask(const GPUMaterial& m) { return m.alphaCutoff > 0.0f && m.baseColorTex != INVALID_TEXTURE_INDEX; }

void prepareCorpus(soc::Context& ctx, Corpus& c, SceneData& s, const std::string& name) {
    c.name = name;
    c.s = &s;
    c.g = uploadGeometry(ctx, s);
    const u32 meshes = s.scene.getMeshCount();
    c.meshMaterial.assign(meshes, ~0u);
    c.meshInstances.assign(meshes, 0);
    for (const auto& i : s.instances) {
        u32& mm = c.meshMaterial[i.meshIndex];
        if (mm != ~0u && mm != i.materialIndex) throw soc::BenchError("mesh used with two materials: BLAS per mesh would be wrong");
        mm = i.materialIndex;
        c.meshInstances[i.meshIndex]++;
    }
    // Textures of the masked materials only: CPU mips + GPU texture with the same levels.
    c.mips.assign(s.textures.size(), {});
    c.texTable = ctx.buffer(std::max<size_t>(s.textures.size(), 1) * sizeof(MTL::ResourceID));
    std::memset(c.texTable->contents(), 0, c.texTable->length());
    for (const GPUMaterial& m : s.materials) {
        if (!texturedMask(m) || m.baseColorTex >= s.textures.size() || c.mips[m.baseColorTex].valid()) continue;
        const CpuTexture& t = s.textures[m.baseColorTex];
        c.mips[m.baseColorTex] = buildMips(t);
        const Mips& mp = c.mips[m.baseColorTex];
        MTL::TextureDescriptor* d = MTL::TextureDescriptor::texture2DDescriptor(
            t.sRGB ? MTL::PixelFormatRGBA8Unorm_sRGB : MTL::PixelFormatRGBA8Unorm, t.width, t.height, true);
        d->setUsage(MTL::TextureUsageShaderRead);
        d->setStorageMode(MTL::StorageModeShared);
        MTL::Texture* tex = ctx.texture(d);
        if (tex->mipmapLevelCount() != mp.lv.size()) throw soc::BenchError("mip count differs from the CPU chain");
        for (u32 l = 0; l < mp.lv.size(); ++l)
            tex->replaceRegion(MTL::Region::Make2D(0, 0, mp.lv[l].w, mp.lv[l].h), l, mp.lv[l].rgba.data(), mp.lv[l].w * 4);
        static_cast<MTL::ResourceID*>(c.texTable->contents())[m.baseColorTex] = tex->gpuResourceID();
    }
    c.mats = ctx.buffer(std::max<size_t>(s.materials.size(), 1) * sizeof(GPUMaterial));
    std::memcpy(c.mats->contents(), s.materials.data(), s.materials.size() * sizeof(GPUMaterial));
    c.recs = ctx.buffer(std::max<size_t>(s.instances.size(), 1) * sizeof(InstRec));
    auto* r = static_cast<InstRec*>(c.recs->contents());
    for (u32 i = 0; i < s.instances.size(); ++i) {
        const auto& mi = s.scene.meshInfos()[s.instances[i].meshIndex];
        r[i] = {mi.vertexOffset, mi.indexOffset, s.instances[i].materialIndex, 0};
    }
    c.params = ctx.buffer(64);
    c.counters = ctx.buffer((2 + s.materials.size()) * 4 + 64);
    // Per-triangle UVs (primitive data), each mesh start aligned to 256 B.
    u64 total = 0;
    c.pdOffset.assign(meshes, 0);
    for (u32 m = 0; m < meshes; ++m) {
        c.pdOffset[m] = total;
        total += (u64(s.meshTriangles(m)) * sizeof(PrimUV) + 255) / 256 * 256;
    }
    c.pd = ctx.buffer(std::max<u64>(total, 256));
    auto* pd = static_cast<soc::u8*>(c.pd->contents());
    for (u32 m = 0; m < meshes; ++m) {
        const auto& mi = s.scene.meshInfos()[m];
        for (u32 t = 0; t < mi.indexCount / 3; ++t) {
            PrimUV p;
            for (u32 k = 0; k < 3; ++k) {
                const phosphor::GPUVertex& v = s.scene.vertices()[mi.vertexOffset + s.scene.indices()[mi.indexOffset + 3 * t + k]];
                p.uv[2 * k] = v.u; p.uv[2 * k + 1] = v.v;
            }
            std::memcpy(pd + c.pdOffset[m] + size_t(t) * sizeof(PrimUV), &p, sizeof p);
        }
    }
    ctx.commitResidency();
    c.soup = s.soup();
    c.bvh = std::make_unique<CpuBvh>(c.soup);
    std::set<u32> mm;
    for (u32 m = 0; m < s.materials.size(); ++m) c.maskedMaterials += s.materials[m].alphaCutoff > 0.0f;
    for (u32 t = 0; t < c.soup.count(); ++t) c.maskedTriangles += c.masked(c.soup.instance[t]);
}

// ---------------------------------------------------------------------------
// CPU alpha rule
// ---------------------------------------------------------------------------

struct Rule {
    const Corpus& c;
    /// 0 reject, 1 accept, 2 ambiguous (|alpha - cutoff| <= band).  `alpha` out (optional).
    int test(u32 tri, f64 u, f64 v, f64 lod, f64* alphaOut = nullptr) const {
        const u32 inst = c.soup.instance[tri];
        const GPUMaterial& m = c.s->materials[c.s->instances[inst].materialIndex];
        if (m.alphaCutoff <= 0.0f) return 1;
        const f64 a = alpha(tri, u, v, lod);
        if (alphaOut) *alphaOut = a;
        const bool textured = m.baseColorTex != INVALID_TEXTURE_INDEX;
        if (textured && std::fabs(a - m.alphaCutoff) <= kBand) return 2;
        return a >= f64(m.alphaCutoff) ? 1 : 0;
    }
    f64 alpha(u32 tri, f64 u, f64 v, f64 lod) const {
        const u32 inst = c.soup.instance[tri];
        const GPUMaterial& m = c.s->materials[c.s->instances[inst].materialIndex];
        f64 a = m.baseColor[3];
        if (m.baseColorTex != INVALID_TEXTURE_INDEX) {
            f64 uu, vv;
            uv(tri, u, v, uu, vv);
            a *= sampleA(c.mips[m.baseColorTex], uu, vv, lod);
        }
        return a;
    }
    void uv(u32 tri, f64 u, f64 v, f64& uu, f64& vv) const {
        const u32 inst = c.soup.instance[tri];
        const u32 mesh = c.s->instances[inst].meshIndex;
        const auto& mi = c.s->scene.meshInfos()[mesh];
        const u32 prim = c.soup.primitive[tri];
        f64 w[3] = {1.0 - u - v, u, v};
        uu = vv = 0;
        for (u32 k = 0; k < 3; ++k) {
            const phosphor::GPUVertex& vx = c.s->scene.vertices()[mi.vertexOffset + c.s->scene.indices()[mi.indexOffset + 3 * prim + k]];
            uu += w[k] * f64(vx.u);
            vv += w[k] * f64(vx.v);
        }
    }
    CpuBvh::Filter strict(f64 lod) const { return [this, lod](u32 tri, f64, f64 u, f64 v) { return test(tri, u, v, lod) == 1; }; }
    CpuBvh::Filter loose(f64 lod) const { return [this, lod](u32 tri, f64, f64 u, f64 v) { return test(tri, u, v, lod) != 0; }; }
    CpuBvh::Filter opaqueOnly() const {
        return [this](u32 tri, f64, f64, f64) { return !c.masked(c.soup.instance[tri]); };
    }
};

// ---------------------------------------------------------------------------
// Ray sets and CPU references
// ---------------------------------------------------------------------------

struct Cam {
    V3 eye{}, fwd{}, right{}, up{};
    f64 th = 0, aspect = 1, vfov = 60;
    u32 w = 0, h = 0;
    Cam() = default;
    Cam(V3 e, V3 target, f64 fovDeg, u32 w_, u32 h_) : eye(e), vfov(fovDeg), w(w_), h(h_) {
        fwd = normalize(target - e);
        right = normalize(cross(fwd, V3{0, 1, 0}));
        up = cross(right, fwd);
        th = std::tan(fovDeg * 0.5 * 3.14159265358979323846 / 180.0);
        aspect = f64(w_) / f64(h_);
    }
    [[nodiscard]] Ray pixel(f64 px, f64 py) const { // px, py in pixels (centres at +0.5)
        const f64 u = (px / w * 2.0 - 1.0) * th * aspect, v = (1.0 - py / h * 2.0) * th;
        return {eye, normalize(fwd + right * u + up * v), 0.0, 1e30};
    }
    [[nodiscard]] f64 pixelSpread() const { return 2.0 * th / f64(h); } // radians per pixel (small angle)
};

struct RaySet {
    std::string name;
    std::vector<Ray> rays;
    std::vector<f64> pathBase; // shadow: length of the primary path to the origin (cone LOD study)
    Cam cam;                   // primary: the camera (pixel i = y * w + x)
    bool shadow = false;
    // GPU buffers
    MTL::Buffer *rb = nullptr, *hb = nullptr;
    // references
    std::vector<CpuHit> lo, hi;       // nearest, ambiguous texels rejected / accepted (primary)
    std::vector<soc::u8> anyLo, anyHi, opaqueOcc; // shadow
    f64 refLod = 0;
};

void uploadRays(soc::Context& ctx, RaySet& s) {
    s.rb = ctx.buffer(std::max<size_t>(s.rays.size(), 1) * sizeof(GpuRay));
    s.hb = ctx.buffer(std::max<size_t>(s.rays.size(), 1) * sizeof(GpuHit));
    auto* r = static_cast<GpuRay*>(s.rb->contents());
    for (size_t i = 0; i < s.rays.size(); ++i) r[i] = toGpu(s.rays[i]);
}

void computeRefs(const Corpus& c, RaySet& s, f64 lod) {
    const Rule rule{c};
    s.refLod = lod;
    const u32 n = u32(s.rays.size());
    if (!s.shadow) {
        s.lo.assign(n, {});
        s.hi.assign(n, {});
        const auto fs = rule.strict(lod), fl = rule.loose(lod);
        parallelFor(n, [&](u32 i) { s.lo[i] = c.bvh->nearest(s.rays[i], fs); s.hi[i] = c.bvh->nearest(s.rays[i], fl); });
    } else {
        s.anyLo.assign(n, 0); s.anyHi.assign(n, 0); s.opaqueOcc.assign(n, 0);
        const auto fs = rule.strict(lod), fl = rule.loose(lod), fo = rule.opaqueOnly();
        parallelFor(n, [&](u32 i) {
            s.anyLo[i] = c.bvh->any(s.rays[i], fs);
            s.anyHi[i] = c.bvh->any(s.rays[i], fl);
            s.opaqueOcc[i] = c.bvh->any(s.rays[i], fo);
        });
    }
}

struct Cmp {
    u32 rays = 0, wrong = 0, ambiguous = 0, leaks = 0, extra = 0, tMismatch = 0, opaqueLeaks = 0;
    std::string first;
    [[nodiscard]] f64 disagreePct() const { return rays ? 100.0 * wrong / rays : 0.0; }
};

Cmp compare(const RaySet& s, const std::vector<GpuHit>& g) {
    Cmp c;
    c.rays = u32(s.rays.size());
    for (u32 i = 0; i < c.rays; ++i) {
        const bool gHit = g[i].t >= 0;
        bool bad = false;
        if (!s.shadow) {
            const CpuHit &lo = s.lo[i], &hi = s.hi[i];
            const bool amb = lo.hit() != hi.hit() || (lo.hit() && std::fabs(lo.t - hi.t) > 2e-4 * std::max(1.0, lo.t));
            if (amb) { ++c.ambiguous; continue; }
            if (gHit != lo.hit()) {
                bad = true;
                (gHit ? c.extra : c.leaks)++;
            } else if (gHit && std::fabs(lo.t - f64(g[i].t)) > 2e-4 * std::max(1.0, lo.t)) {
                bad = true;
                ++c.tMismatch;
            }
        } else {
            if (s.opaqueOcc[i] && !gHit) ++c.opaqueLeaks;
            if (s.anyLo[i] != s.anyHi[i]) { ++c.ambiguous; continue; }
            if (gHit != bool(s.anyLo[i])) {
                bad = true;
                (gHit ? c.extra : c.leaks)++;
            }
        }
        if (bad) {
            ++c.wrong;
            if (c.first.empty()) {
                char b[200];
                std::snprintf(b, sizeof b, "%s ray %u: cpu %s t %.5g, gpu %s t %.5g inst %u prim %u", s.name.c_str(), i,
                              s.shadow ? (s.anyLo[i] ? "blocked" : "free") : (s.lo[i].hit() ? "hit" : "miss"),
                              s.shadow ? 0.0 : s.lo[i].t, gHit ? "hit" : "miss", double(g[i].t), g[i].instance, g[i].primitive);
                c.first = b;
            }
        }
    }
    return c;
}

// ---------------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------------

enum class Geo { Masked, All, Opaque }; // which geometries are non-opaque

struct Spec {
    std::string name;
    Geo geo;
    bool slots;      // per-material table slots (B)
    bool primData;   // A_pd
    bool ift;        // false: no table (C)
    std::vector<std::string> fns, dbgFns;
    bool needsApple10;
};

const std::vector<Spec>& specs() {
    static const std::vector<Spec> v = {
        {"A", Geo::Masked, false, false, true, {"alpha_generic"}, {"alpha_generic_dbg"}, false},
        {"A_pd", Geo::Masked, false, true, true, {"alpha_pd"}, {"alpha_pd_dbg"}, false},
        {"A_all", Geo::All, false, false, true, {"alpha_generic"}, {"alpha_generic_dbg"}, false},
        {"B", Geo::Masked, true, false, true, {"alpha_tex", "alpha_const"}, {"alpha_tex_dbg", "alpha_const_dbg"}, true},
        {"B_all", Geo::All, true, false, true, {"alpha_tex", "alpha_const"}, {"alpha_tex_dbg", "alpha_const_dbg"}, true},
        {"C", Geo::Opaque, false, false, false, {}, {}, false},
    };
    return v;
}

struct Pso {
    MTL::ComputePipelineState* pso = nullptr;
    MTL::IntersectionFunctionTable* ift = nullptr;
};

struct Built {
    const Spec* spec = nullptr;
    std::vector<Blas> blases; // per used mesh
    std::vector<int> meshToBlas;
    Tlas tlas;
    Pso primary, shadow, dbg;
    double blasMs = 0;
};

MTL4::PrimitiveAccelerationStructureDescriptor* blasDesc(soc::Context& ctx, const Corpus& c, u32 mesh, bool opaque,
                                                         u32 iftOffset, bool primData) {
    const phosphor::GPUMeshInfo& mi = c.s->scene.meshInfos()[mesh];
    auto* geo = MTL4::AccelerationStructureTriangleGeometryDescriptor::alloc()->init();
    geo->setVertexBuffer(range(c.g.vertices, u64(mi.vertexOffset) * sizeof(phosphor::GPUVertex)));
    geo->setVertexFormat(MTL::AttributeFormatFloat3);
    geo->setVertexStride(sizeof(phosphor::GPUVertex));
    geo->setIndexBuffer(range(c.g.indices, u64(mi.indexOffset) * sizeof(u32), u64(mi.indexCount) * sizeof(u32)));
    geo->setIndexType(MTL::IndexTypeUInt32);
    geo->setTriangleCount(mi.indexCount / 3);
    geo->setOpaque(opaque);
    geo->setIntersectionFunctionTableOffset(iftOffset);
    if (primData) {
        geo->setPrimitiveDataBuffer(range(c.pd, c.pdOffset[mesh], u64(mi.indexCount / 3) * sizeof(PrimUV)));
        geo->setPrimitiveDataElementSize(sizeof(PrimUV));
        geo->setPrimitiveDataStride(sizeof(PrimUV));
    }
    ctx.keep(geo);
    auto* d = MTL4::PrimitiveAccelerationStructureDescriptor::alloc()->init();
    d->setGeometryDescriptors(NS::Array::array(geo));
    ctx.keep(d);
    return d;
}

Pso makePso(soc::Context& ctx, MTL::Library* lib, const Corpus& c, const std::string& kernel,
            const std::vector<std::string>& fns, const Spec& sp, std::string& err) {
    Pso p;
    p.pso = linkedPipeline(ctx, lib, kernel, fns);
    if (!sp.ift) return p;
    auto* d = MTL::IntersectionFunctionTableDescriptor::alloc()->init();
    d->setFunctionCount(sp.slots ? c.s->materials.size() : 1);
    p.ift = p.pso->newIntersectionFunctionTable(d);
    d->release();
    if (!p.ift) { err = "newIntersectionFunctionTable failed"; return p; }
    auto handle = [&](const std::string& n) { return p.pso->functionHandle(NS::String::string(n.c_str(), NS::UTF8StringEncoding)); };
    if (!sp.slots) {
        MTL::FunctionHandle* h = handle(fns[0]);
        if (!h) { err = "functionHandle(" + fns[0] + ") failed"; return p; }
        p.ift->setFunction(h, 0);
    } else {
        MTL::FunctionHandle* hTex = handle(fns[0]);
        MTL::FunctionHandle* hConst = handle(fns[1]);
        if (!hTex || !hConst) { err = "functionHandle failed for the B functions"; return p; }
        for (u32 m = 0; m < c.s->materials.size(); ++m) {
            const GPUMaterial& mat = c.s->materials[m];
            if (mat.alphaCutoff <= 0.0f)
                p.ift->setOpaqueTriangleIntersectionFunction(
                    MTL::IntersectionFunctionSignature(MTL::IntersectionFunctionSignatureTriangleData | MTL::IntersectionFunctionSignatureInstancing), m);
            else
                p.ift->setFunction(texturedMask(mat) ? hTex : hConst, m);
        }
    }
    // Function resources: the Metal 3 table-bound buffers.
    p.ift->setBuffer(c.mats, 0, 0);
    p.ift->setBuffer(c.texTable, 0, 1);
    p.ift->setBuffer(c.g.vertices, 0, 2);
    p.ift->setBuffer(c.g.indices, 0, 3);
    p.ift->setBuffer(c.recs, 0, 4);
    p.ift->setBuffer(c.params, 0, 5);
    p.ift->setBuffer(c.counters, 0, 6);
    ctx.adopt(p.ift);
    return p;
}

bool build(soc::Context& ctx, const Corpus& c, const Spec& sp, Built& b, std::string& err) {
    b.spec = &sp;
    const u32 meshes = c.s->scene.getMeshCount();
    b.meshToBlas.assign(meshes, -1);
    for (u32 m = 0; m < meshes; ++m) {
        if (c.meshMaterial[m] == ~0u) continue;
        const GPUMaterial& mat = c.s->materials[c.meshMaterial[m]];
        const bool masked = mat.alphaCutoff > 0.0f;
        const bool opaque = sp.geo == Geo::Opaque || (sp.geo == Geo::Masked && !masked);
        const u32 off = sp.slots ? c.meshMaterial[m] : 0;
        Blas bl;
        bl.desc = blasDesc(ctx, c, m, opaque, off, sp.primData && !opaque);
        bl.sizes = ctx.device()->accelerationStructureSizes(bl.desc);
        bl.as = newAccelerationStructure(ctx, bl.sizes.accelerationStructureSize, AsPlacement::Device);
        bl.triangles = c.s->meshTriangles(m);
        b.meshToBlas[m] = int(b.blases.size());
        b.blases.push_back(bl);
    }
    ctx.commitResidency();
    MTL::Buffer* scratch = scratchFor(ctx, b.blases);
    b.blasMs = buildBlases(ctx, b.blases, scratch);
    b.tlas = allocateTlas(ctx, u32(c.s->instances.size()), MTL::AccelerationStructureUsageNone, AsPlacement::Device);
    auto* d = static_cast<InstanceDesc*>(b.tlas.instances->contents());
    for (u32 i = 0; i < c.s->instances.size(); ++i)
        d[i] = toInstanceDesc(c.s->instances[i], b.blases[b.meshToBlas[c.s->instances[i].meshIndex]].as->gpuResourceID(), i, 0);
    timeEncoder(ctx, [&](MTL4::ComputeCommandEncoder* e) { e->buildAccelerationStructure(b.tlas.as, b.tlas.desc, range(b.tlas.scratch)); });
    MTL::Library* lib = f9Library(ctx, "s5_alpha.metal");
    const char* pk = sp.ift ? "primary_ift" : "primary_noift";
    const char* sk = sp.ift ? "shadow_ift" : "shadow_noift";
    b.primary = makePso(ctx, lib, c, pk, sp.fns, sp, err);
    b.shadow = makePso(ctx, lib, c, sk, sp.fns, sp, err);
    if (sp.ift) b.dbg = makePso(ctx, lib, c, "primary_ift", sp.dbgFns, sp, err);
    ctx.commitResidency();
    return err.empty();
}

/// One dispatch of `p` over `set`; returns the span in ms (ComputeTimer).
double dispatch(soc::Context& ctx, const Built& b, const Pso& p, RaySet& set, u32 count = 0) {
    const u32 n = count ? count : u32(set.rays.size());
    TraceParams tp{n, 0xFF, 0, 0};
    MTL::Buffer* pb = ctx.buffer(16);
    std::memcpy(pb->contents(), &tp, sizeof tp);
    soc::ComputeTimer t(ctx);
    MTL4::ComputeCommandEncoder* e = t.begin();
    ctx.table()->setResource(b.tlas.as->gpuResourceID(), 0);
    ctx.table()->setAddress(set.rb->gpuAddress(), 1);
    ctx.table()->setAddress(set.hb->gpuAddress(), 2);
    ctx.table()->setAddress(pb->gpuAddress(), 3);
    if (p.ift) ctx.table()->setResource(p.ift->gpuResourceID(), 4);
    e->setComputePipelineState(p.pso);
    e->setArgumentTable(ctx.table());
    e->dispatchThreads(MTL::Size::Make(n, 1, 1), MTL::Size::Make(64, 1, 1));
    t.lap();
    return t.finish()[0];
}

std::vector<GpuHit> readHits(const RaySet& set) {
    std::vector<GpuHit> h(set.rays.size());
    std::memcpy(h.data(), set.hb->contents(), h.size() * sizeof(GpuHit));
    return h;
}

void setLod(const Corpus& c, float lod) {
    AlphaParams p{lod, 0, 0, 0};
    std::memcpy(c.params->contents(), &p, sizeof p);
}

// ---------------------------------------------------------------------------
// Scenes
// ---------------------------------------------------------------------------

struct Synthetic {
    SceneData s;
    f64 x0[5] = {};
};

void makeSynthetic(SceneData& s) {
    s.name = "synthetic";
    auto addPlane = [&](f64 xa, f64 xb, f64 ya, f64 yb, f64 z, u32 n, u32 mat) {
        std::vector<glm::vec3> pos;
        std::vector<glm::vec2> uv;
        std::vector<u32> idx;
        for (u32 j = 0; j <= n; ++j)
            for (u32 i = 0; i <= n; ++i) {
                const f64 fx = f64(i) / n, fy = f64(j) / n;
                pos.push_back({float(xa + (xb - xa) * fx), float(ya + (yb - ya) * fy), float(z)});
                uv.push_back({float(fx), float(1.0 - fy)});
            }
        for (u32 j = 0; j < n; ++j)
            for (u32 i = 0; i < n; ++i) {
                const u32 a = j * (n + 1) + i, b = a + 1, cc = a + n + 1, d = cc + 1;
                for (u32 k : {a, b, cc, b, d, cc}) idx.push_back(k);
            }
        const phosphor::MeshHandle h = addMesh(s, pos, idx, uv);
        phosphor::GPUInstance inst{};
        const glm::mat4 id(1.0f);
        std::memcpy(inst.modelMatrix, glm::value_ptr(id), sizeof(inst.modelMatrix));
        inst.meshIndex = h;
        inst.materialIndex = mat;
        inst.flags = phosphor::INSTANCE_FLAG_VALID;
        inst.generation = u32(s.instances.size()) + 1;
        s.instances.push_back(inst);
    };
    auto material = [&](float cutoff, float alpha, u32 tex) {
        GPUMaterial m{};
        m.baseColor[0] = m.baseColor[1] = m.baseColor[2] = 1.0f;
        m.baseColor[3] = alpha;
        m.baseColorTex = tex;
        m.normalTex = m.metallicRoughnessTex = m.occlusionTex = m.emissiveTex = INVALID_TEXTURE_INDEX;
        m.alphaCutoff = cutoff;
        s.materials.push_back(m);
    };
    // Textures: 0 checker (256^2, 32-texel cells), 1 disc (512^2, radius 0.35 in UV).
    CpuTexture chk;
    chk.width = chk.height = 256; chk.sRGB = true;
    chk.rgba.resize(256 * 256 * 4);
    for (u32 y = 0; y < 256; ++y)
        for (u32 x = 0; x < 256; ++x) {
            soc::u8* p = &chk.rgba[(size_t(y) * 256 + x) * 4];
            const bool solid = (((x / 32) + (y / 32)) & 1) == 0;
            p[0] = soc::u8(x); p[1] = soc::u8(y); p[2] = 128; p[3] = solid ? 255 : 0;
        }
    CpuTexture disc;
    disc.width = disc.height = 512; disc.sRGB = true;
    disc.rgba.resize(512 * 512 * 4);
    for (u32 y = 0; y < 512; ++y)
        for (u32 x = 0; x < 512; ++x) {
            soc::u8* p = &disc.rgba[(size_t(y) * 512 + x) * 4];
            const f64 du = (x + 0.5) / 512 - 0.5, dv = (y + 0.5) / 512 - 0.5;
            p[0] = soc::u8(x / 2); p[1] = soc::u8(y / 2); p[2] = 64; p[3] = std::sqrt(du * du + dv * dv) < 0.35 ? 255 : 0;
        }
    s.textures = {chk, disc};
    material(0.0f, 1.0f, INVALID_TEXTURE_INDEX); // 0 opaque backdrop
    material(0.5f, 1.0f, 0);                     // 1 checker
    material(0.5f, 1.0f, 1);                     // 2 disc
    material(0.5f, 0.25f, INVALID_TEXTURE_INDEX); // 3 constant alpha below cutoff (always rejected)
    material(0.5f, 0.75f, INVALID_TEXTURE_INDEX); // 4 constant alpha above cutoff (always accepted)
    addPlane(0, 4, 0, 4, 0.0, 8, 1);
    addPlane(5, 9, 0, 4, 0.0, 8, 2);
    addPlane(10, 12, 0, 4, 0.0, 4, 3);
    addPlane(12, 14, 0, 4, 0.0, 4, 4);
    addPlane(-1, 15, -1, 5, -1.0, 4, 0); // backdrop
}

/// Analytic expectation of an orthogonal ray at (x, y): 1 front plane hit, 0 pass through to the
/// backdrop, -1 unknown (outside the front planes, near a hole boundary).
int analytic(f64 x, f64 y) {
    if (y < 0 || y > 4) return -1;
    auto nearEdge = [](f64 v, f64 period, f64 texels, f64 tex) { // distance to a multiple of `period` texels in texel units
        const f64 t = v * tex;
        const f64 d = std::fabs(t / period - std::round(t / period)) * period;
        return d < texels;
    };
    if (x >= 0 && x <= 4) { // checker, u = x/4, v = 1 - y/4
        const f64 u = x / 4, v = 1.0 - y / 4;
        if (u <= 0 || u >= 1 || v <= 0 || v >= 1) return -1;
        if (nearEdge(u, 32, 1.5, 256) || nearEdge(v, 32, 1.5, 256)) return -1;
        const int cx = int(u * 8), cy = int(v * 8);
        return ((cx + cy) & 1) == 0 ? 1 : 0;
    }
    if (x >= 5 && x <= 9) {
        const f64 u = (x - 5) / 4, v = 1.0 - y / 4;
        if (u <= 0 || u >= 1 || v <= 0 || v >= 1) return -1;
        const f64 r = std::sqrt((u - 0.5) * (u - 0.5) + (v - 0.5) * (v - 0.5));
        if (std::fabs(r - 0.35) * 512 < 2.5) return -1;
        return r < 0.35 ? 1 : 0;
    }
    if (x >= 10 && x < 12) return 0;
    if (x >= 12 && x <= 14) return 1;
    return -1;
}

// ---------------------------------------------------------------------------
// LOD rule study (CPU): raster-like derivative LOD vs LOD 0 vs ray cone LOD
// ---------------------------------------------------------------------------

struct LodStudy {
    u32 rays = 0, involved = 0;
    u32 changed0C = 0, changed0R = 0, changedCR = 0;
    u32 lod0KillsVsRaster = 0, lod0AddsVsRaster = 0, coneKillsVsRaster = 0, coneAddsVsRaster = 0;
    f64 meanLodRaster = 0, meanLodCone = 0;
    u64 lodSamples = 0;
    // shadow
    u32 shRays = 0, shBlockedOpaque = 0, shBlocked0 = 0, shBlockedCone = 0, shBlockedFine = 0, shChanged0C = 0, shMaskedOcc = 0;
};

bool changed(const CpuHit& a, const CpuHit& b) {
    if (a.hit() != b.hit()) return true;
    return a.hit() && std::fabs(a.t - b.t) > 1e-6 * std::max(1.0, a.t);
}

struct TriGeo {
    V3 p[3];
    f64 uv[3][2];
};

TriGeo triGeo(const Corpus& c, const Rule& rule, u32 tri) {
    TriGeo g;
    for (u32 k = 0; k < 3; ++k) g.p[k] = c.soup.v[3 * size_t(tri) + k];
    const u32 inst = c.soup.instance[tri];
    const u32 mesh = c.s->instances[inst].meshIndex;
    const auto& mi = c.s->scene.meshInfos()[mesh];
    const u32 prim = c.soup.primitive[tri];
    for (u32 k = 0; k < 3; ++k) {
        const phosphor::GPUVertex& vx = c.s->scene.vertices()[mi.vertexOffset + c.s->scene.indices()[mi.indexOffset + 3 * prim + k]];
        g.uv[k][0] = vx.u; g.uv[k][1] = vx.v;
    }
    (void)rule;
    return g;
}

/// UV at the intersection of ray `r` with the plane of the triangle (extrapolated).
void planeUV(const TriGeo& g, const Ray& r, f64& u, f64& v) {
    const V3 e1 = g.p[1] - g.p[0], e2 = g.p[2] - g.p[0];
    const V3 n = cross(e1, e2);
    const f64 den = dot(r.d, n);
    const f64 t = std::fabs(den) > 1e-300 ? dot(g.p[0] - r.o, n) / den : 0.0;
    const V3 q = (r.o + r.d * t) - g.p[0];
    const f64 d00 = dot(e1, e1), d01 = dot(e1, e2), d11 = dot(e2, e2), d20 = dot(q, e1), d21 = dot(q, e2);
    const f64 det = d00 * d11 - d01 * d01;
    const f64 b1 = det != 0 ? (d11 * d20 - d01 * d21) / det : 0, b2 = det != 0 ? (d00 * d21 - d01 * d20) / det : 0;
    const f64 b0 = 1 - b1 - b2;
    u = b0 * g.uv[0][0] + b1 * g.uv[1][0] + b2 * g.uv[2][0];
    v = b0 * g.uv[0][1] + b1 * g.uv[1][1] + b2 * g.uv[2][1];
}

f64 triArea(const TriGeo& g) { return 0.5 * length(cross(g.p[1] - g.p[0], g.p[2] - g.p[0])); }
f64 uvArea(const TriGeo& g) {
    const f64 ax = g.uv[1][0] - g.uv[0][0], ay = g.uv[1][1] - g.uv[0][1], bx = g.uv[2][0] - g.uv[0][0], by = g.uv[2][1] - g.uv[0][1];
    return 0.5 * std::fabs(ax * by - ay * bx);
}

/// Isotropic ray cone LOD: cone width at the hit over the world size of a texel of mip 0.
f64 coneLod(const Corpus& c, const Rule& rule, u32 tri, f64 pathLength, f64 spread) {
    const GPUMaterial& m = c.s->materials[c.s->instances[c.soup.instance[tri]].materialIndex];
    if (m.baseColorTex == INVALID_TEXTURE_INDEX) return 0;
    const CpuTexture& tx = c.s->textures[m.baseColorTex];
    const TriGeo g = triGeo(c, rule, tri);
    const f64 ua = uvArea(g), wa = triArea(g);
    if (ua <= 0 || wa <= 0) return 0;
    const f64 texelWorld = std::sqrt(wa / (ua * f64(tx.width) * f64(tx.height)));
    return std::max(0.0, std::log2(std::max(1e-12, pathLength * spread) / texelWorld));
}

/// Raster-like LOD from the UV derivatives of the neighbouring pixel rays (anisotropy 8, like the engine sampler).
f64 rasterLod(const Corpus& c, const Rule& rule, u32 tri, const Cam& cam, u32 px, u32 py, f64 u0, f64 v0) {
    const GPUMaterial& m = c.s->materials[c.s->instances[c.soup.instance[tri]].materialIndex];
    if (m.baseColorTex == INVALID_TEXTURE_INDEX) return 0;
    const CpuTexture& tx = c.s->textures[m.baseColorTex];
    const TriGeo g = triGeo(c, rule, tri);
    f64 ux, vx, uy, vy;
    planeUV(g, cam.pixel(px + 1.5, py + 0.5), ux, vx);
    planeUV(g, cam.pixel(px + 0.5, py + 1.5), uy, vy);
    const f64 dx = std::hypot((ux - u0) * tx.width, (vx - v0) * tx.height);
    const f64 dy = std::hypot((uy - u0) * tx.width, (vy - v0) * tx.height);
    const f64 pmax = std::max(dx, dy), pmin = std::min(dx, dy);
    if (pmax <= 0) return 0;
    const f64 n = std::min(std::ceil(pmax / std::max(pmin, 1e-12)), 8.0);
    return std::max(0.0, std::log2(pmax / n));
}

LodStudy lodStudy(const Corpus& c, const RaySet& prim, const RaySet& shadow) {
    const Rule rule{c};
    LodStudy st;
    const u32 n = u32(prim.rays.size());
    const f64 spread = prim.cam.pixelSpread();
    std::vector<CpuHit> hr(n), h0(n), hc(n);
    std::vector<f64> lodR(n, -1), lodC(n, -1);
    parallelFor(n, [&](u32 i) {
        const u32 px = i % prim.cam.w, py = i / prim.cam.w;
        const Ray& r = prim.rays[i];
        auto test0 = [&](u32 tri, f64, f64 u, f64 v) {
            if (!c.masked(c.soup.instance[tri])) return true;
            return rule.alpha(tri, u, v, 0) >= f64(c.s->materials[c.s->instances[c.soup.instance[tri]].materialIndex].alphaCutoff);
        };
        auto testC = [&](u32 tri, f64 t, f64 u, f64 v) {
            if (!c.masked(c.soup.instance[tri])) return true;
            const f64 lod = coneLod(c, rule, tri, t, spread);
            return rule.alpha(tri, u, v, lod) >= f64(c.s->materials[c.s->instances[c.soup.instance[tri]].materialIndex].alphaCutoff);
        };
        auto testR = [&](u32 tri, f64, f64 u, f64 v) {
            if (!c.masked(c.soup.instance[tri])) return true;
            f64 uu, vv;
            rule.uv(tri, u, v, uu, vv);
            const f64 lod = rasterLod(c, rule, tri, prim.cam, px, py, uu, vv);
            return rule.alpha(tri, u, v, lod) >= f64(c.s->materials[c.s->instances[c.soup.instance[tri]].materialIndex].alphaCutoff);
        };
        h0[i] = c.bvh->nearest(r, test0);
        hc[i] = c.bvh->nearest(r, testC);
        hr[i] = c.bvh->nearest(r, testR);
        if (hr[i].hit() && c.masked(c.soup.instance[hr[i].tri])) {
            f64 uu, vv;
            rule.uv(hr[i].tri, hr[i].u, hr[i].v, uu, vv);
            lodR[i] = rasterLod(c, rule, hr[i].tri, prim.cam, px, py, uu, vv);
            lodC[i] = coneLod(c, rule, hr[i].tri, hr[i].t, spread);
        }
    });
    st.rays = n;
    for (u32 i = 0; i < n; ++i) {
        const bool inv = (hr[i].hit() && c.masked(c.soup.instance[hr[i].tri])) || (h0[i].hit() && c.masked(c.soup.instance[h0[i].tri])) ||
                         (hc[i].hit() && c.masked(c.soup.instance[hc[i].tri]));
        if (!inv) continue;
        ++st.involved;
        st.changed0C += changed(h0[i], hc[i]);
        st.changed0R += changed(h0[i], hr[i]);
        st.changedCR += changed(hc[i], hr[i]);
        // direction: "kills" = the RT rule hits a masked surface that the raster-like rule passes through
        auto maskedHit = [&](const CpuHit& h) { return h.hit() && c.masked(c.soup.instance[h.tri]); };
        if (changed(h0[i], hr[i])) { (maskedHit(h0[i]) && !maskedHit(hr[i]) ? st.lod0KillsVsRaster : st.lod0AddsVsRaster)++; }
        if (changed(hc[i], hr[i])) { (maskedHit(hc[i]) && !maskedHit(hr[i]) ? st.coneKillsVsRaster : st.coneAddsVsRaster)++; }
        if (lodR[i] >= 0) { st.meanLodRaster += lodR[i]; st.meanLodCone += lodC[i]; ++st.lodSamples; }
    }
    if (st.lodSamples) { st.meanLodRaster /= f64(st.lodSamples); st.meanLodCone /= f64(st.lodSamples); }
    // Shadow rays: occluded fraction with opaque-only / LOD 0 / cone LOD / a finer LOD (LOD 0 minus nothing = same) occluders.
    const u32 ns = u32(shadow.rays.size());
    std::vector<soc::u8> bo(ns), b0(ns), bc(ns), mo(ns);
    parallelFor(ns, [&](u32 i) {
        const Ray& r = shadow.rays[i];
        const f64 base = shadow.pathBase[i];
        auto lodTest = [&](bool cone) {
            return [&, cone](u32 tri, f64 t, f64 u, f64 v) {
                if (!c.masked(c.soup.instance[tri])) return true;
                const f64 lod = cone ? coneLod(c, rule, tri, base + t, spread) : 0.0;
                return rule.alpha(tri, u, v, lod) >= f64(c.s->materials[c.s->instances[c.soup.instance[tri]].materialIndex].alphaCutoff);
            };
        };
        bo[i] = c.bvh->any(r, rule.opaqueOnly());
        b0[i] = c.bvh->any(r, lodTest(false));
        bc[i] = c.bvh->any(r, lodTest(true));
        mo[i] = b0[i] && !bo[i];
    });
    st.shRays = ns;
    for (u32 i = 0; i < ns; ++i) {
        st.shBlockedOpaque += bo[i];
        st.shBlocked0 += b0[i];
        st.shBlockedCone += bc[i];
        st.shChanged0C += b0[i] != bc[i];
        st.shMaskedOcc += mo[i];
    }
    return st;
}

// ---------------------------------------------------------------------------
// The benchmark
// ---------------------------------------------------------------------------

struct SceneResult {
    bool ok = true;
    std::string detail;
    bool controlFails = false; // C disagrees with the reference
    u32 cWrong = 0;
};

void runScene(soc::Context& ctx, soc::Report& rep, Corpus& c, std::vector<RaySet*> prim, std::vector<RaySet*> shad,
              const std::string& tag, SceneResult& res, std::function<void(const Built&, const Spec&, std::vector<std::string>&)> extra = {}) {
    for (const Spec& sp : specs()) {
        if (sp.needsApple10 && !ctx.apple10()) {
            rep.note(tag + "." + sp.name + ": skipped (needs the M5 per-material table path; ctx.apple10() is false, Apple9 uses A only)");
            continue;
        }
        Built b;
        std::string err;
        if (!build(ctx, c, sp, b, err)) {
            rep.note(tag + "." + sp.name + ": unavailable: " + err);
            res.ok = false;
            res.detail += tag + "." + sp.name + " build: " + err + "; ";
            continue;
        }
        const std::string pre = "alpha." + tag + "." + sp.name;
        setLod(c, 0.0f);
        u32 wrongAll = 0, amb = 0, leaks = 0, extraBlocks = 0, opaqueLeaks = 0, raysAll = 0;
        std::string first;
        for (int kind = 0; kind < 2; ++kind) {
            const auto& sets = kind == 0 ? prim : shad;
            Cmp tot;
            for (RaySet* set : sets) {
                std::memset(set->hb->contents(), 0xFF, set->hb->length());
                dispatch(ctx, b, kind == 0 ? b.primary : b.shadow, *set);
                const Cmp cm = compare(*set, readHits(*set));
                tot.rays += cm.rays; tot.wrong += cm.wrong; tot.ambiguous += cm.ambiguous; tot.leaks += cm.leaks;
                tot.extra += cm.extra; tot.opaqueLeaks += cm.opaqueLeaks;
                if (tot.first.empty()) tot.first = cm.first;
                if (set->refLod == 0 && sets.size() > 1 && kind == 0)
                    rep.value(pre + ".view." + set->name + ".wrong", "rays", cm.wrong, {{"rays", double(cm.rays)}, {"ambiguous", double(cm.ambiguous)}}, false);
            }
            const char* kn = kind == 0 ? "primary" : "shadow";
            rep.value(pre + "." + kn + ".wrong", "rays", tot.wrong,
                      {{"rays", double(tot.rays)}, {"ambiguous", double(tot.ambiguous)}, {"missed_occluder", double(tot.leaks)}, {"extra_hit", double(tot.extra)}}, false);
            if (kind == 1) rep.value(pre + ".shadow.opaque_leaks", "rays", tot.opaqueLeaks, {{"rays", double(tot.rays)}}, false);
            wrongAll += tot.wrong; amb += tot.ambiguous; leaks += tot.leaks; extraBlocks += tot.extra; raysAll += tot.rays;
            opaqueLeaks += tot.opaqueLeaks;
            if (first.empty()) first = tot.first;
        }
        rep.value(pre + ".wrong", "rays", wrongAll, {{"rays", double(raysAll)}, {"ambiguous", double(amb)}}, false);
        rep.value(pre + ".disagree_pct", "%", raysAll ? 100.0 * wrongAll / raysAll : 0.0, {{"rays", double(raysAll)}}, false);
        if (sp.name == "C") {
            res.cWrong = wrongAll;
            res.controlFails = wrongAll > 0;
            ctx.log("[S5 %s] C (all opaque): %u of %u rays disagree with the alpha reference (control, must be > 0)", tag.c_str(), wrongAll, raysAll);
        } else {
            ctx.log("[S5 %s] %s: %u wrong / %u rays (%u ambiguous, %u missed occluders, %u extra hits, %u opaque leaks)%s%s", tag.c_str(),
                    sp.name.c_str(), wrongAll, raysAll, amb, leaks, extraBlocks, opaqueLeaks, first.empty() ? "" : " first: ", first.c_str());
            if (wrongAll || opaqueLeaks) {
                res.ok = false;
                res.detail += tag + "." + sp.name + ": " + std::to_string(wrongAll) + " wrong, " + std::to_string(opaqueLeaks) + " opaque leaks; ";
            }
        }
        // Timing (ns per ray), the primary set (largest) and the shadow set.
        if (!prim.empty() && !shad.empty()) {
            ctx.keepWarm(30);
            RaySet* tp = prim[0];
            RaySet* ts = shad[0];
            const double np = double(tp->rays.size()), ns = double(ts->rays.size());
            const soc::Stats sp1 = ctx.measure([&] { return dispatch(ctx, b, b.primary, *tp); });
            rep.metric(pre + ".primary.ns", "ns", scaled(sp1, 1e6 / np), {{"rays", np}}, false);
            ctx.keepWarm(30);
            const soc::Stats sp2 = ctx.measure([&] { return dispatch(ctx, b, b.shadow, *ts); });
            rep.metric(pre + ".shadow.ns", "ns", scaled(sp2, 1e6 / ns), {{"rays", ns}}, false);
        }
        // Debug variant: intersection function invocations (primary + shadow).
        if (sp.ift) {
            for (int kind = 0; kind < 2; ++kind) {
                const auto& sets = kind == 0 ? prim : shad;
                std::memset(c.counters->contents(), 0, c.counters->length());
                u64 rays = 0;
                for (RaySet* set : sets) {
                    dispatch(ctx, b, b.dbg, *set);
                    rays += set->rays.size();
                }
                // The dbg PSO is the primary kernel; run the shadow sets through a shadow-kernel dbg too.
                const auto* cnt = static_cast<const u32*>(c.counters->contents());
                if (kind == 1) { /* shadow rays traced closest-hit by the dbg kernel: invocation count is an upper bound */ }
                const char* kn = kind == 0 ? "primary" : "shadow_closest";
                rep.value(pre + "." + kn + ".calls_per_ray", "calls", rays ? double(cnt[0]) / double(rays) : 0.0, {{"rays", double(rays)}}, false);
                rep.value(pre + "." + kn + ".opaque_calls", "calls", cnt[1], {{"calls", double(cnt[0])}}, false);
                if (sp.geo == Geo::Masked && cnt[1] != 0) {
                    res.ok = false;
                    res.detail += tag + "." + sp.name + ": function called on an opaque material (" + std::to_string(cnt[1]) + "); ";
                }
                if (kind == 0 && sp.name == "A") {
                    std::string per;
                    for (u32 m = 0; m < c.s->materials.size(); ++m)
                        if (cnt[2 + m]) per += " mat" + std::to_string(m) + "=" + std::to_string(cnt[2 + m]);
                    ctx.log("[S5 %s] A calls per material (primary):%s", tag.c_str(), per.c_str());
                }
            }
        }
        if (extra) {
            std::vector<std::string> notes;
            extra(b, sp, notes);
            for (auto& n : notes) rep.note(n);
        }
        ctx.keepWarm(20);
    }
}

void alphaBench(soc::Context& ctx, soc::Report& rep) {
    SceneData sponza;
    std::string err;
    const bool haveSponza = loadSponza(sponza, err);
    if (!haveSponza) rep.note("Sponza skipped: " + err);
    SceneResult sr, yr;

    // ------------------------------------------------------------ synthetic
    SceneData syn;
    makeSynthetic(syn);
    Corpus sc;
    prepareCorpus(ctx, sc, syn, "synthetic");
    std::vector<RaySet> synSets(3);
    {
        const u32 n = ctx.quick() ? 20000 : 100000;
        RaySet& orth = synSets[0];
        orth.name = "orthogonal";
        std::vector<int> expect;
        for (u32 i = 0; i < n; ++i) {
            const f64 x = rnd01(i, 1) * 14.0, y = rnd01(i, 2) * 4.0;
            orth.rays.push_back({{x, y, 5.0}, {0, 0, -1}, 0.0, 1e30});
        }
        RaySet& obl = synSets[1];
        obl.name = "oblique";
        for (u32 i = 0; i < n; ++i) {
            const f64 x = rnd01(i, 11) * 14.0, y = rnd01(i, 12) * 4.0;
            obl.rays.push_back({{x, y, 5.0}, normalize(V3{(rnd01(i, 13) - 0.5) * 0.8, (rnd01(i, 14) - 0.5) * 0.8, -1.0}), 0.0, 1e30});
        }
        RaySet& sh = synSets[2];
        sh.name = "shadow";
        sh.shadow = true;
        for (u32 i = 0; i < n; ++i) { // from the backdrop towards the front planes (+z), tmax past the front planes only
            const f64 x = rnd01(i, 21) * 14.0, y = rnd01(i, 22) * 4.0;
            sh.rays.push_back({{x, y, -1.0}, {0, 0, 1}, 1e-3, 3.0});
            sh.pathBase.push_back(0);
        }
        for (RaySet& r : synSets) { uploadRays(ctx, r); computeRefs(sc, r, 0); }
    }
    runScene(ctx, rep, sc, {&synSets[0], &synSets[1]}, {&synSets[2]}, "synthetic", yr);
    // Analytic check (independent of the CPU reference code) on the orthogonal rays, per strategy.
    {
        u32 wrongA = 0, wrongC = 0, checked = 0;
        for (const Spec& sp : specs()) {
            if (sp.needsApple10 && !ctx.apple10()) continue;
            Built b;
            std::string e;
            if (!build(ctx, sc, sp, b, e)) continue;
            setLod(sc, 0.0f);
            RaySet& o = synSets[0];
            dispatch(ctx, b, b.primary, o);
            const auto h = readHits(o);
            u32 wrong = 0, n = 0;
            for (size_t i = 0; i < o.rays.size(); ++i) {
                const int ex = analytic(o.rays[i].o.x, o.rays[i].o.y);
                if (ex < 0) continue;
                ++n;
                const bool front = h[i].t >= 0 && std::fabs(h[i].t - 5.0f) < 1e-3f;
                const bool back = h[i].t >= 0 && std::fabs(h[i].t - 6.0f) < 1e-3f;
                if ((ex == 1 && !front) || (ex == 0 && !back)) ++wrong;
            }
            rep.value("alpha.synthetic." + sp.name + ".analytic_wrong", "rays", wrong, {{"checked", double(n)}}, false);
            if (sp.name == "C") wrongC = wrong; else wrongA += wrong;
            checked = n;
        }
        ctx.log("[S5 synthetic] analytic expectation (known holes) on %u orthogonal rays: strategies %u wrong total, C %u wrong (control)", checked, wrongA, wrongC);
        if (wrongA) { yr.ok = false; yr.detail += "analytic mismatch " + std::to_string(wrongA) + "; "; }
        if (wrongC == 0) yr.controlFails = false;
    }
    // Non-zero LOD: the explicit LOD argument works (checker/disc at LOD 2 against the CPU at LOD 2).
    {
        Built b;
        std::string e;
        const Spec& sp = specs()[0];
        if (build(ctx, sc, sp, b, e)) {
            u32 wrong = 0, amb = 0, n = 0;
            for (int k = 0; k < 2; ++k) {
                RaySet& o = synSets[k];
                computeRefs(sc, o, 2.0);
                setLod(sc, 2.0f);
                dispatch(ctx, b, b.primary, o);
                const Cmp cm = compare(o, readHits(o));
                wrong += cm.wrong; amb += cm.ambiguous; n += cm.rays;
            }
            setLod(sc, 0.0f);
            for (int k = 0; k < 2; ++k) computeRefs(sc, synSets[k], 0);
            rep.value("alpha.synthetic.A.lod2.wrong", "rays", wrong, {{"rays", double(n)}, {"ambiguous", double(amb)}}, false);
            ctx.log("[S5 synthetic] A at explicit LOD 2: %u wrong / %u rays (%u ambiguous)", wrong, n, amb);
            if (wrong) { yr.ok = false; yr.detail += "lod2 " + std::to_string(wrong) + " wrong; "; }
        }
    }

    // --------------------------------------------------------------- Sponza
    double shiftWrongPct = -1;
    LodStudy ls;
    bool haveLod = false;
    if (haveSponza) {
        Corpus pc;
        prepareCorpus(ctx, pc, sponza, "sponza");
        // Corpus listing.
        {
            std::string txt;
            for (u32 m = 0; m < sponza.materials.size(); ++m) {
                const GPUMaterial& mat = sponza.materials[m];
                if (mat.alphaCutoff <= 0) continue;
                u32 ni = 0, nm = 0;
                u64 tris = 0;
                std::set<u32> meshes;
                for (const auto& i : sponza.instances)
                    if (i.materialIndex == m) { ++ni; tris += sponza.meshTriangles(i.meshIndex); meshes.insert(i.meshIndex); }
                nm = u32(meshes.size());
                const CpuTexture* t = mat.baseColorTex < sponza.textures.size() ? &sponza.textures[mat.baseColorTex] : nullptr;
                char b[200];
                std::snprintf(b, sizeof b, " [material %u cutoff %.2f baseColor.a %.2f tex %u %ux%u: %u instances, %u meshes, %llu triangles]", m, mat.alphaCutoff,
                              mat.baseColor[3], mat.baseColorTex, t ? t->width : 0, t ? t->height : 0, ni, nm, (unsigned long long)tris);
                txt += b;
                rep.value("alpha.sponza.material." + std::to_string(m) + ".triangles", "triangles", double(tris), {{"cutoff", mat.alphaCutoff}, {"instances", ni}, {"meshes", nm}}, false);
            }
            rep.value("alpha.sponza.masked_materials", "materials", pc.maskedMaterials, {{"materials", double(sponza.materials.size())}}, false);
            rep.value("alpha.sponza.masked_triangles", "triangles", double(pc.maskedTriangles), {{"triangles", double(pc.soup.count())}}, false);
            rep.note("Sponza alpha-masked materials:" + txt);
            ctx.log("[S5 sponza] %u masked materials, %llu of %u triangles masked:%s", pc.maskedMaterials, (unsigned long long)pc.maskedTriangles, pc.soup.count(), txt.c_str());
        }
        const u32 w = ctx.quick() ? 256 : 480, h = ctx.quick() ? 144 : 270;
        struct View { const char* name; V3 eye, target; f64 fov; };
        const View views[] = {{"hall", {-12, 2, 0.3}, {8, 3, -0.3}, 70.0},
                              {"plantA", {1.0, 1.6, 4.0}, {4.0, 1.1, 1.6}, 60.0},
                              {"plantB", {-8.0, 1.6, 3.5}, {-5.0, 1.1, 1.1}, 60.0},
                              {"leaves", {2.0, 1.5, 1.0}, {3.9, 1.4, -1.75}, 60.0}};
        std::vector<RaySet> prim(4), shad(4);
        for (u32 v = 0; v < 4; ++v) {
            RaySet& ps = prim[v];
            ps.name = views[v].name;
            ps.cam = Cam(views[v].eye, views[v].target, views[v].fov, w, h);
            for (u32 y = 0; y < h; ++y)
                for (u32 x = 0; x < w; ++x) ps.rays.push_back(ps.cam.pixel(x + 0.5, y + 0.5));
            uploadRays(ctx, ps);
            computeRefs(pc, ps, 0);
            // Shadow rays from the primary hits (every 2nd pixel): half towards the sun, half random upward, short.
            RaySet& ss = shad[v];
            ss.name = std::string(views[v].name) + ".shadow";
            ss.shadow = true;
            u32 masked = 0, hits = 0;
            for (u32 i = 0; i < ps.rays.size(); ++i) {
                const CpuHit& hh = ps.lo[i];
                if (hh.hit()) { ++hits; masked += pc.masked(pc.soup.instance[hh.tri]); }
                if (!hh.hit() || (i & 1)) continue;
                const Ray& r = ps.rays[i];
                const V3 o = r.o + r.d * hh.t;
                V3 d = normalize(V3{0.35, 1.0, 0.2});
                f64 tmax = 1e30;
                if ((i >> 1) & 1) {
                    d = normalize(V3{(rnd01(i, 31) - 0.5) * 2, 0.3 + rnd01(i, 32), (rnd01(i, 33) - 0.5) * 2});
                    tmax = 6.0;
                }
                ss.rays.push_back({o, d, 2e-3, tmax});
                ss.pathBase.push_back(hh.t);
            }
            uploadRays(ctx, ss);
            computeRefs(pc, ss, 0);
            rep.value(std::string("alpha.sponza.view.") + views[v].name + ".masked_hit_pct", "%", hits ? 100.0 * masked / hits : 0.0, {{"hits", double(hits)}, {"rays", double(ps.rays.size())}}, false);
            ctx.log("[S5 sponza] view %s: %zu rays, %.1f%% of the hits on alpha-masked materials, %zu shadow rays", views[v].name, ps.rays.size(), hits ? 100.0 * masked / hits : 0.0, ss.rays.size());
        }
        // Moving the cutoff by +0.25 must make the CPU reference disagree with the (correct) GPU result.
        runScene(ctx, rep, pc, {&prim[0], &prim[1], &prim[2], &prim[3]}, {&shad[0], &shad[1], &shad[2], &shad[3]}, "sponza", sr);
        {
            // Resolution control of the reference check: same GPU hits (A), CPU filter with a shifted cutoff.
            Built b;
            std::string e;
            if (build(ctx, pc, specs()[0], b, e)) {
                setLod(pc, 0.0f);
                u32 wrong = 0, n = 0;
                for (RaySet& ps : prim) {
                    dispatch(ctx, b, b.primary, ps);
                    const auto g = readHits(ps);
                    for (u32 i = 0; i < ps.rays.size(); ++i) {
                        const bool gh = g[i].t >= 0;
                        const CpuHit sh = pc.bvh->nearest(ps.rays[i], [&](u32 tri, f64, f64 u, f64 v) {
                            if (!pc.masked(pc.soup.instance[tri])) return true;
                            return Rule{pc}.alpha(tri, u, v, 0) >= 0.75;
                        });
                        ++n;
                        wrong += (sh.hit() != gh) || (gh && std::fabs(sh.t - g[i].t) > 2e-4 * std::max(1.0, sh.t));
                    }
                }
                shiftWrongPct = n ? 100.0 * wrong / n : 0;
                rep.value("alpha.sponza.A.shifted_cutoff_disagree_pct", "%", shiftWrongPct, {{"rays", double(n)}}, false);
                ctx.log("[S5 sponza] reference resolution control: CPU cutoff 0.75 instead of 0.5 vs GPU A: %.4f%% of primary rays differ (must be > 0)", shiftWrongPct);
            }
        }
        // Explicit LOD on Sponza: one view at LOD 2 against the CPU at LOD 2.
        {
            Built b;
            std::string e;
            if (build(ctx, pc, specs()[0], b, e)) {
                RaySet& ps = prim[1];
                computeRefs(pc, ps, 2.0);
                setLod(pc, 2.0f);
                dispatch(ctx, b, b.primary, ps);
                const Cmp cm = compare(ps, readHits(ps));
                setLod(pc, 0.0f);
                computeRefs(pc, ps, 0.0);
                rep.value("alpha.sponza.A.lod2.wrong", "rays", cm.wrong, {{"rays", double(cm.rays)}, {"ambiguous", double(cm.ambiguous)}}, false);
                ctx.log("[S5 sponza] A at explicit LOD 2 (view plantA): %u wrong / %u rays (%u ambiguous)", cm.wrong, cm.rays, cm.ambiguous);
                if (cm.wrong) { sr.ok = false; sr.detail += "lod2 " + std::to_string(cm.wrong) + " wrong; "; }
            }
        }
        // LOD rule study over the plant views (CPU).
        for (u32 v = 1; v < 4; ++v) {
            const LodStudy s1 = lodStudy(pc, prim[v], shad[v]);
            ls.rays += s1.rays; ls.involved += s1.involved; ls.changed0C += s1.changed0C; ls.changed0R += s1.changed0R; ls.changedCR += s1.changedCR;
            ls.lod0KillsVsRaster += s1.lod0KillsVsRaster; ls.lod0AddsVsRaster += s1.lod0AddsVsRaster;
            ls.coneKillsVsRaster += s1.coneKillsVsRaster; ls.coneAddsVsRaster += s1.coneAddsVsRaster;
            ls.meanLodRaster += s1.meanLodRaster * s1.lodSamples; ls.meanLodCone += s1.meanLodCone * s1.lodSamples; ls.lodSamples += s1.lodSamples;
            ls.shRays += s1.shRays; ls.shBlockedOpaque += s1.shBlockedOpaque; ls.shBlocked0 += s1.shBlocked0; ls.shBlockedCone += s1.shBlockedCone;
            ls.shChanged0C += s1.shChanged0C; ls.shMaskedOcc += s1.shMaskedOcc;
            const f64 inv = s1.involved ? 100.0 / s1.involved : 0;
            rep.value(std::string("alpha.lod_mismatch_pct.") + views[v].name, "%", s1.changed0C * inv, {{"involved_rays", double(s1.involved)}, {"rays", double(s1.rays)}}, false);
        }
        if (ls.lodSamples) { ls.meanLodRaster /= f64(ls.lodSamples); ls.meanLodCone /= f64(ls.lodSamples); }
        haveLod = true;
        const f64 inv = ls.involved ? 100.0 / ls.involved : 0;
        rep.value("alpha.lod_mismatch_pct", "%", ls.changed0C * inv, {{"involved_rays", double(ls.involved)}, {"rays", double(ls.rays)}, {"changed", double(ls.changed0C)}}, false);
        rep.value("alpha.lod0_vs_raster_pct", "%", ls.changed0R * inv, {{"involved_rays", double(ls.involved)}, {"kills", double(ls.lod0KillsVsRaster)}, {"adds", double(ls.lod0AddsVsRaster)}}, false);
        rep.value("alpha.cone_vs_raster_pct", "%", ls.changedCR * inv, {{"involved_rays", double(ls.involved)}, {"kills", double(ls.coneKillsVsRaster)}, {"adds", double(ls.coneAddsVsRaster)}}, false);
        rep.value("alpha.lod.mean_raster", "lod", ls.meanLodRaster, {{"samples", double(ls.lodSamples)}}, false);
        rep.value("alpha.lod.mean_cone", "lod", ls.meanLodCone, {{"samples", double(ls.lodSamples)}}, false);
        const f64 sinv = ls.shRays ? 100.0 / ls.shRays : 0;
        rep.value("alpha.shadow.blocked_pct.opaque_only", "%", ls.shBlockedOpaque * sinv, {{"rays", double(ls.shRays)}}, false);
        rep.value("alpha.shadow.blocked_pct.lod0", "%", ls.shBlocked0 * sinv, {{"rays", double(ls.shRays)}}, false);
        rep.value("alpha.shadow.blocked_pct.cone", "%", ls.shBlockedCone * sinv, {{"rays", double(ls.shRays)}}, false);
        rep.value("alpha.shadow.lod_mismatch_pct", "%", ls.shChanged0C * sinv, {{"rays", double(ls.shRays)}, {"changed", double(ls.shChanged0C)}}, false);
        rep.value("alpha.shadow.masked_occluded_pct", "%", ls.shMaskedOcc * sinv, {{"rays", double(ls.shRays)}}, false);
        ctx.log("[S5 sponza] LOD study (plant views, %u rays, %u touching masked surfaces): LOD0 vs cone differ on %u (%.2f%%); LOD0 vs raster-like %u (%.2f%%: RT hits what raster discards %u, RT passes what raster draws %u); "
                "cone vs raster-like %u (%.2f%%: %u / %u); mean LOD raster %.2f cone %.2f",
                ls.rays, ls.involved, ls.changed0C, ls.changed0C * inv, ls.changed0R, ls.changed0R * inv, ls.lod0KillsVsRaster, ls.lod0AddsVsRaster, ls.changedCR,
                ls.changedCR * inv, ls.coneKillsVsRaster, ls.coneAddsVsRaster, ls.meanLodRaster, ls.meanLodCone);
        ctx.log("[S5 sponza] shadow rays %u: blocked opaque-only %.2f%%, with masked occluders at LOD0 %.2f%%, at cone LOD %.2f%%; LOD0 vs cone differ on %.2f%% (%u rays)",
                ls.shRays, ls.shBlockedOpaque * sinv, ls.shBlocked0 * sinv, ls.shBlockedCone * sinv, ls.shChanged0C * sinv, ls.shChanged0C);
    }

    // ------------------------------------------------------------- verdict
    if (!ctx.apple10())
        rep.note("--force-family apple9 / Apple9 device: only the portable strategies run (A, A_pd, A_all, C); B and B_all (per-material table slots, M5 hardware table indexing) are skipped. "
                 "This verifies the fallback path of the software choice; the M5 hardware still performs the table indexing.");
    rep.note("Intersection function resources are bound with MTL::IntersectionFunctionTable::setBuffer (Metal 3 style) and read through [[buffer(n)]] in the functions; textures are a bindless "
             "buffer of MTL::ResourceID. CPU mips are built with a 2x2 box filter and uploaded level by level (the engine uses generateMipmaps: same LOD 0, mip contents may differ slightly).");
    const bool ok = sr.ok && yr.ok;
    const bool controls = yr.controlFails && (!haveSponza || (sr.controlFails && shiftWrongPct > 0));
    if (!ok) rep.status(soc::Status::Failed, "alpha strategies disagree with the CPU reference: " + sr.detail + yr.detail);
    rep.negative(ok && controls,
                 std::string("strategies exact vs the CPU alpha reference: ") + (ok ? "yes" : "NO (" + sr.detail + yr.detail + ")") + "; control C (all opaque) disagrees on " +
                     std::to_string(yr.cWrong) + " synthetic rays" + (haveSponza ? " and " + std::to_string(sr.cWrong) + " Sponza rays" : "") + " (must be > 0)" +
                     (haveSponza ? "; shifted-cutoff reference disagrees on " + std::to_string(shiftWrongPct) + "% of Sponza rays (must be > 0)" : ""));
    (void)haveLod;
}

} // namespace

SOC_BENCH("F9-S5", "alpha_rt", "Alpha test in RT: intersection function strategies (generic / per-material / opaque) vs the CPU alpha rule, LOD coherence with the raster", alphaBench);

} // namespace f9
