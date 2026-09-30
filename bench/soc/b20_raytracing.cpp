// B-20: ray tracing throughput (Grays/s).  Scene: a ~1M-triangle displaced
// terrain BLAS + 1000 small rock instances (one 200-triangle BLAS) in a TLAS.
// Rays: primary coherent (one per pixel of a 2048^2 camera) and incoherent
// (random origins, random lower-hemisphere directions); API `intersector`
// vs `intersection_query`; opaque geometry vs an alpha-test-like rejection
// (intersection function table for the intersector, the same test inline in
// the query loop).  A sample of the rays (dumped by the shader) is re-traced
// on the CPU (brute-force nearest hit over every triangle) and the GPU hit
// distance and instance/primitive id must match.
//
// Serves S-RT-1..3 of docs/APPLE_SOC_PLAYBOOK.md.

#include "harness.h"

#include <Metal/MTL4AccelerationStructure.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <thread>

namespace soc {
namespace {

using f64 = double;
struct V3 { f64 x, y, z; };
V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
f64 dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

f64 rayTri(V3 o, V3 d, V3 a, V3 b, V3 c) {
    const V3 e1 = b - a, e2 = c - a, p = cross(d, e2);
    const f64 det = dot(e1, p);
    if (std::fabs(det) < 1e-14) return -1;
    const f64 inv = 1.0 / det;
    const V3 s = o - a;
    const f64 u = dot(s, p) * inv;
    if (u < 0 || u > 1) return -1;
    const V3 q = cross(s, e1);
    const f64 v = dot(d, q) * inv;
    if (v < 0 || u + v > 1) return -1;
    const f64 t = dot(e2, q) * inv;
    return t > 0 ? t : -1;
}

u32 mix(u32 h) { h ^= h >> 16; h *= 0x7FEB352Du; h ^= h >> 15; h *= 0x846CA68Bu; h ^= h >> 16; return h; }
float rnd01(u32 a, u32 b) { return float(mix(a * 0x9E3779B1u ^ mix(b + 0x85EBCA6Bu)) >> 8) * (1.0f / 16777216.0f); }
bool alphaPass(u32 prim) { return ((prim * 2654435761u) >> 30) != 0u; }

struct Mesh {
    std::vector<float> v;
    std::vector<u32> idx;
    u32 tris() const { return u32(idx.size() / 3); }
    V3 vert(u32 i) const { return {v[3 * size_t(i)], v[3 * size_t(i) + 1], v[3 * size_t(i) + 2]}; }
};

Mesh makeGrid(u32 g, float size, float x0, float z0, float amp, float jitter, u32 seed) {
    Mesh m;
    m.v.resize(size_t(g + 1) * (g + 1) * 3);
    for (u32 j = 0; j <= g; ++j)
        for (u32 i = 0; i <= g; ++i) {
            const float x = x0 + size * float(i) / float(g), z = z0 + size * float(j) / float(g);
            const float y = amp * (std::sin(x * 0.02f) + std::cos(z * 0.017f)) + jitter * (rnd01(i + j * (g + 1), seed) - 0.5f);
            float* p = &m.v[(size_t(j) * (g + 1) + i) * 3];
            p[0] = x; p[1] = y; p[2] = z;
        }
    m.idx.reserve(size_t(g) * g * 6);
    for (u32 j = 0; j < g; ++j)
        for (u32 i = 0; i < g; ++i) {
            const u32 a = j * (g + 1) + i, b = a + 1, c = a + (g + 1), d = c + 1;
            for (u32 k : {a, c, b, b, c, d}) m.idx.push_back(k);
        }
    return m;
}

template <typename F> void parallelFor(u32 n, F&& fn) {
    const u32 nt = std::max(1u, std::min(n, std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (u32 t = 0; t < nt; ++t)
        th.emplace_back([&, t] { for (u32 i = t; i < n; i += nt) fn(i); });
    for (auto& x : th) x.join();
}

MTL4::BufferRange range(MTL::Buffer* b) { return MTL4::BufferRange::Make(b->gpuAddress(), b->length()); }

struct Blas {
    Mesh mesh;
    MTL::Buffer *vb = nullptr, *ib = nullptr;
    MTL::AccelerationStructure* as = nullptr;
};

template <typename F> double timeEnc(Context& ctx, F&& fn) {
    CommandTimer t(ctx);
    MTL4::CommandBuffer* cmd = t.begin();
    MTL4::ComputeCommandEncoder* e = cmd->computeCommandEncoder();
    fn(e);
    e->endEncoding();
    return t.finish();
}

Blas buildBlas(Context& ctx, Mesh&& m) {
    Blas b;
    b.mesh = std::move(m);
    b.vb = ctx.buffer(b.mesh.v.size() * 4);
    b.ib = ctx.buffer(b.mesh.idx.size() * 4);
    std::memcpy(b.vb->contents(), b.mesh.v.data(), b.mesh.v.size() * 4);
    std::memcpy(b.ib->contents(), b.mesh.idx.data(), b.mesh.idx.size() * 4);
    auto* g = MTL4::AccelerationStructureTriangleGeometryDescriptor::alloc()->init();
    g->setVertexBuffer(range(b.vb));
    g->setVertexFormat(MTL::AttributeFormatFloat3);
    g->setVertexStride(12);
    g->setIndexBuffer(range(b.ib));
    g->setIndexType(MTL::IndexTypeUInt32);
    g->setTriangleCount(b.mesh.tris());
    g->setOpaque(true);
    ctx.keep(g);
    auto* d = MTL4::PrimitiveAccelerationStructureDescriptor::alloc()->init();
    d->setGeometryDescriptors(NS::Array::array(g));
    ctx.keep(d);
    const MTL::AccelerationStructureSizes sz = ctx.device()->accelerationStructureSizes(d);
    b.as = ctx.device()->newAccelerationStructure(sz.accelerationStructureSize);
    if (!b.as) throw BenchError("newAccelerationStructure failed");
    ctx.adopt(b.as);
    MTL::Buffer* scratch = ctx.buffer(std::max<size_t>(sz.buildScratchBufferSize, 4096));
    ctx.commitResidency();
    timeEnc(ctx, [&](MTL4::ComputeCommandEncoder* e) { e->buildAccelerationStructure(b.as, d, range(scratch)); });
    return b;
}

#pragma pack(push, 1)
struct InstDesc {
    float m[12];
    u32 options, mask, iftOffset, userID;
    MTL::ResourceID blasID;
};
#pragma pack(pop)
static_assert(sizeof(InstDesc) == sizeof(MTL::IndirectAccelerationStructureInstanceDescriptor));

struct RtParams {
    float camPos[4], camFwd[4], camRight[4], camUp[4];
    u32 width, height, kind, seed, dumpStride, pad0, pad1, pad2;
    float bounds[4];
};
static_assert(sizeof(RtParams) == 112);

struct Inst { float s, x, y, z; };

float terrainY(float x, float z) { return 30.0f * (std::sin(x * 0.02f) + std::cos(z * 0.017f)); }

struct Scene {
    Blas terrain, rock;
    std::vector<Inst> rocks; // instances 1..R (instance 0 is the terrain, identity)
    MTL::AccelerationStructure *tlasOpaque = nullptr, *tlasNonOpaque = nullptr;
};

MTL::AccelerationStructure* buildTlas(Context& ctx, const Scene& sc, u32 options) {
    const u32 n = u32(sc.rocks.size()) + 1;
    MTL::Buffer* ib = ctx.buffer(size_t(n) * sizeof(InstDesc));
    auto* d = static_cast<InstDesc*>(ib->contents());
    for (u32 i = 0; i < n; ++i) {
        Inst s = i == 0 ? Inst{1, 0, 0, 0} : sc.rocks[i - 1];
        const float m[12] = {s.s, 0, 0, 0, s.s, 0, 0, 0, s.s, s.x, s.y, s.z};
        std::memcpy(d[i].m, m, sizeof(m));
        d[i].options = options; d[i].mask = 0xFF; d[i].iftOffset = 0; d[i].userID = i;
        d[i].blasID = (i == 0 ? sc.terrain.as : sc.rock.as)->gpuResourceID();
    }
    auto* td = MTL4::InstanceAccelerationStructureDescriptor::alloc()->init();
    td->setInstanceCount(n);
    td->setInstanceDescriptorBuffer(range(ib));
    td->setInstanceDescriptorStride(sizeof(InstDesc));
    td->setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorTypeIndirect);
    td->setInstanceTransformationMatrixLayout(MTL::MatrixLayoutColumnMajor);
    ctx.keep(td);
    const MTL::AccelerationStructureSizes sz = ctx.device()->accelerationStructureSizes(td);
    MTL::AccelerationStructure* tlas = ctx.device()->newAccelerationStructure(sz.accelerationStructureSize);
    if (!tlas) throw BenchError("newAccelerationStructure(TLAS) failed");
    ctx.adopt(tlas);
    MTL::Buffer* scratch = ctx.buffer(std::max<size_t>(sz.buildScratchBufferSize, 4096));
    ctx.commitResidency();
    timeEnc(ctx, [&](MTL4::ComputeCommandEncoder* e) { e->buildAccelerationStructure(tlas, td, range(scratch)); });
    return tlas;
}

Stats toGrays(const Stats& s, double rays) { // ms stats -> Grays/s stats
    const double k = rays * 1e-6;
    Stats r = s;
    r.median = k / s.median; r.min = k / s.max; r.max = k / s.min;
    r.p10 = k / s.p90; r.p90 = k / s.p10; r.mean = k / s.mean;
    return r;
}

struct Variant { bool coherent, query, alpha; };
std::string variantName(const Variant& v) {
    return std::string(v.coherent ? "coherent" : "incoherent") + "." + (v.query ? "query" : "intersector") + "." + (v.alpha ? "if" : "noif");
}

// Pipeline with a statically linked intersection function + the table holding it.
struct IfPipe { MTL::ComputePipelineState* pso = nullptr; MTL::IntersectionFunctionTable* ift = nullptr; std::string err; };
IfPipe makeIfPipeline(Context& ctx, MTL::Library* lib) {
    IfPipe r;
    auto* alpha = ctx.function(lib, "alpha_test");
    auto* sl = MTL4::StaticLinkingDescriptor::alloc()->init();
    sl->setFunctionDescriptors(NS::Array::array(alpha));
    ctx.keep(sl);
    auto* d = MTL4::ComputePipelineDescriptor::alloc()->init();
    d->setComputeFunctionDescriptor(ctx.function(lib, "rays_isect_if"));
    d->setStaticLinkingDescriptor(sl);
    NS::Error* err = nullptr;
    r.pso = ctx.compiler()->newComputePipelineState(d, nullptr, &err);
    d->release();
    if (!r.pso) { r.err = err ? err->localizedDescription()->utf8String() : "unknown"; return r; }
    ctx.keep(r.pso);
    auto* id = MTL::IntersectionFunctionTableDescriptor::alloc()->init();
    id->setFunctionCount(1);
    r.ift = r.pso->newIntersectionFunctionTable(id);
    id->release();
    if (!r.ift) { r.err = "newIntersectionFunctionTable failed"; return r; }
    MTL::FunctionHandle* h = r.pso->functionHandle(NS::String::string("alpha_test", NS::UTF8StringEncoding));
    if (!h) { r.err = "functionHandle(alpha_test) failed"; return r; }
    r.ift->setFunction(h, 0);
    ctx.adopt(r.ift);
    ctx.commitResidency();
    return r;
}

void benchRays(Context& ctx, Report& rep) {
    constexpr u32 kW = 2048, kH = 2048;
    constexpr u32 kRays = kW * kH;
    const u32 stride = ctx.quick() ? 32768 : 16384;
    constexpr float kSize = 1000.0f;

    // --- scene ---------------------------------------------------------------
    Scene sc;
    sc.terrain = buildBlas(ctx, makeGrid(708, kSize, 0, 0, 30.0f, 0.6f * kSize / 708.0f, 1)); // 1,002,528 triangles
    {
        Mesh rock = makeGrid(10, 2.0f, -1.0f, -1.0f, 0.0f, 0.0f, 99);
        for (size_t i = 0; i < rock.v.size(); i += 3) rock.v[i + 1] = 0.6f * std::cos(rock.v[i] * 1.5f) * std::cos(rock.v[i + 2] * 1.5f);
        sc.rock = buildBlas(ctx, std::move(rock));
    }
    constexpr u32 kRocks = 1000;
    for (u32 i = 0; i < kRocks; ++i) {
        Inst s;
        s.s = 2.0f + 4.0f * rnd01(i, 21);
        s.x = rnd01(i, 22) * kSize;
        s.z = rnd01(i, 23) * kSize;
        s.y = terrainY(s.x, s.z) + 0.5f;
        sc.rocks.push_back(s);
    }
    sc.tlasOpaque = buildTlas(ctx, sc, MTL::AccelerationStructureInstanceOptionOpaque);
    sc.tlasNonOpaque = buildTlas(ctx, sc, MTL::AccelerationStructureInstanceOptionNonOpaque);
    const u32 totalTris = sc.terrain.mesh.tris() + kRocks * sc.rock.mesh.tris();

    // --- pipelines -------------------------------------------------------------
    MTL::Library* lib = ctx.library("b20_rays.metal");
    MTL::ComputePipelineState* psoIsect = ctx.compute(lib, "rays_isect");
    MTL::ComputePipelineState* psoQuery = ctx.compute(lib, "rays_query");
    MTL::ComputePipelineState* psoQueryIf = ctx.compute(lib, "rays_query_if");
    IfPipe ifp = makeIfPipeline(ctx, lib);
    if (!ifp.ift) rep.note("intersection-function pipeline unavailable: " + ifp.err);

    // --- buffers ----------------------------------------------------------------
    const u32 maxRays = kRays * 2;
    MTL::Buffer* out = ctx.buffer(size_t(maxRays) * 8);
    MTL::Buffer* dump = ctx.buffer(size_t(maxRays / stride + 1) * 32);
    MTL::Buffer* params = ctx.buffer(256);

    RtParams base{};
    {
        const V3 pos{500, 260, -100}, target{500, 0, 400};
        V3 f = target - pos;
        const f64 fl = std::sqrt(dot(f, f));
        f = {f.x / fl, f.y / fl, f.z / fl};
        V3 r = cross(f, V3{0, 1, 0});
        const f64 rl = std::sqrt(dot(r, r));
        r = {r.x / rl, r.y / rl, r.z / rl};
        const V3 u = cross(r, f);
        const f64 th = std::tan(30.0 * M_PI / 180.0);
        auto put = [](float* d, V3 v, f64 k) { d[0] = float(v.x * k); d[1] = float(v.y * k); d[2] = float(v.z * k); d[3] = 0; };
        put(base.camPos, pos, 1); put(base.camFwd, f, 1); put(base.camRight, r, th); put(base.camUp, u, th);
        base.width = kW; base.height = kH; base.seed = 0x1234u; base.dumpStride = stride;
        base.bounds[0] = 0; base.bounds[1] = 0; base.bounds[2] = kSize; base.bounds[3] = 150.0f;
    }

    auto run = [&](const Variant& v, u32 mult) -> double {
        RtParams p = base;
        p.kind = v.coherent ? 0 : 1;
        std::memcpy(params->contents(), &p, sizeof(p));
        MTL::AccelerationStructure* tlas = v.alpha ? sc.tlasNonOpaque : sc.tlasOpaque;
        MTL::ComputePipelineState* pso = v.query ? (v.alpha ? psoQueryIf : psoQuery) : (v.alpha ? ifp.pso : psoIsect);
        ComputeTimer t(ctx);
        MTL4::ComputeCommandEncoder* e = t.begin();
        ctx.table()->setResource(tlas->gpuResourceID(), 0);
        ctx.table()->setAddress(out->gpuAddress(), 1);
        ctx.table()->setAddress(params->gpuAddress(), 2);
        if (!v.query && v.alpha) ctx.table()->setResource(ifp.ift->gpuResourceID(), 3);
        ctx.table()->setAddress(dump->gpuAddress(), 4);
        e->setComputePipelineState(pso);
        e->setArgumentTable(ctx.table());
        e->dispatchThreads(MTL::Size::Make(kW, kH * mult, 1), MTL::Size::Make(8, 8, 1));
        t.lap();
        return t.finish()[0];
    };

    // --- CPU reference ----------------------------------------------------------
    const Mesh& tm = sc.terrain.mesh;
    const Mesh& rm = sc.rock.mesh;
    struct Hit { f64 t = -1; u32 prim = 0, inst = 0; };
    struct Ray { V3 o, d; };
    auto triT = [&](const Ray& r, u32 inst, u32 prim) -> f64 {
        if (inst == 0) {
            if (prim >= tm.tris()) return -1;
            return rayTri(r.o, r.d, tm.vert(tm.idx[3 * prim]), tm.vert(tm.idx[3 * prim + 1]), tm.vert(tm.idx[3 * prim + 2]));
        }
        if (inst > kRocks || prim >= rm.tris()) return -1;
        const Inst& s = sc.rocks[inst - 1];
        auto w = [&](u32 vi) { const V3 v = rm.vert(rm.idx[3 * prim + vi]); return V3{v.x * s.s + s.x, v.y * s.s + s.y, v.z * s.s + s.z}; };
        return rayTri(r.o, r.d, w(0), w(1), w(2));
    };
    auto nearest = [&](const Ray& r, bool alpha) {
        Hit h;
        for (u32 p = 0; p < tm.tris(); ++p) {
            if (alpha && !alphaPass(p)) continue;
            const f64 t = triT(r, 0, p);
            if (t > 0 && (h.t < 0 || t < h.t)) { h.t = t; h.prim = p; h.inst = 0; }
        }
        for (u32 i = 1; i <= kRocks; ++i) {
            const Inst& s = sc.rocks[i - 1];
            const V3 c{s.x, s.y, s.z};
            const V3 oc = c - r.o;
            const f64 tc = dot(oc, r.d);
            const f64 d2 = dot(oc, oc) - tc * tc;
            const f64 rad = 1.8 * s.s;
            if (d2 > rad * rad) continue;
            for (u32 p = 0; p < rm.tris(); ++p) {
                if (alpha && !alphaPass(p)) continue;
                const f64 t = triT(r, i, p);
                if (t > 0 && (h.t < 0 || t < h.t)) { h.t = t; h.prim = p; h.inst = i; }
            }
        }
        return h;
    };
    // Returns the number of mismatching sampled rays (and the hit fraction).
    auto verify = [&](const Variant& v, double& hitRate) -> u32 {
        const u32 n = kRays / stride;
        std::vector<u32> bad(n, 0), hits(n, 0);
        const auto* dp = static_cast<const float*>(dump->contents());
        const auto* op = static_cast<const u32*>(out->contents());
        parallelFor(n, [&](u32 s) {
            const Ray r{{dp[8 * s], dp[8 * s + 1], dp[8 * s + 2]}, {dp[8 * s + 4], dp[8 * s + 5], dp[8 * s + 6]}};
            const u32 i = s * stride;
            const u32 tb = op[2 * i], id = op[2 * i + 1];
            const Hit c = nearest(r, v.alpha);
            const bool gMiss = tb == 0xFFFFFFFFu;
            if (c.t < 0 || gMiss) { bad[s] = (c.t < 0) != gMiss; return; }
            hits[s] = 1;
            float tg;
            std::memcpy(&tg, &tb, 4);
            const f64 tol = 5e-5 * std::max(1.0, c.t) + 1e-4;
            if (std::fabs(c.t - tg) > tol) { bad[s] = 1; return; }
            const u32 inst = id >> 20, prim = id & 0xFFFFFu;
            const f64 t2 = triT(r, inst, prim);
            if (t2 < 0 || std::fabs(t2 - tg) > tol || (v.alpha && !alphaPass(prim))) bad[s] = 1;
        });
        u32 nb = 0, nh = 0;
        for (u32 s = 0; s < n; ++s) { nb += bad[s]; nh += hits[s]; }
        hitRate = double(nh) / double(n);
        return nb;
    };

    // --- measurements -----------------------------------------------------------
    bool allValid = true;
    std::string detail;
    std::map<std::string, Stats> ms;
    for (const bool coherent : {true, false})
        for (const bool query : {false, true})
            for (const bool alpha : {false, true}) {
                const Variant v{coherent, query, alpha};
                const std::string name = variantName(v);
                if (!query && alpha && !ifp.ift) { rep.note(name + " skipped (no intersection-function pipeline)"); allValid = false; continue; }
                std::memset(out->contents(), 0xFF, size_t(kRays) * 8);
                run(v, 1);
                double hr = 0;
                const u32 bad = verify(v, hr);
                const u32 nSamples = kRays / stride;
                if (bad) { allValid = false; detail += name + ": " + std::to_string(bad) + "/" + std::to_string(nSamples) + " rays differ from the CPU; "; }
                if (hr < 0.3) { allValid = false; detail += name + ": hit rate " + std::to_string(hr) + " too low; "; }
                ctx.keepWarm(30);
                const Stats s = ctx.measure([&] { return run(v, 1); });
                ms[name] = s;
                rep.metric("rays." + name, "Grays/s", toGrays(s, kRays),
                           {{"rays", double(kRays)}, {"ms", s.median}, {"tris", double(totalTris)}, {"instances", double(kRocks + 1)},
                            {"hit_rate", hr}, {"rays_checked", double(nSamples)}, {"rays_wrong", double(bad)}});
                ctx.keepWarm(20);
            }
    // Required names: intersector, no intersection function.
    for (const bool coherent : {true, false}) {
        const std::string name = variantName({coherent, false, false});
        if (!ms.count(name)) continue;
        rep.metric(coherent ? "rays.coherent" : "rays.incoherent", "Grays/s", toGrays(ms[name], kRays),
                   {{"rays", double(kRays)}, {"ms", ms[name].median}, {"tris", double(totalTris)}});
    }

    // --- controls ------------------------------------------------------------------
    std::string ctl;
    bool linOk = true;
    for (const bool coherent : {true, false}) {
        const Variant v{coherent, false, false};
        double ratio = 0;
        for (int attempt = 0; attempt < 3; ++attempt) { // other GPU clients add noise
            ctx.keepWarm(30);
            const Stats a = ctx.measure([&] { return run(v, 1); });
            ctx.keepWarm(30);
            const Stats b = ctx.measure([&] { return run(v, 2); });
            ratio = b.median / a.median;
            if (ratio > 1.8 && ratio < 2.2) break;
        }
        rep.value(std::string("rays.") + (coherent ? "coherent" : "incoherent") + ".scaling_2x", "ratio", ratio, {{"rays", double(kRays)}});
        ctl += std::string(coherent ? "coherent" : "incoherent") + " 2x rays->" + std::to_string(ratio).substr(0, 4) + "x; ";
        linOk &= ratio > 1.8 && ratio < 2.2;
    }
    const std::string cn = variantName({true, false, false}), inn = variantName({false, false, false});
    const bool cohFaster = ms.count(cn) && ms.count(inn) && ms[cn].median * 1.15 < ms[inn].median;
    if (ms.count(cn) && ms.count(inn)) rep.value("rays.coherent_vs_incoherent", "ratio", ms[inn].median / ms[cn].median, {{"rays", double(kRays)}});
    rep.note("scene: " + std::to_string(sc.terrain.mesh.tris()) + "-triangle terrain BLAS + " + std::to_string(kRocks) + " instances of a " +
             std::to_string(rm.tris()) + "-triangle BLAS in one TLAS; 2048x2048 = 4.19 M rays per run; coherent = pinhole camera (one ray per pixel), "
             "incoherent = random origin/direction; alpha-test = 25% of the primitives rejected (intersection function table for the intersector, "
             "inline in the query loop); the CPU re-traces " + std::to_string(kRays / stride) + " dumped rays per variant. "
             "Apple10-only RT features (hardware instance transforms, IFB indexing) cannot be toggled: the same path runs on Apple9");
    rep.negative(allValid && linOk && cohFaster,
                 std::string(allValid ? "sampled rays match the CPU in every variant; " : "VALIDATION FAILED: " + detail) + ctl +
                     (cohFaster ? "coherent >15% faster than incoherent (" : "coherent NOT >15% faster than incoherent (") +
                     std::to_string(ms.count(cn) && ms.count(inn) ? ms[inn].median / ms[cn].median : 0.0).substr(0, 4) + "x)");
    if (!allValid) rep.status(Status::Failed, "validation failed: " + detail);
}

} // namespace

SOC_BENCH("B-20", "raytracing", "Ray throughput: coherent vs incoherent, intersector vs intersection_query, with/without intersection function", benchRays);

} // namespace soc
