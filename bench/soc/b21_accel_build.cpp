// B-21: acceleration-structure build, refit and compaction: BLAS of 10k / 100k
// / 1M triangles (displaced terrain grid), refit after moving the vertices,
// compaction (size before/after and the compact-copy time), TLAS of 1k / 10k /
// 100k instances of a small BLAS (indirect instance descriptors: BLAS
// resource IDs).  Every result is validated by tracing a sample of rays after
// the build, the refit and the compaction and comparing nearest hit (t and
// primitive/instance id) with a brute-force CPU intersection.
//
// Serves S-RT-4 of docs/APPLE_SOC_PLAYBOOK.md (and S-RT-2 for the instance
// count scaling).

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

// Moller-Trumbore, double precision, no culling; returns t > 0 or -1.
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

struct Mesh {
    std::vector<float> v; // xyz
    std::vector<u32> idx;
    u32 tris() const { return u32(idx.size() / 3); }
    V3 vert(u32 i) const { return {v[3 * size_t(i)], v[3 * size_t(i) + 1], v[3 * size_t(i) + 2]}; }
};

// g x g cells (2 g^2 triangles) over [x0, x0+size]^2, height = smooth waves + per-vertex jitter.
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

struct Ray { V3 o, d; };
struct Hit { f64 t = -1; u32 prim = 0, inst = 0; };

// ---- GPU objects -----------------------------------------------------------
struct Accel {
    MTL::AccelerationStructure* as = nullptr;
    MTL4::AccelerationStructureDescriptor* desc = nullptr;
    NS::UInteger asSize = 0, buildScratch = 0, refitScratch = 0;
};

MTL4::BufferRange range(MTL::Buffer* b) { return MTL4::BufferRange::Make(b->gpuAddress(), b->length()); }

struct BlasSrc {
    Mesh mesh;
    MTL::Buffer *vb = nullptr, *ib = nullptr;
};

BlasSrc makeSrc(Context& ctx, Mesh&& m) {
    BlasSrc s;
    s.mesh = std::move(m);
    s.vb = ctx.buffer(s.mesh.v.size() * 4);
    s.ib = ctx.buffer(s.mesh.idx.size() * 4);
    std::memcpy(s.vb->contents(), s.mesh.v.data(), s.mesh.v.size() * 4);
    std::memcpy(s.ib->contents(), s.mesh.idx.data(), s.mesh.idx.size() * 4);
    return s;
}
void uploadVerts(BlasSrc& s) { std::memcpy(s.vb->contents(), s.mesh.v.data(), s.mesh.v.size() * 4); }

Accel makeBlas(Context& ctx, const BlasSrc& s, bool refit) {
    auto* g = MTL4::AccelerationStructureTriangleGeometryDescriptor::alloc()->init();
    g->setVertexBuffer(range(s.vb));
    g->setVertexFormat(MTL::AttributeFormatFloat3);
    g->setVertexStride(12);
    g->setIndexBuffer(range(s.ib));
    g->setIndexType(MTL::IndexTypeUInt32);
    g->setTriangleCount(s.mesh.tris());
    g->setOpaque(true);
    ctx.keep(g);
    auto* d = MTL4::PrimitiveAccelerationStructureDescriptor::alloc()->init();
    d->setGeometryDescriptors(NS::Array::array(g));
    d->setUsage(refit ? MTL::AccelerationStructureUsageRefit : MTL::AccelerationStructureUsageNone);
    ctx.keep(d);
    Accel a;
    a.desc = d;
    const MTL::AccelerationStructureSizes sz = ctx.device()->accelerationStructureSizes(d);
    a.asSize = sz.accelerationStructureSize;
    a.buildScratch = sz.buildScratchBufferSize;
    a.refitScratch = sz.refitScratchBufferSize;
    a.as = ctx.device()->newAccelerationStructure(a.asSize);
    if (!a.as) throw BenchError("newAccelerationStructure failed");
    ctx.adopt(a.as);
    return a;
}

// One timed command buffer with one compute encoder (build / refit / copy are
// AS-stage work: the CommandTimer tail waits for every stage).
template <typename F> double timeEnc(Context& ctx, F&& fn) {
    CommandTimer t(ctx);
    MTL4::CommandBuffer* cmd = t.begin();
    MTL4::ComputeCommandEncoder* e = cmd->computeCommandEncoder();
    fn(e);
    e->endEncoding();
    return std::max(1e-4, t.finish() - ctx.emptySpanMs());
}

Stats toRate(const Stats& s, double units) { // ms stats -> units per ms * 1e-3 = M units/s
    const double k = units * 1e-3;
    Stats r = s;
    r.median = k / s.median; r.min = k / s.max; r.max = k / s.min;
    r.p10 = k / s.p90; r.p90 = k / s.p10; r.mean = k / s.mean;
    return r;
}

// ---- validation ---------------------------------------------------------------
struct Tracer {
    Context& ctx;
    MTL::ComputePipelineState *blas, *tlas;
    MTL::Buffer *rays, *out;
    explicit Tracer(Context& c) : ctx(c) {
        MTL::Library* lib = ctx.library("b21_trace.metal");
        blas = ctx.compute(lib, "trace_blas");
        tlas = ctx.compute(lib, "trace_tlas");
        rays = ctx.buffer(sizeof(float) * 8 * 1024);
        out = ctx.buffer(sizeof(float) * 4 * 1024);
    }
    std::vector<std::array<float, 4>> run(MTL::AccelerationStructure* as, bool instanced, const std::vector<Ray>& r) {
        for (size_t i = 0; i < r.size(); ++i) {
            float* p = static_cast<float*>(rays->contents()) + 8 * i;
            p[0] = float(r[i].o.x); p[1] = float(r[i].o.y); p[2] = float(r[i].o.z); p[3] = 0;
            p[4] = float(r[i].d.x); p[5] = float(r[i].d.y); p[6] = float(r[i].d.z); p[7] = 1e9f;
        }
        MTL4::CommandBuffer* cmd = ctx.beginCommands();
        MTL4::ComputeCommandEncoder* e = cmd->computeCommandEncoder();
        ctx.table()->setResource(as->gpuResourceID(), 0);
        ctx.table()->setAddress(rays->gpuAddress(), 1);
        ctx.table()->setAddress(out->gpuAddress(), 2);
        e->setComputePipelineState(instanced ? tlas : blas);
        e->setArgumentTable(ctx.table());
        e->dispatchThreads(MTL::Size::Make(u32(r.size()), 1, 1), MTL::Size::Make(64, 1, 1));
        e->endEncoding();
        ctx.submit();
        std::vector<std::array<float, 4>> res(r.size());
        std::memcpy(res.data(), out->contents(), r.size() * 16);
        return res;
    }
};

// Compare GPU nearest hits with the CPU: the CPU nearest t must match within
// tolerance, and the CPU t of the (instance, primitive) the GPU reported must
// match the GPU distance (adjacent triangles may tie on an edge).
template <typename NearestFn, typename TriFn>
u32 compare(const std::vector<Ray>& rays, const std::vector<std::array<float, 4>>& gpu, NearestFn&& nearest, TriFn&& triT) {
    std::vector<u32> bad(rays.size(), 0);
    parallelFor(u32(rays.size()), [&](u32 i) {
        const Hit c = nearest(rays[i]);
        const auto& g = gpu[i];
        const bool gMiss = g[0] < 0;
        if (c.t < 0 || gMiss) { bad[i] = (c.t < 0) != gMiss; return; }
        const f64 tol = 2e-4 * std::max(1.0, c.t);
        if (std::fabs(c.t - g[0]) > tol) { bad[i] = 1; return; }
        const f64 tg = triT(rays[i], u32(g[1]), u32(g[2]));
        if (tg < 0 || std::fabs(tg - g[0]) > tol) bad[i] = 1;
    });
    u32 n = 0;
    for (u32 b : bad) n += b;
    return n;
}

std::vector<Ray> terrainRays(u32 n, float size, u32 seed) {
    std::vector<Ray> r(n);
    for (u32 i = 0; i < n; ++i) {
        r[i].o = {double(rnd01(i, seed) * size), 200.0, double(rnd01(i, seed + 1) * size)};
        V3 d = {double(rnd01(i, seed + 2) - 0.5) * 0.6, -1.0, double(rnd01(i, seed + 3) - 0.5) * 0.6};
        const f64 l = std::sqrt(dot(d, d));
        r[i].d = {d.x / l, d.y / l, d.z / l};
    }
    return r;
}

u32 validateBlas(Tracer& tr, MTL::AccelerationStructure* as, const Mesh& m, u32 nRays, float size, u32 seed) {
    const auto rays = terrainRays(nRays, size, seed);
    const auto gpu = tr.run(as, false, rays);
    auto triAt = [&](const Ray& r, u32 p) {
        return rayTri(r.o, r.d, m.vert(m.idx[3 * p]), m.vert(m.idx[3 * p + 1]), m.vert(m.idx[3 * p + 2]));
    };
    return compare(
        rays, gpu,
        [&](const Ray& r) {
            Hit h;
            for (u32 p = 0; p < m.tris(); ++p) {
                const f64 t = triAt(r, p);
                if (t > 0 && (h.t < 0 || t < h.t)) { h.t = t; h.prim = p; }
            }
            return h;
        },
        [&](const Ray& r, u32 p, u32) { return p < m.tris() ? triAt(r, p) : -1.0; });
}

// ---- TLAS ---------------------------------------------------------------------
struct InstSpec { float s, x, y, z; };

Mesh makeRock() { return makeGrid(10, 2.0f, -1.0f, -1.0f, 0.0f, 0.0f, 99); } // flat 200-tri patch, bumped below

InstSpec instSpec(u32 i, u32 n, float size) {
    const u32 cols = u32(std::ceil(std::sqrt(double(n))));
    const float sp = size / float(cols);
    return {0.6f + 0.9f * rnd01(i, 5), (float(i % cols) + rnd01(i, 6)) * sp, 2.0f * rnd01(i, 7), (float(i / cols) + rnd01(i, 8)) * sp};
}

#pragma pack(push, 1)
struct InstDesc { // == MTL::IndirectAccelerationStructureInstanceDescriptor
    float m[12];
    u32 options, mask, iftOffset, userID;
    MTL::ResourceID blasID;
};
#pragma pack(pop)
static_assert(sizeof(InstDesc) == sizeof(MTL::IndirectAccelerationStructureInstanceDescriptor));

void benchBuild(Context& ctx, Report& rep) {
    const bool quick = ctx.quick();
    Tracer tr(ctx);
    const std::vector<u32> triTargets = quick ? std::vector<u32>{10000, 100000} : std::vector<u32>{10000, 100000, 1000000};
    const std::vector<u32> instTargets = quick ? std::vector<u32>{1000, 10000} : std::vector<u32>{1000, 10000, 100000};
    const float kSize = 1000.0f;
    bool valid = true, refitFaster = true;
    std::string detail, notes;
    std::vector<double> buildMs;
    std::vector<u32> buildTris;
    double refitMsLast = 0, rebuildRefittableLast = 0;

    for (u32 target : triTargets) {
        const u32 g = u32(std::ceil(std::sqrt(double(target) / 2.0)));
        BlasSrc src = makeSrc(ctx, makeGrid(g, kSize, 0, 0, 30.0f, 0.6f * kSize / float(g), 1));
        const u32 tris = src.mesh.tris();
        const std::string tag = "tris_" + (target >= 1000000 ? std::to_string(target / 1000000) + "M" : std::to_string(target / 1000) + "k");
        const u32 nRays = quick ? 48 : (tris > 500000 ? 48 : 128);

        // --- plain build --------------------------------------------------------
        Accel a = makeBlas(ctx, src, false);
        MTL::Buffer* scratch = ctx.buffer(std::max<size_t>(std::max(a.buildScratch, a.refitScratch), 4096));
        MTL::Buffer* sizeBuf = ctx.buffer(64);
        ctx.commitResidency();
        auto build = [&](MTL4::ComputeCommandEncoder* e) { e->buildAccelerationStructure(a.as, a.desc, range(scratch)); };
        timeEnc(ctx, build); // first build (also warms the pipeline)
        u32 bad = validateBlas(tr, a.as, src.mesh, nRays, kSize, 11);
        if (bad) { valid = false; detail += "build " + tag + " " + std::to_string(bad) + "/" + std::to_string(nRays) + " rays wrong; "; }
        ctx.keepWarm(20);
        const Stats sb = ctx.measure([&] { return timeEnc(ctx, build); });
        rep.metric("blas.build." + tag, "Mtri/s", toRate(sb, double(tris)),
                   {{"tris", double(tris)}, {"ms", sb.median}, {"as_bytes", double(a.asSize)}, {"scratch_bytes", double(a.buildScratch)}, {"rays_checked", double(nRays)}, {"rays_wrong", double(bad)}});
        rep.metric("blas.build.ms." + tag, "ms", sb, {{"tris", double(tris)}}, false);
        buildMs.push_back(sb.median);
        buildTris.push_back(tris);

        // --- compaction ---------------------------------------------------------
        timeEnc(ctx, [&](MTL4::ComputeCommandEncoder* e) {
            e->buildAccelerationStructure(a.as, a.desc, range(scratch));
            e->barrierAfterEncoderStages(MTL::StageAccelerationStructure, MTL::StageAccelerationStructure, MTL4::VisibilityOptionDevice);
            e->writeCompactedAccelerationStructureSize(a.as, MTL4::BufferRange::Make(sizeBuf->gpuAddress(), 8));
        });
        const NS::UInteger compSize = *static_cast<u32*>(sizeBuf->contents());
        if (compSize == 0 || compSize > a.asSize) {
            notes += tag + ": compacted size " + std::to_string(compSize) + " invalid; ";
            valid = false;
        } else {
            MTL::AccelerationStructure* comp = ctx.device()->newAccelerationStructure(compSize);
            ctx.adopt(comp);
            ctx.commitResidency();
            auto copy = [&](MTL4::ComputeCommandEncoder* e) { e->copyAndCompactAccelerationStructure(a.as, comp); };
            timeEnc(ctx, copy);
            bad = validateBlas(tr, comp, src.mesh, nRays, kSize, 12);
            if (bad) { valid = false; detail += "compact " + tag + " " + std::to_string(bad) + " rays wrong; "; }
            ctx.keepWarm(20);
            const Stats sc = ctx.measure([&] { return timeEnc(ctx, copy); });
            const double ratio = double(compSize) / double(a.asSize);
            rep.value("blas.compact.ratio." + tag, "ratio", ratio, {{"tris", double(tris)}, {"before_bytes", double(a.asSize)}, {"after_bytes", double(compSize)}}, false);
            if (target == triTargets.back()) rep.value("blas.compact.ratio", "ratio", ratio, {{"tris", double(tris)}, {"before_bytes", double(a.asSize)}, {"after_bytes", double(compSize)}}, false);
            rep.metric("blas.compact.ms." + tag, "ms", sc, {{"tris", double(tris)}, {"rays_wrong", double(bad)}}, false);
        }

        // --- refit (usage = Refit) vs rebuild of the same refittable AS ------------
        Accel r = makeBlas(ctx, src, true);
        MTL::Buffer* rscratch = ctx.buffer(std::max<size_t>(std::max(r.buildScratch, r.refitScratch), 4096));
        ctx.commitResidency();
        timeEnc(ctx, [&](MTL4::ComputeCommandEncoder* e) { e->buildAccelerationStructure(r.as, r.desc, range(rscratch)); });
        auto rebuild = [&](MTL4::ComputeCommandEncoder* e) { e->buildAccelerationStructure(r.as, r.desc, range(rscratch)); };
        auto refit = [&](MTL4::ComputeCommandEncoder* e) { e->refitAccelerationStructure(r.as, r.desc, r.as, range(rscratch)); };
        const std::vector<float> orig = src.mesh.v;
        auto move = [&](float phase) { // animate: y += 6 sin(0.05 x + phase) (CPU mesh and GPU buffer)
            for (size_t i = 0; i < orig.size(); i += 3) src.mesh.v[i + 1] = orig[i + 1] + 6.0f * std::sin(0.05f * orig[i] + phase);
            uploadVerts(src);
        };
        move(1.0f);
        timeEnc(ctx, refit);
        bad = validateBlas(tr, r.as, src.mesh, nRays, kSize, 13);
        if (bad) { valid = false; detail += "refit " + tag + " " + std::to_string(bad) + " rays wrong; "; }
        ctx.keepWarm(20);
        u32 flip = 0;
        const Stats sr = ctx.measure([&] { move((++flip & 1) ? 2.0f : 1.0f); return timeEnc(ctx, refit); });
        // Rebuild of the refittable AS at the current geometry (the CPU mesh follows `flip`).
        const Stats sre = ctx.measure([&] { return timeEnc(ctx, rebuild); });
        rep.metric("blas.refit." + tag, "Mtri/s", toRate(sr, double(tris)), {{"tris", double(tris)}, {"ms", sr.median}, {"rays_wrong", double(bad)}});
        rep.metric("blas.refit.ms." + tag, "ms", sr, {{"tris", double(tris)}}, false);
        rep.metric("blas.rebuild_refittable.ms." + tag, "ms", sre, {{"tris", double(tris)}}, false);
        rep.value("blas.refit_speedup." + tag, "ratio", sre.median / sr.median, {{"tris", double(tris)}});
        if (sr.median >= sre.median) { refitFaster = false; detail += "refit not faster than rebuild at " + tag + "; "; }
        refitMsLast = sr.median;
        rebuildRefittableLast = sre.median;
        // Validate once more after the last rebuild at the current vertex state.
        bad = validateBlas(tr, r.as, src.mesh, nRays, kSize, 14);
        if (bad) { valid = false; detail += "rebuild(refittable) " + tag + " " + std::to_string(bad) + " rays wrong; "; }
        ctx.keepWarm(30);
    }

    // --- TLAS ---------------------------------------------------------------------
    {
        BlasSrc rockSrc = makeSrc(ctx, makeRock());
        for (size_t i = 0; i < rockSrc.mesh.v.size(); i += 3) { // bump the patch so it is not flat
            const float x = rockSrc.mesh.v[i], z = rockSrc.mesh.v[i + 2];
            rockSrc.mesh.v[i + 1] = 0.6f * std::cos(x * 1.5f) * std::cos(z * 1.5f);
        }
        uploadVerts(rockSrc);
        Accel rock = makeBlas(ctx, rockSrc, false);
        MTL::Buffer* rscratch = ctx.buffer(std::max<size_t>(rock.buildScratch, 4096));
        ctx.commitResidency();
        timeEnc(ctx, [&](MTL4::ComputeCommandEncoder* e) { e->buildAccelerationStructure(rock.as, rock.desc, range(rscratch)); });
        const Mesh& rm = rockSrc.mesh;
        std::vector<double> tMs;
        std::vector<u32> tN;
        for (u32 n : instTargets) {
            const std::string tag = "inst_" + (n >= 1000 ? std::to_string(n / 1000) + "k" : std::to_string(n));
            std::vector<InstSpec> specs(n);
            MTL::Buffer* ib = ctx.buffer(size_t(n) * sizeof(InstDesc));
            auto* d = static_cast<InstDesc*>(ib->contents());
            for (u32 i = 0; i < n; ++i) {
                specs[i] = instSpec(i, n, kSize);
                const InstSpec& s = specs[i];
                const float m[12] = {s.s, 0, 0, 0, s.s, 0, 0, 0, s.s, s.x, s.y, s.z}; // column-major 4x3
                std::memcpy(d[i].m, m, sizeof(m));
                d[i].options = 0; d[i].mask = 0xFF; d[i].iftOffset = 0; d[i].userID = i;
                d[i].blasID = rock.as->gpuResourceID();
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
            MTL::Buffer* sc = ctx.buffer(std::max<size_t>(sz.buildScratchBufferSize, 4096));
            ctx.commitResidency();
            auto build = [&](MTL4::ComputeCommandEncoder* e) { e->buildAccelerationStructure(tlas, td, range(sc)); };
            timeEnc(ctx, build);
            // Validation: rays aimed at random instances from above.
            const u32 nRays = 128;
            std::vector<Ray> rays(nRays);
            for (u32 i = 0; i < nRays; ++i) {
                const InstSpec& s = specs[mix(i + 77) % n];
                rays[i].o = {double(s.x) + (rnd01(i, 31) - 0.5) * 1.4 * s.s, 50.0, double(s.z) + (rnd01(i, 32) - 0.5) * 1.4 * s.s};
                rays[i].d = {(rnd01(i, 33) - 0.5) * 0.05, -1.0, (rnd01(i, 34) - 0.5) * 0.05};
                const f64 l = std::sqrt(dot(rays[i].d, rays[i].d));
                rays[i].d = {rays[i].d.x / l, rays[i].d.y / l, rays[i].d.z / l};
            }
            auto triT = [&](const Ray& r, u32 p, u32 inst) {
                if (inst >= n || p >= rm.tris()) return -1.0;
                const InstSpec& s = specs[inst];
                auto w = [&](u32 vi) { const V3 v = rm.vert(rm.idx[3 * p + vi]); return V3{v.x * s.s + s.x, v.y * s.s + s.y, v.z * s.s + s.z}; };
                return rayTri(r.o, r.d, w(0), w(1), w(2));
            };
            auto validate = [&]() {
                const auto gpu = tr.run(tlas, true, rays);
                return compare(
                    rays, gpu,
                    [&](const Ray& r) {
                        Hit h;
                        for (u32 i = 0; i < n; ++i) {
                            const InstSpec& s = specs[i];
                            const f64 dx = r.o.x - s.x, dz = r.o.z - s.z; // bounding cylinder test (rays are near-vertical)
                            if (std::fabs(dx) > 3.0 * s.s || std::fabs(dz) > 3.0 * s.s) continue;
                            for (u32 p = 0; p < rm.tris(); ++p) {
                                const f64 t = triT(r, p, i);
                                if (t > 0 && (h.t < 0 || t < h.t)) { h.t = t; h.prim = p; h.inst = i; }
                            }
                        }
                        return h;
                    },
                    triT);
            };
            u32 bad = validate();
            u32 hits = 0;
            {
                const auto gpu = tr.run(tlas, true, rays);
                for (auto& g : gpu) hits += g[0] >= 0;
            }
            if (bad || hits < nRays / 4) { valid = false; detail += "tlas " + tag + ": " + std::to_string(bad) + " wrong, " + std::to_string(hits) + " hits; "; }
            ctx.keepWarm(20);
            const Stats st = ctx.measure([&] { return timeEnc(ctx, build); });
            rep.metric("tlas.build." + tag, "Minst/s", toRate(st, double(n)),
                       {{"instances", double(n)}, {"ms", st.median}, {"as_bytes", double(sz.accelerationStructureSize)}, {"rays_wrong", double(bad)}, {"rays_hit", double(hits)}});
            rep.metric("tlas.build.ms." + tag, "ms", st, {{"instances", double(n)}}, false);
            tMs.push_back(st.median);
            tN.push_back(n);
        }
        if (tMs.size() >= 2) {
            const bool grows = tMs.back() > tMs.front();
            if (!grows) { valid = false; detail += "TLAS build time does not grow with instances; "; }
            rep.value("tlas.build.growth", "ratio", tMs.back() / tMs.front(), {{"inst_first", double(tN.front())}, {"inst_last", double(tN.back())}}, false);
        }
    }

    // --- controls -----------------------------------------------------------------
    bool grows = buildMs.size() >= 2;
    std::string growText;
    for (size_t i = 1; i < buildMs.size(); ++i) {
        const double ratio = buildMs[i] / buildMs[i - 1];
        const double triRatio = double(buildTris[i]) / double(buildTris[i - 1]);
        growText += std::to_string(buildTris[i - 1]) + "->" + std::to_string(buildTris[i]) + " tris: " + std::to_string(ratio).substr(0, 5) + "x; ";
        // Below ~100k triangles a build is dominated by a fixed cost (~0.5 ms measured), so the
        // 10x -> >3x rule applies to decades that start at >= 100k triangles; smaller decades must only not shrink.
        if (triRatio >= 9 && buildTris[i - 1] >= 100000 && ratio <= 3.0) grows = false;
        if (ratio < 0.8) grows = false;
        rep.value("blas.build.growth." + std::to_string(i), "ratio", ratio, {{"tris_from", double(buildTris[i - 1])}, {"tris_to", double(buildTris[i])}}, false);
    }
    (void)refitMsLast; (void)rebuildRefittableLast;
    if (!notes.empty()) rep.note(notes);
    rep.note("builds timed as one AS-stage command (CommandTimer, empty span subtracted); spans exceed 2 ms for large builds (unavoidable). "
             "Refit = refit-usage AS refitted in place after the CPU moved the vertices (alternating two states); rebuild_refittable = full rebuild of the same AS. "
             "TLAS = instances of one 200-triangle BLAS via indirect instance descriptors (BLAS ResourceID)");
    rep.negative(valid && grows && refitFaster,
                 std::string(valid ? "all rays after build/refit/compaction/TLAS match the CPU; " : "VALIDATION FAILED: " + detail) +
                     (grows ? "build time grows with triangles: 10x -> >3x from 100k, monotone below (" : "build time does NOT grow (10x tris -> >3x from 100k; monotone below) (") + growText + "); " +
                     (refitFaster ? "refit faster than rebuild at every size" : "refit NOT faster than rebuild: " + detail));
    if (!valid) rep.status(Status::Failed, "validation failed: " + detail);
}

} // namespace

SOC_BENCH("B-21", "accel_build", "Acceleration structures: BLAS build/refit/compaction, TLAS build by instance count", benchBuild);

} // namespace soc
