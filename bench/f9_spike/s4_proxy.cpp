// F9-S4: simplified ray-tracing proxy geometry (meshoptimizer) and its
// hit / shadow error against the full geometry.
//
// Every proxy is INDEX-ONLY: meshopt_simplify / simplifySloppy run over the
// original GPUVertex positions and produce a new mesh-local index buffer; the
// proxy BLAS reads the same vertex buffer as the raster mesh (UV/material
// mapping is preserved by construction).  Levels (per mesh): "full" (a fresh
// BLAS over a copy of the full indices: the identity control of the method),
// r50/r25/r10 (target ratio, border free), r50b/r25b/r10b (border locked),
// e3/e2 (error driven, relative 1e-3 / 1e-2), s10 (sloppy 0.1) and s01
// (sloppy 0.01: the deliberately terrible proxy).
//
// Error is measured on the GPU: a reference TLAS over the original BLASes and
// a proxy TLAS with the same instances (same order, same transforms) trace
//   (1) primary rays: hit/miss disagreement and |dt| of the hit distance;
//   (2) sun shadow rays from the FULL-geometry hit points (offset along the
//       geometric normal by 1e-3 m and 1e-2 m): disagreement split in false
//       shadows (proxy occluded, full not) and leaks (proxy lit, full
//       occluded), traced against all instances and against the receiver's
//       own instance only (self-shadow acne of the proxy vs the full mesh);
//   (3) blame per mesh (the proxy occluder of a false shadow, the full
//       occluder of a leak, the receiver of a bad primary / acne ray) and
//       error by primary distance (<5 m, 5-15 m, >15 m).
// A policy then starts every mesh at the coarsest level of a ladder and
// upgrades, by measurement, the meshes that cause the error until global
// thresholds hold.  Negative controls: full-vs-full is exactly 0, the s01
// proxy exceeds the thresholds, and a single sabotaged mesh is the top
// offender and is flagged by the policy rule.
#include "f9_common.h"
#include "renderer/rt_proxy.h"

#include <meshoptimizer.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <optional>

namespace f9 {
namespace {

constexpr u32 kNone = 0xFFFFFFFFu;

// ---------------------------------------------------------------------------
// Levels
// ---------------------------------------------------------------------------

struct LevelSpec {
    const char* name;
    enum Kind { Copy, Simplify, Sloppy } kind;
    float ratio;       // target triangle ratio (0: error driven)
    bool lockBorder;
    float targetError; // relative, error driven only
};

constexpr LevelSpec kLevels[] = {
    {"full", LevelSpec::Copy, 1.0f, false, 0.0f},      // 0
    {"r50", LevelSpec::Simplify, 0.5f, false, 0.0f},   // 1
    {"r25", LevelSpec::Simplify, 0.25f, false, 0.0f},  // 2
    {"r10", LevelSpec::Simplify, 0.1f, false, 0.0f},   // 3
    {"r50b", LevelSpec::Simplify, 0.5f, true, 0.0f},   // 4
    {"r25b", LevelSpec::Simplify, 0.25f, true, 0.0f},  // 5
    {"r10b", LevelSpec::Simplify, 0.1f, true, 0.0f},   // 6
    {"e3", LevelSpec::Simplify, 0.0f, false, 1e-3f},   // 7
    {"e2", LevelSpec::Simplify, 0.0f, false, 1e-2f},   // 8
    {"s10", LevelSpec::Sloppy, 0.1f, false, 0.0f},     // 9
    {"s01", LevelSpec::Sloppy, 0.01f, false, 0.0f},    // 10
};
constexpr int kLevelCount = int(sizeof(kLevels) / sizeof(kLevels[0]));
constexpr int kFull = 0, kS01 = 10;

struct Thresholds {
    double shadowPct = 0.5;  // shadow disagreement (all instances, offset 1e-3) of receivers
    double primBadPct = 0.2; // primary rays whose hit/miss differs or |dt| > 5 cm, of all rays
    double dt95Cm = 1.0;     // p95 of |dt| over rays that hit both
    double acnePct = 0.5;    // own-instance proxy hits the full mesh does not have, of receivers
};

// ---------------------------------------------------------------------------
// GPU tracing with persistent buffers
// ---------------------------------------------------------------------------

class Tracer {
public:
    explicit Tracer(soc::Context& ctx) : ctx_(ctx) {
        nearest_ = ctx.compute(f9Library(ctx, "f9_trace.metal"), "trace_nearest");
        shadow_ = ctx.compute(f9Library(ctx, "s4_proxy.metal"), "px_shadow");
        rb_ = ctx.buffer(size_t(kBatch) * sizeof(GpuRay) + 64);
        hb_ = ctx.buffer(size_t(kBatch) * sizeof(GpuHit) + 64);
        ob_ = ctx.buffer(size_t(kBatch) * 4 + 64);
        wb_ = ctx.buffer(size_t(kBatch) * 4 + 64);
        pb_ = ctx.buffer(16);
    }

    void nearest(MTL::AccelerationStructure* as, const std::vector<GpuRay>& rays, std::vector<GpuHit>& out) {
        out.resize(rays.size());
        for (size_t first = 0; first < rays.size(); first += kBatch) {
            const u32 n = u32(std::min<size_t>(kBatch, rays.size() - first));
            std::memcpy(rb_->contents(), rays.data() + first, size_t(n) * sizeof(GpuRay));
            dispatch(nearest_, as, n, 0);
            std::memcpy(out.data() + first, hb_->contents(), size_t(n) * sizeof(GpuHit));
        }
    }

    /// First occluding instance per ray (kNone: unoccluded).  mode 1 only
    /// counts hits on instance owner[i].
    void shadow(MTL::AccelerationStructure* as, const std::vector<GpuRay>& rays, const std::vector<u32>& owner,
                u32 mode, std::vector<u32>& out) {
        out.resize(rays.size());
        for (size_t first = 0; first < rays.size(); first += kBatch) {
            const u32 n = u32(std::min<size_t>(kBatch, rays.size() - first));
            std::memcpy(rb_->contents(), rays.data() + first, size_t(n) * sizeof(GpuRay));
            std::memcpy(wb_->contents(), owner.data() + first, size_t(n) * 4);
            dispatch(shadow_, as, n, mode);
            std::memcpy(out.data() + first, ob_->contents(), size_t(n) * 4);
        }
    }

private:
    static constexpr u32 kBatch = 1u << 20; // 1M rays per command buffer
    void dispatch(MTL::ComputePipelineState* pso, MTL::AccelerationStructure* as, u32 n, u32 mode) {
        const u32 params[4] = {n, pso == shadow_ ? mode : 0xFFu, 0, 0};
        std::memcpy(pb_->contents(), params, sizeof params);
        MTL4::CommandBuffer* cmd = ctx_.beginCommands();
        MTL4::ComputeCommandEncoder* e = cmd->computeCommandEncoder();
        ctx_.table()->setResource(as->gpuResourceID(), 0);
        ctx_.table()->setAddress(rb_->gpuAddress(), 1);
        ctx_.table()->setAddress(pso == shadow_ ? ob_->gpuAddress() : hb_->gpuAddress(), 2);
        ctx_.table()->setAddress(pb_->gpuAddress(), 3);
        ctx_.table()->setAddress(wb_->gpuAddress(), 4);
        e->setComputePipelineState(pso);
        e->setArgumentTable(ctx_.table());
        e->dispatchThreads(MTL::Size::Make(n, 1, 1), MTL::Size::Make(64, 1, 1));
        e->endEncoding();
        ctx_.submit();
    }
    soc::Context& ctx_;
    MTL::ComputePipelineState* nearest_ = nullptr;
    MTL::ComputePipelineState* shadow_ = nullptr;
    MTL::Buffer *rb_, *hb_, *ob_, *wb_, *pb_;
};

// ---------------------------------------------------------------------------
// Proxy levels
// ---------------------------------------------------------------------------

struct LevelData {
    std::string name;
    MTL::Buffer* indices = nullptr;
    std::vector<u32> offset, count; // per mesh, in indices
    std::vector<Blas> blas;
    MTL::Buffer* scratch = nullptr;
    u64 tris = 0, bytes = 0;
    double simplifyMs = 0, errRelMax = 0, errAbsMax = 0, errAbsSum = 0;
    u32 meshesWithError = 0;
    u32 emptyFallbacks = 0; // meshes whose result was empty (kept at the 4-triangle minimum)
};

MTL4::PrimitiveAccelerationStructureDescriptor* proxyDescriptor(soc::Context& ctx, const SceneData& s,
                                                                const GpuGeometry& g, const LevelData& L, u32 mesh) {
    const phosphor::GPUMeshInfo& mi = s.scene.meshInfos()[mesh];
    auto* geo = MTL4::AccelerationStructureTriangleGeometryDescriptor::alloc()->init();
    geo->setVertexBuffer(range(g.vertices, u64(mi.vertexOffset) * sizeof(phosphor::GPUVertex)));
    geo->setVertexFormat(MTL::AttributeFormatFloat3);
    geo->setVertexStride(sizeof(phosphor::GPUVertex));
    geo->setIndexBuffer(range(L.indices, u64(L.offset[mesh]) * sizeof(u32), u64(L.count[mesh]) * sizeof(u32)));
    geo->setIndexType(MTL::IndexTypeUInt32);
    geo->setTriangleCount(L.count[mesh] / 3);
    geo->setOpaque(true);
    ctx.keep(geo);
    auto* d = MTL4::PrimitiveAccelerationStructureDescriptor::alloc()->init();
    d->setGeometryDescriptors(NS::Array::array(geo));
    d->setUsage(MTL::AccelerationStructureUsageNone);
    ctx.keep(d);
    return d;
}

/// World scale of mesh `m` (length of the first matrix column of its first
/// instance; 1 if it has none): object-space errors times this are meters.
double meshWorldScale(const SceneData& s, u32 m) {
    for (const auto& i : s.instances)
        if (i.meshIndex == m) return glm::length(glm::vec3(i.modelMatrix[0], i.modelMatrix[1], i.modelMatrix[2]));
    return 1.0;
}

LevelData makeLevel(soc::Context& ctx, const SceneData& s, const GpuGeometry& g, const LevelSpec& spec) {
    LevelData L;
    L.name = spec.name;
    const u32 meshes = s.scene.getMeshCount();
    const auto& verts = s.scene.vertices();
    const auto& idx = s.scene.indices();
    std::vector<std::vector<u32>> out(meshes);
    const double t0 = soc::nowMs();
    for (u32 m = 0; m < meshes; ++m) {
        const phosphor::GPUMeshInfo& mi = s.scene.meshInfos()[m];
        const u32* src = idx.data() + mi.indexOffset;
        const size_t n = mi.indexCount;
        if (spec.kind == LevelSpec::Copy || n <= 12) {
            out[m].assign(src, src + n);
            continue;
        }
        u32 maxIdx = 0;
        for (size_t i = 0; i < n; ++i) maxIdx = std::max(maxIdx, src[i]);
        const size_t vcount = size_t(maxIdx) + 1;
        const float* pos = &verts[mi.vertexOffset].px;
        const size_t stride = sizeof(phosphor::GPUVertex);
        const float scale = meshopt_simplifyScale(pos, vcount, stride);
        size_t target = spec.ratio > 0 ? std::max<size_t>(12, size_t(double(n) * spec.ratio) / 3 * 3) : 0;
        target = std::min(target, n);
        std::vector<u32> dst(n);
        float err = 0;
        size_t got;
        if (spec.lockBorder) {
            phosphor::RtProxyLevel level;
            if (!phosphor::rtProxyParseLevel(spec.name, level))
                throw std::runtime_error("unknown production RT proxy level");
            auto cooked = phosphor::rtCookProxyMesh(std::span(verts).subspan(mi.vertexOffset, vcount),
                                                    std::span(src, n), level);
            dst = std::move(cooked.indices);
            got = dst.size();
            err = cooked.relativeError;
        } else if (spec.kind == LevelSpec::Sloppy) {
            got = meshopt_simplifySloppy(dst.data(), src, n, pos, vcount, stride, nullptr, target, FLT_MAX, &err);
        } else {
            const unsigned opts = spec.lockBorder ? meshopt_SimplifyLockBorder : 0u;
            got = meshopt_simplify(dst.data(), src, n, pos, vcount, stride, target,
                                   spec.ratio > 0 ? 1.0f : spec.targetError, opts, &err);
        }
        if (got < 3) { // a BLAS cannot be empty: keep the 4-triangle minimum
            L.emptyFallbacks++;
            got = meshopt_simplify(dst.data(), src, n, pos, vcount, stride, 12, 1.0f, 0, &err);
        }
        dst.resize(got);
        out[m] = std::move(dst);
        L.errRelMax = std::max(L.errRelMax, double(err));
        const double abs = double(err) * double(scale) * meshWorldScale(s, m);
        L.errAbsMax = std::max(L.errAbsMax, abs);
        L.errAbsSum += abs;
        L.meshesWithError++;
    }
    L.simplifyMs = soc::nowMs() - t0;
    u64 total = 0;
    L.offset.resize(meshes);
    L.count.resize(meshes);
    for (u32 m = 0; m < meshes; ++m) {
        L.offset[m] = u32(total);
        L.count[m] = u32(out[m].size());
        total += out[m].size();
    }
    L.indices = ctx.buffer(std::max<u64>(total * 4, 16));
    auto* dst = static_cast<u32*>(L.indices->contents());
    for (u32 m = 0; m < meshes; ++m) std::memcpy(dst + L.offset[m], out[m].data(), out[m].size() * 4);
    L.tris = total / 3;
    L.blas.resize(meshes);
    for (u32 m = 0; m < meshes; ++m) {
        L.blas[m].desc = proxyDescriptor(ctx, s, g, L, m);
        L.blas[m].sizes = ctx.device()->accelerationStructureSizes(L.blas[m].desc);
        L.blas[m].as = newAccelerationStructure(ctx, L.blas[m].sizes.accelerationStructureSize, AsPlacement::Device);
        L.blas[m].triangles = L.count[m] / 3;
        L.bytes += L.blas[m].sizes.accelerationStructureSize;
    }
    ctx.commitResidency();
    L.scratch = scratchFor(ctx, L.blas);
    buildBlases(ctx, L.blas, L.scratch);
    return L;
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

struct Eval {
    u64 rays = 0, hitMiss = 0, both = 0, dt1 = 0, dt5 = 0, primBad = 0;
    double dt50 = 0, dt95 = 0, dtMax = 0; // cm, over rays that hit both
    u64 recv = 0;
    u64 dis[2][2] = {}, fal[2][2] = {}, lea[2][2] = {}; // [offset 1e-3, 1e-2][mode all, own]
    u64 bRays[3] = {}, bBad[3] = {}, bRecv[3] = {}, bDis[3] = {};
    std::vector<u32> bPrim, bShadow, bSelf; // blame per mesh
    static double pct(u64 a, u64 b) { return b ? 100.0 * double(a) / double(b) : 0.0; }
    [[nodiscard]] double score(u32 m) const { return double(bPrim[m]) + double(bShadow[m]) + double(bSelf[m]); }
    [[nodiscard]] bool meets(const Thresholds& t) const {
        return pct(dis[0][0], recv) <= t.shadowPct && pct(primBad, rays) <= t.primBadPct && dt95 <= t.dt95Cm &&
               pct(fal[0][1], recv) <= t.acnePct;
    }
};

struct SceneEval {
    SceneData* s = nullptr;
    GpuGeometry g;
    std::vector<Blas> fullBlas;
    std::vector<GpuRay> prim;
    std::vector<GpuHit> fullHit;
    std::vector<GpuRay> shRays[2]; // offsets 1e-3 and 1e-2
    std::vector<u32> owner;        // instance of the receiver
    std::vector<float> recvT;      // primary distance of the receiver
    std::vector<u32> shFull[2][2]; // full-geometry results [offset][mode]
    Tlas refTlas;
};

Tlas buildTlas(soc::Context& ctx, const SceneData& s, const std::vector<const std::vector<Blas>*>& perMesh) {
    Tlas t = allocateTlas(ctx, u32(s.instances.size()), MTL::AccelerationStructureUsageNone, AsPlacement::Device);
    auto* d = static_cast<InstanceDesc*>(t.instances->contents());
    for (u32 i = 0; i < s.instances.size(); ++i) {
        const u32 m = s.instances[i].meshIndex;
        d[i] = toInstanceDesc(s.instances[i], (*perMesh[m])[m].as->gpuResourceID(), i,
                              MTL::AccelerationStructureInstanceOptionOpaque);
    }
    timeEncoder(ctx, [&](MTL4::ComputeCommandEncoder* e) { e->buildAccelerationStructure(t.as, t.desc, range(t.scratch)); });
    return t;
}

int bucketOf(float t) { return t < 5.0f ? 0 : (t < 15.0f ? 1 : 2); }

Eval evaluate(soc::Context& ctx, Tracer& tr, SceneEval& se, const std::vector<int>& cfg,
              const std::vector<LevelData>& levels) {
    const SceneData& s = *se.s;
    std::vector<const std::vector<Blas>*> perMesh(cfg.size());
    for (size_t m = 0; m < cfg.size(); ++m) perMesh[m] = &levels[size_t(cfg[m])].blas;
    Tlas tlas = buildTlas(ctx, s, perMesh);
    Eval ev;
    ev.bPrim.assign(cfg.size(), 0);
    ev.bShadow.assign(cfg.size(), 0);
    ev.bSelf.assign(cfg.size(), 0);
    std::vector<GpuHit> ph;
    tr.nearest(tlas.as, se.prim, ph);
    ev.rays = se.prim.size();
    std::vector<float> dts;
    for (size_t i = 0; i < ph.size(); ++i) {
        const GpuHit& f = se.fullHit[i];
        const GpuHit& p = ph[i];
        const bool fh = f.t >= 0, phit = p.t >= 0;
        const int b = fh ? bucketOf(f.t) : 2;
        ev.bRays[b]++;
        bool bad = false;
        u32 blame = 0;
        if (fh != phit) {
            ev.hitMiss++;
            bad = true;
            blame = fh ? s.instances[f.instance].meshIndex : s.instances[p.instance].meshIndex;
        } else if (fh) {
            ev.both++;
            const double dt = std::fabs(double(p.t) - double(f.t)) * 100.0;
            dts.push_back(float(dt));
            ev.dtMax = std::max(ev.dtMax, dt);
            ev.dt1 += dt > 1.0;
            if (dt > 5.0) {
                ev.dt5++;
                bad = true;
                blame = s.instances[p.t < f.t ? p.instance : f.instance].meshIndex; // nearer proxy surface is the culprit
            }
        }
        if (bad) {
            ev.primBad++;
            ev.bBad[b]++;
            ev.bPrim[blame]++;
        }
    }
    if (!dts.empty()) {
        auto q = [&](double f) {
            const size_t k = std::min(dts.size() - 1, size_t(double(dts.size()) * f));
            std::nth_element(dts.begin(), dts.begin() + long(k), dts.end());
            return double(dts[k]);
        };
        ev.dt50 = q(0.50);
        ev.dt95 = q(0.95);
    }
    ev.recv = se.owner.size();
    std::vector<u32> res;
    for (int off = 0; off < 2; ++off)
        for (int mode = 0; mode < 2; ++mode) {
            tr.shadow(tlas.as, se.shRays[off], se.owner, u32(mode), res);
            const std::vector<u32>& ref = se.shFull[off][mode];
            for (size_t i = 0; i < res.size(); ++i) {
                const bool p = res[i] != kNone, f = ref[i] != kNone;
                const int b = bucketOf(se.recvT[i]);
                if (off == 0 && mode == 0) ev.bRecv[b]++;
                if (p == f) continue;
                ev.dis[off][mode]++;
                if (off == 0 && mode == 0) ev.bDis[b]++;
                if (p) { // false shadow / acne
                    ev.fal[off][mode]++;
                    if (off == 0 && mode == 0) ev.bShadow[s.instances[res[i]].meshIndex]++;
                    if (off == 0 && mode == 1) ev.bSelf[s.instances[se.owner[i]].meshIndex]++;
                } else { // leak
                    ev.lea[off][mode]++;
                    if (off == 0 && mode == 0) ev.bShadow[s.instances[ref[i]].meshIndex]++;
                }
            }
        }
    return ev;
}

/// Hit points, shadow receivers and the full-geometry reference results.
void prepare(soc::Context& ctx, Tracer& tr, SceneEval& se, const std::vector<Ray>& primRays, V3 sun) {
    const SceneData& s = *se.s;
    se.g = uploadGeometry(ctx, s);
    se.fullBlas = allocateBlases(ctx, s, se.g, {});
    MTL::Buffer* scr = scratchFor(ctx, se.fullBlas);
    buildBlases(ctx, se.fullBlas, scr);
    std::vector<const std::vector<Blas>*> pm(s.scene.getMeshCount(), &se.fullBlas);
    se.refTlas = buildTlas(ctx, s, pm);
    se.prim.reserve(primRays.size());
    for (const Ray& r : primRays) se.prim.push_back(toGpu(r));
    tr.nearest(se.refTlas.as, se.prim, se.fullHit);
    const TriangleSoup soup = s.soup();
    const SoupIndex sidx(soup);
    const f64 offs[2] = {1e-3, 1e-2};
    for (size_t i = 0; i < se.prim.size(); ++i) {
        const GpuHit& h = se.fullHit[i];
        if (h.t < 0) continue;
        const u32 tri = sidx(h.instance, h.geometry, h.primitive);
        if (tri == kNone) continue;
        const V3 a = soup.v[3 * tri], b = soup.v[3 * tri + 1], c = soup.v[3 * tri + 2];
        const f64 u = h.u, v = h.v;
        const V3 p = a * (1.0 - u - v) + b * u + c * v;
        V3 n = normalize(cross(b - a, c - a));
        const V3 d = {se.prim[i].d[0], se.prim[i].d[1], se.prim[i].d[2]};
        if (dot(n, d) > 0) n = n * -1.0;
        if (dot(n, sun) <= 0.05) continue; // facing away from the sun: shadowed by definition
        for (int k = 0; k < 2; ++k) se.shRays[k].push_back(toGpu({p + n * offs[k], sun, 0.0, 1e30}));
        se.owner.push_back(h.instance);
        se.recvT.push_back(h.t);
    }
    for (int off = 0; off < 2; ++off)
        for (int mode = 0; mode < 2; ++mode)
            tr.shadow(se.refTlas.as, se.shRays[off], se.owner, u32(mode), se.shFull[off][mode]);
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

void emit(soc::Report& rep, const std::string& pre, u64 fullTris, u64 fullBytes, const LevelData& L, const Eval& e,
          u64 trisInstanced, u64 fullInstanced) {
    const char* kB[3] = {"d0_5", "d5_15", "d15p"};
    auto V = [&](const std::string& n, const char* unit, double v) { rep.value(pre + n, unit, v, {}, false); };
    V(".tris", "tris", double(L.tris));
    V(".tris_pct", "%", Eval::pct(L.tris, fullTris));
    V(".tris_instanced_pct", "%", Eval::pct(trisInstanced, fullInstanced));
    V(".blas_bytes", "B", double(L.bytes));
    V(".blas_bytes_pct", "%", Eval::pct(L.bytes, fullBytes));
    V(".simplify_ms", "ms", L.simplifyMs);
    V(".err_rel_max", "ratio", L.errRelMax);
    V(".err_abs_max_cm", "cm", L.errAbsMax * 100.0);
    V(".err_abs_mean_cm", "cm", L.meshesWithError ? L.errAbsSum / L.meshesWithError * 100.0 : 0.0);
    V(".prim_hitmiss_pct", "%", Eval::pct(e.hitMiss, e.rays));
    V(".prim_bad_pct", "%", Eval::pct(e.primBad, e.rays));
    V(".dt_gt1cm_pct", "%", Eval::pct(e.dt1, e.both));
    V(".dt_gt5cm_pct", "%", Eval::pct(e.dt5, e.both));
    V(".dt50_cm", "cm", e.dt50);
    V(".dt95_cm", "cm", e.dt95);
    V(".dtmax_cm", "cm", e.dtMax);
    V(".shadow_disagree_pct", "%", Eval::pct(e.dis[0][0], e.recv));
    V(".shadow_false_pct", "%", Eval::pct(e.fal[0][0], e.recv));
    V(".shadow_leak_pct", "%", Eval::pct(e.lea[0][0], e.recv));
    V(".shadow_disagree_pct_off1e2", "%", Eval::pct(e.dis[1][0], e.recv));
    V(".shadow_false_pct_off1e2", "%", Eval::pct(e.fal[1][0], e.recv));
    V(".shadow_leak_pct_off1e2", "%", Eval::pct(e.lea[1][0], e.recv));
    V(".acne_pct", "%", Eval::pct(e.fal[0][1], e.recv));
    V(".acne_pct_off1e2", "%", Eval::pct(e.fal[1][1], e.recv));
    V(".self_disagree_pct", "%", Eval::pct(e.dis[0][1], e.recv));
    for (int b = 0; b < 3; ++b) {
        V(std::string(".prim_bad_pct_") + kB[b], "%", Eval::pct(e.bBad[b], e.bRays[b]));
        V(std::string(".shadow_disagree_pct_") + kB[b], "%", Eval::pct(e.bDis[b], e.bRecv[b]));
    }
}

/// "mesh i (tris, min extent, share of blame)" of the `k` highest-blame meshes.
std::string topOffenders(const SceneData& s, const Eval& e, u32 k) {
    const u32 meshes = u32(e.bPrim.size());
    std::vector<u32> order(meshes);
    for (u32 i = 0; i < meshes; ++i) order[i] = i;
    double total = 0;
    for (u32 m = 0; m < meshes; ++m) total += e.score(m);
    std::sort(order.begin(), order.end(), [&](u32 a, u32 b) { return e.score(a) > e.score(b); });
    std::string out;
    for (u32 i = 0; i < std::min(k, meshes); ++i) {
        const u32 m = order[i];
        if (e.score(m) <= 0) break;
        const phosphor::GPUMeshInfo& mi = s.scene.meshInfos()[m];
        glm::vec3 lo(1e30f), hi(-1e30f);
        for (u32 j = 0; j < mi.indexCount; ++j) {
            const phosphor::GPUVertex& v = s.scene.vertices()[mi.vertexOffset + s.scene.indices()[mi.indexOffset + j]];
            lo = glm::min(lo, glm::vec3(v.px, v.py, v.pz));
            hi = glm::max(hi, glm::vec3(v.px, v.py, v.pz));
        }
        const glm::vec3 ext = (hi - lo) * float(meshWorldScale(s, m));
        char buf[160];
        std::snprintf(buf, sizeof buf, "%smesh %u (%u tris, min extent %.1f cm, %.0f%% of blame)", out.empty() ? "" : "; ", m,
                      mi.indexCount / 3, double(std::min({ext.x, ext.y, ext.z})) * 100.0, 100.0 * e.score(m) / total);
        out += buf;
    }
    return out.empty() ? "none" : out;
}

/// Meshes the policy rule upgrades one ladder step: blame share >= 2% (at
/// least the top one when anything is blamed).
std::vector<u32> flagged(const Eval& e, const std::vector<char>* eligible = nullptr) {
    const u32 meshes = u32(e.bPrim.size());
    double total = 0, best = 0;
    u32 top = 0;
    for (u32 m = 0; m < meshes; ++m) {
        if (eligible && !(*eligible)[m]) continue;
        const double sc = e.score(m);
        total += sc;
        if (sc > best) { best = sc; top = m; }
    }
    std::vector<u32> out;
    if (total <= 0) return out;
    for (u32 m = 0; m < meshes; ++m)
        if ((!eligible || (*eligible)[m]) && e.score(m) >= 0.02 * total) out.push_back(m);
    if (out.empty()) out.push_back(top);
    return out;
}

struct PolicyResult {
    std::vector<int> cfg;
    Eval ev;
    int iterations = 0;
    bool met = false;
    u64 tris = 0, bytes = 0, trisInstanced = 0;
    std::vector<u32> atLevel = std::vector<u32>(kLevelCount, 0); // meshes per level
};

/// `ladder[m]`: levels of mesh m from the coarsest to the finest (ends with
/// the full mesh).  Each iteration traces the current configuration and
/// upgrades one ladder step the meshes blamed for the error.
PolicyResult runPolicy(soc::Context& ctx, Tracer& tr, SceneEval& se, const std::vector<LevelData>& levels,
                       const std::vector<std::vector<int>>& ladder, const Thresholds& thr, bool productionPolicy = false) {
    const u32 meshes = se.s->scene.getMeshCount();
    PolicyResult r;
    std::vector<u32> pos(meshes, 0);
    for (int iter = 0; iter < 60; ++iter) {
        r.cfg.assign(meshes, 0);
        for (u32 m = 0; m < meshes; ++m) r.cfg[m] = ladder[m][pos[m]];
        r.ev = evaluate(ctx, tr, se, r.cfg, levels);
        r.iterations = iter + 1;
        if (r.ev.meets(thr)) { r.met = true; break; }
        // Only meshes that still have a finer level can be upgraded (blame can
        // also land on a mesh that is already full when a neighbour's proxy is the cause).
        std::vector<char> up(meshes);
        for (u32 m = 0; m < meshes; ++m) up[m] = pos[m] + 1 < ladder[m].size();
        if (productionPolicy) {
            std::vector<phosphor::RtProxyLevel> productionLevels(meshes);
            std::vector<double> blame(meshes);
            for (u32 m = 0; m < meshes; ++m) {
                if (!phosphor::rtProxyParseLevel(kLevels[r.cfg[m]].name, productionLevels[m]))
                    throw std::runtime_error("invalid production RT proxy ladder");
                blame[m] = r.ev.score(m);
            }
            if (!phosphor::rtProxyPromote(productionLevels, blame)) break;
            for (u32 m = 0; m < meshes; ++m)
                if (std::string(phosphor::rtProxyLevelName(productionLevels[m])) != kLevels[r.cfg[m]].name) ++pos[m];
        } else {
            const std::vector<u32> fl = flagged(r.ev, &up);
            if (fl.empty()) break;
            for (u32 m : fl) pos[m]++;
        }
    }
    std::vector<u32> instCount(meshes, 0);
    for (const auto& i : se.s->instances) instCount[i.meshIndex]++;
    for (u32 m = 0; m < meshes; ++m) {
        const LevelData& L = levels[size_t(r.cfg[m])];
        r.tris += L.count[m] / 3;
        r.bytes += L.blas[m].sizes.accelerationStructureSize;
        r.trisInstanced += u64(L.count[m] / 3) * instCount[m];
        r.atLevel[size_t(r.cfg[m])]++;
    }
    return r;
}

/// Same ladder (a list of level indices) for every mesh.
std::vector<std::vector<int>> sameLadder(u32 meshes, std::initializer_list<int> l) {
    return std::vector<std::vector<int>>(meshes, std::vector<int>(l));
}

/// Per mesh: every non-sloppy level sorted by this mesh's triangle count
/// (levels that do not reduce the mesh are dropped), ending with the full mesh.
std::vector<std::vector<int>> sortedLadder(const std::vector<LevelData>& levels, u32 meshes) {
    std::vector<std::vector<int>> out(meshes);
    for (u32 m = 0; m < meshes; ++m) {
        std::vector<int> c;
        for (int l = 1; l <= 8; ++l)
            if (levels[size_t(l)].count[m] < levels[size_t(kFull)].count[m]) c.push_back(l);
        std::stable_sort(c.begin(), c.end(), [&](int a, int b) { return levels[size_t(a)].count[m] < levels[size_t(b)].count[m]; });
        std::vector<int> d;
        for (int l : c)
            if (d.empty() || levels[size_t(l)].count[m] != levels[size_t(d.back())].count[m]) d.push_back(l);
        d.push_back(kFull);
        out[m] = std::move(d);
    }
    return out;
}

u64 instancedTris(const SceneData& s, const LevelData& L) {
    u64 t = 0;
    for (const auto& i : s.instances) t += L.count[i.meshIndex] / 3;
    return t;
}

// ---------------------------------------------------------------------------
// One scene
// ---------------------------------------------------------------------------

struct SceneResult {
    bool identityExact = false;
    double s01ShadowPct = 0, s01PrimBadPct = 0, s01Dt95 = 0;
    bool s01Violates = false;
    bool sabotageDetected = false;
    bool policyMet[3] = {false, false, false};
    std::string detail;
};

SceneResult runScene(soc::Context& ctx, soc::Report& rep, Tracer& tr, const std::string& tag, SceneData& s,
                     const std::vector<Ray>& primRays, V3 sun, const Thresholds& thr) {
    SceneResult R;
    SceneEval se;
    se.s = &s;
    prepare(ctx, tr, se, primRays, sun);
    const u32 meshes = s.scene.getMeshCount();
    u64 fullTris = 0;
    for (u32 m = 0; m < meshes; ++m) fullTris += s.meshTriangles(m);
    const u64 fullInstanced = s.totalTriangles();
    u64 fullBytes = 0;
    for (const Blas& b : se.fullBlas) fullBytes += b.sizes.accelerationStructureSize;
    u32 hits = 0;
    for (const GpuHit& h : se.fullHit) hits += h.t >= 0;
    ctx.log("S4 %s: %u meshes, %u instances, %zu primary rays (%u hit), %zu shadow receivers", tag.c_str(), meshes,
            u32(s.instances.size()), se.prim.size(), hits, se.owner.size());
    rep.value("proxy." + tag + ".meshes", "meshes", meshes, {{"instances", double(s.instances.size())}}, false);
    rep.value("proxy." + tag + ".rays", "rays", double(se.prim.size()), {{"hits", double(hits)}}, false);
    rep.value("proxy." + tag + ".receivers", "rays", double(se.owner.size()), {}, false);
    rep.value("proxy." + tag + ".orig.tris", "tris", double(fullTris), {{"instanced", double(fullInstanced)}}, false);
    rep.value("proxy." + tag + ".orig.blas_bytes", "B", double(fullBytes), {}, false);

    std::vector<LevelData> levels;
    levels.reserve(kLevelCount);
    std::vector<Eval> evals;
    for (int l = 0; l < kLevelCount; ++l) {
        levels.push_back(makeLevel(ctx, s, se.g, kLevels[l]));
        LevelData& L = levels.back();
        std::vector<int> cfg(meshes, l);
        evals.push_back(evaluate(ctx, tr, se, cfg, levels));
        const Eval& e = evals.back();
        const std::string pre = "proxy." + tag + "." + L.name;
        emit(rep, pre, fullTris, fullBytes, L, e, instancedTris(s, L), fullInstanced);
        // BLAS build time of the whole level (median over repetitions).
        const soc::Stats st = ctx.measure([&] { return buildBlases(ctx, L.blas, L.scratch); });
        rep.metric(pre + ".blas_build_ms", "ms", st, {{"meshes", double(meshes)}}, false);
        ctx.log("S4 %s %-5s tris %8llu (%5.1f%%) bytes %5.1f%% | prim bad %6.3f%% dt95 %7.3f cm | shadow dis %6.3f%% (F %.3f L %.3f) acne %6.3f%% | simplify %.0f ms",
                tag.c_str(), L.name.c_str(), (unsigned long long)L.tris, Eval::pct(L.tris, fullTris),
                Eval::pct(L.bytes, fullBytes), Eval::pct(e.primBad, e.rays), e.dt95, Eval::pct(e.dis[0][0], e.recv),
                Eval::pct(e.fal[0][0], e.recv), Eval::pct(e.lea[0][0], e.recv), Eval::pct(e.fal[0][1], e.recv),
                L.simplifyMs);
        if (l != kFull)
            rep.note(tag + " " + L.name + " top offenders: " + topOffenders(s, e, 5));
    }

    // Control 1: the fresh full copy must be exactly identical.
    const Eval& e0 = evals[kFull];
    R.identityExact = e0.hitMiss == 0 && e0.dt1 == 0 && e0.dtMax == 0 && e0.primBad == 0 && e0.dis[0][0] == 0 &&
                      e0.dis[0][1] == 0 && e0.dis[1][0] == 0 && e0.dis[1][1] == 0 && e0.recv > 0;
    // Control 2: the s01 proxy must violate the thresholds.
    const Eval& e1 = evals[kS01];
    R.s01ShadowPct = Eval::pct(e1.dis[0][0], e1.recv);
    R.s01PrimBadPct = Eval::pct(e1.primBad, e1.rays);
    R.s01Dt95 = e1.dt95;
    R.s01Violates = !e1.meets(thr);

    // Policies: three ladders (border free, border locked, every non-sloppy
    // level sorted per mesh by triangle count).
    const char* pname[3] = {"policy_free", "policy_lock", "policy_all"};
    std::vector<std::vector<int>> ladders[3] = {sameLadder(meshes, {3, 2, 1, 0}), sameLadder(meshes, {6, 5, 4, 0}),
                                                      sortedLadder(levels, meshes)};
    const auto protectedMeshes = phosphor::rtProxyProtectedMeshes(meshes, s.instances, s.materials);
    for (u32 mesh : protectedMeshes) ladders[1][mesh] = {kFull};
    rep.value("proxy." + tag + ".policy_lock.protected_meshes", "meshes", double(protectedMeshes.size()), {}, false);
    std::optional<phosphor::RtProxyManifest> exportManifest;
    for (int k = 0; k < 3; ++k) {
        PolicyResult p = runPolicy(ctx, tr, se, levels, ladders[k], thr, k == 1);
        R.policyMet[k] = p.met;
        if (k == 1 && p.met && tag == "sponza" && std::getenv("PHOSPHOR_RT_PROXY_EXPORT_DIR")) {
            if (ctx.quick()) throw std::runtime_error("RT proxy export requires the full three-camera S4 corpus, no --quick");
            std::vector<phosphor::RtProxyLevel> selected(meshes);
            for (u32 m = 0; m < meshes; ++m)
                if (!phosphor::rtProxyParseLevel(kLevels[p.cfg[m]].name, selected[m]))
                    throw std::runtime_error("invalid exported RT proxy level");
            phosphor::RtProxyMeasurements measured{p.ev.rays, p.ev.recv,
                Eval::pct(p.ev.dis[0][0], p.ev.recv), Eval::pct(p.ev.primBad, p.ev.rays),
                p.ev.dt95, Eval::pct(p.ev.fal[0][1], p.ev.recv)};
            exportManifest = phosphor::rtMakeProxyManifest(s.scene, selected, measured, "sponza",
                "F9-S4 three Sponza cameras at 960x540; sun=(0.30,0.85,0.20) normalized; "
                "full geometry receivers, offsets 0.001/0.01m; MASK/emissive meshes full");
            // Never attach measured errors to regenerated indices unless they
            // match the ACTUAL geometry used in this GPU evaluation exactly.
            for (u32 m = 0; m < meshes; ++m) {
                const auto& actual = levels[size_t(p.cfg[m])];
                const auto indices = std::span(static_cast<const u32*>(actual.indices->contents()) + actual.offset[m],
                                              actual.count[m]);
                if (phosphor::rtProxyIndexFingerprint(indices) != exportManifest->meshes[m].indexFingerprint)
                    throw std::runtime_error("exported RT proxy differs from measured index stream");
            }
        }
        const std::string pre = "proxy." + tag + "." + pname[k];
        LevelData agg;
        agg.tris = p.tris;
        agg.bytes = p.bytes;
        emit(rep, pre, fullTris, fullBytes, agg, p.ev, p.trisInstanced, fullInstanced);
        rep.value(pre + ".iterations", "iter", p.iterations, {}, false);
        rep.value(pre + ".met", "bool", p.met ? 1 : 0, {}, true);
        std::string hist;
        for (int l = 0; l < kLevelCount; ++l) {
            if (!p.atLevel[size_t(l)]) continue;
            rep.value(pre + ".meshes_at_" + kLevels[l].name, "meshes", p.atLevel[size_t(l)], {}, false);
            hist += std::string(" ") + kLevels[l].name + "=" + std::to_string(p.atLevel[size_t(l)]);
        }
        ctx.log("S4 %s %s: %s after %d iterations, tris %.1f%%, bytes %.1f%% | prim bad %.3f%% dt95 %.3f cm shadow %.3f%% acne %.3f%% | meshes:%s",
                tag.c_str(), pname[k], p.met ? "MET" : "NOT MET", p.iterations, Eval::pct(p.tris, fullTris),
                Eval::pct(p.bytes, fullBytes), Eval::pct(p.ev.primBad, p.ev.rays), p.ev.dt95,
                Eval::pct(p.ev.dis[0][0], p.ev.recv), Eval::pct(p.ev.fal[0][1], p.ev.recv), hist.c_str());
        rep.note(tag + " " + pname[k] + (p.met ? " met" : " NOT met") + " thresholds after " + std::to_string(p.iterations) +
                 " iterations (meshes per level:" + hist + "); remaining offenders: " + topOffenders(s, p.ev, 3));
    }

    // Control 3: one sabotaged mesh (the worst offender at s01, all others full)
    // must be the top offender and be flagged by the policy rule.
    u32 victim = 0;
    double best = -1;
    for (u32 m = 0; m < meshes; ++m)
        if (e1.score(m) > best) { best = e1.score(m); victim = m; }
    std::vector<int> cfg(meshes, kFull);
    cfg[victim] = kS01;
    const Eval es = evaluate(ctx, tr, se, cfg, levels);
    u32 top = 0;
    double sabTotal = 0;
    for (u32 m = 0; m < meshes; ++m) {
        sabTotal += es.score(m);
        if (es.score(m) > es.score(top)) top = m;
    }
    const std::vector<u32> fl = flagged(es);
    const bool isFlagged = std::find(fl.begin(), fl.end(), victim) != fl.end();
    R.sabotageDetected = top == victim && es.score(victim) > 0 && isFlagged;
    rep.value("proxy." + tag + ".sabotage.victim_mesh", "mesh", victim, {}, false);
    rep.value("proxy." + tag + ".sabotage.victim_score", "rays", es.score(victim), {}, false);
    rep.value("proxy." + tag + ".sabotage.detected", "bool", R.sabotageDetected ? 1 : 0, {}, true);
    rep.value("proxy." + tag + ".sabotage.shadow_disagree_pct", "%", Eval::pct(es.dis[0][0], es.recv), {}, false);
    ctx.log("S4 %s sabotage: mesh %u at s01, top offender %u, flagged %d (score %.0f of %.0f)", tag.c_str(), victim, top,
            int(isFlagged), es.score(victim), sabTotal);

    char buf[640];
    std::snprintf(buf, sizeof buf,
                  "%s: full-vs-full %s (%llu receivers differ, dtmax %.3g cm); s01 shadow %.2f%% primbad %.2f%% dt95 %.1f cm (%s thresholds); sabotage mesh %u %s; policy free %s lock %s all %s",
                  tag.c_str(), R.identityExact ? "exact 0" : "NOT exact", (unsigned long long)(e0.dis[0][0] + e0.dis[1][0]),
                  e0.dtMax, R.s01ShadowPct, R.s01PrimBadPct, R.s01Dt95, R.s01Violates ? "exceeds" : "does NOT exceed", victim,
                  R.sabotageDetected ? "detected" : "NOT detected", R.policyMet[0] ? "met" : "NOT met",
                  R.policyMet[1] ? "met" : "NOT met", R.policyMet[2] ? "met" : "NOT met");
    R.detail = buf;
    if (exportManifest && R.identityExact && R.s01Violates && R.sabotageDetected && R.policyMet[1]) {
        const std::filesystem::path directory(std::getenv("PHOSPHOR_RT_PROXY_EXPORT_DIR"));
        std::filesystem::create_directories(directory);
        const auto path = directory / "sponza.rtproxy.json";
        std::string error;
        if (!phosphor::rtWriteProxyManifest(path.string(), *exportManifest, error)) throw std::runtime_error(error);
        rep.note("Measured production RT proxy manifest exported: " + path.string());
        ctx.log("S4 exported measured production proxy: %s", path.c_str());
    }
    return R;
}

// ---------------------------------------------------------------------------
// Scenes
// ---------------------------------------------------------------------------

/// Procedural corpus plus a ground quad and a 5 mm thick plate (two grids)
/// above the first sphere: thin geometry whose shadow must survive.
void proceduralThin(SceneData& s) {
    proceduralScene(s);
    auto addInstance = [&](phosphor::MeshHandle h, glm::vec3 pos) {
        phosphor::GPUInstance inst{};
        const glm::mat4 w = glm::translate(glm::mat4(1.0f), pos);
        std::memcpy(inst.modelMatrix, glm::value_ptr(w), sizeof(inst.modelMatrix));
        inst.meshIndex = h;
        inst.materialIndex = 0;
        inst.flags = phosphor::INSTANCE_FLAG_VALID;
        inst.generation = u32(s.instances.size()) + 1;
        s.instances.push_back(inst);
    };
    {
        std::vector<glm::vec3> pos;
        std::vector<u32> idx;
        constexpr u32 N = 16;
        for (u32 j = 0; j <= N; ++j)
            for (u32 i = 0; i <= N; ++i)
                pos.push_back({-8.0f + 30.0f * float(i) / N, 0.0f, -10.0f + 20.0f * float(j) / N});
        for (u32 j = 0; j < N; ++j)
            for (u32 i = 0; i < N; ++i) {
                const u32 a = j * (N + 1) + i, b = a + 1, c = a + N + 1, d = c + 1;
                idx.insert(idx.end(), {a, c, b, b, c, d});
            }
        addInstance(addMesh(s, pos, idx), {0, -1.0f, 0});
    }
    {
        std::vector<glm::vec3> pos;
        std::vector<u32> idx;
        constexpr u32 N = 24;
        for (int layer = 0; layer < 2; ++layer)
            for (u32 j = 0; j <= N; ++j)
                for (u32 i = 0; i <= N; ++i) {
                    // Slight dish so the simplifier has a reason to move vertices.
                    const float x = -1.0f + 2.0f * float(i) / N, z = -1.0f + 2.0f * float(j) / N;
                    pos.push_back({x, (layer ? -0.0025f : 0.0025f) + 0.02f * (x * x + z * z), z});
                }
        const u32 L = (N + 1) * (N + 1);
        for (u32 j = 0; j < N; ++j)
            for (u32 i = 0; i < N; ++i) {
                const u32 a = j * (N + 1) + i, b = a + 1, c = a + N + 1, d = c + 1;
                idx.insert(idx.end(), {a, c, b, b, c, d});
                idx.insert(idx.end(), {L + a, L + b, L + c, L + b, L + d, L + c});
            }
        addInstance(addMesh(s, pos, idx), {0, 2.2f, 0});
    }
}

void bench(soc::Context& ctx, soc::Report& rep) {
    Thresholds thr;
    rep.value("proxy.thr.shadow_pct", "%", thr.shadowPct, {}, false);
    rep.value("proxy.thr.prim_bad_pct", "%", thr.primBadPct, {}, false);
    rep.value("proxy.thr.dt95_cm", "cm", thr.dt95Cm, {}, false);
    rep.value("proxy.thr.acne_pct", "%", thr.acnePct, {}, false);
    Tracer tr(ctx);
    const u32 W = ctx.quick() ? 480 : 960, H = ctx.quick() ? 270 : 540;
    const V3 sun = normalize({0.30, 0.85, 0.20});
    std::vector<SceneResult> results;

    SceneData sp;
    std::string err;
    if (loadSponza(sp, err)) {
        std::vector<Ray> rays = cameraRays({-8, 2, 0.5}, {8, 4, -0.5}, 70.0, W, H);
        if (!ctx.quick()) {
            for (const Ray& r : cameraRays({8, 1.6, -3.5}, {-8, 3, 3}, 70.0, W, H)) rays.push_back(r);
            for (const Ray& r : cameraRays({0, 6, 0}, {-10, 1, 2}, 75.0, W, H)) rays.push_back(r);
        }
        V3 lo, hi;
        sp.bounds(lo, hi);
        ctx.log("S4 sponza bounds (%.2f %.2f %.2f) .. (%.2f %.2f %.2f)", lo.x, lo.y, lo.z, hi.x, hi.y, hi.z);
        results.push_back(runScene(ctx, rep, tr, "sponza", sp, rays, sun, thr));
    } else {
        rep.note("Sponza skipped: " + err);
    }

    SceneData pr;
    proceduralThin(pr);
    {
        const std::vector<Ray> rays = cameraRays({7, 4.5, 9}, {5, 0, 0}, 65.0, W, H);
        results.push_back(runScene(ctx, rep, tr, "proc", pr, rays, sun, thr));
    }

    bool ok = !results.empty();
    std::string detail;
    for (const SceneResult& r : results) {
        ok = ok && r.identityExact && r.s01Violates && r.sabotageDetected && r.policyMet[0] && r.policyMet[1] && r.policyMet[2];
        detail += r.detail + " | ";
    }
    rep.negative(ok, detail);
    if (!ok) rep.status(soc::Status::Failed, "controls or policy failed: " + detail);
}

} // namespace

SOC_BENCH("F9-S4", "proxy", "RT proxy geometry (meshoptimizer): hit and shadow error vs full geometry", bench);

} // namespace f9
