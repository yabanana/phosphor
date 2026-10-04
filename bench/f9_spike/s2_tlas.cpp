// F9-S2: per-frame TLAS written by compute from the engine's GPU scene.
//
// A GPUInstance[] buffer (engine layout, capacity >= live count, ~5% invalid
// slots) is animated by a kernel; a second kernel writes
// MTLIndirectAccelerationStructureInstanceDescriptor[] from the slots; the TLAS
// is built or refitted from them.  Three descriptor strategies:
//   A   one descriptor per slot (invalid slots: mask 0), instanceCount = capacity,
//       MTL4::InstanceAccelerationStructureDescriptor; instance_id == slot;
//   B   compaction with an atomic counter (userID = slot, unstable order),
//       MTL4::IndirectInstanceAccelerationStructureDescriptor with an
//       instanceCountBuffer written by the GPU (no CPU readback in the timed path);
//   Bs  the same with a stable compaction (tile counts + scan, slot order).
// For each size and strategy it measures the descriptor kernel, TLAS build
// (usage None / PreferFastBuild / Refit) and refit, the combined update spans, and
// the AS/scratch bytes; and it checks every TLAS (after build, after one refit,
// after five refit frames, after delete/reuse of recycled slots) against an exact
// CPU two-level reference (per-mesh double-precision BVH + the GPU-animated
// matrices read back): hit/miss, t, instance, userID, generation, mesh, descriptor
// index, front face.  A separate experiment compares triangle_front_facing with
// the world-space winding for mirrored instances under every instance option.
//
// Negative controls (each must FAIL its check):
//   NC1 4x3 transform written transposed -> wrong hits;
//   NC2 deleted slots with mask 0xFF -> hits on deleted instances;
//   NC3 the opposite front-face rule on mirrored instances -> front mismatches.
#include "f9_common.h"

#include "scene/procedural.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cstring>
#include <memory>

namespace f9 {
namespace {

using phosphor::GPUInstance;

// ---- shared layouts (mirror shaders/s2_tlas.metal) ----------------------------------------
struct S2Args {
    u64 inst, base, desc, blas, count, blockCount;
    u32 capacity, numMeshes;
};
static_assert(sizeof(S2Args) == 56);

struct S2Mode {
    float time = 0;
    u32 ccwMode = 0;
    u32 maskInvalid = 0;
    u32 transposeBug = 0;
    u32 mask = 0xFF;
    u32 pad[3] = {0, 0, 0};
};
static_assert(sizeof(S2Mode) == 32);

struct S2Hit {
    float t;
    u32 instance, userID, primitive, generation, mesh, front, flags;
};
static_assert(sizeof(S2Hit) == 32);

enum Strat { kA = 0, kB = 1, kBs = 2 };
const char* const kStratName[3] = {"A", "B", "Bs"};
enum Use { kNone = 0, kFast = 1, kRefit = 2 };
const char* const kUseName[3] = {"none", "fast", "refit"};
constexpr MTL::AccelerationStructureUsage kUseFlag[3] = {MTL::AccelerationStructureUsageNone,
                                                         MTL::AccelerationStructureUsagePreferFastBuild,
                                                         MTL::AccelerationStructureUsageRefit};

constexpr u32 kMeshes = 4;
constexpr u32 kTile = 256;

using phosphor::INSTANCE_FLAG_MIRRORED;
using phosphor::INSTANCE_FLAG_VALID;

// ---- meshes ---------------------------------------------------------------------------------
struct Meshes {
    SceneData s;
    GpuGeometry g;
    std::vector<Blas> blases;
    MTL::Buffer* table = nullptr; // u64 raw resource id per mesh
    std::vector<std::unique_ptr<TriangleSoup>> soups;
    std::vector<std::unique_ptr<CpuBvh>> bvhs;
    std::vector<f64> radius;
};

void buildMeshes(soc::Context& ctx, Meshes& M) {
    using namespace phosphor::ProceduralMeshes;
    const std::vector<phosphor::MeshData> meshes = {generateCube(1.0f), generateSphere(1.0f, 32, 16),
                                                    generateTorus(1.0f, 0.35f, 32, 16),
                                                    generateIcosahedron(1.0f, 2, false)};
    {
        QuietStderr quiet; // uploadMesh logs with LOG_INFO
        for (const phosphor::MeshData& m : meshes) M.s.scene.uploadMesh(m.positions, m.normals, m.tangents, m.uvs, m.indices);
    }
    if (M.s.scene.getMeshCount() != kMeshes) throw soc::BenchError("unexpected mesh count");
    M.g = uploadGeometry(ctx, M.s);
    M.blases = allocateBlases(ctx, M.s, M.g, {});
    MTL::Buffer* scratch = scratchFor(ctx, M.blases);
    buildBlases(ctx, M.blases, scratch);
    M.table = ctx.buffer(kMeshes * sizeof(u64));
    auto* t = static_cast<u64*>(M.table->contents());
    const auto& infos = M.s.scene.meshInfos();
    const auto& idx = M.s.scene.indices();
    for (u32 m = 0; m < kMeshes; ++m) {
        t[m] = M.blases[m].as->gpuResourceID()._impl;
        auto soup = std::make_unique<TriangleSoup>();
        f64 r = 0;
        for (u32 tri = 0; tri < infos[m].indexCount / 3; ++tri) {
            V3 p[3];
            for (u32 k = 0; k < 3; ++k) {
                p[k] = M.s.vertex(m, idx[infos[m].indexOffset + 3 * tri + k]);
                r = std::max(r, length(p[k]));
            }
            soup->add(p[0], p[1], p[2], 0, tri, 0);
        }
        M.radius.push_back(r);
        M.bvhs.push_back(std::make_unique<CpuBvh>(*soup));
        M.soups.push_back(std::move(soup));
    }
    ctx.commitResidency();
}

// ---- GPU pipelines ----------------------------------------------------------------------------
struct Psos {
    MTL::ComputePipelineState *animate, *descA, *reset, *descB, *countBlocks, *scan, *writeBs, *trace;
};

Psos makePsos(soc::Context& ctx) {
    MTL::Library* lib = f9Library(ctx, "s2_tlas.metal");
    Psos p;
    p.animate = ctx.compute(lib, "s2_animate");
    p.descA = ctx.compute(lib, "s2_desc_a");
    p.reset = ctx.compute(lib, "s2_reset");
    p.descB = ctx.compute(lib, "s2_desc_b");
    p.countBlocks = ctx.compute(lib, "s2_count_blocks");
    p.scan = ctx.compute(lib, "s2_scan_blocks");
    p.writeBs = ctx.compute(lib, "s2_write_bs");
    p.trace = ctx.compute(lib, "s2_trace");
    if (p.scan->maxTotalThreadsPerThreadgroup() < 1024) throw soc::BenchError("scan kernel needs 1024 threads");
    return p;
}

// ---- world --------------------------------------------------------------------------------------
struct TlasV {
    MTL::AccelerationStructure* as = nullptr;
    MTL4::AccelerationStructureDescriptor* desc = nullptr;
    MTL::AccelerationStructureSizes sizes{};
};

struct World {
    u32 N = 0, capacity = 0;
    MTL::Buffer *inst = nullptr, *base = nullptr, *desc = nullptr, *count = nullptr, *blocks = nullptr, *args = nullptr,
                *params = nullptr, *rayBuf = nullptr, *hitBuf = nullptr, *cntBuf = nullptr, *scratch = nullptr;
    float time = 0;
    u32 paramNext = 0;
    std::vector<GPUInstance> baseCpu, origBase, cur;
    std::vector<glm::dmat4> inv;
    std::vector<f64> rad;
    std::vector<u32> liveSlots;
    TlasV tl[3][3];
    static constexpr u32 kMaxRays = 16384;
};

constexpr u32 kParamSlots = 1024, kParamStride = 256;

u64 pushMode(World& W, const S2Mode& m) {
    const u32 i = W.paramNext++ % kParamSlots;
    std::memcpy(static_cast<char*>(W.params->contents()) + size_t(i) * kParamStride, &m, sizeof m);
    return W.params->gpuAddress() + u64(i) * kParamStride;
}

void disp(soc::Context& ctx, World& W, MTL4::ComputeCommandEncoder* e, MTL::ComputePipelineState* pso, u32 n, u32 tg,
          const S2Mode* m) {
    ctx.table()->setAddress(W.args->gpuAddress(), 0);
    if (m) ctx.table()->setAddress(pushMode(W, *m), 1);
    e->setComputePipelineState(pso);
    e->setArgumentTable(ctx.table());
    e->dispatchThreads(MTL::Size::Make(n, 1, 1), MTL::Size::Make(tg, 1, 1));
}

void barrierDD(MTL4::ComputeCommandEncoder* e) {
    e->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
}
void barrierDA(MTL4::ComputeCommandEncoder* e) {
    e->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageAccelerationStructure, MTL4::VisibilityOptionDevice);
}

void encAnimate(soc::Context& ctx, World& W, const Psos& P, MTL4::ComputeCommandEncoder* e) {
    S2Mode m;
    m.time = W.time;
    disp(ctx, W, e, P.animate, W.capacity, 64, &m);
}

/// The descriptor kernels of strategy `s` (dependent dispatches separated by barriers).
void encDescriptors(soc::Context& ctx, World& W, const Psos& P, MTL4::ComputeCommandEncoder* e, Strat s, S2Mode m) {
    switch (s) {
    case kA:
        disp(ctx, W, e, P.descA, W.capacity, 64, &m);
        break;
    case kB:
        disp(ctx, W, e, P.reset, 1, 1, nullptr);
        barrierDD(e);
        disp(ctx, W, e, P.descB, W.capacity, 64, &m);
        break;
    case kBs: {
        const u32 blocks = (W.capacity + kTile - 1) / kTile;
        disp(ctx, W, e, P.countBlocks, blocks * kTile, kTile, nullptr);
        barrierDD(e);
        disp(ctx, W, e, P.scan, 1024, 1024, nullptr);
        barrierDD(e);
        disp(ctx, W, e, P.writeBs, blocks * kTile, kTile, &m);
        break;
    }
    }
}

/// Untimed encoder + submit.
void run(soc::Context& ctx, const std::function<void(MTL4::ComputeCommandEncoder*)>& fn) {
    MTL4::CommandBuffer* cmd = ctx.beginCommands();
    MTL4::ComputeCommandEncoder* e = cmd->computeCommandEncoder();
    fn(e);
    e->endEncoding();
    ctx.submit();
}

void encBuild(World& W, MTL4::ComputeCommandEncoder* e, const TlasV& t) {
    e->buildAccelerationStructure(t.as, t.desc, range(W.scratch));
}
void encRefit(World& W, MTL4::ComputeCommandEncoder* e, const TlasV& t) {
    e->refitAccelerationStructure(t.as, t.desc, t.as, range(W.scratch));
}

// ---- CPU model ----------------------------------------------------------------------------------
V3 xf(const glm::dmat4& m, V3 p) {
    const glm::dvec4 q = m * glm::dvec4(p.x, p.y, p.z, 1.0);
    return {q.x, q.y, q.z};
}
V3 xfDir(const glm::dmat4& m, V3 d) {
    const glm::dvec4 q = m * glm::dvec4(d.x, d.y, d.z, 0.0);
    return {q.x, q.y, q.z};
}

/// Re-read the animated GPU scene and rebuild the CPU mirror.
void readback(World& W, const Meshes& M) {
    std::memcpy(W.cur.data(), W.inst->contents(), size_t(W.capacity) * sizeof(GPUInstance));
    W.liveSlots.clear();
    parallelFor(W.capacity, [&](u32 i) {
        W.inv[i] = glm::inverse(glm::dmat4(glm::make_mat4(W.cur[i].modelMatrix)));
        const float* m = W.cur[i].modelMatrix;
        f64 fro = 0;
        for (int c = 0; c < 3; ++c)
            for (int r = 0; r < 3; ++r) fro += f64(m[4 * c + r]) * f64(m[4 * c + r]);
        W.rad[i] = M.radius[W.baseCpu[i].meshIndex] * std::sqrt(fro) * 1.0001 + 1e-6;
    });
    for (u32 i = 0; i < W.capacity; ++i)
        if (W.baseCpu[i].flags & INSTANCE_FLAG_VALID) W.liveSlots.push_back(i);
}

struct RefHit {
    f64 t = -1;
    u32 slot = ~0u, prim = ~0u;
};

RefHit refNearest(const World& W, const Meshes& M, const Ray& r) {
    RefHit best;
    const f64 dd = dot(r.d, r.d);
    for (u32 slot : W.liveSlots) {
        const float* m = W.cur[slot].modelMatrix;
        const V3 w = V3{m[12], m[13], m[14]} - r.o;
        const f64 tcl = dot(w, r.d) / dd;
        const f64 R = W.rad[slot];
        const f64 d2 = dot(w, w) - tcl * tcl * dd;
        if (d2 > R * R) continue;
        if (tcl * std::sqrt(dd) < -R) continue;
        Ray o;
        o.o = xf(W.inv[slot], r.o);
        o.d = xfDir(W.inv[slot], r.d);
        o.tmin = 0;
        o.tmax = 1e300;
        const CpuHit h = M.bvhs[W.baseCpu[slot].meshIndex]->nearest(o);
        if (!h.hit()) continue;
        if (best.t < 0 || h.t < best.t || (h.t == best.t && slot < best.slot)) {
            best.t = h.t;
            best.slot = slot;
            best.prim = h.tri;
        }
    }
    return best;
}

/// Rays aimed at `targets` (slots): 75% from outside towards the centre, 25% from (near) the centre outwards.
std::vector<Ray> makeRays(const World& W, const std::vector<u32>& targets, u32 seed, f64 insideFrac = 0.25) {
    std::vector<Ray> rays;
    rays.reserve(targets.size());
    for (u32 i = 0; i < targets.size(); ++i) {
        const u32 slot = targets[i];
        const float* m = W.cur[slot].modelMatrix;
        const V3 c{m[12], m[13], m[14]};
        auto unit = [&](u32 salt) {
            const f64 z = 2.0 * rnd01(seed + i, salt) - 1.0;
            const f64 a = 6.283185307179586 * rnd01(seed + i, salt + 1);
            const f64 s = std::sqrt(std::max(0.0, 1.0 - z * z));
            return V3{s * std::cos(a), s * std::sin(a), z};
        };
        Ray r;
        r.tmin = 0;
        r.tmax = 1e30;
        if (rnd01(seed + i, 90) < insideFrac) {
            r.o = c + unit(30) * 0.05;
            r.d = unit(40);
        } else {
            r.o = c + unit(10) * (3.0 + 5.0 * rnd01(seed + i, 20));
            const V3 aim = c + unit(50) * (0.5 * rnd01(seed + i, 60));
            r.d = normalize(aim - r.o);
        }
        rays.push_back(r);
    }
    return rays;
}

std::vector<u32> randomLive(const World& W, u32 n, u32 seed) {
    std::vector<u32> t(n);
    for (u32 i = 0; i < n; ++i)
        t[i] = W.liveSlots[std::min<u32>(u32(rnd01(seed, i) * float(W.liveSlots.size())), u32(W.liveSlots.size() - 1))];
    return t;
}

// ---- tracing + checking --------------------------------------------------------------------
std::vector<S2Hit> trace(soc::Context& ctx, World& W, const Psos& P, MTL::AccelerationStructure* as,
                         const std::vector<Ray>& rays, u32 mask = 0xFF) {
    if (rays.size() > World::kMaxRays) throw soc::BenchError("too many rays");
    auto* rb = static_cast<GpuRay*>(W.rayBuf->contents());
    for (size_t i = 0; i < rays.size(); ++i) rb[i] = toGpu(rays[i]);
    *static_cast<u32*>(W.cntBuf->contents()) = u32(rays.size());
    S2Mode m;
    m.mask = mask;
    MTL4::CommandBuffer* cmd = ctx.beginCommands();
    MTL4::ComputeCommandEncoder* e = cmd->computeCommandEncoder();
    // The AS was built by an earlier (completed) submission; make its writes visible anyway.
    e->barrierAfterEncoderStages(MTL::StageAccelerationStructure, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
    ctx.table()->setResource(as->gpuResourceID(), 0);
    ctx.table()->setAddress(W.args->gpuAddress(), 1);
    ctx.table()->setAddress(W.rayBuf->gpuAddress(), 2);
    ctx.table()->setAddress(W.hitBuf->gpuAddress(), 3);
    ctx.table()->setAddress(pushMode(W, m), 4);
    ctx.table()->setAddress(W.cntBuf->gpuAddress(), 5);
    e->setComputePipelineState(P.trace);
    e->setArgumentTable(ctx.table());
    e->dispatchThreads(MTL::Size::Make(u32(rays.size()), 1, 1), MTL::Size::Make(64, 1, 1));
    e->endEncoding();
    ctx.submit();
    std::vector<S2Hit> out(rays.size());
    std::memcpy(out.data(), W.hitBuf->contents(), rays.size() * sizeof(S2Hit));
    return out;
}

struct CheckStats {
    u32 rays = 0, wrong = 0, missMismatch = 0, tMismatch = 0, idMismatch = 0, tie = 0, descBad = 0, metaBad = 0,
        onDeleted = 0, hits = 0;
    u32 frontN[2] = {0, 0}, frontEq[2] = {0, 0}; // [mirrored]: rays whose hit agrees with the CPU / GPU front == CPU front
    std::string firstError;
    void add(const CheckStats& o) {
        rays += o.rays; wrong += o.wrong; missMismatch += o.missMismatch; tMismatch += o.tMismatch;
        idMismatch += o.idMismatch; tie += o.tie; descBad += o.descBad; metaBad += o.metaBad;
        onDeleted += o.onDeleted; hits += o.hits;
        for (int k = 0; k < 2; ++k) { frontN[k] += o.frontN[k]; frontEq[k] += o.frontEq[k]; }
        if (firstError.empty()) firstError = o.firstError;
    }
};

/// Compare GPU hits with the CPU two-level reference.  `desc` = descriptors the TLAS was built from,
/// `descCount` their number (A: capacity; B/Bs: the GPU count); strict instance mapping: A requires
/// instance_id == userID.
CheckStats checkHits(const World& W, const Meshes& M, const std::vector<Ray>& rays, const std::vector<S2Hit>& hits,
                     const InstanceDesc* desc, u32 descCount, Strat strat, f64 tol = 2e-4) {
    CheckStats c;
    c.rays = u32(rays.size());
    enum : u32 { kMiss = 1, kT = 2, kId = 4, kDesc = 8, kMeta = 16, kDel = 32, kTie = 64 };
    std::vector<u32> bits(rays.size(), 0);
    std::vector<signed char> fcls(rays.size(), -1), feq(rays.size(), 0);
    parallelFor(c.rays, [&](u32 i) {
        const Ray& r = rays[i];
        const S2Hit& g = hits[i];
        const RefHit ref = refNearest(W, M, r);
        const bool gHit = g.t >= 0;
        u32 b = 0;
        if ((ref.t >= 0) != gHit) { bits[i] = kMiss; return; }
        if (!gHit) return;
        const f64 scale = std::max(1.0, ref.t * length(r.d));
        const f64 err = std::fabs(ref.t - f64(g.t)) * length(r.d);
        // Same slot and primitive as the reference: the identity is right, only t is judged, with a wider bound
        // (grazing hits at |p| ~ 30: measured 3e-4 relative on one ray, identical for build and refit).
        const bool sameTri = g.userID == ref.slot && g.primitive == ref.prim;
        if (err > (sameTri ? 20.0 * tol : tol) * scale) b |= kT;
        // Descriptor mapping: instance_id is the descriptor index.
        if (g.instance >= descCount || desc[g.instance].userID != g.userID) b |= kDesc;
        if (strat == kA && g.instance != g.userID) b |= kDesc;
        const bool slotOk = g.userID < W.capacity;
        if (slotOk && !(W.baseCpu[g.userID].flags & INSTANCE_FLAG_VALID)) b |= kDel;
        if (slotOk && (g.generation != W.baseCpu[g.userID].generation || g.mesh != W.baseCpu[g.userID].meshIndex ||
                       g.flags != W.baseCpu[g.userID].flags))
            b |= kMeta;
        bool agree = !(b & (kT | kDesc | kDel)) && slotOk;
        if (!slotOk) b |= kId;
        else if (g.userID != ref.slot || g.primitive != ref.prim) {
            // Tie on a shared edge / equal t: the named triangle must contain the hit up to 2e-3 barycentric
            // (float transforms at |p| ~ 100 limit the GPU hit position to ~1e-5 absolute on 0.05-wide slivers).
            const u32 mesh = W.baseCpu[g.userID].meshIndex;
            const TriangleSoup& s = *M.soups[mesh];
            if (g.primitive >= s.count()) b |= kId;
            else {
                Ray o;
                o.o = xf(W.inv[g.userID], r.o);
                o.d = xfDir(W.inv[g.userID], r.d);
                o.tmin = 0;
                o.tmax = 1e300;
                const f64 tg = rayTriangleLoose(o, s.v[3 * size_t(g.primitive)], s.v[3 * size_t(g.primitive) + 1],
                                                s.v[3 * size_t(g.primitive) + 2], 2e-3);
                if (tg < 0 || std::fabs(tg - f64(g.t)) * length(r.d) > tol * scale) b |= kId;
                else if (g.userID != ref.slot && std::fabs(ref.t - tg) * length(r.d) > tol * scale) b |= kId;
                else b |= kTie;
            }
        }
        if (b & kId) agree = false;
        if (agree) {
            const u32 mesh = W.baseCpu[g.userID].meshIndex;
            const TriangleSoup& s = *M.soups[mesh];
            const glm::dmat4 mm = glm::dmat4(glm::make_mat4(W.cur[g.userID].modelMatrix));
            const V3 a = xf(mm, s.v[3 * size_t(g.primitive)]), bb = xf(mm, s.v[3 * size_t(g.primitive) + 1]),
                     cc = xf(mm, s.v[3 * size_t(g.primitive) + 2]);
            const V3 n = cross(bb - a, cc - a);
            const bool cpuFront = dot(n, r.d) < 0;
            fcls[i] = (W.baseCpu[g.userID].flags & INSTANCE_FLAG_MIRRORED) ? 1 : 0;
            feq[i] = (cpuFront == (g.front != 0)) ? 1 : 0;
        }
        bits[i] = b;
    });
    for (u32 i = 0; i < c.rays; ++i) {
        const u32 b = bits[i];
        c.hits += hits[i].t >= 0;
        if (fcls[i] >= 0) { ++c.frontN[fcls[i]]; c.frontEq[fcls[i]] += feq[i]; }
        if (b & kTie) ++c.tie;
        const u32 bad = b & ~u32(kTie);
        if (!bad) continue;
        ++c.wrong;
        if (bad & kMiss) ++c.missMismatch;
        if (bad & kT) ++c.tMismatch;
        if (bad & kId) ++c.idMismatch;
        if (bad & kDesc) ++c.descBad;
        if (bad & kMeta) ++c.metaBad;
        if (bad & kDel) ++c.onDeleted;
        if (c.firstError.empty()) {
            const RefHit ref = refNearest(W, M, rays[i]);
            char buf[400];
            f64 dbg1 = -9, dbg3 = -9;
            if (hits[i].userID < W.capacity && hits[i].primitive < M.soups[W.baseCpu[hits[i].userID].meshIndex]->count()) {
                const TriangleSoup& ss = *M.soups[W.baseCpu[hits[i].userID].meshIndex];
                Ray o;
                o.o = xf(W.inv[hits[i].userID], rays[i].o);
                o.d = xfDir(W.inv[hits[i].userID], rays[i].d);
                o.tmin = 0; o.tmax = 1e300;
                const size_t p0 = 3 * size_t(hits[i].primitive);
                dbg1 = rayTriangleLoose(o, ss.v[p0], ss.v[p0 + 1], ss.v[p0 + 2], 1e-5);
                dbg3 = rayTriangleLoose(o, ss.v[p0], ss.v[p0 + 1], ss.v[p0 + 2], 1e-3);
            }
            std::snprintf(buf, sizeof buf, "ray %u bits 0x%x: cpu t %.6g slot %d prim %d; gpu t %.6g inst %u user %u prim %u gen %u mesh %u [loose t eps1e-5 %.6g eps1e-3 %.6g]", i,
                          bad, ref.t, int(ref.slot), int(ref.prim), double(hits[i].t), hits[i].instance, hits[i].userID,
                          hits[i].primitive, hits[i].generation, hits[i].mesh, dbg1, dbg3);
            c.firstError = buf;
        }
    }
    return c;
}

CheckStats traceCheck(soc::Context& ctx, World& W, const Meshes& M, const Psos& P, MTL::AccelerationStructure* as,
                      Strat strat, const std::vector<Ray>& rays) {
    const std::vector<S2Hit> hits = trace(ctx, W, P, as, rays);
    const u32 descCount = strat == kA ? W.capacity : *static_cast<u32*>(W.count->contents());
    CheckStats c = checkHits(W, M, rays, hits, static_cast<const InstanceDesc*>(W.desc->contents()), descCount, strat);
    if (strat != kA && descCount != W.liveSlots.size()) {
        ++c.wrong;
        c.firstError = "GPU instance count " + std::to_string(descCount) + " != live " + std::to_string(W.liveSlots.size());
    }
    return c;
}

// ---- world construction ----------------------------------------------------------------------------
void makeWorld(soc::Context& ctx, const Meshes& M, const Psos& P, World& W, u32 N) {
    W.N = N;
    W.capacity = ((u32(double(N) * 1.0527) + 19) / 20) * 20;
    W.baseCpu.assign(W.capacity, GPUInstance{});
    const u32 side = u32(std::ceil(std::cbrt(double(W.capacity))));
    for (u32 i = 0; i < W.capacity; ++i) {
        GPUInstance& g = W.baseCpu[i];
        const glm::vec3 pos = glm::vec3(float(i % side), float((i / side) % side), float(i / (side * side))) * 3.5f +
                              glm::vec3(rnd01(i, 11), rnd01(i, 12), rnd01(i, 13)) * 0.6f - 0.3f;
        const float u1 = rnd01(i, 21), u2 = rnd01(i, 22), u3 = rnd01(i, 23);
        const float a = std::sqrt(1.0f - u1), b = std::sqrt(u1);
        const glm::quat q(b * std::cos(6.2831853f * u3), a * std::sin(6.2831853f * u2), a * std::cos(6.2831853f * u2),
                          b * std::sin(6.2831853f * u3));
        const float s = 0.5f + rnd01(i, 3);
        const bool mirrored = rnd01(i, 5) < 0.10f;
        const glm::mat4 m = glm::translate(glm::mat4(1.0f), pos) * glm::mat4_cast(glm::normalize(q)) *
                            glm::scale(glm::mat4(1.0f), glm::vec3(mirrored ? -s : s, s, s));
        std::memcpy(g.modelMatrix, glm::value_ptr(m), sizeof g.modelMatrix);
        g.meshIndex = std::min<u32>(u32(rnd01(i, 2) * float(kMeshes)), kMeshes - 1);
        g.materialIndex = 0;
        g.flags = ((i % 20) == 19 ? 0u : INSTANCE_FLAG_VALID) | (mirrored ? INSTANCE_FLAG_MIRRORED : 0u);
        g.generation = 1 + i;
    }
    W.origBase = W.baseCpu;
    W.cur.assign(W.capacity, GPUInstance{});
    W.inv.assign(W.capacity, glm::dmat4(1.0));
    W.rad.assign(W.capacity, 0.0);
    const size_t ib = size_t(W.capacity) * sizeof(GPUInstance);
    W.inst = ctx.buffer(ib);
    W.base = ctx.buffer(ib);
    W.desc = ctx.buffer(size_t(W.capacity) * sizeof(InstanceDesc) + 64);
    W.count = ctx.buffer(64);
    W.blocks = ctx.buffer(size_t((W.capacity + kTile - 1) / kTile + 1) * 4 + 64);
    W.args = ctx.buffer(256);
    W.params = ctx.buffer(size_t(kParamSlots) * kParamStride);
    W.rayBuf = ctx.buffer(size_t(World::kMaxRays) * sizeof(GpuRay));
    W.hitBuf = ctx.buffer(size_t(World::kMaxRays) * sizeof(S2Hit));
    W.cntBuf = ctx.buffer(64);
    std::memcpy(W.base->contents(), W.baseCpu.data(), ib);
    std::memset(W.inst->contents(), 0, ib);
    S2Args a{W.inst->gpuAddress(), W.base->gpuAddress(), W.desc->gpuAddress(), M.table->gpuAddress(),
             W.count->gpuAddress(), W.blocks->gpuAddress(), W.capacity, kMeshes};
    std::memcpy(W.args->contents(), &a, sizeof a);
    (void)P;
}

/// All TLAS variants of the world (strategy x usage) and one shared scratch.
void makeTlases(soc::Context& ctx, World& W) {
    u64 scratch = 4096;
    for (int s = 0; s < 3; ++s)
        for (int u = 0; u < 3; ++u) {
            TlasV& t = W.tl[s][u];
            if (s == kA) {
                auto* d = MTL4::InstanceAccelerationStructureDescriptor::alloc()->init();
                d->setInstanceCount(W.capacity);
                d->setInstanceDescriptorBuffer(range(W.desc));
                d->setInstanceDescriptorStride(sizeof(InstanceDesc));
                d->setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorTypeIndirect);
                d->setInstanceTransformationMatrixLayout(MTL::MatrixLayoutColumnMajor);
                d->setUsage(kUseFlag[u]);
                t.desc = d;
            } else {
                auto* d = MTL4::IndirectInstanceAccelerationStructureDescriptor::alloc()->init();
                d->setMaxInstanceCount(W.capacity);
                d->setInstanceCountBuffer(range(W.count, 0, 4));
                d->setInstanceDescriptorBuffer(range(W.desc));
                d->setInstanceDescriptorStride(sizeof(InstanceDesc));
                d->setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorTypeIndirect);
                d->setInstanceTransformationMatrixLayout(MTL::MatrixLayoutColumnMajor);
                d->setUsage(kUseFlag[u]);
                t.desc = d;
            }
            ctx.keep(t.desc);
            t.sizes = ctx.device()->accelerationStructureSizes(t.desc);
            t.as = newAccelerationStructure(ctx, t.sizes.accelerationStructureSize, AsPlacement::Device);
            scratch = std::max<u64>({scratch, t.sizes.buildScratchBufferSize, t.sizes.refitScratchBufferSize});
        }
    W.scratch = ctx.buffer(scratch, MTL::ResourceStorageModePrivate);
    ctx.commitResidency();
}

void uploadBase(World& W) {
    std::memcpy(W.base->contents(), W.baseCpu.data(), size_t(W.capacity) * sizeof(GPUInstance));
}

/// animate at W.time -> descriptors -> (barrier) -> fn, in one untimed submission; refreshes the CPU mirror.
void frame(soc::Context& ctx, World& W, const Meshes& M, const Psos& P, Strat s, const S2Mode& dm,
           const std::function<void(MTL4::ComputeCommandEncoder*)>& after = {}) {
    run(ctx, [&](MTL4::ComputeCommandEncoder* e) {
        encAnimate(ctx, W, P, e);
        barrierDD(e);
        encDescriptors(ctx, W, P, e, s, dm);
        if (after) {
            barrierDA(e);
            after(e);
        }
    });
    readback(W, M);
}

/// animate + descriptors only (untimed, no CPU mirror refresh).
void advance(soc::Context& ctx, World& W, const Psos& P, Strat s, const S2Mode& dm) {
    run(ctx, [&](MTL4::ComputeCommandEncoder* e) {
        encAnimate(ctx, W, P, e);
        barrierDD(e);
        encDescriptors(ctx, W, P, e, s, dm);
    });
}

std::string tagOf(u32 n) { return n >= 1000000 ? std::to_string(n / 1000000) + "m" : std::to_string(n / 1000) + "k"; }

struct Findings {
    std::string failures;
    std::string info;
    void fail(const std::string& s) { failures += s + "; "; }
};

} // namespace

namespace {

// ---- one size, one strategy ----------------------------------------------------------------------
struct NegState {
    u32 nc2OnDeleted = 0, nc2Wrong = 0, nc2Rays = 0;
};

void runStrategy(soc::Context& ctx, soc::Report& rep, World& W, const Meshes& M, const Psos& P, Strat s, Findings& F,
                 NegState& neg) {
    const std::string tag = tagOf(W.N);
    const std::string pre = "tlas." + tag + "." + kStratName[s] + ".";
    const u32 nRays = ctx.quick() ? 2048 : 4096;
    const std::map<std::string, double> prm = {{"instances", double(W.N)}, {"capacity", double(W.capacity)}};
    W.time = 0.0f;
    S2Mode dm;

    // ---- sizes ------------------------------------------------------------------------------------
    for (int u = 0; u < 3; ++u) {
        const TlasV& t = W.tl[s][u];
        rep.value(pre + (u == kNone ? "bytes" : std::string("bytes_") + kUseName[u]), "B", double(t.sizes.accelerationStructureSize),
                  prm, false);
        rep.value(pre + std::string("scratch_build_") + kUseName[u], "B", double(t.sizes.buildScratchBufferSize), prm, false);
    }
    rep.value(pre + "scratch_refit", "B", double(W.tl[s][kRefit].sizes.refitScratchBufferSize), prm, false);

    // ---- correctness: build of the three usages -------------------------------------------------------
    frame(ctx, W, M, P, s, dm);
    {
        const std::vector<Ray> rays = makeRays(W, randomLive(W, nRays, 1000 + s), 5000 + s);
        CheckStats tot;
        for (int u = 0; u < 3; ++u) {
            run(ctx, [&](MTL4::ComputeCommandEncoder* e) { encBuild(W, e, W.tl[s][u]); });
            tot.add(traceCheck(ctx, W, M, P, W.tl[s][u].as, s, rays));
        }
        rep.value(pre + "wrong.build", "rays", double(tot.wrong), {{"rays", double(tot.rays)}, {"hits", double(tot.hits)}, {"ties", double(tot.tie)}}, false);
        if (tot.wrong) F.fail(std::string(kStratName[s]) + " " + tag + " build: " + tot.firstError);
    }
    // ---- correctness: refit after 1 frame, after 5 frames ------------------------------------------------
    std::vector<Ray> rays5;
    for (int f = 1; f <= 5; ++f) {
        W.time += 0.37f;
        frame(ctx, W, M, P, s, dm, [&](MTL4::ComputeCommandEncoder* e) { encRefit(W, e, W.tl[s][kRefit]); });
        if (f == 1 || f == 5) {
            const std::vector<Ray> rays = makeRays(W, randomLive(W, nRays, 2000 + f + s), 6000 + f + s);
            if (f == 5) rays5 = rays;
            const CheckStats c = traceCheck(ctx, W, M, P, W.tl[s][kRefit].as, s, rays);
            rep.value(pre + (f == 1 ? "wrong.refit1" : "wrong.refit5"), "rays", double(c.wrong), {{"rays", double(c.rays)}, {"hits", double(c.hits)}}, false);
            if (c.wrong) {
                // Atomic compaction reorders descriptors every frame: a failing refit there is a finding, not a bug.
                if (s == kB) F.info += std::string("B refit frame ") + std::to_string(f) + " " + tag + " wrong " + std::to_string(c.wrong) + "/" + std::to_string(c.rays) + " (" + c.firstError + "); ";
                else F.fail(std::string(kStratName[s]) + " " + tag + " refit frame " + std::to_string(f) + ": " + c.firstError);
            }
        }
    }
    // A fresh build at the same frame must agree too (the refit AS may be compared with a rebuild).
    {
        run(ctx, [&](MTL4::ComputeCommandEncoder* e) { encBuild(W, e, W.tl[s][kFast]); });
        const CheckStats c = traceCheck(ctx, W, M, P, W.tl[s][kFast].as, s, rays5); // same rays as the refit check
        rep.value(pre + "wrong.rebuild5", "rays", double(c.wrong), {{"rays", double(c.rays)}}, false);
        if (c.wrong) F.fail(std::string(kStratName[s]) + " " + tag + " rebuild frame 5: " + c.firstError);
    }

    // ---- delete / reuse of recycled slots ----------------------------------------------------------------
    {
        std::vector<u32> deleted, reused, gone;
        std::vector<V3> goneCenter;
        for (u32 slot : W.liveSlots)
            if (rnd01(slot, 77 + s) < 0.10f) deleted.push_back(slot);
        const std::vector<GPUInstance> keep = W.baseCpu;
        const std::vector<GPUInstance> curBefore = W.cur;
        for (u32 k = 0; k < deleted.size(); ++k) (k & 1 ? reused : gone).push_back(deleted[k]);
        for (u32 slot : deleted) W.baseCpu[slot].flags &= ~INSTANCE_FLAG_VALID;
        std::vector<u32> newMesh(W.capacity, 0), newGen(W.capacity, 0);
        for (u32 slot : reused) {
            GPUInstance& g = W.baseCpu[slot];
            g.meshIndex = (g.meshIndex + 1 + u32(rnd01(slot, 88) * 3.0f)) % kMeshes;
            if (g.meshIndex == keep[slot].meshIndex) g.meshIndex = (g.meshIndex + 1) % kMeshes;
            g.generation = keep[slot].generation + 1000;
            g.flags |= INSTANCE_FLAG_VALID;
            newMesh[slot] = g.meshIndex;
            newGen[slot] = g.generation;
        }
        uploadBase(W);
        frame(ctx, W, M, P, s, dm); // same time: unchanged slots keep their pose
        // Rays at the old positions of the deleted slots and at the reused ones.
        World centers; // only cur is used by makeRays
        centers.cur = curBefore;
        std::vector<Ray> rays = makeRays(centers, gone, 8000 + s, 0.0);
        const std::vector<Ray> rr = makeRays(centers, reused, 9000 + s, 0.0);
        rays.insert(rays.end(), rr.begin(), rr.end());
        std::vector<u32> targets = gone;
        targets.insert(targets.end(), reused.begin(), reused.end());
        const std::vector<Ray> extra = makeRays(W, randomLive(W, nRays / 2, 4000 + s), 10000 + s);
        rays.insert(rays.end(), extra.begin(), extra.end());
        if (rays.size() > World::kMaxRays) rays.resize(World::kMaxRays);
        run(ctx, [&](MTL4::ComputeCommandEncoder* e) { encBuild(W, e, W.tl[s][kNone]); });
        const std::vector<S2Hit> hits = trace(ctx, W, P, W.tl[s][kNone].as, rays);
        const u32 descCount = s == kA ? W.capacity : *static_cast<u32*>(W.count->contents());
        CheckStats c = checkHits(W, M, rays, hits, static_cast<const InstanceDesc*>(W.desc->contents()), descCount, s);
        u32 goneHitDeleted = 0, reuseOnTarget = 0, reuseHitNewMesh = 0, reuseRays = 0;
        for (u32 i = 0; i < targets.size() && i < rays.size(); ++i) {
            const bool isReused = i >= gone.size();
            if (!isReused) goneHitDeleted += hits[i].t >= 0 && hits[i].userID == targets[i];
            else {
                ++reuseRays;
                if (hits[i].t >= 0 && hits[i].userID == targets[i]) {
                    ++reuseOnTarget;
                    reuseHitNewMesh += hits[i].mesh == newMesh[targets[i]] && hits[i].generation == newGen[targets[i]];
                }
            }
        }
        rep.value(pre + "wrong.delete_reuse", "rays", double(c.wrong), {{"rays", double(c.rays)}, {"deleted", double(gone.size())}, {"reused", double(reused.size())}, {"hit_on_deleted", double(c.onDeleted + goneHitDeleted)}}, false);
        rep.value(pre + "reuse.on_target", "rays", double(reuseOnTarget), {{"aimed", double(reuseRays)}, {"new_mesh_and_gen", double(reuseHitNewMesh)}}, true);
        if (c.wrong) F.fail(std::string(kStratName[s]) + " " + tag + " delete/reuse: " + c.firstError);
        if (goneHitDeleted) F.fail(std::string(kStratName[s]) + " " + tag + " ray aimed at a deleted slot hit it");
        if (reuseRays > 4 && (reuseOnTarget == 0 || reuseHitNewMesh != reuseOnTarget))
            F.fail(std::string(kStratName[s]) + " " + tag + " reused slots: " + std::to_string(reuseHitNewMesh) + "/" + std::to_string(reuseOnTarget) + " hits carry the new mesh+generation");

        if (s == kA) {
            // NC2: deleted slots not masked out (mask 0xFF) must produce hits on deleted instances.
            S2Mode bad = dm;
            bad.maskInvalid = 0xFF;
            run(ctx, [&](MTL4::ComputeCommandEncoder* e) {
                encDescriptors(ctx, W, P, e, s, bad);
                barrierDA(e);
                encBuild(W, e, W.tl[s][kNone]);
            });
            const std::vector<S2Hit> h2 = trace(ctx, W, P, W.tl[s][kNone].as, rays);
            const CheckStats c2 = checkHits(W, M, rays, h2, static_cast<const InstanceDesc*>(W.desc->contents()), descCount, s);
            neg.nc2OnDeleted += c2.onDeleted;
            neg.nc2Wrong += c2.wrong;
            neg.nc2Rays += c2.rays;
            rep.value(pre + "nc2.hits_on_deleted", "rays", double(c2.onDeleted), {{"wrong", double(c2.wrong)}, {"rays", double(c2.rays)}}, false);
            // Refit of the Refit-usage TLAS after the masks changed (information: is a mask change legal in a refit?).
            run(ctx, [&](MTL4::ComputeCommandEncoder* e) {
                encDescriptors(ctx, W, P, e, s, dm);
                barrierDA(e);
                encRefit(W, e, W.tl[s][kRefit]);
            });
            const CheckStats c3 = traceCheck(ctx, W, M, P, W.tl[s][kRefit].as, s, rays);
            rep.value(pre + "info.refit_after_delete.wrong", "rays", double(c3.wrong), {{"rays", double(c3.rays)}}, false);
            if (c3.wrong) F.info += std::string("A refit after delete/reuse ") + tag + " wrong " + std::to_string(c3.wrong) + "/" + std::to_string(c3.rays) + "; ";
        }
        // restore the original scene (valid flags, meshes, generations)
        W.baseCpu = keep;
        uploadBase(W);
        frame(ctx, W, M, P, s, dm);
    }

    // ---- timing ----------------------------------------------------------------------------------------------
    // The whole build span: descriptors are written first (untimed) so the TLAS builds see real data.
    W.time = 0.0f;
    frame(ctx, W, M, P, s, dm);
    ctx.keepWarm();
    {
        // animate kernel (strategy independent; reported under tlas.<size>.animate.ms once)
        if (s == kA) {
            const soc::Stats st = ctx.measure([&] {
                soc::ComputeTimer t(ctx);
                MTL4::ComputeCommandEncoder* e = t.begin();
                encAnimate(ctx, W, P, e);
                t.lap();
                return t.finish()[0];
            });
            rep.metric("tlas." + tag + ".animate.ms", "ms", st, prm, false);
        }
        // descriptor kernels (reset + write [+ scan] included)
        const soc::Stats sd = ctx.measure([&] {
            soc::ComputeTimer t(ctx);
            MTL4::ComputeCommandEncoder* e = t.begin();
            encDescriptors(ctx, W, P, e, s, dm);
            t.lap();
            return t.finish()[0];
        });
        rep.metric(pre + "descriptors.ms", "ms", sd, prm, false);
    }
    for (int u = 0; u < 3; ++u) {
        const TlasV& t = W.tl[s][u];
        const soc::Stats sb = ctx.measure([&] { return timeEncoder(ctx, [&](MTL4::ComputeCommandEncoder* e) { encBuild(W, e, t); }); });
        rep.metric(pre + "build_" + kUseName[u] + ".ms", "ms", sb, prm, false);
    }
    {
        // refit: after one frame of motion (animate + descriptors untimed, the refit alone timed)
        const TlasV& t = W.tl[s][kRefit];
        run(ctx, [&](MTL4::ComputeCommandEncoder* e) { encBuild(W, e, t); });
        const soc::Stats sr = ctx.measure([&] {
            W.time += 0.37f;
            advance(ctx, W, P, s, dm);
            return timeEncoder(ctx, [&](MTL4::ComputeCommandEncoder* e) { encRefit(W, e, t); });
        });
        rep.metric(pre + "refit.ms", "ms", sr, prm, false);
    }
    // Combined spans: descriptors + barrier + AS command in one encoder (what the engine would pay per frame).
    for (int u = 0; u < 3; ++u) {
        const TlasV& t = W.tl[s][u];
        const soc::Stats su = ctx.measure([&] {
            if (u == kRefit) { W.time += 0.37f; advance(ctx, W, P, s, dm); }
            return timeEncoder(ctx, [&](MTL4::ComputeCommandEncoder* e) {
                encDescriptors(ctx, W, P, e, s, dm);
                barrierDA(e);
                if (u == kRefit) encRefit(W, e, t);
                else encBuild(W, e, t);
            });
        });
        rep.metric(pre + std::string("update_") + (u == kRefit ? "refit" : kUseName[u]) + ".ms", "ms", su, prm, false);
    }
    ctx.keepWarm();
    ctx.log("S2 %s %s: done", tag.c_str(), kStratName[s]);
}

// ---- front-face experiment -----------------------------------------------------------------------------
struct FrontResult {
    u32 n[4][2] = {}, eq[4][2] = {}; // [ccwMode][mirrored]
};

void frontExperiment(soc::Context& ctx, soc::Report& rep, World& W, const Meshes& M, const Psos& P, Findings& F,
                     FrontResult& fr, bool& pure) {
    W.time = 0.0f;
    S2Mode dm;
    frame(ctx, W, M, P, kA, dm);
    const u32 nRays = ctx.quick() ? 4096 : 8192;
    // Half the rays start inside the meshes (back faces), half outside (front faces).
    const std::vector<Ray> rays = makeRays(W, randomLive(W, nRays, 31337), 777, 0.5);
    const char* modeName[4] = {"none", "ccw_all", "ccw_mirrored_only", "ccw_nonmirrored_only"};
    pure = true;
    for (u32 mode = 0; mode < 4; ++mode) {
        S2Mode m = dm;
        m.ccwMode = mode;
        run(ctx, [&](MTL4::ComputeCommandEncoder* e) {
            encDescriptors(ctx, W, P, e, kA, m);
            barrierDA(e);
            encBuild(W, e, W.tl[kA][kNone]);
        });
        const CheckStats c = traceCheck(ctx, W, M, P, W.tl[kA][kNone].as, kA, rays);
        for (int k = 0; k < 2; ++k) { fr.n[mode][k] = c.frontN[k]; fr.eq[mode][k] = c.frontEq[k]; }
        for (int k = 0; k < 2; ++k) {
            const char* cls = k ? "mirrored" : "nonmirrored";
            rep.value(std::string("front.") + modeName[mode] + "." + cls + ".mismatch_raw", "rays", double(c.frontN[k] - c.frontEq[k]), {{"rays", double(c.frontN[k])}}, false);
            rep.value(std::string("front.") + modeName[mode] + "." + cls + ".mismatch_flipped", "rays", double(c.frontEq[k]), {{"rays", double(c.frontN[k])}}, false);
        }
        if (c.wrong) F.fail(std::string("front-face rays (mode ") + modeName[mode] + "): " + c.firstError);
    }
    // Each class must be pure (all rays agree with raw, or all with the flipped value) in every mode.
    for (u32 mode = 0; mode < 4; ++mode)
        for (int k = 0; k < 2; ++k) {
            if (fr.n[mode][k] < 50) { F.fail("too few front-face samples"); pure = false; }
            else if (fr.eq[mode][k] != 0 && fr.eq[mode][k] != fr.n[mode][k]) pure = false;
        }
    // restore the default options
    run(ctx, [&](MTL4::ComputeCommandEncoder* e) { encDescriptors(ctx, W, P, e, kA, dm); });
}

// ---- the benchmark ----------------------------------------------------------------------------------------
void tlasGpuScene(soc::Context& ctx, soc::Report& rep) {
    Meshes M;
    buildMeshes(ctx, M);
    const Psos P = makePsos(ctx);
    Findings F;
    NegState neg;
    std::vector<u32> sizes = ctx.quick() ? std::vector<u32>{1000, 10000} : std::vector<u32>{1000, 10000, 100000};
    FrontResult fr;
    bool frontPure = false;
    u32 nc1Wrong = 0, nc1Rays = 0;
    ctx.warmUp(3.0);
    for (u32 N : sizes) {
        World W;
        makeWorld(ctx, M, P, W, N);
        makeTlases(ctx, W);
        S2Mode dm;
        frame(ctx, W, M, P, kA, dm);
        rep.value("tlas." + tagOf(N) + ".live", "instances", double(W.liveSlots.size()), {{"capacity", double(W.capacity)}}, true);
        std::string note = "size " + tagOf(N) + ": capacity " + std::to_string(W.capacity) + ", live " + std::to_string(W.liveSlots.size());
        ctx.log("S2 %s", note.c_str());
        if (N == sizes[0]) {
            // NC1: transposed 4x3 must produce wrong hits.
            S2Mode bad = dm;
            bad.transposeBug = 1;
            run(ctx, [&](MTL4::ComputeCommandEncoder* e) {
                encDescriptors(ctx, W, P, e, kA, bad);
                barrierDA(e);
                encBuild(W, e, W.tl[kA][kNone]);
            });
            const std::vector<Ray> rays = makeRays(W, randomLive(W, 2048, 4242), 4343);
            const CheckStats c = traceCheck(ctx, W, M, P, W.tl[kA][kNone].as, kA, rays);
            nc1Wrong = c.wrong;
            nc1Rays = c.rays;
            rep.value("nc1.transposed.wrong", "rays", double(c.wrong), {{"rays", double(c.rays)}}, false);
        }
        for (int s = 0; s < 3; ++s) runStrategy(ctx, rep, W, M, P, Strat(s), F, neg);
        if (N == (ctx.quick() ? 1000u : 10000u)) frontExperiment(ctx, rep, W, M, P, F, fr, frontPure);
    }

    // ---- front-face rule ------------------------------------------------------------------------------------
    // Rule per (mode, class): "raw" = use triangle_front_facing as is; "flip" = invert it.
    auto rule = [&](u32 mode, int k) -> std::string {
        if (fr.n[mode][k] == 0) return "n/a";
        if (fr.eq[mode][k] == fr.n[mode][k]) return "raw";
        if (fr.eq[mode][k] == 0) return "flip";
        return "mixed";
    };
    const char* modeName[4] = {"none", "ccw_all", "ccw_mirrored_only", "ccw_nonmirrored_only"};
    std::string frontNote;
    for (u32 mode = 0; mode < 4; ++mode)
        frontNote += std::string(modeName[mode]) + ": non-mirrored " + rule(mode, 0) + ", mirrored " + rule(mode, 1) + " (n " +
                     std::to_string(fr.n[mode][0]) + "/" + std::to_string(fr.n[mode][1]) + "); ";
    // Proposed rule: CCW option exactly on mirrored instances (mode 2), triangle_front_facing used as is.
    // The other valid rule: default options (mode 0), invert the result for mirrored instances.
    // NC3: ignoring the mirrored flag (mode 0, raw) must mismatch on every mirrored hit.
    const bool ruleOk = rule(2, 0) == "raw" && rule(2, 1) == "raw" && rule(0, 0) == "raw" && rule(0, 1) == "flip";
    const u32 mirrN = fr.n[0][1];
    const u32 nc3 = mirrN - fr.eq[0][1]; // mode 0 raw mismatches on mirrored
    rep.value("nc3.wrong_front_rule.mismatch", "rays", double(nc3), {{"mirrored_rays", double(mirrN)}}, false);
    rep.note(std::string("Proposed engine rule: set MTL::AccelerationStructureInstanceOptionTriangleFrontFacingWindingCounterClockwise exactly on INSTANCE_FLAG_MIRRORED instances and use triangle_front_facing unchanged (verified: ") + (ruleOk ? "0 mismatches" : "NOT VERIFIED") + "); equivalent: no option and invert front_facing for mirrored. ");
    rep.note("Front-face matrix (GPU triangle_front_facing vs world-space CCW winding): " + frontNote);
    rep.note("userID: intersector<triangle_data, instancing> result .user_instance_id (needs __HAVE_RAYTRACING_USER_INSTANCE_ID__) = descriptor userID; "
             "instance_id = the descriptor index (A: slot; B: compacted rank, unstable; Bs: compacted rank in slot order).");
    if (!F.info.empty()) rep.note("Findings (not failures): " + F.info);

    const bool nc1 = nc1Wrong > 0;
    const bool nc2 = neg.nc2OnDeleted > 0;
    const bool nc3ok = frontPure && ruleOk && nc3 > 0;
    if (!frontPure) F.fail("front-face classes are not pure (see the matrix note)");
    if (!F.failures.empty()) rep.status(soc::Status::Failed, F.failures);
    rep.negative(nc1 && nc2 && nc3ok,
                 "NC1 transposed 4x3: " + std::to_string(nc1Wrong) + "/" + std::to_string(nc1Rays) + " rays wrong (must be > 0); NC2 deleted slots with mask 0xFF: " +
                     std::to_string(neg.nc2OnDeleted) + " hits on deleted instances, " + std::to_string(neg.nc2Wrong) + "/" + std::to_string(neg.nc2Rays) +
                     " rays wrong (must be > 0); NC3 front face of mirrored instances read raw without the CCW option (rule ignoring mirroring): " +
                     std::to_string(nc3) + "/" + std::to_string(mirrN) + " mismatches (must be > 0)");
}

} // namespace

SOC_BENCH("F9-S2", "tlas_gpu_scene",
          "F9: per-frame TLAS written by compute from the GPU scene (A one descriptor per slot, B/Bs compacted, indirect count), 1K/10K/100K",
          tlasGpuScene);

} // namespace f9
