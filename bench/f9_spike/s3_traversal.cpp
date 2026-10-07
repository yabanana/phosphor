// F9-S3: traversal.  Sponza (one opaque BLAS per mesh, TLAS of its instances),
// 1920x1080 (quick: 960x540), two fixed cameras and a directional sun.
//
// Ray types: primary (camera), shadow (hit point -> sun, tmax inf), AO
// (cosine hemisphere, tmax 0.5), diffuse (cosine hemisphere, tmax inf).  A
// first kernel stores the primary hits (point, normal, ids, Waechter-Binder
// offset points); the secondary kernels read them, so their timings exclude
// the primary rays.
//
// Variants: intersector closest hit (assume_geometry_type(triangle)),
// intersector without the assumption, intersector accept_any_intersection,
// intersection_query closest and any (commit the first candidate and stop),
// intersector on instances forced non-opaque with and without
// force_opacity(opaque).
//
// Self-intersection (shadow and diffuse rays): origin = hit point with
// tmin 0 / 1e-4 / 1e-3, and the Waechter & Binder normal offset (RTG ch. 6).
// Counts of false self hits and agreement of the shadow bit with the exact
// double-precision CPU reference (light leaks: GPU unoccluded, CPU occluded;
// acne: GPU occluded, CPU not).
//
// Checks: primary closest hits and diffuse closest hits vs CpuBvh::nearest
// (sample), every fast variant vs the baseline closest-hit variant (occluded
// bit for any-hit variants; id and t for closest variants).
//
// Negative control: strategy "none" (origin = hit point, tmin 0) must show
// self hits; the control fails if it does not.
#include "f9_common.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace f9 {
namespace {

struct S3Params {
    u32 width, height, seed, strategy;
    float eye[3];
    float aoTmax;
    float fwd[3];
    float pad0;
    float rightS[3];
    float pad1;
    float upS[3];
    float pad2;
    float sun[3];
    float pad3;
};
static_assert(sizeof(S3Params) == 96);

struct S3Prim {
    float p[3];
    float t;
    float n[3];
    u32 instance;
    float pf[3];
    u32 primitive;
    float pb[3];
    u32 pad;
};
static_assert(sizeof(S3Prim) == 64);

struct S3Res {
    float t;
    u32 instance, primitive, pad;
};
static_assert(sizeof(S3Res) == 16);

enum Type : u32 { kPrimary = 0, kShadow = 1, kAo = 2, kDiffuse = 3 };
const char* const kTypeName[4] = {"primary", "shadow", "ao", "diffuse"};
const char* const kStrategyName[5] = {"none", "tmin1e-4", "tmin1e-3", "offset_wb", "offset_wb_world"};
// Origin strategy of the timed kernels and the CPU closest-hit check: the world-space Waechter-Binder offset (the
// object-space one self-hits on instances with a small scale, see selfhit.offset_wb.diffuse.count).
constexpr u32 kHeadline = 4;

struct Variant {
    const char* name;
    u32 mode; // 0 isect closest, 1 isect any, 2 query closest, 3 query any
    bool assumeTri, forceOpaque;
    u32 tlas; // 0 opaque instances, 1 non-opaque instances
    u32 types; // bit mask of ray types
};
constexpr u32 kAll = 0xF, kShadowAo = (1u << kShadow) | (1u << kAo);
const Variant kVariants[] = {
    {"intersector", 0, true, false, 0, kAll},
    {"intersector_generic", 0, false, false, 0, kAll},
    {"anyhit.intersector", 1, true, false, 0, kShadowAo},
    {"query", 2, true, false, 0, kAll},
    {"anyhit.query", 3, true, false, 0, kShadowAo},
    {"intersector_nonopaque", 0, true, false, 1, kAll},
    {"intersector_forceopaque", 0, true, true, 1, kAll},
};

struct Camera {
    const char* suffix;
    V3 eye, target;
    f64 vfov;
};

soc::Stats scaled(soc::Stats s, double k) {
    s.median *= k; s.min *= k; s.max *= k; s.p10 *= k; s.p90 *= k; s.mean *= k;
    return s;
}

struct Sums {
    u64 primWrong = 0, primChecked = 0, diffWrong = 0, diffChecked = 0;
    u64 boolMismatch = 0, idMismatch = 0; // fast variants vs baseline
    u64 selfSame[5][2] = {}, selfNear[5][2] = {}, falseNear[5] = {}; // [strategy][shadow, diffuse]
    u64 agree[5] = {}, leaks[5] = {}, acne[5] = {}, samples = 0;
    u64 validTotal = 0;
    std::string firstError;
};

void s3(soc::Context& ctx, soc::Report& rep) {
    SceneData sd;
    std::string err;
    if (!loadSponza(sd, err)) {
        rep.status(soc::Status::Unsupported, "Sponza missing: " + err);
        return;
    }
    const u32 W = ctx.quick() ? 960 : 1920, H = ctx.quick() ? 540 : 1080;
    const u32 npix = W * H;
    const u32 sampleN = ctx.quick() ? 20000 : 100000;

    // ---- scene -------------------------------------------------------------
    const GpuGeometry g = uploadGeometry(ctx, sd);
    std::vector<Blas> blases = allocateBlases(ctx, sd, g, {});
    MTL::Buffer* scratch = scratchFor(ctx, blases);
    buildBlases(ctx, blases, scratch);
    Tlas tlas[2];
    for (u32 k = 0; k < 2; ++k) {
        tlas[k] = allocateTlas(ctx, u32(sd.instances.size()), MTL::AccelerationStructureUsageNone, AsPlacement::Device);
        auto* d = static_cast<InstanceDesc*>(tlas[k].instances->contents());
        for (u32 i = 0; i < sd.instances.size(); ++i)
            d[i] = toInstanceDesc(sd.instances[i], blases[sd.instances[i].meshIndex].as->gpuResourceID(), i,
                                  k == 0 ? MTL::AccelerationStructureInstanceOptionOpaque
                                         : MTL::AccelerationStructureInstanceOptionNonOpaque);
        MTL::Buffer* sc = tlas[k].scratch;
        timeEncoder(ctx, [&](MTL4::ComputeCommandEncoder* e) {
            e->buildAccelerationStructure(tlas[k].as, tlas[k].desc, range(sc));
        });
    }
    const TriangleSoup soup = sd.soup();
    const CpuBvh bvh(soup);
    const SoupIndex sidx(soup);

    MTL::Buffer* meshBuf = ctx.buffer(std::max<size_t>(sd.scene.meshInfos().size() * sizeof(phosphor::GPUMeshInfo), 64));
    std::memcpy(meshBuf->contents(), sd.scene.meshInfos().data(), sd.scene.meshInfos().size() * sizeof(phosphor::GPUMeshInfo));
    MTL::Buffer* instBuf = ctx.buffer(std::max<size_t>(sd.instances.size() * sizeof(phosphor::GPUInstance), 64));
    std::memcpy(instBuf->contents(), sd.instances.data(), sd.instances.size() * sizeof(phosphor::GPUInstance));
    MTL::Buffer* paramBuf = ctx.buffer(sizeof(S3Params));
    MTL::Buffer* primBuf = ctx.buffer(size_t(npix) * sizeof(S3Prim));
    MTL::Buffer* outBuf = ctx.buffer(size_t(npix) * sizeof(GpuRay)); // S3Res (16 B) or rays (32 B)

    MTL::Library* lib = f9Library(ctx, "s3_traversal.metal");
    auto pso = [&](const char* fn, u32 type, u32 mode, bool tri, bool force) {
        auto* fc = MTL::FunctionConstantValues::alloc()->init();
        const bool bt = tri, bf = force;
        fc->setConstantValue(&type, MTL::DataTypeUInt, NS::UInteger(0));
        fc->setConstantValue(&mode, MTL::DataTypeUInt, NS::UInteger(1));
        fc->setConstantValue(&bt, MTL::DataTypeBool, NS::UInteger(2));
        fc->setConstantValue(&bf, MTL::DataTypeBool, NS::UInteger(3));
        ctx.keep(fc);
        return ctx.compute(lib, fn, fc);
    };
    MTL::ComputePipelineState* psoPrim = ctx.compute(lib, "s3_primary_hits");

    auto dispatch = [&](MTL::ComputePipelineState* p, MTL::AccelerationStructure* as) -> double {
        soc::ComputeTimer t(ctx);
        MTL4::ComputeCommandEncoder* e = t.begin();
        ctx.table()->setResource(as->gpuResourceID(), 0);
        ctx.table()->setAddress(paramBuf->gpuAddress(), 1);
        ctx.table()->setAddress(primBuf->gpuAddress(), 2);
        ctx.table()->setAddress(outBuf->gpuAddress(), 3);
        ctx.table()->setAddress(g.vertices->gpuAddress(), 4);
        ctx.table()->setAddress(g.indices->gpuAddress(), 5);
        ctx.table()->setAddress(meshBuf->gpuAddress(), 6);
        ctx.table()->setAddress(instBuf->gpuAddress(), 7);
        e->setComputePipelineState(p);
        e->setArgumentTable(ctx.table());
        const u32 tgy = std::max<u32>(1, std::min<u32>(8, u32(p->maxTotalThreadsPerThreadgroup()) / 8));
        e->dispatchThreads(MTL::Size::Make(W, H, 1), MTL::Size::Make(8, tgy, 1));
        t.lap();
        return t.finish()[0];
    };

    const V3 sunDir = normalize(V3{0.3, 1.0, 0.2}); // toward the sun (light travels along (-0.3,-1,-0.2))
    const Camera cams[2] = {{"", {-8, 2, 0.5}, {8, 4, -0.5}, 70.0}, {".camB", {8, 2.5, 0}, {-8, 3, 0}, 70.0}};

    Sums sum;
    const double empty = 0;
    (void)empty;

    for (const Camera& cam : cams) {
        // ---- params -----------------------------------------------------------
        S3Params P{};
        P.width = W; P.height = H; P.seed = 0x1234u; P.strategy = kHeadline; P.aoTmax = 0.5f;
        {
            const V3 fwd = normalize(cam.target - cam.eye);
            const V3 right = normalize(cross(fwd, V3{0, 1, 0}));
            const V3 up = cross(right, fwd);
            const f64 th = std::tan(cam.vfov * 0.5 * 3.14159265358979323846 / 180.0);
            const f64 aspect = f64(W) / f64(H);
            auto put = [](float* d, V3 v) { d[0] = float(v.x); d[1] = float(v.y); d[2] = float(v.z); };
            put(P.eye, cam.eye); put(P.fwd, fwd); put(P.rightS, right * (th * aspect)); put(P.upS, up * th); put(P.sun, sunDir);
        }
        auto setStrategy = [&](u32 s) {
            P.strategy = s;
            std::memcpy(paramBuf->contents(), &P, sizeof P);
        };
        setStrategy(kHeadline);

        // ---- primary hits kernel -------------------------------------------------
        dispatch(psoPrim, tlas[0].as);
        const std::vector<S3Prim> prims(static_cast<S3Prim*>(primBuf->contents()), static_cast<S3Prim*>(primBuf->contents()) + npix);
        std::vector<u32> validIdx;
        for (u32 i = 0; i < npix; ++i)
            if (prims[i].t >= 0) validIdx.push_back(i);
        const u32 nValid = u32(validIdx.size());
        sum.validTotal += nValid;
        ctx.log("S3%s: %u of %u pixels hit (%.1f%%)", cam.suffix, nValid, npix, 100.0 * nValid / npix);
        {
            ctx.keepWarm();
            const soc::Stats st = ctx.measure([&] { return dispatch(psoPrim, tlas[0].as); });
            rep.metric(std::string("primary_hits_kernel.ns") + cam.suffix, "ns", scaled(st, 1e6 / npix), {{"pixels", double(npix)}}, false);
        }
        if (nValid < npix / 20) { rep.status(soc::Status::Failed, "too few primary hits: bad camera"); return; }

        auto readRes = [&] { return std::vector<S3Res>(static_cast<S3Res*>(outBuf->contents()), static_cast<S3Res*>(outBuf->contents()) + npix); };
        auto genRays = [&](u32 type) {
            dispatch(pso("s3_gen_rays", type, 0, true, false), tlas[0].as);
            return std::vector<GpuRay>(static_cast<GpuRay*>(outBuf->contents()), static_cast<GpuRay*>(outBuf->contents()) + npix);
        };

        // Sample of valid pixels.
        const u32 step = std::max<u32>(1, nValid / sampleN);
        std::vector<u32> sample;
        for (u32 k = 0; k < nValid; k += step) sample.push_back(validIdx[k]);

        // ---- timing matrix and cross-variant checks -----------------------------------------
        std::vector<S3Res> base[4];
        for (const Variant& v : kVariants) {
            for (u32 type = 0; type < 4; ++type) {
                if (!(v.types & (1u << type))) continue;
                setStrategy(kHeadline);
                MTL::ComputePipelineState* p = pso("s3_trace", type, v.mode, v.assumeTri, v.forceOpaque);
                ctx.keepWarm();
                const soc::Stats st = ctx.measure([&] { return dispatch(p, tlas[v.tlas].as); });
                const double rays = type == kPrimary ? double(npix) : double(nValid);
                const std::string key = std::string("rays.") + kTypeName[type] + "." + v.name;
                rep.metric(key + ".ns" + cam.suffix, "ns", scaled(st, 1e6 / rays), {{"rays", rays}}, false);
                rep.value(key + ".grays" + cam.suffix, "Grays/s", rays / (st.median * 1e6), {{"rays", rays}}, true);
                std::vector<S3Res> res = readRes();
                if (&v == &kVariants[0]) {
                    base[type] = res;
                    u64 hit = 0, miss = 0;
                    for (const S3Res& r : res) { hit += r.t >= 0; miss += r.t == -1.0f; }
                    rep.value(std::string("rays.") + kTypeName[type] + ".hit" + cam.suffix, "rays", double(hit), {{"rays", rays}}, true);
                    rep.value(std::string("rays.") + kTypeName[type] + ".miss" + cam.suffix, "rays", double(miss), {{"rays", rays}}, true);
                } else {
                    u64 bm = 0, im = 0;
                    const bool closest = v.mode == 0 || v.mode == 2;
                    for (u32 i = 0; i < npix; ++i) {
                        const S3Res& a = base[type][i];
                        const S3Res& b = res[i];
                        if ((a.t >= 0) != (b.t >= 0)) { ++bm; continue; }
                        if (closest && a.t >= 0 &&
                            (a.instance != b.instance || a.primitive != b.primitive || std::fabs(a.t - b.t) > 1e-5f * std::max(1.0f, a.t)))
                            ++im;
                    }
                    sum.boolMismatch += bm;
                    sum.idMismatch += im;
                    rep.value("agree." + std::string(kTypeName[type]) + "." + v.name + ".occ_mismatch" + cam.suffix, "rays", double(bm), {}, false);
                    if (closest) rep.value("agree." + std::string(kTypeName[type]) + "." + v.name + ".id_mismatch" + cam.suffix, "rays", double(im), {}, false);
                    if (bm && sum.firstError.empty()) sum.firstError = std::string(kTypeName[type]) + "/" + v.name + ": hit/miss differs from the baseline closest hit";
                }
            }
        }

        // ---- primary vs CPU ---------------------------------------------------------------------
        {
            const std::vector<GpuRay> pr = genRays(kPrimary);
            std::vector<Ray> rays;
            std::vector<GpuHit> hits, hitsQ;
            std::vector<u32> picks;
            const u32 pstep = std::max<u32>(1, npix / sampleN);
            for (u32 i = 0; i < npix; i += pstep) picks.push_back(i);
            for (u32 i : picks) {
                rays.push_back(fromGpu(pr[i]));
                GpuHit h{};
                h.t = base[kPrimary][i].t; h.instance = base[kPrimary][i].instance; h.primitive = base[kPrimary][i].primitive;
                hits.push_back(h);
            }
            auto map = [&](const GpuHit& h) { return sidx(h.instance, h.geometry, h.primitive); };
            const HitCheck c = checkNearest(bvh, rays, hits, map);
            sum.primWrong += c.wrong; sum.primChecked += c.rays;
            if (!c.ok() && sum.firstError.empty()) sum.firstError = "primary: " + c.firstError;
            // The primary-hits kernel must name the same triangles as the timed primary kernel.
            u64 pmis = 0;
            for (u32 i : picks) pmis += (prims[i].t >= 0) != (base[kPrimary][i].t >= 0) || (prims[i].t >= 0 && (prims[i].instance != base[kPrimary][i].instance || prims[i].primitive != base[kPrimary][i].primitive));
            rep.value(std::string("check.primary_hits_vs_trace.mismatch") + cam.suffix, "rays", double(pmis), {}, false);
            sum.idMismatch += pmis;
        }

        // ---- diffuse (offset origin) closest hits vs CPU ---------------------------------------------
        {
            setStrategy(kHeadline);
            const std::vector<GpuRay> dr = genRays(kDiffuse);
            std::vector<Ray> rays;
            std::vector<GpuHit> hits;
            for (u32 i : sample) {
                rays.push_back(fromGpu(dr[i]));
                GpuHit h{};
                h.t = base[kDiffuse][i].t; h.instance = base[kDiffuse][i].instance; h.primitive = base[kDiffuse][i].primitive;
                hits.push_back(h);
            }
            const HitCheck c = checkNearest(bvh, rays, hits, [&](const GpuHit& h) { return sidx(h.instance, h.geometry, h.primitive); });
            sum.diffWrong += c.wrong; sum.diffChecked += c.rays;
            rep.value(std::string("check.diffuse.wrong") + cam.suffix, "rays", double(c.wrong),
                      {{"rays", double(c.rays)}, {"miss_mismatch", double(c.missMismatch)}, {"t_mismatch", double(c.tMismatch)}, {"id_mismatch", double(c.idMismatch)}, {"max_rel_err", c.maxRelErr}}, false);
            if (!c.ok()) ctx.log("S3%s: diffuse closest hit differs from the CPU in %u of %u rays: %s", cam.suffix, c.wrong, c.rays, c.firstError.c_str());
        }

        // ---- self-intersection: origin strategies ---------------------------------------------------------
        const std::vector<GpuRay> primRays = genRays(kPrimary);
        std::vector<char> cpuOcc(sample.size()); // not vector<bool>: written by several threads
        std::vector<Ray> cpuRay(sample.size());
        parallelFor(u32(sample.size()), [&](u32 k) {
            const u32 i = sample[k];
            const S3Prim& h = prims[i];
            const u32 tri = sidx(h.instance, 0, h.primitive);
            const Ray pr = fromGpu(primRays[i]);
            const V3 a = soup.v[3 * tri], b = soup.v[3 * tri + 1], c = soup.v[3 * tri + 2];
            f64 t = rayTriangle(pr, a, b, c);
            if (t < 0) t = rayTriangleLoose(pr, a, b, c, 1e-5);
            const V3 p = t >= 0 ? pr.o + pr.d * t : V3{h.p[0], h.p[1], h.p[2]};
            const Ray sr{p, {double(P.sun[0]), double(P.sun[1]), double(P.sun[2])}, 0.0, 1e30};
            cpuRay[k] = sr;
            cpuOcc[k] = bvh.any(sr, [&](u32 tt, f64 th, f64, f64) { return tt != tri && th > 1e-7; });
        });
        u32 dbg = 0, wbDbg = 0, wbNearDbg = 0;
        for (u32 s = 0; s < 5; ++s) {
            for (u32 type : {u32(kShadow), u32(kDiffuse)}) {
                setStrategy(s);
                dispatch(pso("s3_trace", type, 0, true, false), tlas[0].as);
                const std::vector<S3Res> res = readRes();
                u64 same = 0, near = 0;
                for (u32 i : validIdx) {
                    const S3Res& r = res[i];
                    if (r.t < 0) continue;
                    if (r.instance == prims[i].instance && r.primitive == prims[i].primitive) {
                        ++same;
                        if (s == 3 && ++wbDbg <= 4) {
                            const auto& m = sd.instances[r.instance].modelMatrix;
                            ctx.log("S3 debug wb %s self hit: px %u inst %u (mesh %u) prim %u t %g, hit point %.4f %.4f %.4f, |col0| %.4f", kTypeName[type], i,
                                    r.instance, sd.instances[r.instance].meshIndex, r.primitive, r.t, prims[i].p[0], prims[i].p[1], prims[i].p[2],
                                    std::sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]));
                        }
                    } else if (r.t < 1e-3f) {
                        ++near;
                        if (s == 3 && type == kShadow && ++wbNearDbg <= 4) {
                            ctx.log("S3 debug wb near hit: px %u start inst %u prim %u, hit inst %u prim %u t %g", i, prims[i].instance, prims[i].primitive, r.instance,
                                    r.primitive, r.t);
                        }
                    }
                }
                sum.selfSame[s][type == kShadow ? 0 : 1] += same;
                sum.selfNear[s][type == kShadow ? 0 : 1] += near;
                if (type == kShadow) {
                    for (u32 k = 0; k < sample.size(); ++k) {
                        const S3Res& r = res[sample[k]];
                        const bool gpu = r.t >= 0;
                        sum.agree[s] += gpu == bool(cpuOcc[k]);
                        sum.leaks[s] += !gpu && cpuOcc[k];
                        sum.acne[s] += gpu && !cpuOcc[k];
                        sum.falseNear[s] += gpu && r.t < 1e-3f && !cpuOcc[k];
                        if (s == 3 && gpu != bool(cpuOcc[k]) && dbg < 6) {
                            ++dbg;
                            const S3Prim& h = prims[sample[k]];
                            const CpuHit nh = bvh.nearest(cpuRay[k]);
                            ctx.log("   cpu ray o %.6f %.6f %.6f nearest t %g tri %u (inst %u prim %u) start tri %u", cpuRay[k].o.x, cpuRay[k].o.y, cpuRay[k].o.z, nh.t, nh.tri,
                                    nh.hit() ? soup.instance[nh.tri] : 0u, nh.hit() ? soup.primitive[nh.tri] : 0u, sidx(h.instance, 0, h.primitive));
                            ctx.log("S3 debug %s (wb): px %u start inst %u prim %u, GPU t %g inst %u prim %u, n.L %.4f", gpu ? "acne" : "leak", sample[k],
                                    h.instance, h.primitive, r.t, r.instance, r.primitive,
                                    h.n[0] * P.sun[0] + h.n[1] * P.sun[1] + h.n[2] * P.sun[2]);
                        }
                    }
                    if (s == 0) sum.samples += sample.size();
                }
            }
        }
        {
            u64 occ = 0;
            for (bool b : cpuOcc) occ += b;
            rep.value(std::string("shadow.cpu_occluded_fraction") + cam.suffix, "ratio", double(occ) / double(sample.size()), {{"samples", double(sample.size())}}, true);
        }
    }

    // ---- totals over both cameras -----------------------------------------------------------------------
    rep.value("check.primary.wrong", "rays", double(sum.primWrong), {{"rays", double(sum.primChecked)}}, false);
    rep.value("check.diffuse.wrong_total", "rays", double(sum.diffWrong), {{"rays", double(sum.diffChecked)}}, false);
    rep.value("agree.occ_mismatch_total", "rays", double(sum.boolMismatch), {}, false);
    rep.value("agree.id_mismatch_total", "rays", double(sum.idMismatch), {}, false);
    for (u32 s = 0; s < 5; ++s) {
        const std::string n = kStrategyName[s];
        rep.value("selfhit." + n + ".count", "rays", double(sum.selfSame[s][0]), {{"valid_rays", double(sum.validTotal)}}, false);
        rep.value("selfhit." + n + ".near_count", "rays", double(sum.selfNear[s][0]), {}, false);
        rep.value("selfhit." + n + ".false_near", "rays", double(sum.falseNear[s]), {{"samples", double(sum.samples)}}, false);
        rep.value("selfhit." + n + ".diffuse.count", "rays", double(sum.selfSame[s][1]), {{"valid_rays", double(sum.validTotal)}}, false);
        rep.value("selfhit." + n + ".diffuse.near_count", "rays", double(sum.selfNear[s][1]), {}, false);
        rep.value("shadow." + n + ".agree_pct", "%", 100.0 * double(sum.agree[s]) / double(sum.samples), {{"samples", double(sum.samples)}}, true);
        rep.value("shadow." + n + ".leaks", "rays", double(sum.leaks[s]), {{"samples", double(sum.samples)}}, false);
        rep.value("shadow." + n + ".acne", "rays", double(sum.acne[s]), {{"samples", double(sum.samples)}}, false);
        ctx.log("S3 %-10s self-hits shadow %llu (near %llu) diffuse %llu (near %llu) | shadow agree %.3f%% leaks %llu acne %llu (of %llu)", n.c_str(),
                (unsigned long long)sum.selfSame[s][0], (unsigned long long)sum.selfNear[s][0], (unsigned long long)sum.selfSame[s][1],
                (unsigned long long)sum.selfNear[s][1], 100.0 * double(sum.agree[s]) / double(sum.samples), (unsigned long long)sum.leaks[s],
                (unsigned long long)sum.acne[s], (unsigned long long)sum.samples);
    }
    if (sum.primWrong) rep.status(soc::Status::Failed, "primary hits differ from the CPU: " + sum.firstError);
    else if (sum.boolMismatch) rep.status(soc::Status::Failed, "fast variant disagrees with the closest hit: " + sum.firstError);
    else if (sum.idMismatch) rep.status(soc::Status::Failed, "closest-hit variants name different hits (ids or t)");
    rep.negative(sum.selfSame[0][0] > 0 && sum.selfSame[0][0] > sum.selfSame[3][0],
                 "origin = hit point with tmin 0 must self-hit: " + std::to_string(sum.selfSame[0][0]) + " shadow self hits (none) vs " +
                     std::to_string(sum.selfSame[3][0]) + " (offset_wb); primary " + std::to_string(sum.primWrong) + "/" + std::to_string(sum.primChecked) +
                     " wrong");
}

} // namespace

SOC_BENCH("F9-S3", "traversal", "F9 traversal: intersector vs query, any/closest hit, coherence, self-intersection", s3);

} // namespace f9
