// F9-S0: smoke test of the shared F9 spike helpers (f9_common): BLAS per
// mesh reading the engine's GPUVertex/index layout in place, TLAS of indirect
// instance descriptors, camera rays traced with the intersector and checked
// against the exact CPU BVH.  Negative control: the same rays checked against
// a soup whose first instance is shifted by 1% of the scene size must fail.
#include "f9_common.h"

#include <glm/gtc/type_ptr.hpp>

#include <cstring>

namespace f9 {
namespace {

void smoke(soc::Context& ctx, soc::Report& rep) {
    SceneData s;
    proceduralScene(s);
    const GpuGeometry g = uploadGeometry(ctx, s);
    std::vector<Blas> blases = allocateBlases(ctx, s, g, {});
    MTL::Buffer* scratch = scratchFor(ctx, blases);
    const double blasMs = buildBlases(ctx, blases, scratch);
    Tlas t = allocateTlas(ctx, u32(s.instances.size()), MTL::AccelerationStructureUsageNone, AsPlacement::Device);
    auto* d = static_cast<InstanceDesc*>(t.instances->contents());
    for (u32 i = 0; i < s.instances.size(); ++i)
        d[i] = toInstanceDesc(s.instances[i], blases[s.instances[i].meshIndex].as->gpuResourceID(), i,
                              MTL::AccelerationStructureInstanceOptionOpaque);
    const double tlasMs = timeEncoder(ctx, [&](MTL4::ComputeCommandEncoder* e) {
        e->buildAccelerationStructure(t.as, t.desc, range(t.scratch));
    });
    const std::vector<Ray> rays = cameraRays({6, 2.5, 5}, {6, 0, 0}, 90.0, 320, 180);
    const std::vector<GpuHit> hits = traceNearest(ctx, t.as, true, rays);
    const TriangleSoup soup = s.soup();
    const CpuBvh bvh(soup);
    const SoupIndex idx(soup);
    auto map = [&](const GpuHit& h) { return idx(h.instance, h.geometry, h.primitive); };
    const HitCheck c = checkNearest(bvh, rays, hits, map);
    u32 hitCount = 0;
    for (const GpuHit& h : hits) hitCount += h.t >= 0;

    SceneData moved;
    proceduralScene(moved);
    moved.instances[0].modelMatrix[12] += 0.15f;
    const TriangleSoup soupN = moved.soup();
    const CpuBvh bvhN(soupN);
    const HitCheck n = checkNearest(bvhN, rays, hits, map);

    rep.value("blas.build.ms", "ms", blasMs, {{"meshes", double(blases.size())}}, false);
    rep.value("tlas.build.ms", "ms", tlasMs, {{"instances", double(t.count)}}, false);
    rep.value("rays.wrong", "rays", double(c.wrong), {{"rays", double(c.rays)}, {"hits", double(hitCount)}}, false);
    rep.value("rays.max_rel_err", "ratio", c.maxRelErr, {}, false);
    rep.value("negative.wrong", "rays", double(n.wrong), {}, true);
    // Sponza (if present): one BLAS per mesh, TLAS of its instances, 1/4-res camera rays.
    HitCheck sp;
    SceneData sz;
    std::string err;
    if (loadSponza(sz, err)) {
        const GpuGeometry sg = uploadGeometry(ctx, sz);
        std::vector<Blas> sb = allocateBlases(ctx, sz, sg, {});
        MTL::Buffer* ss = scratchFor(ctx, sb);
        const double sBlasMs = buildBlases(ctx, sb, ss);
        Tlas st = allocateTlas(ctx, u32(sz.instances.size()), MTL::AccelerationStructureUsageNone, AsPlacement::Device);
        auto* sd = static_cast<InstanceDesc*>(st.instances->contents());
        for (u32 i = 0; i < sz.instances.size(); ++i)
            sd[i] = toInstanceDesc(sz.instances[i], sb[sz.instances[i].meshIndex].as->gpuResourceID(), i,
                                   MTL::AccelerationStructureInstanceOptionOpaque);
        timeEncoder(ctx, [&](MTL4::ComputeCommandEncoder* e) { e->buildAccelerationStructure(st.as, st.desc, range(st.scratch)); });
        const std::vector<Ray> srays = cameraRays({-8, 2, 0.5}, {8, 4, -0.5}, 70.0, 480, 270);
        const std::vector<GpuHit> shits = traceNearest(ctx, st.as, true, srays);
        const TriangleSoup ssoup = sz.soup();
        const CpuBvh sbvh(ssoup);
        const SoupIndex sidx(ssoup);
        sp = checkNearest(sbvh, srays, shits, [&](const GpuHit& h) { return sidx(h.instance, h.geometry, h.primitive); });
        rep.value("sponza.blas.build.ms", "ms", sBlasMs, {{"meshes", double(sb.size())}, {"triangles", double(sz.totalTriangles())}}, false);
        rep.value("sponza.rays.wrong", "rays", double(sp.wrong), {{"rays", double(sp.rays)}}, false);
        if (!sp.ok()) rep.status(soc::Status::Failed, "Sponza GPU hits differ from the CPU: " + sp.firstError);
    } else {
        rep.note("Sponza skipped: " + err);
    }
    if (!c.ok()) rep.status(soc::Status::Failed, "GPU hits differ from the CPU: " + c.firstError);
    rep.negative(c.ok() && sp.ok() && n.wrong > 0 && hitCount > c.rays / 10,
                 "Sponza " + std::to_string(sp.wrong) + "/" + std::to_string(sp.rays) + " wrong; exact: " + std::to_string(c.wrong) + "/" + std::to_string(c.rays) + " wrong (" + std::to_string(hitCount) +
                     " hits); shifted-instance control: " + std::to_string(n.wrong) + " wrong (must be > 0)");
}

} // namespace

SOC_BENCH("F9-S0", "smoke", "F9 helpers: BLAS/TLAS of the engine layout, rays vs exact CPU BVH", smoke);

} // namespace f9
