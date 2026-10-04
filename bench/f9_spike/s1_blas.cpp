// F9-S1: BLAS lifecycle on Metal 4 (build, placement, compaction, refit,
// ordering).  Registered as four benchmarks so that each stays small:
//   F9-S1   placement/residency (device vs placement heaps, several BLAS packed
//           in one heap), building the Sponza BLAS with one shared scratch vs
//           disjoint scratch ranges, scratch sizes and usage flags;
//   F9-S1b  asynchronous compaction (size query in one command buffer, compact
//           copy in a later one), sizes and bitwise identical hits;
//   F9-S1c  refit after a compute deformation of the shared vertex buffer
//           (in place and out of place), refit vs rebuild, refit quality;
//   F9-S1d  ordering build -> trace and vertex write -> build: which barrier is
//           needed (method of the F5-S5 spike: a slow producer, two states).
// Every GPU result is compared with the exact CPU reference of f9_common.
#include "f9_common.h"

#include "scene/procedural.h"

#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>

namespace f9 {
namespace {

using soc::Context;
using soc::Report;
using soc::Stats;
using soc::Status;

constexpr MTL::Stages kAS = MTL::StageAccelerationStructure;
constexpr MTL::Stages kDisp = MTL::StageDispatch;

void barrier(MTL4::ComputeCommandEncoder* e, MTL::Stages after, MTL::Stages before) {
    e->barrierAfterEncoderStages(after, before, MTL4::VisibilityOptionDevice);
}
u64 alignUp(u64 v, u64 a) { return (v + a - 1) / a * a; }

template <typename T> MTL::Buffer* constBuf(Context& ctx, const T& v) {
    MTL::Buffer* b = ctx.buffer(std::max<size_t>(sizeof(T), 16));
    std::memcpy(b->contents(), &v, sizeof(T));
    return b;
}

void dispatch1d(Context& ctx, MTL4::ComputeCommandEncoder* e, MTL::ComputePipelineState* pso,
                std::initializer_list<u64> addrs, u32 n) {
    u32 i = 0;
    for (u64 a : addrs) ctx.table()->setAddress(a, i++);
    e->setComputePipelineState(pso);
    e->setArgumentTable(ctx.table());
    e->dispatchThreads(MTL::Size::Make(n, 1, 1), MTL::Size::Make(64, 1, 1));
}

/// Verdict of a benchmark: the first failure wins the status text.
struct Verdict {
    bool ok = true;
    std::string why;
    void fail(const std::string& w) {
        if (ok) why = w;
        else why += "; " + w;
        ok = false;
    }
};

struct UsageCase {
    const char* name;
    MTL::AccelerationStructureUsage usage;
};
const UsageCase kUsages[] = {{"none", MTL::AccelerationStructureUsageNone},
                             {"refit", MTL::AccelerationStructureUsageRefit},
                             {"fast_build", MTL::AccelerationStructureUsagePreferFastBuild},
                             {"min_memory", MTL::AccelerationStructureUsageMinimizeMemory},
                             {"fast_intersect", MTL::AccelerationStructureUsagePreferFastIntersection}};

// ---------------------------------------------------------------------------
// Corpus: a scene, its CPU reference and the camera rays
// ---------------------------------------------------------------------------

struct Corpus {
    std::string tag;
    SceneData s;
    GpuGeometry g;
    std::vector<Ray> rays;
    TriangleSoup soup;
    std::unique_ptr<CpuBvh> bvh;
    std::unique_ptr<SoupIndex> idx;
};

std::unique_ptr<Corpus> makeCorpus(Context& ctx, bool sponza, std::string& err) {
    auto c = std::make_unique<Corpus>();
    if (sponza) {
        c->tag = "sponza";
        if (!loadSponza(c->s, err)) return nullptr;
        const bool q = ctx.quick();
        c->rays = cameraRays({-8, 2, 0.5}, {8, 4, -0.5}, 70.0, q ? 320 : 480, q ? 180 : 270);
    } else {
        c->tag = "proc";
        proceduralScene(c->s);
        c->rays = cameraRays({6, 2.5, 5}, {6, 0, 0}, 90.0, 320, 180);
    }
    c->g = uploadGeometry(ctx, c->s);
    c->soup = c->s.soup();
    c->bvh = std::make_unique<CpuBvh>(c->soup);
    c->idx = std::make_unique<SoupIndex>(c->soup);
    return c;
}

/// TLAS over the instances of `c` referencing `as[mesh]`, built, then the
/// camera rays traced; returns the check against the CPU.
HitCheck traceAndCheck(Context& ctx, const Corpus& c, const std::vector<MTL::AccelerationStructure*>& as,
                       std::vector<GpuHit>* hitsOut = nullptr) {
    Tlas t = allocateTlas(ctx, u32(c.s.instances.size()), MTL::AccelerationStructureUsageNone, AsPlacement::Device);
    auto* d = static_cast<InstanceDesc*>(t.instances->contents());
    for (u32 i = 0; i < c.s.instances.size(); ++i)
        d[i] = toInstanceDesc(c.s.instances[i], as[c.s.instances[i].meshIndex]->gpuResourceID(), i,
                              MTL::AccelerationStructureInstanceOptionOpaque);
    timeEncoder(ctx, [&](MTL4::ComputeCommandEncoder* e) { e->buildAccelerationStructure(t.as, t.desc, range(t.scratch)); });
    std::vector<GpuHit> hits = traceNearest(ctx, t.as, true, c.rays);
    const SoupIndex& idx = *c.idx;
    HitCheck chk = checkNearest(*c.bvh, c.rays, hits, [&](const GpuHit& h) { return idx(h.instance, h.geometry, h.primitive); });
    if (hitsOut) *hitsOut = std::move(hits);
    return chk;
}

std::vector<MTL::AccelerationStructure*> handles(const std::vector<Blas>& b) {
    std::vector<MTL::AccelerationStructure*> o;
    for (const Blas& x : b) o.push_back(x.as);
    return o;
}

// ---------------------------------------------------------------------------
// Allocation: standalone, one heap each, or several AS packed in one heap
// ---------------------------------------------------------------------------

enum class Mode { Device, HeapEach, HeapPacked };
const char* modeName(Mode m) { return m == Mode::Device ? "device" : m == Mode::HeapEach ? "heap_each" : "heap_packed"; }

struct PackInfo {
    u64 sumSize = 0;        // sum of accelerationStructureSize
    u64 sumAligned = 0;     // sum of heap sizes (size rounded up to its alignment)
    u64 heapBytes = 0;      // bytes of the heap(s) created
    u64 minAlign = ~0ull, maxAlign = 0;
    u64 allocDelta = 0;     // MTLDevice currentAllocatedSize before/after the allocation
    std::vector<MTL::SizeAndAlign> sa;
};

/// Private untracked placement heap.  RETAINED ONCE MORE ON PURPOSE (leaked until the process ends): releasing
/// a heap that held acceleration structures made a LATER benchmark of the same process crash (SIGSEGV, exit 139)
/// under MTL_DEBUG_LAYER + MTL_SHADER_VALIDATION, on macOS 27.2 / M5 Max.
MTL::Heap* leakedHeap(Context& ctx, u64 bytes) {
    MTL::HeapDescriptor* hd = MTL::HeapDescriptor::alloc()->init();
    hd->setType(MTL::HeapTypePlacement);
    hd->setStorageMode(MTL::StorageModePrivate);
    hd->setHazardTrackingMode(MTL::HazardTrackingModeUntracked);
    hd->setSize(bytes);
    MTL::Heap* heap = ctx.heap(hd);
    hd->release();
    if (!heap) throw soc::BenchError("placement heap failed (" + std::to_string(bytes) + " B)");
    heap->retain();
    return heap;
}

std::vector<MTL::AccelerationStructure*> allocateAs(Context& ctx, const std::vector<u64>& sizes, Mode mode, PackInfo& info) {
    info = PackInfo{};
    std::vector<MTL::AccelerationStructure*> out;
    const u64 before = ctx.device()->currentAllocatedSize();
    for (u64 sz : sizes) {
        const MTL::SizeAndAlign sa = ctx.device()->heapAccelerationStructureSizeAndAlign(sz);
        info.sa.push_back(sa);
        info.sumSize += sz;
        info.sumAligned += alignUp(sa.size, sa.align);
        info.minAlign = std::min<u64>(info.minAlign, sa.align);
        info.maxAlign = std::max<u64>(info.maxAlign, sa.align);
    }
    if (mode == Mode::Device) {
        for (u64 sz : sizes) {
            MTL::AccelerationStructure* a = ctx.device()->newAccelerationStructure(sz);
            if (!a) throw soc::BenchError("newAccelerationStructure failed");
            ctx.adopt(a);
            out.push_back(a);
        }
        info.heapBytes = 0;
    } else if (mode == Mode::HeapEach) {
        for (size_t i = 0; i < sizes.size(); ++i) {
            const u64 hb = alignUp(info.sa[i].size, info.sa[i].align);
            MTL::AccelerationStructure* a = leakedHeap(ctx, hb)->newAccelerationStructure(sizes[i], 0);
            if (!a) throw soc::BenchError("heap->newAccelerationStructure(" + std::to_string(sizes[i]) + ") failed");
            ctx.keep(a);
            out.push_back(a);
            info.heapBytes += hb;
        }
    } else {
        std::vector<u64> offs;
        u64 off = 0;
        for (const MTL::SizeAndAlign& sa : info.sa) {
            off = alignUp(off, sa.align);
            offs.push_back(off);
            off += sa.size;
        }
        info.heapBytes = alignUp(off, std::max<u64>(info.maxAlign, 1));
        MTL::Heap* heap = leakedHeap(ctx, info.heapBytes);
        for (size_t i = 0; i < sizes.size(); ++i) {
            MTL::AccelerationStructure* a = heap->newAccelerationStructure(sizes[i], offs[i]);
            if (!a) throw soc::BenchError("heap->newAccelerationStructure(" + std::to_string(sizes[i]) + ", offset " + std::to_string(offs[i]) + ") failed");
            ctx.keep(a); // the heap is resident, so its sub-allocations need no residency entry of their own
            out.push_back(a);
        }
    }
    ctx.commitResidency();
    info.allocDelta = ctx.device()->currentAllocatedSize() - before;
    return out;
}

std::vector<Blas> makeBlases(Context& ctx, const Corpus& c, MTL::AccelerationStructureUsage usage, Mode mode, PackInfo& info) {
    BlasOptions o;
    o.usage = usage;
    std::vector<Blas> out(c.s.scene.getMeshCount());
    std::vector<u64> sizes;
    for (u32 m = 0; m < out.size(); ++m) {
        out[m].desc = blasDescriptor(ctx, c.s, c.g, m, o);
        out[m].sizes = ctx.device()->accelerationStructureSizes(out[m].desc);
        out[m].triangles = c.s.meshTriangles(m);
        sizes.push_back(out[m].sizes.accelerationStructureSize);
    }
    const std::vector<MTL::AccelerationStructure*> as = allocateAs(ctx, sizes, mode, info);
    for (u32 m = 0; m < out.size(); ++m) out[m].as = as[m];
    return out;
}

u64 scratchAlign() {
    const char* e = std::getenv("F9_S1_SCRATCH_ALIGN"); // experiment: alignment of disjoint scratch ranges
    return e ? std::strtoull(e, nullptr, 10) : 256;
}

/// Disjoint scratch ranges: offsets of every build scratch inside one buffer.
std::vector<u64> disjointOffsets(const std::vector<Blas>& b, u64 align, u64& total) {
    std::vector<u64> offs;
    u64 off = 0;
    // Experiment knob: F9_S1_SCRATCH_SKEW adds N bytes after every range (probes unaligned offsets).
    const char* sk = std::getenv("F9_S1_SCRATCH_SKEW");
    const u64 skew = sk ? std::strtoull(sk, nullptr, 10) : 0;
    for (const Blas& x : b) {
        off = alignUp(off, align);
        offs.push_back(off);
        off += std::max<u64>(x.sizes.buildScratchBufferSize, 16) + skew;
    }
    total = alignUp(off, align);
    return offs;
}

double buildDisjoint(Context& ctx, std::vector<Blas>& b, MTL::Buffer* scratch, const std::vector<u64>& offs) {
    return timeEncoder(ctx, [&](MTL4::ComputeCommandEncoder* e) {
        for (size_t i = 0; i < b.size(); ++i)
            e->buildAccelerationStructure(b[i].as, b[i].desc, range(scratch, offs[i], std::max<u64>(b[i].sizes.buildScratchBufferSize, 16)));
    });
}

u64 maxBuildScratch(const std::vector<Blas>& b) {
    u64 n = 0;
    for (const Blas& x : b) n = std::max<u64>(n, x.sizes.buildScratchBufferSize);
    return n;
}
u64 maxRefitScratch(const std::vector<Blas>& b) {
    u64 n = 0;
    for (const Blas& x : b) n = std::max<u64>(n, x.sizes.refitScratchBufferSize);
    return n;
}
u64 sumAsBytes(const std::vector<Blas>& b) {
    u64 n = 0;
    for (const Blas& x : b) n += x.sizes.accelerationStructureSize;
    return n;
}

std::string wrongDetail(const HitCheck& h) {
    return std::to_string(h.wrong) + "/" + std::to_string(h.rays) + " wrong" + (h.firstError.empty() ? "" : " (" + h.firstError + ")");
}

/// Negative control of the checker: the same hits against a soup in which every
/// 5th triangle is moved by 2% of the scene diagonal must disagree.
u32 shiftedSoupWrong(const Corpus& c, const std::vector<GpuHit>& hits) {
    TriangleSoup moved = c.soup;
    V3 lo = {1e300, 1e300, 1e300}, hi = {-1e300, -1e300, -1e300};
    for (const V3& p : moved.v) {
        lo = {std::min(lo.x, p.x), std::min(lo.y, p.y), std::min(lo.z, p.z)};
        hi = {std::max(hi.x, p.x), std::max(hi.y, p.y), std::max(hi.z, p.z)};
    }
    const f64 sh = 0.02 * length(hi - lo);
    for (u32 t = 0; t < moved.count(); t += 5)
        for (u32 k = 0; k < 3; ++k) moved.v[3 * size_t(t) + k] = moved.v[3 * size_t(t) + k] + V3{sh, sh, sh};
    const CpuBvh bvh(moved);
    const SoupIndex& idx = *c.idx;
    return checkNearest(bvh, c.rays, hits, [&](const GpuHit& h) { return idx(h.instance, h.geometry, h.primitive); }).wrong;
}

// ---------------------------------------------------------------------------
// F9-S1: placement, scratch, usage
// ---------------------------------------------------------------------------

struct S1Out {
    u32 wrongTotal = 0;
    u32 negWrong = 0;
};

void lifecycleCorpus(Context& ctx, Report& rep, Corpus& c, Verdict& v, S1Out& out) {
    const std::string tag = c.tag;
    const u32 nMesh = c.s.scene.getMeshCount();
    ctx.log("S1 %s: %u meshes, %llu triangles", tag.c_str(), nMesh, (unsigned long long)c.s.totalTriangles());
    std::vector<GpuHit> baseHits;

    // --- 1. placement / residency -------------------------------------------
    for (Mode mode : {Mode::Device, Mode::HeapEach, Mode::HeapPacked}) {
        PackInfo info;
        std::vector<Blas> b = makeBlases(ctx, c, MTL::AccelerationStructureUsageNone, mode, info);
        MTL::Buffer* scratch = scratchFor(ctx, b);
        const double firstMs = buildBlases(ctx, b, scratch);
        std::vector<GpuHit> hits;
        const HitCheck chk = traceAndCheck(ctx, c, handles(b), &hits);
        if (mode == Mode::Device) baseHits = hits;
        const std::string pre = "blas." + tag + ".place." + modeName(mode);
        rep.value(pre + ".wrong", "rays", double(chk.wrong), {{"rays", double(chk.rays)}}, false);
        rep.value(pre + ".build_first.ms", "ms", firstMs, {{"meshes", double(nMesh)}}, false);
        rep.value(pre + ".bytes.as_sum", "B", double(info.sumSize), {{"meshes", double(nMesh)}}, false);
        rep.value(pre + ".bytes.heap_sizes_aligned", "B", double(info.sumAligned), {}, false);
        rep.value(pre + ".bytes.heap_total", "B", double(info.heapBytes), {{"align_min", double(info.minAlign)}, {"align_max", double(info.maxAlign)}}, false);
        rep.value(pre + ".bytes.alloc_delta", "B", double(info.allocDelta), {{"as_sum", double(info.sumSize)}}, false);
        out.wrongTotal += chk.wrong;
        if (!chk.ok()) v.fail(tag + " " + modeName(mode) + " placement: " + wrongDetail(chk));
        if (mode == Mode::Device) {
            // Size and alignment of every mesh (heapAccelerationStructureSizeAndAlign).
            std::string note = tag + " per-mesh as size / heap size / align (first 12 of " + std::to_string(nMesh) + "): ";
            for (u32 m = 0; m < std::min<u32>(12, nMesh); ++m)
                note += std::to_string(b[m].sizes.accelerationStructureSize) + "/" + std::to_string(info.sa[m].size) + "/" + std::to_string(info.sa[m].align) + " ";
            rep.note(note);
            rep.value("blas." + tag + ".bytes.device", "B", double(info.sumSize), {{"meshes", double(nMesh)}}, false);
            rep.value("blas." + tag + ".align_min", "B", double(info.minAlign), {}, false);
            rep.value("blas." + tag + ".align_max", "B", double(info.maxAlign), {}, false);
        } else if (mode == Mode::HeapPacked) {
            rep.value("blas." + tag + ".bytes.heap_packed", "B", double(info.heapBytes), {{"as_sum", double(info.sumSize)}}, false);
        }
    }

    // --- 2. build: one shared scratch + barriers vs disjoint scratch, no barriers ---
    {
        PackInfo info;
        std::vector<Blas> b = makeBlases(ctx, c, MTL::AccelerationStructureUsageNone, Mode::Device, info);
        MTL::Buffer* scratch = scratchFor(ctx, b);
        const double firstMs = buildBlases(ctx, b, scratch);
        std::vector<GpuHit> hits;
        const HitCheck chk = traceAndCheck(ctx, c, handles(b), &hits);
        ctx.keepWarm(30);
        const Stats st = ctx.measure([&] { return buildBlases(ctx, b, scratch); });
        rep.metric("blas." + tag + ".build_shared.ms", "ms", st, {{"meshes", double(nMesh)}, {"first_ms", firstMs}, {"wrong", double(chk.wrong)}}, false);
        rep.value("blas." + tag + ".build_shared.wrong", "rays", double(chk.wrong), {{"rays", double(chk.rays)}}, false);
        rep.value("blas." + tag + ".scratch.bytes.shared", "B", double(scratch->length()), {{"max_build", double(maxBuildScratch(b))}, {"max_refit", double(maxRefitScratch(b))}}, false);
        out.wrongTotal += chk.wrong;
        if (!chk.ok()) v.fail(tag + " shared-scratch build: " + wrongDetail(chk));
    }
    {
        PackInfo info;
        std::vector<Blas> b = makeBlases(ctx, c, MTL::AccelerationStructureUsageNone, Mode::Device, info);
        u64 total = 0;
        const std::vector<u64> offs = disjointOffsets(b, scratchAlign(), total);
        MTL::Buffer* scratch = ctx.buffer(std::max<u64>(total, 4096), MTL::ResourceStorageModePrivate);
        const double firstMs = buildDisjoint(ctx, b, scratch, offs);
        std::vector<GpuHit> hits;
        const HitCheck chk = traceAndCheck(ctx, c, handles(b), &hits);
        ctx.keepWarm(30);
        const Stats st = ctx.measure([&] { return buildDisjoint(ctx, b, scratch, offs); });
        rep.metric("blas." + tag + ".build_disjoint.ms", "ms", st, {{"meshes", double(nMesh)}, {"first_ms", firstMs}, {"wrong", double(chk.wrong)}, {"align", double(scratchAlign())}}, false);
        rep.value("blas." + tag + ".build_disjoint.wrong", "rays", double(chk.wrong), {{"rays", double(chk.rays)}}, false);
        rep.value("blas." + tag + ".scratch.bytes.disjoint", "B", double(total), {{"align", double(scratchAlign())}}, false);
        out.wrongTotal += chk.wrong;
        if (!chk.ok()) v.fail(tag + " disjoint-scratch build: " + wrongDetail(chk));
    }

    // --- 3. sizes and scratch per usage flag ----------------------------------
    for (const UsageCase& u : kUsages) {
        PackInfo info;
        std::vector<Blas> b = makeBlases(ctx, c, u.usage, Mode::Device, info);
        u64 sumBuild = 0, sumRefit = 0;
        for (const Blas& x : b) { sumBuild += x.sizes.buildScratchBufferSize; sumRefit += x.sizes.refitScratchBufferSize; }
        MTL::Buffer* scratch = scratchFor(ctx, b);
        const double firstMs = buildBlases(ctx, b, scratch);
        const HitCheck chk = traceAndCheck(ctx, c, handles(b));
        const std::string pre = "usage." + tag + "." + u.name;
        rep.value(pre + ".bytes.as", "B", double(sumAsBytes(b)), {{"meshes", double(nMesh)}}, false);
        rep.value(pre + ".bytes.scratch_build_sum", "B", double(sumBuild), {}, false);
        rep.value(pre + ".bytes.scratch_build_max", "B", double(maxBuildScratch(b)), {}, false);
        rep.value(pre + ".bytes.scratch_refit_sum", "B", double(sumRefit), {}, false);
        rep.value(pre + ".bytes.scratch_refit_max", "B", double(maxRefitScratch(b)), {}, false);
        rep.value(pre + ".build_first.ms", "ms", firstMs, {}, false);
        rep.value(pre + ".wrong", "rays", double(chk.wrong), {{"rays", double(chk.rays)}}, false);
        out.wrongTotal += chk.wrong;
        if (!chk.ok()) v.fail(tag + " usage " + u.name + ": " + wrongDetail(chk));
        if (tag == "proc") {
            for (u32 m = 0; m < nMesh; ++m)
                rep.value("usage.proc.mesh" + std::to_string(m) + "." + u.name + ".bytes.as", "B", double(b[m].sizes.accelerationStructureSize),
                          {{"triangles", double(b[m].triangles)}, {"scratch_build", double(b[m].sizes.buildScratchBufferSize)}, {"scratch_refit", double(b[m].sizes.refitScratchBufferSize)}}, false);
        }
    }

    // --- control: hits of the first build against a perturbed soup must differ ----
    out.negWrong = shiftedSoupWrong(c, baseHits);
}

void s1Placement(Context& ctx, Report& rep) {
    Verdict v;
    S1Out total;
    u32 negProc = 0, negSponza = 0;
    std::string sponzaNote;
    {
        std::string err;
        auto c = makeCorpus(ctx, false, err);
        S1Out o;
        lifecycleCorpus(ctx, rep, *c, v, o);
        total.wrongTotal += o.wrongTotal;
        negProc = o.negWrong;
    }
    {
        std::string err;
        auto c = makeCorpus(ctx, true, err);
        if (!c) {
            rep.note("Sponza skipped: " + err);
            rep.status(Status::Partial, "Sponza missing: " + err);
        } else {
            S1Out o;
            lifecycleCorpus(ctx, rep, *c, v, o);
            total.wrongTotal += o.wrongTotal;
            negSponza = o.negWrong;
            sponzaNote = "; Sponza shifted-soup control " + std::to_string(negSponza) + " wrong";
        }
    }
    if (!v.ok) rep.status(Status::Failed, v.why);
    rep.negative(v.ok && negProc > 0, std::to_string(total.wrongTotal) + " wrong rays over every placement/build/usage variant; shifted-soup control (procedural) " +
                                          std::to_string(negProc) + " wrong (must be > 0)" + sponzaNote);
}

// ---------------------------------------------------------------------------
// F9-S1b: compaction
// ---------------------------------------------------------------------------

bool sameHit(const GpuHit& a, const GpuHit& b) { return std::memcmp(&a, &b, sizeof(GpuHit)) == 0; }

void compactCorpus(Context& ctx, Report& rep, Corpus& c, Verdict& v, u32& negWrong) {
    const std::string tag = c.tag;
    const u32 n = c.s.scene.getMeshCount();
    PackInfo info;
    std::vector<Blas> b = makeBlases(ctx, c, MTL::AccelerationStructureUsageNone, Mode::Device, info);
    MTL::Buffer* scratch = scratchFor(ctx, b);
    // Size slots of 8 bytes pre-filled with a sentinel: the upper word shows how many bytes the write covers.
    MTL::Buffer* sizeBuf = ctx.buffer(size_t(n) * 8);
    std::memset(sizeBuf->contents(), 0xFF, size_t(n) * 8);
    const double buildMs = timeEncoder(ctx, [&](MTL4::ComputeCommandEncoder* e) {
        for (u32 i = 0; i < n; ++i) {
            if (i) barrier(e, kAS, kAS);
            e->buildAccelerationStructure(b[i].as, b[i].desc, range(scratch));
            barrier(e, kAS, kAS);
            e->writeCompactedAccelerationStructureSize(b[i].as, MTL4::BufferRange::Make(sizeBuf->gpuAddress() + 8ull * i, 8));
        }
    });
    // (the encoder is complete here: a LATER command buffer allocates and copies)
    std::vector<u64> csz(n);
    u32 upperWritten = 0;
    u64 totalBefore = 0, totalAfter = 0;
    std::vector<double> ratios;
    for (u32 i = 0; i < n; ++i) {
        const u32* w = reinterpret_cast<const u32*>(static_cast<const char*>(sizeBuf->contents()) + 8 * size_t(i));
        csz[i] = w[0];
        upperWritten += (w[1] != 0xFFFFFFFFu);
        const u64 orig = b[i].sizes.accelerationStructureSize;
        if (csz[i] == 0 || csz[i] > orig) v.fail(tag + " mesh " + std::to_string(i) + " compacted size " + std::to_string(csz[i]) + " invalid (original " + std::to_string(orig) + ")");
        totalBefore += orig;
        totalAfter += csz[i];
        ratios.push_back(double(csz[i]) / double(orig));
    }
    std::vector<double> sorted = ratios;
    std::sort(sorted.begin(), sorted.end());
    const std::string pre = "blas." + tag;
    rep.value(pre + ".compact.size_write_bytes", "B", upperWritten ? 8.0 : 4.0, {{"upper_words_written", double(upperWritten)}}, false);
    rep.value(pre + ".bytes.device", "B", double(totalBefore), {{"meshes", double(n)}}, false);
    rep.value(pre + ".bytes.compact", "B", double(totalAfter), {{"meshes", double(n)}}, false);
    rep.value(pre + ".compact.ratio", "ratio", double(totalAfter) / double(totalBefore), {{"before", double(totalBefore)}, {"after", double(totalAfter)}}, false);
    rep.value(pre + ".compact.ratio_mesh_min", "ratio", sorted.front(), {}, false);
    rep.value(pre + ".compact.ratio_mesh_median", "ratio", sorted[sorted.size() / 2], {}, false);
    rep.value(pre + ".compact.ratio_mesh_max", "ratio", sorted.back(), {}, false);
    rep.value(pre + ".compact.build_and_size.ms", "ms", buildMs, {{"meshes", double(n)}}, false);
    if (c.tag == "proc")
        for (u32 i = 0; i < n; ++i)
            rep.value(pre + ".compact.mesh" + std::to_string(i) + ".ratio", "ratio", ratios[i],
                      {{"before", double(b[i].sizes.accelerationStructureSize)}, {"after", double(csz[i])}, {"triangles", double(b[i].triangles)}}, false);

    std::vector<GpuHit> before;
    const HitCheck chkBefore = traceAndCheck(ctx, c, handles(b), &before);
    rep.value(pre + ".compact.before.wrong", "rays", double(chkBefore.wrong), {{"rays", double(chkBefore.rays)}}, false);
    if (!chkBefore.ok()) v.fail(tag + " before compaction: " + wrongDetail(chkBefore));

    for (Mode mode : {Mode::Device, Mode::HeapPacked}) {
        PackInfo ci;
        const std::vector<MTL::AccelerationStructure*> dst = allocateAs(ctx, csz, mode, ci);
        auto copy = [&](MTL4::ComputeCommandEncoder* e) {
            for (u32 i = 0; i < n; ++i) e->copyAndCompactAccelerationStructure(b[i].as, dst[i]);
        };
        const double firstMs = timeEncoder(ctx, copy);
        std::vector<GpuHit> after;
        const HitCheck chk = traceAndCheck(ctx, c, dst, &after);
        u32 diffs = 0;
        for (size_t i = 0; i < before.size(); ++i) diffs += !sameHit(before[i], after[i]);
        ctx.keepWarm(30);
        const Stats st = ctx.measure([&] { return timeEncoder(ctx, copy); });
        const std::string p2 = pre + ".compact.to_" + modeName(mode);
        rep.metric(p2 + ".copy.ms", "ms", st, {{"meshes", double(n)}, {"first_ms", firstMs}}, false);
        rep.value(p2 + ".wrong", "rays", double(chk.wrong), {{"rays", double(chk.rays)}}, false);
        rep.value(p2 + ".hit_diffs", "rays", double(diffs), {{"rays", double(before.size())}}, false);
        rep.value(p2 + ".bytes.alloc_delta", "B", double(ci.allocDelta), {{"as_sum", double(ci.sumSize)}, {"heap_total", double(ci.heapBytes)}}, false);
        if (!chk.ok()) v.fail(tag + " compact(" + modeName(mode) + "): " + wrongDetail(chk));
        if (diffs) v.fail(tag + " compact(" + modeName(mode) + "): " + std::to_string(diffs) + " hits differ bitwise from the uncompacted AS");
    }
    negWrong = shiftedSoupWrong(c, before);
}

void s1bCompaction(Context& ctx, Report& rep) {
    Verdict v;
    u32 negProc = 0, negSponza = 0;
    {
        std::string err;
        auto c = makeCorpus(ctx, false, err);
        compactCorpus(ctx, rep, *c, v, negProc);
    }
    {
        std::string err;
        auto c = makeCorpus(ctx, true, err);
        if (!c) {
            rep.note("Sponza skipped: " + err);
            rep.status(Status::Partial, "Sponza missing: " + err);
        } else {
            compactCorpus(ctx, rep, *c, v, negSponza);
        }
    }
    if (!v.ok) rep.status(Status::Failed, v.why);
    rep.negative(v.ok && negProc > 0, std::string(v.ok ? "compacted AS: identical hits (bitwise) and 0 wrong vs the CPU; " : "FAILED: " + v.why + "; ") +
                                          "shifted-soup control " + std::to_string(negProc) + " wrong (must be > 0)");
}

// ---------------------------------------------------------------------------
// F9-S1c: refit
// ---------------------------------------------------------------------------

struct Ref {
    TriangleSoup soup;
    std::unique_ptr<CpuBvh> bvh;
    std::unique_ptr<SoupIndex> idx;
};

/// CPU reference of the mesh as the GPU vertex buffer holds it now.
std::unique_ptr<Ref> refFromGpu(const SceneData& s, const GpuGeometry& g, u32 mesh) {
    auto r = std::make_unique<Ref>();
    const phosphor::GPUMeshInfo& mi = s.scene.meshInfos()[mesh];
    const auto* vtx = static_cast<const phosphor::GPUVertex*>(g.vertices->contents()) + mi.vertexOffset;
    const u32* idx = s.scene.indices().data() + mi.indexOffset;
    auto P = [&](u32 i) { return V3{vtx[i].px, vtx[i].py, vtx[i].pz}; };
    for (u32 t = 0; t < mi.indexCount / 3; ++t) r->soup.add(P(idx[3 * t]), P(idx[3 * t + 1]), P(idx[3 * t + 2]), 0, t, 0);
    r->bvh = std::make_unique<CpuBvh>(r->soup);
    r->idx = std::make_unique<SoupIndex>(r->soup);
    return r;
}

/// Hit/miss and t must agree for every ray.  After a large deformation a few rays at grazing, stretched
/// triangles name a neighbouring triangle with the same t (float vs double edge ties, also seen with a
/// freshly rebuilt AS): up to 1 per 10,000 rays of such id-only mismatches are tolerated (and reported).
HitCheck checkRef(const Ref& r, const std::vector<Ray>& rays, const std::vector<GpuHit>& hits) {
    HitCheck c = checkNearest(*r.bvh, rays, hits, [&](const GpuHit& h) { return (*r.idx)(h.instance, h.geometry, h.primitive); });
    if (c.missMismatch + c.tMismatch == 0 && c.idMismatch <= c.rays / 10000 + 1) c.wrong = 0;
    return c;
}

/// Random rays from a shell around the box towards random points inside it.
std::vector<Ray> boxRays(V3 lo, V3 hi, u32 n, u32 seed) {
    const V3 center = (lo + hi) * 0.5;
    const f64 radius = 0.75 * length(hi - lo) + 1e-3;
    std::vector<Ray> rays;
    for (u32 i = 0; i < n; ++i) {
        const f64 z = 2.0 * rnd01(i, seed) - 1.0, phi = 6.283185307179586 * rnd01(i, seed + 1);
        const f64 r = std::sqrt(std::max(0.0, 1.0 - z * z));
        const V3 dir = {r * std::cos(phi), z, r * std::sin(phi)};
        const V3 target = {lo.x + (hi.x - lo.x) * rnd01(i, seed + 2), lo.y + (hi.y - lo.y) * rnd01(i, seed + 3), lo.z + (hi.z - lo.z) * rnd01(i, seed + 4)};
        const V3 o = center + dir * radius;
        rays.push_back({o, normalize(target - o), 0.0, 1e30});
    }
    return rays;
}

struct DeformParams {
    u32 first, count, strideFloats;
    float amplitude, k, phase;
    u32 pad0 = 0, pad1 = 0;
};
struct TraceParams {
    u32 count, mask, pad0 = 0, pad1 = 0;
};

struct RefitCase {
    std::string tag;
    const SceneData* s = nullptr;
    GpuGeometry g;
    u32 mesh = 0, vFirst = 0, vCount = 0, tris = 0;
    MTL::Buffer* base = nullptr;
    double extent = 1;
    std::vector<Ray> rays;
    MTL::Buffer *rayBuf = nullptr, *hitBuf = nullptr, *trParams = nullptr, *scratch = nullptr;
    MTL4::PrimitiveAccelerationStructureDescriptor* desc = nullptr;
    MTL::AccelerationStructureSizes sizes{};
    MTL::ComputePipelineState *deform = nullptr, *trace = nullptr;
};

void encodeDeform(Context& ctx, RefitCase& rc, MTL4::ComputeCommandEncoder* e, float amp, float k, float phase) {
    const DeformParams p{rc.vFirst, rc.vCount, u32(sizeof(phosphor::GPUVertex) / 4), amp, k, phase};
    dispatch1d(ctx, e, rc.deform, {rc.g.vertices->gpuAddress(), rc.base->gpuAddress(), constBuf(ctx, p)->gpuAddress()}, rc.vCount);
}
void encodeTrace(Context& ctx, RefitCase& rc, MTL4::ComputeCommandEncoder* e, MTL::AccelerationStructure* as, bool setState = true) {
    if (setState) { // repeated dispatches in one encoder keep the state (the validation layer rejects redundant sets)
        ctx.table()->setResource(as->gpuResourceID(), 0);
        ctx.table()->setAddress(rc.rayBuf->gpuAddress(), 1);
        ctx.table()->setAddress(rc.hitBuf->gpuAddress(), 2);
        ctx.table()->setAddress(rc.trParams->gpuAddress(), 3);
        e->setComputePipelineState(rc.trace);
        e->setArgumentTable(ctx.table());
    }
    e->dispatchThreads(MTL::Size::Make(u32(rc.rays.size()), 1, 1), MTL::Size::Make(64, 1, 1));
}
std::vector<GpuHit> readHits(const RefitCase& rc) {
    std::vector<GpuHit> h(rc.rays.size());
    std::memcpy(h.data(), rc.hitBuf->contents(), h.size() * sizeof(GpuHit));
    return h;
}
// Trace time of the case's rays; the same dispatch is repeated kTraceRepeat times (one encoder, identical
// results written to the same hit buffer) so that the interval is long enough to time.
u32 traceRepeat(const Context& ctx) { return ctx.quick() ? 16 : 64; }
double traceMs(Context& ctx, RefitCase& rc, MTL::AccelerationStructure* as) {
    return timeEncoder(ctx, [&](MTL4::ComputeCommandEncoder* e) {
        for (u32 i = 0; i < traceRepeat(ctx); ++i) encodeTrace(ctx, rc, e, as, i == 0);
    });
}

/// deform -> barrier(Dispatch,AS) -> refit src->dst -> barrier(AS,Dispatch) -> trace(dst), one encoder.
std::vector<GpuHit> deformRefitTrace(Context& ctx, RefitCase& rc, float amp, float k, float phase, MTL::AccelerationStructure* src,
                                     MTL::AccelerationStructure* dst) {
    MTL4::CommandBuffer* cmd = ctx.beginCommands();
    MTL4::ComputeCommandEncoder* e = cmd->computeCommandEncoder();
    encodeDeform(ctx, rc, e, amp, k, phase);
    barrier(e, kDisp, kAS);
    e->refitAccelerationStructure(src, rc.desc, dst, range(rc.scratch));
    barrier(e, kAS, kDisp);
    encodeTrace(ctx, rc, e, dst);
    e->endEncoding();
    ctx.submit();
    return readHits(rc);
}

void refitCase(Context& ctx, Report& rep, RefitCase& rc, Verdict& v, u32& negWrong) {
    const std::string pre = "refit." + rc.tag;
    const phosphor::GPUMeshInfo& mi = rc.s->scene.meshInfos()[rc.mesh];
    rc.vFirst = mi.vertexOffset;
    rc.vCount = (rc.mesh + 1 < rc.s->scene.getMeshCount() ? rc.s->scene.meshInfos()[rc.mesh + 1].vertexOffset : u32(rc.s->scene.vertices().size())) - mi.vertexOffset;
    rc.tris = mi.indexCount / 3;
    // Base copy (packed float3) and the bounds.
    rc.base = ctx.buffer(size_t(rc.vCount) * 12);
    auto* bp = static_cast<float*>(rc.base->contents());
    const auto* vtx = static_cast<const phosphor::GPUVertex*>(rc.g.vertices->contents()) + rc.vFirst;
    V3 lo = {1e300, 1e300, 1e300}, hi = {-1e300, -1e300, -1e300};
    for (u32 i = 0; i < rc.vCount; ++i) {
        bp[3 * i] = vtx[i].px; bp[3 * i + 1] = vtx[i].py; bp[3 * i + 2] = vtx[i].pz;
        lo = {std::min<f64>(lo.x, vtx[i].px), std::min<f64>(lo.y, vtx[i].py), std::min<f64>(lo.z, vtx[i].pz)};
        hi = {std::max<f64>(hi.x, vtx[i].px), std::max<f64>(hi.y, vtx[i].py), std::max<f64>(hi.z, vtx[i].pz)};
    }
    rc.extent = std::max({hi.x - lo.x, hi.y - lo.y, hi.z - lo.z});
    const float smallA = float(0.04 * rc.extent), smallK = float(6.2831853 * 3.0 / rc.extent);
    const float largeA = float(0.35 * rc.extent), largeK = float(6.2831853 * 9.0 / rc.extent);
    const V3 rlo = {lo.x, lo.y - largeA, lo.z}, rhi = {hi.x, hi.y + largeA, hi.z};
    const u32 nRays = ctx.quick() ? 32768 : 131072;
    rc.rays = boxRays(rlo, rhi, nRays, 1234);
    rc.rayBuf = ctx.buffer(rc.rays.size() * sizeof(GpuRay));
    rc.hitBuf = ctx.buffer(rc.rays.size() * sizeof(GpuHit));
    auto* gr = static_cast<GpuRay*>(rc.rayBuf->contents());
    for (size_t i = 0; i < rc.rays.size(); ++i) gr[i] = toGpu(rc.rays[i]);
    rc.trParams = constBuf(ctx, TraceParams{u32(rc.rays.size()), 0xFF});
    rc.deform = ctx.compute(f9Library(ctx, "s1_blas.metal"), "deform_sine");
    rc.trace = ctx.compute(f9Library(ctx, "f9_trace.metal"), "trace_nearest_blas");

    BlasOptions o;
    o.usage = MTL::AccelerationStructureUsageRefit;
    rc.desc = blasDescriptor(ctx, *rc.s, rc.g, rc.mesh, o);
    rc.sizes = ctx.device()->accelerationStructureSizes(rc.desc);
    rc.scratch = ctx.buffer(std::max<u64>(std::max(rc.sizes.buildScratchBufferSize, rc.sizes.refitScratchBufferSize), 4096), MTL::ResourceStorageModePrivate);
    MTL::AccelerationStructure* asA = newAccelerationStructure(ctx, rc.sizes.accelerationStructureSize, AsPlacement::Device);
    MTL::AccelerationStructure* asB = newAccelerationStructure(ctx, rc.sizes.accelerationStructureSize, AsPlacement::Device);
    MTL::AccelerationStructure* asR = newAccelerationStructure(ctx, rc.sizes.accelerationStructureSize, AsPlacement::Device);
    ctx.commitResidency();
    auto build = [&](MTL::AccelerationStructure* as) {
        return timeEncoder(ctx, [&](MTL4::ComputeCommandEncoder* e) { e->buildAccelerationStructure(as, rc.desc, range(rc.scratch)); });
    };
    auto refit = [&](MTL::AccelerationStructure* src, MTL::AccelerationStructure* dst) {
        return timeEncoder(ctx, [&](MTL4::ComputeCommandEncoder* e) { e->refitAccelerationStructure(src, rc.desc, dst, range(rc.scratch)); });
    };
    auto traceCheck = [&](MTL::AccelerationStructure* as, const Ref& ref, const std::string& what) {
        timeEncoder(ctx, [&](MTL4::ComputeCommandEncoder* e) { encodeTrace(ctx, rc, e, as); });
        const std::vector<GpuHit> h = readHits(rc);
        const HitCheck chk = checkRef(ref, rc.rays, h);
        rep.value(pre + "." + what + ".wrong", "rays", double(chk.wrong), {{"rays", double(chk.rays)}}, false);
        if (!chk.ok()) v.fail(rc.tag + " " + what + ": " + wrongDetail(chk));
        return h;
    };
    // The deformation as the GPU wrote it against double precision (information only).
    auto deformErr = [&](float amp, float k, float phase) {
        const auto* cur = static_cast<const phosphor::GPUVertex*>(rc.g.vertices->contents()) + rc.vFirst;
        f64 maxErr = 0;
        for (u32 i = 0; i < rc.vCount; ++i) {
            const f64 y = f64(bp[3 * i + 1]) + f64(amp) * std::sin(f64(k) * f64(bp[3 * i]) + f64(phase));
            maxErr = std::max(maxErr, std::fabs(y - f64(cur[i].py)) + std::fabs(f64(cur[i].px) - f64(bp[3 * i])) + std::fabs(f64(cur[i].pz) - f64(bp[3 * i + 2])));
        }
        return maxErr / rc.extent;
    };

    // State 0: undeformed.
    const std::unique_ptr<Ref> ref0 = refFromGpu(*rc.s, rc.g, rc.mesh);
    build(asA);
    build(asR);
    const std::vector<GpuHit> hits0 = traceCheck(asA, *ref0, "state0");
    const double trace0 = ctx.measure([&] { return traceMs(ctx, rc, asR); }).median;

    // State 1: small deformation, in-place refit, all in ONE encoder.
    {
        const std::vector<GpuHit> h = deformRefitTrace(ctx, rc, smallA, smallK, 0.7f, asA, asA);
        const std::unique_ptr<Ref> ref = refFromGpu(*rc.s, rc.g, rc.mesh);
        const HitCheck chk = checkRef(*ref, rc.rays, h);
        rep.value(pre + ".inplace.wrong", "rays", double(chk.wrong), {{"rays", double(chk.rays)}}, false);
        rep.value(pre + ".deform.max_err_rel", "ratio", deformErr(smallA, smallK, 0.7f), {}, false);
        if (!chk.ok()) v.fail(rc.tag + " in-place refit: " + wrongDetail(chk));
        u32 changed = 0;
        for (size_t i = 0; i < h.size(); ++i) changed += !sameHit(h[i], hits0[i]);
        rep.value(pre + ".inplace.hits_changed", "rays", double(changed), {}, true);
        negWrong = std::max(negWrong, checkRef(*ref0, rc.rays, h).wrong); // the undeformed reference must disagree

        // State 2: out-of-place refit asA -> asB, the source must stay valid (state 1).
        const std::vector<GpuHit> h2 = deformRefitTrace(ctx, rc, smallA, smallK, 1.9f, asA, asB);
        const std::unique_ptr<Ref> ref2 = refFromGpu(*rc.s, rc.g, rc.mesh);
        const HitCheck chk2 = checkRef(*ref2, rc.rays, h2);
        rep.value(pre + ".outofplace.wrong", "rays", double(chk2.wrong), {{"rays", double(chk2.rays)}}, false);
        if (!chk2.ok()) v.fail(rc.tag + " out-of-place refit: " + wrongDetail(chk2));
        timeEncoder(ctx, [&](MTL4::ComputeCommandEncoder* e) { encodeTrace(ctx, rc, e, asA); });
        const HitCheck chkSrc = checkRef(*ref, rc.rays, readHits(rc));
        rep.value(pre + ".outofplace.source_untouched.wrong", "rays", double(chkSrc.wrong), {{"rays", double(chkSrc.rays)}}, false);
        if (!chkSrc.ok()) v.fail(rc.tag + " out-of-place refit modified its source: " + wrongDetail(chkSrc));
    }

    // State 3: LARGE deformation: refit chain (asB) vs full rebuild (asR).
    {
        const std::vector<GpuHit> h = deformRefitTrace(ctx, rc, largeA, largeK, 0.3f, asB, asB);
        const std::unique_ptr<Ref> ref3 = refFromGpu(*rc.s, rc.g, rc.mesh);
        const HitCheck chk = checkRef(*ref3, rc.rays, h);
        rep.value(pre + ".large.refit.wrong", "rays", double(chk.wrong), {{"rays", double(chk.rays)}}, false);
        if (!chk.ok()) v.fail(rc.tag + " large-deformation refit: " + wrongDetail(chk));
        build(asR);
        traceCheck(asR, *ref3, "large.rebuild");
        ctx.keepWarm(30);
        const Stats tr = ctx.measure([&] { return traceMs(ctx, rc, asB); });
        const Stats tb = ctx.measure([&] { return traceMs(ctx, rc, asR); });
        rep.metric(pre + ".large.trace.refit.ms", "ms", tr, {{"rays", double(nRays) * traceRepeat(ctx)}}, false);
        rep.metric(pre + ".large.trace.rebuild.ms", "ms", tb, {{"rays", double(nRays) * traceRepeat(ctx)}}, false);
        rep.value(pre + ".large.trace.undeformed.ms", "ms", trace0, {{"rays", double(nRays) * traceRepeat(ctx)}}, false);
        rep.value(pre + ".large.trace.ratio_refit_over_rebuild", "ratio", tr.median / tb.median, {}, false);
        // Build / refit costs at this state.
        const Stats sIn = ctx.measure([&] { return refit(asB, asB); });
        const Stats sOut = ctx.measure([&] { return refit(asB, asA); });
        const Stats sRb = ctx.measure([&] { return build(asR); });
        rep.metric(pre + ".time.refit_inplace.ms", "ms", sIn, {{"triangles", double(rc.tris)}}, false);
        rep.metric(pre + ".time.refit_outofplace.ms", "ms", sOut, {{"triangles", double(rc.tris)}}, false);
        rep.metric(pre + ".time.rebuild.ms", "ms", sRb, {{"triangles", double(rc.tris)}}, false);
        rep.value(pre + ".time.rebuild_over_refit", "ratio", sRb.median / sIn.median, {}, true);
    }
    rep.value(pre + ".bytes.as", "B", double(rc.sizes.accelerationStructureSize), {{"triangles", double(rc.tris)}}, false);
    rep.value(pre + ".bytes.scratch_build", "B", double(rc.sizes.buildScratchBufferSize), {}, false);
    rep.value(pre + ".bytes.scratch_refit", "B", double(rc.sizes.refitScratchBufferSize), {}, false);
}

void s1cRefit(Context& ctx, Report& rep) {
    Verdict v;
    u32 neg = 0;
    // Procedural: sphere and a 64x64 plane (own scene: GpuScene is not copyable).
    SceneData sp;
    sp.name = "refit";
    {
        QuietStderr quiet; // the engine logs every mesh upload (counted by --validate)
        using namespace phosphor::ProceduralMeshes;
        for (const phosphor::MeshData& m : {generateSphere(1.0f, 64, 32), generatePlane(8.0f, 8.0f, 64, 64)})
            sp.scene.uploadMesh(m.positions, m.normals, m.tangents, m.uvs, m.indices);
    }
    const GpuGeometry gp = uploadGeometry(ctx, sp);
    const char* names[2] = {"sphere", "plane"};
    for (u32 m = 0; m < 2; ++m) {
        RefitCase rc;
        rc.tag = names[m];
        rc.s = &sp;
        rc.g = gp;
        rc.mesh = m;
        ctx.log("S1c %s", rc.tag.c_str());
        u32 n = 0;
        refitCase(ctx, rep, rc, v, n);
        neg = std::max(neg, n);
    }
    // One large Sponza mesh.
    SceneData sz;
    std::string err;
    if (loadSponza(sz, err)) {
        u32 big = 0;
        for (u32 m = 0; m < sz.scene.getMeshCount(); ++m)
            if (sz.scene.meshInfos()[m].indexCount > sz.scene.meshInfos()[big].indexCount) big = m;
        RefitCase rc;
        rc.tag = "sponza_big";
        rc.s = &sz;
        rc.g = uploadGeometry(ctx, sz);
        rc.mesh = big;
        ctx.log("S1c sponza mesh %u (%u triangles)", big, sz.meshTriangles(big));
        u32 n = 0;
        refitCase(ctx, rep, rc, v, n);
        neg = std::max(neg, n);
    } else {
        rep.note("Sponza skipped: " + err);
        rep.status(Status::Partial, "Sponza missing: " + err);
    }
    if (!v.ok) rep.status(Status::Failed, v.why);
    rep.negative(v.ok && neg > 0, std::string(v.ok ? "in-place, out-of-place and rebuilt AS all match the CPU reference of the deformed vertices; " : "FAILED: " + v.why + "; ") +
                                      "control: the hits of the refitted AS checked against the UNDEFORMED mesh: " + std::to_string(neg) + " wrong (must be > 0)");
}

// ---------------------------------------------------------------------------
// F9-S1d: ordering build -> trace and vertex write -> build
// ---------------------------------------------------------------------------

enum class Prod { Barrier, None };
enum class Cons { Enc, None, DispDisp, TwoEncQueue, TwoEncNone, SkipOp };
enum class Op { Build, Refit };

struct OrderVar {
    const char* pair;   // "build_trace", "write_build" or "control"
    const char* name;
    Prod prod;
    Cons cons;
    bool correct;       // an ordering the API requires (must never fail)
    bool slow = false;  // the vertex producer spins (a slow producer: widens the race window)
};

struct Plane {
    u32 n = 0;
    float cell = 0;
    u32 rays = 0;
    MTL::Buffer *verts = nullptr, *indices = nullptr, *ts = nullptr, *scratch = nullptr, *pA = nullptr, *pB = nullptr, *pBslow = nullptr, *pTrace = nullptr;
    MTL::AccelerationStructure* as = nullptr;
    MTL4::PrimitiveAccelerationStructureDescriptor* desc = nullptr;
    MTL::ComputePipelineState *setState = nullptr, *traceDown = nullptr;
};

struct PlaneParams {
    u32 n;
    float cell, y;
    u32 count;
};

void encodeSetState(Context& ctx, Plane& p, MTL4::ComputeCommandEncoder* e, MTL::Buffer* params) {
    dispatch1d(ctx, e, p.setState, {p.verts->gpuAddress(), params->gpuAddress()}, (p.n + 1) * (p.n + 1));
}
void encodeTraceDown(Context& ctx, Plane& p, MTL4::ComputeCommandEncoder* e) {
    ctx.table()->setResource(p.as->gpuResourceID(), 0);
    ctx.table()->setAddress(p.ts->gpuAddress(), 1);
    ctx.table()->setAddress(p.pTrace->gpuAddress(), 2);
    e->setComputePipelineState(p.traceDown);
    e->setArgumentTable(ctx.table());
    e->dispatchThreads(MTL::Size::Make(p.rays, 1, 1), MTL::Size::Make(64, 1, 1));
}
void encodeOp(Plane& p, MTL4::ComputeCommandEncoder* e, Op op, MTL::Buffer* scratchBuf) {
    if (op == Op::Build) e->buildAccelerationStructure(p.as, p.desc, range(scratchBuf));
    else e->refitAccelerationStructure(p.as, p.desc, p.as, range(scratchBuf));
}

struct OrderRes {
    u32 failures = 0, reps = 0, maxWrong = 0;
    Stats ms;
};

OrderRes runOrder(Context& ctx, Plane& p, const OrderVar& var, Op op, u32 reps) {
    OrderRes r;
    r.reps = reps;
    const Stats st = ctx.measure(
        [&] {
            // (1) state A in an earlier command buffer, completed.
            {
                MTL4::CommandBuffer* cmd = ctx.beginCommands();
                MTL4::ComputeCommandEncoder* e = cmd->computeCommandEncoder();
                encodeSetState(ctx, p, e, p.pA);
                barrier(e, kDisp, kAS);
                e->buildAccelerationStructure(p.as, p.desc, range(p.scratch));
                e->endEncoding();
                ctx.submit();
            }
            std::fill_n(static_cast<float*>(p.ts->contents()), p.rays, -7.0f);
            // (2) the test: state B, build/refit, trace.
            MTL4::CommandBuffer* cmd = ctx.beginCommands();
            MTL4::ComputeCommandEncoder* e = cmd->computeCommandEncoder();
            encodeSetState(ctx, p, e, var.slow ? p.pBslow : p.pB);
            if (var.prod == Prod::Barrier) barrier(e, kDisp, kAS);
            if (var.cons != Cons::SkipOp) encodeOp(p, e, op, p.scratch);
            switch (var.cons) {
            case Cons::Enc:
            case Cons::SkipOp:
                barrier(e, kAS, kDisp);
                encodeTraceDown(ctx, p, e);
                break;
            case Cons::None:
                encodeTraceDown(ctx, p, e);
                break;
            case Cons::DispDisp:
                barrier(e, kDisp, kDisp);
                encodeTraceDown(ctx, p, e);
                break;
            case Cons::TwoEncQueue:
            case Cons::TwoEncNone:
                e->endEncoding();
                e = cmd->computeCommandEncoder();
                if (var.cons == Cons::TwoEncQueue) e->barrierAfterQueueStages(kAS, kDisp, MTL4::VisibilityOptionDevice);
                encodeTraceDown(ctx, p, e);
                break;
            }
            e->endEncoding();
            const double ms = ctx.submit();
            const float* t = static_cast<const float*>(p.ts->contents());
            u32 wrong = 0;
            for (u32 i = 0; i < p.rays; ++i) wrong += !(std::fabs(t[i] - 1.0f) < 1e-3f);
            r.failures += wrong > 0;
            r.maxWrong = std::max(r.maxWrong, wrong);
            return ms;
        },
        reps);
    r.ms = st;
    return r;
}

void s1dOrdering(Context& ctx, Report& rep) {
    Verdict v;
    Plane p;
    p.n = ctx.quick() ? 354 : 708; // 250,632 / 1,002,528 triangles
    p.cell = 2.0f / float(p.n);
    p.rays = ctx.quick() ? 65536 : 262144;
    const u32 side = p.n + 1, tris = 2 * p.n * p.n;
    p.verts = ctx.buffer(size_t(side) * side * 12);
    p.indices = ctx.buffer(size_t(tris) * 12);
    auto* ix = static_cast<u32*>(p.indices->contents());
    for (u32 z = 0; z < p.n; ++z)
        for (u32 x = 0; x < p.n; ++x) {
            const u32 q = 6 * (z * p.n + x), v0 = z * side + x, v1 = v0 + 1, v2 = v0 + side, v3 = v2 + 1;
            ix[q] = v0; ix[q + 1] = v2; ix[q + 2] = v1; ix[q + 3] = v1; ix[q + 4] = v2; ix[q + 5] = v3;
        }
    p.ts = ctx.buffer(size_t(p.rays) * 4);
    p.pA = constBuf(ctx, PlaneParams{p.n, p.cell, 0.0f, 0});
    p.pB = constBuf(ctx, PlaneParams{p.n, p.cell, 1.0f, 0});
    p.pBslow = constBuf(ctx, PlaneParams{p.n, p.cell, 1.0f, ctx.quick() ? 4000u : 12000u});
    p.pTrace = constBuf(ctx, PlaneParams{p.n, p.cell, 0.0f, p.rays});
    MTL::Library* lib = f9Library(ctx, "s1_blas.metal");
    p.setState = ctx.compute(lib, "plane_set_state");
    p.traceDown = ctx.compute(lib, "trace_down");
    auto* geo = MTL4::AccelerationStructureTriangleGeometryDescriptor::alloc()->init();
    geo->setVertexBuffer(range(p.verts));
    geo->setVertexFormat(MTL::AttributeFormatFloat3);
    geo->setVertexStride(12);
    geo->setIndexBuffer(range(p.indices));
    geo->setIndexType(MTL::IndexTypeUInt32);
    geo->setTriangleCount(tris);
    geo->setOpaque(true);
    ctx.keep(geo);
    p.desc = MTL4::PrimitiveAccelerationStructureDescriptor::alloc()->init();
    p.desc->setGeometryDescriptors(NS::Array::array(geo));
    p.desc->setUsage(MTL::AccelerationStructureUsageRefit);
    ctx.keep(p.desc);
    const MTL::AccelerationStructureSizes sz = ctx.device()->accelerationStructureSizes(p.desc);
    p.as = newAccelerationStructure(ctx, sz.accelerationStructureSize, AsPlacement::Device);
    p.scratch = ctx.buffer(std::max<u64>(std::max(sz.buildScratchBufferSize, sz.refitScratchBufferSize), 4096), MTL::ResourceStorageModePrivate);
    ctx.commitResidency();
    rep.value("order.plane.triangles", "tri", double(tris), {{"as_bytes", double(sz.accelerationStructureSize)}, {"scratch_build", double(sz.buildScratchBufferSize)}, {"scratch_refit", double(sz.refitScratchBufferSize)}}, false);

    // Sanity: state A traces to t = 2, state B to t = 1 (the checker can tell them apart).
    {
        MTL4::CommandBuffer* cmd = ctx.beginCommands();
        MTL4::ComputeCommandEncoder* e = cmd->computeCommandEncoder();
        encodeSetState(ctx, p, e, p.pA);
        barrier(e, kDisp, kAS);
        e->buildAccelerationStructure(p.as, p.desc, range(p.scratch));
        barrier(e, kAS, kDisp);
        encodeTraceDown(ctx, p, e);
        e->endEncoding();
        ctx.submit();
        const float* t = static_cast<const float*>(p.ts->contents());
        u32 nA = 0;
        for (u32 i = 0; i < p.rays; ++i) nA += std::fabs(t[i] - 2.0f) < 1e-3f;
        rep.value("order.sanity.state_a_rays_at_t2", "rays", double(nA), {{"rays", double(p.rays)}}, true);
        if (nA != p.rays) v.fail("state A did not trace to t=2 for every ray (" + std::to_string(nA) + "/" + std::to_string(p.rays) + ")");
    }

    const u32 reps = ctx.quick() ? 6 : 20;
    const std::vector<OrderVar> vars = {
        {"build_trace", "barrier_enc", Prod::Barrier, Cons::Enc, true},
        {"build_trace", "none", Prod::Barrier, Cons::None, false},
        {"build_trace", "dispatch_dispatch", Prod::Barrier, Cons::DispDisp, false},
        {"build_trace", "two_enc_queue_barrier", Prod::Barrier, Cons::TwoEncQueue, true},
        {"build_trace", "two_enc_no_barrier", Prod::Barrier, Cons::TwoEncNone, false},
        {"write_build", "barrier", Prod::Barrier, Cons::Enc, true},
        {"write_build", "none", Prod::None, Cons::Enc, false},
        {"write_build", "slow_barrier", Prod::Barrier, Cons::Enc, true, true},
        {"write_build", "slow_none", Prod::None, Cons::Enc, false, true},
    };
    u32 unorderedFailing = 0, unorderedVariants = 0;
    std::string unorderedList;
    for (Op op : {Op::Build, Op::Refit}) {
        const char* opName = op == Op::Build ? "build" : "refit";
        for (const OrderVar& var : vars) {
            ctx.log("S1d %s.%s (%s)", var.pair, var.name, opName);
            const OrderRes r = runOrder(ctx, p, var, op, reps);
            const std::string pre = std::string("order.") + var.pair + "." + var.name + "." + opName;
            rep.value(pre + ".failures", "reps", double(r.failures), {{"reps", double(r.reps)}, {"max_wrong_rays", double(r.maxWrong)}, {"correct_variant", var.correct ? 1.0 : 0.0}}, false);
            rep.metric(pre + ".gpu.ms", "ms", r.ms, {{"triangles", double(tris)}}, false);
            if (var.correct && r.failures) v.fail(std::string("REQUIRED ordering ") + var.pair + "." + var.name + " (" + opName + ") failed in " + std::to_string(r.failures) + "/" + std::to_string(r.reps) + " reps");
            if (!var.correct) {
                ++unorderedVariants;
                if (r.failures) { ++unorderedFailing; unorderedList += std::string(var.pair) + "." + var.name + "." + opName + "(" + std::to_string(r.failures) + "/" + std::to_string(r.reps) + ") "; }
            }
        }
    }
    // Control of the checker: the op is skipped, the trace sees state A -> every rep must fail.
    const OrderRes ctl = runOrder(ctx, p, {"control", "no_op", Prod::Barrier, Cons::SkipOp, false}, Op::Build, reps);
    rep.value("order.control.no_op.failures", "reps", double(ctl.failures), {{"reps", double(ctl.reps)}, {"max_wrong_rays", double(ctl.maxWrong)}}, true);
    rep.value("order.unordered.variants_failing", "variants", double(unorderedFailing), {{"variants", double(unorderedVariants)}}, true);
    if (!v.ok) rep.status(Status::Failed, v.why);
    const bool detects = ctl.failures == ctl.reps;
    const bool raced = unorderedFailing > 0;
    if (!raced) rep.note("race control INCONCLUSIVE: none of the unordered variants ever failed in " + std::to_string(reps) + " repetitions; the method cannot prove that a barrier is needed for these pairs on this device/OS (a missing barrier is not shown to be harmless: the queue may simply serialise these dispatches)");
    else rep.note("unordered variants that produced wrong rays: " + unorderedList);
    rep.negative(v.ok && detects && raced,
                 std::string(v.ok ? "required orderings: 0 failures in every repetition; " : "FAILED: " + v.why + "; ") + "checker control (op skipped): " + std::to_string(ctl.failures) + "/" +
                     std::to_string(ctl.reps) + " reps fail (must be all); unordered variants failing: " + std::to_string(unorderedFailing) + "/" + std::to_string(unorderedVariants) +
                     (raced ? " [" + unorderedList + "]" : " (race control inconclusive: none failed)"));
}

} // namespace

SOC_BENCH("F9-S1", "blas_lifecycle", "F9 BLAS lifecycle: placement heaps, shared vs disjoint scratch, usage flags and scratch sizes", s1Placement);
SOC_BENCH("F9-S1b", "blas_compaction", "F9 BLAS async compaction: size query, later compact copy, bitwise identical hits", s1bCompaction);
SOC_BENCH("F9-S1c", "blas_refit", "F9 BLAS refit after a compute deformation: in/out of place, refit vs rebuild, quality", s1cRefit);
SOC_BENCH("F9-S1d", "blas_ordering", "F9 BLAS ordering: vertex write -> build/refit -> trace barriers, racing variants", s1dOrdering);

} // namespace f9
