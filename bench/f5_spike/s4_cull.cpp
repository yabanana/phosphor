// F5-S4: culling of 1,048,576 instances + compaction of the visible slots
// (docs/ROADMAP.md F5; opt-log "F5 — Spike").  Also the shared scene / CPU
// reference / GPU rig of F5-S7 (s4_cull.h).
//
// Scene: 8 buckets (= meshes) of random sizes in contiguous slot ranges, random
// positions in a +-1000 cube, random rotations, non-uniform scales 0.3-3, ~5%
// mirrored.  Camera at the origin looking down -Z, fov 60, 16:9, 3200x1800,
// reverse-Z infinite projection as src/scene/camera.cpp builds it.
// Cull test (one thread per slot): world sphere (model * centre, radius * max
// column scale) against 5 planes (left, right, bottom, top, near; no far),
// distance cull (distance - radius > 800) and screen-size cull (projected
// diameter in pixels < 1).  cullCpu() and shaders/s4_cull.metal use the same
// formulas and operation order; the CPU reference is compiled without FMA
// contraction.
//
// Variants:
//   (a) atomics: clear + one atomic per SIMD-group when its lanes share the
//       bucket (simd_prefix_exclusive_sum for lane offsets), per-lane atomic
//       otherwise; per bucket region [bucketFirst, bucketFirst + size).
//   (b) stable reduce-then-scan, 3 dispatches: flags + group counts / scan of
//       the 1024 counts / write list + per-slot prefix.  One global list in
//       slot order; per bucket first = prefix[bucketFirst], count = difference.
//   (b') single-pass decoupled look-back: NOT implemented -- it needs forward
//       progress between threadgroups (a threadgroup spinning on a predecessor
//       that is not resident deadlocks), which Apple GPUs do not guarantee and
//       the harness cannot prove robust; the reduce-then-scan of (b) is the
//       portable stable variant (3 dispatches, all with independent groups).
//
// Checks: visible set per bucket == CPU reference outside a +-1e-4 relative
// band around each decision boundary (default library = fast math, and
// MathModeSafe); (b) lists and prefixes are also compared EXACTLY with the
// compaction of the GPU's own flags; determinism: 10 runs per variant, (b)
// byte-identical, (a) order differences counted; negative controls: one plane
// sign flipped must fail the CPU check, and a swap of two list entries must be
// detected by the determinism comparison.
//
// Metrics (ms per dispatch, ComputeTimer laps): cull.atomics.{clear,cull,total}.ms,
// cull.stable.{flags,scan,write,total}.ms (+ the same under "cull.safe." for
// MathModeSafe); cull.visible_count, cull.ambiguous_count,
// cull.<variant>.<mode>.{mismatch_outside_band,mismatch_inside_band,structural_errors},
// cull.<variant>.<mode>.runs_differing (of 10 runs vs run 1).

#include "s4_cull.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>

#pragma STDC FP_CONTRACT OFF

namespace f5::cull {

using soc::BenchError;
using soc::Context;
using soc::Report;
using soc::Status;

namespace {

float rnd01(u64& s) { return float(soc::xorshift64(s) >> 40) * (1.0f / 16777216.0f); }

using Mat = std::array<std::array<float, 4>, 4>; // m[col][row], like glm

Mat mul(const Mat& a, const Mat& b) {
    Mat r{};
    for (int c = 0; c < 4; ++c)
        for (int row = 0; row < 4; ++row) {
            float s = 0.0f;
            for (int k = 0; k < 4; ++k) s += a[k][row] * b[c][k];
            r[c][row] = s;
        }
    return r;
}

// Camera::updateMatrices(): reverse-Z infinite far plane (near -> 1, infinity -> 0), no jitter.
Mat projection(float fovY, float aspect, float nearPlane) {
    Mat p{};
    const float f = 1.0f / std::tan(fovY * 0.5f);
    p[0][0]       = f / aspect;
    p[1][1]       = f;
    p[2][2]       = 0.0f;
    p[2][3]       = -1.0f;
    p[3][2]       = nearPlane;
    return p;
}

} // namespace

Scene makeScene(u64 seed) {
    Scene s;
    u64 rng = seed ? seed : 1;
    for (int i = 0; i < 8; ++i) soc::xorshift64(rng);
    // Random odd bucket sizes (so SIMD-groups straddle bucket boundaries); the last takes the rest (odd as well).
    float w[kBuckets], sum = 0.0f;
    for (u32 b = 0; b < kBuckets; ++b) sum += w[b] = 0.5f + 2.0f * rnd01(rng);
    u32 first = 0;
    for (u32 b = 0; b < kBuckets; ++b) {
        s.bucketFirst[b] = first;
        u32 size = b + 1 < kBuckets ? (u32(double(kN) * w[b] / sum) | 1u) : kN - first;
        first += size;
    }
    s.bucketFirst[kBuckets] = kN;
    for (u32 b = 0; b < kBuckets; ++b) {
        s.spheres[b][0] = rnd01(rng) - 0.5f;
        s.spheres[b][1] = rnd01(rng) - 0.5f;
        s.spheres[b][2] = rnd01(rng) - 0.5f;
        s.spheres[b][3] = 0.5f + 1.5f * rnd01(rng);
    }
    s.inst.resize(kN);
    const float twoPi = 6.28318530718f;
    for (u32 b = 0; b < kBuckets; ++b) {
        for (u32 i = s.bucketFirst[b]; i < s.bucketFirst[b + 1]; ++i) {
            Instance& in = s.inst[i];
            const float u1 = rnd01(rng), u2 = rnd01(rng), u3 = rnd01(rng);
            const float qx = std::sqrt(1.0f - u1) * std::sin(twoPi * u2), qy = std::sqrt(1.0f - u1) * std::cos(twoPi * u2),
                        qz = std::sqrt(u1) * std::sin(twoPi * u3), qw = std::sqrt(u1) * std::cos(twoPi * u3);
            float sc[3] = {0.3f + 2.7f * rnd01(rng), 0.3f + 2.7f * rnd01(rng), 0.3f + 2.7f * rnd01(rng)};
            if (rnd01(rng) < 0.05f) {
                sc[soc::xorshift64(rng) % 3] *= -1.0f;
                ++s.mirrored;
            }
            const float c0[3] = {1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy + qz * qw), 2 * (qx * qz - qy * qw)};
            const float c1[3] = {2 * (qx * qy - qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz + qx * qw)};
            const float c2[3] = {2 * (qx * qz + qy * qw), 2 * (qy * qz - qx * qw), 1 - 2 * (qx * qx + qy * qy)};
            for (int k = 0; k < 3; ++k) {
                in.m[0 + k] = c0[k] * sc[0];
                in.m[4 + k] = c1[k] * sc[1];
                in.m[8 + k] = c2[k] * sc[2];
            }
            in.m[3] = in.m[7] = in.m[11] = 0.0f;
            in.m[12] = (rnd01(rng) * 2.0f - 1.0f) * 1000.0f;
            in.m[13] = (rnd01(rng) * 2.0f - 1.0f) * 1000.0f;
            in.m[14] = (rnd01(rng) * 2.0f - 1.0f) * 1000.0f;
            in.m[15] = 1.0f;
            in.mesh  = b;
            in.pad[0] = in.pad[1] = in.pad[2] = 0;
        }
    }
    return s;
}

CullParams makeParams(const Scene& scene, float maxDistance, float minPixels) {
    constexpr float kFov = 60.0f * 3.14159265358979f / 180.0f, kNear = 0.05f, kW = 3200.0f, kH = 1800.0f;
    Mat view{}; // lookAt(origin, origin + (0,0,-1), up) is the identity
    for (int i = 0; i < 4; ++i) view[i][i] = 1.0f;
    const Mat proj = projection(kFov, kW / kH, kNear);
    const Mat m    = mul(proj, view);
    // Plane extraction from the rows of VP.  Metal clip volume: -w <= x <= w, -w <= y <= w, 0 <= z <= w.
    // Reverse-Z: z_clip = near * w_h is constant, ndc z = near / depth -> the NEAR plane is z <= w
    // (row3 - row2), the far plane at infinity (z >= 0) always holds and is not tested.  Note that
    // Camera::getFrustumPlanes() slot 4 (row3 + row2) is the GL-style near plane and is WRONG for this
    // projection (it is z <= +near: it accepts points behind the camera); its slot 5 (row3 - row2) is the near plane.
    auto row = [&](int r) { return std::array<float, 4>{m[0][r], m[1][r], m[2][r], m[3][r]}; };
    const auto r0 = row(0), r1 = row(1), r2 = row(2), r3 = row(3);
    std::array<std::array<float, 4>, 5> pl;
    for (int k = 0; k < 4; ++k) {
        pl[0][k] = r3[k] + r0[k]; // left
        pl[1][k] = r3[k] - r0[k]; // right
        pl[2][k] = r3[k] + r1[k]; // bottom
        pl[3][k] = r3[k] - r1[k]; // top
        pl[4][k] = r3[k] - r2[k]; // near (reverse-Z)
    }
    for (auto& p : pl) {
        const float len = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
        for (float& v : p) v /= len;
    }
    // Unit check on the CPU: hand-picked points with known classification, and random points against
    // an independent clip-space test (-w <= x,y <= w, z <= w, z >= 0).
    auto inPlanes = [&](float x, float y, float z) {
        for (const auto& p : pl)
            if (p[0] * x + p[1] * y + p[2] * z + p[3] < 0.0f) return false;
        return true;
    };
    auto inClip = [&](double x, double y, double z) {
        double c[4];
        for (int r = 0; r < 4; ++r) c[r] = double(m[0][r]) * x + double(m[1][r]) * y + double(m[2][r]) * z + double(m[3][r]);
        return std::fabs(c[0]) <= c[3] && std::fabs(c[1]) <= c[3] && c[2] <= c[3] && c[2] >= 0.0 && c[3] > 0.0;
    };
    struct P {
        float x, y, z;
        bool inside;
    };
    const float hx = kW / kH / std::tan(kFov * 0.5f) * 0.0f; // (unused, keeps the intent readable below)
    (void)hx;
    const float halfH = std::tan(kFov * 0.5f), halfW = halfH * kW / kH; // half extents per unit depth
    const P pts[] = {{0, 0, -10, true},
                     {0, 0, -0.06f, true},
                     {0, 0, -0.04f, false}, // in front of the near plane
                     {0, 0, 10, false},     // behind the camera
                     {0, 0, 0.04f, false},  // behind the camera but z <= +near (the engine's slot 4 would accept it)
                     {0, 0, -1.0e6f, true}, // infinite far plane
                     {halfW * 10 * 0.99f, 0, -10, true},
                     {halfW * 10 * 1.01f, 0, -10, false},
                     {-halfW * 10 * 0.99f, 0, -10, true},
                     {-halfW * 10 * 1.01f, 0, -10, false},
                     {0, halfH * 10 * 0.99f, -10, true},
                     {0, halfH * 10 * 1.01f, -10, false},
                     {0, -halfH * 10 * 0.99f, -10, true},
                     {0, -halfH * 10 * 1.01f, -10, false}};
    for (const P& q : pts) {
        if (inPlanes(q.x, q.y, q.z) != q.inside || inClip(q.x, q.y, q.z) != q.inside)
            throw BenchError("frustum planes: point (" + std::to_string(q.x) + "," + std::to_string(q.y) + "," + std::to_string(q.z) +
                             ") classified wrong (planes " + std::to_string(inPlanes(q.x, q.y, q.z)) + ", clip " +
                             std::to_string(inClip(q.x, q.y, q.z)) + ", expected " + std::to_string(q.inside) + ")");
    }
    u64 rng = 0xC0FFEE;
    u32 randomInside = 0;
    for (u32 i = 0; i < 20000; ++i) {
        const float x = (rnd01(rng) * 2 - 1) * 40.0f, y = (rnd01(rng) * 2 - 1) * 40.0f, z = (rnd01(rng) * 2 - 1) * 40.0f;
        const bool a = inPlanes(x, y, z), b = inClip(x, y, z);
        if (a != b) throw BenchError("frustum planes disagree with the clip-space test at a random point");
        randomInside += a;
    }
    if (randomInside == 0 || randomInside == 20000) throw BenchError("frustum unit check: random points all inside or all outside");
    (void)scene;

    CullParams p{};
    for (int i = 0; i < 5; ++i)
        for (int k = 0; k < 4; ++k) p.planes[i * 4 + k] = pl[i][k];
    p.cam[0] = p.cam[1] = p.cam[2] = 0.0f;
    p.fwd[0] = 0.0f;
    p.fwd[1] = 0.0f;
    p.fwd[2] = -1.0f;
    p.maxDistance   = maxDistance;
    p.minPixels     = minPixels;
    p.projY         = proj[1][1];
    p.halfViewportH = kH * 0.5f;
    p.nearPlane     = kNear;
    p.count         = kN;
    return p;
}

namespace {

struct Eval {
    bool vis;
    float rel[7]; // relative signed margin of each test (>= 0: passes): 5 planes, distance, screen size
};

// Same formulas and operation order as cullVisible() in s4_cull.metal.
Eval evalOne(const Instance& in, const float* sph, const CullParams& p) {
    Eval e{};
    const float cx = sph[0], cy = sph[1], cz = sph[2];
    const float wx = in.m[0] * cx + in.m[4] * cy + in.m[8] * cz + in.m[12];
    const float wy = in.m[1] * cx + in.m[5] * cy + in.m[9] * cz + in.m[13];
    const float wz = in.m[2] * cx + in.m[6] * cy + in.m[10] * cz + in.m[14];
    const float s0 = in.m[0] * in.m[0] + in.m[1] * in.m[1] + in.m[2] * in.m[2];
    const float s1 = in.m[4] * in.m[4] + in.m[5] * in.m[5] + in.m[6] * in.m[6];
    const float s2 = in.m[8] * in.m[8] + in.m[9] * in.m[9] + in.m[10] * in.m[10];
    const float smax   = std::max(std::max(s0, s1), s2);
    const float radius = sph[3] * std::sqrt(smax);
    bool vis = true;
    for (u32 i = 0; i < 5; ++i) {
        const float dist = p.planes[i * 4 + 0] * wx + p.planes[i * 4 + 1] * wy + p.planes[i * 4 + 2] * wz + p.planes[i * 4 + 3];
        if (dist + radius < 0.0f) vis = false;
        e.rel[i] = (dist + radius) / (std::fabs(dist) + radius + 1.0f);
    }
    const float dx  = wx - p.cam[0];
    const float dy  = wy - p.cam[1];
    const float dz  = wz - p.cam[2];
    const float len = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (len - radius > p.maxDistance) vis = false;
    e.rel[5] = (p.maxDistance - (len - radius)) / (std::fabs(len - radius) + p.maxDistance);
    float depth = dx * p.fwd[0] + dy * p.fwd[1] + dz * p.fwd[2];
    depth       = std::max(depth, p.nearPlane);
    const float diam = 2.0f * radius * p.projY * p.halfViewportH / depth;
    if (diam < p.minPixels) vis = false;
    e.rel[6] = (diam - p.minPixels) / (diam + p.minPixels);
    e.vis    = vis;
    return e;
}

} // namespace

CpuRef cullCpu(const Scene& scene, const CullParams& p) {
    CpuRef r;
    r.vis.assign(kN, 0);
    r.ambiguous.assign(kN, 0);
    for (u32 b = 0; b < kBuckets; ++b)
        for (u32 i = scene.bucketFirst[b]; i < scene.bucketFirst[b + 1]; ++i) {
            const Eval e = evalOne(scene.inst[i], scene.spheres[scene.inst[i].mesh], p);
            bool loose = true, strict = true;
            for (float rel : e.rel) {
                loose &= rel >= -kBand;
                strict &= rel >= kBand;
            }
            r.vis[i]       = e.vis;
            r.ambiguous[i] = loose && !strict;
            r.visible += e.vis;
            r.bucketVisible[b] += e.vis;
            r.ambiguousCount += r.ambiguous[i];
        }
    return r;
}

// ---- GPU rig ------------------------------------------------------------------------------------------

namespace {

Pipes makePipes(Context& ctx, MTL::Library* lib) {
    Pipes p{};
    p.clear  = ctx.compute(lib, "s4_clear");
    p.atomic = ctx.compute(lib, "s4_cull_atomic");
    p.flags  = ctx.compute(lib, "s4_flags");
    p.scan   = ctx.compute(lib, "s4_scan_groups");
    p.write  = ctx.compute(lib, "s4_write_list");
    for (auto* pso : {p.flags, p.scan, p.write})
        if (pso->maxTotalThreadsPerThreadgroup() < kGroupSz)
            throw BenchError("pipeline max threads per threadgroup " + std::to_string(pso->maxTotalThreadsPerThreadgroup()) + " < 1024");
    return p;
}

} // namespace

Rig makeRig(Context& ctx, const Scene& scene, const CullParams& p) {
    Rig r;
    MTL::Library* fast = f5::f5Library(ctx, "s4_cull.metal", true);
    MTL::Library* safe = f5::f5Library(ctx, "s4_cull.metal", false);
    r.fast             = makePipes(ctx, fast);
    r.safe             = makePipes(ctx, safe);
    r.inst             = ctx.buffer(size_t(kN) * sizeof(Instance));
    std::memcpy(r.inst->contents(), scene.inst.data(), size_t(kN) * sizeof(Instance));
    r.spheres = ctx.buffer(kBuckets * 16);
    std::memcpy(r.spheres->contents(), scene.spheres, kBuckets * 16);
    r.bucketFirst = ctx.buffer(256);
    std::memcpy(r.bucketFirst->contents(), scene.bucketFirst, sizeof(scene.bucketFirst));
    r.params = ctx.buffer(256);
    setParams(r, p);
    r.counters    = ctx.buffer(256);
    r.listAtomic  = ctx.buffer(size_t(kN) * 4);
    r.flags       = ctx.buffer(kN);
    r.groupCounts = ctx.buffer(kGroups * 4);
    r.groupOffs   = ctx.buffer((kGroups + 1) * 4);
    r.prefix      = ctx.buffer(size_t(kN + 1) * 4);
    r.listStable  = ctx.buffer(size_t(kN) * 4);
    for (MTL::Buffer* b : {r.listAtomic, r.listStable, r.prefix, r.groupCounts, r.groupOffs, r.flags, r.counters}) std::memset(b->contents(), 0xFF, b->length());
    return r;
}

void setParams(Rig& rig, const CullParams& p) { std::memcpy(rig.params->contents(), &p, sizeof(p)); }

namespace {
void bind(Context& ctx, MTL4::ComputeCommandEncoder* e, MTL::ComputePipelineState* pso, std::initializer_list<MTL::Buffer*> bufs) {
    u32 i = 0;
    for (MTL::Buffer* b : bufs) ctx.table()->setAddress(b->gpuAddress(), i++);
    e->setComputePipelineState(pso);
    e->setArgumentTable(ctx.table());
}
} // namespace

void encodeAtomic(Context& ctx, MTL4::ComputeCommandEncoder* e, const Rig& rig, const Pipes& pl, const std::function<void()>& lap) {
    bind(ctx, e, pl.clear, {rig.counters});
    e->dispatchThreads(MTL::Size::Make(32, 1, 1), MTL::Size::Make(32, 1, 1));
    lap();
    bind(ctx, e, pl.atomic, {rig.inst, rig.spheres, rig.params, rig.counters, rig.listAtomic, rig.bucketFirst});
    e->dispatchThreads(MTL::Size::Make(kN, 1, 1), MTL::Size::Make(256, 1, 1));
    lap();
}

void encodeStable(Context& ctx, MTL4::ComputeCommandEncoder* e, const Rig& rig, const Pipes& pl, const std::function<void()>& lap) {
    bind(ctx, e, pl.flags, {rig.inst, rig.spheres, rig.params, rig.flags, rig.groupCounts});
    e->dispatchThreadgroups(MTL::Size::Make(kGroups, 1, 1), MTL::Size::Make(kGroupSz, 1, 1));
    lap();
    bind(ctx, e, pl.scan, {rig.groupCounts, rig.groupOffs});
    e->dispatchThreadgroups(MTL::Size::Make(1, 1, 1), MTL::Size::Make(kGroupSz, 1, 1));
    lap();
    bind(ctx, e, pl.write, {rig.flags, rig.groupOffs, rig.prefix, rig.listStable, rig.params});
    e->dispatchThreadgroups(MTL::Size::Make(kGroups, 1, 1), MTL::Size::Make(kGroupSz, 1, 1));
    lap();
}

// ---- benchmark -------------------------------------------------------------------------------------------

namespace {

constexpr u32 kDetRuns = 10;

std::vector<double> runVariant(Context& ctx, const Rig& rig, const Pipes& pl, bool stable) {
    soc::ComputeTimer t(ctx);
    MTL4::ComputeCommandEncoder* e = t.begin();
    if (stable) encodeStable(ctx, e, rig, pl, [&] { t.lap(); });
    else encodeAtomic(ctx, e, rig, pl, [&] { t.lap(); });
    return t.finish();
}

struct Gpu {
    std::vector<u8> member;  // slot -> in the GPU's list
    std::vector<u32> snap;   // determinism snapshot (counts + lists [+ prefix])
    u32 total = 0;
    u32 structural = 0;      // out of range / duplicate / unsorted / inconsistent entries (must be 0)
};

Gpu readAtomic(const Rig& rig, const Scene& sc) {
    Gpu g;
    g.member.assign(kN, 0);
    const auto* cnt  = static_cast<const u32*>(rig.counters->contents());
    const auto* list = static_cast<const u32*>(rig.listAtomic->contents());
    for (u32 b = 0; b < kBuckets; ++b) g.snap.push_back(cnt[b]);
    for (u32 b = 0; b < kBuckets; ++b) {
        const u32 lo = sc.bucketFirst[b], size = sc.bucketFirst[b + 1] - lo;
        u32 n = cnt[b];
        if (n > size) {
            ++g.structural;
            n = size;
        }
        g.total += n;
        for (u32 k = 0; k < n; ++k) {
            const u32 slot = list[lo + k];
            g.snap.push_back(slot);
            if (slot < lo || slot >= lo + size || g.member[slot]) ++g.structural;
            else g.member[slot] = 1;
        }
    }
    return g;
}

Gpu readStable(const Rig& rig, const Scene& sc) {
    Gpu g;
    g.member.assign(kN, 0);
    const auto* prefix = static_cast<const u32*>(rig.prefix->contents());
    const auto* list   = static_cast<const u32*>(rig.listStable->contents());
    const auto* flags  = static_cast<const u8*>(rig.flags->contents());
    g.total            = prefix[kN];
    // Exact check against the GPU's own flags: prefix = running count, list = compaction in slot order.
    u32 run = 0;
    for (u32 i = 0; i < kN; ++i) {
        if (prefix[i] != run) ++g.structural;
        if (flags[i]) {
            if (run >= g.total || list[run] != i) ++g.structural;
            ++run;
        }
    }
    if (run != g.total) ++g.structural;
    for (u32 b = 0; b < kBuckets; ++b) {
        const u32 lo = sc.bucketFirst[b], hi = sc.bucketFirst[b + 1];
        const u32 first = prefix[lo], count = prefix[hi] - first;
        g.snap.push_back(first);
        g.snap.push_back(count);
        for (u32 k = 0; k < count; ++k) {
            const u32 slot = list[first + k];
            if (slot < lo || slot >= hi || g.member[slot] || (k > 0 && slot <= list[first + k - 1])) ++g.structural;
            else g.member[slot] = 1;
        }
    }
    g.snap.insert(g.snap.end(), list, list + g.total);
    g.snap.insert(g.snap.end(), prefix, prefix + kN + 1);
    return g;
}

struct Cmp {
    u32 outside = 0, inside = 0;
};
Cmp compare(const std::vector<u8>& member, const CpuRef& ref) {
    Cmp c;
    for (u32 i = 0; i < kN; ++i)
        if ((member[i] != 0) != (ref.vis[i] != 0)) (ref.ambiguous[i] ? c.inside : c.outside) += 1;
    return c;
}

u32 crossDiff(const std::vector<u8>& a, const std::vector<u8>& b) {
    u32 n = 0;
    for (u32 i = 0; i < kN; ++i) n += (a[i] != 0) != (b[i] != 0);
    return n;
}

// The determinism comparison and its self-test: a copy with two entries swapped must be detected.
bool sameBytes(const std::vector<u32>& a, const std::vector<u32>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
}
bool detectorWorks(const Gpu& g, bool atomicsLayout) {
    std::vector<u32> copy = g.snap;
    if (!sameBytes(copy, g.snap)) return false; // identical copies must compare equal
    // Swap two different entries inside the first list region (after the 8 counts / 16 first+count values).
    const size_t base = atomicsLayout ? kBuckets : 2 * kBuckets;
    for (size_t i = base; i + 1 < copy.size(); ++i)
        if (copy[i] != copy[i + 1]) {
            std::swap(copy[i], copy[i + 1]);
            return !sameBytes(copy, g.snap);
        }
    return false;
}

std::string str(double v, size_t n = 7) { return std::to_string(v).substr(0, n); }

struct ModeResult {
    u32 outsideA = 0, insideA = 0, structA = 0, outsideB = 0, insideB = 0, structB = 0, cross = 0;
    u32 diffRunsA = 0, diffRunsB = 0, visibleA = 0, visibleB = 0;
    bool detectA = false, detectB = false;
};

ModeResult checkMode(Context& ctx, const Rig& rig, const Pipes& pl, const Scene& sc, const CpuRef& ref) {
    ModeResult m;
    // (a)
    runVariant(ctx, rig, pl, false);
    const Gpu a1 = readAtomic(rig, sc);
    const Cmp ca = compare(a1.member, ref);
    m.outsideA = ca.outside;
    m.insideA  = ca.inside;
    m.structA  = a1.structural;
    m.visibleA = a1.total;
    m.detectA  = detectorWorks(a1, true);
    for (u32 r = 1; r < kDetRuns; ++r) {
        runVariant(ctx, rig, pl, false);
        const Gpu g = readAtomic(rig, sc);
        if (!sameBytes(g.snap, a1.snap)) ++m.diffRunsA;
    }
    // (b)
    runVariant(ctx, rig, pl, true);
    const Gpu b1 = readStable(rig, sc);
    const Cmp cb = compare(b1.member, ref);
    m.outsideB = cb.outside;
    m.insideB  = cb.inside;
    m.structB  = b1.structural;
    m.visibleB = b1.total;
    m.detectB  = detectorWorks(b1, false);
    for (u32 r = 1; r < kDetRuns; ++r) {
        runVariant(ctx, rig, pl, true);
        const Gpu g = readStable(rig, sc);
        if (!sameBytes(g.snap, b1.snap)) ++m.diffRunsB;
        m.structB += g.structural;
    }
    m.cross = crossDiff(a1.member, b1.member);
    return m;
}

void benchCull(Context& ctx, Report& rep) {
    const Scene scene = makeScene(0xF5A4);
    CullParams params = makeParams(scene);
    Rig rig           = makeRig(ctx, scene, params);
    const CpuRef ref  = cullCpu(scene, params);
    ctx.log("F5-S4: %u instances, %u visible (CPU), %u ambiguous within +-%.0e, %u mirrored; bucket sizes %u %u %u %u %u %u %u %u",
            kN, ref.visible, ref.ambiguousCount, double(kBand), scene.mirrored, scene.bucketFirst[1] - scene.bucketFirst[0],
            scene.bucketFirst[2] - scene.bucketFirst[1], scene.bucketFirst[3] - scene.bucketFirst[2],
            scene.bucketFirst[4] - scene.bucketFirst[3], scene.bucketFirst[5] - scene.bucketFirst[4],
            scene.bucketFirst[6] - scene.bucketFirst[5], scene.bucketFirst[7] - scene.bucketFirst[6],
            scene.bucketFirst[8] - scene.bucketFirst[7]);
    rep.value("cull.visible_count", "count", ref.visible, {{"instances", double(kN)}}, false);
    rep.value("cull.ambiguous_count", "count", ref.ambiguousCount, {{"band", double(kBand)}}, false);
    rep.value("cull.mirrored_count", "count", scene.mirrored, {}, false);
    ctx.warmUp(5.0);

    // ---- correctness + determinism, both math modes -----------------------------------------------------
    struct Mode {
        const char* name;
        const Pipes* pl;
    };
    const Mode modes[] = {{"fast", &rig.fast}, {"safe", &rig.safe}};
    bool ok = true;
    std::string failWhy, detail;
    bool detectors = true;
    for (const Mode& md : modes) {
        const ModeResult m = checkMode(ctx, rig, *md.pl, scene, ref);
        const std::string t = std::string(".") + md.name;
        rep.value("cull.atomics.mismatch_outside_band" + t, "count", m.outsideA, {}, false);
        rep.value("cull.atomics.mismatch_inside_band" + t, "count", m.insideA, {}, false);
        rep.value("cull.atomics.structural_errors" + t, "count", m.structA, {}, false);
        rep.value("cull.atomics.visible" + t, "count", m.visibleA, {}, false);
        rep.value("cull.atomics.runs_differing" + t, "runs", m.diffRunsA, {{"runs", double(kDetRuns)}}, false);
        rep.value("cull.stable.mismatch_outside_band" + t, "count", m.outsideB, {}, false);
        rep.value("cull.stable.mismatch_inside_band" + t, "count", m.insideB, {}, false);
        rep.value("cull.stable.structural_errors" + t, "count", m.structB, {}, false);
        rep.value("cull.stable.visible" + t, "count", m.visibleB, {}, false);
        rep.value("cull.stable.runs_differing" + t, "runs", m.diffRunsB, {{"runs", double(kDetRuns)}}, false);
        rep.value("cull.cross_variant_diff" + t, "count", m.cross, {}, false);
        ctx.log("F5-S4 [%s]: (a) visible %u outside-band %u inside-band %u struct %u, %u/%u runs differ in order | (b) visible %u "
                "outside-band %u inside-band %u struct %u, %u/%u runs differ | a-vs-b %u slots",
                md.name, m.visibleA, m.outsideA, m.insideA, m.structA, m.diffRunsA, kDetRuns - 1, m.visibleB, m.outsideB, m.insideB,
                m.structB, m.diffRunsB, kDetRuns - 1, m.cross);
        if (m.outsideA || m.outsideB || m.structA || m.structB) {
            ok = false;
            failWhy += std::string(md.name) + ": mismatches outside the band / structural errors; ";
        }
        if (m.diffRunsB) {
            ok = false;
            failWhy += std::string(md.name) + ": stable variant not deterministic; ";
        }
        detectors &= m.detectA && m.detectB;
        detail += std::string(md.name) + " a:" + std::to_string(m.diffRunsA) + "/" + std::to_string(kDetRuns - 1) + " runs differ (b:" +
                  std::to_string(m.diffRunsB) + "), ";
    }

    // ---- negative control: flip the sign of one plane (the left plane): the CPU check must FAIL ------------
    CullParams flipped = params;
    for (int k = 0; k < 4; ++k) flipped.planes[k] = -flipped.planes[k];
    setParams(rig, flipped);
    runVariant(ctx, rig, rig.fast, true);
    const Cmp flip = compare(readStable(rig, scene).member, ref);
    setParams(rig, params);
    const bool flipDetected = flip.outside > ref.visible / 100; // flipping a side plane changes the visible set by far more than 1%
    ctx.log("F5-S4 control: left plane flipped -> %u slots outside the band (visible %u)", flip.outside, ref.visible);

    // ---- timing (ComputeTimer laps), both math modes ------------------------------------------------------
    for (const Mode& md : modes) {
        const std::string pre = md.name[0] == 'f' ? "cull." : "cull.safe.";
        ctx.keepWarm(50);
        for (int stable = 0; stable < 2; ++stable) {
            const size_t nl = stable ? 3 : 2;
            std::vector<std::vector<double>> laps(nl);
            const soc::Stats total = ctx.measure([&] {
                const std::vector<double> l = runVariant(ctx, rig, *md.pl, stable != 0);
                double s = 0;
                for (size_t i = 0; i < nl; ++i) {
                    laps[i].push_back(l[i]);
                    s += l[i];
                }
                return s;
            });
            static const char* an[] = {"clear", "cull"};
            static const char* sn[] = {"flags", "scan", "write"};
            for (size_t i = 0; i < nl; ++i)
                rep.metric(pre + (stable ? "stable." : "atomics.") + (stable ? sn[i] : an[i]) + ".ms", "ms",
                           phosphor::soc::computeStats(laps[i]), {{"instances", double(kN)}}, false);
            rep.metric(pre + (stable ? "stable" : "atomics") + ".total.ms", "ms", total, {{"instances", double(kN)}}, false);
            ctx.keepWarm(20);
        }
    }

    rep.negative(flipDetected && detectors,
                 std::string("left plane flipped -> ") + std::to_string(flip.outside) + " slots differ from the CPU reference outside the band (must be > " +
                     std::to_string(ref.visible / 100) + "); determinism comparison detects a swap of two list entries: " +
                     (detectors ? "yes" : "NO") + " [" + detail + "]");
    if (!ok) rep.status(Status::Failed, failWhy);
    rep.note("fast = default library, safe = MathModeSafe; band = relative margin " + str(double(kBand) * 1e4, 3) +
             "e-4 (margin / (|dist| + radius + 1) for planes, / (|len - r| + maxDistance) for distance, / (diam + minPixels) for screen size); "
             "ComputeTimer laps: one dispatch each, Dispatch->Dispatch barrier between. (b') decoupled look-back not implemented: no forward-progress "
             "guarantee between threadgroups on Apple GPUs. Camera::getFrustumPlanes() slot 4 (row3+row2) is not the near plane of the reverse-Z "
             "projection (z <= +near, accepts points behind the camera): slot 5 (row3-row2) is; this spike uses row3-row2.");
}

} // namespace

} // namespace f5::cull

namespace soc {
SOC_BENCH("F5-S4", "scene.cull", "F5-S4: culling of 1M instances + stable compaction (scan) vs atomics", f5::cull::benchCull);
} // namespace soc
