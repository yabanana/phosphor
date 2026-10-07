#include <metal_stdlib>
#include "platform/metal/rt_visibility_check.h"
#include "renderer/visibility_math.h"
using namespace metal;
using namespace phosphor;

// Buffer ABI: counters0, params1, full primary rays2, full primary hits3,
// candidates A4 / B5, instances6, materials7. Texture0=R32Uint V-buffer,
// texture1=reverse-Z depth. Both stages use the engine's versioned graph refs.
static_assert(sizeof(GPURtVisibilityParams) == 176, "RT visibility params ABI");
static_assert(sizeof(GPURtVisibilityCounters) == 204, "RT visibility counters ABI");

kernel void rt_visibility_clear(device atomic_uint* counters [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    if (tid < sizeof(GPURtVisibilityCounters) / sizeof(uint))
        atomic_store_explicit(counters + tid, 0u, memory_order_relaxed);
}

inline float4x4 rtVisibilityMatrix(constant float* m) {
    return float4x4(float4(m[0], m[1], m[2], m[3]), float4(m[4], m[5], m[6], m[7]),
                    float4(m[8], m[9], m[10], m[11]), float4(m[12], m[13], m[14], m[15]));
}
struct RtRasterPixel { uint slot; bool hit, valid, mask; };
inline RtRasterPixel rtRasterPixel(uint id, constant GPURtVisibilityParams& p,
                                   const device GPUMeshletCandidate* a, const device GPUMeshletCandidate* b,
                                   const device GPUInstance* instances, const device GPUMaterial* materials) {
    RtRasterPixel r{~0u, id != VISIBILITY_BACKGROUND, true, false};
    if (!r.hit) return r;
    // Exactly visibility_resolve::candidateOf: ID is biased by one; phase B
    // clusters start at candidateCapacity, NOT at the current phase-A count.
    const uint cluster = visibilityCluster(id);
    if (cluster >= 2u * p.candidateCapacity) { r.valid = false; return r; }
    const GPUMeshletCandidate c = cluster < p.candidateCapacity ? a[cluster] : b[cluster - p.candidateCapacity];
    r.slot = c.slot;
    if (r.slot >= p.slotCount) { r.valid = false; return r; }
    const device GPUInstance& instance = instances[r.slot];
    if ((instance.flags & INSTANCE_FLAG_VALID) == 0u || instance.materialIndex >= p.materialCount) {
        r.valid = false;
        return r;
    }
    r.mask = materials[instance.materialIndex].alphaCutoff > 0.0f;
    return r;
}

inline void rtCount(device atomic_uint* counter, uint value, uint lane) {
    const uint total = simd_sum(value);
    if (lane == 0u && total != 0u) atomic_fetch_add_explicit(counter, total, memory_order_relaxed);
}

kernel void rt_visibility_compare(device GPURtVisibilityCounters* output [[buffer(0)]],
                                  constant GPURtVisibilityParams& p [[buffer(1)]],
                                  const device GPURtRay* rays [[buffer(2)]],
                                  const device GPURtHit* hits [[buffer(3)]],
                                  const device GPUMeshletCandidate* a [[buffer(4)]],
                                  const device GPUMeshletCandidate* b [[buffer(5)]],
                                  const device GPUInstance* instances [[buffer(6)]],
                                  const device GPUMaterial* materials [[buffer(7)]],
                                  texture2d<uint, access::read> visibility [[texture(0)]],
                                  texture2d<float, access::read> depth [[texture(1)]],
                                  uint tid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    if (tid >= p.rayCount) return;
    const uint2 pixel(tid % p.width, tid / p.width);
    const uint id = visibility.read(pixel).x;
    const RtRasterPixel raster = rtRasterPixel(id, p, a, b, instances, materials);
    const GPURtRay ray = rays[tid];
    const GPURtHit hit = hits[tid];
    bool rtValid = ray.pad == tid && ray.type == RT_PROBE_PRIMARY;
    const bool rtHit = hit.hit != 0u;
    bool rtMask = false;
    if (rtHit) {
        rtValid &= isfinite(hit.t) && hit.t >= ray.tmin && hit.t <= ray.tmax && hit.slot < p.slotCount;
        if (hit.slot < p.slotCount) {
            const device GPUInstance& instance = instances[hit.slot];
            rtValid &= (instance.flags & INSTANCE_FLAG_VALID) != 0u && instance.generation == hit.generation &&
                       instance.materialIndex < p.materialCount;
            if (instance.materialIndex < p.materialCount) rtMask = materials[instance.materialIndex].alphaCutoff > 0.0f;
        }
    } else rtValid &= hit.t == -1.0f; // -2 means skipped/invalid ray, not a valid sky miss
    const bool mask = raster.mask || rtMask;
    bool edge = false, maskNeighbor = false;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            const int2 neighbor = int2(pixel) + int2(x, y);
            if (any(neighbor < 0) || any(neighbor >= int2(p.width, p.height))) continue;
            const uint neighborId = visibility.read(uint2(neighbor)).x;
            edge |= neighborId != id;
            maskNeighbor |= rtRasterPixel(neighborId, p, a, b, instances, materials).mask;
        }
    }
    GPURtVisibilityCounters c{};
    c.compared = 1;
    c.bothBackground = !raster.hit && !rtHit;
    c.bothHit = raster.hit && rtHit;
    c.rasterHits = raster.hit;
    c.rtHits = rtHit;
    c.hitMiss = raster.hit != rtHit;
    c.slot = raster.hit && rtHit && raster.valid && rtValid && raster.slot != hit.slot;
    c.edgePixels = edge;
    c.maskPixels = mask;
    c.maskNeighborPixels = maskNeighbor;
    c.invalidVisibility = !raster.valid;
    c.invalidRt = !rtValid;
    const float rasterDepth = depth.read(pixel).x;
    c.depthWithoutVisibility = !raster.hit && rasterDepth != 0.0f;
    c.invalidDepth = !isfinite(rasterDepth) || rasterDepth < 0.0f || rasterDepth > 1.0f;
    uint maxUlp = 0;
    float depthError = 0, relativeError = 0;
    if (c.bothHit && raster.valid && rtValid && c.invalidDepth == 0u) {
        const float3 origin(ray.ox, ray.oy, ray.oz), direction(ray.dx, ray.dy, ray.dz);
        const float2 ndc = (float2(pixel) + 0.5f) * float2(2.0f, -2.0f) / float2(p.width, p.height) + float2(-1, 1);
        const float4 rasterWorld = rtVisibilityMatrix(p.inverseViewProjection) * float4(ndc, rasterDepth, 1);
        const float4 rtClip = rtVisibilityMatrix(p.viewProjection) * float4(origin + hit.t * direction, 1);
        const float dd = dot(direction, direction);
        if (!all(isfinite(rasterWorld)) || !all(isfinite(rtClip)) || abs(rasterWorld.w) < 1e-30f ||
            rtClip.w <= 0.0f || !isfinite(dd) || dd <= 0.0f) {
            c.invalidDepth = 1;
        } else {
            const float rasterT = dot(rasterWorld.xyz / rasterWorld.w - origin, direction) / dd;
            const float rtDepth = rtClip.z / rtClip.w;
            if (!isfinite(rasterT) || !isfinite(rtDepth) || rtDepth < 0.0f || rtDepth > 1.0f) {
                c.invalidDepth = 1;
            } else {
                relativeError = abs(hit.t - rasterT) / max(1.0f, abs(rasterT));
                c.depth = relativeError > p.relativeDistanceTolerance;
                depthError = abs(rtDepth - rasterDepth);
                const uint ra = as_type<uint>(max(rasterDepth, 0.0f)), rb = as_type<uint>(max(rtDepth, 0.0f));
                maxUlp = max(ra, rb) - min(ra, rb);
                c.depthBitDifferent = maxUlp != 0u;
                const uint bin = maxUlp == 0u ? 0u : maxUlp == 1u ? 1u : maxUlp <= 4u ? 2u :
                                 maxUlp <= 16u ? 3u : maxUlp <= 64u ? 4u : maxUlp <= 256u ? 5u : 6u;
                c.depthUlpHistogram[bin] = 1;
            }
        }
    }
    c.mismatches = c.hitMiss || c.slot || c.depth || c.invalidVisibility || c.invalidRt || c.invalidDepth ||
                   c.depthWithoutVisibility;
    const uint category = (edge ? 1u : 0u) + (mask ? 2u : 0u);
    c.category[category] = {1u, c.mismatches, c.hitMiss, c.slot, c.depth};
    // Scalar-only layout: all counts before frameLo are reduced identically.
    // The three maxima are filled separately below, not accumulated as counts.
    const thread uint* values = reinterpret_cast<const thread uint*>(&c);
    device atomic_uint* counters = reinterpret_cast<device atomic_uint*>(output);
    constexpr uint countWords = __builtin_offsetof(GPURtVisibilityCounters, frameLo) / sizeof(uint);
    for (uint i = 0; i < countWords; ++i) rtCount(counters + i, values[i], lane);
    const uint ulpMax = simd_max(maxUlp);
    const uint errorMax = simd_max(as_type<uint>(depthError));
    const uint relativeMax = simd_max(as_type<uint>(relativeError));
    if (lane == 0u) {
        atomic_fetch_max_explicit(reinterpret_cast<device atomic_uint*>(&output->maxDepthUlp), ulpMax, memory_order_relaxed);
        atomic_fetch_max_explicit(reinterpret_cast<device atomic_uint*>(&output->maxDepthErrorBits), errorMax, memory_order_relaxed);
        atomic_fetch_max_explicit(reinterpret_cast<device atomic_uint*>(&output->maxRelativeDistanceErrorBits), relativeMax, memory_order_relaxed);
    }
    if (tid == 0u) {
        output->frameLo = p.frameLo; output->frameHi = p.frameHi;
        output->width = p.width; output->height = p.height;
    }
}
