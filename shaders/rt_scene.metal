#include "rt_common.h"

// Buffer indices are deliberately explicit so PipelineCache hot reload and
// host argument tables cannot silently disagree with an opaque pointer block.
// Descriptor pipeline: instances0, materials1, meshes2, descriptors3, params4,
// counters5. Dispatch 1D, rounded up to a SIMD-group (32 on Apple9/Apple10).
// Ray/debug pipelines: AS0, rays1, hits2, probe3, IFT4, instances5, meshes6,
// vertices7, RT indices8, counters9, primaryHits10, primaryRays11; color texture0.
// All ray kernels dispatch a 1D grid. Pixel = (tid % width, tid / width).
// The counters use GPURtCounters word order. Clear once before descriptors;
// alpha counters are enabled by probe.flags bit0. bits1..2 select debug mode:
// 0 instance-slot color, 1 geometric world normal, 2 binary hit.

kernel void rt_clear_counters(device atomic_uint* counters [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    if (tid < sizeof(GPURtCounters) / sizeof(uint)) atomic_store_explicit(&counters[tid], 0u, memory_order_relaxed);
}

inline bool rtHasBlas(const device GPURtMesh& mesh) { return (mesh.blasLo | mesh.blasHi) != 0u; }

kernel void rt_write_instances(const device GPUInstance* instances [[buffer(0)]],
                                const device GPUMaterial* materials [[buffer(1)]],
                                const device GPURtMesh* meshes [[buffer(2)]],
                                device GPURtInstanceDesc* descriptors [[buffer(3)]],
                                constant GPURtParams& p [[buffer(4)]],
                                device atomic_uint* counters [[buffer(5)]],
                                uint tid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint active = 0u, masked = 0u, invalidMesh = 0u;
    // Host precondition: meshCount > 0, mesh 0 has a valid placeholder BLAS.
    // The masked identity descriptor must be well formed even for dead slots.
    if (tid < p.slotCount && p.meshCount > 0u) {
        const device GPUInstance& instance = instances[tid];
        const bool live = (instance.flags & INSTANCE_FLAG_VALID) != 0u;
        const bool meshValid = instance.meshIndex < p.meshCount && rtHasBlas(meshes[instance.meshIndex]);
        const bool materialValid = instance.materialIndex < p.materialCount;
        bool transformValid = true;
        for (uint i = 0; i < 16; ++i) transformValid &= isfinite(instance.modelMatrix[i]);
        const device float* m = instance.modelMatrix;
        const float3 c0(m[0], m[1], m[2]), c1(m[4], m[5], m[6]), c2(m[8], m[9], m[10]);
        const float det = dot(c0, cross(c1, c2));
        transformValid &= isfinite(det) && abs(det) > 1e-20f;
        const bool valid = live && meshValid && materialValid && transformValid;
        GPURtInstanceDesc d{};
        d.transform[0] = d.transform[4] = d.transform[8] = 1.0f;
        d.userID = tid; // descriptor index == stable scene slot == user ID
        uint meshIndex = 0u;
        if (valid) {
            meshIndex = instance.meshIndex;
            for (uint c = 0; c < 4; ++c)
                for (uint r = 0; r < 3; ++r) d.transform[c * 3u + r] = m[c * 4u + r];
            if ((instance.flags & 1u) != 0u) d.mask |= RT_MASK_PRIMARY | RT_MASK_INDIRECT;
            if ((instance.flags & 2u) != 0u) d.mask |= RT_MASK_SHADOW;
            const device GPUMaterial& material = materials[instance.materialIndex];
            // Exact values from MTLAccelerationStructure.hpp. S2's CCW option
            // on mirrored instances reports world-space geometric facing, but
            // would cull the wrong side for the raster's mirrored CullModeFront.
            // Keep object-space front-facing for hardware culling; rtTrace
            // converts the reported face to world-facing after traversal.
            if ((material.flags & MATERIAL_FLAG_DOUBLE_SIDED) != 0u) d.options |= 1u;
            if (material.alphaCutoff <= 0.0f) d.options |= 4u;
            if (p.corruption == RT_CORRUPT_TRANSFORM) d.transform[9] += 100000.0f;
            if (p.corruption == RT_CORRUPT_MASK) d.mask = 0u;
            if (p.corruption == RT_CORRUPT_BLAS) {
                // Never fabricate a resource ID. A different valid BLAS tests
                // the mapping independently of the mask control. Attribute
                // fetches still check primitive against the original mesh's
                // indexCount, so mismatched geometry cannot escape its ranges.
                const uint alternate = (meshIndex + 1u) % p.meshCount;
                const bool different = meshes[alternate].blasLo != meshes[meshIndex].blasLo ||
                                       meshes[alternate].blasHi != meshes[meshIndex].blasHi;
                if (different && rtHasBlas(meshes[alternate])) meshIndex = alternate;
                else d.mask = 0u; // one mesh / aliased IDs: safe deterministic mismatch
            }
        }
        d.blasLo = meshes[meshIndex].blasLo;
        d.blasHi = meshes[meshIndex].blasHi;
        descriptors[tid] = d;
        active = d.mask != 0u;
        masked = d.mask == 0u;
        invalidMesh = live && !meshValid;
    }
    const uint activeSum = simd_sum(active), maskedSum = simd_sum(masked), invalidSum = simd_sum(invalidMesh);
    if (lane == 0u) {
        if (activeSum) atomic_fetch_add_explicit(&counters[0], activeSum, memory_order_relaxed);
        if (maskedSum) atomic_fetch_add_explicit(&counters[1], maskedSum, memory_order_relaxed);
        if (invalidSum) atomic_fetch_add_explicit(&counters[2], invalidSum, memory_order_relaxed);
    }
}

inline float4x4 rtInverseViewProjection(constant GPURtProbeParams& p) {
    const constant float* m = p.inverseViewProjection;
    return float4x4(float4(m[0], m[1], m[2], m[3]), float4(m[4], m[5], m[6], m[7]),
                    float4(m[8], m[9], m[10], m[11]), float4(m[12], m[13], m[14], m[15]));
}
inline float3 rtPrimaryDirection(constant GPURtProbeParams& p, float2 pixel) {
    const float2 ndc = float2(2.0f, -2.0f) * pixel / float2(p.width, p.height) + float2(-1.0f, 1.0f);
    // Reverse-Z: depth 1 is the finite near plane. Using depth 0 would produce
    // w=0 with the engine's infinite-far projection.
    const float4 near = rtInverseViewProjection(p) * float4(ndc, 1.0f, 1.0f);
    return normalize(near.xyz / near.w - float3(p.cameraPosition[0], p.cameraPosition[1], p.cameraPosition[2]));
}

kernel void rt_generate_primary(device GPURtRay* rays [[buffer(1)]], constant GPURtProbeParams& p [[buffer(3)]],
                                 uint tid [[thread_position_in_grid]]) {
    if (tid >= p.rayCount || p.width == 0u || p.height == 0u) return;
    const float2 pixel = float2(tid % p.width, tid / p.width) + 0.5f;
    const float3 direction = rtPrimaryDirection(p, pixel);
    const float3 dx = rtPrimaryDirection(p, pixel + float2(1, 0));
    const float3 dy = rtPrimaryDirection(p, pixel + float2(0, 1));
    GPURtRay r{};
    r.ox = p.cameraPosition[0]; r.oy = p.cameraPosition[1]; r.oz = p.cameraPosition[2];
    r.dx = direction.x; r.dy = direction.y; r.dz = direction.z;
    r.tmax = 1e30f;
    r.mask = RT_MASK_PRIMARY;
    r.type = RT_PROBE_PRIMARY;
    r.coneWidth = max(length(dx - direction), length(dy - direction));
    rays[tid] = r;
}

inline uint rtHash(uint h) { h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; return h ^ (h >> 16); }
inline float rtRandom(uint pixel, uint seed) { return float(rtHash(pixel * 0x9e3779b1u ^ rtHash(seed)) >> 8) / 16777216.0f; }
inline float3 rtCosineDirection(float3 normal, float u, float v) {
    const float3 axis = abs(normal.x) > 0.9f ? float3(0, 1, 0) : float3(1, 0, 0);
    const float3 tangent = normalize(cross(axis, normal)), bitangent = cross(normal, tangent);
    const float radius = sqrt(u), angle = 6.28318530718f * v;
    return normalize(radius * (cos(angle) * tangent + sin(angle) * bitangent) + sqrt(max(0.0f, 1.0f - u)) * normal);
}

// Source primary rays/hits must remain separate from destination rays/hits.
// No invocation reads a neighbour; a Dispatch->Dispatch barrier orders stages.
kernel void rt_generate_secondary(device GPURtRay* rays [[buffer(1)]], constant GPURtProbeParams& p [[buffer(3)]],
                                   const device GPUInstance* instances [[buffer(5)]],
                                   const device GPURtMesh* meshes [[buffer(6)]],
                                   const device GPUVertex* vertices [[buffer(7)]], const device uint* indices [[buffer(8)]],
                                   const device GPURtHit* primaryHits [[buffer(10)]],
                                   const device GPURtRay* primaryRays [[buffer(11)]], uint tid [[thread_position_in_grid]]) {
    if (tid >= p.rayCount) return;
    GPURtRay r{};
    r.tmax = -1.0f; // skipped secondary: primary ray missed
    r.type = p.probeType;
    r.mask = p.probeType == RT_PROBE_SHADOW ? RT_MASK_SHADOW : RT_MASK_INDIRECT;
    float3 point, normal;
    if (rtSurface(primaryHits[tid], instances, meshes, vertices, indices, p.slotCount, p.meshCount, point, normal)) {
        const GPURtRay primary = primaryRays[tid];
        if (dot(normal, float3(primary.dx, primary.dy, primary.dz)) > 0.0f) normal = -normal;
        const float3 towardLight(p.lightDirection[0], p.lightDirection[1], p.lightDirection[2]);
        const float3 direction = p.probeType == RT_PROBE_SHADOW
                                     ? normalize(towardLight)
                                     : rtCosineDirection(normal, rtRandom(tid, p.frameIndex),
                                                          rtRandom(tid ^ 0x68bc21ebu, p.frameIndex + 7u));
        point = rtOffsetRay(point, dot(normal, direction) >= 0.0f ? normal : -normal);
        r.ox = point.x; r.oy = point.y; r.oz = point.z;
        r.dx = direction.x; r.dy = direction.y; r.dz = direction.z;
        r.tmax = p.lightDirection[3] > 0.0f ? p.lightDirection[3] : (p.probeType == RT_PROBE_AO ? 1.0f : 1e30f);
    }
    rays[tid] = r;
}

kernel void rt_trace_rays(instance_acceleration_structure as [[buffer(0)]], const device GPURtRay* rays [[buffer(1)]],
                           device GPURtHit* hits [[buffer(2)]], constant GPURtProbeParams& p [[buffer(3)]],
                           intersection_function_table<triangle_data, instancing> ift [[buffer(4)]],
                           const device GPUInstance* instances [[buffer(5)]], device atomic_uint* counters [[buffer(9)]],
                           uint tid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    RtPayload payload{};
    GPURtHit hit = rtMiss(-2.0f);
    if (tid < p.rayCount) {
        hit = rtTrace(rays[tid], as, ift, instances, p.slotCount, payload);
        hits[tid] = hit;
    }
    const uint rayCount = simd_sum(uint(tid < p.rayCount && hit.t != -2.0f));
    const uint hitCount = simd_sum(hit.hit);
    const uint alphaCount = simd_sum(payload.alphaTests), opaqueCount = simd_sum(payload.opaqueAlphaTests);
    if (lane == 0u) {
        if (rayCount) atomic_fetch_add_explicit(&counters[3], rayCount, memory_order_relaxed);
        if (hitCount) atomic_fetch_add_explicit(&counters[4], hitCount, memory_order_relaxed);
        if ((p.flags & 1u) != 0u) {
            if (alphaCount) atomic_fetch_add_explicit(&counters[5], alphaCount, memory_order_relaxed);
            if (opaqueCount) atomic_fetch_add_explicit(&counters[6], opaqueCount, memory_order_relaxed);
        }
    }
}

kernel void rt_debug_view(const device GPURtHit* hits [[buffer(2)]], constant GPURtProbeParams& p [[buffer(3)]],
                           const device GPUInstance* instances [[buffer(5)]], const device GPURtMesh* meshes [[buffer(6)]],
                           const device GPUVertex* vertices [[buffer(7)]], const device uint* indices [[buffer(8)]],
                           texture2d<float, access::write> color [[texture(0)]], uint tid [[thread_position_in_grid]]) {
    if (tid >= p.rayCount || p.width == 0u) return;
    const uint2 pixel(tid % p.width, tid / p.width);
    if (pixel.x >= color.get_width() || pixel.y >= color.get_height()) return;
    const GPURtHit h = hits[tid];
    float3 rgb(0.02f, 0.025f, 0.035f);
    if (h.hit != 0u) {
        const uint mode = (p.flags >> 1u) & 3u;
        const uint hash = rtHash(h.slot + 1u);
        rgb = 0.2f + 0.8f * float3(hash & 255u, (hash >> 8u) & 255u, (hash >> 16u) & 255u) / 255.0f;
        if (mode == 1u) {
            float3 point, normal;
            if (rtSurface(h, instances, meshes, vertices, indices, p.slotCount, p.meshCount, point, normal)) rgb = normal * 0.5f + 0.5f;
        } else if (mode == 2u) rgb = float3(1);
    }
    color.write(float4(rgb, 1), pixel);
}

// The debug image is linear RGBA16F. Present through a render pass because an
// sRGB drawable cannot be a compute-writable texture; the target performs the
// final linear -> sRGB conversion. The graph orders Dispatch -> Fragment.
struct RtPresentVertex {
    float4 position [[position]];
    float2 uv;
};
vertex RtPresentVertex rt_present_vs(uint id [[vertex_id]]) {
    const float2 xy = id == 0u ? float2(-1, -1) : id == 1u ? float2(3, -1) : float2(-1, 3);
    return {float4(xy, 0, 1), xy * float2(0.5f, -0.5f) + 0.5f};
}
fragment float4 rt_present_fs(RtPresentVertex in [[stage_in]], texture2d<float> image [[texture(0)]]) {
    constexpr sampler point(filter::nearest, address::clamp_to_edge);
    return float4(image.sample(point, in.uv).rgb, 1);
}
