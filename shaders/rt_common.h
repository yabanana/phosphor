#pragma once
#include <metal_stdlib>
#include <metal_raytracing>
#include "renderer/gpu_types.h"

using namespace metal;
using namespace raytracing;
using namespace phosphor;

// F9 public shader ABI (no private pointer block):
// rt_trace_rays: AS0, rays1, hits2, GPURtProbeParams3, IFT4, instances5,
//               meshes6, vertices7, RT indices8, counters9.
// IFT function 0 rt_alpha_generic: materials0, texture handles1, vertices2,
//               RT indices3, instances4, GPURtMesh5, GPURtParams6.
// Texture handles match SceneRenderer's bindless array of 64-bit resource IDs.
// Host validates mesh/index/vertex ranges and material texture indices on upload.
struct RtTextureHandle { texture2d<float> tex; };
static_assert(sizeof(RtTextureHandle) == 8, "RT texture handle ABI");
static_assert(sizeof(GPURtInstanceDesc) == 72, "RT packed instance descriptor ABI");
static_assert(sizeof(GPURtRay) == 48 && sizeof(GPURtHit) == 32, "RT ray/hit ABI");

// Payload is local to traversal, not a host ABI. Counters are accumulated per
// ray, then reduced by SIMD-group; alpha tests do not atomically contend.
struct RtPayload {
    float coneWidth;
    uint type;
    uint alphaTests;
    uint opaqueAlphaTests;
};

inline float3 rtPosition(const device GPUVertex& v) { return float3(v.px, v.py, v.pz); }
inline float3 rtWorldPoint(const device GPUInstance& i, float3 p) {
    const device float* m = i.modelMatrix;
    return float3(m[0], m[1], m[2]) * p.x + float3(m[4], m[5], m[6]) * p.y +
           float3(m[8], m[9], m[10]) * p.z + float3(m[12], m[13], m[14]);
}

// Waechter & Binder, Ray Tracing Gems ch. 6. Apply to the reconstructed WORLD
// point and geometric WORLD normal, after choosing the outgoing hemisphere.
inline float3 rtOffsetRay(float3 p, float3 n) {
    const float origin = 1.0f / 32.0f, floatScale = 1.0f / 65536.0f;
    const int3 offset = int3(256.0f * n);
    const int3 bits = as_type<int3>(p);
    const float3 pi = as_type<float3>(bits + select(offset, -offset, p < 0.0f));
    return select(pi, p + floatScale * n, abs(p) < origin);
}

inline GPURtHit rtMiss(float distance = -1.0f) {
    GPURtHit h{};
    h.t = distance;
    h.slot = ~0u;
    h.primitive = ~0u;
    return h;
}

inline float rtConeLod(float width, float distance, float3 a, float3 b, float3 c,
                       float2 uv0, float2 uv1, float2 uv2, uint2 textureSize) {
    const float worldArea2 = length(cross(b - a, c - a));
    const float2 du = uv1 - uv0, dv = uv2 - uv0;
    const float uvArea2 = abs(du.x * dv.y - du.y * dv.x);
    if (!(width > 0.0f && worldArea2 > 0.0f && uvArea2 > 0.0f)) return 0.0f;
    const float texelWorld = sqrt(worldArea2 / (uvArea2 * float(textureSize.x) * float(textureSize.y)));
    return max(0.0f, log2(max(distance * width, 1e-12f) / max(texelWorld, 1e-20f)));
}

// Match the raster alpha arithmetic, including its half precision texel.
// LOD is explicit: no spatial derivatives exist in intersection functions.
constexpr sampler kRtAlphaSampler(filter::linear, mip_filter::linear, address::repeat);

[[intersection(triangle, triangle_data, instancing)]]
bool rt_alpha_generic(uint primitive [[primitive_id]], uint slot [[user_instance_id]],
                       float2 bary [[barycentric_coord]], float distance [[distance]],
                       ray_data RtPayload& payload [[payload]],
                       const device GPUMaterial* materials [[buffer(0)]],
                       const device RtTextureHandle* textures [[buffer(1)]],
                       const device GPUVertex* vertices [[buffer(2)]],
                       const device uint* indices [[buffer(3)]],
                       const device GPUInstance* instances [[buffer(4)]],
                       const device GPURtMesh* meshes [[buffer(5)]],
                       constant GPURtParams& params [[buffer(6)]]) {
    ++payload.alphaTests;
    if (slot >= params.slotCount) return false;
    const device GPUInstance& instance = instances[slot];
    if (instance.meshIndex >= params.meshCount || instance.materialIndex >= params.materialCount) return false;
    const device GPUMaterial& m = materials[instance.materialIndex];
    if (m.alphaCutoff <= 0.0f) {
        // Must stay zero: opaque instances have the hardware Opaque option.
        ++payload.opaqueAlphaTests;
        return true;
    }
    const device GPURtMesh& mesh = meshes[instance.meshIndex];
    if (primitive >= mesh.indexCount / 3u) return false;
    if (m.baseColorTex == INVALID_TEXTURE_INDEX) return m.baseColor[3] >= m.alphaCutoff;
    const uint base = mesh.indexOffset + 3u * primitive;
    const device GPUVertex& v0 = vertices[mesh.vertexOffset + indices[base]];
    const device GPUVertex& v1 = vertices[mesh.vertexOffset + indices[base + 1u]];
    const device GPUVertex& v2 = vertices[mesh.vertexOffset + indices[base + 2u]];
    const float2 uv0(v0.u, v0.v), uv1(v1.u, v1.v), uv2(v2.u, v2.v);
    const float2 uv = uv0 * (1.0f - bary.x - bary.y) + uv1 * bary.x + uv2 * bary.y;
    const auto texture = textures[m.baseColorTex].tex;
    float lod = 0.0f;
    if (payload.type == RT_PROBE_PRIMARY && payload.coneWidth > 0.0f) {
        lod = rtConeLod(payload.coneWidth, distance,
                        rtWorldPoint(instance, rtPosition(v0)), rtWorldPoint(instance, rtPosition(v1)),
                        rtWorldPoint(instance, rtPosition(v2)), uv0, uv1, uv2,
                        uint2(texture.get_width(), texture.get_height()));
    }
    const float alpha = m.baseColor[3] * float(half(texture.sample(kRtAlphaSampler, uv, level(lod)).a));
    return alpha >= m.alphaCutoff;
}

// All triangles use the generic alpha function at IFT slot 0. Hardware opacity
// on an instance bypasses it entirely. Explicit instance Opaque/NonOpaque
// overrides allow a mesh to be shared by opaque and MASK materials dynamically.
inline GPURtHit rtTrace(const GPURtRay r, instance_acceleration_structure as,
                        intersection_function_table<triangle_data, instancing> ift,
                        const device GPUInstance* instances, uint slotCount,
                        thread RtPayload& payload) {
    if (!(r.tmax >= r.tmin && r.tmin >= 0.0f) || r.mask == 0u) return rtMiss(-2.0f);
    const float3 origin(r.ox, r.oy, r.oz), direction(r.dx, r.dy, r.dz);
    if (!all(isfinite(origin)) || !all(isfinite(direction)) || dot(direction, direction) <= 0.0f)
        return rtMiss(-2.0f);
    intersector<triangle_data, instancing> traversal;
    traversal.assume_geometry_type(geometry_type::triangle);
    if (r.type == RT_PROBE_PRIMARY) traversal.set_triangle_cull_mode(triangle_cull_mode::back);
    if (r.type == RT_PROBE_SHADOW) traversal.accept_any_intersection(true);
    payload.coneWidth = r.coneWidth;
    payload.type = r.type;
    const auto result = traversal.intersect(ray(origin, direction, r.tmin, r.tmax), as, r.mask, ift, payload);
    if (result.type != intersection_type::triangle || result.user_instance_id >= slotCount) return rtMiss();
    GPURtHit hit{};
    hit.t = result.distance;
    hit.u = result.triangle_barycentric_coord.x;
    hit.v = result.triangle_barycentric_coord.y;
    hit.slot = result.user_instance_id;
    hit.primitive = result.primitive_id;
    hit.generation = instances[hit.slot].generation;
    // The raster reverses its cull mode for mirrored objects. Hardware RT
    // culls in object space (no descriptor CCW override), then we report the
    // world-geometric facing as S2 did. This does not change ray culling.
    hit.frontFacing = (result.triangle_front_facing !=
                       ((instances[hit.slot].flags & INSTANCE_FLAG_MIRRORED) != 0u)) ? 1u : 0u;
    hit.hit = 1u;
    return hit;
}

// Geometry fetch shared by secondary ray generation and the debug view. Proxy
// primitive IDs refer to the RT index stream, never the raster index stream.
inline bool rtSurface(GPURtHit hit, const device GPUInstance* instances, const device GPURtMesh* meshes,
                       const device GPUVertex* vertices, const device uint* indices,
                       uint slotCount, uint meshCount, thread float3& point, thread float3& normal) {
    if (hit.hit == 0u || hit.slot >= slotCount) return false;
    const device GPUInstance& instance = instances[hit.slot];
    if (instance.meshIndex >= meshCount || instance.generation != hit.generation) return false;
    const device GPURtMesh& mesh = meshes[instance.meshIndex];
    if (hit.primitive >= mesh.indexCount / 3u) return false;
    const uint base = mesh.indexOffset + 3u * hit.primitive;
    const float3 a = rtWorldPoint(instance, rtPosition(vertices[mesh.vertexOffset + indices[base]]));
    const float3 b = rtWorldPoint(instance, rtPosition(vertices[mesh.vertexOffset + indices[base + 1u]]));
    const float3 c = rtWorldPoint(instance, rtPosition(vertices[mesh.vertexOffset + indices[base + 2u]]));
    point = a * (1.0f - hit.u - hit.v) + b * hit.u + c * hit.v;
    const float3 n = cross(b - a, c - a);
    if (dot(n, n) <= 1e-30f) return false;
    normal = normalize(n);
    return all(isfinite(point)) && all(isfinite(normal));
}
