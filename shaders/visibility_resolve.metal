#include "material_shading.h"
#include "temporal_motion.h"
#include "renderer/visibility_math.h"
#include "renderer/visibility_layout.h"

static float4x4 visibilityMatrix(const device float *m) {
    return float4x4(float4(m[0], m[1], m[2], m[3]), float4(m[4], m[5], m[6], m[7]), float4(m[8], m[9], m[10], m[11]),
                    float4(m[12], m[13], m[14], m[15]));
}
static float4x4 visibilityMatrix(constant float *m) {
    return float4x4(float4(m[0], m[1], m[2], m[3]), float4(m[4], m[5], m[6], m[7]), float4(m[8], m[9], m[10], m[11]),
                    float4(m[12], m[13], m[14], m[15]));
}
static GPUMeshletCandidate candidateOf(uint id, constant GPUVisibilityParams &p, const device GPUMeshletCandidate *a,
                                       const device GPUMeshletCandidate *b) {
    const uint cluster = visibilityCluster(id);
    return cluster < p.candidateCapacity ? a[cluster] : b[cluster - p.candidateCapacity];
}
static uint materialClass(const device GPUMaterial &m, constant GPUVisibilityParams &p) {
    const uint normal = (m.normalTex != INVALID_TEXTURE_INDEX && m.normalTex != p.defaultNormal) ? 1u : 0u;
    const uint emissive = (m.emissive[0] != 0 || m.emissive[1] != 0 || m.emissive[2] != 0) ? 2u : 0u;
    return normal | emissive;
}

kernel void visibility_clear(constant GPUVisibilityParams &p [[buffer(11)]], device uint *args [[buffer(13)]],
                             texture2d<float, access::write> color [[texture(1)]],
                             texture2d<float, access::write> normal [[texture(2)]],
                             texture2d<float, access::write> diffuse [[texture(3)]],
                             texture2d<float, access::write> specular [[texture(4)]],
                             texture2d<float, access::write> motion [[texture(5)]],
                             texture2d<float, access::write> reactive [[texture(6)]],
                             uint2 pixel [[thread_position_in_grid]]) {
    if (pixel.y == 0 && pixel.x < 16)
        args[pixel.x] = pixel.x >= 12 ? 0u : (pixel.x % 3 == 0) ? 0u : 1u;
    if (pixel.x >= p.outputWidth || pixel.y >= p.outputHeight)
        return;
    color.write(float4(0.02f, 0.025f, 0.035f, 1), pixel);
    normal.write(float4(0), pixel);
    diffuse.write(float4(0), pixel);
    specular.write(float4(0), pixel);
    motion.write(float4(0), pixel);
    reactive.write(float4(1), pixel);
}

kernel void visibility_classify(const device GPUInstance *instances [[buffer(2)]],
                                const device GPUMaterial *materials [[buffer(3)]],
                                const device GPUMeshletCandidate *a [[buffer(9)]],
                                const device GPUMeshletCandidate *b [[buffer(10)]],
                                constant GPUVisibilityParams &p [[buffer(11)]], device uint *bins [[buffer(12)]],
                                device atomic_uint *args [[buffer(13)]],
                                texture2d<uint, access::read> visibility [[texture(0)]],
                                uint2 pixel [[thread_position_in_grid]], uint2 tile [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint mask;
    if (tid == 0)
        atomic_store_explicit(&mask, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (pixel.x < p.width && pixel.y < p.height) {
        const uint id = visibility.read(pixel).x;
        if (id != VISIBILITY_BACKGROUND && visibilityCluster(id) < 2u * p.candidateCapacity) {
            const auto candidate = candidateOf(id, p, a, b);
            const uint cls = materialClass(materials[instances[candidate.slot].materialIndex], p);
            atomic_fetch_or_explicit(&mask, 1u << cls, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        const uint bits = atomic_load_explicit(&mask, memory_order_relaxed);
        const uint tiles = p.tilesX * p.tilesY;
        for (uint cls = 0; cls < VISIBILITY_CLASSES; ++cls)
            if (bits & (1u << cls)) {
                const uint index = atomic_fetch_add_explicit(&args[cls * 3], 1u, memory_order_relaxed);
                if (index < tiles)
                    bins[cls * tiles + index] = tile.y * p.tilesX + tile.x;
            }
    }
}

static void resolveVisibilityPixel(constant FrameConstants &frame, const device GPUVertex *vertices,
                                   const device GPUInstance *instances, const device GPUMaterial *materials,
                                   const device GPULight *lights, const device TextureHandle *textures,
                                   const device GPUMeshlet *meshlets, const device uint *meshletVertices,
                                   const device uchar *triangles, const device GPUMeshletCandidate *a,
                                   const device GPUMeshletCandidate *b, constant GPUVisibilityParams &p,
                                   const device GPUInstance *previousInstances, constant GPUTemporalParams &temporal,
                                   texture2d<float, access::write> color, texture2d<float, access::write> normal,
                                   texture2d<float, access::write> diffuse, texture2d<float, access::write> specular,
                                   texture2d<float, access::write> motion, texture2d<float, access::write> reactive,
                                   uint2 pixel, uint id, bool reuse, thread ShadingResult &cached,
                                   float sunVisibility = 1.0f, uint sunIndex = ~0u, bool replaceLocal = false,
                                   float3 localDirect = float3(0), bool useGI = false, float3 irradiance = float3(0)) {
    if (pixel.x >= p.width || pixel.y >= p.height)
        return;
    if (id == VISIBILITY_BACKGROUND || visibilityCluster(id) >= 2u * p.candidateCapacity)
        return;
    const GPUMeshletCandidate candidate = candidateOf(id, p, a, b);
    const device GPUInstance &instance = instances[candidate.slot];
    if (instance.materialIndex >= p.materialCount)
        return;
    if (kResolveClass < 4u && materialClass(materials[instance.materialIndex], p) != kResolveClass)
        return;
    const GPUMeshlet m = meshlets[candidate.meshlet];
    const uint triangle = visibilityTriangle(id);
    if (triangle >= m.triangleCount)
        return;
    const GPUVertex v0 = vertices[meshletVertices[m.vertexOffset + triangles[m.triangleOffset + triangle * 3]]];
    const GPUVertex v1 = vertices[meshletVertices[m.vertexOffset + triangles[m.triangleOffset + triangle * 3 + 1]]];
    const GPUVertex v2 = vertices[meshletVertices[m.vertexOffset + triangles[m.triangleOffset + triangle * 3 + 2]]];
    const float4x4 model = visibilityMatrix(instance.modelMatrix);
    const float4x4 vp = visibilityMatrix(frame.viewProjection);
    const float4 w0 = model * float4(v0.px, v0.py, v0.pz, 1), w1 = model * float4(v1.px, v1.py, v1.pz, 1),
                 w2 = model * float4(v2.px, v2.py, v2.pz, 1);
    const float4 c0 = vp * w0, c1 = vp * w1, c2 = vp * w2;
    const auto bary =
        visibilityBarycentrics(c0.x, c0.y, c0.w, c1.x, c1.y, c1.w, c2.x, c2.y, c2.w, float(pixel.x) + 0.5f,
                               float(pixel.y) + 0.5f, float(p.width), float(p.height));
    if (!bary.valid)
        return;
    const float3 weights = float3(bary.value[0], bary.value[1], bary.value[2]);
    const float2 uv0 = float2(v0.u, v0.v), uv1 = float2(v1.u, v1.v), uv2 = float2(v2.u, v2.v);
    const float3x3 normalMatrix = float3x3(model[0].xyz, model[1].xyz, model[2].xyz);
    SurfaceInput surface;
    surface.worldPos = w0.xyz * weights.x + w1.xyz * weights.y + w2.xyz * weights.z;
    surface.normal =
        surfaceNormal(model, float3(v0.nx, v0.ny, v0.nz) * weights.x + float3(v1.nx, v1.ny, v1.nz) * weights.y +
                                 float3(v2.nx, v2.ny, v2.nz) * weights.z);
    surface.tangent =
        float4(normalMatrix * (float3(v0.tx, v0.ty, v0.tz) * weights.x + float3(v1.tx, v1.ty, v1.tz) * weights.y +
                               float3(v2.tx, v2.ty, v2.tz) * weights.z),
               (v0.tw * weights.x + v1.tw * weights.y + v2.tw * weights.z) *
                   ((instance.flags & INSTANCE_FLAG_MIRRORED) ? -1.0f : 1.0f));
    surface.uv = uv0 * weights.x + uv1 * weights.y + uv2 * weights.z;
    const float bias = exp2(p.mipBias);
    surface.uvDx = (uv0 * bary.dx[0] + uv1 * bary.dx[1] + uv2 * bary.dx[2]) * bias;
    surface.uvDy = (uv0 * bary.dy[0] + uv1 * bary.dy[1] + uv2 * bary.dy[2]) * bias;
    surface.materialIndex = instance.materialIndex;
    surface.mirrored = (instance.flags & INSTANCE_FLAG_MIRRORED) != 0;
    const float3 eye = float3(frame.cameraPosition[0], frame.cameraPosition[1], frame.cameraPosition[2]);
    surface.frontFacing = dot(cross(w1.xyz - w0.xyz, w2.xyz - w0.xyz), eye - surface.worldPos) > 0;
    const ShadingResult value = reuse ? cached : shadeSurface(surface, frame, materials, lights, textures, false,
                                                              sunVisibility, sunIndex, replaceLocal, localDirect, useGI, irradiance);
    cached = value;
    color.write(float4(p.debugMode == 1   ? value.normal * 0.5f + 0.5f
                       : p.debugMode == 2 ? value.baseColor
                                          : value.color,
                       1),
                pixel);
    normal.write(float4(value.normal, value.roughness), pixel);
    diffuse.write(float4(value.diffuseAlbedo + ((temporal.debugFlags & 2u) ? float3(0.25f, 0, 0) : float3(0)), 1),
                  pixel);
    specular.write(float4(value.specularAlbedo, 1), pixel);
    float2 velocity = 0;
    bool validHistory = false;
    if (temporal.historyValid) {
        const device GPUInstance &previous = previousInstances[candidate.slot];
        if (previous.generation == instance.generation && (previous.flags & INSTANCE_FLAG_VALID)) {
            validHistory = true;
            const float3 local = float3(v0.px, v0.py, v0.pz) * weights.x + float3(v1.px, v1.py, v1.pz) * weights.y +
                                 float3(v2.px, v2.py, v2.pz) * weights.z;
            const float4 oldWorld = temporalMatrix(previous.modelMatrix) * float4(local, 1);
            velocity = temporalMotion(temporalMatrix(temporal.currentViewProjection) * float4(surface.worldPos, 1),
                                      temporalMatrix(temporal.previousViewProjection) * oldWorld, temporal, true);
        }
    }
    motion.write(float4(velocity, 0, 0), pixel);
    cached.alpha = length(velocity);
    const float reactiveValue = !validHistory                                          ? 1.0f
                                : materials[instance.materialIndex].alphaCutoff > 0.0f ? 0.75f
                                                                                       : 0.0f;
    reactive.write(float4(reactiveValue), pixel);
}

kernel void visibility_resolve(
    constant FrameConstants &frame [[buffer(0)]], const device GPUVertex *vertices [[buffer(1)]],
    const device GPUInstance *instances [[buffer(2)]], const device GPUMaterial *materials [[buffer(3)]],
    const device GPULight *lights [[buffer(4)]], const device TextureHandle *textures [[buffer(5)]],
    const device GPUMeshlet *meshlets [[buffer(6)]], const device uint *meshletVertices [[buffer(7)]],
    const device uchar *triangles [[buffer(8)]], const device GPUMeshletCandidate *a [[buffer(9)]],
    const device GPUMeshletCandidate *b [[buffer(10)]], constant GPUVisibilityParams &p [[buffer(11)]],
    const device uint *bins [[buffer(12)]], const device GPUInstance *previousInstances [[buffer(14)]],
    constant GPUTemporalParams &temporal [[buffer(15)]], texture2d<uint, access::read> visibility [[texture(0)]],
    texture2d<float, access::write> color [[texture(1)]], texture2d<float, access::write> normal [[texture(2)]],
    texture2d<float, access::write> diffuse [[texture(3)]], texture2d<float, access::write> specular [[texture(4)]],
    texture2d<float, access::write> motion [[texture(5)]], texture2d<float, access::write> reactive [[texture(6)]],
    uint2 tile [[threadgroup_position_in_grid]], uint2 local [[thread_position_in_threadgroup]]) {
    uint index = tile.y * p.tilesX + tile.x;
    if (kResolveClass < 4u)
        index = bins[kResolveClass * p.tilesX * p.tilesY + tile.x];
    const uint2 pixel = uint2(index % p.tilesX, index / p.tilesX) * VISIBILITY_TILE + local;
    if (pixel.x >= p.width || pixel.y >= p.height)
        return;
    const uint id = visibility.read(pixel).x;
    ShadingResult cached{};
    resolveVisibilityPixel(frame, vertices, instances, materials, lights, textures, meshlets, meshletVertices,
                           triangles, a, b, p, previousInstances, temporal, color, normal, diffuse, specular, motion,
                           reactive, pixel, id, false, cached);
}
kernel void visibility_lit_resolve(
    constant FrameConstants &frame [[buffer(0)]], const device GPUVertex *vertices [[buffer(1)]],
    const device GPUInstance *instances [[buffer(2)]], const device GPUMaterial *materials [[buffer(3)]],
    const device GPULight *lights [[buffer(4)]], const device TextureHandle *textures [[buffer(5)]],
    const device GPUMeshlet *meshlets [[buffer(6)]], const device uint *meshletVertices [[buffer(7)]],
    const device uchar *triangles [[buffer(8)]], const device GPUMeshletCandidate *a [[buffer(9)]],
    const device GPUMeshletCandidate *b [[buffer(10)]], constant GPUVisibilityParams &p [[buffer(11)]],
    const device uint *bins [[buffer(12)]], const device GPUInstance *previousInstances [[buffer(14)]],
    constant GPUTemporalParams &temporal [[buffer(15)]], texture2d<uint, access::read> visibility [[texture(0)]],
    texture2d<float, access::write> color [[texture(1)]], texture2d<float, access::write> normal [[texture(2)]],
    texture2d<float, access::write> diffuse [[texture(3)]], texture2d<float, access::write> specular [[texture(4)]],
    texture2d<float, access::write> motion [[texture(5)]], texture2d<float, access::write> reactive [[texture(6)]],
    constant GPUResolveLightingParams& lighting [[buffer(17)]],
    texture2d<float, access::read> sun [[texture(7)]],
    texture2d<float, access::read> localDirect [[texture(8)]],
    texture2d<float, access::read> irradiance [[texture(9)]],
    uint2 tile [[threadgroup_position_in_grid]], uint2 local [[thread_position_in_threadgroup]]) {
    uint index = tile.y * p.tilesX + tile.x;
    if (kResolveClass < 4u)
        index = bins[kResolveClass * p.tilesX * p.tilesY + tile.x];
    const uint2 pixel = uint2(index % p.tilesX, index / p.tilesX) * VISIBILITY_TILE + local;
    if (pixel.x >= p.width || pixel.y >= p.height)
        return;
    const uint id = visibility.read(pixel).x;
    ShadingResult cached{};
    resolveVisibilityPixel(frame, vertices, instances, materials, lights, textures, meshlets, meshletVertices,
                           triangles, a, b, p, previousInstances, temporal, color, normal, diffuse, specular, motion,
                           reactive, pixel, id, false, cached,
                           (lighting.flags & 1u) ? sun.read(pixel).x : 1.0f, lighting.sunIndex,
                           (lighting.flags & 2u) != 0, (lighting.flags & 2u) ? localDirect.read(pixel).xyz : float3(0),
                           (lighting.flags & 4u) != 0, (lighting.flags & 4u) ? irradiance.read(pixel).xyz : float3(0));
}
kernel void visibility_adaptive(
    constant FrameConstants &frame [[buffer(0)]], const device GPUVertex *vertices [[buffer(1)]],
    const device GPUInstance *instances [[buffer(2)]], const device GPUMaterial *materials [[buffer(3)]],
    const device GPULight *lights [[buffer(4)]], const device TextureHandle *textures [[buffer(5)]],
    const device GPUMeshlet *meshlets [[buffer(6)]], const device uint *meshletVertices [[buffer(7)]],
    const device uchar *triangles [[buffer(8)]], const device GPUMeshletCandidate *a [[buffer(9)]],
    const device GPUMeshletCandidate *b [[buffer(10)]], constant GPUVisibilityParams &p [[buffer(11)]],
    const device uint *bins [[buffer(12)]], const device GPUInstance *previousInstances [[buffer(14)]],
    constant GPUTemporalParams &temporal [[buffer(15)]], const device GPUShadingHistory *history [[buffer(16)]],
    device atomic_uint *statistics [[buffer(13)]], texture2d<uint, access::read> visibility [[texture(0)]],
    texture2d<float, access::write> color [[texture(1)]], texture2d<float, access::write> normal [[texture(2)]],
    texture2d<float, access::write> diffuse [[texture(3)]], texture2d<float, access::write> specular [[texture(4)]],
    texture2d<float, access::write> motion [[texture(5)]], texture2d<float, access::write> reactive [[texture(6)]],
    uint2 tile [[threadgroup_position_in_grid]], uint2 local [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint2 origin = tile * VISIBILITY_TILE + local * 2u;
    const bool inside = all(origin < uint2(p.width, p.height));
    const uint firstID = inside ? visibility.read(origin).x : 0u;
    ShadingResult cached{};
    resolveVisibilityPixel(frame, vertices, instances, materials, lights, textures, meshlets, meshletVertices,
                           triangles, a, b, p, previousInstances, temporal, color, normal, diffuse, specular, motion,
                           reactive, origin, firstID, false, cached);
    bool coarse = (p.pad2 & 2u) && !(temporal.debugFlags & 4u) && temporal.historyValid && firstID != 0u &&
                  all(origin + 1u < uint2(p.width, p.height)) && cached.alpha < 0.25f;
    if (coarse) {
        const auto candidate = candidateOf(firstID, p, a, b);
        const auto instance = instances[candidate.slot];
        const device auto &material = materials[instance.materialIndex];
        const uint primitive = meshlets[candidate.meshlet].triangleOffset + visibilityTriangle(firstID) * 3u;
        coarse = material.roughness >= 0.75f && material.metallic <= 0.05f && material.alphaCutoff == 0 &&
                 materialClass(material, p) == 0u;
        float lo = INFINITY, hi = 0;
        for (uint y = 0; y < 2; ++y)
            for (uint x = 0; x < 2; ++x) {
                const uint2 pixel = origin + uint2(x, y);
                const auto prior = history[pixel.y * p.outputWidth + pixel.x];
                coarse &= visibility.read(pixel).x == firstID && prior.generation == instance.generation &&
                          prior.primitive == primitive && prior.roughness >= 0.75f && isfinite(prior.luminance);
                lo = min(lo, prior.luminance);
                hi = max(hi, prior.luminance);
            }
        coarse &= (hi - lo) <= max(0.002f, hi * 0.02f);
    }
    uint shaded = inside && firstID != 0;
    for (uint i = 1; i < 4; ++i) {
        const uint2 pixel = origin + uint2(i & 1u, i >> 1u);
        const bool valid = all(pixel < uint2(p.width, p.height));
        const uint id = valid ? visibility.read(pixel).x : 0u;
        resolveVisibilityPixel(frame, vertices, instances, materials, lights, textures, meshlets, meshletVertices,
                               triangles, a, b, p, previousInstances, temporal, color, normal, diffuse, specular,
                               motion, reactive, pixel, id, coarse, cached);
        shaded += valid && id != 0 && !coarse;
    }
    const uint total = simd_sum(shaded), reused = simd_sum(coarse ? 3u : 0u);
    if (lane == 0) {
        atomic_fetch_add_explicit(&statistics[12], total, memory_order_relaxed);
        atomic_fetch_add_explicit(&statistics[13], reused, memory_order_relaxed);
    }
}

kernel void visibility_history(
    const device GPUInstance *instances [[buffer(2)]], const device GPUMeshlet *meshlets [[buffer(6)]],
    const device GPUMeshletCandidate *a [[buffer(9)]], const device GPUMeshletCandidate *b [[buffer(10)]],
    constant GPUVisibilityParams &p [[buffer(11)]], device GPUShadingHistory *history [[buffer(16)]],
    texture2d<uint, access::read> visibility [[texture(0)]], texture2d<float, access::read> color [[texture(1)]],
    texture2d<float, access::read> normal [[texture(2)]], uint2 pixel [[thread_position_in_grid]]) {
    if (any(pixel >= uint2(p.outputWidth, p.outputHeight)))
        return;
    GPUShadingHistory entry{};
    const uint id = visibility.read(pixel).x;
    if (all(pixel < uint2(p.width, p.height)) && id != 0 && visibilityCluster(id) < 2u * p.candidateCapacity) {
        const auto candidate = candidateOf(id, p, a, b);
        entry.generation = instances[candidate.slot].generation;
        entry.primitive = meshlets[candidate.meshlet].triangleOffset + visibilityTriangle(id) * 3u;
        entry.luminance = dot(color.read(pixel).rgb, float3(0.2126f, 0.7152f, 0.0722f));
        entry.roughness = normal.read(pixel).w;
    }
    history[pixel.y * p.outputWidth + pixel.x] = entry;
}
struct VisibilityTile {
    uint id [[color(0)]];
};
kernel void visibility_tile(
    constant FrameConstants &frame [[buffer(0)]], const device GPUVertex *vertices [[buffer(1)]],
    const device GPUInstance *instances [[buffer(2)]], const device GPUMaterial *materials [[buffer(3)]],
    const device GPULight *lights [[buffer(4)]], const device TextureHandle *textures [[buffer(5)]],
    const device GPUMeshlet *meshlets [[buffer(6)]], const device uint *meshletVertices [[buffer(7)]],
    const device uchar *triangles [[buffer(8)]], const device GPUMeshletCandidate *a [[buffer(9)]],
    const device GPUMeshletCandidate *b [[buffer(10)]], constant GPUVisibilityParams &p [[buffer(11)]],
    const device uint *bins [[buffer(12)]], const device GPUInstance *previousInstances [[buffer(14)]],
    constant GPUTemporalParams &temporal [[buffer(15)]], imageblock<VisibilityTile, imageblock_layout_implicit> image,
    texture2d<float, access::write> color [[texture(1)]], texture2d<float, access::write> normal [[texture(2)]],
    texture2d<float, access::write> diffuse [[texture(3)]], texture2d<float, access::write> specular [[texture(4)]],
    texture2d<float, access::write> motion [[texture(5)]], texture2d<float, access::write> reactive [[texture(6)]],
    uint2 tile [[threadgroup_position_in_grid]], uint2 local [[thread_position_in_threadgroup]]) {
    const uint2 pixel = tile * VISIBILITY_TILE + local;
    if (pixel.x >= p.outputWidth || pixel.y >= p.outputHeight)
        return;
    color.write(float4(0.02f, 0.025f, 0.035f, 1), pixel);
    normal.write(float4(0), pixel);
    diffuse.write(float4(0), pixel);
    specular.write(float4(0), pixel);
    motion.write(float4(0), pixel);
    reactive.write(float4(1), pixel);
    if (pixel.x >= p.width || pixel.y >= p.height)
        return;
    const uint id = image.read(ushort2(local)).id;
    ShadingResult cached{};
    resolveVisibilityPixel(frame, vertices, instances, materials, lights, textures, meshlets, meshletVertices,
                           triangles, a, b, p, previousInstances, temporal, color, normal, diffuse, specular, motion,
                           reactive, pixel, id, false, cached);
}

struct PresentVertex {
    float4 position [[position]];
    float2 uv;
};
vertex PresentVertex visibility_present_vs(uint id [[vertex_id]]) {
    const float2 uv = float2((id << 1) & 2, id & 2);
    return {float4(uv.x * 2 - 1, 1 - uv.y * 2, 0, 1), uv};
}
fragment half4 visibility_present_fs(PresentVertex in [[stage_in]], constant GPUVisibilityParams &p [[buffer(11)]],
                                     texture2d<float> hdr [[texture(1)]], texture2d<float> normal [[texture(2)]],
                                     texture2d<float> diffuse [[texture(3)]]) {
    constexpr sampler sampling(filter::nearest, address::clamp_to_edge);
    const float4 value = hdr.sample(sampling, in.uv);
    if (dot(normal.sample(sampling, in.uv).xyz, normal.sample(sampling, in.uv).xyz) == 0)
        return half4(0.02h, 0.025h, 0.035h, 1.0h);
#ifdef PHOSPHOR_HOT_RELOAD_PROBE
    return half4(1, 0, 1, 1);
#endif
    if (p.debugMode == 1)
        return half4(half3(normal.sample(sampling, in.uv).xyz * 0.5f + 0.5f), 1.0h);
    if (p.debugMode == 2)
        return half4(half3(value.rgb), 1.0h);
    return half4(half3(tonemapACES(value.rgb * p.exposure)), 1.0h);
}

kernel void material_lighting_guides(
    constant FrameConstants &frame [[buffer(0)]], const device GPUVertex *vertices [[buffer(1)]],
    const device GPUInstance *instances [[buffer(2)]], const device GPUMaterial *materials [[buffer(3)]],
    const device GPULight *lights [[buffer(4)]], const device TextureHandle *textures [[buffer(5)]],
    const device GPUMeshlet *meshlets [[buffer(6)]], const device uint *meshletVertices [[buffer(7)]],
    const device uchar *triangles [[buffer(8)]], const device GPUMeshletCandidate *a [[buffer(9)]],
    const device GPUMeshletCandidate *b [[buffer(10)]], constant GPUVisibilityParams &p [[buffer(11)]],
    const device GPUInstance *previousInstances [[buffer(14)]], constant GPUTemporalParams &temporal [[buffer(15)]],
    device GPUDISurface* output [[buffer(16)]], const device GPUShadowSurface* geometry [[buffer(17)]],
    constant GPUDIParams& di [[buffer(18)]], texture2d<uint,access::read> visibility [[texture(0)]],
    texture2d<float,access::write> motion [[texture(7)]],
    texture2d<float,access::write> fallbackShading [[texture(8)]],
    texture2d<float,access::write> fallbackAlbedo [[texture(9)]],uint2 pixel [[thread_position_in_grid]]) {
    if(any(pixel>=uint2(p.width,p.height)))return;
    const uint index=pixel.y*p.width+pixel.x;
    output[index]=GPUDISurface{};motion.write(float4(0),pixel);
    fallbackShading.write(float4(0),pixel);fallbackAlbedo.write(float4(0),pixel);
    const uint id=visibility.read(pixel).x;
    if(id==VISIBILITY_BACKGROUND || visibilityCluster(id)>=2u*p.candidateCapacity || !geometry[index].valid)return;
    const GPUMeshletCandidate candidate = candidateOf(id, p, a, b);
    const device GPUInstance &instance = instances[candidate.slot];
    if (instance.materialIndex >= p.materialCount)
        return;
    if (kResolveClass < 4u && materialClass(materials[instance.materialIndex], p) != kResolveClass)
        return;
    const GPUMeshlet m = meshlets[candidate.meshlet];
    const uint triangle = visibilityTriangle(id);
    if (triangle >= m.triangleCount)
        return;
    const GPUVertex v0 = vertices[meshletVertices[m.vertexOffset + triangles[m.triangleOffset + triangle * 3]]];
    const GPUVertex v1 = vertices[meshletVertices[m.vertexOffset + triangles[m.triangleOffset + triangle * 3 + 1]]];
    const GPUVertex v2 = vertices[meshletVertices[m.vertexOffset + triangles[m.triangleOffset + triangle * 3 + 2]]];
    const float4x4 model = visibilityMatrix(instance.modelMatrix);
    const float4x4 vp = visibilityMatrix(frame.viewProjection);
    const float4 w0 = model * float4(v0.px, v0.py, v0.pz, 1), w1 = model * float4(v1.px, v1.py, v1.pz, 1),
                 w2 = model * float4(v2.px, v2.py, v2.pz, 1);
    const float4 c0 = vp * w0, c1 = vp * w1, c2 = vp * w2;
    const auto bary =
        visibilityBarycentrics(c0.x, c0.y, c0.w, c1.x, c1.y, c1.w, c2.x, c2.y, c2.w, float(pixel.x) + 0.5f,
                               float(pixel.y) + 0.5f, float(p.width), float(p.height));
    if (!bary.valid)
        return;
    const float3 weights = float3(bary.value[0], bary.value[1], bary.value[2]);
    const float2 uv0 = float2(v0.u, v0.v), uv1 = float2(v1.u, v1.v), uv2 = float2(v2.u, v2.v);
    const float3x3 normalMatrix = float3x3(model[0].xyz, model[1].xyz, model[2].xyz);
    SurfaceInput surface;
    surface.worldPos = w0.xyz * weights.x + w1.xyz * weights.y + w2.xyz * weights.z;
    surface.normal =
        surfaceNormal(model, float3(v0.nx, v0.ny, v0.nz) * weights.x + float3(v1.nx, v1.ny, v1.nz) * weights.y +
                                 float3(v2.nx, v2.ny, v2.nz) * weights.z);
    surface.tangent =
        float4(normalMatrix * (float3(v0.tx, v0.ty, v0.tz) * weights.x + float3(v1.tx, v1.ty, v1.tz) * weights.y +
                               float3(v2.tx, v2.ty, v2.tz) * weights.z),
               (v0.tw * weights.x + v1.tw * weights.y + v2.tw * weights.z) *
                   ((instance.flags & INSTANCE_FLAG_MIRRORED) ? -1.0f : 1.0f));
    surface.uv = uv0 * weights.x + uv1 * weights.y + uv2 * weights.z;
    const float bias = exp2(p.mipBias);
    surface.uvDx = (uv0 * bary.dx[0] + uv1 * bary.dx[1] + uv2 * bary.dx[2]) * bias;
    surface.uvDy = (uv0 * bary.dy[0] + uv1 * bary.dy[1] + uv2 * bary.dy[2]) * bias;
    surface.materialIndex = instance.materialIndex;
    surface.mirrored = (instance.flags & INSTANCE_FLAG_MIRRORED) != 0;
    const float3 eye = float3(frame.cameraPosition[0], frame.cameraPosition[1], frame.cameraPosition[2]);
    surface.frontFacing = dot(cross(w1.xyz - w0.xyz, w2.xyz - w0.xyz), eye - surface.worldPos) > 0;
    const ShadingResult value=shadeSurface(surface,frame,materials,lights,textures,false);
    GPUDISurface guide{};const GPUShadowSurface geometric=geometry[index];
    for(uint c=0;c<3;++c) {
        guide.position[c]=geometric.position[c];guide.geometricNormal[c]=geometric.geometricNormal[c];
        guide.shadingNormal[c]=value.normal[c];guide.albedo[c]=value.baseColor[c];
        guide.viewDirection[c]=normalize(eye-surface.worldPos)[c];
    }
    guide.depth=geometric.viewDepth;guide.roughness=value.roughness;guide.metallic=value.metallic;
    guide.materialRevision=di.historyEpoch;guide.instanceSlot=candidate.slot;guide.instanceGeneration=instance.generation;guide.valid=1;
    output[index]=guide;
    const float3 objectPoint=float3(v0.px,v0.py,v0.pz)*weights.x+float3(v1.px,v1.py,v1.pz)*weights.y+float3(v2.px,v2.py,v2.pz)*weights.z;
    const auto old=previousInstances[candidate.slot];
    const bool valid=temporal.historyValid && old.generation==instance.generation && (old.flags&INSTANCE_FLAG_VALID);
    const float4 current=temporalMatrix(temporal.currentViewProjection)*float4(surface.worldPos,1);
    const float4 previous=valid?temporalMatrix(temporal.previousViewProjection)*visibilityMatrix(old.modelMatrix)*float4(objectPoint,1):current;
    motion.write(float4(temporalMotion(current,previous,temporal,valid),0,0),pixel);
}
