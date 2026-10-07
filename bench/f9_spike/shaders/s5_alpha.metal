// F9-S5: alpha test in ray tracing with intersection functions.
//
// Raster rule mirrored (shaders/meshlet.metal visibility_alpha_fs, forward.metal):
//     alpha = material.baseColor.a * float(half(baseColorTex.sample(...).a))
//     discard when alpha < material.alphaCutoff
// i.e. a hit is ACCEPTED when alpha >= cutoff.  The RT function samples at an
// explicit LOD (no derivatives exist in an intersection function) with a
// linear filter and repeat addressing.
//
// Resources reach the functions through the intersection function table
// (MTL::IntersectionFunctionTable::setBuffer, indices below), the Metal 3
// mechanism; the instance's userID is its index in the scene (instance record
// -> mesh offsets + material).
#include "f9_rt.h"
#include "renderer/gpu_types.h"

using namespace phosphor;

struct TextureHandle {
    texture2d<float> tex;
};

struct InstRec {
    uint vertexOffset;
    uint indexOffset;
    uint material;
    uint pad;
};

struct AlphaParams {
    float lod;   // explicit mip level (integer values are exact)
    uint pad0, pad1, pad2;
};

struct PrimUV { // per-triangle primitive data (alternative to the vertex fetch)
    float2 uv0, uv1, uv2;
};

struct TraceParams {
    uint count;
    uint mask;
    uint pad0, pad1;
};

// Counters of the debug variant: [0] calls, [1] calls on materials without a
// cutoff (must be 0 when the geometry is flagged opaque), [2 + material] calls per material.
constexpr sampler kAlphaSampler(filter::linear, mip_filter::nearest, address::repeat);

inline bool alphaPass(const device GPUMaterial* mats, const device TextureHandle* texs, uint material, float2 uv,
                      float lod) {
    const device GPUMaterial& m = mats[material];
    if (m.alphaCutoff <= 0.0f) return true; // opaque material reached through a non-opaque geometry
    float a = m.baseColor[3];
    if (m.baseColorTex != INVALID_TEXTURE_INDEX)
        a *= float(half(texs[m.baseColorTex].tex.sample(kAlphaSampler, uv, level(lod)).a));
    return a >= m.alphaCutoff;
}

inline float2 uvFromVertices(const device GPUVertex* verts, const device uint* idx, InstRec r, uint prim, float2 b) {
    const device GPUVertex& v0 = verts[r.vertexOffset + idx[r.indexOffset + 3 * prim]];
    const device GPUVertex& v1 = verts[r.vertexOffset + idx[r.indexOffset + 3 * prim + 1]];
    const device GPUVertex& v2 = verts[r.vertexOffset + idx[r.indexOffset + 3 * prim + 2]];
    return float2(v0.u, v0.v) * (1.0f - b.x - b.y) + float2(v1.u, v1.v) * b.x + float2(v2.u, v2.v) * b.y;
}

inline void count(device atomic_uint* c, const device GPUMaterial* mats, uint material) {
    atomic_fetch_add_explicit(&c[0], 1u, memory_order_relaxed);
    if (mats[material].alphaCutoff <= 0.0f) atomic_fetch_add_explicit(&c[1], 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(&c[2 + material], 1u, memory_order_relaxed);
}

// Buffer layout of every function (IFT setBuffer indices).
#define ISECT_ARGS                                                                                                    \
    uint prim [[primitive_id]], uint inst [[user_instance_id]], float2 bary [[barycentric_coord]],                     \
        const device GPUMaterial* mats [[buffer(0)]], const device TextureHandle* texs [[buffer(1)]],                  \
        const device GPUVertex* verts [[buffer(2)]], const device uint* idx [[buffer(3)]],                             \
        const device InstRec* recs [[buffer(4)]], constant AlphaParams& prm [[buffer(5)]],                             \
        device atomic_uint* counters [[buffer(6)]]
#define ISECT_ATTR [[intersection(triangle, triangle_data, instancing)]]

// (A) generic: material lookup inside the function (the portable path).
template <bool kDebug>
inline bool genericBody(uint prim, uint inst, float2 bary, const device GPUMaterial* mats, const device TextureHandle* texs,
                        const device GPUVertex* verts, const device uint* idx, const device InstRec* recs,
                        constant AlphaParams& prm, device atomic_uint* counters) {
    const InstRec r = recs[inst];
    if (kDebug) count(counters, mats, r.material);
    if (mats[r.material].alphaCutoff <= 0.0f) return true;
    return alphaPass(mats, texs, r.material, uvFromVertices(verts, idx, r, prim, bary), prm.lod);
}
ISECT_ATTR bool alpha_generic(ISECT_ARGS) { return genericBody<false>(prim, inst, bary, mats, texs, verts, idx, recs, prm, counters); }
ISECT_ATTR bool alpha_generic_dbg(ISECT_ARGS) { return genericBody<true>(prim, inst, bary, mats, texs, verts, idx, recs, prm, counters); }

// (A, primitive data) UVs from the per-triangle primitive data instead of the vertex/index fetch.
template <bool kDebug>
inline bool pdBody(uint inst, float2 bary, const device PrimUV* pd, const device GPUMaterial* mats,
                   const device TextureHandle* texs, const device InstRec* recs, constant AlphaParams& prm,
                   device atomic_uint* counters) {
    const InstRec r = recs[inst];
    if (kDebug) count(counters, mats, r.material);
    if (mats[r.material].alphaCutoff <= 0.0f) return true;
    const float2 uv = pd->uv0 * (1.0f - bary.x - bary.y) + pd->uv1 * bary.x + pd->uv2 * bary.y;
    return alphaPass(mats, texs, r.material, uv, prm.lod);
}
[[intersection(triangle, triangle_data, instancing)]] bool alpha_pd(
    uint inst [[user_instance_id]], float2 bary [[barycentric_coord]], const device PrimUV* pd [[primitive_data]],
    const device GPUMaterial* mats [[buffer(0)]], const device TextureHandle* texs [[buffer(1)]],
    const device InstRec* recs [[buffer(4)]], constant AlphaParams& prm [[buffer(5)]],
    device atomic_uint* counters [[buffer(6)]]) {
    return pdBody<false>(inst, bary, pd, mats, texs, recs, prm, counters);
}
[[intersection(triangle, triangle_data, instancing)]] bool alpha_pd_dbg(
    uint inst [[user_instance_id]], float2 bary [[barycentric_coord]], const device PrimUV* pd [[primitive_data]],
    const device GPUMaterial* mats [[buffer(0)]], const device TextureHandle* texs [[buffer(1)]],
    const device InstRec* recs [[buffer(4)]], constant AlphaParams& prm [[buffer(5)]],
    device atomic_uint* counters [[buffer(6)]]) {
    return pdBody<true>(inst, bary, pd, mats, texs, recs, prm, counters);
}

// (B) specialised per material slot: textured cutoff / constant alpha.  The
// slot is known from the table index, so there is no opaque or no-texture branch.
template <bool kDebug>
inline bool texBody(uint prim, uint inst, float2 bary, const device GPUMaterial* mats, const device TextureHandle* texs,
                    const device GPUVertex* verts, const device uint* idx, const device InstRec* recs,
                    constant AlphaParams& prm, device atomic_uint* counters) {
    const InstRec r = recs[inst];
    if (kDebug) count(counters, mats, r.material);
    const device GPUMaterial& m = mats[r.material];
    const float2 uv = uvFromVertices(verts, idx, r, prim, bary);
    const float a = m.baseColor[3] * float(half(texs[m.baseColorTex].tex.sample(kAlphaSampler, uv, level(prm.lod)).a));
    return a >= m.alphaCutoff;
}
template <bool kDebug>
inline bool constBody(uint inst, const device GPUMaterial* mats, const device InstRec* recs, device atomic_uint* counters) {
    const InstRec r = recs[inst];
    if (kDebug) count(counters, mats, r.material);
    return mats[r.material].baseColor[3] >= mats[r.material].alphaCutoff;
}
ISECT_ATTR bool alpha_tex(ISECT_ARGS) { return texBody<false>(prim, inst, bary, mats, texs, verts, idx, recs, prm, counters); }
ISECT_ATTR bool alpha_tex_dbg(ISECT_ARGS) { return texBody<true>(prim, inst, bary, mats, texs, verts, idx, recs, prm, counters); }
ISECT_ATTR bool alpha_const(ISECT_ARGS) { return constBody<false>(inst, mats, recs, counters); }
ISECT_ATTR bool alpha_const_dbg(ISECT_ARGS) { return constBody<true>(inst, mats, recs, counters); }

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

kernel void primary_ift(instance_acceleration_structure as [[buffer(0)]], device const RayIn* rays [[buffer(1)]],
                        device HitOut* hits [[buffer(2)]], constant TraceParams& p [[buffer(3)]],
                        intersection_function_table<triangle_data, instancing> ift [[buffer(4)]],
                        uint tid [[thread_position_in_grid]]) {
    if (tid >= p.count) return;
    const RayIn r = rays[tid];
    intersector<triangle_data, instancing> isect;
    isect.assume_geometry_type(geometry_type::triangle);
    hits[tid] = toHit(isect.intersect(ray(float3(r.o), float3(r.d), r.tmin, r.tmax), as, p.mask, ift));
}

kernel void shadow_ift(instance_acceleration_structure as [[buffer(0)]], device const RayIn* rays [[buffer(1)]],
                       device HitOut* hits [[buffer(2)]], constant TraceParams& p [[buffer(3)]],
                       intersection_function_table<triangle_data, instancing> ift [[buffer(4)]],
                       uint tid [[thread_position_in_grid]]) {
    if (tid >= p.count) return;
    const RayIn r = rays[tid];
    intersector<triangle_data, instancing> isect;
    isect.assume_geometry_type(geometry_type::triangle);
    isect.accept_any_intersection(true);
    hits[tid] = toHit(isect.intersect(ray(float3(r.o), float3(r.d), r.tmin, r.tmax), as, p.mask, ift));
}

kernel void primary_noift(instance_acceleration_structure as [[buffer(0)]], device const RayIn* rays [[buffer(1)]],
                          device HitOut* hits [[buffer(2)]], constant TraceParams& p [[buffer(3)]],
                          uint tid [[thread_position_in_grid]]) {
    if (tid >= p.count) return;
    const RayIn r = rays[tid];
    intersector<triangle_data, instancing> isect;
    isect.assume_geometry_type(geometry_type::triangle);
    hits[tid] = toHit(isect.intersect(ray(float3(r.o), float3(r.d), r.tmin, r.tmax), as, p.mask));
}

kernel void shadow_noift(instance_acceleration_structure as [[buffer(0)]], device const RayIn* rays [[buffer(1)]],
                         device HitOut* hits [[buffer(2)]], constant TraceParams& p [[buffer(3)]],
                         uint tid [[thread_position_in_grid]]) {
    if (tid >= p.count) return;
    const RayIn r = rays[tid];
    intersector<triangle_data, instancing> isect;
    isect.assume_geometry_type(geometry_type::triangle);
    isect.accept_any_intersection(true);
    hits[tid] = toHit(isect.intersect(ray(float3(r.o), float3(r.d), r.tmin, r.tmax), as, p.mask));
}
