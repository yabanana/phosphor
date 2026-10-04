#pragma once
// F9 spikes: ray/hit layout shared with bench/f9_spike/f9_common.h (GpuRay,
// GpuHit, 32 B each).
#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace raytracing;

struct RayIn {
    packed_float3 o;
    float tmin;
    packed_float3 d;
    float tmax;
};

struct HitOut {
    float t;        // < 0: miss
    uint instance;
    uint primitive;
    uint geometry;
    float u, v;
    uint front;
    uint extra;
};

inline HitOut missHit() {
    HitOut h;
    h.t = -1.0f; h.instance = 0xFFFFFFFFu; h.primitive = 0xFFFFFFFFu; h.geometry = 0xFFFFFFFFu;
    h.u = 0.0f; h.v = 0.0f; h.front = 0u; h.extra = 0u;
    return h;
}

template <typename R>
inline HitOut toHitBlas(R r) {
    if (r.type != intersection_type::triangle) return missHit();
    HitOut h;
    h.t = r.distance;
    h.instance = 0u;
    h.primitive = r.primitive_id;
    h.geometry = r.geometry_id;
    h.u = r.triangle_barycentric_coord.x;
    h.v = r.triangle_barycentric_coord.y;
    h.front = r.triangle_front_facing ? 1u : 0u;
    h.extra = 0u;
    return h;
}

template <typename R>
inline HitOut toHit(R r) {
    if (r.type != intersection_type::triangle) return missHit();
    HitOut h;
    h.t = r.distance;
    h.instance = r.instance_id;
    h.primitive = r.primitive_id;
    h.geometry = r.geometry_id;
    h.u = r.triangle_barycentric_coord.x;
    h.v = r.triangle_barycentric_coord.y;
    h.front = r.triangle_front_facing ? 1u : 0u;
    h.extra = 0u;
    return h;
}
