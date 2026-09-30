// B-21: validation traces after an acceleration-structure build / refit /
// compaction (a few hundred rays, nearest hit checked against the CPU).
#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace raytracing;

// rays: 2 x float4 per ray: (origin, tmin) and (direction, tmax).
// out: float4 per ray: (t or -1, primitive_id, instance_id, 0).
kernel void trace_blas(primitive_acceleration_structure as [[buffer(0)]], device const float4* rays [[buffer(1)]],
                       device float4* out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    const float4 o = rays[2 * i], d = rays[2 * i + 1];
    ray r(o.xyz, d.xyz, o.w, d.w);
    intersector<triangle_data> isect;
    auto h = isect.intersect(r, as);
    out[i] = h.type == intersection_type::triangle ? float4(h.distance, float(h.primitive_id), 0, 0) : float4(-1, 0, 0, 0);
}

kernel void trace_tlas(instance_acceleration_structure as [[buffer(0)]], device const float4* rays [[buffer(1)]],
                       device float4* out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    const float4 o = rays[2 * i], d = rays[2 * i + 1];
    ray r(o.xyz, d.xyz, o.w, d.w);
    intersector<triangle_data, instancing> isect;
    auto h = isect.intersect(r, as);
    out[i] = h.type == intersection_type::triangle
                 ? float4(h.distance, float(h.primitive_id), float(h.instance_id), 0)
                 : float4(-1, 0, 0, 0);
}
