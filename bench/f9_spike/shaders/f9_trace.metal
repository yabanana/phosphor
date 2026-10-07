// F9 spikes: generic nearest-hit trace through a TLAS (intersector,
// triangle_data + instancing), used to validate builds against the CPU.
#include "f9_rt.h"

struct TraceParams {
    uint count;
    uint mask;
    uint pad0, pad1;
};

kernel void trace_nearest(instance_acceleration_structure as [[buffer(0)]], device const RayIn* rays [[buffer(1)]],
                          device HitOut* hits [[buffer(2)]], constant TraceParams& p [[buffer(3)]],
                          uint tid [[thread_position_in_grid]]) {
    if (tid >= p.count) return;
    const RayIn r = rays[tid];
    intersector<triangle_data, instancing> isect;
    isect.assume_geometry_type(geometry_type::triangle);
    hits[tid] = toHit(isect.intersect(ray(float3(r.o), float3(r.d), r.tmin, r.tmax), as, p.mask));
}

kernel void trace_nearest_blas(primitive_acceleration_structure as [[buffer(0)]], device const RayIn* rays [[buffer(1)]],
                               device HitOut* hits [[buffer(2)]], constant TraceParams& p [[buffer(3)]],
                               uint tid [[thread_position_in_grid]]) {
    if (tid >= p.count) return;
    const RayIn r = rays[tid];
    intersector<triangle_data> isect;
    isect.assume_geometry_type(geometry_type::triangle);
    hits[tid] = toHitBlas(isect.intersect(ray(float3(r.o), float3(r.d), r.tmin, r.tmax), as));
}
