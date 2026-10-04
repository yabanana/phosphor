// F9-S4: any-hit shadow query against a TLAS, optionally restricted to the
// receiver's own instance (proxy self-shadow / acne measurement).
// Output per ray: the instance id of the first candidate triangle found, or
// 0xFFFFFFFF if the ray is unoccluded.  The geometry is forced non-opaque so
// the query reports every candidate and the kernel decides (the instance
// filter has to run in the kernel).
#include "f9_rt.h"

struct PxParams {
    uint count;
    uint mode; // 0: any instance, 1: only the instance named in owner[tid]
    uint pad0, pad1;
};

kernel void px_shadow(instance_acceleration_structure as [[buffer(0)]], device const RayIn* rays [[buffer(1)]],
                      device uint* out [[buffer(2)]], constant PxParams& p [[buffer(3)]],
                      device const uint* owner [[buffer(4)]], uint tid [[thread_position_in_grid]]) {
    if (tid >= p.count) return;
    const RayIn r = rays[tid];
    intersection_params params;
    params.force_opacity(forced_opacity::non_opaque);
    params.assume_geometry_type(geometry_type::triangle);
    intersection_query<triangle_data, instancing> q(ray(float3(r.o), float3(r.d), r.tmin, r.tmax), as, params);
    const uint own = owner[tid];
    uint res = 0xFFFFFFFFu;
    uint guard = 0u;
    while (q.next() && guard++ < 1000000u) {
        if (q.get_candidate_intersection_type() == intersection_type::triangle) {
            const uint inst = q.get_candidate_instance_id();
            if (p.mode == 0u || inst == own) {
                res = inst;
                break;
            }
        }
    }
    out[tid] = res;
}
