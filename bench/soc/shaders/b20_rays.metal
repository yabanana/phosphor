// B-20: ray throughput.  A TLAS over a ~1M-triangle displaced terrain plus
// scattered small instances is traced with primary coherent rays (one per
// pixel of a 2048^2 camera) or incoherent rays (random origins and
// directions), with the `intersector` API or `intersection_query`, opaque
// geometry or an alpha-test-like rejection (intersection function table for
// the intersector, the same test inline in the query loop).
//
// Output per ray: uint2 (float bits of t or 0xFFFFFFFF for a miss, instance << 20 | primitive).
// Every dumpStride-th ray is dumped (origin, direction) so the CPU can
// re-trace exactly the rays the GPU generated.
#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace raytracing;

struct RtParams {
    float4 camPos, camFwd, camRight, camUp; // right/up already scaled by tan(fov/2) (and aspect)
    uint width, height;                     // pixel grid of one pass (2048 x 2048)
    uint kind;                              // 0 coherent, 1 incoherent
    uint seed;
    uint dumpStride;
    uint pad0, pad1, pad2;
    float4 bounds;                          // incoherent: x0, z0, size, height
};

inline uint hash(uint x) {
    x ^= x >> 16; x *= 0x7FEB352Du; x ^= x >> 15; x *= 0x846CA68Bu; x ^= x >> 16;
    return x;
}
inline float rnd(uint a, uint b) { return float(hash(a * 0x9E3779B1u ^ hash(b + 0x85EBCA6Bu)) >> 8) * (1.0f / 16777216.0f); }

// Alpha-test-like rule shared with the CPU: 25% of the primitives are holes.
inline bool alphaPass(uint prim) { return ((prim * 2654435761u) >> 30) != 0u; }

inline void makeRay(constant RtParams& p, uint2 tid, thread float3& o, thread float3& d) {
    const uint pass = tid.y / p.height;
    const uint y = tid.y % p.height;
    if (p.kind == 0) {
        const float jit = 0.5f + 0.37f * float(pass);
        const float u = (float(tid.x) + jit) / float(p.width) * 2.0f - 1.0f;
        const float v = 1.0f - (float(y) + jit) / float(p.height) * 2.0f;
        o = p.camPos.xyz;
        d = normalize(p.camFwd.xyz + u * p.camRight.xyz + v * p.camUp.xyz);
    } else {
        const uint i = tid.y * p.width + tid.x;
        o = float3(p.bounds.x + rnd(i, p.seed) * p.bounds.z, p.bounds.w, p.bounds.y + rnd(i, p.seed + 1) * p.bounds.z);
        const float cz = -rnd(i, p.seed + 2);                       // cos(theta) in (-1, 0]: lower hemisphere
        const float ph = 6.2831853f * rnd(i, p.seed + 3);
        const float sz = sqrt(max(0.0f, 1.0f - cz * cz));
        d = float3(sz * cos(ph), cz, sz * sin(ph));
    }
}

inline void store(device uint2* out, device float4* dump, constant RtParams& p, uint2 tid, float3 o, float3 d, bool hit,
                  float t, uint inst, uint prim) {
    const uint i = tid.y * p.width + tid.x;
    out[i] = uint2(hit ? as_type<uint>(t) : 0xFFFFFFFFu, hit ? ((inst << 20) | prim) : 0xFFFFFFFFu);
    if ((i % p.dumpStride) == 0u) {
        dump[2 * (i / p.dumpStride)] = float4(o, 0.0f);
        dump[2 * (i / p.dumpStride) + 1] = float4(d, 0.0f);
    }
}

kernel void rays_isect(instance_acceleration_structure as [[buffer(0)]], device uint2* out [[buffer(1)]],
                       constant RtParams& p [[buffer(2)]], device float4* dump [[buffer(4)]],
                       uint2 tid [[thread_position_in_grid]]) {
    float3 o, d;
    makeRay(p, tid, o, d);
    intersector<triangle_data, instancing> isect;
    isect.assume_geometry_type(geometry_type::triangle);
    auto h = isect.intersect(ray(o, d, 0.0f, 1e9f), as);
    const bool hit = h.type == intersection_type::triangle;
    store(out, dump, p, tid, o, d, hit, h.distance, h.instance_id, h.primitive_id);
}

[[intersection(triangle, triangle_data, instancing)]]
bool alpha_test(uint primitive_id [[primitive_id]]) { return alphaPass(primitive_id); }

kernel void rays_isect_if(instance_acceleration_structure as [[buffer(0)]], device uint2* out [[buffer(1)]],
                          constant RtParams& p [[buffer(2)]],
                          intersection_function_table<triangle_data, instancing> ift [[buffer(3)]],
                          device float4* dump [[buffer(4)]], uint2 tid [[thread_position_in_grid]]) {
    float3 o, d;
    makeRay(p, tid, o, d);
    intersector<triangle_data, instancing> isect;
    isect.assume_geometry_type(geometry_type::triangle);
    auto h = isect.intersect(ray(o, d, 0.0f, 1e9f), as, ift);
    const bool hit = h.type == intersection_type::triangle;
    store(out, dump, p, tid, o, d, hit, h.distance, h.instance_id, h.primitive_id);
}

kernel void rays_query(instance_acceleration_structure as [[buffer(0)]], device uint2* out [[buffer(1)]],
                       constant RtParams& p [[buffer(2)]], device float4* dump [[buffer(4)]],
                       uint2 tid [[thread_position_in_grid]]) {
    float3 o, d;
    makeRay(p, tid, o, d);
    intersection_query<triangle_data, instancing> q(ray(o, d, 0.0f, 1e9f), as);
    while (q.next()) {
        if (q.get_candidate_intersection_type() == intersection_type::triangle) q.commit_triangle_intersection();
    }
    const bool hit = q.get_committed_intersection_type() == intersection_type::triangle;
    store(out, dump, p, tid, o, d, hit, q.get_committed_distance(), q.get_committed_instance_id(),
          q.get_committed_primitive_id());
}

kernel void rays_query_if(instance_acceleration_structure as [[buffer(0)]], device uint2* out [[buffer(1)]],
                          constant RtParams& p [[buffer(2)]], device float4* dump [[buffer(4)]],
                          uint2 tid [[thread_position_in_grid]]) {
    float3 o, d;
    makeRay(p, tid, o, d);
    intersection_query<triangle_data, instancing> q(ray(o, d, 0.0f, 1e9f), as);
    while (q.next()) {
        if (q.get_candidate_intersection_type() == intersection_type::triangle &&
            alphaPass(q.get_candidate_primitive_id()))
            q.commit_triangle_intersection();
    }
    const bool hit = q.get_committed_intersection_type() == intersection_type::triangle;
    store(out, dump, p, tid, o, d, hit, q.get_committed_distance(), q.get_committed_instance_id(),
          q.get_committed_primitive_id());
}
