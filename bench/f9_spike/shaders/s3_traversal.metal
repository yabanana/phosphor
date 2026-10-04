// F9-S3: traversal cost and robustness.  Ray types (camera primary, shadow,
// AO, diffuse) traced with the intersector or intersection_query, closest or
// any hit, plus the origin strategies of the self-intersection study.
// Layouts shared with bench/f9_spike/s3_traversal.cpp (S3Params, S3Prim, S3Res).
#include "f9_rt.h"
#include "renderer/gpu_types.h"

struct S3Params {
    uint width, height, seed, strategy; // strategy: 0 none, 1 tmin 1e-4, 2 tmin 1e-3, 3 Waechter-Binder offset (object space), 4 same in world space
    float eye[3];
    float aoTmax;
    float fwd[3];
    float pad0;
    float rightS[3]; // right * tan(fov/2) * aspect
    float pad1;
    float upS[3];    // up * tan(fov/2)
    float pad2;
    float sun[3];    // unit vector TOWARD the sun
    float pad3;
};

struct S3Prim {   // 64 B, one per pixel (primary hit)
    float p[3];   // hit point (barycentric, object space -> world)
    float t;      // < 0: miss
    float n[3];   // unit geometric normal, facing the camera
    uint instance;
    float pf[3];  // Waechter-Binder offset point on the side of n
    uint primitive;
    float pb[3];  // ... on the opposite side
    uint pad;
};

struct S3Res {    // 16 B, one per pixel and kernel
    float t;      // >= 0 hit, -1 miss, -2 no origin (primary miss)
    uint instance;
    uint primitive;
    uint pad;
};

constant uint c_type [[function_constant(0)]];       // 0 primary, 1 shadow, 2 AO, 3 diffuse
constant uint c_mode [[function_constant(1)]];       // 0 isect closest, 1 isect any, 2 query closest, 3 query any
constant bool c_assume_tri [[function_constant(2)]];
constant bool c_force_opaque [[function_constant(3)]];

inline float3 f3(const device float* p) { return float3(p[0], p[1], p[2]); }
inline float3 f3(constant float* p) { return float3(p[0], p[1], p[2]); }

inline uint hashu(uint h) { h ^= h >> 16; h *= 0x7FEB352Du; h ^= h >> 15; h *= 0x846CA68Bu; h ^= h >> 16; return h; }
inline float rnd(uint a, uint b) { return float(hashu(a * 0x9E3779B1u ^ hashu(b + 0x85EBCA6Bu)) >> 8) * (1.0f / 16777216.0f); }

inline float3 cosineDir(float3 n, float u1, float u2) {
    const float r = sqrt(u1), phi = 6.2831853f * u2;
    const float3 a = abs(n.x) > 0.9f ? float3(0, 1, 0) : float3(1, 0, 0);
    const float3 t = normalize(cross(a, n));
    const float3 b = cross(n, t);
    return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(0.0f, 1.0f - u1)));
}

// Waechter & Binder, "A Fast and Robust Method for Avoiding Self-Intersection"
// (Ray Tracing Gems ch. 6).
inline float3 offsetRay(float3 p, float3 n) {
    const float origin = 1.0f / 32.0f, floatScale = 1.0f / 65536.0f, intScale = 256.0f;
    const int3 of = int3(intScale * n.x, intScale * n.y, intScale * n.z);
    const float3 pi = float3(as_type<float>(as_type<int>(p.x) + (p.x < 0 ? -of.x : of.x)),
                             as_type<float>(as_type<int>(p.y) + (p.y < 0 ? -of.y : of.y)),
                             as_type<float>(as_type<int>(p.z) + (p.z < 0 ? -of.z : of.z)));
    return float3(abs(p.x) < origin ? p.x + floatScale * n.x : pi.x,
                  abs(p.y) < origin ? p.y + floatScale * n.y : pi.y,
                  abs(p.z) < origin ? p.z + floatScale * n.z : pi.z);
}

inline float3 xformPoint(device const phosphor::GPUInstance& m, float3 p) {
    return float3(m.modelMatrix[0] * p.x + m.modelMatrix[4] * p.y + m.modelMatrix[8] * p.z + m.modelMatrix[12],
                  m.modelMatrix[1] * p.x + m.modelMatrix[5] * p.y + m.modelMatrix[9] * p.z + m.modelMatrix[13],
                  m.modelMatrix[2] * p.x + m.modelMatrix[6] * p.y + m.modelMatrix[10] * p.z + m.modelMatrix[14]);
}

inline float3 primaryDir(constant S3Params& P, uint2 tid) {
    const float u = (float(tid.x) + 0.5f) / float(P.width) * 2.0f - 1.0f;
    const float v = 1.0f - (float(tid.y) + 0.5f) / float(P.height) * 2.0f;
    return normalize(f3(P.fwd) + f3(P.rightS) * u + f3(P.upS) * v);
}

// The ray of pixel `tid` for the compile-time ray type; valid = false when a
// secondary ray has no origin (primary miss).
inline ray makeRay(constant S3Params& P, device const S3Prim* prims, uint2 tid, thread bool& valid) {
    valid = true;
    if (c_type == 0) return ray(f3(P.eye), primaryDir(P, tid), 0.0f, 1e30f);
    const uint lin = tid.y * P.width + tid.x;
    const device S3Prim& h = prims[lin];
    if (h.t < 0.0f) { valid = false; return ray(float3(0), float3(0, 1, 0), 0.0f, 0.0f); }
    const float3 n = f3(h.n);
    float3 d;
    if (c_type == 1) d = f3(P.sun);
    else d = cosineDir(n, rnd(lin, P.seed), rnd(lin ^ 0x68BC21EBu, P.seed + 7u));
    const bool front = dot(n, d) >= 0.0f;
    float3 o = f3(h.p);
    float tmin = 0.0f;
    if (P.strategy == 1) tmin = 1e-4f;
    else if (P.strategy == 2) tmin = 1e-3f;
    else if (P.strategy == 3) o = front ? f3(h.pf) : f3(h.pb);
    else if (P.strategy == 4) o = offsetRay(o, front ? n : -n); // same offset, applied to the world-space point and normal
    return ray(o, d, tmin, c_type == 2 ? P.aoTmax : 1e30f);
}

inline S3Res traceRay(ray r, instance_acceleration_structure as) {
    S3Res o;
    o.t = -1.0f; o.instance = 0xFFFFFFFFu; o.primitive = 0xFFFFFFFFu; o.pad = 0;
    if (c_mode < 2) {
        intersector<triangle_data, instancing> isect;
        if (c_assume_tri) isect.assume_geometry_type(geometry_type::triangle);
        if (c_force_opaque) isect.force_opacity(forced_opacity::opaque);
        if (c_mode == 1) isect.accept_any_intersection(true);
        const auto h = isect.intersect(r, as, 0xFFu);
        if (h.type == intersection_type::triangle) { o.t = h.distance; o.instance = h.instance_id; o.primitive = h.primitive_id; }
    } else {
        intersection_params params;
        if (c_assume_tri) params.assume_geometry_type(geometry_type::triangle);
        if (c_force_opaque) params.force_opacity(forced_opacity::opaque);
        intersection_query<triangle_data, instancing> q(r, as, 0xFFu, params);
        if (c_mode == 3) {
            // Any hit: commit the first candidate and stop (bounded: the loop ends at the first triangle).
            while (q.next()) {
                if (q.get_candidate_intersection_type() == intersection_type::triangle) {
                    q.commit_triangle_intersection();
                    break;
                }
            }
        } else {
            while (q.next()) {
                if (q.get_candidate_intersection_type() == intersection_type::triangle) q.commit_triangle_intersection();
            }
        }
        if (q.get_committed_intersection_type() == intersection_type::triangle) {
            o.t = q.get_committed_distance();
            o.instance = q.get_committed_instance_id();
            o.primitive = q.get_committed_primitive_id();
        }
    }
    return o;
}

// Timed / checked kernel: one thread per pixel.
kernel void s3_trace(instance_acceleration_structure as [[buffer(0)]], constant S3Params& P [[buffer(1)]],
                     device const S3Prim* prims [[buffer(2)]], device S3Res* out [[buffer(3)]],
                     uint2 tid [[thread_position_in_grid]]) {
    if (tid.x >= P.width || tid.y >= P.height) return;
    bool valid;
    const ray r = makeRay(P, prims, tid, valid);
    S3Res o;
    if (valid) o = traceRay(r, as);
    else { o.t = -2.0f; o.instance = 0xFFFFFFFFu; o.primitive = 0xFFFFFFFFu; o.pad = 0; }
    out[tid.y * P.width + tid.x] = o;
}

// The rays of s3_trace, dumped (CPU checks): invalid pixels get tmax = -1.
kernel void s3_gen_rays(constant S3Params& P [[buffer(1)]], device const S3Prim* prims [[buffer(2)]],
                        device RayIn* rays [[buffer(3)]], uint2 tid [[thread_position_in_grid]]) {
    if (tid.x >= P.width || tid.y >= P.height) return;
    bool valid;
    const ray r = makeRay(P, prims, tid, valid);
    RayIn o;
    o.o = r.origin; o.d = r.direction; o.tmin = r.min_distance; o.tmax = valid ? r.max_distance : -1.0f;
    rays[tid.y * P.width + tid.x] = o;
}

// Primary hits with everything the secondary kernels need (position,
// normal, Waechter-Binder offset points on both sides).
kernel void s3_primary_hits(instance_acceleration_structure as [[buffer(0)]], constant S3Params& P [[buffer(1)]],
                            device S3Prim* prims [[buffer(2)]], device const phosphor::GPUVertex* verts [[buffer(4)]],
                            device const uint* indices [[buffer(5)]], device const phosphor::GPUMeshInfo* meshes [[buffer(6)]],
                            device const phosphor::GPUInstance* instances [[buffer(7)]], uint2 tid [[thread_position_in_grid]]) {
    if (tid.x >= P.width || tid.y >= P.height) return;
    const uint lin = tid.y * P.width + tid.x;
    const float3 d = primaryDir(P, tid);
    intersector<triangle_data, instancing> isect;
    isect.assume_geometry_type(geometry_type::triangle);
    const auto h = isect.intersect(ray(f3(P.eye), d, 0.0f, 1e30f), as, 0xFFu);
    S3Prim o;
    o.t = -1.0f; o.instance = 0xFFFFFFFFu; o.primitive = 0xFFFFFFFFu; o.pad = 0;
    for (int i = 0; i < 3; ++i) { o.p[i] = 0; o.n[i] = 0; o.pf[i] = 0; o.pb[i] = 0; }
    if (h.type == intersection_type::triangle) {
        const device phosphor::GPUInstance& inst = instances[h.instance_id];
        const device phosphor::GPUMeshInfo& mi = meshes[inst.meshIndex];
        const uint base = mi.indexOffset + 3u * h.primitive_id;
        const device phosphor::GPUVertex& a = verts[mi.vertexOffset + indices[base]];
        const device phosphor::GPUVertex& b = verts[mi.vertexOffset + indices[base + 1]];
        const device phosphor::GPUVertex& c = verts[mi.vertexOffset + indices[base + 2]];
        const float3 pa = float3(a.px, a.py, a.pz), pb = float3(b.px, b.py, b.pz), pc = float3(c.px, c.py, c.pz);
        const float u = h.triangle_barycentric_coord.x, v = h.triangle_barycentric_coord.y;
        const float3 pObj = pa * (1.0f - u - v) + pb * u + pc * v;
        const float3 nObj = normalize(cross(pb - pa, pc - pa));
        // World geometric normal = cross of the world edges (correct for mirrored and scaled instances).
        const float3 wa = xformPoint(inst, pa), wb = xformPoint(inst, pb), wc = xformPoint(inst, pc);
        float3 nW = normalize(cross(wb - wa, wc - wa));
        const float3 pW = xformPoint(inst, pObj);
        // Offset points computed in object space (as in the paper), then transformed.
        const float3 q0 = xformPoint(inst, offsetRay(pObj, nObj));
        const float3 q1 = xformPoint(inst, offsetRay(pObj, -nObj));
        if (dot(nW, d) > 0.0f) nW = -nW; // face the camera
        // pf: side of the (camera-facing) normal; a mirrored instance swaps the object-space sides.
        const bool q0Front = dot(q0 - pW, nW) >= 0.0f;
        const float3 pf = q0Front ? q0 : q1, pbk = q0Front ? q1 : q0;
        o.t = h.distance; o.instance = h.instance_id; o.primitive = h.primitive_id;
        for (int i = 0; i < 3; ++i) { o.p[i] = pW[i]; o.n[i] = nW[i]; o.pf[i] = pf[i]; o.pb[i] = pbk[i]; }
    }
    prims[lin] = o;
}
