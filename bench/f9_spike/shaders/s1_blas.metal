// F9-S1: kernels of the BLAS lifecycle spike (deformation producers and a
// tracer with in-kernel rays).  Every loop is bounded by the grid.
#include "f9_rt.h"

struct DeformParams {
    uint first;   // first vertex (index in the GPUVertex array)
    uint count;
    uint strideFloats; // GPUVertex stride in floats (12)
    float amplitude;
    float k;
    float phase;
    uint pad0, pad1;
};

// positions = base + (0, A sin(k x + phase), 0); the base copy is a packed
// float3 array, so repeated deformations never accumulate.
kernel void deform_sine(device float* vertices [[buffer(0)]], device const packed_float3* base [[buffer(1)]],
                        constant DeformParams& p [[buffer(2)]], uint tid [[thread_position_in_grid]]) {
    if (tid >= p.count) return;
    const float3 b = float3(base[tid]);
    device float* v = vertices + size_t(p.first + tid) * p.strideFloats;
    v[0] = b.x;
    v[1] = b.y + p.amplitude * sin(p.k * b.x + p.phase);
    v[2] = b.z;
}

struct PlaneParams {
    uint n;      // quads per side: (n+1)^2 vertices
    float cell;  // quad edge
    float y;     // the state: every vertex at this height
    uint count;  // rays (trace_down) / spin iterations of the producer (plane_set_state, bounded)
};

// Plane vertices (packed float3, row-major), all at height p.y.
kernel void plane_set_state(device packed_float3* verts [[buffer(0)]], constant PlaneParams& p [[buffer(1)]],
                            uint tid [[thread_position_in_grid]]) {
    const uint side = p.n + 1;
    if (tid >= side * side) return;
    const uint ix = tid % side, iz = tid / side;
    // Optional slow producer: a bounded dependent chain before the write (the compare keeps it alive).
    float acc = float(tid & 7u) + 1.0f;
    const uint spin = min(p.count, 50000u);
    for (uint i = 0; i < spin; ++i) acc = fma(acc, 1.0000001f, 1e-7f);
    const float y = (acc == -12345.678f) ? p.y + 1e-30f : p.y;
    verts[tid] = packed_float3(float(ix) * p.cell, y, float(iz) * p.cell);
}

// Vertical rays from y = 2 at hashed positions strictly inside the plane;
// writes t (or -1).  State A (y = 0) gives t = 2, state B (y = 1) gives t = 1.
kernel void trace_down(primitive_acceleration_structure as [[buffer(0)]], device float* ts [[buffer(1)]],
                       constant PlaneParams& p [[buffer(2)]], uint tid [[thread_position_in_grid]]) {
    if (tid >= p.count) return;
    uint h = tid * 0x9E3779B1u + 0x7F4A7C15u;
    h ^= h >> 16; h *= 0x7FEB352Du; h ^= h >> 15; h *= 0x846CA68Bu; h ^= h >> 16;
    uint g = h * 0x85EBCA6Bu + tid;
    g ^= g >> 13; g *= 0xC2B2AE35u; g ^= g >> 16;
    const float extent = float(p.n) * p.cell;
    const float x = (0.02f + 0.96f * float(h >> 8) * (1.0f / 16777216.0f)) * extent;
    const float z = (0.02f + 0.96f * float(g >> 8) * (1.0f / 16777216.0f)) * extent;
    intersector<> isect;
    const auto r = isect.intersect(ray(float3(x, 2.0f, z), float3(0, -1, 0), 0.0f, 100.0f), as);
    ts[tid] = (r.type == intersection_type::triangle) ? r.distance : -1.0f;
}
