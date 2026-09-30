// B-16: the same triangle grid drawn through a classic indexed vertex
// pipeline and through object + mesh shaders (meshlets = BW x BH blocks of
// grid cells).  A grid of G x G cells, 2 triangles per cell, vertex (i, j) at
// NDC (-1 + 2 i / G, -1 + 2 j / G), rendered to a 1024^2 R8 target by a
// fragment shader that writes 1.  Vertex and mesh paths read the same
// position buffer, so the rasterised image is identical.
#include <metal_stdlib>
using namespace metal;

struct Params {
    uint meshletCount; // meshlets to process (half of them for the 2x control)
    uint meshletsPerRow;
    int  cullParity;   // -1: keep all; 0/1: object shader culls meshlets with ((mx + my) & 1) == parity
    uint zero;         // run-time zero (keeps payload reads alive)
    uint gridCells;    // G
    uint pad[3];
};

struct VertexOut {
    float4 pos [[position]];
};

// ---- classic pipeline ----
vertex VertexOut b16_vs(uint vid [[vertex_id]], constant Params& p [[buffer(0)]], device const float2* pos [[buffer(1)]]) {
    VertexOut o;
    o.pos = float4(pos[vid], 0.5, 1.0);
    return o;
}
fragment half4 b16_fs() { return half4(1.0h); }

// ---- object + mesh ----
template <uint PW>
struct Payload {
    uint ids[32]; // visible meshlet ids (compacted)
    uint pad[PW > 0 ? PW : 1]; // payload ballast: written by the object shader, read by the mesh shader
};

// Object threadgroup = one SIMD-group = 32 consecutive meshlets.
template <uint PW>
inline void objectBody(object_data Payload<PW>& payload, mesh_grid_properties mgp, constant Params& p, uint lane, uint tgid) {
    const uint id = tgid * 32u + lane;
    bool vis = id < p.meshletCount;
    if (vis && p.cullParity >= 0) {
        const uint mx = id % p.meshletsPerRow, my = id / p.meshletsPerRow;
        vis = int((mx + my) & 1u) != p.cullParity;
    }
    const uint prefix = simd_prefix_exclusive_sum(vis ? 1u : 0u);
    const uint count  = simd_sum(vis ? 1u : 0u);
    if (vis) payload.ids[prefix] = id;
    for (uint i = lane; i < PW; i += 32u) payload.pad[i] = id + i;
    if (lane == 0u) mgp.set_threadgroups_per_grid(uint3(count, 1, 1));
}

template <uint BW, uint BH, uint PW, uint T, uint MAXV, uint MAXP>
inline void meshBody(mesh<VertexOut, void, MAXV, MAXP, topology::triangle> m, const object_data Payload<PW>& payload,
                     constant Params& p, device const float2* pos, uint tid, uint gid) {
    constexpr uint nv = (BW + 1u) * (BH + 1u);
    constexpr uint np = 2u * BW * BH;
    const uint id = payload.ids[gid];
    const uint mx = id % p.meshletsPerRow, my = id / p.meshletsPerRow;
    const uint stride = p.gridCells + 1u;
    const uint base = (my * BH) * stride + mx * BW;
    // Reads the payload ballast (keeps its write alive): contributes 0 through the run-time zero.
    const float z = PW > 0 ? float(payload.pad[tid % (PW > 0 ? PW : 1u)] & p.zero) : 0.0f;
    if (tid == 0u) m.set_primitive_count(np);
    for (uint v = tid; v < nv; v += T) {
        const uint i = v % (BW + 1u), j = v / (BW + 1u);
        VertexOut o;
        o.pos = float4(pos[base + j * stride + i] + float2(z, 0.0f), 0.5, 1.0);
        m.set_vertex(v, o);
    }
    for (uint q = tid; q < np; q += T) {
        const uint cell = q >> 1, ci = cell % BW, cj = cell / BW;
        const uint a = cj * (BW + 1u) + ci, b = a + 1u, c = a + (BW + 1u), d = c + 1u;
        if ((q & 1u) == 0u) {
            m.set_index(3u * q + 0u, a);
            m.set_index(3u * q + 1u, b);
            m.set_index(3u * q + 2u, c);
        } else {
            m.set_index(3u * q + 0u, b);
            m.set_index(3u * q + 1u, d);
            m.set_index(3u * q + 2u, c);
        }
    }
}

#define B16_MESHLET(NAME, BW, BH, PW, T, MAXV, MAXP)                                                                     \
    [[object, max_total_threads_per_threadgroup(32)]] void NAME##_obj(                                                   \
        object_data Payload<PW>& payload [[payload]], mesh_grid_properties mgp, constant Params& p [[buffer(0)]],          \
        uint lane [[thread_index_in_threadgroup]], uint tgid [[threadgroup_position_in_grid]]) {                         \
        objectBody<PW>(payload, mgp, p, lane, tgid);                                                                     \
    }                                                                                                                    \
    [[mesh, max_total_threads_per_threadgroup(T)]] void NAME##_mesh(                                                     \
        mesh<VertexOut, void, MAXV, MAXP, topology::triangle> m, const object_data Payload<PW>& payload [[payload]],      \
        constant Params& p [[buffer(0)]], device const float2* pos [[buffer(1)]],                                        \
        uint tid [[thread_index_in_threadgroup]], uint gid [[threadgroup_position_in_grid]]) {                           \
        meshBody<BW, BH, PW, T, MAXV, MAXP>(m, payload, p, pos, tid, gid);                                               \
    }

// name, cells per meshlet (BW x BH), payload ballast words, mesh threads, max vertices, max primitives
B16_MESHLET(b16_m32, 4, 4, 0, 32, 25, 32)
B16_MESHLET(b16_m64, 7, 7, 0, 64, 64, 98)
B16_MESHLET(b16_m96, 8, 8, 0, 96, 81, 128)
B16_MESHLET(b16_m128, 10, 10, 0, 128, 121, 200)
B16_MESHLET(b16_m256, 15, 15, 0, 256, 256, 450)
// payload sizes at the 128-vertex meshlet: ids (128 B) + ballast = 1 KiB and 16 KiB (the maximum)
B16_MESHLET(b16_m128_p1k, 10, 10, 224, 128, 121, 200)
B16_MESHLET(b16_m128_p16k, 10, 10, 4064, 128, 121, 200)
