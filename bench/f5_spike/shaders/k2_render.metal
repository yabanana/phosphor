// F5-K2: render side of the GPU scene kernel check (bench/f5_spike/k2_gpu_scene.cpp).
// Reads the visible list like the engine's forward vertex shader will:
// instances[visible[instance_id]] (instance_id includes the base instance,
// vertex_id the base vertex).  Draws a tiny fan per instance at its
// translation (x, y) / 500 with a unique depth per slot and writes slot + 1
// into an R32Uint target; drawn[slot] = 1 proves the instance was processed.
#include <metal_stdlib>
using namespace metal;

struct K2Inst {
    float m[16]; // column-major model matrix (GPUInstance)
    uint  mesh;
    uint  material;
    uint  flags;
    uint  pad;
};

struct K2Out {
    float4 pos [[position]];
    uint   slot [[flat]];
};

vertex K2Out k2_vs(uint vid [[vertex_id]], uint iid [[instance_id]], const device K2Inst* inst [[buffer(0)]],
                   const device uint* vis [[buffer(1)]], const device float2* verts [[buffer(2)]],
                   device uint* drawn [[buffer(3)]]) {
    const uint slot = vis[iid];
    const device K2Inst& in = inst[slot];
    drawn[slot] = 1u; // idempotent: exact coverage check (vertex invocation counts are not exact)
    float2 v = verts[vid];
    if ((in.flags & (1u << 3)) != 0u) v.x = -v.x; // INSTANCE_FLAG_MIRRORED
    K2Out o;
    const float depth = float((slot * 2654435761u) % (1u << 24) + 1u) / float(1u << 24) * 0.98f;
    o.pos             = float4(float2(in.m[12], in.m[13]) * (1.0f / 500.0f) + v * 0.01f, depth, 1.0f);
    o.slot            = slot;
    return o;
}

fragment uint4 k2_fs(K2Out in [[stage_in]]) { return uint4(in.slot + 1u, 0u, 0u, 0u); }
