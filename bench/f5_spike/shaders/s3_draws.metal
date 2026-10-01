// F5-S3: draw emission.  One global vertex buffer (float2) + one u32 index
// buffer, B buckets (one small mesh each), 100k instances in contiguous slot
// ranges per bucket, a deterministic "cull" stand-in that produces a compacted
// visible list in draw order (class, bucket, slot), and the three ways of
// turning it into draws: GPU-encoded ICB, indirect draw per bucket, direct.
#include <metal_stdlib>
using namespace metal;

struct Params {
    uint buckets;
    uint emptyMode; // 0: buckets with b % 10 == 3 are empty; 1: every bucket except b % 10 == 3 is empty
    uint drop;      // bucket whose draw is forced to 0 instances (negative control); ~0u = none
    uint halfMode;  // 1: odd buckets are not encoded (v4: stale-command detection)
};

// Mirrors the engine's GPUMeshInfo + DrawBatch (scalars only).
struct Bucket {
    uint firstSlot, slotCount; // contiguous instance slots of the bucket
    uint indexCount, indexOffset;
    int  vertexOffset;
    uint cls;                  // 0 Back, 1 BackMirrored (cull Front), 2 None
    uint pos;                  // position in draw order (class, bucket)
    uint pad;
};

struct Inst {
    float2 off;
    float  depth;
    float  scale; // negative: x is mirrored
};

inline uint hash32(uint x) { // lowbias32, identical on the CPU
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}
inline bool slotVisible(uint slot) { return (hash32(slot ^ 0x9E3779B9u) & 3u) != 0u; }
inline bool bucketEmpty(uint b, uint mode) { return mode == 0u ? (b % 10u == 3u) : (b % 10u != 3u); }

// ---- cull stand-in ---------------------------------------------------------------

// buffer(0) buckets, (1) visCount out, (2) params
kernel void s3_count(device const Bucket* bk [[buffer(0)]], device uint* visCount [[buffer(1)]],
                     constant Params& p [[buffer(2)]], uint b [[thread_position_in_grid]]) {
    if (b >= p.buckets) return;
    uint n = 0;
    if (!bucketEmpty(b, p.emptyMode)) {
        const Bucket k = bk[b];
        for (uint s = k.firstSlot; s < k.firstSlot + k.slotCount; ++s) n += slotVisible(s) ? 1u : 0u;
    }
    visCount[b] = n;
}

// One thread: exclusive scan in draw order + the compacted command index and
// the per-class execution ranges {location, length} of IndirectCommandBufferExecutionRange:
// ranges[c] = non-compacted (one command per bucket), ranges[3 + c] = compacted.
// buffer(0) buckets, (1) order (pos -> bucket), (2) visCount, (3) firstVis out, (4) cmdIdx out,
// (5) ranges out, (6) params
kernel void s3_scan(device const Bucket* bk [[buffer(0)]], device const uint* order [[buffer(1)]],
                    device const uint* visCount [[buffer(2)]], device uint* firstVis [[buffer(3)]],
                    device uint* cmdIdx [[buffer(4)]], device uint2* ranges [[buffer(5)]],
                    constant Params& p [[buffer(6)]], uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    uint run = 0, nonEmpty = 0;
    uint cstart[3] = {0, 0, 0}, ccnt[3] = {0, 0, 0}, nstart[3] = {0, 0, 0}, ncnt[3] = {0, 0, 0};
    for (uint pos = 0; pos < p.buckets; ++pos) {
        const uint b = order[pos];
        const uint c = bk[b].cls;
        if (ccnt[c] == 0) { cstart[c] = pos; nstart[c] = nonEmpty; }
        ccnt[c] += 1;
        const uint n = visCount[b];
        firstVis[b] = run;
        run += n;
        cmdIdx[b] = nonEmpty;
        if (n > 0) { nonEmpty += 1; ncnt[c] += 1; }
    }
    for (uint c = 0; c < 3; ++c) {
        ranges[c]     = uint2(cstart[c], ccnt[c]);
        ranges[3 + c] = uint2(nstart[c], ncnt[c]);
    }
}

// buffer(0) buckets, (1) firstVis, (2) visCount, (3) vis out
kernel void s3_fill(device const Bucket* bk [[buffer(0)]], device const uint* firstVis [[buffer(1)]],
                    device const uint* visCount [[buffer(2)]], device uint* vis [[buffer(3)]],
                    constant Params& p [[buffer(4)]], uint b [[thread_position_in_grid]]) {
    if (b >= p.buckets || visCount[b] == 0) return;
    const Bucket k = bk[b];
    uint o = firstVis[b];
    for (uint s = k.firstSlot; s < k.firstSlot + k.slotCount; ++s)
        if (slotVisible(s)) vis[o++] = s;
}

// ---- draw emission ---------------------------------------------------------------

struct ICBContainer {
    command_buffer icb [[id(0)]];
};

// buffer(0) ICB container, (1) buckets, (2) visCount, (3) firstVis, (4) params, (5) index buffer, (6) cmdIdx
// Empty bucket: reset() (this variant), draw with 0 instances (zero / a10), no command (compact).
kernel void s3_encode_reset(constant ICBContainer& c [[buffer(0)]], device const Bucket* bk [[buffer(1)]],
                            device const uint* visCount [[buffer(2)]], device const uint* firstVis [[buffer(3)]],
                            constant Params& p [[buffer(4)]], device const uint* idx [[buffer(5)]],
                            uint b [[thread_position_in_grid]]) {
    if (b >= p.buckets || (p.halfMode != 0u && (b & 1u) != 0u)) return;
    const Bucket k = bk[b];
    uint n = visCount[b];
    if (b == p.drop) n = 0;
    render_command cmd(c.icb, k.pos);
    if (n == 0) { cmd.reset(); return; }
    cmd.draw_indexed_primitives(primitive_type::triangle, k.indexCount, idx + k.indexOffset, n, uint(k.vertexOffset), firstVis[b]);
}

kernel void s3_encode_zero(constant ICBContainer& c [[buffer(0)]], device const Bucket* bk [[buffer(1)]],
                           device const uint* visCount [[buffer(2)]], device const uint* firstVis [[buffer(3)]],
                           constant Params& p [[buffer(4)]], device const uint* idx [[buffer(5)]],
                           uint b [[thread_position_in_grid]]) {
    if (b >= p.buckets || (p.halfMode != 0u && (b & 1u) != 0u)) return;
    const Bucket k = bk[b];
    uint n = visCount[b];
    if (b == p.drop) n = 0;
    render_command cmd(c.icb, k.pos);
    cmd.draw_indexed_primitives(primitive_type::triangle, k.indexCount, idx + k.indexOffset, n, uint(k.vertexOffset), firstVis[b]);
}

kernel void s3_encode_compact(constant ICBContainer& c [[buffer(0)]], device const Bucket* bk [[buffer(1)]],
                              device const uint* visCount [[buffer(2)]], device const uint* firstVis [[buffer(3)]],
                              constant Params& p [[buffer(4)]], device const uint* idx [[buffer(5)]],
                              device const uint* cmdIdx [[buffer(6)]], uint b [[thread_position_in_grid]]) {
    if (b >= p.buckets || (p.halfMode != 0u && (b & 1u) != 0u)) return;
    const Bucket k = bk[b];
    uint n = visCount[b];
    if (n == 0) return;
    if (b == p.drop) n = 0;
    render_command cmd(c.icb, cmdIdx[b]);
    cmd.draw_indexed_primitives(primitive_type::triangle, k.indexCount, idx + k.indexOffset, n, uint(k.vertexOffset), firstVis[b]);
}

// Apple10: per-command cull mode and winding (ICB with inheritCullMode = inheritFrontFacingWinding = false);
// a single range, one command per bucket in draw order, no CPU state commands.
kernel void s3_encode_a10(constant ICBContainer& c [[buffer(0)]], device const Bucket* bk [[buffer(1)]],
                          device const uint* visCount [[buffer(2)]], device const uint* firstVis [[buffer(3)]],
                          constant Params& p [[buffer(4)]], device const uint* idx [[buffer(5)]],
                          uint b [[thread_position_in_grid]]) {
    if (b >= p.buckets || (p.halfMode != 0u && (b & 1u) != 0u)) return;
    const Bucket k = bk[b];
    uint n = visCount[b];
    if (b == p.drop) n = 0;
    render_command cmd(c.icb, k.pos);
    cmd.set_front_facing_winding(winding::counterclockwise);
    cmd.set_cull_mode(k.cls == 0u ? cull_mode::back : (k.cls == 1u ? cull_mode::front : cull_mode::none));
    cmd.draw_indexed_primitives(primitive_type::triangle, k.indexCount, idx + k.indexOffset, n, uint(k.vertexOffset), firstVis[b]);
}

// DrawIndexedPrimitivesIndirectArguments {indexCount, instanceCount, indexStart, baseVertex, baseInstance}
// at args[pos * 5].  buffer(0) buckets, (1) visCount, (2) firstVis, (3) args, (4) params
kernel void s3_args(device const Bucket* bk [[buffer(0)]], device const uint* visCount [[buffer(1)]],
                    device const uint* firstVis [[buffer(2)]], device uint* args [[buffer(3)]],
                    constant Params& p [[buffer(4)]], uint b [[thread_position_in_grid]]) {
    if (b >= p.buckets) return;
    const Bucket k = bk[b];
    device uint* a = args + k.pos * 5u;
    a[0] = k.indexCount;
    a[1] = visCount[b];
    a[2] = k.indexOffset;
    a[3] = as_type<uint>(k.vertexOffset);
    a[4] = firstVis[b];
}

// ---- render ------------------------------------------------------------------------

struct VOut {
    float4 pos [[position]];
    uint   slot [[flat]];
};

// buffer(0) instances, (1) visible list, (2) vertices, (3) drawn flags.  vertex_id includes the base
// vertex and instance_id the base instance (like the engine's forward_vs).
vertex VOut s3_vs(uint vid [[vertex_id]], uint iid [[instance_id]], device const Inst* inst [[buffer(0)]],
                  device const uint* vis [[buffer(1)]], device const float2* verts [[buffer(2)]],
                  device uint* drawn [[buffer(3)]]) {
    const uint slot = vis[iid];
    const Inst in = inst[slot];
    drawn[slot] = 1u; // idempotent: exact coverage check (vertex invocation counts are not exact)
    float2 v = verts[vid];
    if (in.scale < 0.0f) v.x = -v.x;
    VOut o;
    o.pos  = float4(in.off + v * fabs(in.scale), in.depth, 1.0f);
    o.slot = slot;
    return o;
}

fragment uint4 s3_fs(VOut in [[stage_in]]) { return uint4(in.slot + 1u, 0u, 0u, 0u); }
