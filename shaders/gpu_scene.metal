// gpu_scene.metal -- F5 GPU scene kernels: queue clear, delta scatter (F5.1),
// instance culling + stable compaction (F5.5, D4) and the ICB draw build
// (F5.3, D2).  Kernel names, argument-table slots (= [[buffer(n)]]), threadgroup
// sizes and parameter blocks are the contract of renderer/gpu_scene_layout.h;
// the cull math is renderer/cull_math.h (shared with the C++ reference).
//
// Dispatch rules (the host encodes every one of them every frame):
//   scene_queue_clear   dispatchThreads(max(queues, 1)); thread 0 also zeroes the counters
//   scene_scatter       dispatchThreads(max(count, 1), SCENE_SCATTER_GROUP)
//   scene_cull_flags    dispatchThreadgroups(groupCount, SCENE_CULL_GROUP threads): whole
//   scene_cull_write      groups, because the SIMD-group reductions need all 32 lanes
//   scene_cull_scan     dispatchThreadgroups(1, SCENE_CULL_GROUP threads)
//   scene_draw_build    dispatchThreads(commandCount, SCENE_DRAW_GROUP)
// groupCount = ceil(slotCount / 1024) >= 1.  Between consecutive kernels:
// Dispatch -> Dispatch barrier (S5).

#include <metal_stdlib>

#include "renderer/cull_math.h"
#include "renderer/gpu_queue.h"
#include "renderer/gpu_scene_layout.h"
#include "renderer/gpu_types.h"

using namespace metal;
using namespace phosphor;

// The attribute slots below are literals (an attribute needs a constant
// expression); keep them equal to the layout header.
static_assert(SB_CLEAR_QUEUES == 1 && SB_CLEAR_COUNTERS == 2, "scene_queue_clear slots");
static_assert(SB_SCATTER_RECORDS == 1 && SB_SCATTER_DST == 2, "scene_scatter slots");
static_assert(SB_CULL_INSTANCES == 1 && SB_CULL_MESHES == 2 && SB_CULL_FLAGS == 3 && SB_CULL_GROUPS == 4 &&
                  SB_CULL_COUNTERS == 5 && SB_CULL_VISIBLE == 6 && SB_CULL_PREFIX == 7,
              "scene_cull_* slots");
static_assert(SB_DRAW_BUCKETS == 1 && SB_DRAW_COMMANDS == 2 && SB_DRAW_PREFIX == 3 && SB_DRAW_ICB == 4 &&
                  SB_DRAW_INDICES == 5 && SB_DRAW_ARGS == 6 && SB_DRAW_COUNTERS == 7 && SB_DRAW_GATE == 8,
              "scene_draw_build slots");
static_assert(SCENE_CULL_GROUP == 1024, "the cull kernels are written for 32 SIMD-groups of 32");

// ---- queue clear / scatter -----------------------------------------------------------

// Queues 1 .. queues live back to back (queue L at (L - 1) * strideBytes): clear
// count / overflow / groups of each (capacity stays) and the counters.
kernel void scene_queue_clear(constant GPUQueueClearParams& p [[buffer(0)]], device uchar* queues [[buffer(1)]],
                              device uint* counters [[buffer(2)]], uint tid [[thread_position_in_grid]]) {
    if (tid < p.queues) {
        device GPUQueueHeader& h = *reinterpret_cast<device GPUQueueHeader*>(queues + ulong(tid) * ulong(p.strideBytes));
        h.count    = 0u;
        h.overflow = 0u;
        h.groups[0] = 0u;
        h.groups[1] = 0u;
        h.groups[2] = 0u;
    }
    if (tid == 0u) {
        for (uint i = 0; i < sizeof(GPUSceneCounters) / 4u; ++i) counters[i] = 0u;
    }
}

// dst[slot * words + w] = payload[w]: one thread per record.  Records of one
// dispatch target distinct slots (the store coalesces its deltas), so the
// order does not matter.
kernel void scene_scatter(constant GPUScatterParams& p [[buffer(0)]], const device GPUDeltaRecord* records [[buffer(1)]],
                          device uint* dst [[buffer(2)]], uint tid [[thread_position_in_grid]]) {
    if (tid >= p.count) return;
    const device GPUDeltaRecord& r = records[tid];
    const ulong base               = ulong(r.slot) * ulong(p.words);
    for (uint w = 0; w < p.words; ++w) dst[base + w] = r.payload[w];
}

// ---- instance culling (D4: stable reduce-then-scan) ----------------------------------

// (1) one thread per slot: world sphere + cullSphere.  flags[slot]: 0 invalid,
// 1 visible, 2 * reason culled.  Per-group visible count, counters with one
// atomic per SIMD-group and counter.
kernel void scene_cull_flags(constant GPUCullParams& p [[buffer(0)]], const device GPUInstance* instances [[buffer(1)]],
                             const device GPUMeshInfo* meshes [[buffer(2)]], device uint* flags [[buffer(3)]],
                             device uint* groupCounts [[buffer(4)]], device atomic_uint* counters [[buffer(5)]],
                             uint tid [[thread_position_in_grid]], uint gid [[threadgroup_position_in_grid]],
                             uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint sums[32];
    uint reason = 0u;
    bool valid  = false;
    if (tid < p.slotCount) {
        const device GPUInstance& in = instances[tid];
        if ((in.flags & INSTANCE_FLAG_VALID) != 0u) {
            valid = true;
            const GPUWorldSphere w = cullWorldSphere(in.modelMatrix, meshes[in.meshIndex].boundingSphere);
            reason                 = cullSphere(p, w.x, w.y, w.z, w.r);
        }
        flags[tid] = !valid ? 0u : (reason == CULL_REASON_VISIBLE ? CULL_FLAG_OUT_VISIBLE : reason * 2u);
    }
    const bool vis  = valid && reason == CULL_REASON_VISIBLE;
    const uint nVis = simd_sum(vis ? 1u : 0u);
    const uint nVal = simd_sum(valid ? 1u : 0u);
    const uint nF   = simd_sum(valid && reason == CULL_REASON_FRUSTUM ? 1u : 0u);
    const uint nD   = simd_sum(valid && reason == CULL_REASON_DISTANCE ? 1u : 0u);
    const uint nS   = simd_sum(valid && reason == CULL_REASON_SIZE ? 1u : 0u);
    if (lane == 0u) {
        sums[sg] = nVis;
        if (nVal != 0u) atomic_fetch_add_explicit(&counters[SCENE_COUNTER_TESTED], nVal, memory_order_relaxed);
        if (nVis != 0u) atomic_fetch_add_explicit(&counters[SCENE_COUNTER_VISIBLE], nVis, memory_order_relaxed);
        if (nF != 0u) atomic_fetch_add_explicit(&counters[SCENE_COUNTER_CULLED_FRUSTUM], nF, memory_order_relaxed);
        if (nD != 0u) atomic_fetch_add_explicit(&counters[SCENE_COUNTER_CULLED_DISTANCE], nD, memory_order_relaxed);
        if (nS != 0u) atomic_fetch_add_explicit(&counters[SCENE_COUNTER_CULLED_SIZE], nS, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        const uint t = simd_sum(sums[lane]);
        if (lane == 0u) groupCounts[gid] = t;
    }
}

// (2) ONE threadgroup of 1024 threads: exclusive scan of groupCount group
// counts, in place (counts -> offsets), in chunks of 1024 with a running carry:
// any groupCount works.
kernel void scene_cull_scan(constant GPUCullParams& p [[buffer(0)]], device uint* groups [[buffer(4)]],
                            uint tid [[thread_position_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint sums[32];
    threadgroup uint carry;
    threadgroup uint chunkTotal;
    if (tid == 0u) carry = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint base = 0u; base < p.groupCount; base += SCENE_CULL_GROUP) {
        const uint idx  = base + tid;
        const uint v    = idx < p.groupCount ? groups[idx] : 0u;
        const uint excl = simd_prefix_exclusive_sum(v);
        if (lane == 31u) sums[sg] = excl + v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            const uint s = sums[lane];
            sums[lane]   = simd_prefix_exclusive_sum(s);
            const uint t = simd_sum(s);
            if (lane == 0u) chunkTotal = t;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (idx < p.groupCount) groups[idx] = carry + sums[sg] + excl;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) carry += chunkTotal;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// (3) every visible slot goes to groupOffset + intra-group exclusive prefix: the
// visible list in slot order and the exclusive prefix of the flags (slotCount + 1
// entries, the last = total visible).
kernel void scene_cull_write(constant GPUCullParams& p [[buffer(0)]], const device uint* flags [[buffer(3)]],
                             const device uint* groupOffsets [[buffer(4)]], device uint* visible [[buffer(6)]],
                             device uint* prefix [[buffer(7)]], uint tid [[thread_position_in_grid]],
                             uint gid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint sums[32];
    const uint f    = (tid < p.slotCount && (flags[tid] & CULL_FLAG_OUT_VISIBLE) != 0u) ? 1u : 0u;
    const uint excl = simd_prefix_exclusive_sum(f);
    if (lane == 31u) sums[sg] = excl + f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) sums[lane] = simd_prefix_exclusive_sum(sums[lane]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint pos = groupOffsets[gid] + sums[sg] + excl;
    if (tid < p.slotCount) {
        prefix[tid] = pos;
        if (f != 0u) visible[pos] = tid;
        if (tid == p.slotCount - 1u) prefix[p.slotCount] = pos + f;
    }
}

// ---- draw build (D2) -----------------------------------------------------------------

struct GPUIcbContainer {
    command_buffer icb [[id(0)]];
};

// One thread per ICB command.  A sentinel or an empty bucket is reset; otherwise
// one indexed draw of the bucket's visible instances: instanceCount from the
// prefix over the bucket's slot region, baseInstance = prefix[firstSlot] (the
// forward vertex shader reads instances[visible[instance_id]]).  drawArgs
// repeats {instanceCount, baseInstance} for the host self-check.  The bucket
// regions must lie inside the prefix (firstSlot + capacity <= slotCount).
kernel void scene_draw_build(constant GPUDrawParams& p [[buffer(0)]], const device GPUDrawBucket* buckets [[buffer(1)]],
                             const device uint* commandBuckets [[buffer(2)]], const device uint* prefix [[buffer(3)]],
                             constant GPUIcbContainer& c [[buffer(4)]], const device uint* indices [[buffer(5)]],
                             device uint* drawArgs [[buffer(6)]], device atomic_uint* counters [[buffer(7)]],
                             const device uint* gate [[buffer(8)]], uint tid [[thread_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]]) {
    uint drawn = 0u;
    const bool enabled = gate[0] != 0u;
    if (tid < p.commandCount) {
        const uint b       = commandBuckets[tid];
        uint instanceCount = 0u;
        uint baseInstance  = 0u;
        if (b != 0xFFFFFFFFu) {
            const GPUDrawBucket k = buckets[b];
            baseInstance          = prefix[k.firstSlot];
            instanceCount         = prefix[k.firstSlot + k.capacity] - baseInstance;
        }
        render_command cmd(c.icb, tid);
        if (instanceCount == 0u || !enabled) {
            cmd.reset();
            if (instanceCount == 0u) baseInstance = 0u;
        } else {
            const GPUDrawBucket k = buckets[b];
            cmd.draw_indexed_primitives(primitive_type::triangle, k.indexCount, indices + k.indexOffset, instanceCount,
                                        k.vertexOffset, baseInstance);
            drawn = 1u;
        }
        drawArgs[tid * 2u + 0u] = instanceCount;
        drawArgs[tid * 2u + 1u] = baseInstance;
    }
    const uint n = simd_sum(drawn);
    if (lane == 0u && n != 0u) atomic_fetch_add_explicit(&counters[SCENE_COUNTER_DRAW_COMMANDS], n, memory_order_relaxed);
}
