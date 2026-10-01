// transforms.metal -- F5.2 / F5.4: transform hierarchy and procedural motion
// on the GPU, with producer/consumer queues and indirect dispatch.
//
// Kernels (names, slots and parameter blocks: renderer/gpu_scene_layout.h):
//   scene_motion      one thread per motion slot: instances[slot].modelMatrix =
//                     motionWorld (renderer/transform_math.h).
//   scene_queue_args  1 thread: queue header `groups` from `count`.
//   scene_hier_level  one thread per entry of queueIn (indirect dispatch):
//                     level 0: entries are parents whose world matrix is final
//                     (roots): append all their children to queueOut;
//                     level >= 1: entries are nodes: world = world(parent) *
//                     local (bit-identical to glm), then append their children.
//
// Queues (renderer/gpu_queue.h) are appended to with ONE atomic per
// SIMD-group (simd_prefix_exclusive_sum for the lane offsets); an append past
// the capacity is dropped and counted (queue overflow + GPUSceneCounters::
// queueOverflow), never written out of bounds.  Every dispatch is encoded
// every frame: empty work early-outs here (O8).  Barriers between dispatches
// are Dispatch -> Dispatch (S5).
//
// A node reached twice (duplicated queue entry) computes the same value twice:
// both threads write identical bits.

#include <metal_stdlib>
#include "renderer/gpu_scene_layout.h"
#include "renderer/gpu_queue.h"
#include "renderer/transform_math.h"

using namespace metal;
using namespace phosphor;

// Test hook of bench/f5_spike/k3_transforms.cpp (negative control): the bench
// defines it when it expands this file; production builds never do.
#ifdef PHOSPHOR_TRANSFORMS_TEST_DROP_APPEND
constant bool kDropAppend = true;
#else
constant bool kDropAppend = false;
#endif

kernel void scene_motion(constant GPUMotionFrame& frame      [[buffer(0)]],
                         device const uint*       slots      [[buffer(SB_MOTION_SLOTS)]],
                         device const GPUMotion*  motions    [[buffer(SB_MOTION_RECORDS)]],
                         device GPUInstance*      instances  [[buffer(SB_MOTION_INSTANCES)]],
                         uint t [[thread_position_in_grid]]) {
    if (t >= frame.motionCount) return; // also the 1-thread dispatch of an empty list
    const uint slot = slots[t];
    const GPUMotion m = motions[slot];
    const uint k = min(m.speedClass, SCENE_MOTION_CLASSES - 1u);
    float w[16];
    motionWorld(m, frame.sinCos[2 * k], frame.sinCos[2 * k + 1], w);
    for (uint i = 0; i < 16; ++i) instances[slot].modelMatrix[i] = w[i];
}

kernel void scene_queue_args(constant GPUHierParams& params [[buffer(0)]],
                             device uint*             queueIn [[buffer(SB_HIER_QUEUE_IN)]]) {
    const uint n = min(queueIn[GPU_QUEUE_WORD_COUNT], queueIn[GPU_QUEUE_WORD_CAPACITY]);
    queueIn[GPU_QUEUE_WORD_GROUPS + 0] = (n + SCENE_HIER_GROUP - 1u) / SCENE_HIER_GROUP;
    queueIn[GPU_QUEUE_WORD_GROUPS + 1] = 1u;
    queueIn[GPU_QUEUE_WORD_GROUPS + 2] = 1u;
}

kernel void scene_hier_level(constant GPUHierParams&   params       [[buffer(0)]],
                             device const uint*        queueIn      [[buffer(SB_HIER_QUEUE_IN)]],
                             device atomic_uint*       queueOut     [[buffer(SB_HIER_QUEUE_OUT)]],
                             device const GPUTransformNode* nodes   [[buffer(SB_HIER_NODES)]],
                             device GPUInstance*       instances    [[buffer(SB_HIER_INSTANCES)]],
                             device const uint*        childOffsets [[buffer(SB_HIER_CHILD_OFFSETS)]],
                             device const uint*        childSlots   [[buffer(SB_HIER_CHILD_SLOTS)]],
                             device GPUSceneCounters*  counters     [[buffer(SB_HIER_COUNTERS)]],
                             uint t    [[thread_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]]) {
    // Entries to process: the header count clamped to the capacity (overflowed
    // appends were dropped).  All lanes of a SIMD-group reach the SIMD
    // operations below: no early return.
    const uint n = min(queueIn[GPU_QUEUE_WORD_COUNT], queueIn[GPU_QUEUE_WORD_CAPACITY]);
    const bool active = t < n;

    uint first = 0, nchild = 0, updated = 0;
    if (active) {
        const uint slot = queueIn[GPU_QUEUE_WORD_ENTRIES + t];
        if (params.level >= 1u) {
            const GPUTransformNode node = nodes[slot];
            float parent[16], local[16], w[16];
            for (uint i = 0; i < 16; ++i) {
                parent[i] = instances[node.parentSlot].modelMatrix[i];
                local[i]  = node.local[i];
            }
            mat4Mul(parent, local, w);
            for (uint i = 0; i < 16; ++i) instances[slot].modelMatrix[i] = w[i];
            updated = 1u;
        }
        first  = childOffsets[slot];
        nchild = childOffsets[slot + 1u] - first;
    }

    // The last level has no queue after it (queue index SCENE_MAX_LEVELS does not exist).
    const bool append = params.level + 1u < SCENE_MAX_LEVELS;
    if (kDropAppend && lane == 0u && nchild > 0u) nchild -= 1u; // negative control only

    const uint nodesUpdated = simd_sum(updated);
    if (lane == 0u && nodesUpdated > 0u)
        atomic_fetch_add_explicit((device atomic_uint*)&counters->nodesUpdated, nodesUpdated, memory_order_relaxed);
    if (!append) return; // uniform across the dispatch

    // One atomic per SIMD-group (gpuQueueReserve): lanes take their slice of queueOut.
    const GPUQueueSlice slice = gpuQueueReserve(queueOut, nchild, lane);
    for (uint k = 0; k < slice.stored; ++k) gpuQueuePut(queueOut, slice, k, childSlots[first + k]);
    const uint dropped = gpuQueueFinish(queueOut, slice, nchild, lane);
    if (lane == 0u && dropped > 0u)
        atomic_fetch_add_explicit((device atomic_uint*)&counters->queueOverflow, dropped, memory_order_relaxed);
}
