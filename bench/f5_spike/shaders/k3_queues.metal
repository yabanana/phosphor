// F5-K3 helper kernels (bench/f5_spike/k3_transforms.cpp).  Compiled through the
// bench's include expander (renderer/ headers inlined).
//
//   k3_queue_clear  stand-in for scene_queue_clear (shaders/gpu_scene.metal,
//                   another task): same layout and effect (queues 1.. of one
//                   buffer: count/overflow/groups cleared, capacity kept;
//                   counters cleared).
//   k3_synth_stage  synthetic producer/consumer of F5.4: one thread per entry
//                   of queueIn; appends a DATA-DEPENDENT number of entries
//                   (0 .. mod-1, a quarter of the entries append none) to
//                   queueOut with the production append primitive
//                   (gpuQueueReserve / gpuQueuePut / gpuQueueFinish).
#include <metal_stdlib>
#include "renderer/gpu_scene_layout.h"
#include "renderer/gpu_queue.h"

using namespace metal;
using namespace phosphor;

kernel void k3_queue_clear(constant GPUQueueClearParams& p [[buffer(0)]],
                           device uchar*                 queues   [[buffer(SB_CLEAR_QUEUES)]],
                           device GPUSceneCounters*      counters [[buffer(SB_CLEAR_COUNTERS)]],
                           uint t [[thread_position_in_grid]]) {
    if (t == 0u) {
        device uint* c = (device uint*)counters;
        for (uint i = 0; i < 8u; ++i) c[i] = 0u;
    }
    if (t >= p.queues) return;
    device uint* q = (device uint*)(queues + ulong(t) * p.strideBytes);
    q[GPU_QUEUE_WORD_COUNT]    = 0u;
    q[GPU_QUEUE_WORD_OVERFLOW] = 0u;
    q[GPU_QUEUE_WORD_GROUPS + 0] = 0u;
    q[GPU_QUEUE_WORD_GROUPS + 1] = 1u;
    q[GPU_QUEUE_WORD_GROUPS + 2] = 1u;
}

struct K3Synth {
    uint stage;
    uint mod;
    uint pad[2];
};

static uint k3hash(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

// The CPU model in k3_transforms.cpp has the same two functions.
static uint k3count(uint v, uint stage, uint mod) {
    const uint h = k3hash(v * 0x9E3779B1U + stage * 0x85EBCA6BU + 12345U);
    return (h & 3u) == 0u ? 0u : (h >> 2) % mod;
}
static uint k3value(uint v, uint j, uint stage) { return k3hash(v ^ (j * 0x27d4eb2fU) ^ (stage << 24)); }

kernel void k3_synth_stage(constant K3Synth&      p        [[buffer(0)]],
                           device const uint*     queueIn  [[buffer(SB_HIER_QUEUE_IN)]],
                           device atomic_uint*    queueOut [[buffer(SB_HIER_QUEUE_OUT)]],
                           uint t    [[thread_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]]) {
    const uint n = min(queueIn[GPU_QUEUE_WORD_COUNT], queueIn[GPU_QUEUE_WORD_CAPACITY]);
    uint v = 0, count = 0;
    if (t < n) {
        v = queueIn[GPU_QUEUE_WORD_ENTRIES + t];
        count = k3count(v, p.stage, p.mod);
    }
    const GPUQueueSlice slice = gpuQueueReserve(queueOut, count, lane);
    for (uint k = 0; k < slice.stored; ++k) gpuQueuePut(queueOut, slice, k, k3value(v, k, p.stage));
    gpuQueueFinish(queueOut, slice, count, lane);
}
