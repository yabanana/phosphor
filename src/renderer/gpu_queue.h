#pragma once

// ---------------------------------------------------------------------------
// F5.4 -- GPU producer/consumer queues and indirect dispatch chains (Metal
// has no work graphs).  Shared by C++ and MSL like gpu_types.h.
//
// A queue lives in one device buffer: a 32-byte header followed by
// `capacity` u32 entries.  Producers append with one atomic per SIMD-group
// (simd_prefix_exclusive_sum for the lane offsets, S-SIMD-1); an append past
// the capacity is dropped and counted in `overflow` (never written out of
// bounds).  A one-thread "args" kernel (queue_args in shaders/transforms.metal)
// turns `count` into the dispatchThreadgroups arguments stored in the header,
// so the consumer is encoded by the CPU up front with
// dispatchThreadgroups(headerAddress + GPU_QUEUE_ARGS_OFFSET, tptg) and no
// read-back.  Barriers (spike S5): producer -> args -> consumer are
// Dispatch -> Dispatch (encoder barrier inside one compute encoder).
// Each frame clears the queues it uses (count, overflow, groups; capacity
// stays) with one kernel before the first producer (scene_queue_clear); a
// queue written by the CPU (the dirty roots, in the frame upload ring) gets
// its header, groups included, from the CPU.
// ---------------------------------------------------------------------------

#ifdef __METAL_VERSION__
#include <metal_stdlib>
#define PHOSPHOR_QUEUE_STATIC_ASSERT(cond, msg)
#else
#include "core/types.h"
#define PHOSPHOR_QUEUE_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#endif

namespace phosphor {

#ifdef __METAL_VERSION__
using u32 = uint;
#endif

struct GPUQueueHeader {
    u32 count;      // entries appended (may exceed capacity: overflow)
    u32 capacity;   // entries that fit after the header
    u32 overflow;   // appends dropped
    u32 pad0;
    u32 groups[3];  // dispatchThreadgroups arguments (MTLDispatchThreadgroupsIndirectArguments)
    u32 pad1;
};
PHOSPHOR_QUEUE_STATIC_ASSERT(sizeof(GPUQueueHeader) == 32, "GPUQueueHeader layout");

#ifdef __METAL_VERSION__
constant u32 GPU_QUEUE_ARGS_OFFSET = 16;  // offset of `groups` in the header
#else
inline constexpr u32 GPU_QUEUE_ARGS_OFFSET = 16;
/// Bytes of a queue buffer with room for `capacity` entries.
inline constexpr u64 gpuQueueBytes(u32 capacity) { return sizeof(GPUQueueHeader) + u64(capacity) * sizeof(u32); }
/// Threadgroups the args kernel writes for `count` entries (count clamped to
/// the capacity, `threads` per group).
inline constexpr u32 gpuQueueGroups(u32 count, u32 capacity, u32 threads) {
    const u32 n = count < capacity ? count : capacity;
    return (n + threads - 1) / threads;
}
#endif

} // namespace phosphor
