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

#include <cstring>
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
// Word indices of the header fields and of the first entry (kernels see a
// queue as a u32 / atomic_uint array: count and overflow are atomics).
constant u32 GPU_QUEUE_WORD_COUNT    = 0;
constant u32 GPU_QUEUE_WORD_CAPACITY = 1;
constant u32 GPU_QUEUE_WORD_OVERFLOW = 2;
constant u32 GPU_QUEUE_WORD_GROUPS   = 4;
constant u32 GPU_QUEUE_WORD_ENTRIES  = 8;

/// Where the calling lane may store its entries: [lo, lo + stored) of the
/// queue (stored < n when the queue is full: the rest is dropped).
struct GPUQueueSlice {
    u32 lo;
    u32 stored;
};

/// SIMD-aggregated append, step 1: reserve room for this lane's `n` entries
/// with ONE atomic per SIMD-group (simd_prefix_exclusive_sum for the lane
/// offsets, S-SIMD-1).  Every lane of the SIMD-group must call it (n may be
/// 0), no lane may have returned earlier.  `q` is the queue buffer viewed as
/// an atomic word array.
inline GPUQueueSlice gpuQueueReserve(device metal::atomic_uint* q, u32 n, u32 lane) {
    const u32 prefix = metal::simd_prefix_exclusive_sum(n);
    const u32 total  = metal::simd_sum(n);
    u32 base = 0;
    if (lane == 0u && total > 0u)
        base = metal::atomic_fetch_add_explicit(&q[GPU_QUEUE_WORD_COUNT], total, metal::memory_order_relaxed);
    base = metal::simd_broadcast_first(base);
    const u32 capacity = metal::atomic_load_explicit(&q[GPU_QUEUE_WORD_CAPACITY], metal::memory_order_relaxed);
    GPUQueueSlice s;
    s.lo     = base + prefix;
    s.stored = s.lo >= capacity ? 0u : metal::min(n, capacity - s.lo);
    return s;
}

/// Step 2: store entry k (< slice.stored) of this lane.
inline void gpuQueuePut(device metal::atomic_uint* q, GPUQueueSlice s, u32 k, u32 value) {
    metal::atomic_store_explicit(&q[GPU_QUEUE_WORD_ENTRIES + s.lo + k], value, metal::memory_order_relaxed);
}

/// Step 3 (all lanes, uniform): count the entries dropped by a full queue in
/// its header (one atomic per SIMD-group); returns the dropped entries of the
/// whole SIMD-group (the same value on every lane) so the caller can add them
/// to a global counter.
inline u32 gpuQueueFinish(device metal::atomic_uint* q, GPUQueueSlice s, u32 n, u32 lane) {
    const u32 dropped = metal::simd_sum(n - s.stored);
    if (lane == 0u && dropped > 0u)
        metal::atomic_fetch_add_explicit(&q[GPU_QUEUE_WORD_OVERFLOW], dropped, metal::memory_order_relaxed);
    return dropped;
}
#else
inline constexpr u32 GPU_QUEUE_ARGS_OFFSET = 16;
inline constexpr u32 GPU_QUEUE_WORD_COUNT    = 0;
inline constexpr u32 GPU_QUEUE_WORD_CAPACITY = 1;
inline constexpr u32 GPU_QUEUE_WORD_OVERFLOW = 2;
inline constexpr u32 GPU_QUEUE_WORD_GROUPS   = 4;
inline constexpr u32 GPU_QUEUE_WORD_ENTRIES  = 8;
static_assert(GPU_QUEUE_WORD_ENTRIES * sizeof(u32) == sizeof(GPUQueueHeader), "entries follow the header");
static_assert(GPU_QUEUE_WORD_GROUPS * sizeof(u32) == GPU_QUEUE_ARGS_OFFSET, "groups offset");
/// Bytes of a queue buffer with room for `capacity` entries.
inline constexpr u64 gpuQueueBytes(u32 capacity) { return sizeof(GPUQueueHeader) + u64(capacity) * sizeof(u32); }
/// Threadgroups the args kernel writes for `count` entries (count clamped to
/// the capacity, `threads` per group).
inline constexpr u32 gpuQueueGroups(u32 count, u32 capacity, u32 threads) {
    const u32 n = count < capacity ? count : capacity;
    return (n + threads - 1) / threads;
}

/// Writes a CPU-filled queue (header + entries) into `dst` (at least
/// gpuQueueBytes(capacity) bytes) with the same semantics as the GPU: `count`
/// is the number of entries offered (may exceed the capacity), only
/// min(count, capacity) are stored and `overflow` = the dropped ones; `groups`
/// is what the args kernel would write, so the consumer can be dispatched
/// indirectly without an args kernel for this queue (the dirty roots, queue 0
/// in the frame upload ring).  Returns the bytes written (header + stored
/// entries).
inline u64 gpuQueueWrite(void* dst, const u32* entries, u32 count, u32 capacity, u32 threads) {
    auto* words = static_cast<u32*>(dst);
    const u32 stored = count < capacity ? count : capacity;
    GPUQueueHeader h{};
    h.count    = count;
    h.capacity = capacity;
    h.overflow = count - stored;
    h.groups[0] = gpuQueueGroups(count, capacity, threads);
    h.groups[1] = 1;
    h.groups[2] = 1;
    std::memcpy(words, &h, sizeof(h));
    for (u32 i = 0; i < stored; ++i) words[GPU_QUEUE_WORD_ENTRIES + i] = entries[i];
    return gpuQueueBytes(0) + u64(stored) * sizeof(u32);
}

/// Header of an empty queue (what scene_queue_clear leaves: capacity kept,
/// groups (0, 1, 1)).
inline GPUQueueHeader gpuQueueEmptyHeader(u32 capacity) {
    GPUQueueHeader h{};
    h.capacity  = capacity;
    h.groups[1] = 1;
    h.groups[2] = 1;
    return h;
}
#endif

} // namespace phosphor
