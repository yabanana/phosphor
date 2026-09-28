#pragma once

#include "core/types.h"

namespace phosphor {

class MetalContext;

struct MemoryStressResult {
    u64  baselineBytes = 0; // device allocation before the test
    u64  warmBytes     = 0; // after the warm-up cycles, everything released (heaps kept)
    u64  finalBytes    = 0; // after all cycles, everything released (heaps kept)
    u64  trimmedBytes  = 0; // after trimming every empty heap
    u32  heapsPeak     = 0;
    bool countsRestored = false; // per-category allocation counts back to baseline
    bool passed        = false; // trimmedBytes == baselineBytes and counts restored
};

struct TransientAliasResult {
    u64  heapSize      = 0;
    u64  textureOffset = 0;
    bool buffersOk     = false; // aliased buffers read back what was written
    bool texturesOk    = false; // aliased textures read back what was written
    bool memoryShared  = false; // A re-read after writing B shows B's bytes: real overlap
    bool passed        = false;
};

/// F1.1: place two buffers and two textures at the same offsets of a
/// TransientHeap, write/read A, alias barrier, write/read B, and check both
/// read-backs.  Must run between frames.
TransientAliasResult runTransientAliasTest(MetalContext& context);

/// F1.6: create and destroy `cycles` mixed GPU resources through GpuMemory
/// (private buffers and textures in placement heaps, shared buffers
/// standalone), at most 64 alive at once, deterministic sizes.  Passes when,
/// after releasing everything and trimming every empty heap, device memory
/// and allocation counts are exactly back to their starting values: nothing
/// leaked.  (Between warm-up and the end the heap pool may grow with the peak
/// of live bytes; heaps are kept until a trim by design.)
/// Must run between frames (it waits for the GPU and collects garbage).
MemoryStressResult runMemoryStress(MetalContext& context, u32 cycles, u32 warmup = 1000);

} // namespace phosphor
