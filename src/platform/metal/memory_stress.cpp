#include "platform/metal/memory_stress.h"
#include "platform/metal/gpu_memory.h"
#include "platform/metal/metal_context.h"
#include "core/log.h"

#include <algorithm>
#include <array>
#include <random>
#include <vector>

namespace phosphor {

namespace {

struct Live {
    MTL::Resource* resource;
    MemoryCategory category;
};

u64 deviceBytes(MetalContext& context) {
    return context.device()->currentAllocatedSize();
}

u32 heapCount(const GpuMemory& memory) {
    std::vector<GpuMemory::HeapStats> heaps;
    memory.heapStats(heaps);
    return static_cast<u32>(heaps.size());
}

} // namespace

MemoryStressResult runMemoryStress(MetalContext& context, u32 cycles, u32 warmup) {
    constexpr size_t kMaxLive = 64;
    GpuMemory& memory = context.memory();
    std::mt19937 rng(20260928);
    std::vector<Live> live;
    live.reserve(kMaxLive);
    MemoryStressResult result;

    auto releaseAll = [&] {
        for (const Live& l : live) memory.release(l.resource, l.category);
        live.clear();
        context.collectGarbage();
    };

    context.collectGarbage();
    result.baselineBytes = deviceBytes(context);
    std::array<u32, MEMORY_CATEGORY_COUNT> baselineCounts{};
    for (u32 c = 0; c < MEMORY_CATEGORY_COUNT; ++c) {
        baselineCounts[c] = memory.stats(static_cast<MemoryCategory>(c)).count;
    }

    for (u32 cycle = 0; cycle < cycles; ++cycle) {
        if (live.size() == kMaxLive || (!live.empty() && rng() % 3 == 0)) {
            const size_t i = rng() % live.size();
            memory.release(live[i].resource, live[i].category);
            live[i] = live.back();
            live.pop_back();
        }

        switch (rng() % 3) {
        case 0: { // private buffer: placement heap
            const u64 size = u64{4096} << (rng() % 12); // 4 KiB .. 8 MiB
            live.push_back({memory.newBuffer(size, MTL::ResourceStorageModePrivate, MemoryCategory::Geometry,
                                             "Stress buffer"),
                            MemoryCategory::Geometry});
            break;
        }
        case 1: { // private texture: placement heap, sometimes mipmapped
            const u32 side = 64u << (rng() % 6); // 64 .. 2048
            MTL::TextureDescriptor* desc = MTL::TextureDescriptor::texture2DDescriptor(
                MTL::PixelFormatRGBA8Unorm, side, side, rng() % 2 == 0);
            desc->setUsage(MTL::TextureUsageShaderRead);
            desc->setStorageMode(MTL::StorageModePrivate);
            live.push_back({memory.newTexture(desc, MemoryCategory::Textures, "Stress texture"),
                            MemoryCategory::Textures});
            break;
        }
        default: { // shared buffer: standalone, residency add/remove
            const u64 size = u64{4096} << (rng() % 9); // 4 KiB .. 1 MiB
            live.push_back({memory.newBuffer(size, MTL::ResourceStorageModeShared, MemoryCategory::Other,
                                             "Stress shared buffer"),
                            MemoryCategory::Other});
            break;
        }
        }
        if (!live.back().resource) {
            LOG_ERROR("Memory stress: allocation failed at cycle %u", cycle);
            live.pop_back();
        }

        result.heapsPeak = std::max(result.heapsPeak, heapCount(memory));
        // No frames run during the test: collect deferred releases periodically.
        if (cycle % 100 == 99) context.collectGarbage();
        if (cycle + 1 == warmup) {
            releaseAll();
            result.warmBytes = deviceBytes(context);
        }
    }

    releaseAll();
    result.finalBytes = deviceBytes(context);
    memory.trimEmptyHeaps(/*keepSpare*/ false);
    context.commitResidency(); // removed heaps are freed once the sets drop them
    result.trimmedBytes = deviceBytes(context);
    result.countsRestored = true;
    for (u32 c = 0; c < MEMORY_CATEGORY_COUNT; ++c) {
        result.countsRestored &= memory.stats(static_cast<MemoryCategory>(c)).count == baselineCounts[c];
    }
    result.passed = result.trimmedBytes == result.baselineBytes && result.countsRestored;
    return result;
}

} // namespace phosphor
