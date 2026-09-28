#include "platform/metal/memory_stress.h"
#include "platform/metal/gpu_memory.h"
#include "platform/metal/metal_context.h"
#include "platform/metal/transient_heap.h"
#include "core/log.h"

#include <algorithm>
#include <cstring>
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

TransientAliasResult runTransientAliasTest(MetalContext& context) {
    constexpr u64 kBufferBytes = 1ull << 20;
    constexpr u32 kSide = 256;
    constexpr u64 kTextureBytes = u64{kSide} * kSide * 4;
    GpuMemory& memory = context.memory();
    TransientAliasResult result;

    context.collectGarbage();
    TransientHeap heap(context);

    MTL::TextureDescriptor* desc =
        MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatRGBA8Unorm, kSide, kSide, false);
    desc->setUsage(MTL::TextureUsageShaderRead);
    desc->setStorageMode(MTL::StorageModePrivate);
    const MTL::SizeAndAlign bufferSa  = heap.sizeAndAlign(kBufferBytes);
    const MTL::SizeAndAlign textureSa = heap.sizeAndAlign(desc);
    // Buffers alias each other at 0; textures alias each other after them.
    result.textureOffset = (bufferSa.size + textureSa.align - 1) / textureSa.align * textureSa.align;
    if (!heap.reserve(result.textureOffset + textureSa.size)) return result;
    result.heapSize = heap.size();

    MTL::Buffer*  a  = heap.createBuffer(kBufferBytes, 0, "Alias buffer A");
    MTL::Buffer*  b  = heap.createBuffer(kBufferBytes, 0, "Alias buffer B");
    MTL::Texture* ta = heap.createTexture(desc, result.textureOffset, "Alias texture A");
    MTL::Texture* tb = heap.createTexture(desc, result.textureOffset, "Alias texture B");
    MTL::Buffer* readA  = memory.newBuffer(kBufferBytes, MTL::ResourceStorageModeShared, MemoryCategory::Other, "Read A");
    MTL::Buffer* readB  = memory.newBuffer(kBufferBytes, MTL::ResourceStorageModeShared, MemoryCategory::Other, "Read B");
    MTL::Buffer* readA2 = memory.newBuffer(kBufferBytes, MTL::ResourceStorageModeShared, MemoryCategory::Other, "Read A again");
    MTL::Buffer* readTA = memory.newBuffer(kTextureBytes, MTL::ResourceStorageModeShared, MemoryCategory::Other, "Read TA");
    MTL::Buffer* readTB = memory.newBuffer(kTextureBytes, MTL::ResourceStorageModeShared, MemoryCategory::Other, "Read TB");
    if (!a || !b || !ta || !tb) return result;

    // Two different patterns for the textures.
    const UploadRing::Slice p1 = context.stagingAllocate(kTextureBytes);
    const UploadRing::Slice p2 = context.stagingAllocate(kTextureBytes);
    for (u64 i = 0; i < kTextureBytes; ++i) {
        p1.cpu[i] = static_cast<u8>(i * 7 + 1);
        p2.cpu[i] = static_cast<u8>(255 - i * 3);
    }

    const MTL::Size extent = MTL::Size::Make(kSide, kSide, 1);
    const MTL::Origin zero = MTL::Origin::Make(0, 0, 0);
    const auto blitToBlit = [](MTL4::ComputeCommandEncoder* enc, MTL4::VisibilityOptions visibility) {
        enc->barrierAfterEncoderStages(MTL::StageBlit, MTL::StageBlit, visibility);
    };
    const MTL4::VisibilityOptions alias =
        static_cast<MTL4::VisibilityOptions>(MTL4::VisibilityOptionDevice | MTL4::VisibilityOptionResourceAlias);
    context.submitAndWait([&](MTL4::ComputeCommandEncoder* enc) {
        enc->fillBuffer(a, NS::Range::Make(0, kBufferBytes), 0xAA);
        blitToBlit(enc, MTL4::VisibilityOptionDevice);
        enc->copyFromBuffer(a, 0, readA, 0, kBufferBytes);
        blitToBlit(enc, alias); // B takes over A's memory
        enc->fillBuffer(b, NS::Range::Make(0, kBufferBytes), 0x55);
        blitToBlit(enc, MTL4::VisibilityOptionDevice);
        enc->copyFromBuffer(b, 0, readB, 0, kBufferBytes);
        // Diagnostic only (undefined for textures): A's view of the shared bytes.
        blitToBlit(enc, alias);
        enc->copyFromBuffer(a, 0, readA2, 0, kBufferBytes);

        enc->copyFromBuffer(p1.buffer, p1.offset, kSide * 4, 0, extent, ta, 0, 0, zero);
        blitToBlit(enc, MTL4::VisibilityOptionDevice);
        enc->copyFromTexture(ta, 0, 0, zero, extent, readTA, 0, kSide * 4, kTextureBytes);
        blitToBlit(enc, alias); // TB takes over TA's memory
        enc->copyFromBuffer(p2.buffer, p2.offset, kSide * 4, 0, extent, tb, 0, 0, zero);
        blitToBlit(enc, MTL4::VisibilityOptionDevice);
        enc->copyFromTexture(tb, 0, 0, zero, extent, readTB, 0, kSide * 4, kTextureBytes);
    });

    const auto allBytes = [](MTL::Buffer* buf, u8 value) {
        const auto* p = static_cast<const u8*>(buf->contents());
        return std::all_of(p, p + buf->length(), [&](u8 v) { return v == value; });
    };
    result.buffersOk  = allBytes(readA, 0xAA) && allBytes(readB, 0x55);
    result.texturesOk = std::memcmp(readTA->contents(), p1.cpu, kTextureBytes) == 0 &&
                        std::memcmp(readTB->contents(), p2.cpu, kTextureBytes) == 0;
    result.memoryShared = allBytes(readA2, 0x55);
    result.passed = result.buffersOk && result.texturesOk && result.memoryShared;

    context.flushUploads(); // recycle the staging slices
    for (MTL::Resource* r : {static_cast<MTL::Resource*>(a), static_cast<MTL::Resource*>(b),
                             static_cast<MTL::Resource*>(ta), static_cast<MTL::Resource*>(tb)}) {
        heap.release(r);
    }
    for (MTL::Buffer* r : {readA, readB, readA2, readTA, readTB}) memory.release(r, MemoryCategory::Other);
    return result;
}

} // namespace phosphor
