// F5.4: GPU queue layout (renderer/gpu_queue.h) and its host helpers.

#include <doctest/doctest.h>

#include "renderer/gpu_queue.h"
#include "renderer/gpu_scene_layout.h"

#include <cstddef>
#include <cstring>
#include <vector>

using namespace phosphor;

TEST_CASE("GPUQueueHeader layout") {
    CHECK(sizeof(GPUQueueHeader) == 32);
    CHECK(offsetof(GPUQueueHeader, count) == 0);
    CHECK(offsetof(GPUQueueHeader, capacity) == 4);
    CHECK(offsetof(GPUQueueHeader, overflow) == 8);
    CHECK(offsetof(GPUQueueHeader, groups) == GPU_QUEUE_ARGS_OFFSET);
    CHECK(GPU_QUEUE_WORD_GROUPS * 4 == offsetof(GPUQueueHeader, groups));
    CHECK(GPU_QUEUE_WORD_ENTRIES * 4 == sizeof(GPUQueueHeader));
    // Indirect arguments = three u32 (MTLDispatchThreadgroupsIndirectArguments).
    CHECK(sizeof(GPUQueueHeader::groups) == 12);
}

TEST_CASE("gpuQueueBytes") {
    CHECK(gpuQueueBytes(0) == 32);
    CHECK(gpuQueueBytes(1) == 36);
    CHECK(gpuQueueBytes(1024) == 32 + 4096);
    CHECK(gpuQueueBytes(1u << 30) == 32ull + (1ull << 32)); // 64-bit: does not wrap
    CHECK((gpuQueueBytes(1016) % 32) == 0); // a stride of scene_queue_clear must be a multiple of 32
}

TEST_CASE("gpuQueueGroups edge cases") {
    const u32 T = SCENE_HIER_GROUP;
    CHECK(gpuQueueGroups(0, 100, T) == 0);
    CHECK(gpuQueueGroups(1, 100, T) == 1);
    CHECK(gpuQueueGroups(T, 1000, T) == 1);
    CHECK(gpuQueueGroups(T + 1, 1000, T) == 2);
    CHECK(gpuQueueGroups(100, 100, T) == 2);
    CHECK(gpuQueueGroups(1000, 100, T) == 2); // count > capacity: clamped
    CHECK(gpuQueueGroups(0xFFFFFFFFu, 128, T) == 2);
    CHECK(gpuQueueGroups(5, 0, T) == 0);
    CHECK(gpuQueueGroups(10, 10, 1) == 10);
}

TEST_CASE("gpuQueueWrite: CPU-filled queue 0") {
    const u32 entries[5] = {7, 8, 9, 10, 11};
    std::vector<u32> buf(GPU_QUEUE_WORD_ENTRIES + 8, 0xDEADBEEFu);
    const u64 bytes = gpuQueueWrite(buf.data(), entries, 5, 8, SCENE_HIER_GROUP);
    CHECK(bytes == 32 + 5 * 4);
    GPUQueueHeader h;
    std::memcpy(&h, buf.data(), sizeof(h));
    CHECK(h.count == 5);
    CHECK(h.capacity == 8);
    CHECK(h.overflow == 0);
    CHECK(h.groups[0] == 1);
    CHECK(h.groups[1] == 1);
    CHECK(h.groups[2] == 1);
    for (u32 i = 0; i < 5; ++i) CHECK(buf[GPU_QUEUE_WORD_ENTRIES + i] == entries[i]);
    CHECK(buf[GPU_QUEUE_WORD_ENTRIES + 5] == 0xDEADBEEFu); // nothing written past the entries

    SUBCASE("more entries than the capacity: stored up to the capacity, the rest counted") {
        std::vector<u32> big(GPU_QUEUE_WORD_ENTRIES + 3, 0xDEADBEEFu); // exact size: an overrun would be caught by ASan
        const u64 b = gpuQueueWrite(big.data(), entries, 5, 3, SCENE_HIER_GROUP);
        CHECK(b == 32 + 3 * 4);
        std::memcpy(&h, big.data(), sizeof(h));
        CHECK(h.count == 5);
        CHECK(h.overflow == 2);
        CHECK(h.groups[0] == 1);
        CHECK(big[GPU_QUEUE_WORD_ENTRIES + 2] == 9);
    }
    SUBCASE("empty") {
        gpuQueueWrite(buf.data(), nullptr, 0, 8, SCENE_HIER_GROUP);
        std::memcpy(&h, buf.data(), sizeof(h));
        CHECK(h.count == 0);
        CHECK(h.groups[0] == 0);
        CHECK(h.groups[1] == 1);
    }
}

TEST_CASE("gpuQueueEmptyHeader") {
    const GPUQueueHeader h = gpuQueueEmptyHeader(77);
    CHECK(h.count == 0);
    CHECK(h.capacity == 77);
    CHECK(h.overflow == 0);
    CHECK(h.groups[0] == 0);
    CHECK(h.groups[1] == 0);
    CHECK(h.groups[2] == 0);
}
