#include "core/memory/tlsf_allocator.h"

#include <doctest/doctest.h>

#include <algorithm>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace phosphor;

namespace {

void requireValid(const TlsfAllocator& a) {
    std::string error;
    const bool ok = a.validate(&error);
    INFO(error);
    REQUIRE(ok);
}

} // namespace

TEST_CASE("tlsf: allocate, free and merge back to one block") {
    TlsfAllocator a(1 << 20);
    const auto x = a.allocate(1000);
    const auto y = a.allocate(5000);
    const auto z = a.allocate(1);
    REQUIRE(x.valid());
    REQUIRE(y.valid());
    REQUIRE(z.valid());
    CHECK(x.offset == 0);
    CHECK(y.offset >= x.offset + 1000);
    CHECK(z.offset >= y.offset + 5000);
    CHECK(a.stats().allocationCount == 3);
    requireValid(a);

    // Free the middle first: its neighbours are used, so it stays separate.
    a.free(y.handle);
    requireValid(a);
    CHECK(a.stats().freeBlockCount == 2);
    a.free(x.handle);
    a.free(z.handle);
    requireValid(a);
    const auto s = a.stats();
    CHECK(s.allocationCount == 0);
    CHECK(s.usedBytes == 0);
    CHECK(s.freeBlockCount == 1);
    CHECK(s.largestFreeBlock == a.capacity());
    CHECK(s.fragmentation() == doctest::Approx(0.0f));
}

TEST_CASE("tlsf: alignment from 4 bytes to 64 KiB") {
    TlsfAllocator a(64ull << 20);
    std::vector<TlsfAllocator::Allocation> live;
    for (u64 align = 4; align <= 64 * 1024; align <<= 1) {
        // An odd-sized allocation first so the next offset is misaligned.
        live.push_back(a.allocate(3));
        const auto aligned = a.allocate(777, align);
        REQUIRE(aligned.valid());
        CHECK(aligned.offset % align == 0);
        live.push_back(aligned);
        requireValid(a);
    }
    for (const auto& l : live) a.free(l.handle);
    requireValid(a);
    CHECK(a.stats().freeBlockCount == 1);
}

TEST_CASE("tlsf: rejects impossible requests and ignores bad frees") {
    TlsfAllocator a(4096);
    CHECK_FALSE(a.allocate(0).valid());
    CHECK_FALSE(a.allocate(4097).valid());
    CHECK_FALSE(a.allocate(16, 3).valid()); // alignment not a power of two
    const auto whole = a.allocate(4096);
    REQUIRE(whole.valid());
    CHECK_FALSE(a.allocate(1).valid()); // full
    a.free(TlsfAllocator::INVALID_HANDLE);
    a.free(12345);
    a.free(whole.handle);
    a.free(whole.handle); // double free is ignored
    requireValid(a);
    CHECK(a.stats().usedBytes == 0);
}

TEST_CASE("tlsf: a freed block is reused") {
    TlsfAllocator a(1 << 16);
    const auto x = a.allocate(1024);
    const auto y = a.allocate(1024);
    a.free(x.handle);
    const auto again = a.allocate(1024);
    REQUIRE(again.valid());
    CHECK(again.offset == x.offset);
    a.free(y.handle);
    a.free(again.handle);
    requireValid(a);
}

TEST_CASE("tlsf: very large ranges") {
    const u64 capacity = 1ull << 40; // 1 TiB of address range, nothing is touched
    TlsfAllocator a(capacity);
    const auto big = a.allocate(capacity / 2, 1ull << 16);
    REQUIRE(big.valid());
    const auto small = a.allocate(64);
    REQUIRE(small.valid());
    requireValid(a);
    a.free(big.handle);
    a.free(small.handle);
    requireValid(a);
    CHECK(a.stats().largestFreeBlock == capacity);
}

TEST_CASE("tlsf: reset releases everything") {
    TlsfAllocator a(1 << 20);
    for (int i = 0; i < 100; ++i) (void)a.allocate(100 + i);
    a.reset();
    requireValid(a);
    CHECK(a.stats().allocationCount == 0);
    CHECK(a.stats().largestFreeBlock == a.capacity());
}

TEST_CASE("tlsf: randomized allocate/free keeps every invariant") {
    // Deterministic fuzz: 100k operations, sizes from bytes to megabytes,
    // alignments up to 64 KiB, checked against an independent overlap map.
    constexpr u64 capacity = 256ull << 20;
    TlsfAllocator a(capacity);
    std::mt19937_64 rng(0x9E3779B97F4A7C15ull);
    std::vector<TlsfAllocator::Allocation> live;
    std::map<u64, u64> ranges; // offset -> end, of live allocations
    u32 failures = 0;

    for (int op = 0; op < 100000; ++op) {
        const bool doAlloc = live.empty() || (rng() % 100) < 55;
        if (doAlloc) {
            const u32 bucket = static_cast<u32>(rng() % 3);
            const u64 size = bucket == 0 ? 1 + rng() % 256
                           : bucket == 1 ? 1 + rng() % (64 << 10)
                                         : 1 + rng() % (4 << 20);
            const u64 align = u64{1} << (rng() % 17);
            const auto alloc = a.allocate(size, align);
            if (!alloc.valid()) {
                ++failures;
                continue;
            }
            REQUIRE(alloc.offset % align == 0);
            REQUIRE(alloc.offset + size <= capacity);
            // No overlap with any live allocation.
            auto next = ranges.lower_bound(alloc.offset);
            if (next != ranges.end()) REQUIRE(next->first >= alloc.offset + size);
            if (next != ranges.begin()) REQUIRE(std::prev(next)->second <= alloc.offset);
            ranges[alloc.offset] = alloc.offset + size;
            live.push_back(alloc);
        } else {
            const size_t i = rng() % live.size();
            ranges.erase(live[i].offset);
            a.free(live[i].handle);
            live[i] = live.back();
            live.pop_back();
        }
        if (op % 1000 == 0) requireValid(a);
    }
    requireValid(a);
    CHECK(a.stats().allocationCount == live.size());
    // The fuzz should exercise exhaustion, but mostly succeed.
    CHECK(failures < 100000 / 10);

    for (const auto& l : live) a.free(l.handle);
    requireValid(a);
    CHECK(a.stats().freeBlockCount == 1);
    CHECK(a.stats().usedBytes == 0);
}

TEST_CASE("tlsf: no metadata growth in steady state") {
    // After warm-up, alloc/free cycles reuse node slots (O7 for the CPU side).
    TlsfAllocator a(64 << 20);
    std::vector<TlsfAllocator::Allocation> live;
    auto cycle = [&] {
        for (int i = 0; i < 64; ++i) live.push_back(a.allocate(4096 + i * 256, 256));
        for (const auto& l : live) a.free(l.handle);
        live.clear();
    };
    cycle();
    const size_t nodes = a.metadataNodeCount();
    for (int i = 0; i < 10000; ++i) cycle();
    CHECK(a.metadataNodeCount() == nodes);
    CHECK(a.stats().usedBytes == 0);
    CHECK(a.stats().freeBlockCount == 1);
    requireValid(a);
}
