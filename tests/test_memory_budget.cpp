#include "core/memory/memory_budget.h"

#include <doctest/doctest.h>

#include <string>

using namespace phosphor;

TEST_CASE("memory budget: tier detection from the device name") {
    CHECK(detectTier("Apple M3", false).tier == HardwareTier::T0Base);
    CHECK(detectTier("Apple M4 Pro", false).tier == HardwareTier::T1Pro);
    CHECK(detectTier("Apple M3 Max", false).tier == HardwareTier::T2Max);
    CHECK(detectTier("Apple M2 Ultra", false).tier == HardwareTier::T2Max);
    CHECK(detectTier("Apple A18 Pro GPU", false).tier == HardwareTier::T0Base); // roadmap: A18 Pro is T0
    CHECK_FALSE(detectTier("Apple A19 Pro GPU", true).neural);
    CHECK(detectTier("Something Maxwell", false).tier == HardwareTier::T0Base); // whole words only
    CHECK(detectTier("", true).tier == HardwareTier::T0Base);

    // T3 Neural: Apple10 on Pro/Max only.
    CHECK(detectTier("Apple M5 Max", true).neural);
    CHECK(detectTier("Apple M5 Pro", true).neural);
    CHECK_FALSE(detectTier("Apple M5", true).neural);
    CHECK_FALSE(detectTier("Apple M4 Max", false).neural);
    CHECK(std::string(tierName(detectTier("Apple M5 Max", true))) == "T2 Max + T3 Neural");
}

TEST_CASE("memory budget: category shares cover the engine budget") {
    float total = 0.0f;
    for (u32 c = 0; c < MEMORY_CATEGORY_COUNT; ++c) {
        const float s = MemoryBudget::share(static_cast<MemoryCategory>(c));
        CHECK(s > 0.0f);
        total += s;
    }
    CHECK(total == doctest::Approx(1.0f));
}

TEST_CASE("memory budget: limits scale with the working set") {
    const u64 gib = 1ull << 30;
    const MemoryBudget small(12 * gib, detectTier("Apple M3", false));
    const MemoryBudget large(96 * gib, detectTier("Apple M5 Max", true));
    CHECK(small.engineLimit() == static_cast<u64>(12.0 * gib * MemoryBudget::ENGINE_SHARE));
    CHECK(large.limit(MemoryCategory::Textures) == 8 * small.limit(MemoryCategory::Textures));

    u64 sum = 0;
    for (u32 c = 0; c < MEMORY_CATEGORY_COUNT; ++c) sum += large.limit(static_cast<MemoryCategory>(c));
    CHECK(sum <= large.engineLimit());
}

TEST_CASE("memory budget: pool sizes follow the working set") {
    const u64 mib = 1ull << 20, gib = 1ull << 30;
    const MemoryBudget t0(10 * gib, detectTier("Apple M3", false));       // ~16 GB machine
    const MemoryBudget t2(107 * gib, detectTier("Apple M5 Max", true));   // 128 GB machine
    const MemoryBudget tiny(64 * mib, TierInfo{});

    // Clamped ranges, whole MiB, never smaller on the bigger machine.
    for (const MemoryBudget* b : {&t0, &t2, &tiny}) {
        CHECK(b->frameUploadRingSize() >= 16 * mib);
        CHECK(b->frameUploadRingSize() <= 128 * mib);
        CHECK(b->frameUploadRingSize() % mib == 0);
        CHECK(b->stagingRingSize() >= 16 * mib);
        CHECK(b->stagingRingSize() <= 256 * mib);
        const u64 page = b->heapPageSize();
        CHECK(page >= 16 * mib);
        CHECK(page <= 128 * mib);
        CHECK((page & (page - 1)) == 0);
    }
    CHECK(t2.frameUploadRingSize() >= t0.frameUploadRingSize());
    CHECK(t2.heapPageSize() >= t0.heapPageSize());
    CHECK(t2.frameUploadRingSize() == 128 * mib);
    CHECK(t2.heapPageSize() == 128 * mib);
    CHECK(tiny.frameUploadRingSize() == 16 * mib);
    CHECK(tiny.heapPageSize() == 16 * mib);
}

TEST_CASE("memory budget: levels") {
    const MemoryBudget b(1000 * 1000, TierInfo{});
    const u64 lim = b.limit(MemoryCategory::Geometry);
    CHECK(b.level(MemoryCategory::Geometry, 0) == MemoryBudget::Level::Ok);
    CHECK(b.level(MemoryCategory::Geometry, lim / 2) == MemoryBudget::Level::Ok);
    CHECK(b.level(MemoryCategory::Geometry, lim * 9 / 10) == MemoryBudget::Level::Warning);
    CHECK(b.level(MemoryCategory::Geometry, lim) == MemoryBudget::Level::Warning);
    CHECK(b.level(MemoryCategory::Geometry, lim + 1) == MemoryBudget::Level::Over);
}
