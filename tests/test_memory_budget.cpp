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

TEST_CASE("memory budget: levels") {
    const MemoryBudget b(1000 * 1000, TierInfo{});
    const u64 lim = b.limit(MemoryCategory::Geometry);
    CHECK(b.level(MemoryCategory::Geometry, 0) == MemoryBudget::Level::Ok);
    CHECK(b.level(MemoryCategory::Geometry, lim / 2) == MemoryBudget::Level::Ok);
    CHECK(b.level(MemoryCategory::Geometry, lim * 9 / 10) == MemoryBudget::Level::Warning);
    CHECK(b.level(MemoryCategory::Geometry, lim) == MemoryBudget::Level::Warning);
    CHECK(b.level(MemoryCategory::Geometry, lim + 1) == MemoryBudget::Level::Over);
}
