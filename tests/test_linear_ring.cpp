#include "core/memory/linear_ring.h"

#include <doctest/doctest.h>

#include <deque>
#include <random>
#include <vector>

using namespace phosphor;

TEST_CASE("linear ring: bump allocation with alignment") {
    LinearRing r(1000, 3);
    r.beginFrame(0);
    CHECK(r.allocate(10, 1) == 0u);
    CHECK(r.allocate(10, 256) == 256u);
    CHECK(r.allocate(1, 4) == 268u);
    r.endFrame();
    CHECK(r.stats().frameBytes == 269u);
    CHECK(r.stats().overflows == 0);
}

TEST_CASE("linear ring: frames in flight are protected until their slot comes back") {
    LinearRing r(300, 2);
    r.beginFrame(0);
    REQUIRE(r.allocate(100, 1));
    r.endFrame();
    r.beginFrame(1);
    REQUIRE(r.allocate(100, 1));
    r.endFrame();
    // Frame 2 reuses slot 0: frame 0 is done, frame 1 still in flight.
    r.beginFrame(2);
    CHECK(r.stats().inFlightBytes == 100u);
    CHECK(r.allocate(100, 1) == 200u);
    // Next allocation would overwrite frame 1: wraps to 0 (frame 0's bytes), fits.
    CHECK(r.allocate(100, 1) == 0u);
    CHECK_FALSE(r.allocate(1, 1).has_value()); // only frame 1's bytes are left
    CHECK(r.stats().overflows == 1);
    r.endFrame();
}

TEST_CASE("linear ring: allocations never straddle the end and stay aligned") {
    LinearRing r(1000, 1);
    r.beginFrame(0);
    REQUIRE(r.allocate(900, 1));
    r.endFrame();
    r.beginFrame(1); // single frame in flight: previous frame recycled
    // 100 bytes left at the end: a 200-byte request goes to the next lap.
    CHECK(r.allocate(200, 64) == 0u);
    CHECK(r.allocate(10, 256) == 256u);
    r.endFrame();
}

TEST_CASE("linear ring: rejects impossible requests") {
    LinearRing r(128, 2);
    r.beginFrame(0);
    CHECK_FALSE(r.allocate(0, 1));
    CHECK_FALSE(r.allocate(129, 1));
    CHECK_FALSE(r.allocate(8, 3));
    CHECK(r.stats().overflows == 3);
}

TEST_CASE("linear ring: randomized frames never overlap live data") {
    // Model: every allocation of the last `framesInFlight` frames is live.
    constexpr u64 capacity = 1 << 16;
    constexpr u32 inFlight = 3;
    LinearRing r(capacity, inFlight);
    std::mt19937 rng(1234);
    struct Range { u64 begin, end; };
    std::deque<std::vector<Range>> frames;

    for (u64 f = 0; f < 5000; ++f) {
        r.beginFrame(f);
        if (frames.size() == inFlight) frames.pop_front();
        std::vector<Range> current;
        const int count = static_cast<int>(rng() % 12);
        for (int i = 0; i < count; ++i) {
            const u64 size = 1 + rng() % 4000;
            const u64 align = u64{1} << (rng() % 9);
            const auto off = r.allocate(size, align);
            if (!off) continue;
            REQUIRE(*off % align == 0);
            REQUIRE(*off + size <= capacity);
            const Range mine{*off, *off + size};
            for (const auto& frame : frames)
                for (const Range& other : frame) REQUIRE((mine.end <= other.begin || other.end <= mine.begin));
            for (const Range& other : current) REQUIRE((mine.end <= other.begin || other.end <= mine.begin));
            current.push_back(mine);
        }
        r.endFrame();
        frames.push_back(std::move(current));
    }
    CHECK(r.stats().peakFrameBytes > 0);
}

TEST_CASE("linear ring: resize resets") {
    LinearRing r(100, 2);
    r.beginFrame(0);
    REQUIRE(r.allocate(80, 1));
    r.endFrame();
    r.resize(1000);
    r.beginFrame(1);
    CHECK(r.stats().inFlightBytes == 0u);
    CHECK(r.allocate(900, 1) == 0u);
}
