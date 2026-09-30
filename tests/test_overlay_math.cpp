#include "diagnostics/overlay_math.h"

#include <doctest/doctest.h>

#include <cmath>

using namespace phosphor;
using namespace phosphor::overlay;

TEST_CASE("overlay: the heat palette hits its stops and stays inside [0, 1]") {
    for (u32 s = 0; s < HEAT_STOP_COUNT; ++s) {
        const float t = static_cast<float>(s) / static_cast<float>(HEAT_STOP_COUNT - 1);
        const Rgb c = heatColor(t);
        CHECK(c.r == doctest::Approx(HEAT_STOPS[s * 3 + 0]));
        CHECK(c.g == doctest::Approx(HEAT_STOPS[s * 3 + 1]));
        CHECK(c.b == doctest::Approx(HEAT_STOPS[s * 3 + 2]));
    }
    // Clamped outside [0, 1], halfway between two stops is their mean.
    CHECK(heatColor(-3.0f) == heatColor(0.0f));
    CHECK(heatColor(7.0f) == heatColor(1.0f));
    const Rgb mid = heatColor(0.1f);
    CHECK(mid.g == doctest::Approx(0.5f * (HEAT_STOPS[1] + HEAT_STOPS[4])));
    for (int i = 0; i <= 200; ++i) {
        const Rgb c = heatColor(static_cast<float>(i) / 200.0f);
        for (const float v : {c.r, c.g, c.b}) {
            CHECK(v >= 0.0f);
            CHECK(v <= 1.0f);
        }
    }
}

TEST_CASE("overlay: scales map their range to [0, 1] and saturate") {
    CHECK(normalize(KIND_OVERDRAW, 0.0f) == 0.0f);
    CHECK(normalize(KIND_OVERDRAW, OVERDRAW_MAX / 2) == doctest::Approx(0.5f));
    CHECK(normalize(KIND_OVERDRAW, 100.0f) == 1.0f);
    for (const u32 kind : {KIND_LIGHTS, KIND_TILE_COST}) {
        CAPTURE(kind);
        CHECK(normalize(kind, 0.0f) == 0.0f);
        CHECK(normalize(kind, -5.0f) == 0.0f);
        CHECK(normalize(kind, scaleMax(kind)) == doctest::Approx(1.0f));
        CHECK(normalize(kind, scaleMax(kind) * 10) == 1.0f);
        // Logarithmic: monotonic, and the middle of the value range is well
        // above the middle of the scale.
        CHECK(normalize(kind, 1.0f) < normalize(kind, 2.0f));
        CHECK(normalize(kind, scaleMax(kind) / 2) > 0.8f);
    }
}

TEST_CASE("overlay: pixels without data") {
    CHECK_FALSE(hasData(KIND_OVERDRAW, 0.0f));
    CHECK(hasData(KIND_OVERDRAW, 1.0f));
    CHECK_FALSE(hasData(KIND_TILE_COST, 0.0f));
    CHECK_FALSE(hasData(KIND_LIGHTS, -1.0f)); // cleared: no geometry
    CHECK(hasData(KIND_LIGHTS, 0.0f));        // geometry lit by no light
}

TEST_CASE("overlay: tile cost reference") {
    // 40 x 33: 2 x 2 tiles, the right column 8 wide, the bottom row 1 high.
    const u32 w = 40, h = 33;
    std::vector<float> overdraw(w * h, 0.0f), lights(w * h, -1.0f);
    CHECK(tileCount(w) == 2);
    CHECK(tileCount(h) == 2);

    // Tile (0,0): overdraw 2 with 3 lights everywhere -> 2 x 4 = 8.
    for (u32 y = 0; y < 32; ++y)
        for (u32 x = 0; x < 32; ++x) {
            overdraw[y * w + x] = 2.0f;
            lights[y * w + x]   = 3.0f;
        }
    // Tile (1,0): half of its 8 x 32 pixels drawn once, no lights (0), the
    // rest empty (lights -1 counts as 0): average 0.5.
    for (u32 y = 0; y < 32; ++y)
        for (u32 x = 32; x < 36; ++x) {
            overdraw[y * w + x] = 1.0f;
            lights[y * w + x]   = 0.0f;
        }
    // Tile (0,1): a single 32 x 1 row, overdraw 5, 1 light -> 10.
    for (u32 x = 0; x < 32; ++x) {
        overdraw[32 * w + x] = 5.0f;
        lights[32 * w + x]   = 1.0f;
    }
    const std::vector<float> tiles = tileCostReference(overdraw.data(), lights.data(), w, h);
    REQUIRE(tiles.size() == 4);
    CHECK(tiles[0] == doctest::Approx(8.0f));
    CHECK(tiles[1] == doctest::Approx(0.5f));
    CHECK(tiles[2] == doctest::Approx(10.0f));
    CHECK(tiles[3] == 0.0f);
}
