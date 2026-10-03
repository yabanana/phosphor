#include "renderer/history_registry.h"
#include "renderer/exposure.h"
#include <doctest/doctest.h>
#include <cmath>
#include <limits>
using namespace phosphor;
TEST_CASE("history has per-view ownership, explicit resets and GPU timeline requirements") {
    HistoryRegistry registry;
    const HistoryRegistry::Extent extent{640, 360, 1280, 720};
    const float matrix[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    CHECK(registry.begin(0, extent, 1).reset);
    registry.read(0, 1);
    registry.write(0, 1, matrix);
    CHECK_FALSE(registry.begin(0, extent, 1).reset);
    CHECK(registry.begin(1, extent, 1).reset);
    CHECK(registry.get(0).sample == 1);
    CHECK(registry.get(1).sample == 0);
    registry.read(0, 3);
    CHECK(registry.begin(0, extent, 1).requiredCompletion == 3);
    CHECK_THROWS(registry.write(0, 2, matrix));
    registry.write(0, 3, matrix);
    CHECK(registry.begin(0, {800, 450, 1280, 720}, 1).reset);
    registry.write(0, 4, matrix);
    CHECK(registry.begin(0, {800, 450, 1280, 720}, 1, true).reset);
    CHECK_THROWS(registry.begin(0, {}, 1));
}
TEST_CASE("Halton jitter is bounded, deterministic and independent of frame slots") {
    CHECK(temporalJitter(0)[0] == 0.0f);
    CHECK(temporalJitter(0)[1] == doctest::Approx(-1.0 / 6));
    for (u32 i = 0; i < 100; ++i) {
        const auto j = temporalJitter(i);
        CHECK(j[0] >= -0.5f);
        CHECK(j[0] < 0.5f);
        CHECK(j[1] >= -0.5f);
        CHECK(j[1] < 0.5f);
        CHECK(j == temporalJitter(i + 16));
    }
}
TEST_CASE("exposure rejects invalid luminance and adapts consistently in real time") {
    CHECK(exposureBin(0) == 0);
    CHECK(exposureBin(-1) == 0);
    CHECK(exposureBin(std::numeric_limits<float>::quiet_NaN()) == 0);
    std::array<u32, ExposureBins> h{};
    CHECK(exposureTarget(h) == 1);
    h[exposureBin(0.18f)] = 10000;
    CHECK(exposureTarget(h) == doctest::Approx(1).epsilon(0.05));
    h[255] = 1;
    CHECK(exposureTarget(h) == doctest::Approx(1).epsilon(0.05));
    float e30 = 1, e60 = 1;
    for (u32 i = 0; i < 30; ++i)
        e30 = adaptExposure(e30, 0.25f, 1.0f / 30);
    for (u32 i = 0; i < 60; ++i)
        e60 = adaptExposure(e60, 0.25f, 1.0f / 60);
    CHECK(e30 == doctest::Approx(e60).epsilon(0.00001));
    CHECK(adaptExposure(1, 2, 0) == 1);
}

TEST_CASE("raster jitter uses explicit right-down pixel displacement") {
    const float base[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    float result[16];
    applyRasterJitter(result, base, 0.25f, -0.125f, 800, 600);
    const float x = (result[12] / result[15] * 0.5f + 0.5f) * 800;
    const float y = (0.5f - result[13] / result[15] * 0.5f) * 600;
    CHECK(x == doctest::Approx(400.0f + 0.25f));
    CHECK(y == doctest::Approx(300.0f - 0.125f));
}
