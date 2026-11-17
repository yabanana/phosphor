#include "testbench/corpus_rig.h"

#include <doctest/doctest.h>
#include <glm/gtc/constants.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

using namespace phosphor;

namespace {
constexpr float fovY = glm::radians(60.0f);

// Project a world point to NDC for the framed camera (right-handed look-at, vertical fov `fovY`).
glm::vec2 ndc(const CorpusRigView& view, glm::vec3 point, float aspect) {
    const glm::vec3 forward = glm::normalize(view.target - view.position);
    const glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0, 1, 0)));
    const glm::vec3 up = glm::cross(right, forward);
    const glm::vec3 d = point - view.position;
    const float depth = glm::dot(d, forward), t = std::tan(0.5f * fovY);
    return {glm::dot(d, right) / (depth * t * aspect), glm::dot(d, up) / (depth * t)};
}
} // namespace

TEST_CASE("F15 corpus rig: a flat wide box fills the width, a tall one the height, with the declared margin") {
    const float aspect = 4.0f / 3.0f;
    SUBCASE("wide") {
        const auto view = frameCorpusRig({-4, -1, -0.1f}, {4, 1, 0.1f}, 0, 0, fovY, aspect);
        // Front view of a thin slab: the horizontal extent decides (4 / (tan30 * 4/3) * margin + the half depth).
        const float expected = 4.0f / (std::tan(0.5f * fovY) * aspect) * CorpusRig::margin + 0.1f;
        CHECK(view.distance == doctest::Approx(expected));
        CHECK(view.position.z == doctest::Approx(expected));
        CHECK(view.target.x == doctest::Approx(0.0f));
        // The front face corners project inside the frame, the widest one close to 1 / margin of the half width (slab depth shifts it a little).
        const glm::vec2 corner = ndc(view, {4, 1, 0.1f}, aspect);
        CHECK(std::abs(corner.x) < 1.0f);
        CHECK(std::abs(corner.x) > 0.8f);
    }
    SUBCASE("tall") {
        const auto view = frameCorpusRig({-0.5f, -3, -0.1f}, {0.5f, 3, 0.1f}, 0, 0, fovY, aspect);
        const float expected = 3.0f / std::tan(0.5f * fovY) * CorpusRig::margin + 0.1f;
        CHECK(view.distance == doctest::Approx(expected));
    }
}

TEST_CASE("F15 corpus rig: every bounds corner stays inside the frame for rotated views") {
    const float aspect = 16.0f / 9.0f;
    const glm::vec3 minimum(-1.0f, -0.3f, -2.0f), maximum(1.5f, 2.0f, 0.5f);
    for (const float yaw : {0.0f, 35.0f, 90.0f, -140.0f}) {
        for (const float pitch : {0.0f, 20.0f, -30.0f}) {
            const auto view = frameCorpusRig(minimum, maximum, yaw, pitch, fovY, aspect);
            float widest = 0.0f;
            for (unsigned corner = 0; corner < 8; ++corner) {
                const glm::vec3 point((corner & 1) ? maximum.x : minimum.x, (corner & 2) ? maximum.y : minimum.y, (corner & 4) ? maximum.z : minimum.z);
                const glm::vec2 p = ndc(view, point, aspect);
                CHECK(std::abs(p.x) <= 1.0f);
                CHECK(std::abs(p.y) <= 1.0f);
                widest = std::max(widest, std::max(std::abs(p.x), std::abs(p.y)));
            }
            // The fit is exact: the extreme corner sits at 1 / margin of the half frame (not a loose sphere or front-face fit).
            CHECK(widest == doctest::Approx(1.0f / CorpusRig::margin).epsilon(1e-3));
            CHECK(glm::length(view.position - view.target) == doctest::Approx(view.distance));
        }
    }
}

TEST_CASE("F15 corpus rig: yaw 0 looks from +Z, the sun comes from the upper left behind the camera") {
    const auto view = frameCorpusRig({-1, -1, -1}, {1, 1, 1}, 0, 0, fovY, 1.0f);
    CHECK(view.position.z > view.target.z);
    CHECK(view.position.x == doctest::Approx(view.target.x));
    CHECK(glm::length(view.towardSun) == doctest::Approx(1.0f));
    CHECK(view.towardSun.x < 0.0f);
    CHECK(view.towardSun.y > 0.0f);
    CHECK(view.towardSun.z > 0.0f);
    // A model-space point lit from the camera side: the sun is not behind the model.
    const auto turned = frameCorpusRig({-1, -1, -1}, {1, 1, 1}, 90, 0, fovY, 1.0f);
    CHECK(turned.position.x > turned.target.x);
}

TEST_CASE("F15 corpus rig: empty or non-finite bounds fall back to a unit box") {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (const auto& bounds : {std::pair{glm::vec3(1), glm::vec3(-1)}, std::pair{glm::vec3(nan), glm::vec3(0)},
                               std::pair{glm::vec3(std::numeric_limits<float>::max()), glm::vec3(std::numeric_limits<float>::lowest())}}) {
        const auto view = frameCorpusRig(bounds.first, bounds.second, 0, 0, fovY, 1.0f);
        CHECK(std::isfinite(view.distance));
        CHECK(view.distance > 0.0f);
        CHECK(view.target == glm::vec3(0.0f));
    }
}
