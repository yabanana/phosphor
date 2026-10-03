#include "renderer/visibility_math.h"
#include "renderer/normal_transform.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <doctest/doctest.h>
#include <cmath>

using namespace phosphor;

TEST_CASE("visibility IDs distinguish instances and reserve the background sentinel") {
    for (u32 cluster : {0u, 1u, 1234u, VISIBILITY_CLUSTER_LIMIT - 1}) {
        for (u32 triangle : {0u, 1u, 124u, 127u}) {
            const u32 id = visibilityPack(cluster, triangle);
            CHECK(id != VISIBILITY_BACKGROUND);
            CHECK(visibilityCluster(id) == cluster);
            CHECK(visibilityTriangle(id) == triangle);
        }
    }
    CHECK(visibilityPack(VISIBILITY_CLUSTER_LIMIT, 0) == VISIBILITY_BACKGROUND);
    CHECK(visibilityPack(1, 128) == VISIBILITY_BACKGROUND);
    CHECK(visibilityPack(1, 0) != visibilityPack(2, 0));
}

TEST_CASE("homogeneous barycentrics reconstruct known perspective samples and derivatives") {
    // Independently construct a clip-space point from known world weights,
    // project it and ask the inverse to recover those weights.
    for (float aw : {0.0f, 0.2f, 1.0f, -0.2f}) {
        const float ax = -0.7f, ay = -0.5f, bx = 0.8f, by = -0.4f, bw = 1.2f, cx = 0.1f, cy = 0.9f, cw = 2.0f;
        const float w[3] = {0.2f, 0.3f, 0.5f};
        const float clipW = w[0] * aw + w[1] * bw + w[2] * cw;
        const float px = ((w[0] * ax + w[1] * bx + w[2] * cx) / clipW + 1) * 400;
        const float py = (1 - (w[0] * ay + w[1] * by + w[2] * cy) / clipW) * 300;
        const auto eval = [&](float x, float y) {
            return visibilityBarycentrics(ax, ay, aw, bx, by, bw, cx, cy, cw, x, y, 800, 600);
        };
        const auto r = eval(px, py);
        REQUIRE(r.valid);
        const auto left = eval(px - 0.25f, py), right = eval(px + 0.25f, py);
        const auto up = eval(px, py - 0.25f), down = eval(px, py + 0.25f);
        float dxSum = 0, dySum = 0;
        for (u32 i = 0; i < 3; ++i) {
            CHECK(r.value[i] == doctest::Approx(w[i]).epsilon(0.0001));
            CHECK(r.dx[i] == doctest::Approx((right.value[i] - left.value[i]) * 2).epsilon(0.003));
            CHECK(r.dy[i] == doctest::Approx((down.value[i] - up.value[i]) * 2).epsilon(0.003));
            dxSum += r.dx[i];
            dySum += r.dy[i];
        }
        CHECK(std::abs(dxSum) < 1e-6f);
        CHECK(std::abs(dySum) < 1e-6f);
    }
}

TEST_CASE("visibility rejects degenerate triangles and zero extents") {
    CHECK_FALSE(visibilityBarycentrics(0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 100, 100).valid);
    CHECK_FALSE(visibilityBarycentrics(-1, -1, 1, 1, -1, 1, 0, 1, 1, 0, 0, 0, 100).valid);
}

TEST_CASE("surface normals use inverse transpose for anisotropy, reflection and shear") {
    for (float x : {-3.0f, 0.01f, 1.0f, 4.0f}) {
        glm::mat3 m(glm::rotate(glm::mat4(1), 0.7f, glm::normalize(glm::vec3(1, 2, 3))));
        m[0] *= x;
        m[1] *= 0.4f;
        m[2] *= 2.0f;
        m[1] += m[0] * 0.35f;
        const glm::vec3 n = glm::normalize(glm::vec3(1, 2, 1));
        const auto transformed = transformSurfaceNormal(m[0].x, m[0].y, m[0].z, m[1].x, m[1].y, m[1].z, m[2].x, m[2].y,
                                                        m[2].z, n.x, n.y, n.z);
        const glm::vec3 actual = glm::normalize(glm::vec3(transformed.x, transformed.y, transformed.z));
        const glm::vec3 expected = glm::normalize(glm::transpose(glm::inverse(m)) * n);
        for (int i = 0; i < 3; ++i)
            CHECK(actual[i] == doctest::Approx(expected[i]).epsilon(0.00001));
        const glm::vec3 tangent = glm::normalize(glm::cross(n, glm::vec3(0, 1, 0)));
        CHECK(std::abs(glm::dot(actual, glm::normalize(m * tangent))) < 1e-5f);
    }
}
