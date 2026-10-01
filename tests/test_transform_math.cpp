// F5.2: transform math shared with the GPU (renderer/transform_math.h) and its
// CPU reference (renderer/transform_reference.h).

#include <doctest/doctest.h>

#include "renderer/transform_math.h"
#include "renderer/transform_reference.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

using namespace phosphor;

namespace {

struct Rng {
    std::mt19937_64 g{0x5EEDu};
    float f(float lo, float hi) { return lo + (hi - lo) * std::uniform_real_distribution<float>(0.0f, 1.0f)(g); }
    u32   u(u32 n) { return static_cast<u32>(g() % n); }
};

// TRS -> mat4 like TransformComponent::updateMatrix; ~30% non-uniform scale,
// ~15% mirrored.
glm::mat4 randomTrs(Rng& r) {
    glm::quat q{r.f(-1, 1), r.f(-1, 1), r.f(-1, 1), r.f(-1, 1)};
    if (glm::dot(q, q) < 1e-6f) q = glm::quat{1, 0, 0, 0};
    glm::vec3 s{r.f(0.5f, 1.5f)};
    if (r.u(100) < 30) s = {r.f(0.5f, 1.5f), r.f(0.5f, 1.5f), r.f(0.5f, 1.5f)};
    if (r.u(100) < 15) s.x = -s.x;
    return glm::translate(glm::mat4{1.0f}, {r.f(-10, 10), r.f(-10, 10), r.f(-10, 10)}) * glm::mat4_cast(glm::normalize(q))
         * glm::scale(glm::mat4{1.0f}, s);
}

glm::mat4 randomDense(Rng& r) {
    glm::mat4 m;
    for (int c = 0; c < 4; ++c)
        for (int e = 0; e < 4; ++e) m[c][e] = r.f(-4, 4);
    return m;
}

bool bitEqual(const float* a, const float* b, size_t n) { return std::memcmp(a, b, n * sizeof(float)) == 0; }

// A different summation order (reversed): the control that proves the
// bit-exact test is sensitive to the order of operations.
void mat4MulReversedOrder(const float* a, const float* b, float* r) {
#pragma clang fp contract(off)
    for (u32 j = 0; j < 4; ++j)
        for (u32 i = 0; i < 4; ++i)
            r[4 * j + i] = a[12 + i] * b[4 * j + 3] + a[8 + i] * b[4 * j + 2] + a[4 + i] * b[4 * j + 1] + a[i] * b[4 * j];
}

GPUMotion randomMotion(Rng& r, u32 speedClass) {
    GPUMotion m{};
    for (float& c : m.centre) c = r.f(-20, 20);
    m.radius = r.f(0, 15);
    const float phase = r.f(0, 6.28f);
    m.cosPhase = std::cos(phase);
    m.sinPhase = std::sin(phase);
    m.height = r.f(-3, 3);
    m.speedClass = speedClass;
    const glm::mat4 base = randomTrs(r);
    for (u32 j = 0; j < 4; ++j)
        for (u32 i = 0; i < 3; ++i) m.base[3 * j + i] = base[static_cast<int>(j)][static_cast<int>(i)];
    return m;
}

} // namespace

TEST_CASE("mat4Mul is bit-identical to glm::mat4 operator* (random, mirrored, non-uniform, dense)") {
    Rng r;
    u32 mirrored = 0;
    for (int n = 0; n < 5000; ++n) {
        const glm::mat4 A = randomTrs(r), B = randomTrs(r);
        mirrored += glm::determinant(A) < 0 || glm::determinant(B) < 0;
        float got[16];
        mat4Mul(&A[0][0], &B[0][0], got);
        const glm::mat4 want = A * B;
        CHECK(bitEqual(got, &want[0][0], 16));
    }
    for (int n = 0; n < 2000; ++n) { // no zero structure: every product term matters
        const glm::mat4 A = randomDense(r), B = randomDense(r);
        float got[16];
        mat4Mul(&A[0][0], &B[0][0], got);
        const glm::mat4 want = A * B;
        CHECK(bitEqual(got, &want[0][0], 16));
    }
    CHECK(mirrored > 1000);
}

TEST_CASE("mat4Mul: a different summation order is NOT bit-identical (negative control)") {
    Rng r;
    u32 differing = 0;
    for (int n = 0; n < 2000; ++n) {
        const glm::mat4 A = randomDense(r), B = randomDense(r);
        float got[16];
        mat4MulReversedOrder(&A[0][0], &B[0][0], got);
        const glm::mat4 want = A * B;
        differing += !bitEqual(got, &want[0][0], 16);
    }
    CHECK(differing > 1000); // the bit-exact test above can fail
}

TEST_CASE("motionWorld matches translate * rotateY * base within tolerance and is deterministic") {
    Rng r;
    for (int n = 0; n < 2000; ++n) {
        const GPUMotion m = randomMotion(r, r.u(SCENE_MOTION_CLASSES));
        const float a = r.f(-20, 20);
        const float s = std::sin(a), c = std::cos(a);
        float w[16], w2[16];
        motionWorld(m, s, c, w);
        motionWorld(m, s, c, w2);
        CHECK(bitEqual(w, w2, 16));

        glm::mat4 base{1.0f};
        for (u32 j = 0; j < 4; ++j)
            for (u32 i = 0; i < 3; ++i) base[static_cast<int>(j)][static_cast<int>(i)] = m.base[3 * j + i];
        glm::mat4 rot{1.0f}; // rotateY(a) with the same sin/cos (glm::rotate about +Y)
        rot[0][0] = c;  rot[0][2] = -s;
        rot[2][0] = s;  rot[2][2] = c;
        const glm::vec3 orbit{m.centre[0] + m.radius * (c * m.cosPhase - s * m.sinPhase), m.centre[1] + m.height,
                              m.centre[2] + m.radius * (s * m.cosPhase + c * m.sinPhase)};
        const glm::mat4 want = glm::translate(glm::mat4{1.0f}, orbit) * rot * base;
        for (int e = 0; e < 16; ++e) {
            const float ref = (&want[0][0])[e];
            CHECK(std::fabs(w[e] - ref) <= 1e-5f * (1.0f + std::fabs(ref)));
        }
        // The last row stays (0 0 0 1); a rotation about Y keeps the determinant sign.
        CHECK(w[3] == 0.0f);
        CHECK(w[7] == 0.0f);
        CHECK(w[11] == 0.0f);
        CHECK(w[15] == 1.0f);
        CHECK((glm::determinant(glm::mat3(want)) < 0) == (glm::determinant(glm::mat3(base)) < 0));
    }
}

TEST_CASE("motionWorld with radius 0 and angle 0 keeps the base plus centre/height") {
    GPUMotion m{};
    m.centre[0] = 1; m.centre[1] = 2; m.centre[2] = 3;
    m.height = 0.5f;
    m.cosPhase = 1.0f;
    m.base[0] = 2; m.base[4] = 3; m.base[8] = 4; // diagonal scale
    m.base[9] = 10; m.base[10] = 20; m.base[11] = 30;
    float w[16];
    motionWorld(m, 0.0f, 1.0f, w);
    CHECK(w[0] == 2);
    CHECK(w[5] == 3);
    CHECK(w[10] == 4);
    CHECK(w[12] == 11);
    CHECK(w[13] == 22.5f);
    CHECK(w[14] == 33);
    CHECK(w[15] == 1);
}

TEST_CASE("motionSinCosTable: documented ramp, double precision, parity sign") {
    float t0[SCENE_MOTION_CLASSES * 2], t1[SCENE_MOTION_CLASSES * 2], t1b[SCENE_MOTION_CLASSES * 2];
    motionSinCosTable(0.0, t0);
    for (u32 k = 0; k < SCENE_MOTION_CLASSES; ++k) {
        CHECK(t0[2 * k] == 0.0f);
        CHECK(t0[2 * k + 1] == 1.0f);
    }
    CHECK(motionClassSpeed(0) == doctest::Approx(0.05));
    CHECK(motionClassSpeed(63) == doctest::Approx(-1.55)); // odd class: backwards
    CHECK(motionClassSpeed(62) > 0.0);
    motionSinCosTable(12.5, t1);
    motionSinCosTable(12.5, t1b);
    CHECK(bitEqual(t1, t1b, SCENE_MOTION_CLASSES * 2));
    for (u32 k = 0; k < SCENE_MOTION_CLASSES; ++k) {
        const double a = 12.5 * motionClassSpeed(k);
        CHECK(t1[2 * k] == static_cast<float>(std::sin(a)));
        CHECK(t1[2 * k + 1] == static_cast<float>(std::cos(a)));
        CHECK(std::fabs(t1[2 * k] * t1[2 * k] + t1[2 * k + 1] * t1[2 * k + 1] - 1.0f) < 1e-6f);
    }
    CHECK(t1[2] == static_cast<float>(std::sin(12.5 * -(0.05 + 1.5 / 63.0)))); // class 1 runs backwards
}

namespace {

// Slot-space forest builder for the reference tests.
struct Forest {
    std::vector<GPUInstance>      instances;
    std::vector<GPUTransformNode> nodes;
    std::vector<GPUMotion>        motions;
    std::vector<u32>              motionSlots, childOffsets, childSlots;
    std::vector<u32>              parent; // ~0u for roots
    std::vector<glm::mat4>        locals;

    HierarchyView view() const { return {instances, nodes, motions, motionSlots, childOffsets, childSlots}; }

    void finish() {
        const u32 n = static_cast<u32>(instances.size());
        childOffsets.assign(n + 1, 0);
        for (u32 s = 0; s < n; ++s)
            if (parent[s] != ~0u) ++childOffsets[parent[s] + 1];
        for (u32 s = 0; s < n; ++s) childOffsets[s + 1] += childOffsets[s];
        childSlots.assign(childOffsets[n], 0);
        std::vector<u32> fill(n, 0);
        for (u32 s = 0; s < n; ++s)
            if (parent[s] != ~0u) childSlots[childOffsets[parent[s]] + fill[parent[s]]++] = s;
    }
};

Forest buildHandForest(Rng& r) {
    Forest f;
    const u32 n = 12;
    f.instances.assign(n, GPUInstance{});
    f.nodes.assign(n, GPUTransformNode{});
    f.motions.assign(n, GPUMotion{});
    f.parent.assign(n, ~0u);
    f.locals.assign(n, glm::mat4{1.0f});
    // Slot 0 is a moving root with a chain 0 -> 1 -> ... -> 7 (depth 7);
    // slot 8 is a static root with children 9 and 10, and 10 has child 11.
    for (u32 s = 1; s <= 7; ++s) f.parent[s] = s - 1;
    f.parent[9] = 8;
    f.parent[10] = 8;
    f.parent[11] = 10;
    for (u32 s = 0; s < n; ++s) {
        const glm::mat4 m = randomTrs(r);
        std::memcpy(f.instances[s].modelMatrix, &m[0][0], 64); // static roots: the mirror matrix; children: stale
        if (f.parent[s] != ~0u) {
            f.locals[s] = m;
            std::memcpy(f.nodes[s].local, &m[0][0], 64);
            f.nodes[s].parentSlot = f.parent[s];
            f.nodes[s].depth = s <= 7 ? s : (s == 11 ? 2 : 1);
        }
    }
    f.motions[0] = randomMotion(r, 5);
    f.motionSlots = {0};
    f.finish();
    return f;
}

} // namespace

TEST_CASE("referenceWorlds: hand-built forest with a depth-7 chain, a motion parent and static roots") {
    Rng r;
    const Forest f = buildHandForest(r);
    float sinCos[SCENE_MOTION_CLASSES * 2];
    motionSinCosTable(3.25, sinCos);
    std::vector<float> world;
    referenceWorlds(f.view(), sinCos, world);
    REQUIRE(world.size() == 12 * 16);

    // Independent expectation with glm.
    std::vector<glm::mat4> want(12);
    float w0[16];
    motionWorld(f.motions[0], sinCos[10], sinCos[11], w0);
    std::memcpy(&want[0][0][0], w0, 64);
    for (u32 s = 1; s <= 7; ++s) want[s] = want[s - 1] * f.locals[s];
    std::memcpy(&want[8][0][0], f.instances[8].modelMatrix, 64);
    want[9]  = want[8] * f.locals[9];
    want[10] = want[8] * f.locals[10];
    want[11] = want[10] * f.locals[11];
    for (u32 s = 0; s < 12; ++s) CHECK_MESSAGE(bitEqual(&world[s * 16], &want[s][0][0], 16), "slot ", s);

    // Moving the parent moves the whole chain: another time gives other worlds.
    motionSinCosTable(4.5, sinCos);
    std::vector<float> later;
    referenceWorlds(f.view(), sinCos, later);
    CHECK(!bitEqual(&world[7 * 16], &later[7 * 16], 16));
    CHECK(bitEqual(&world[11 * 16], &later[11 * 16], 16)); // static subtree unchanged
}

TEST_CASE("referenceWorlds: no hierarchy and empty input") {
    HierarchyView empty;
    std::vector<float> world{1.0f};
    float sinCos[SCENE_MOTION_CLASSES * 2] = {};
    referenceWorlds(empty, sinCos, world);
    CHECK(world.empty());

    Rng r;
    std::vector<GPUInstance> inst(3);
    for (auto& i : inst) {
        const glm::mat4 m = randomTrs(r);
        std::memcpy(i.modelMatrix, &m[0][0], 64);
    }
    HierarchyView v;
    v.instances = inst;
    referenceWorlds(v, sinCos, world); // empty CSR: every slot keeps its matrix
    REQUIRE(world.size() == 48);
    for (u32 s = 0; s < 3; ++s) CHECK(bitEqual(&world[s * 16], inst[s].modelMatrix, 16));
}
