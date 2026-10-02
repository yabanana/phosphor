#include <doctest/doctest.h>

#include "renderer/cull_math.h"
#include "renderer/cull_reference.h"
#include "scene/camera.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

using namespace phosphor;

namespace {

constexpr float kNear = 0.05f;

struct Rng {
    u64 s;
    float next() { // [0, 1)
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return float(s >> 40) * (1.0f / 16777216.0f);
    }
    float range(float a, float b) { return a + (b - a) * next(); }
};

Camera makeCamera(glm::vec3 pos, float yaw, float pitch) {
    Camera cam(glm::radians(60.0f), 16.0f / 9.0f, kNear, 1000.0f);
    cam.setPosition(pos);
    cam.setYawPitch(yaw, pitch);
    cam.updateMatrices();
    return cam;
}

// Independent classification: clip space in double (Metal volume, reverse-Z).
bool insideClip(const glm::mat4& vp, glm::vec3 p) {
    double c[4];
    for (int r = 0; r < 4; ++r) c[r] = double(vp[0][r]) * p.x + double(vp[1][r]) * p.y + double(vp[2][r]) * p.z + double(vp[3][r]);
    return c[3] > 0.0 && std::fabs(c[0]) <= c[3] && std::fabs(c[1]) <= c[3] && c[2] >= 0.0 && c[2] <= c[3];
}

float planeDistance(const glm::vec4& pl, glm::vec3 p) { return pl.x * p.x + pl.y * p.y + pl.z * p.z + pl.w; }

bool insidePlanes(const std::array<glm::vec4, 5>& planes, glm::vec3 p) {
    for (const glm::vec4& pl : planes)
        if (planeDistance(pl, p) < 0.0f) return false;
    return true;
}

GPUCullParams paramsFor(const Camera& cam, u32 flags, float maxDistance = 100.0f, float minPixels = 2.0f, u32 slots = 1) {
    return makeCullParams(cam.getViewProjection(), cam.getProjection()[1][1], 1080, cam.getPosition(), cam.getFront(), kNear, flags,
                          maxDistance, minPixels, slots);
}

u32 reasonOf(const GPUCullParams& p, glm::vec3 c, float r) { return cullSphere(p, c.x, c.y, c.z, r); }

GPUInstance makeInstance(const glm::mat4& m, u32 mesh, u32 flags = INSTANCE_FLAG_VALID) {
    GPUInstance in{};
    std::memcpy(in.modelMatrix, glm::value_ptr(m), sizeof(in.modelMatrix));
    in.meshIndex = mesh;
    in.flags     = flags;
    return in;
}

GPUMeshInfo makeMesh(glm::vec3 c, float r) {
    GPUMeshInfo m{};
    m.boundingSphere[0] = c.x;
    m.boundingSphere[1] = c.y;
    m.boundingSphere[2] = c.z;
    m.boundingSphere[3] = r;
    return m;
}

} // namespace

// ---- planes --------------------------------------------------------------------------------

TEST_CASE("frustum planes of the reverse-Z infinite projection match clip-space classification") {
    Rng rng{0x9E3779B97F4A7C15ull};
    u32 inside = 0, total = 0, behindRejected = 0;
    for (int pose = 0; pose < 6; ++pose) {
        const glm::vec3 pos(rng.range(-20, 20), rng.range(-5, 5), rng.range(-20, 20));
        const Camera cam = makeCamera(pos, rng.range(-180, 180), rng.range(-60, 60));
        const auto planes = cam.getFrustumPlanes();
        const glm::mat4& vp = cam.getViewProjection();
        for (int i = 0; i < 5000; ++i) {
            glm::vec3 p = pos + glm::vec3(rng.range(-40, 40), rng.range(-40, 40), rng.range(-40, 40));
            if (i % 4 == 0) { // the slab between the camera and the near plane, where the OpenGL near plane is wrong
                const float depth = rng.range(0.002f, 0.1f);
                p = pos + cam.getFront() * depth + (cam.getRight() * rng.range(-0.3f, 0.3f) + cam.getUp() * rng.range(-0.3f, 0.3f)) * depth;
            }
            // Skip points within 1e-3 of a plane: float vs double rounding.
            float nearest = 1e9f;
            for (const glm::vec4& pl : planes) nearest = std::min(nearest, std::fabs(planeDistance(pl, p)));
            if (nearest < 1.0e-3f) continue;
            const bool a = insidePlanes(planes, p);
            const bool b = insideClip(vp, p);
            REQUIRE(a == b);
            ++total;
            inside += a;
            if (glm::dot(p - pos, cam.getFront()) < 0.0f && !a) ++behindRejected;
        }
    }
    CHECK(inside > 0);
    CHECK(inside < total);
    CHECK(behindRejected > 1000); // points behind the camera are classified outside
}

TEST_CASE("frustum planes: unit normals, near normal is the camera forward axis, 5 planes in order") {
    const Camera cam = makeCamera({1, 2, 3}, 30.0f, -20.0f);
    const auto planes = cam.getFrustumPlanes();
    static_assert(std::tuple_size_v<decltype(planes)> == 5);
    for (const glm::vec4& pl : planes) CHECK(glm::length(glm::vec3(pl)) == doctest::Approx(1.0f).epsilon(1e-5));
    CHECK(glm::length(glm::vec3(planes[4]) - cam.getFront()) < 1e-4f);
    // The near plane passes through camera + near * forward.
    CHECK(std::fabs(planeDistance(planes[4], cam.getPosition() + kNear * cam.getFront())) < 1e-4f);
    // Left/right/bottom/top pass through the camera position.
    for (int i = 0; i < 4; ++i) CHECK(std::fabs(planeDistance(planes[i], cam.getPosition())) < 1e-4f);
    // The forward point is inside, with the camera right of a left plane.
    CHECK(insidePlanes(planes, cam.getPosition() + 10.0f * cam.getFront()));
    CHECK(planeDistance(planes[0], cam.getPosition() + cam.getRight()) > 0.0f);
    CHECK(planeDistance(planes[1], cam.getPosition() + cam.getRight()) < 0.0f);
}

TEST_CASE("negative control: the OpenGL near plane (row3 + row2) fails on a point behind the camera") {
    const Camera cam = makeCamera({0, 0, 0}, -90.0f, 0.0f); // looks along -Z
    const glm::mat4& m = cam.getViewProjection();
    const glm::vec4 glNear(m[0][3] + m[0][2], m[1][3] + m[1][2], m[2][3] + m[2][2], m[3][3] + m[3][2]);
    const glm::vec3 behind(0.0f, 0.0f, 0.04f); // behind the camera, z <= +near
    CHECK_FALSE(insideClip(m, behind));
    CHECK(planeDistance(glNear, behind) / glm::length(glm::vec3(glNear)) > 0.0f); // the old plane ACCEPTS it
    CHECK_FALSE(insidePlanes(cam.getFrustumPlanes(), behind));                    // the fixed planes reject it
    // The four side planes already reject the points behind the camera, so with the old near plane
    // the whole test is wrong only in the slab between the camera and the near plane (in front of
    // the camera, inside the pyramid): clip space rejects it, the old plane accepts it.
    Rng rng{42};
    u32 wrong = 0, wrongFixed = 0;
    auto old = cam.getFrustumPlanes();
    old[4]   = glNear / glm::length(glm::vec3(glNear));
    for (int i = 0; i < 2000; ++i) {
        const float depth = rng.range(0.002f, 0.9f * kNear);
        const glm::vec3 p(rng.range(-0.3f, 0.3f) * depth, rng.range(-0.3f, 0.3f) * depth, -depth);
        wrong += insidePlanes(old, p) != insideClip(m, p);
        wrongFixed += insidePlanes(cam.getFrustumPlanes(), p) != insideClip(m, p);
    }
    CHECK(wrong > 1900);
    CHECK(wrongFixed == 0);
}

// ---- world sphere -------------------------------------------------------------------------

namespace {
float maxColumnLength(const glm::mat4& m) {
    return std::max({glm::length(glm::vec3(m[0])), glm::length(glm::vec3(m[1])), glm::length(glm::vec3(m[2]))});
}
} // namespace

TEST_CASE("world sphere: rotation, non-uniform scale, negative scale") {
    const glm::vec3 c(1.0f, -2.0f, 0.5f);
    const float r = 1.5f;
    const float sph[4] = {c.x, c.y, c.z, r};
    const glm::mat4 rot = glm::rotate(glm::mat4(1.0f), 0.7f, glm::normalize(glm::vec3(1, 2, 3)));
    const glm::mat4 cases[] = {
        glm::translate(glm::mat4(1.0f), {4, 5, 6}),
        glm::translate(glm::mat4(1.0f), {4, 5, 6}) * rot,
        glm::translate(glm::mat4(1.0f), {-3, 2, 1}) * rot * glm::scale(glm::mat4(1.0f), {2.0f, 0.5f, 3.0f}),
        glm::translate(glm::mat4(1.0f), {1, 1, 1}) * rot * glm::scale(glm::mat4(1.0f), {-2.0f, 1.0f, 1.0f}),
        glm::scale(glm::mat4(1.0f), {-0.5f, -0.5f, -0.5f}),
    };
    for (const glm::mat4& m : cases) {
        const GPUWorldSphere w = cullWorldSphere(glm::value_ptr(m), sph);
        const glm::vec3 wc     = glm::vec3(m * glm::vec4(c, 1.0f));
        CHECK(w.x == doctest::Approx(wc.x).epsilon(1e-5));
        CHECK(w.y == doctest::Approx(wc.y).epsilon(1e-5));
        CHECK(w.z == doctest::Approx(wc.z).epsilon(1e-5));
        // F6: Gershgorin bound of the spectral norm, exact for orthogonal
        // columns (rotation x scale) up to the 2^-16 inflation and rounding.
        CHECK(w.r >= r * maxColumnLength(m));
        CHECK(w.r == doctest::Approx(r * maxColumnLength(m) * (1.0f + 1.0f / 65536.0f)).epsilon(1e-5));
        // Conservative: every point of the mesh-space sphere lands inside the world sphere.
        Rng rng{7};
        for (int i = 0; i < 200; ++i) {
            glm::vec3 d(rng.range(-1, 1), rng.range(-1, 1), rng.range(-1, 1));
            if (glm::dot(d, d) < 1e-6f) continue;
            const glm::vec3 p = c + glm::normalize(d) * r;
            const glm::vec3 q = glm::vec3(m * glm::vec4(p, 1.0f));
            CHECK(glm::length(q - glm::vec3(w.x, w.y, w.z)) <= w.r * 1.0001f);
        }
    }
    // Non-uniform scale: the radius follows the LONGEST column (3), not the average.
    CHECK(cullWorldSphere(glm::value_ptr(cases[2]), sph).r ==
          doctest::Approx(4.5f * (1.0f + 1.0f / 65536.0f)).epsilon(1e-5));
}

TEST_CASE("world sphere: shear (F6 fix: the longest column is not a bound)") {
    // A child rotated 45 degrees under a parent scaled (3, 1, 1): the product
    // has non-orthogonal columns.  Its largest stretch exceeds the longest
    // column, so the F5 radius (r * longest column) missed points; the
    // Gershgorin bound must contain every transformed point.
    const glm::mat4 parent = glm::scale(glm::mat4(1.0f), {3.0f, 1.0f, 1.0f});
    const glm::mat4 child  = glm::rotate(glm::mat4(1.0f), glm::radians(45.0f), glm::vec3(0, 0, 1));
    const glm::mat4 m      = parent * child;
    const glm::vec3 c(0.0f);
    const float r = 1.0f;
    const float sph[4] = {c.x, c.y, c.z, r};
    const GPUWorldSphere w = cullWorldSphere(glm::value_ptr(m), sph);
    float maxStretch = 0.0f;
    for (int i = 0; i < 3600; ++i) {
        const float a = glm::radians(static_cast<float>(i) * 0.1f);
        const glm::vec3 q = glm::vec3(m * glm::vec4(std::cos(a), std::sin(a), 0.0f, 1.0f));
        maxStretch = std::max(maxStretch, glm::length(q));
    }
    CHECK(maxStretch > maxColumnLength(m) * 1.1f); // the old bound was too small
    CHECK(w.r >= maxStretch);
    // A shear matrix [[1,1],[0,1]]: stretch = golden ratio > sqrt(2) (longest column).
    glm::mat4 shear(1.0f);
    shear[1][0] = 1.0f;
    const GPUWorldSphere s = cullWorldSphere(glm::value_ptr(shear), sph);
    CHECK(s.r >= 1.6180339f);
}

// ---- cullSphere / margins -----------------------------------------------------------------

TEST_CASE("cullSphere: every reason, tested in order and only when its flag is set") {
    // Camera at the origin looking down -Z.
    const Camera cam = makeCamera({0, 0, 0}, -90.0f, 0.0f);
    const u32 all    = CULL_FLAG_FRUSTUM | CULL_FLAG_DISTANCE | CULL_FLAG_SIZE;
    const GPUCullParams p = paramsFor(cam, all, 100.0f, 2.0f);

    CHECK(reasonOf(p, {0, 0, -10}, 1.0f) == CULL_REASON_VISIBLE);
    CHECK(reasonOf(p, {0, 0, +10}, 1.0f) == CULL_REASON_FRUSTUM);   // behind the camera
    CHECK(reasonOf(p, {500, 0, -10}, 1.0f) == CULL_REASON_FRUSTUM); // far to the side
    CHECK(reasonOf(p, {0, 0, -200}, 1.0f) == CULL_REASON_DISTANCE); // beyond maxDistance, big enough on screen
    // 1080 px, proj11 = 1/tan(30 deg): pixelScale ~ 935; at depth 50 a radius 0.01 sphere is ~0.4 px < 2 px.
    CHECK(reasonOf(p, {0, 0, -50}, 0.01f) == CULL_REASON_SIZE);
    // A sphere failing both distance and size is reported as DISTANCE (test order).
    CHECK(reasonOf(p, {0, 0, -200}, 0.001f) == CULL_REASON_DISTANCE);
    // A sphere failing frustum and distance is reported as FRUSTUM.
    CHECK(reasonOf(p, {0, 0, +200}, 1.0f) == CULL_REASON_FRUSTUM);

    // Flags off: the test does not run.
    GPUCullParams q = p;
    q.flags         = CULL_FLAG_DISTANCE | CULL_FLAG_SIZE;
    CHECK(reasonOf(q, {0, 0, +10}, 1.0f) == CULL_REASON_VISIBLE);
    q.flags = CULL_FLAG_FRUSTUM | CULL_FLAG_SIZE;
    CHECK(reasonOf(q, {0, 0, -200}, 1.0f) == CULL_REASON_VISIBLE);
    q.flags = CULL_FLAG_FRUSTUM | CULL_FLAG_DISTANCE;
    CHECK(reasonOf(q, {0, 0, -50}, 0.01f) == CULL_REASON_VISIBLE);
    q.flags = 0;
    CHECK(reasonOf(q, {0, 0, +10}, 1.0f) == CULL_REASON_VISIBLE);
    CHECK(reasonOf(q, {0, 0, -1e6f}, 0.0f) == CULL_REASON_VISIBLE);
}

TEST_CASE("cullSphere: size test uses the depth along the forward axis, clamped to the near plane") {
    const Camera cam = makeCamera({0, 0, 0}, -90.0f, 0.0f);
    GPUCullParams p  = paramsFor(cam, CULL_FLAG_SIZE, 1e9f, 10.0f);
    const float diameterAt1 = 2.0f * p.pixelScale; // pixels of a radius-1 sphere at depth 1
    // Depth 100, radius 1: diameter = diameterAt1 / 100 ~ 18.7 px > 10.
    CHECK(reasonOf(p, {0, 0, -100}, 1.0f) == CULL_REASON_VISIBLE);
    // Depth 100 with minPixels just above the diameter: culled.
    p.minPixels = diameterAt1 / 100.0f * 1.01f;
    CHECK(reasonOf(p, {0, 0, -100}, 1.0f) == CULL_REASON_SIZE);
    p.minPixels = diameterAt1 / 100.0f * 0.99f;
    CHECK(reasonOf(p, {0, 0, -100}, 1.0f) == CULL_REASON_VISIBLE);
    // Off-axis: the distance does not matter, the depth does (same depth, large lateral offset).
    CHECK(reasonOf(p, {30.0f, 0, -100}, 1.0f) == CULL_REASON_VISIBLE);
    // Behind the camera (negative depth): clamped to the near plane, not a negative diameter test.
    p.minPixels = 1.0f;
    CHECK(reasonOf(p, {0, 0, +5}, 1.0f) == CULL_REASON_VISIBLE); // diameter = diameterAt1 / near: huge
}

TEST_CASE("cullSphereEval: reason equals cullSphere, margin sign and boundary behaviour") {
    const Camera cam = makeCamera({3, 1, -2}, 20.0f, 10.0f);
    const u32 all    = CULL_FLAG_FRUSTUM | CULL_FLAG_DISTANCE | CULL_FLAG_SIZE;
    const GPUCullParams p = paramsFor(cam, all, 60.0f, 3.0f);
    Rng rng{1234};
    u32 reasons[4] = {};
    for (int i = 0; i < 20000; ++i) {
        const glm::vec3 c = cam.getPosition() + glm::vec3(rng.range(-120, 120), rng.range(-120, 120), rng.range(-120, 120));
        const float r     = rng.range(0.0f, 1.0f) < 0.5f ? rng.range(0.001f, 0.05f) : rng.range(0.1f, 5.0f);
        const CullEval e  = cullSphereEval(p, c.x, c.y, c.z, r);
        REQUIRE(e.reason == cullSphere(p, c.x, c.y, c.z, r));
        ++reasons[e.reason];
        if (e.reason == CULL_REASON_VISIBLE) CHECK(e.margin >= 0.0f);
        else CHECK(e.margin <= 0.0f);
    }
    for (u32 k = 0; k < 4; ++k) CHECK(reasons[k] > 20); // every reason is exercised

    // A sphere exactly on a boundary has a tiny margin; one well inside a large one.
    const GPUCullParams q = paramsFor(makeCamera({0, 0, 0}, -90.0f, 0.0f), CULL_FLAG_DISTANCE, 100.0f, 0.0f);
    CHECK(std::fabs(cullSphereEval(q, 0, 0, -101.0f, 1.0f).margin) < 1e-6f);
    CHECK(cullSphereEval(q, 0, 0, -10.0f, 1.0f).margin > 0.4f);
    CHECK(cullSphereEval(q, 0, 0, -300.0f, 1.0f).margin < -0.4f);
    // No test enabled: visible with the largest margin.
    GPUCullParams none = q;
    none.flags         = 0;
    CHECK(cullSphereEval(none, 0, 0, -300.0f, 1.0f).margin > 1e30f);
    // Culled by frustum with a borderline-passing earlier test has no earlier test: margin of the failing one.
    // Culled by distance while the frustum plane is borderline: the small frustum margin limits the margin.
    const Camera cam0 = makeCamera({0, 0, 0}, -90.0f, 0.0f);
    GPUCullParams two = paramsFor(cam0, CULL_FLAG_FRUSTUM | CULL_FLAG_DISTANCE, 100.0f, 0.0f);
    const auto planes = cam0.getFrustumPlanes();
    // Centre on the left plane at depth 200 (distance fails clearly): frustum margin ~ r/(r+1) at best, make r tiny.
    glm::vec3 onLeft = glm::vec3(-200.0f * std::tan(glm::radians(30.0f)) * 16.0f / 9.0f, 0.0f, -200.0f);
    CHECK(std::fabs(planeDistance(planes[0], onLeft)) < 1e-3f);
    const CullEval ev = cullSphereEval(two, onLeft.x, onLeft.y, onLeft.z, 1e-4f);
    CHECK(std::fabs(ev.margin) < 1e-4f); // ambiguous: the frustum test is a coin flip, so the distance reason is too
}

TEST_CASE("cullSphere is conservative: a sphere that touches the frustum is visible") {
    Rng rng{99};
    for (int pose = 0; pose < 4; ++pose) {
        const glm::vec3 pos(rng.range(-30, 30), rng.range(-10, 10), rng.range(-30, 30));
        const Camera cam = makeCamera(pos, rng.range(-180, 180), rng.range(-50, 50));
        const GPUCullParams p = paramsFor(cam, CULL_FLAG_FRUSTUM);
        const auto planes     = cam.getFrustumPlanes();
        u32 tested = 0;
        for (int i = 0; i < 20000; ++i) {
            const glm::vec3 inside = pos + glm::vec3(rng.range(-30, 30), rng.range(-30, 30), rng.range(-30, 30));
            if (!insidePlanes(planes, inside)) continue;
            // A sphere centred anywhere within r of an inside point intersects the frustum.
            glm::vec3 d(rng.range(-1, 1), rng.range(-1, 1), rng.range(-1, 1));
            if (glm::dot(d, d) < 1e-6f) continue;
            const float r     = rng.range(0.01f, 20.0f);
            const glm::vec3 c = inside + glm::normalize(d) * (r * 0.999f);
            ++tested;
            REQUIRE(reasonOf(p, c, r) == CULL_REASON_VISIBLE);
        }
        CHECK(tested > 100);
    }
    // Spheres straddling each plane (centre on the plane, any radius > 0): visible.
    const Camera cam = makeCamera({0, 0, 0}, -90.0f, 0.0f);
    const GPUCullParams p = paramsFor(cam, CULL_FLAG_FRUSTUM);
    const float tanY = std::tan(glm::radians(30.0f)), tanX = tanY * 16.0f / 9.0f;
    const glm::vec3 centres[] = {{-20 * tanX, 0, -20}, {20 * tanX, 0, -20}, {0, -20 * tanY, -20}, {0, 20 * tanY, -20}, {0, 0, -kNear}};
    for (const glm::vec3& c : centres) CHECK(reasonOf(p, c, 0.01f) == CULL_REASON_VISIBLE);
    // And entirely outside by more than the radius: culled.
    CHECK(reasonOf(p, {-20 * tanX - 1.0f, 0, -20}, 0.5f) == CULL_REASON_FRUSTUM);
    CHECK(reasonOf(p, {0, 0, -kNear + 0.2f}, 0.1f) == CULL_REASON_FRUSTUM); // in front of the near plane
}

// ---- reference kernels -------------------------------------------------------------------

TEST_CASE("makeCullParams fills the contract fields") {
    const Camera cam = makeCamera({1, 2, 3}, 10.0f, 5.0f);
    const GPUCullParams p = makeCullParams(cam.getViewProjection(), cam.getProjection()[1][1], 1080, cam.getPosition(), cam.getFront(),
                                           kNear, CULL_FLAG_FRUSTUM | CULL_FLAG_SIZE, 250.0f, 2.5f, 2049);
    const auto planes = cam.getFrustumPlanes();
    for (int i = 0; i < 5; ++i)
        for (int k = 0; k < 4; ++k) CHECK(p.planes[i * 4 + k] == planes[i][k]);
    CHECK(p.cameraPosition[0] == 1.0f);
    CHECK(p.cameraPosition[1] == 2.0f);
    CHECK(p.cameraPosition[2] == 3.0f);
    CHECK(p.maxDistance == 250.0f);
    CHECK(p.minPixels == 2.5f);
    CHECK(p.pixelScale == doctest::Approx(cam.getProjection()[1][1] * 540.0f));
    CHECK(p.nearPlane == kNear);
    CHECK(p.slotCount == 2049u);
    CHECK(p.groupCount == 3u); // ceil(2049 / 1024)
    CHECK(p.flags == (CULL_FLAG_FRUSTUM | CULL_FLAG_SIZE));
    // The size axis stored implicitly (near-plane normal) is the camera forward vector.
    CHECK(p.planes[16] == doctest::Approx(cam.getFront().x).epsilon(1e-4));
    CHECK(p.planes[17] == doctest::Approx(cam.getFront().y).epsilon(1e-4));
    CHECK(p.planes[18] == doctest::Approx(cam.getFront().z).epsilon(1e-4));
    CHECK(offsetof(GPUSceneCounters, tested) == SCENE_COUNTER_TESTED * 4);
    CHECK(offsetof(GPUSceneCounters, visible) == SCENE_COUNTER_VISIBLE * 4);
    CHECK(offsetof(GPUSceneCounters, culledFrustum) == SCENE_COUNTER_CULLED_FRUSTUM * 4);
    CHECK(offsetof(GPUSceneCounters, culledDistance) == SCENE_COUNTER_CULLED_DISTANCE * 4);
    CHECK(offsetof(GPUSceneCounters, culledSize) == SCENE_COUNTER_CULLED_SIZE * 4);
    CHECK(offsetof(GPUSceneCounters, drawCommands) == SCENE_COUNTER_DRAW_COMMANDS * 4);
    CHECK(offsetof(GPUSceneCounters, nodesUpdated) == SCENE_COUNTER_NODES_UPDATED * 4);
    CHECK(offsetof(GPUSceneCounters, queueOverflow) == SCENE_COUNTER_QUEUE_OVERFLOW * 4);
}

TEST_CASE("cullReference: reasons, invalid slots, mirrored instances, margins") {
    const Camera cam = makeCamera({0, 0, 0}, -90.0f, 0.0f);
    const std::vector<GPUMeshInfo> meshes = {makeMesh({0, 0, 0}, 1.0f), makeMesh({0, 0, -2.0f}, 0.5f)};
    const auto T = [](float x, float y, float z) { return glm::translate(glm::mat4(1.0f), glm::vec3(x, y, z)); };
    std::vector<GPUInstance> inst = {
        makeInstance(T(0, 0, -10), 0),                                                      // 0 visible
        makeInstance(T(0, 0, +10), 0),                                                      // 1 behind: frustum
        makeInstance(T(0, 0, -200), 0),                                                     // 2 distance
        makeInstance(T(0, 0, -50) * glm::scale(glm::mat4(1.0f), glm::vec3(0.01f)), 0),      // 3 size
        makeInstance(T(0, 0, 0), 0, 0),                                                     // 4 slack slot (no VALID)
        makeInstance(T(0, 0, -8) * glm::scale(glm::mat4(1.0f), glm::vec3(-1.0f, 1, 1)), 1,
                     INSTANCE_FLAG_VALID | INSTANCE_FLAG_MIRRORED),                         // 5 mirrored, visible (mesh centre offset)
        makeInstance(T(0, 0, 0), 1),                                                        // 6 sphere centre at z = -2: still visible
    };
    // The mirrored instance's world centre: model * (0,0,-2) = (0,0,-10).
    const GPUCullParams p = paramsFor(cam, CULL_FLAG_FRUSTUM | CULL_FLAG_DISTANCE | CULL_FLAG_SIZE, 100.0f, 2.0f,
                                      static_cast<u32>(inst.size()));
    std::vector<u8> result;
    std::vector<float> margins;
    cullReference(inst, meshes, p, result, &margins);
    REQUIRE(result.size() == inst.size());
    CHECK(result[0] == 0);
    CHECK(result[1] == CULL_REASON_FRUSTUM);
    CHECK(result[2] == CULL_REASON_DISTANCE);
    CHECK(result[3] == CULL_REASON_SIZE);
    CHECK(result[4] == CULL_RESULT_INVALID);
    CHECK(result[5] == 0);
    CHECK(result[6] == 0);
    CHECK(margins[4] == 0.0f);
    CHECK(margins[0] > 0.0f);
    CHECK(margins[1] < 0.0f);

    // Slot count smaller than the array: only the first slots are tested.
    GPUCullParams few = p;
    few.slotCount     = 2;
    cullReference(inst, meshes, few, result);
    CHECK(result.size() == 2);
}

TEST_CASE("compactReference: stable list and exclusive prefix") {
    const u8 inv = CULL_RESULT_INVALID;
    const std::vector<u8> result = {0, 1, inv, 0, 0, 2, 3, inv, 0};
    std::vector<u32> visible, prefix;
    compactReference(result, visible, prefix);
    CHECK(visible == std::vector<u32>{0, 3, 4, 8});
    CHECK(prefix == std::vector<u32>{0, 1, 1, 1, 2, 3, 3, 3, 3, 4});

    compactReference(std::vector<u8>{}, visible, prefix);
    CHECK(visible.empty());
    CHECK(prefix == std::vector<u32>{0});

    compactReference(std::vector<u8>{1, 2, 3, inv}, visible, prefix);
    CHECK(visible.empty());
    CHECK(prefix == std::vector<u32>{0, 0, 0, 0, 0});

    // Slot order, not input order of anything else: all visible -> identity.
    compactReference(std::vector<u8>(70, 0), visible, prefix);
    REQUIRE(visible.size() == 70);
    for (u32 i = 0; i < 70; ++i) CHECK(visible[i] == i);
    CHECK(prefix[70] == 70);
}

TEST_CASE("drawArgsReference: buckets, empty buckets, sentinels, slack slots") {
    // Slots: bucket 0 = [0, 8) (3 live + slack), bucket 1 = [8, 16) (all culled), bucket 2 = [16, 24), bucket 3 = [24, 28).
    std::vector<u8> result(28, CULL_RESULT_INVALID);
    for (u32 s : {0u, 1u, 2u}) result[s] = 0;                    // bucket 0: slack after the 3 live slots
    result[1] = 1;                                               // slot 1 culled
    for (u32 s = 8; s < 14; ++s) result[s] = 3;                  // bucket 1: live but culled
    for (u32 s = 16; s < 24; ++s) result[s] = (s % 2) ? 0 : 2;   // bucket 2: 4 visible
    for (u32 s = 24; s < 28; ++s) result[s] = 0;                 // bucket 3: all visible
    std::vector<u32> visible, prefix;
    compactReference(result, visible, prefix);

    const std::vector<GPUDrawBucket> buckets = {
        {0, 8, 0, 0, 36, 0, 0, 0},   {8, 8, 1, 0, 36, 36, 8, 1},
        {16, 8, 2, 1, 36, 72, 16, 3}, {24, 4, 3, 2, 12, 108, 24, 5},
    };
    // Commands: class 0: buckets 0, 1, sentinel; class 1: bucket 2, sentinel; class 2: bucket 3, sentinel.
    const u32 S = DRAW_COMMAND_SENTINEL;
    const std::vector<u32> commands = {0, 1, S, 2, S, 3, S};
    std::vector<u32> args;
    drawArgsReference(buckets, commands, prefix, args);
    REQUIRE(args.size() == 14);
    // bucket 0: visible slots {0, 2}; baseInstance = prefix[0] = 0
    CHECK(args[0] == 2);
    CHECK(args[1] == 0);
    // bucket 1: empty -> {0, 0}
    CHECK(args[2] == 0);
    CHECK(args[3] == 0);
    // sentinel
    CHECK(args[4] == 0);
    CHECK(args[5] == 0);
    // bucket 2: 4 visible, baseInstance = prefix[16] = 2
    CHECK(args[6] == 4);
    CHECK(args[7] == prefix[16]);
    CHECK(args[7] == 2);
    CHECK(args[8] == 0);
    // bucket 3: 4 visible, baseInstance = 6
    CHECK(args[10] == 4);
    CHECK(args[11] == 6);
    CHECK(args[12] == 0);
    CHECK(args[13] == 0);
    // Consistency: the instances of the draws tile the visible list in command order.
    CHECK(args[0] + args[6] + args[10] == visible.size());

    // No commands: nothing.
    drawArgsReference(buckets, std::vector<u32>{}, prefix, args);
    CHECK(args.empty());
}
