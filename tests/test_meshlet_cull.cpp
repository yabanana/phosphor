#include <doctest/doctest.h>

#include "renderer/cull_math.h"
#include "renderer/cull_reference.h"
#include "renderer/meshlet_cull_math.h"
#include "renderer/meshlet_cull_reference.h"
#include "renderer/scene_extract.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <array>
#include <cmath>
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
    u32 below(u32 n) { return u32(next() * float(n)) % n; }
    glm::vec3 unit() {
        for (;;) {
            const glm::vec3 v(range(-1, 1), range(-1, 1), range(-1, 1));
            const float l = glm::length(v);
            if (l > 0.05f && l <= 1.0f) return v / l;
        }
    }
};

// Reverse-Z infinite projection, exactly as Camera::updateMatrices.
glm::mat4 reverseZ(float fovY, float aspect, float nearPlane) {
    const float f = 1.0f / std::tan(fovY * 0.5f);
    glm::mat4 proj(0.0f);
    proj[0][0] = f / aspect;
    proj[1][1] = f;
    proj[2][2] = 0.0f;
    proj[2][3] = -1.0f;
    proj[3][2] = nearPlane;
    return proj;
}

glm::mat4 viewProjFor(glm::vec3 eye, glm::vec3 target, u32 vw, u32 vh, float fovY = glm::radians(60.0f)) {
    glm::vec3 fwd = glm::normalize(target - eye);
    glm::vec3 up = std::fabs(fwd.y) > 0.99f ? glm::vec3(0, 0, 1) : glm::vec3(0, 1, 0);
    return reverseZ(fovY, float(vw) / float(vh), kNear) * glm::lookAt(eye, target, up);
}

GPUMeshletCullParams makeParams(const glm::mat4& vp, const glm::mat4& prevVp, glm::vec3 cam, u32 vw, u32 vh, u32 flags) {
    GPUMeshletCullParams p{};
    std::memcpy(p.viewProj, glm::value_ptr(vp), sizeof(p.viewProj));
    std::memcpy(p.prevViewProj, glm::value_ptr(prevVp), sizeof(p.prevViewProj));
    const auto planes = extractFrustumPlanesReverseZ(vp);
    for (int i = 0; i < 5; ++i)
        for (int k = 0; k < 4; ++k) p.planes[i * 4 + k] = planes[i][k];
    p.cameraPosition[0] = cam.x;
    p.cameraPosition[1] = cam.y;
    p.cameraPosition[2] = cam.z;
    p.nearPlane = kNear;
    p.viewport[0] = float(vw);
    p.viewport[1] = float(vh);
    p.hizSize[0] = hizLevel0Size(vw);
    p.hizSize[1] = hizLevel0Size(vh);
    p.hizLevels = hizLevelCount(p.hizSize[0], p.hizSize[1]);
    p.flags = flags;
    p.minPixels = 2.0f;
    return p;
}

struct Projected {
    bool ok;     // clip w above the near plane
    double sx, sy;
    double depth; // reverse-Z NDC depth (clip z / w)
    double w;
};

Projected project(const glm::mat4& vp, glm::dvec3 p, u32 vw, u32 vh) {
    const glm::dvec4 c = glm::dmat4(vp) * glm::dvec4(p, 1.0);
    Projected r{};
    r.w = c.w;
    r.ok = c.w > double(kNear);
    if (!r.ok) return r;
    r.sx = (c.x / c.w * 0.5 + 0.5) * double(vw);
    r.sy = (0.5 - c.y / c.w * 0.5) * double(vh);
    r.depth = c.z / c.w;
    return r;
}

glm::dvec3 randomPointInSphere(Rng& rng, glm::dvec3 c, double r) {
    const glm::vec3 d = rng.unit();
    const double k = std::cbrt(double(rng.next()));
    return c + glm::dvec3(d) * (r * k);
}

// Largest singular value of the 3x3 part of a column-major matrix (double,
// power iteration on M^T M with many iterations, then a Rayleigh quotient).
double spectralNorm(const float* m) {
    double a[3][3]; // a[row][col]
    for (int c = 0; c < 3; ++c)
        for (int r = 0; r < 3; ++r) a[r][c] = m[c * 4 + r];
    double g[3][3] = {};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) g[i][j] += a[k][i] * a[k][j];
    double v[3] = {0.577, 0.5771, 0.5772};
    double lambda = 0.0;
    for (int it = 0; it < 2000; ++it) {
        double w[3] = {};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) w[i] += g[i][j] * v[j];
        const double n = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
        if (n == 0.0) return 0.0;
        for (int i = 0; i < 3; ++i) v[i] = w[i] / n;
        lambda = n;
    }
    return std::sqrt(lambda);
}

void fillMatrix(float* m, const glm::mat3& lin) {
    std::memset(m, 0, 16 * sizeof(float));
    for (int c = 0; c < 3; ++c)
        for (int r = 0; r < 3; ++r) m[c * 4 + r] = lin[c][r];
    m[15] = 1.0f;
}

GPUInstance makeInstance(glm::vec3 t, float scale = 1.0f) {
    GPUInstance in{};
    fillMatrix(in.modelMatrix, glm::mat3(scale));
    in.modelMatrix[12] = t.x;
    in.modelMatrix[13] = t.y;
    in.modelMatrix[14] = t.z;
    in.flags = INSTANCE_FLAG_VALID;
    return in;
}

GPUMeshletBounds sphereBounds(float r) {
    GPUMeshletBounds b{};
    b.radius = r;
    b.coneCutoff = 1.0f; // no cone
    return b;
}

} // namespace

TEST_CASE("meshlet cull: cullScaleBound is an upper bound of the spectral norm") {
    Rng rng{0xABCDEF12ull};
    for (int i = 0; i < 1000; ++i) {
        glm::mat3 lin;
        const int kind = i % 4;
        if (kind == 0) { // random
            for (int c = 0; c < 3; ++c)
                for (int r = 0; r < 3; ++r) lin[c][r] = rng.range(-3, 3);
        } else if (kind == 1) { // shear
            lin = glm::mat3(1.0f);
            lin[1][0] = rng.range(-4, 4);
            lin[2][0] = rng.range(-4, 4);
            lin[2][1] = rng.range(-4, 4);
            lin = lin * glm::mat3(rng.range(0.1f, 3), 0, 0, 0, rng.range(0.1f, 3), 0, 0, 0, rng.range(0.1f, 3));
        } else if (kind == 2) { // mirror * rotation * scale
            const glm::mat4 rot = glm::rotate(glm::mat4(1.0f), rng.range(0, 6.28f), rng.unit());
            lin = glm::mat3(rot) * glm::mat3(-rng.range(0.1f, 4), 0, 0, 0, rng.range(0.1f, 4), 0, 0, 0, rng.range(0.1f, 4));
        } else { // near-singular
            lin = glm::mat3(1.0f, 0.0f, 0.0f, 1.0f, 1e-3f, 0.0f, rng.range(-1, 1), rng.range(-1, 1), 1e-4f);
        }
        float m[16];
        fillMatrix(m, lin);
        const double sigma = spectralNorm(m);
        const double bound = cullScaleBound(m);
        CAPTURE(i);
        CHECK(bound >= sigma);
        CHECK(bound <= sigma * 3.0 + 1e-6); // Gershgorin is within sqrt(3) of the norm
    }
    // Rotation x diagonal scale: the longest column, within the 2^-16 inflation.
    for (int i = 0; i < 200; ++i) {
        const glm::mat4 rot = glm::rotate(glm::mat4(1.0f), rng.range(0, 6.28f), rng.unit());
        const glm::vec3 s(rng.range(0.2f, 4), rng.range(0.2f, 4), rng.range(0.2f, 4));
        float m[16];
        fillMatrix(m, glm::mat3(rot) * glm::mat3(s.x, 0, 0, 0, s.y, 0, 0, 0, s.z));
        const double longest = std::max({double(s.x), double(s.y), double(s.z)});
        const double bound = cullScaleBound(m);
        CHECK(bound >= longest * (1.0 - 1e-6));
        // The documented inflation is 2^-16 = 1.53e-5 (plus float rounding).
        CHECK(bound <= longest * (1.0 + 2.5e-5));
    }
}

TEST_CASE("meshlet cull: frustum test never culls a sphere with a point inside") {
    Rng rng{0x1234567ull};
    u32 culled = 0, kept = 0;
    for (int cam = 0; cam < 20; ++cam) {
        const glm::vec3 eye(rng.range(-5, 5), rng.range(-5, 5), rng.range(-5, 5));
        const glm::vec3 target(rng.range(-3, 3), rng.range(-3, 3), rng.range(-3, 3));
        if (glm::length(target - eye) < 1.0f) continue;
        const glm::mat4 vp = viewProjFor(eye, target, 1920, 1080);
        const GPUMeshletCullParams p = makeParams(vp, vp, eye, 1920, 1080, MESHLET_CULL_FRUSTUM);
        const auto planes = extractFrustumPlanesReverseZ(vp);
        const glm::vec3 fwd = glm::normalize(target - eye);
        for (int s = 0; s < 300; ++s) {
            const glm::vec3 c = eye + fwd * rng.range(-2, 30) + glm::vec3(rng.range(-25, 25), rng.range(-25, 25), rng.range(-25, 25)) * 0.6f;
            const float r = rng.range(0.05f, 4.0f);
            const GPUWorldSphere w{c.x, c.y, c.z, r};
            bool anyInside = false;
            for (int k = 0; k < 400 && !anyInside; ++k) {
                const glm::dvec3 pt = randomPointInSphere(rng, glm::dvec3(c), r);
                bool in = true;
                for (const glm::vec4& pl : planes) in = in && (pl.x * pt.x + pl.y * pt.y + pl.z * pt.z + pl.w >= 1e-4);
                anyInside = in;
            }
            const bool isCulled = meshletFrustumCulled(p, w);
            if (anyInside) CHECK_FALSE(isCulled);
            (isCulled ? culled : kept)++;
        }
    }
    CHECK(culled > 100);
    CHECK(kept > 100);
}

namespace {
struct Viewport {
    u32 w, h;
};
const Viewport kViewports[] = {{1920, 1080}, {1919, 1081}, {64, 7}};
} // namespace

TEST_CASE("meshlet cull: Hi-Z footprint is conservative") {
    Rng rng{0xF00DCAFEull};
    u32 usableCount = 0, unusableCount = 0, pointChecks = 0;
    for (const Viewport& vpx : kViewports) {
        for (int c = 0; c < 15; ++c) {
            const glm::vec3 eye(rng.range(-5, 5), rng.range(-5, 5), rng.range(-5, 5));
            const glm::vec3 target(rng.range(-3, 3), rng.range(-3, 3), rng.range(-3, 3));
            if (glm::length(target - eye) < 1.0f) continue;
            const glm::mat4 vp = viewProjFor(eye, target, vpx.w, vpx.h);
            const GPUMeshletCullParams p = makeParams(vp, vp, eye, vpx.w, vpx.h, 0);
            const glm::vec3 fwd = glm::normalize(target - eye);
            for (int s = 0; s < 80; ++s) {
                const float dist = rng.range(-1.0f, 40.0f);
                const glm::vec3 c = eye + fwd * dist + glm::vec3(rng.range(-1, 1), rng.range(-1, 1), rng.range(-1, 1)) * dist * 0.5f;
                const float r = rng.range(0.01f, 3.0f) * (s % 7 == 0 ? 5.0f : 1.0f);
                const GPUWorldSphere w{c.x, c.y, c.z, r};
                const HiZFootprint f = hizFootprint(p.viewProj, w, p);

                // Crossing the near plane / behind the camera: not testable.
                const double cw = (glm::dmat4(vp) * glm::dvec4(glm::dvec3(c), 1.0)).w;
                if (cw - double(r) <= double(kNear) - 1e-4) {
                    CHECK(f.usable == 0);
                    ++unusableCount;
                    continue;
                }
                if (f.usable == 0) {
                    ++unusableCount;
                    continue;
                }
                ++usableCount;
                CHECK(f.x1 - f.x0 < HIZ_TEST_SPAN);
                CHECK(f.y1 - f.y0 < HIZ_TEST_SPAN);
                CHECK(f.x0 <= f.x1);
                CHECK(f.y0 <= f.y1);
                REQUIRE(f.level < p.hizLevels);
                CHECK(f.x1 < hizLevelSize(p.hizSize[0], f.level));
                CHECK(f.y1 < hizLevelSize(p.hizSize[1], f.level));
                const u32 sh = f.level + 1;
                const i64 lox = i64(f.x0) << sh, hix = (i64(f.x1 + 1) << sh) - 1;
                const i64 loy = i64(f.y0) << sh, hiy = (i64(f.y1 + 1) << sh) - 1;
                for (int k = 0; k < 500; ++k) {
                    const glm::dvec3 pt = randomPointInSphere(rng, glm::dvec3(c), r);
                    const Projected pr = project(vp, pt, vpx.w, vpx.h);
                    if (!pr.ok) continue;
                    if (pr.sx < 0 || pr.sy < 0 || pr.sx >= vpx.w || pr.sy >= vpx.h) continue;
                    const i64 px = i64(std::floor(pr.sx)), py = i64(std::floor(pr.sy));
                    ++pointChecks;
                    if (px < lox || px > hix || py < loy || py > hiy) FAIL("projected pixel outside the footprint rectangle");
                    if (pr.depth > double(f.nearestDepth) * (1.0 + 1e-6)) FAIL("a sphere point is nearer than nearestDepth");
                }
            }
        }
    }
    CHECK(usableCount > 200);
    CHECK(unusableCount > 50);
    CHECK(pointChecks > 10000);
}

namespace {

// Brute-force texel value: min of the depth pixels covered, 1.0 when none.
float bruteTexel(const std::vector<float>& depth, u32 w, u32 h, u32 level, u32 x, u32 y) {
    const u64 span = u64(1) << (level + 1);
    const u64 x0 = u64(x) * span, y0 = u64(y) * span;
    float m = 1.0f;
    for (u64 py = y0; py < std::min<u64>(y0 + span, h); ++py)
        for (u64 px = x0; px < std::min<u64>(x0 + span, w); ++px) m = std::min(m, depth[py * w + px]);
    return m;
}

bool covered(u32 w, u32 h, u32 level, u32 x, u32 y) {
    const u64 span = u64(1) << (level + 1);
    return u64(x) * span < w && u64(y) * span < h;
}

bool coversPixel(u32 level, u32 x, u32 y, u32 px, u32 py) {
    return (px >> (level + 1)) == x && (py >> (level + 1)) == y;
}

} // namespace

TEST_CASE("meshlet cull: Hi-Z reference sizes and exactness") {
    struct Case {
        u32 w, h;
    };
    const Case cases[] = {{1, 1}, {1, 17}, {17, 1}, {1920, 1080}, {1919, 1081}, {3200, 1800}};
    Rng rng{0x777ull};
    for (const Case& cs : cases) {
        CAPTURE(cs.w);
        CAPTURE(cs.h);
        std::vector<float> depth(size_t(cs.w) * cs.h);
        for (float& d : depth) d = rng.next();
        const HiZPyramid pyr = buildHiZReference(depth.data(), cs.w, cs.h);

        // Sizes: power of two >= ceil(v / 2) (and the smallest such), last level 1x1.
        const u32 hw = (cs.w + 1) / 2, hh = (cs.h + 1) / 2;
        CHECK(pyr.width0 >= hw);
        CHECK((pyr.width0 & (pyr.width0 - 1)) == 0);
        CHECK((pyr.width0 == 1 || pyr.width0 / 2 < hw));
        CHECK(pyr.height0 >= hh);
        CHECK((pyr.height0 & (pyr.height0 - 1)) == 0);
        CHECK((pyr.height0 == 1 || pyr.height0 / 2 < hh));
        REQUIRE(pyr.levels >= 1);
        REQUIRE(pyr.level.size() == pyr.levels);
        CHECK(hizLevelSize(pyr.width0, pyr.levels - 1) == 1);
        CHECK(hizLevelSize(pyr.height0, pyr.levels - 1) == 1);
        if (pyr.levels > 1) CHECK((hizLevelSize(pyr.width0, pyr.levels - 2) > 1 || hizLevelSize(pyr.height0, pyr.levels - 2) > 1));

        for (u32 l = 0; l < pyr.levels; ++l) {
            const u32 lw = hizLevelSize(pyr.width0, l), lh = hizLevelSize(pyr.height0, l);
            REQUIRE(pyr.level[l].size() == size_t(lw) * lh);
            u64 bad = 0;
            for (u32 y = 0; y < lh; ++y)
                for (u32 x = 0; x < lw; ++x)
                    if (pyr.at(l, x, y) != bruteTexel(depth, cs.w, cs.h, l, x, y)) ++bad;
            CHECK(bad == 0);
        }
    }
}

TEST_CASE("meshlet cull: Hi-Z holes, empty and full occluders") {
    const u32 w = 100, h = 37;
    {
        std::vector<float> depth(size_t(w) * h, 0.5f);
        const u32 hx = 63, hy = 20;
        depth[size_t(hy) * w + hx] = 0.0f;
        const HiZPyramid pyr = buildHiZReference(depth.data(), w, h);
        for (u32 l = 0; l < pyr.levels; ++l) {
            for (u32 y = 0; y < hizLevelSize(pyr.height0, l); ++y)
                for (u32 x = 0; x < hizLevelSize(pyr.width0, l); ++x) {
                    float expect = covered(w, h, l, x, y) ? 0.5f : 1.0f;
                    if (coversPixel(l, x, y, hx, hy)) expect = 0.0f;
                    CHECK(pyr.at(l, x, y) == expect);
                }
        }
        // The top level holds the hole.
        CHECK(pyr.at(pyr.levels - 1, 0, 0) == 0.0f);
    }
    for (float value : {0.0f, 0.8f}) {
        std::vector<float> depth(size_t(w) * h, value);
        const HiZPyramid pyr = buildHiZReference(depth.data(), w, h);
        for (u32 l = 0; l < pyr.levels; ++l)
            for (u32 y = 0; y < hizLevelSize(pyr.height0, l); ++y)
                for (u32 x = 0; x < hizLevelSize(pyr.width0, l); ++x)
                    CHECK(pyr.at(l, x, y) == (covered(w, h, l, x, y) ? value : 1.0f));
    }
}

TEST_CASE("meshlet cull: occlusion never hides a visible sphere, hides a covered one") {
    const u32 vw = 256, vh = 256;
    Rng rng{0xBEEF1234ull};

    SUBCASE("random depth images, brute-force visibility") {
        u32 occludedCount = 0, visibleChecks = 0;
        for (int img = 0; img < 25; ++img) {
            std::vector<float> depth(size_t(vw) * vh, rng.next() < 0.5f ? 0.0f : rng.range(0.0f, 0.4f));
            const int rects = 1 + int(rng.below(5));
            for (int r = 0; r < rects; ++r) {
                const u32 x0 = rng.below(vw), y0 = rng.below(vh);
                const u32 x1 = std::min(vw, x0 + 1 + rng.below(120)), y1 = std::min(vh, y0 + 1 + rng.below(120));
                const float d = rng.next() < 0.2f ? 0.0f : rng.range(0.0f, 1.0f);
                for (u32 y = y0; y < y1; ++y)
                    for (u32 x = x0; x < x1; ++x) depth[size_t(y) * vw + x] = d;
            }
            const HiZPyramid pyr = buildHiZReference(depth.data(), vw, vh);
            const glm::vec3 eye(0, 0, 0);
            const glm::mat4 vp = viewProjFor(eye, glm::vec3(0, 0, -1), vw, vh);
            const GPUMeshletCullParams p = makeParams(vp, vp, eye, vw, vh, 0);
            for (int s = 0; s < 150; ++s) {
                const float z = rng.range(0.3f, 60.0f);
                const glm::vec3 c(rng.range(-0.8f, 0.8f) * z, rng.range(-0.8f, 0.8f) * z, -z);
                const float r = rng.range(0.02f, 0.3f) * z * (s % 5 == 0 ? 0.2f : 1.0f);
                const GPUWorldSphere w{c.x, c.y, c.z, r};
                const HiZFootprint f = hizFootprint(p.viewProj, w, p);
                const bool occluded = f.usable != 0 && hizOccluded(f.nearestDepth, hizMinOverFootprint(pyr, f));
                bool visible = false;
                for (int k = 0; k < 300 && !visible; ++k) {
                    const glm::dvec3 pt = randomPointInSphere(rng, glm::dvec3(c), r);
                    const Projected pr = project(vp, pt, vw, vh);
                    if (!pr.ok || pr.sx < 0 || pr.sy < 0 || pr.sx >= vw || pr.sy >= vh) continue;
                    const float img = depth[size_t(std::floor(pr.sy)) * vw + size_t(std::floor(pr.sx))];
                    if (double(img) <= pr.depth) visible = true;
                }
                if (visible) {
                    ++visibleChecks;
                    if (occluded) FAIL("a sphere visible at one of its points was reported occluded");
                }
                if (occluded) ++occludedCount;
            }
        }
        CHECK(occludedCount > 50);
        CHECK(visibleChecks > 50);
    }

    SUBCASE("full-screen occluder: positive and negative control") {
        const float occluderZ = 5.0f; // view distance of the occluder
        std::vector<float> depth(size_t(vw) * vh, kNear / occluderZ);
        const HiZPyramid pyr = buildHiZReference(depth.data(), vw, vh);
        const glm::vec3 eye(0, 0, 0);
        const glm::mat4 vp = viewProjFor(eye, glm::vec3(0, 0, -1), vw, vh);
        const GPUMeshletCullParams p = makeParams(vp, vp, eye, vw, vh, 0);
        auto test = [&](glm::vec3 c, float r) {
            const GPUWorldSphere w{c.x, c.y, c.z, r};
            const HiZFootprint f = hizFootprint(p.viewProj, w, p);
            REQUIRE(f.usable != 0);
            return hizOccluded(f.nearestDepth, hizMinOverFootprint(pyr, f));
        };
        CHECK(test(glm::vec3(0, 0, -20), 1.0f));          // far behind, centered
        CHECK(test(glm::vec3(2, 1, -30), 2.0f));          // far behind
        CHECK_FALSE(test(glm::vec3(0, 0, -2), 0.5f));      // in front of the occluder
        CHECK_FALSE(test(glm::vec3(0, 0, -6), 3.0f));      // straddles the occluder
        // Margin helper agrees: positive occluded, negative not.
        CHECK(hizOcclusionMargin(0.01f, 0.5f) > 0.0f);
        CHECK(hizOcclusionMargin(0.9f, 0.5f) < 0.0f);
        CHECK(hizOcclusionMargin(0.5f, 0.0f) < 0.0f);
    }
}

namespace {

struct Scene {
    std::vector<GPUMeshInfo> meshes;
    std::vector<GPUMaterial> materials;
    std::vector<GPUInstance> instances;
    std::vector<u32> flags;
};

Scene handScene() {
    Scene s;
    auto mesh = [&](u32 count, u32 offset) {
        GPUMeshInfo m{};
        m.meshletCount = count;
        m.meshletOffset = offset;
        s.meshes.push_back(m);
    };
    mesh(0, 100); // 0
    mesh(1, 10);  // 1
    mesh(3, 20);  // 2
    GPUMaterial normal{}, dbl{};
    dbl.flags = MATERIAL_FLAG_DOUBLE_SIDED;
    s.materials = {normal, dbl};
    auto inst = [&](u32 meshIdx, u32 mat, u32 flags, u32 sceneFlags) {
        GPUInstance in{};
        in.meshIndex = meshIdx;
        in.materialIndex = mat;
        in.flags = flags;
        s.instances.push_back(in);
        s.flags.push_back(sceneFlags);
    };
    const u32 V = INSTANCE_FLAG_VALID, M = INSTANCE_FLAG_MIRRORED;
    inst(2, 0, V, 1);     // 0: class 0, 3 meshlets
    inst(1, 0, V | M, 1); // 1: class 1
    inst(1, 1, V, 1);     // 2: class 2 (double sided)
    inst(2, 0, V, 0);     // 3: not visible
    inst(2, 0, 0, 1);     // 4: invalid slot
    inst(0, 0, V, 1);     // 5: mesh without meshlets
    inst(1, 1, V | M, 1); // 6: double sided wins over mirrored: class 2
    inst(1, 0, V, 3);     // 7: class 0, other flag bits set
    return s;
}

void checkList(const MeshletListRef& r, const std::vector<GPUMeshletCandidate>& expect, const std::array<u32, 3>& first,
               const std::array<u32, 3>& count, u32 phase) {
    REQUIRE(r.list.size() == expect.size());
    CHECK(r.total == expect.size());
    for (size_t i = 0; i < expect.size(); ++i) {
        CHECK(r.list[i].slot == expect[i].slot);
        CHECK(r.list[i].meshlet == expect[i].meshlet);
    }
    for (u32 c = 0; c < 3; ++c) {
        CHECK(r.ranges[c].first == first[c]);
        CHECK(r.ranges[c].count == count[c]);
        CHECK(r.ranges[c].cullClass == c);
        CHECK(r.ranges[c].phase == phase);
    }
}

} // namespace

TEST_CASE("meshlet cull: candidate and B lists (hand-made scene)") {
    const Scene s = handScene();
    const MeshletListRef cand = buildCandidatesReference(s.instances, s.meshes, s.materials, s.flags, 8);
    const std::vector<GPUMeshletCandidate> expect = {{0, 20}, {0, 21}, {0, 22}, {7, 10}, {1, 10}, {2, 10}, {6, 10}};
    checkList(cand, expect, {0, 4, 5}, {4, 1, 2}, MESHLET_PHASE_A);

    // slotCount limits the scan.
    const MeshletListRef cand7 = buildCandidatesReference(s.instances, s.meshes, s.materials, s.flags, 7);
    checkList(cand7, {{0, 20}, {0, 21}, {0, 22}, {1, 10}, {2, 10}, {6, 10}}, {0, 3, 4}, {3, 1, 2}, MESHLET_PHASE_A);

    // Empty scene.
    const MeshletListRef none = buildCandidatesReference(s.instances, s.meshes, s.materials, s.flags, 0);
    checkList(none, {}, {0, 0, 0}, {0, 0, 0}, MESHLET_PHASE_A);

    // B list.
    const std::vector<u32> all(7, 1), zero(7, 0), alt = {1, 0, 1, 0, 1, 0, 1};
    checkList(buildBListReference(cand, all), expect, {0, 4, 5}, {4, 1, 2}, MESHLET_PHASE_B);
    checkList(buildBListReference(cand, zero), {}, {0, 0, 0}, {0, 0, 0}, MESHLET_PHASE_B);
    checkList(buildBListReference(cand, alt), {{0, 20}, {0, 22}, {1, 10}, {6, 10}}, {0, 2, 3}, {2, 1, 1}, MESHLET_PHASE_B);
    // Non-zero but not 1 counts as set.
    const std::vector<u32> big = {0, 7, 0, 0, 0, 0, 0};
    checkList(buildBListReference(cand, big), {{0, 21}}, {0, 1, 1}, {1, 0, 0}, MESHLET_PHASE_B);
}

TEST_CASE("meshlet cull: candidate capacity bounds every frame") {
    Rng rng{0x424242ull};
    for (int trial = 0; trial < 50; ++trial) {
        Scene s;
        const u32 meshCount = 1 + rng.below(5);
        for (u32 i = 0; i < meshCount; ++i) {
            GPUMeshInfo m{};
            m.meshletCount = rng.below(9); // 0 allowed
            m.meshletOffset = i * 16;
            s.meshes.push_back(m);
        }
        GPUMaterial normal{}, dbl{};
        dbl.flags = MATERIAL_FLAG_DOUBLE_SIDED;
        s.materials = {normal, dbl};
        std::vector<SceneBucket> buckets;
        u32 slot = 0;
        const u32 bucketCount = 1 + rng.below(6);
        for (u32 b = 0; b < bucketCount; ++b) {
            SceneBucket bk;
            bk.mesh = rng.below(meshCount);
            bk.firstSlot = slot;
            bk.capacity = 1 + rng.below(20);
            bk.cull = CullClass::Back;
            buckets.push_back(bk);
            for (u32 i = 0; i < bk.capacity; ++i) {
                GPUInstance in{};
                in.meshIndex = bk.mesh;
                in.materialIndex = rng.below(2);
                in.flags = (rng.next() < 0.8f ? INSTANCE_FLAG_VALID : 0u) | (rng.next() < 0.3f ? INSTANCE_FLAG_MIRRORED : 0u);
                s.instances.push_back(in);
                s.flags.push_back(rng.next() < 0.7f ? 1u : 0u);
            }
            slot += bk.capacity;
        }
        const MeshletListRef r = buildCandidatesReference(s.instances, s.meshes, s.materials, s.flags, slot);
        const u64 cap = meshletCandidateCapacity(buckets, s.meshes);
        CHECK(cap >= r.total);
        // All visible: the bound is reached exactly.
        std::vector<u32> allVisible(slot, 1);
        std::vector<GPUInstance> allValid = s.instances;
        for (GPUInstance& in : allValid) in.flags |= INSTANCE_FLAG_VALID;
        const MeshletListRef full = buildCandidatesReference(allValid, s.meshes, s.materials, allVisible, slot);
        CHECK(full.total == cap);
    }
}

TEST_CASE("meshlet cull: phase A / B decision order and flag gating") {
    const u32 vw = 256, vh = 256;
    const glm::vec3 eye(0, 0, 5);
    const glm::mat4 vp = viewProjFor(eye, glm::vec3(0, 0, 0), vw, vh);
    const GPUInstance inst = makeInstance(glm::vec3(0));
    GPUMeshletBounds bounds = sphereBounds(0.5f);
    // A cone that rejects the camera at +Z: normals face -Z.
    GPUMeshletBounds coneBounds = bounds;
    coneBounds.coneAxis[2] = -1.0f;
    coneBounds.coneCutoff = 0.0f;

    const std::vector<float> occluder(size_t(vw) * vh, 0.9f), clear(size_t(vw) * vh, 0.0f);
    const HiZPyramid history = buildHiZReference(occluder.data(), vw, vh);
    const HiZPyramid empty = buildHiZReference(clear.data(), vw, vh);
    auto phaseA = [&](u32 flags, const GPUMeshletBounds& b, u32 cls, const GPUInstance& in, const HiZPyramid* h) {
        const GPUMeshletCullParams p = makeParams(vp, vp, eye, vw, vh, flags);
        const MeshletDecisionInput di{&in, &b, cls};
        return meshletDecisionPhaseA(p, di, h);
    };
    const u32 F = MESHLET_CULL_FRUSTUM, C = MESHLET_CULL_CONE, O = MESHLET_CULL_OCCLUSION, H = MESHLET_CULL_HISTORY_VALID,
              S = MESHLET_CULL_SIZE;

    // Nothing enabled: drawn.
    CHECK(phaseA(0, coneBounds, 0, inst, &history) == MESHLET_DECISION_DRAWN_A);

    // Frustum before cone.
    const GPUInstance outside = makeInstance(glm::vec3(500, 0, 0));
    CHECK(phaseA(F | C, coneBounds, 0, outside, nullptr) == MESHLET_DECISION_FRUSTUM);
    CHECK(phaseA(C, coneBounds, 0, outside, nullptr) == MESHLET_DECISION_CONE);
    CHECK(phaseA(F, bounds, 0, outside, nullptr) == MESHLET_DECISION_FRUSTUM);
    CHECK(phaseA(F, bounds, 0, inst, nullptr) == MESHLET_DECISION_DRAWN_A);

    // Cone: class None skips it, flag gates it, no-cone bounds are drawn.
    CHECK(phaseA(F | C, coneBounds, 0, inst, nullptr) == MESHLET_DECISION_CONE);
    CHECK(phaseA(F | C, coneBounds, 1, inst, nullptr) == MESHLET_DECISION_CONE);
    CHECK(phaseA(F | C, coneBounds, 2, inst, nullptr) == MESHLET_DECISION_DRAWN_A);
    CHECK(phaseA(F, coneBounds, 0, inst, nullptr) == MESHLET_DECISION_DRAWN_A);
    CHECK(phaseA(F | C, bounds, 0, inst, nullptr) == MESHLET_DECISION_DRAWN_A);

    // Size: only with the flag; cone comes first, history after.
    {
        GPUMeshletCullParams p = makeParams(vp, vp, eye, vw, vh, F | S);
        p.minPixels = 100000.0f;
        const MeshletDecisionInput di{&inst, &bounds, 0};
        CHECK(meshletDecisionPhaseA(p, di, nullptr) == MESHLET_DECISION_SIZE);
        p.flags = F;
        CHECK(meshletDecisionPhaseA(p, di, nullptr) == MESHLET_DECISION_DRAWN_A);
        p.flags = F | S | C;
        const MeshletDecisionInput dc{&inst, &coneBounds, 0};
        CHECK(meshletDecisionPhaseA(p, dc, nullptr) == MESHLET_DECISION_CONE);
        p.flags = F | S | O | H;
        CHECK(meshletDecisionPhaseA(p, di, &history) == MESHLET_DECISION_SIZE);
        p.minPixels = 0.5f; // projected sphere is larger than this
        CHECK(meshletDecisionPhaseA(p, di, &history) == MESHLET_DECISION_HISTORY);
    }

    // History: needs OCCLUSION and HISTORY_VALID and a pyramid; a clear pyramid never occludes.
    CHECK(phaseA(F | O | H, bounds, 0, inst, &history) == MESHLET_DECISION_HISTORY);
    CHECK(phaseA(F | O, bounds, 0, inst, &history) == MESHLET_DECISION_DRAWN_A);
    CHECK(phaseA(F | H, bounds, 0, inst, &history) == MESHLET_DECISION_DRAWN_A);
    CHECK(phaseA(F | O | H, bounds, 0, inst, nullptr) == MESHLET_DECISION_DRAWN_A);
    CHECK(phaseA(F | O | H, bounds, 0, inst, &empty) == MESHLET_DECISION_DRAWN_A);
    CHECK(phaseA(F | C | O | H, coneBounds, 0, inst, &history) == MESHLET_DECISION_CONE);
    // History runs for class None too (the cone is the only class-dependent test).
    CHECK(phaseA(F | C | O | H, coneBounds, 2, inst, &history) == MESHLET_DECISION_HISTORY);
    // A meshlet behind the camera cannot be tested (visible), but the frustum culls it first.
    const GPUInstance behind = makeInstance(glm::vec3(0, 0, 50));
    CHECK(phaseA(O | H, bounds, 0, behind, &history) == MESHLET_DECISION_DRAWN_A);
    CHECK(phaseA(F | O | H, bounds, 0, behind, &history) == MESHLET_DECISION_FRUSTUM);

    // Phase B.
    {
        const GPUMeshletCullParams p = makeParams(vp, vp, eye, vw, vh, F | O);
        const MeshletDecisionInput di{&inst, &bounds, 0};
        CHECK(meshletDecisionPhaseB(p, di, history) == MESHLET_DECISION_OCCLUDED);
        CHECK(meshletDecisionPhaseB(p, di, empty) == MESHLET_DECISION_DRAWN_B);
        const MeshletDecisionInput db{&behind, &bounds, 0};
        CHECK(meshletDecisionPhaseB(p, db, history) == MESHLET_DECISION_DRAWN_B);
    }
}
