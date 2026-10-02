#include <doctest/doctest.h>

#include "renderer/meshlet_builder.h"
#include "renderer/meshlet_cull_math.h"
#include "scene/procedural.h"

#include <glm/glm.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace phosphor;

namespace {

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
};

struct TestMesh {
    std::string name;
    std::vector<glm::vec3> positions;
    std::vector<u32> indices;
};

TestMesh fromData(const char* name, const MeshData& d) { return {name, d.positions, d.indices}; }

TestMesh makeSoup(u32 seed, u32 vertices, u32 triangles) {
    Rng rng{0x9E3779B97F4A7C15ull ^ seed};
    TestMesh m{"soup", {}, {}};
    for (u32 i = 0; i < vertices; ++i) m.positions.emplace_back(rng.range(-1, 1), rng.range(-1, 1), rng.range(-1, 1));
    for (u32 t = 0; t < triangles; ++t) {
        u32 a = rng.below(vertices), b = rng.below(vertices), c = rng.below(vertices);
        while (b == a) b = rng.below(vertices);
        while (c == a || c == b) c = rng.below(vertices);
        m.indices.insert(m.indices.end(), {a, b, c});
    }
    return m;
}

// Triangles with repeated indices, collinear and coincident points mixed with
// regular ones.
TestMesh makeDegenerate() {
    Rng rng{0xDEADBEEFull};
    TestMesh m{"degenerate", {}, {}};
    for (u32 i = 0; i < 40; ++i) m.positions.emplace_back(rng.range(-1, 1), rng.range(-1, 1), rng.range(-1, 1));
    // collinear and coincident vertices
    m.positions.emplace_back(0, 0, 0);   // 40
    m.positions.emplace_back(1, 1, 1);   // 41
    m.positions.emplace_back(2, 2, 2);   // 42 (collinear with 40, 41)
    m.positions.emplace_back(0, 0, 0);   // 43 (coincident with 40)
    for (u32 t = 0; t < 150; ++t) {
        if (t % 5 == 0) {
            const u32 kind = (t / 5) % 3;
            const u32 a = rng.below(40), b = rng.below(40);
            if (kind == 0) m.indices.insert(m.indices.end(), {40, 41, 42});         // collinear
            else if (kind == 1) m.indices.insert(m.indices.end(), {40, 43, a});      // coincident points
            else m.indices.insert(m.indices.end(), {a, a == b ? (b + 1) % 40 : b, a}); // repeated index
        } else {
            u32 a = rng.below(40), b = rng.below(40), c = rng.below(40);
            while (b == a) b = rng.below(40);
            while (c == a || c == b) c = rng.below(40);
            m.indices.insert(m.indices.end(), {a, b, c});
        }
    }
    return m;
}

// Long thin triangles (aspect ratio 1e4) plus a few regular ones.
TestMesh makeSlivers() {
    Rng rng{0x51157ull};
    TestMesh m{"slivers", {}, {}};
    for (u32 t = 0; t < 200; ++t) {
        const glm::vec3 o(rng.range(-1, 1), rng.range(-1, 1), rng.range(-1, 1));
        const glm::vec3 dir = glm::normalize(glm::vec3(rng.range(-1, 1), rng.range(-1, 1), rng.range(-1, 1)) + glm::vec3(0.01f));
        const glm::vec3 side = glm::normalize(glm::cross(dir, glm::vec3(0.3f, 0.5f, 0.8f)));
        const u32 base = u32(m.positions.size());
        m.positions.push_back(o);
        m.positions.push_back(o + dir * 1.5f);
        m.positions.push_back(o + dir * 0.7f + side * 1e-4f);
        m.indices.insert(m.indices.end(), {base, base + 1, base + 2});
    }
    return m;
}

std::vector<TestMesh> testMeshes() {
    std::vector<TestMesh> v;
    v.push_back(fromData("cube", ProceduralMeshes::generateCube(1.0f)));
    v.push_back(fromData("sphere", ProceduralMeshes::generateSphere(1.0f, 48, 24)));
    v.push_back(fromData("torus", ProceduralMeshes::generateTorus(1.0f, 0.3f, 40, 20)));
    v.push_back(makeSoup(1, 300, 700));
    v.push_back(makeDegenerate());
    v.push_back(makeSlivers());
    return v;
}

std::vector<MeshletBuildOptions> optionSets() {
    std::vector<MeshletBuildOptions> v;
    auto std_ = [](u32 mv, u32 mt, bool opt = false) {
        MeshletBuildOptions o;
        o.maxVertices = mv;
        o.maxTriangles = mt;
        o.optimize = opt;
        return o;
    };
    v.push_back(MeshletBuildOptions{}); // 64 / 124 default
    v.push_back(std_(64, 64));
    v.push_back(std_(64, 96));
    v.push_back(std_(64, 128));
    v.push_back(std_(128, 128));
    for (u32 t : {124u, 96u}) {
        MeshletBuildOptions o;
        o.algorithm = MeshletAlgorithm::Spatial;
        o.maxVertices = 64;
        o.maxTriangles = t;
        v.push_back(o);
    }
    v.push_back(std_(64, 124, true));
    return v;
}

using Tri = std::array<u32, 3>;

// Rotation-invariant, orientation-preserving key: the lexicographically
// smallest cyclic rotation.
Tri canonical(Tri t) {
    Tri best = t;
    for (int r = 1; r < 3; ++r) {
        const Tri c{t[(0 + r) % 3], t[(1 + r) % 3], t[(2 + r) % 3]};
        if (c < best) best = c;
    }
    return best;
}

bool isDegenerateIndices(const Tri& t) { return t[0] == t[1] || t[1] == t[2] || t[0] == t[2]; }

glm::dvec3 P(const TestMesh& m, u32 i) { return glm::dvec3(m.positions[i]); }

MeshletBuildResult cook(const TestMesh& m, const MeshletBuildOptions& o) {
    return MeshletBuilder::build(&m.positions[0].x, m.positions.size(), sizeof(glm::vec3), m.indices.data(), m.indices.size(), o);
}

Tri meshletTri(const MeshletBuildResult& r, const Meshlet& ml, u32 t) {
    Tri out{};
    for (u32 k = 0; k < 3; ++k) out[k] = r.meshletVertices[ml.vertexOffset + r.meshletTriangles[ml.triangleOffset + 3 * t + k]];
    return out;
}

} // namespace

TEST_CASE("meshlet options: validation accepts legal and rejects illegal sets") {
    for (const MeshletBuildOptions& o : optionSets()) CHECK_MESSAGE(validateMeshletOptions(o).empty(), meshletOptionsName(o));

    auto bad = [](auto&& mutate) {
        MeshletBuildOptions o;
        mutate(o);
        return !validateMeshletOptions(o).empty();
    };
    CHECK(bad([](MeshletBuildOptions& o) { o.maxVertices = 2; }));
    CHECK(bad([](MeshletBuildOptions& o) { o.maxVertices = 129; }));
    CHECK(bad([](MeshletBuildOptions& o) { o.maxVertices = 300; }));
    CHECK(bad([](MeshletBuildOptions& o) { o.maxTriangles = 0; }));
    CHECK(bad([](MeshletBuildOptions& o) { o.maxTriangles = 129; }));
    CHECK(bad([](MeshletBuildOptions& o) { o.maxTriangles = 600; }));
    CHECK(bad([](MeshletBuildOptions& o) { o.coneWeight = -1.0f; }));
    CHECK(bad([](MeshletBuildOptions& o) { o.coneWeight = 2.0f; }));
    CHECK(bad([](MeshletBuildOptions& o) {
        o.algorithm = MeshletAlgorithm::Spatial;
        o.minTriangles = o.maxTriangles + 1;
    }));
    // The boundaries themselves are legal.
    CHECK_FALSE(bad([](MeshletBuildOptions& o) { o.maxVertices = 3; }));
    CHECK_FALSE(bad([](MeshletBuildOptions& o) { o.maxVertices = 128; o.maxTriangles = 128; }));
    CHECK_FALSE(bad([](MeshletBuildOptions& o) { o.maxTriangles = 1; }));
}

TEST_CASE("meshlet builder: limits, reconstruction, bounds") {
    for (const TestMesh& mesh : testMeshes()) {
        for (const MeshletBuildOptions& o : optionSets()) {
            const std::string label = mesh.name + " / " + meshletOptionsName(o);
            CAPTURE(label);
            const MeshletBuildResult r = cook(mesh, o);
            REQUIRE(r.meshlets.size() == r.bounds.size());
            REQUIRE(!r.meshlets.empty());

            // (c) counts, offsets, local indices.
            u64 triTotal = 0;
            for (const Meshlet& ml : r.meshlets) {
                CHECK(ml.vertexCount >= 1);
                CHECK(ml.vertexCount <= o.maxVertices);
                CHECK(ml.triangleCount >= 1);
                CHECK(ml.triangleCount <= o.maxTriangles);
                REQUIRE(ml.vertexOffset + ml.vertexCount <= r.meshletVertices.size());
                REQUIRE(ml.triangleOffset + 3 * ml.triangleCount <= r.meshletTriangles.size());
                for (u32 i = 0; i < 3 * ml.triangleCount; ++i) CHECK(r.meshletTriangles[ml.triangleOffset + i] < ml.vertexCount);
                for (u32 i = 0; i < ml.vertexCount; ++i) CHECK(r.meshletVertices[ml.vertexOffset + i] < mesh.positions.size());
                triTotal += ml.triangleCount;
            }

            // (b) reconstruction: multiset of cyclic-canonical triangles.
            std::map<Tri, i64> counts;
            u64 originalTris = mesh.indices.size() / 3, originalDegenerateIdx = 0;
            for (size_t t = 0; t < mesh.indices.size(); t += 3) {
                const Tri tri{mesh.indices[t], mesh.indices[t + 1], mesh.indices[t + 2]};
                ++counts[canonical(tri)];
                if (isDegenerateIndices(tri)) ++originalDegenerateIdx;
            }
            for (const Meshlet& ml : r.meshlets) {
                for (u32 t = 0; t < ml.triangleCount; ++t) {
                    // The meshlet triangle must be a cyclic rotation of an original one (orientation kept).
                    --counts[canonical(meshletTri(r, ml, t))];
                }
            }
            // meshoptimizer v1.3 keeps every triangle, including degenerate
            // ones (repeated indices, collinear / coincident points).
            CHECK(triTotal == originalTris);
            i64 mismatches = 0;
            for (const auto& [tri, c] : counts) mismatches += c != 0;
            CHECK(mismatches == 0);
            (void)originalDegenerateIdx;

            // (d) bounds: every vertex of a meshlet inside its sphere.
            for (size_t i = 0; i < r.meshlets.size(); ++i) {
                const Meshlet& ml = r.meshlets[i];
                const MeshletBounds& b = r.bounds[i];
                const glm::dvec3 c(b.center[0], b.center[1], b.center[2]);
                for (u32 k = 0; k < ml.vertexCount; ++k) {
                    const glm::dvec3 p = P(mesh, r.meshletVertices[ml.vertexOffset + k]);
                    CHECK(glm::length(p - c) <= double(b.radius) * (1.0 + 1e-5) + 1e-6);
                }
            }

            // (f) stats.
            const MeshletStats st = MeshletBuilder::stats(r, o);
            CHECK(st.meshlets == r.meshlets.size());
            CHECK(st.triangles == triTotal);
            CHECK(st.vertexRefs == r.meshletVertices.size());
            CHECK(st.bytes == r.meshlets.size() * sizeof(Meshlet) + r.bounds.size() * sizeof(MeshletBounds) +
                                  r.meshletVertices.size() * sizeof(u32) + r.meshletTriangles.size());
            CHECK(st.fillTriangles > 0.0);
            CHECK(st.fillTriangles <= 1.0);
            CHECK(st.coneUsable >= 0.0);
            CHECK(st.coneUsable <= 1.0);
            CHECK(st.avgTriangles <= double(o.maxTriangles));
            CHECK(st.avgVertices <= double(o.maxVertices));
            if (st.coneUsable > 0.0) {
                CHECK(st.coneCutoffMean < 1.0);
                CHECK(st.coneCutoffMean >= -1.0);
            }
        }
    }
}

namespace {

// Matrices as column-major float[16] from a double similarity.
struct Xf {
    float m[16];
    double d[3][3]; // linear part, row-major
    double t[3];
    double det;
};

Xf makeXf(const glm::dmat3& lin, glm::dvec3 t) {
    Xf x{};
    for (int c = 0; c < 3; ++c)
        for (int r = 0; r < 3; ++r) {
            x.m[c * 4 + r] = float(lin[c][r]);
            x.d[r][c] = lin[c][r];
        }
    x.m[12] = float(t.x);
    x.m[13] = float(t.y);
    x.m[14] = float(t.z);
    x.m[15] = 1.0f;
    x.t[0] = t.x;
    x.t[1] = t.y;
    x.t[2] = t.z;
    x.det = glm::determinant(lin);
    return x;
}

glm::dvec3 apply(const Xf& x, glm::dvec3 p) {
    return glm::dvec3(x.d[0][0] * p.x + x.d[0][1] * p.y + x.d[0][2] * p.z + x.t[0], x.d[1][0] * p.x + x.d[1][1] * p.y + x.d[1][2] * p.z + x.t[1],
                      x.d[2][0] * p.x + x.d[2][1] * p.y + x.d[2][2] * p.z + x.t[2]);
}

glm::dmat3 rotation(double ax, double ay, double az) {
    const double cx = std::cos(ax), sx = std::sin(ax), cy = std::cos(ay), sy = std::sin(ay), cz = std::cos(az), sz = std::sin(az);
    const glm::dmat3 rx(1, 0, 0, 0, cx, sx, 0, -sx, cx);
    const glm::dmat3 ry(cy, 0, -sy, 0, 1, 0, sy, 0, cy);
    const glm::dmat3 rz(cz, sz, 0, -sz, cz, 0, 0, 0, 1);
    return rz * ry * rx;
}

// True when every triangle of the meshlet, transformed by `x`, would be culled
// by the renderer's rule for this instance: back faces for det > 0 (CCW front
// faces culled-away = normal facing away), front-face culling for a mirrored
// instance (its transformed winding is reversed, so the geometrically
// back-facing triangles have cross(q1-q0, q2-q0) facing the camera).  Both
// cases: sign * dot(cross, q0 - cam) >= -tol with sign = sign(det).
bool allTrianglesCulled(const TestMesh& mesh, const MeshletBuildResult& r, const Meshlet& ml, const Xf& x, glm::dvec3 cam) {
    const double sign = x.det < 0.0 ? -1.0 : 1.0;
    for (u32 t = 0; t < ml.triangleCount; ++t) {
        const Tri tri = meshletTri(r, ml, t);
        const glm::dvec3 q0 = apply(x, P(mesh, tri[0])), q1 = apply(x, P(mesh, tri[1])), q2 = apply(x, P(mesh, tri[2]));
        const glm::dvec3 n = glm::cross(q1 - q0, q2 - q0);
        const double d = sign * glm::dot(n, q0 - cam);
        const double tol = 1e-6 * glm::length(n) * glm::length(q0 - cam) + 1e-12;
        if (d < -tol) return false; // a visible triangle
    }
    return true;
}

} // namespace

TEST_CASE("meshlet builder: normal cone is conservative (identity, mirrored, non-uniform, shear)") {
    Rng rng{0xC0FFEEull};
    u64 culledIdentity = 0, culledMirrored = 0, culledNonUniform = 0, culledShear = 0, testedCones = 0;
    const std::vector<TestMesh> meshes = testMeshes();
    const std::vector<MeshletBuildOptions> opts = {MeshletBuildOptions{}, optionSets()[4], optionSets()[5]};
    for (const TestMesh& mesh : meshes) {
        for (const MeshletBuildOptions& o : opts) {
            const MeshletBuildResult r = cook(mesh, o);
            const float identity[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
            const Xf mirror = makeXf(rotation(0.4, -0.9, 1.3) * glm::dmat3(-2, 0, 0, 0, 2, 0, 0, 0, 2), glm::dvec3(3, -1, 2));
            const Xf nonUniform = makeXf(rotation(0.2, 0.3, 0.4) * glm::dmat3(1, 0, 0, 0, 3, 0, 0, 0, 0.5), glm::dvec3(0, 1, 0));
            // A sheared, mirrored child (rotation under a non-uniform parent, then a mirror).
            const Xf shear = makeXf(glm::dmat3(-3, 0, 0, 0, 1, 0, 0, 0, 0.7) * rotation(0.0, 0.0, 0.785), glm::dvec3(1, 2, -1));
            const Xf uniformRot = makeXf(rotation(1.1, 0.2, -0.7) * glm::dmat3(1.7), glm::dvec3(-2, 0.5, 1));
            const Xf idXf = makeXf(glm::dmat3(1.0), glm::dvec3(0));
            REQUIRE(mirror.det < 0.0);
            CAPTURE(mesh.name);
            CAPTURE(meshletOptionsName(o));
            for (size_t i = 0; i < r.meshlets.size(); ++i) {
                const Meshlet& ml = r.meshlets[i];
                const MeshletBounds& b = r.bounds[i];
                const bool zeroAxis = b.coneAxis[0] == 0.0f && b.coneAxis[1] == 0.0f && b.coneAxis[2] == 0.0f;
                if (!zeroAxis && b.coneCutoff + MESHLET_CONE_EPS < 1.0f) ++testedCones;
                const glm::dvec3 center(b.center[0], b.center[1], b.center[2]);
                const double radius = std::max(double(b.radius), 1e-3);
                for (int s = 0; s < 2000 / 4; ++s) { // 500 cameras per meshlet per matrix kind (x4 below)
                    // Random direction, distance 0.1 .. 20 radii.
                    glm::dvec3 dir(rng.range(-1, 1), rng.range(-1, 1), rng.range(-1, 1));
                    if (glm::length(dir) < 1e-3) dir = glm::dvec3(0, 1, 0);
                    dir = glm::normalize(dir);
                    const double dist = radius * (0.1 + 20.0 * double(rng.next()) * double(rng.next()));
                    const glm::dvec3 cam = center + dir * dist;
                    const float camF[3] = {float(cam.x), float(cam.y), float(cam.z)};

                    if (meshletConeCulled(identity, b, camF)) {
                        ++culledIdentity;
                        if (!allTrianglesCulled(mesh, r, ml, idXf, cam)) FAIL("identity: cone culled a meshlet with a visible triangle");
                    }
                    {
                        const glm::dvec3 w = apply(uniformRot, center) + dir * dist;
                        const float wf[3] = {float(w.x), float(w.y), float(w.z)};
                        if (meshletConeCulled(uniformRot.m, b, wf) && !allTrianglesCulled(mesh, r, ml, uniformRot, w))
                            FAIL("uniform scale+rotation: visible triangle culled");
                    }
                    {
                        const glm::dvec3 w = apply(mirror, center) + dir * dist;
                        const float wf[3] = {float(w.x), float(w.y), float(w.z)};
                        if (meshletConeCulled(mirror.m, b, wf)) {
                            ++culledMirrored;
                            if (!allTrianglesCulled(mesh, r, ml, mirror, w)) FAIL("mirrored: cone culled a meshlet with a visible triangle");
                        }
                    }
                    {
                        // F6: the test runs in mesh space (camera through M^-1),
                        // exact for any affine map: non-uniform scale too.
                        const glm::dvec3 w = apply(nonUniform, center) + dir * dist;
                        const float wf[3] = {float(w.x), float(w.y), float(w.z)};
                        if (meshletConeCulled(nonUniform.m, b, wf)) {
                            ++culledNonUniform;
                            if (!allTrianglesCulled(mesh, r, ml, nonUniform, w))
                                FAIL("non-uniform scale: cone culled a meshlet with a visible triangle");
                        }
                    }
                    {
                        const glm::dvec3 w = apply(shear, center) + dir * dist;
                        const float wf[3] = {float(w.x), float(w.y), float(w.z)};
                        if (meshletConeCulled(shear.m, b, wf)) {
                            ++culledShear;
                            if (!allTrianglesCulled(mesh, r, ml, shear, w)) FAIL("shear + mirror: cone culled a meshlet with a visible triangle");
                        }
                    }
                    if (zeroAxis) {
                        CHECK_FALSE(meshletConeCulled(identity, b, camF));
                        CHECK_FALSE(meshletConeCulled(mirror.m, b, camF));
                    }
                }
            }
        }
    }
    // Positive controls: the property is not vacuous.
    CHECK(testedCones > 0);
    CHECK(culledIdentity > 1000);
    CHECK(culledMirrored > 1000);
    CHECK(culledNonUniform > 1000);
    CHECK(culledShear > 1000);
}

TEST_CASE("meshlet cone: singular matrices never reject") {
    GPUMeshletBounds b{};
    b.coneAxis[0] = 1.0f;
    b.coneCutoff  = 0.1f;
    const float flat[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}; // z collapsed: det 0
    const float camBehind[3] = {-10, 0, 0};
    CHECK_FALSE(meshletConeCulled(flat, b, camBehind));
}

TEST_CASE("meshlet cone: degenerate bounds are never rejected") {
    const float identity[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    const float cam[3] = {5, 5, 5};
    GPUMeshletBounds zero{};
    CHECK_FALSE(meshletConeCulled(identity, zero, cam));
    GPUMeshletBounds noCone{};
    noCone.coneAxis[2] = 1.0f;
    noCone.coneCutoff = 1.0f;
    noCone.coneApex[2] = -10.0f;
    CHECK_FALSE(meshletConeCulled(identity, noCone, cam));
    // Camera at the apex.
    GPUMeshletBounds atApex{};
    atApex.coneAxis[0] = 1.0f;
    atApex.coneCutoff = 0.1f;
    const float camApex[3] = {0, 0, 0};
    CHECK_FALSE(meshletConeCulled(identity, atApex, camApex));
    // A usable cone rejects a camera behind the faces (axis +X, camera at -X
    // side of the apex, all normals within ~84 degrees of +X face away from it).
    const float camBehind[3] = {-10, 0, 0};
    CHECK(meshletConeCulled(identity, atApex, camBehind)); // apex - cam = +X, dot(+X, axis) = 1 >= cutoff
}
