#include "scene/mesh_processing.h"
#include "scene/procedural.h"

#include <doctest/doctest.h>
#include <glm/glm.hpp>

#include <cmath>

using namespace phosphor;

namespace {

GeometryStreams fromMesh(const MeshData& m, bool keepNormals, bool keepTangents, bool keepUVs) {
    GeometryStreams g;
    g.positions = m.positions;
    if (keepNormals) g.normals = m.normals;
    if (keepTangents) g.tangents = m.tangents;
    if (keepUVs) g.uvs = m.uvs;
    g.indices = m.indices;
    return g;
}

// Same rules as tests/test_procedural.cpp: T along +U, B = cross(N, T) * w
// towards decreasing V (glTF UV origin top-left).  Returns wrong triangles.
u32 wrongTangentFrames(const GeometryStreams& g) {
    u32 wrong = 0;
    for (size_t t = 0; t + 2 < g.indices.size(); t += 3) {
        const u32 i0 = g.indices[t], i1 = g.indices[t + 1], i2 = g.indices[t + 2];
        const glm::vec3 e1 = g.positions[i1] - g.positions[i0];
        const glm::vec3 e2 = g.positions[i2] - g.positions[i0];
        const glm::vec2 d1 = g.uvs[i1] - g.uvs[i0];
        const glm::vec2 d2 = g.uvs[i2] - g.uvs[i0];
        const float det = d1.x * d2.y - d2.x * d1.y;
        if (std::abs(det) < 1e-9f || glm::length(glm::cross(e1, e2)) < 1e-7f) continue;
        const glm::vec3 dPdu = (e1 * d2.y - e2 * d1.y) / det;
        const glm::vec3 dPdv = (e2 * d1.x - e1 * d2.x) / det;
        for (const u32 i : {i0, i1, i2}) {
            // Poles of a UV sphere: the tangent is undefined there.
            if (std::abs(g.normals[i].y) > 0.999f) continue;
            const glm::vec3 t3(g.tangents[i]);
            const glm::vec3 b = glm::cross(g.normals[i], t3) * g.tangents[i].w;
            if (glm::dot(t3, dPdu) <= 0.0f || glm::dot(b, dPdv) >= 0.0f) {
                ++wrong;
                break;
            }
        }
    }
    return wrong;
}

void checkUnitOrthogonal(const GeometryStreams& g) {
    for (size_t v = 0; v < g.positions.size(); ++v) {
        const glm::vec3 t(g.tangents[v]);
        CHECK(glm::length(g.normals[v]) == doctest::Approx(1.0f).epsilon(1e-3));
        CHECK(glm::length(t) == doctest::Approx(1.0f).epsilon(1e-3));
        CHECK(std::abs(glm::dot(t, g.normals[v])) < 1e-3f);
        CHECK(std::abs(std::abs(g.tangents[v].w) - 1.0f) < 1e-6f);
    }
}

} // namespace

TEST_CASE("mesh processing: complete geometry is left untouched") {
    const MeshData cube = ProceduralMeshes::generateCube(1.0f);
    GeometryStreams g = fromMesh(cube, true, true, true);
    completeGeometry(g, true, true, true);
    CHECK(g.positions.size() == cube.positions.size());
    CHECK(g.indices == cube.indices);
}

TEST_CASE("mesh processing: missing normals become flat, CCW normals") {
    // A plane without normals or tangents: flat +Y, shared vertices welded back.
    const MeshData plane = ProceduralMeshes::generatePlane(2.0f, 2.0f, 2, 2);
    GeometryStreams g = fromMesh(plane, false, false, true);
    completeGeometry(g, false, false, true);
    CHECK(g.positions.size() == plane.positions.size()); // nothing split
    CHECK(g.indices.size() == plane.indices.size());
    for (const glm::vec3& n : g.normals) CHECK(n.y == doctest::Approx(1.0f));
    CHECK(wrongTangentFrames(g) == 0);
    checkUnitOrthogonal(g);

    // A cube without normals: every face keeps its own 4 vertices and gets
    // the normal of its face.
    const MeshData cube = ProceduralMeshes::generateCube(1.0f);
    GeometryStreams c = fromMesh(cube, false, false, true);
    completeGeometry(c, false, false, true);
    CHECK(c.positions.size() == 24);
    for (size_t t = 0; t < c.indices.size(); t += 3) {
        const glm::vec3 p0 = c.positions[c.indices[t]];
        const glm::vec3 face = glm::normalize(glm::cross(c.positions[c.indices[t + 1]] - p0,
                                                         c.positions[c.indices[t + 2]] - p0));
        for (int k = 0; k < 3; ++k) CHECK(glm::dot(c.normals[c.indices[t + k]], face) == doctest::Approx(1.0f));
    }
    CHECK(wrongTangentFrames(c) == 0);
}

TEST_CASE("mesh processing: MikkTSpace tangents follow the glTF/engine convention") {
    // Sphere and torus with their smooth normals but no tangents: the result
    // must satisfy the same frame rules as the analytic tangents, and agree
    // with them in direction.
    int which = 0;
    for (const MeshData& mesh : {ProceduralMeshes::generateSphere(1.0f, 24, 12),
                                 ProceduralMeshes::generateTorus(1.5f, 0.5f, 32, 16)}) {
        CAPTURE(which++); // 0 = sphere, 1 = torus
        GeometryStreams g = fromMesh(mesh, true, false, true);
        completeGeometry(g, true, false, true);
        CHECK(wrongTangentFrames(g) == 0);
        checkUnitOrthogonal(g);

        // Compare with the analytic frame at matching positions/UVs.
        u32 disagree = 0, compared = 0;
        for (size_t v = 0; v < g.positions.size(); ++v) {
            for (size_t o = 0; o < mesh.positions.size(); ++o) {
                if (glm::length(mesh.positions[o] - g.positions[v]) < 1e-6f &&
                    glm::length(mesh.uvs[o] - g.uvs[v]) < 1e-6f) {
                    ++compared;
                    if (mesh.normals[o].y > 0.99f || mesh.normals[o].y < -0.99f) break; // poles: tangent undefined
                    if (glm::dot(glm::vec3(g.tangents[v]), glm::vec3(mesh.tangents[o])) < 0.9f ||
                        g.tangents[v].w != mesh.tangents[o].w) {
                        ++disagree;
                    }
                    break;
                }
            }
        }
        CHECK(compared > 0);
        CHECK(disagree == 0);
    }
}

TEST_CASE("mesh processing: no UVs gives any orthonormal tangent") {
    const MeshData sphere = ProceduralMeshes::generateSphere(1.0f, 8, 6);
    GeometryStreams g = fromMesh(sphere, true, false, false);
    g.uvs.clear();
    completeGeometry(g, true, false, false);
    checkUnitOrthogonal(g);
}
