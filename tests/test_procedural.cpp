#include "scene/procedural.h"

#include <doctest/doctest.h>
#include <glm/glm.hpp>

#include <string>
#include <utility>
#include <vector>

using namespace phosphor;

namespace {

// Every generated mesh, labelled for failure messages.
std::vector<std::pair<std::string, MeshData>> allMeshes() {
    return {
        {"torus", ProceduralMeshes::generateTorus(1.5f, 0.5f, 24, 12)},
        {"sphere", ProceduralMeshes::generateSphere(0.5f, 16, 8)},
        {"cube", ProceduralMeshes::generateCube(1.0f)},
        {"plane", ProceduralMeshes::generatePlane(4.0f, 2.0f, 3, 2)},
    };
}

} // namespace

TEST_CASE("procedural meshes: triangles are counter-clockwise about their normals") {
    for (const auto& [name, mesh] : allMeshes()) {
        CAPTURE(name);
        REQUIRE(mesh.indices.size() % 3 == 0);
        u32 wrong = 0;
        u32 degenerate = 0;
        for (size_t t = 0; t < mesh.indices.size(); t += 3) {
            const u32 i0 = mesh.indices[t], i1 = mesh.indices[t + 1], i2 = mesh.indices[t + 2];
            const glm::vec3 faceNormal = glm::cross(mesh.positions[i1] - mesh.positions[i0],
                                                    mesh.positions[i2] - mesh.positions[i0]);
            // Pole triangles of the UV sphere collapse to a point: skip them.
            if (glm::length(faceNormal) < 1e-7f) {
                ++degenerate;
                continue;
            }
            const glm::vec3 vertexNormals = mesh.normals[i0] + mesh.normals[i1] + mesh.normals[i2];
            if (glm::dot(faceNormal, vertexNormals) <= 0.0f) ++wrong;
        }
        CHECK(wrong == 0);
        CHECK(degenerate < mesh.indices.size() / 3);
    }
}

TEST_CASE("procedural meshes: tangent frame matches the UV layout") {
    // The shader builds B = cross(N, T) * w.  glTF puts the UV origin at the
    // image's top-left, so B (tangent-space "up") points towards decreasing V.
    for (const auto& [name, mesh] : allMeshes()) {
        CAPTURE(name);
        u32 wrongTangent = 0;
        u32 wrong = 0;
        for (size_t t = 0; t < mesh.indices.size(); t += 3) {
            const u32 i0 = mesh.indices[t], i1 = mesh.indices[t + 1], i2 = mesh.indices[t + 2];
            const glm::vec3 e1 = mesh.positions[i1] - mesh.positions[i0];
            const glm::vec3 e2 = mesh.positions[i2] - mesh.positions[i0];
            const glm::vec2 d1 = mesh.uvs[i1] - mesh.uvs[i0];
            const glm::vec2 d2 = mesh.uvs[i2] - mesh.uvs[i0];
            const float det = d1.x * d2.y - d2.x * d1.y;
            if (std::abs(det) < 1e-9f || glm::length(glm::cross(e1, e2)) < 1e-7f) continue;
            const glm::vec3 dPdu = (e1 * d2.y - e2 * d1.y) / det;
            const glm::vec3 dPdv = (e2 * d1.x - e1 * d2.x) / det;

            const glm::vec4& tangent = mesh.tangents[i0];
            if (glm::dot(glm::vec3(tangent), dPdu) <= 0.0f) ++wrongTangent;
            const glm::vec3 bitangent = glm::cross(mesh.normals[i0], glm::vec3(tangent)) * tangent.w;
            if (glm::dot(bitangent, dPdv) >= 0.0f) ++wrong;
        }
        CHECK(wrongTangent == 0);
        CHECK(wrong == 0);
    }
}
