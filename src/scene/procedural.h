#pragma once

#include "core/types.h"
#include <glm/glm.hpp>
#include <vector>

namespace phosphor {

struct MeshData {
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec4> tangents; // xyz = tangent direction, w = handedness (+1 or -1)
    std::vector<glm::vec2> uvs;
    std::vector<u32>       indices;
};

namespace ProceduralMeshes {

    // Torus lying in the XZ plane, major circle along XZ, tube revolves around it.
    MeshData generateTorus(float majorRadius, float minorRadius,
                           u32 majorSegments, u32 minorSegments);

    // UV sphere centered at origin.
    MeshData generateSphere(float radius, u32 slices, u32 stacks);

    // Axis-aligned cube centered at origin, side length = 2 * halfExtent.
    // 24 vertices (4 per face) with correct per-face normals and tangents.
    MeshData generateCube(float halfExtent);

    // Subdivided quad on the XZ plane (Y = 0), centered at origin, normal up (+Y).
    MeshData generatePlane(float width, float depth, u32 subdivX, u32 subdivZ);

} // namespace ProceduralMeshes

} // namespace phosphor
