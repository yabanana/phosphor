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

// Every generator follows the glTF conventions the whole engine uses:
// counter-clockwise triangles about the outward normal (front face = CCW),
// UV origin at the image's top-left, bitangent = cross(N, T) * w pointing
// towards decreasing V.
namespace ProceduralMeshes {

    // Torus lying in the XZ plane, major circle along XZ, tube revolves around it.
    MeshData generateTorus(float majorRadius, float minorRadius,
                           u32 majorSegments, u32 minorSegments);

    // UV sphere centered at origin.
    MeshData generateSphere(float radius, u32 slices, u32 stacks);

    // F15.3 (ticket 11): the UV sphere of generateSphere with a "combed" tangent field instead of the longitude
    // one: the tangent of every vertex is the world direction `comb` projected onto its tangent plane (unit length,
    // exactly perpendicular to the normal), `handedness` is stored in w (the engine's UV sphere uses -1: B = cross(N, T) * w).
    // The field has its poles where the normal is +-comb; there (no projection left) the tangent is any unit
    // perpendicular, a vertex-attribute choice only.  UVs are the sphere's: the tangent does NOT follow dP/du,
    // so this mesh is a KHR_materials_anisotropy carrier (a uniform lobe direction facing the camera), not a UV test mesh.
    MeshData generateCombedSphere(float radius, u32 slices, u32 stacks, const glm::vec3& comb, float handedness);

    // Axis-aligned cube centered at origin, side length = 2 * halfExtent.
    // 24 vertices (4 per face) with correct per-face normals and tangents.
    MeshData generateCube(float halfExtent);

    // Subdivided quad on the XZ plane (Y = 0), centered at origin, normal up (+Y).
    MeshData generatePlane(float width, float depth, u32 subdivX, u32 subdivZ);

    // Icosahedron inscribed in a sphere of `radius`.  subdivisions = 0 gives
    // 20 triangles; each level splits every triangle in 4 (20 * 4^n) and
    // projects the new vertices onto the sphere.  `flatNormals`: one normal per
    // face, vertices not shared (faceted look); otherwise vertices are shared
    // where position and UV agree and the normals are radial.  UVs are
    // equirectangular like generateSphere (seam vertices duplicated, a pole
    // vertex takes the mean U of its triangle).
    MeshData generateIcosahedron(float radius, u32 subdivisions = 0, bool flatNormals = true);

    // Octahedron with vertices at +-radius on the axes (8 triangles); same
    // normal/UV options as generateIcosahedron.
    MeshData generateOctahedron(float radius, bool flatNormals = true);

} // namespace ProceduralMeshes

} // namespace phosphor
