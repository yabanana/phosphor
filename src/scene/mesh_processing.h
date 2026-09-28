#pragma once

#include "core/types.h"

#include <glm/glm.hpp>

#include <vector>

namespace phosphor {

// Vertex streams of one triangle-list primitive; indices are local to it.
struct GeometryStreams {
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec4> tangents; // xyz + handedness w
    std::vector<glm::vec2> uvs;
    std::vector<u32>       indices;
};

/// Fill the attributes a glTF primitive may omit, as the specification asks:
///  - missing normals: flat normals (per triangle, counter-clockwise);
///  - missing tangents: MikkTSpace from positions, normals and UVs, with w
///    negated for glTF's top-left UV origin, so that bitangent =
///    cross(N, T) * w points towards decreasing V (engine convention, see
///    tests/test_procedural.cpp).  Without UVs any unit vector
///    perpendicular to the normal is used (no normal map can apply).
/// The streams must hold one entry per vertex for every attribute marked as
/// present (absent ones may be empty).  The primitive is expanded per
/// triangle, completed and re-welded, so shared vertices stay shared unless
/// a new flat normal or tangent splits them.  No-op when nothing is missing.
void completeGeometry(GeometryStreams& geometry, bool hasNormals, bool hasTangents, bool hasUVs);

} // namespace phosphor
