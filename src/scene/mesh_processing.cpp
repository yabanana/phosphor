#include "scene/mesh_processing.h"

#include <meshoptimizer.h>
#include <mikktspace.h>

#include <cmath>

namespace phosphor {

namespace {

// One expanded (per triangle corner) vertex; tightly packed floats, so
// meshopt_generateVertexRemap can weld identical ones bitwise.
struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
    glm::vec4 tangent;
};
static_assert(sizeof(Vertex) == 48, "Vertex must be tightly packed for welding");

glm::vec3 anyPerpendicular(const glm::vec3& n) {
    const glm::vec3 axis = std::abs(n.x) < 0.9f ? glm::vec3(1, 0, 0) : glm::vec3(0, 1, 0);
    return glm::normalize(glm::cross(axis, n));
}

// MikkTSpace callbacks over the expanded vertex array (3 corners per face).
std::vector<Vertex>& vertices(const SMikkTSpaceContext* ctx) {
    return *static_cast<std::vector<Vertex>*>(ctx->m_pUserData);
}
int getNumFaces(const SMikkTSpaceContext* ctx) { return static_cast<int>(vertices(ctx).size() / 3); }
int getNumVerticesOfFace(const SMikkTSpaceContext*, const int) { return 3; }
void getPosition(const SMikkTSpaceContext* ctx, float out[], const int face, const int vert) {
    const glm::vec3& p = vertices(ctx)[static_cast<size_t>(face * 3 + vert)].position;
    out[0] = p.x; out[1] = p.y; out[2] = p.z;
}
void getNormal(const SMikkTSpaceContext* ctx, float out[], const int face, const int vert) {
    const glm::vec3& n = vertices(ctx)[static_cast<size_t>(face * 3 + vert)].normal;
    out[0] = n.x; out[1] = n.y; out[2] = n.z;
}
void getTexCoord(const SMikkTSpaceContext* ctx, float out[], const int face, const int vert) {
    const glm::vec2& t = vertices(ctx)[static_cast<size_t>(face * 3 + vert)].uv;
    out[0] = t.x; out[1] = t.y;
}
void setTSpaceBasic(const SMikkTSpaceContext* ctx, const float tangent[], const float sign, const int face,
                    const int vert) {
    // glTF UVs have their origin at the top-left: flip the handedness so the
    // bitangent points towards decreasing V, like three.js does for glTF.
    vertices(ctx)[static_cast<size_t>(face * 3 + vert)].tangent =
        glm::vec4(tangent[0], tangent[1], tangent[2], -sign);
}

} // namespace

void completeGeometry(GeometryStreams& g, bool hasNormals, bool hasTangents, bool hasUVs) {
    if (hasNormals && hasTangents) return;
    const size_t corners = g.indices.size() - g.indices.size() % 3;
    if (corners == 0) return;

    // Expand per triangle corner.
    std::vector<Vertex> expanded(corners);
    for (size_t c = 0; c < corners; ++c) {
        const u32 i = g.indices[c];
        Vertex& v = expanded[c];
        v.position = g.positions[i];
        v.normal   = hasNormals ? g.normals[i] : glm::vec3(0.0f);
        v.uv       = hasUVs ? g.uvs[i] : glm::vec2(0.0f);
        v.tangent  = hasTangents ? g.tangents[i] : glm::vec4(0.0f);
    }

    if (!hasNormals) {
        for (size_t c = 0; c < corners; c += 3) {
            const glm::vec3 face = glm::cross(expanded[c + 1].position - expanded[c].position,
                                              expanded[c + 2].position - expanded[c].position);
            const float len = glm::length(face);
            const glm::vec3 n = len > 0.0f ? face / len : glm::vec3(0.0f, 0.0f, 1.0f); // degenerate
            for (size_t k = 0; k < 3; ++k) expanded[c + k].normal = n;
        }
    }

    if (!hasTangents) {
        bool generated = false;
        if (hasUVs) {
            SMikkTSpaceInterface callbacks{};
            callbacks.m_getNumFaces          = getNumFaces;
            callbacks.m_getNumVerticesOfFace = getNumVerticesOfFace;
            callbacks.m_getPosition          = getPosition;
            callbacks.m_getNormal            = getNormal;
            callbacks.m_getTexCoord          = getTexCoord;
            callbacks.m_setTSpaceBasic       = setTSpaceBasic;
            SMikkTSpaceContext ctx{&callbacks, &expanded};
            generated = genTangSpaceDefault(&ctx) != 0;
        }
        if (!generated) {
            for (Vertex& v : expanded) v.tangent = glm::vec4(anyPerpendicular(v.normal), 1.0f);
        }
    }

    // The weld below compares bits: quantize generated tangents (MikkTSpace
    // may differ in the last bits for corners of one vertex) and turn -0.0
    // into +0.0 in generated data (cross products produce signed zeros).
    for (Vertex& v : expanded) {
        for (int k = 0; k < 3; ++k) {
            if (!hasTangents) v.tangent[k] = std::round(v.tangent[k] * 65536.0f) / 65536.0f + 0.0f;
            if (!hasNormals) v.normal[k] += 0.0f;
        }
    }

    // Weld identical corners back into shared vertices.
    std::vector<u32> remap(corners);
    const size_t unique = meshopt_generateVertexRemap(remap.data(), nullptr, corners, expanded.data(), corners,
                                                      sizeof(Vertex));
    std::vector<Vertex> welded(unique);
    meshopt_remapVertexBuffer(welded.data(), expanded.data(), corners, sizeof(Vertex), remap.data());

    g.positions.resize(unique);
    g.normals.resize(unique);
    g.uvs.resize(unique);
    g.tangents.resize(unique);
    for (size_t v = 0; v < unique; ++v) {
        g.positions[v] = welded[v].position;
        g.normals[v]   = welded[v].normal;
        g.uvs[v]       = welded[v].uv;
        g.tangents[v]  = welded[v].tangent;
    }
    g.indices.assign(remap.begin(), remap.end());
}

} // namespace phosphor
