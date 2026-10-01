#include "scene/procedural.h"

#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <tuple>

namespace phosphor {
namespace ProceduralMeshes {

// ---------------------------------------------------------------------------
// Torus: parametric surface (theta = major angle, phi = minor/tube angle)
//   P(theta, phi) = ( (R + r*cos(phi)) * cos(theta),
//                      r * sin(phi),
//                     (R + r*cos(phi)) * sin(theta) )
//
// Normals and tangents are computed analytically from the partial derivatives.
// ---------------------------------------------------------------------------
MeshData generateTorus(float majorRadius, float minorRadius,
                       u32 majorSegments, u32 minorSegments) {
    MeshData mesh;
    u32 vertCount = (majorSegments + 1) * (minorSegments + 1);
    mesh.positions.reserve(vertCount);
    mesh.normals.reserve(vertCount);
    mesh.tangents.reserve(vertCount);
    mesh.uvs.reserve(vertCount);
    mesh.indices.reserve(majorSegments * minorSegments * 6);

    float R = majorRadius;
    float r = minorRadius;

    for (u32 i = 0; i <= majorSegments; ++i) {
        float theta = static_cast<float>(i) / static_cast<float>(majorSegments)
                    * glm::two_pi<float>();
        float cosTheta = std::cos(theta);
        float sinTheta = std::sin(theta);

        for (u32 j = 0; j <= minorSegments; ++j) {
            float phi = static_cast<float>(j) / static_cast<float>(minorSegments)
                      * glm::two_pi<float>();
            float cosPhi = std::cos(phi);
            float sinPhi = std::sin(phi);

            // Position
            float x = (R + r * cosPhi) * cosTheta;
            float y = r * sinPhi;
            float z = (R + r * cosPhi) * sinTheta;
            mesh.positions.emplace_back(x, y, z);

            // Normal: direction from the center of the tube circle to the surface point.
            // Center of tube circle at angle theta is (R*cosTheta, 0, R*sinTheta).
            glm::vec3 center(R * cosTheta, 0.0f, R * sinTheta);
            glm::vec3 normal = glm::normalize(glm::vec3(x, y, z) - center);
            mesh.normals.push_back(normal);

            // Tangent: partial derivative with respect to theta (along the major circle).
            // dP/dtheta = (-(R + r*cosPhi)*sinTheta, 0, (R + r*cosPhi)*cosTheta)
            glm::vec3 tangent = glm::normalize(
                glm::vec3(-(R + r * cosPhi) * sinTheta,
                           0.0f,
                           (R + r * cosPhi) * cosTheta));
            mesh.tangents.emplace_back(tangent, 1.0f);

            // UV
            float u = static_cast<float>(i) / static_cast<float>(majorSegments);
            float v = static_cast<float>(j) / static_cast<float>(minorSegments);
            mesh.uvs.emplace_back(u, v);
        }
    }

    // Indices: two triangles per quad
    for (u32 i = 0; i < majorSegments; ++i) {
        for (u32 j = 0; j < minorSegments; ++j) {
            u32 a = i * (minorSegments + 1) + j;
            u32 b = a + minorSegments + 1;
            u32 c = a + 1;
            u32 d = b + 1;

            // Counter-clockwise about the outward normal (engine convention).
            mesh.indices.push_back(a);
            mesh.indices.push_back(c);
            mesh.indices.push_back(b);

            mesh.indices.push_back(c);
            mesh.indices.push_back(d);
            mesh.indices.push_back(b);
        }
    }

    return mesh;
}

// ---------------------------------------------------------------------------
// UV Sphere: standard latitude/longitude parameterization.
// Tangents point along the longitude direction (partial derivative w.r.t. phi).
// ---------------------------------------------------------------------------
MeshData generateSphere(float radius, u32 slices, u32 stacks) {
    MeshData mesh;
    u32 vertCount = (slices + 1) * (stacks + 1);
    mesh.positions.reserve(vertCount);
    mesh.normals.reserve(vertCount);
    mesh.tangents.reserve(vertCount);
    mesh.uvs.reserve(vertCount);
    mesh.indices.reserve(slices * stacks * 6);

    for (u32 stack = 0; stack <= stacks; ++stack) {
        // theta goes from 0 (top pole) to pi (bottom pole)
        float theta = static_cast<float>(stack) / static_cast<float>(stacks)
                    * glm::pi<float>();
        float sinTheta = std::sin(theta);
        float cosTheta = std::cos(theta);

        for (u32 slice = 0; slice <= slices; ++slice) {
            // phi goes from 0 to 2*pi around the equator
            float phi = static_cast<float>(slice) / static_cast<float>(slices)
                      * glm::two_pi<float>();
            float sinPhi = std::sin(phi);
            float cosPhi = std::cos(phi);

            // Normal (unit sphere direction)
            glm::vec3 normal(sinTheta * cosPhi,
                             cosTheta,
                             sinTheta * sinPhi);

            mesh.positions.push_back(normal * radius);
            mesh.normals.push_back(normal);

            // Tangent: dP/dphi normalized (along longitude).
            // dP/dphi = (-sinTheta*sinPhi, 0, sinTheta*cosPhi) -> normalize
            // V grows towards the south pole, so the bitangent (towards -V,
            // glTF convention) is north: cross(N, T) points south, hence w = -1.
            glm::vec3 tangent(-sinPhi, 0.0f, cosPhi);
            mesh.tangents.emplace_back(tangent, -1.0f);

            float u = static_cast<float>(slice) / static_cast<float>(slices);
            float v = static_cast<float>(stack) / static_cast<float>(stacks);
            mesh.uvs.emplace_back(u, v);
        }
    }

    // Indices
    for (u32 stack = 0; stack < stacks; ++stack) {
        for (u32 slice = 0; slice < slices; ++slice) {
            u32 a = stack * (slices + 1) + slice;
            u32 b = a + slices + 1;
            u32 c = a + 1;
            u32 d = b + 1;

            // Counter-clockwise about the outward normal (engine convention).
            mesh.indices.push_back(a);
            mesh.indices.push_back(c);
            mesh.indices.push_back(b);

            mesh.indices.push_back(c);
            mesh.indices.push_back(d);
            mesh.indices.push_back(b);
        }
    }

    return mesh;
}

// ---------------------------------------------------------------------------
// Cube: 6 faces, 4 vertices each (24 total). Per-face normals and tangents.
// ---------------------------------------------------------------------------
MeshData generateCube(float halfExtent) {
    MeshData mesh;
    mesh.positions.reserve(24);
    mesh.normals.reserve(24);
    mesh.tangents.reserve(24);
    mesh.uvs.reserve(24);
    mesh.indices.reserve(36);

    float h = halfExtent;

    // Face data: normal, tangent, and 4 corner positions.
    // UV layout follows glTF (origin at the image's top-left):
    // (0,1) bottom-left, (1,1) bottom-right, (1,0) top-right, (0,0) top-left
    struct Face {
        glm::vec3 normal;
        glm::vec3 tangent;
        glm::vec3 corners[4]; // BL, BR, TR, TL
    };

    Face faces[6] = {
        // +X face
        { { 1, 0, 0}, { 0, 0,-1}, {{ h,-h, h}, { h,-h,-h}, { h, h,-h}, { h, h, h}} },
        // -X face
        { {-1, 0, 0}, { 0, 0, 1}, {{-h,-h,-h}, {-h,-h, h}, {-h, h, h}, {-h, h,-h}} },
        // +Y face (top)
        { { 0, 1, 0}, { 1, 0, 0}, {{-h, h, h}, { h, h, h}, { h, h,-h}, {-h, h,-h}} },
        // -Y face (bottom)
        { { 0,-1, 0}, { 1, 0, 0}, {{-h,-h,-h}, { h,-h,-h}, { h,-h, h}, {-h,-h, h}} },
        // +Z face
        { { 0, 0, 1}, { 1, 0, 0}, {{-h,-h, h}, { h,-h, h}, { h, h, h}, {-h, h, h}} },
        // -Z face
        { { 0, 0,-1}, {-1, 0, 0}, {{ h,-h,-h}, {-h,-h,-h}, {-h, h,-h}, { h, h,-h}} },
    };

    glm::vec2 faceUVs[4] = {
        {0.0f, 1.0f}, {1.0f, 1.0f}, {1.0f, 0.0f}, {0.0f, 0.0f}
    };

    for (u32 f = 0; f < 6; ++f) {
        u32 base = f * 4;
        for (u32 v = 0; v < 4; ++v) {
            mesh.positions.push_back(faces[f].corners[v]);
            mesh.normals.push_back(faces[f].normal);
            mesh.tangents.emplace_back(faces[f].tangent, 1.0f);
            mesh.uvs.push_back(faceUVs[v]);
        }

        // Two triangles per face, counter-clockwise about the face normal
        mesh.indices.push_back(base + 0);
        mesh.indices.push_back(base + 1);
        mesh.indices.push_back(base + 2);

        mesh.indices.push_back(base + 0);
        mesh.indices.push_back(base + 2);
        mesh.indices.push_back(base + 3);
    }

    return mesh;
}

// ---------------------------------------------------------------------------
// Plane: subdivided quad on XZ plane, Y=0, centered at origin.
// Normal points up (+Y), tangent along +X.
// ---------------------------------------------------------------------------
MeshData generatePlane(float width, float depth, u32 subdivX, u32 subdivZ) {
    MeshData mesh;
    u32 vertsX = subdivX + 1;
    u32 vertsZ = subdivZ + 1;
    u32 vertCount = vertsX * vertsZ;
    mesh.positions.reserve(vertCount);
    mesh.normals.reserve(vertCount);
    mesh.tangents.reserve(vertCount);
    mesh.uvs.reserve(vertCount);
    mesh.indices.reserve(subdivX * subdivZ * 6);

    glm::vec3 normal(0.0f, 1.0f, 0.0f);
    glm::vec4 tangent(1.0f, 0.0f, 0.0f, 1.0f);

    float halfW = width * 0.5f;
    float halfD = depth * 0.5f;

    for (u32 iz = 0; iz <= subdivZ; ++iz) {
        float v = static_cast<float>(iz) / static_cast<float>(subdivZ);
        float z = -halfD + v * depth;

        for (u32 ix = 0; ix <= subdivX; ++ix) {
            float u = static_cast<float>(ix) / static_cast<float>(subdivX);
            float x = -halfW + u * width;

            mesh.positions.emplace_back(x, 0.0f, z);
            mesh.normals.push_back(normal);
            mesh.tangents.push_back(tangent);
            mesh.uvs.emplace_back(u, v);
        }
    }

    // Indices: two triangles per quad
    for (u32 iz = 0; iz < subdivZ; ++iz) {
        for (u32 ix = 0; ix < subdivX; ++ix) {
            u32 a = iz * vertsX + ix;
            u32 b = a + vertsX;
            u32 c = a + 1;
            u32 d = b + 1;

            mesh.indices.push_back(a);
            mesh.indices.push_back(b);
            mesh.indices.push_back(c);

            mesh.indices.push_back(c);
            mesh.indices.push_back(b);
            mesh.indices.push_back(d);
        }
    }

    return mesh;
}

// ---------------------------------------------------------------------------
// Polyhedra inscribed in a sphere (icosahedron, octahedron).  Both build a
// triangle list (CCW seen from outside) and share the UV/tangent code below.
// ---------------------------------------------------------------------------
namespace {

using Tri = std::array<glm::vec3, 3>;

MeshData meshFromTriangles(const std::vector<Tri>& tris, float radius, bool flatNormals) {
    constexpr float kPi = glm::pi<float>();
    struct Vert {
        glm::vec3 p;
        glm::vec2 uv;
    };
    std::vector<std::array<Vert, 3>> faces;
    faces.reserve(tris.size());
    for (const Tri& t : tris) {
        std::array<Vert, 3> f;
        float u[3];
        bool pole[3];
        for (int k = 0; k < 3; ++k) {
            const glm::vec3 n = glm::normalize(t[k]);
            pole[k] = std::abs(n.y) > 0.999999f;
            const float uk = std::atan2(n.z, n.x) / (2.0f * kPi);
            u[k] = uk - std::floor(uk);
            f[k].p = n * radius;
            f[k].uv.y = std::acos(std::clamp(n.y, -1.0f, 1.0f)) / kPi;
        }
        // Fix the u seam among the non-pole vertices of this triangle.
        float lo = 2.0f, hi = -1.0f;
        for (int k = 0; k < 3; ++k) {
            if (pole[k]) continue;
            lo = std::min(lo, u[k]);
            hi = std::max(hi, u[k]);
        }
        if (hi - lo > 0.5f) {
            for (int k = 0; k < 3; ++k) {
                if (!pole[k] && u[k] < 0.5f) u[k] += 1.0f;
            }
        }
        // A pole has no longitude: take the mean of the triangle's others.
        float sum = 0.0f;
        int cnt = 0;
        for (int k = 0; k < 3; ++k) {
            if (!pole[k]) { sum += u[k]; ++cnt; }
        }
        for (int k = 0; k < 3; ++k) {
            if (pole[k]) u[k] = cnt ? sum / static_cast<float>(cnt) : 0.0f;
            f[k].uv.x = u[k];
        }
        faces.push_back(f);
    }

    MeshData mesh;
    std::map<std::tuple<float, float, float, float, float>, u32> weld;
    for (const auto& f : faces) {
        const glm::vec3 e1 = f[1].p - f[0].p, e2 = f[2].p - f[0].p;
        const glm::vec3 faceN = glm::normalize(glm::cross(e1, e2));
        // dP/du and dP/dv of the triangle (flat mode takes the tangent from it).
        const glm::vec2 d1 = f[1].uv - f[0].uv, d2 = f[2].uv - f[0].uv;
        const float det = d1.x * d2.y - d2.x * d1.y;
        glm::vec3 faceT(1, 0, 0);
        float faceW = -1.0f;
        if (std::abs(det) > 1e-9f) {
            const glm::vec3 dPdu = (e1 * d2.y - e2 * d1.y) / det;
            const glm::vec3 dPdv = (e2 * d1.x - e1 * d2.x) / det;
            faceT = glm::normalize(dPdu);
            // B = cross(N, T) * w must point towards decreasing V.
            faceW = glm::dot(glm::cross(faceN, faceT), dPdv) < 0.0f ? 1.0f : -1.0f;
        }
        for (int k = 0; k < 3; ++k) {
            glm::vec3 n = faceN;
            glm::vec4 tan(faceT, faceW);
            if (!flatNormals) {
                n = glm::normalize(f[k].p);
                // East direction at this longitude, made orthogonal to N.
                const float phi = f[k].uv.x * 2.0f * kPi;
                glm::vec3 east(-std::sin(phi), 0.0f, std::cos(phi));
                east = glm::normalize(east - n * glm::dot(east, n));
                tan = glm::vec4(east, -1.0f);
                const auto key = std::make_tuple(f[k].p.x, f[k].p.y, f[k].p.z, f[k].uv.x, f[k].uv.y);
                auto [it, inserted] = weld.try_emplace(key, static_cast<u32>(mesh.positions.size()));
                if (!inserted) {
                    mesh.indices.push_back(it->second);
                    continue;
                }
            }
            mesh.indices.push_back(static_cast<u32>(mesh.positions.size()));
            mesh.positions.push_back(f[k].p);
            mesh.normals.push_back(n);
            mesh.tangents.push_back(tan);
            mesh.uvs.push_back(f[k].uv);
        }
    }
    return mesh;
}

// Make `t` counter-clockwise about the outward direction (centroid).
void orientOutward(Tri& t) {
    const glm::vec3 c = t[0] + t[1] + t[2];
    if (glm::dot(glm::cross(t[1] - t[0], t[2] - t[0]), c) < 0.0f) std::swap(t[1], t[2]);
}

} // namespace

MeshData generateIcosahedron(float radius, u32 subdivisions, bool flatNormals) {
    const float t = (1.0f + std::sqrt(5.0f)) * 0.5f;
    const glm::vec3 v[12] = {
        {-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0},
        {0, -1, t}, {0, 1, t}, {0, -1, -t}, {0, 1, -t},
        {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1},
    };
    static constexpr int f[20][3] = {
        {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11},
        {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
        {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9},
        {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1},
    };
    // Rotate so that vertex 0 is the +Y pole: the equirectangular UVs are
    // only well behaved when the poles are vertices (no triangle spans one).
    const glm::vec3 pole0 = glm::normalize(v[0]);
    const glm::vec3 axis = glm::cross(pole0, glm::vec3(0, 1, 0));
    const glm::mat3 toPole = glm::mat3(glm::rotate(glm::mat4(1.0f),
        std::acos(std::clamp(pole0.y, -1.0f, 1.0f)), glm::normalize(axis)));
    const u32 levels = std::min(subdivisions, 8u);
    std::vector<Tri> tris;
    tris.reserve(20u << (2 * levels));
    for (const auto& face : f) {
        tris.push_back({toPole * glm::normalize(v[face[0]]), toPole * glm::normalize(v[face[1]]),
                        toPole * glm::normalize(v[face[2]])});
    }
    for (u32 s = 0; s < levels; ++s) {
        std::vector<Tri> next;
        next.reserve(tris.size() * 4);
        for (const Tri& a : tris) {
            const glm::vec3 m01 = glm::normalize(a[0] + a[1]);
            const glm::vec3 m12 = glm::normalize(a[1] + a[2]);
            const glm::vec3 m20 = glm::normalize(a[2] + a[0]);
            next.push_back({a[0], m01, m20});
            next.push_back({a[1], m12, m01});
            next.push_back({a[2], m20, m12});
            next.push_back({m01, m12, m20});
        }
        tris.swap(next);
    }
    for (Tri& tri : tris) orientOutward(tri);
    return meshFromTriangles(tris, radius, flatNormals);
}

MeshData generateOctahedron(float radius, bool flatNormals) {
    const glm::vec3 px(1, 0, 0), nx(-1, 0, 0), py(0, 1, 0), ny(0, -1, 0), pz(0, 0, 1), nz(0, 0, -1);
    std::vector<Tri> tris = {
        {py, pz, px}, {py, nx, pz}, {py, nz, nx}, {py, px, nz},
        {ny, px, pz}, {ny, pz, nx}, {ny, nx, nz}, {ny, nz, px},
    };
    for (Tri& tri : tris) orientOutward(tri);
    return meshFromTriangles(tris, radius, flatNormals);
}

} // namespace ProceduralMeshes
} // namespace phosphor
