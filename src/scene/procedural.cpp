#include "scene/procedural.h"

#include <glm/gtc/constants.hpp>
#include <cmath>

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
            glm::vec3 tangent(-sinPhi, 0.0f, cosPhi);
            mesh.tangents.emplace_back(tangent, 1.0f);

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
    // UV layout: (0,0) bottom-left, (1,0) bottom-right, (1,1) top-right, (0,1) top-left
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
        { { 0, 1, 0}, { 1, 0, 0}, {{-h, h,-h}, { h, h,-h}, { h, h, h}, {-h, h, h}} },
        // -Y face (bottom)
        { { 0,-1, 0}, { 1, 0, 0}, {{-h,-h, h}, { h,-h, h}, { h,-h,-h}, {-h,-h,-h}} },
        // +Z face
        { { 0, 0, 1}, { 1, 0, 0}, {{-h,-h, h}, { h,-h, h}, { h, h, h}, {-h, h, h}} },
        // -Z face
        { { 0, 0,-1}, {-1, 0, 0}, {{ h,-h,-h}, {-h,-h,-h}, {-h, h,-h}, { h, h,-h}} },
    };

    glm::vec2 faceUVs[4] = {
        {0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f}
    };

    for (u32 f = 0; f < 6; ++f) {
        u32 base = f * 4;
        for (u32 v = 0; v < 4; ++v) {
            mesh.positions.push_back(faces[f].corners[v]);
            mesh.normals.push_back(faces[f].normal);
            mesh.tangents.emplace_back(faces[f].tangent, 1.0f);
            mesh.uvs.push_back(faceUVs[v]);
        }

        // Two triangles per face (CCW winding)
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

} // namespace ProceduralMeshes
} // namespace phosphor
