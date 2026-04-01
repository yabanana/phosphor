#pragma once

#include "core/types.h"
#include <vector>

namespace phosphor {

struct Meshlet {
    u32 vertexOffset;    // into meshlet vertex buffer
    u32 vertexCount;
    u32 triangleOffset;  // into meshlet triangle buffer
    u32 triangleCount;
};

struct MeshletBounds {
    float center[3];     // bounding sphere center
    float radius;        // bounding sphere radius
    float coneApex[3];   // normal cone apex
    float coneCutoff;    // cos(half-angle), <0 means >90 degrees
    float coneAxis[3];   // normal cone axis
    float pad;
};

struct MeshletBuildResult {
    std::vector<Meshlet> meshlets;
    std::vector<u32> meshletVertices;      // vertex indices
    std::vector<u8> meshletTriangles;      // triangle indices (3 bytes per tri)
    std::vector<MeshletBounds> bounds;
};

class MeshletBuilder {
public:
    /// Build meshlets from an indexed triangle mesh.
    /// @param positions      Pointer to vertex positions (float3 per vertex, may be interleaved)
    /// @param vertexCount    Number of vertices
    /// @param vertexStride   Byte stride between consecutive vertex positions
    /// @param indices        Triangle index buffer
    /// @param indexCount     Number of indices (must be a multiple of 3)
    /// @return Packed meshlet data ready for GPU upload
    static MeshletBuildResult build(
        const float* positions, size_t vertexCount, size_t vertexStride,
        const u32* indices, size_t indexCount);
};

} // namespace phosphor
