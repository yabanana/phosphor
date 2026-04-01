#include "renderer/meshlet_builder.h"
#include "core/log.h"

#include <meshoptimizer.h>

#include <cassert>
#include <cstring>

namespace phosphor {

MeshletBuildResult MeshletBuilder::build(
    const float* positions, size_t vertexCount, size_t vertexStride,
    const u32* indices, size_t indexCount) {

    assert(positions != nullptr);
    assert(indices != nullptr);
    assert(indexCount % 3 == 0);
    assert(vertexStride >= sizeof(float) * 3);

    constexpr size_t maxVerts = MESHLET_MAX_VERTICES;   // 64
    constexpr size_t maxTris  = MESHLET_MAX_TRIANGLES;  // 124
    constexpr float  coneWeight = 0.5f;

    // Compute worst-case upper bound for allocation
    const size_t maxMeshlets = meshopt_buildMeshletsBound(indexCount, maxVerts, maxTris);

    // Temporary buffers sized to the upper bound
    std::vector<meshopt_Meshlet> moMeshlets(maxMeshlets);
    std::vector<unsigned int> moVertices(maxMeshlets * maxVerts);
    std::vector<unsigned char> moTriangles(maxMeshlets * maxTris * 3);

    // Build the meshlets
    const size_t meshletCount = meshopt_buildMeshlets(
        moMeshlets.data(),
        moVertices.data(),
        moTriangles.data(),
        indices, indexCount,
        positions, vertexCount, vertexStride,
        maxVerts, maxTris, coneWeight);

    LOG_INFO("Built %zu meshlets from %zu triangles (%zu vertices)",
             meshletCount, indexCount / 3, vertexCount);

    // Trim the temporary meshlet array
    moMeshlets.resize(meshletCount);

    // Determine the tight bounds for the vertex and triangle arrays
    // by looking at the last meshlet's offset + count.
    size_t totalVertices  = 0;
    size_t totalTriangles = 0;
    if (meshletCount > 0) {
        const auto& last = moMeshlets[meshletCount - 1];
        totalVertices  = last.vertex_offset + last.vertex_count;
        totalTriangles = last.triangle_offset + ((last.triangle_count * 3 + 3) & ~size_t(3));
    }

    // Pack output
    MeshletBuildResult result;
    result.meshlets.reserve(meshletCount);
    result.meshletVertices.assign(moVertices.begin(), moVertices.begin() + static_cast<ptrdiff_t>(totalVertices));
    result.meshletTriangles.assign(moTriangles.begin(), moTriangles.begin() + static_cast<ptrdiff_t>(totalTriangles));
    result.bounds.reserve(meshletCount);

    for (size_t i = 0; i < meshletCount; ++i) {
        const auto& mo = moMeshlets[i];

        // Convert to our Meshlet struct
        Meshlet m{};
        m.vertexOffset   = mo.vertex_offset;
        m.vertexCount    = mo.vertex_count;
        m.triangleOffset = mo.triangle_offset;
        m.triangleCount  = mo.triangle_count;
        result.meshlets.push_back(m);

        // Compute bounds for this meshlet
        meshopt_Bounds mb = meshopt_computeMeshletBounds(
            &moVertices[mo.vertex_offset],
            &moTriangles[mo.triangle_offset],
            mo.triangle_count,
            positions, vertexCount, vertexStride);

        MeshletBounds b{};
        b.center[0]   = mb.center[0];
        b.center[1]   = mb.center[1];
        b.center[2]   = mb.center[2];
        b.radius       = mb.radius;
        b.coneApex[0]  = mb.cone_apex[0];
        b.coneApex[1]  = mb.cone_apex[1];
        b.coneApex[2]  = mb.cone_apex[2];
        b.coneCutoff   = mb.cone_cutoff;
        b.coneAxis[0]  = mb.cone_axis[0];
        b.coneAxis[1]  = mb.cone_axis[1];
        b.coneAxis[2]  = mb.cone_axis[2];
        b.pad          = 0.0f;
        result.bounds.push_back(b);
    }

    return result;
}

} // namespace phosphor
