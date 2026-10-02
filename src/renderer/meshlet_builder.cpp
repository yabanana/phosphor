#include "renderer/meshlet_builder.h"
#include "core/log.h"

#include <meshoptimizer.h>

#include <algorithm>
#include <cassert>
#include <cstring>

namespace phosphor {

namespace {

u32 effectiveMinTriangles(const MeshletBuildOptions& o) {
    if (o.minTriangles != 0) return o.minTriangles;
    const u32 quarter = (o.maxTriangles / 4 + 3) & ~3u;
    return std::clamp(quarter, 1u, o.maxTriangles);
}

} // namespace

std::string validateMeshletOptions(const MeshletBuildOptions& o) {
    // meshoptimizer v1.3 asserts (clusterizer.cpp): 3 <= max_vertices <= 256,
    // 1 <= max_triangles <= 512, 1 <= min_triangles <= max_triangles,
    // 0 <= cone_weight <= 1.
    if (o.maxVertices < 3 || o.maxVertices > 256) return "max vertices must be in 3..256 (meshoptimizer)";
    if (o.maxTriangles < 1 || o.maxTriangles > 512) return "max triangles must be in 1..512 (meshoptimizer)";
    if (o.maxVertices > MESHLET_SHADER_MAX_VERTICES) return "max vertices exceeds the mesh shader output (128)";
    if (o.maxTriangles > MESHLET_SHADER_MAX_TRIANGLES) return "max triangles exceeds the mesh shader output (128)";
    if (o.algorithm == MeshletAlgorithm::Spatial) {
        const u32 minT = effectiveMinTriangles(o);
        if (minT < 1 || minT > o.maxTriangles) return "min triangles must be in 1..max triangles";
        if (!(o.fillWeight >= 0.0f && o.fillWeight <= 1.0f)) return "fill weight must be in 0..1";
    } else if (!(o.coneWeight >= 0.0f && o.coneWeight <= 1.0f)) {
        return "cone weight must be in 0..1";
    }
    return {};
}

std::string meshletOptionsName(const MeshletBuildOptions& o) {
    std::string s = o.algorithm == MeshletAlgorithm::Spatial ? "spatial-" : "standard-";
    s += std::to_string(o.maxVertices) + "v" + std::to_string(o.maxTriangles) + "t";
    if (o.optimize) s += "-opt";
    return s;
}

MeshletBuildResult MeshletBuilder::build(
    const float* positions, size_t vertexCount, size_t vertexStride,
    const u32* indices, size_t indexCount, const MeshletBuildOptions& options) {

    assert(positions != nullptr);
    assert(indices != nullptr);
    assert(indexCount % 3 == 0);
    assert(vertexStride >= sizeof(float) * 3);
    assert(validateMeshletOptions(options).empty());

    const size_t maxVerts = options.maxVertices;
    const size_t maxTris  = options.maxTriangles;
    const size_t minTris  = effectiveMinTriangles(options);
    const bool   spatial  = options.algorithm == MeshletAlgorithm::Spatial;

    // Worst-case upper bound for allocation (spatial: bounded with min_triangles).
    const size_t maxMeshlets = meshopt_buildMeshletsBound(indexCount, maxVerts, spatial ? minTris : maxTris);

    // Temporary buffers sized to the upper bound
    std::vector<meshopt_Meshlet> moMeshlets(maxMeshlets);
    std::vector<unsigned int> moVertices(maxMeshlets * maxVerts);
    std::vector<unsigned char> moTriangles(maxMeshlets * maxTris * 3);

    size_t meshletCount = 0;
    if (spatial) {
        meshletCount = meshopt_buildMeshletsSpatial(moMeshlets.data(), moVertices.data(), moTriangles.data(),
                                                    indices, indexCount, positions, vertexCount, vertexStride,
                                                    maxVerts, minTris, maxTris, options.fillWeight);
    } else {
        meshletCount = meshopt_buildMeshlets(moMeshlets.data(), moVertices.data(), moTriangles.data(),
                                             indices, indexCount, positions, vertexCount, vertexStride,
                                             maxVerts, maxTris, options.coneWeight);
    }

    LOG_INFO("Built %zu meshlets (%s) from %zu triangles (%zu vertices)",
             meshletCount, meshletOptionsName(options).c_str(), indexCount / 3, vertexCount);

    moMeshlets.resize(meshletCount);
    if (options.optimize) {
        for (const meshopt_Meshlet& m : moMeshlets) {
            meshopt_optimizeMeshlet(&moVertices[m.vertex_offset], &moTriangles[m.triangle_offset], m.triangle_count,
                                    m.vertex_count);
        }
    }

    // v1.3 packs the arrays without padding (triangle_offset += count * 3):
    // the tight sizes are the last meshlet's offset + count.
    size_t totalVertices  = 0;
    size_t totalTriangles = 0;
    if (meshletCount > 0) {
        const auto& last = moMeshlets[meshletCount - 1];
        totalVertices  = last.vertex_offset + last.vertex_count;
        totalTriangles = last.triangle_offset + last.triangle_count * 3;
    }

    // Pack output
    MeshletBuildResult result;
    result.meshlets.reserve(meshletCount);
    result.meshletVertices.assign(moVertices.begin(), moVertices.begin() + static_cast<ptrdiff_t>(totalVertices));
    result.meshletTriangles.assign(moTriangles.begin(), moTriangles.begin() + static_cast<ptrdiff_t>(totalTriangles));
    result.bounds.reserve(meshletCount);

    for (size_t i = 0; i < meshletCount; ++i) {
        const auto& mo = moMeshlets[i];

        Meshlet m{};
        m.vertexOffset   = mo.vertex_offset;
        m.vertexCount    = mo.vertex_count;
        m.triangleOffset = mo.triangle_offset;
        m.triangleCount  = mo.triangle_count;
        result.meshlets.push_back(m);

        // Bounds of the (possibly optimised) meshlet.
        meshopt_Bounds mb = meshopt_computeMeshletBounds(
            &moVertices[mo.vertex_offset],
            &moTriangles[mo.triangle_offset],
            mo.triangle_count,
            positions, vertexCount, vertexStride);

        MeshletBounds b{};
        b.center[0]   = mb.center[0];
        b.center[1]   = mb.center[1];
        b.center[2]   = mb.center[2];
        b.radius      = mb.radius;
        b.coneApex[0] = mb.cone_apex[0];
        b.coneApex[1] = mb.cone_apex[1];
        b.coneApex[2] = mb.cone_apex[2];
        b.coneCutoff  = mb.cone_cutoff;
        b.coneAxis[0] = mb.cone_axis[0];
        b.coneAxis[1] = mb.cone_axis[1];
        b.coneAxis[2] = mb.cone_axis[2];
        b.pad         = 0.0f;
        result.bounds.push_back(b);
    }

    return result;
}

MeshletStats MeshletBuilder::stats(const MeshletBuildResult& r, const MeshletBuildOptions& options) {
    MeshletStats s;
    s.meshlets   = static_cast<u32>(r.meshlets.size());
    s.vertexRefs = r.meshletVertices.size();
    s.bytes      = r.meshlets.size() * sizeof(Meshlet) + r.bounds.size() * sizeof(MeshletBounds) +
              r.meshletVertices.size() * sizeof(u32) + r.meshletTriangles.size();
    u64 usable = 0;
    double cutoffSum = 0.0;
    for (size_t i = 0; i < r.meshlets.size(); ++i) {
        s.triangles += r.meshlets[i].triangleCount;
        const MeshletBounds& b = r.bounds[i];
        const bool axis = b.coneAxis[0] != 0.0f || b.coneAxis[1] != 0.0f || b.coneAxis[2] != 0.0f;
        if (axis && b.coneCutoff < 1.0f) {
            ++usable;
            cutoffSum += b.coneCutoff;
        }
    }
    if (s.meshlets > 0) {
        s.avgTriangles  = static_cast<double>(s.triangles) / s.meshlets;
        s.avgVertices   = static_cast<double>(s.vertexRefs) / s.meshlets;
        s.fillTriangles = s.avgTriangles / options.maxTriangles;
        s.coneUsable    = static_cast<double>(usable) / s.meshlets;
    }
    if (usable > 0) s.coneCutoffMean = cutoffSum / static_cast<double>(usable);
    return s;
}

} // namespace phosphor
