#pragma once

#include "core/types.h"
#include "renderer/gpu_types.h"

#include <string>
#include <vector>

namespace phosphor {

// ---------------------------------------------------------------------------
// Meshlet cook (F6.1) on meshoptimizer v1.3 (cmake/Dependencies.cmake).
//
// A meshlet is a range of the meshlet vertex buffer (mesh-local vertex
// indices here; GpuScene makes them global) and of the packed triangle buffer
// (3 bytes per triangle, indices local to the meshlet's vertices, BYTE
// offsets).  Bounds follow meshoptimizer's meshopt_computeMeshletBounds of
// the pinned revision: bounding sphere, and a normal cone whose test
// (renderer/meshlet_cull_reference.h) rejects a meshlet whose triangles all
// face away from the camera.  A cluster whose cone spans more than ~168
// degrees gets coneCutoff = 1 (no cone); a cluster of only degenerate
// triangles gets all-zero bounds (zero axis = no cone, never rejected by our
// test even though meshoptimizer calls it "trivial reject").
//
// Limits: the options are validated against meshoptimizer's asserts (3..256
// vertices, 1..512 triangles) AND the static outputs of the engine's mesh
// shader (MESHLET_SHADER_MAX_*): a meshlet the shader cannot emit is never
// built.  The default (64 vertices / 124 triangles, standard builder, cone
// weight 0.5) is the pre-F6 cook, unchanged for existing callers.
// ---------------------------------------------------------------------------

using Meshlet       = GPUMeshlet;
using MeshletBounds = GPUMeshletBounds;

/// Static output limits of the engine's mesh shader (shaders/meshlet.metal):
/// mesh<VertexOut, void, MESHLET_SHADER_MAX_VERTICES, MESHLET_SHADER_MAX_TRIANGLES>.
constexpr u32 MESHLET_SHADER_MAX_VERTICES  = 128;
constexpr u32 MESHLET_SHADER_MAX_TRIANGLES = 128;

enum class MeshletAlgorithm : u8 {
    Standard, // meshopt_buildMeshlets (greedy, cone weight)
    Spatial,  // meshopt_buildMeshletsSpatial (SAH-like spatial splits, fill weight)
};

struct MeshletBuildOptions {
    u32              maxVertices  = MESHLET_MAX_VERTICES;  // 64
    u32              maxTriangles = MESHLET_MAX_TRIANGLES; // 124
    /// Spatial: minimum triangles per meshlet (meshopt min_triangles); 0 =
    /// maxTriangles / 4 rounded up to a multiple of 4 (at least 1).
    u32              minTriangles = 0;
    MeshletAlgorithm algorithm    = MeshletAlgorithm::Standard;
    float            coneWeight   = 0.5f; // Standard: 0 = locality only, 1 = cone tightness only
    float            fillWeight   = 0.5f; // Spatial: meshopt fill_weight
    /// meshopt_optimizeMeshlet on every meshlet (vertex/triangle order for
    /// locality); off in the baseline.
    bool             optimize     = false;

    bool operator==(const MeshletBuildOptions&) const = default;
};

/// Empty string when `options` is legal; otherwise why not.
[[nodiscard]] std::string validateMeshletOptions(const MeshletBuildOptions& options);
/// "standard-64v124t" style name (reports, spike manifests).
[[nodiscard]] std::string meshletOptionsName(const MeshletBuildOptions& options);

struct MeshletBuildResult {
    std::vector<Meshlet> meshlets;
    std::vector<u32> meshletVertices;      // mesh-local vertex indices
    std::vector<u8> meshletTriangles;      // local triangle indices (3 bytes per triangle)
    std::vector<MeshletBounds> bounds;
};

/// Cook statistics (spike S1, reports).
struct MeshletStats {
    u32    meshlets      = 0;
    u64    triangles     = 0;   // non-degenerate and degenerate triangles cooked
    u64    vertexRefs    = 0;   // meshlet vertex entries (>= unique vertices: duplication across meshlets)
    u64    bytes         = 0;   // meshlets + bounds + vertex refs + triangle bytes
    double avgTriangles  = 0.0; // per meshlet
    double avgVertices   = 0.0;
    double fillTriangles = 0.0; // avgTriangles / maxTriangles
    double coneUsable    = 0.0; // fraction of meshlets with a usable cone (cutoff < 1, non-zero axis)
    double coneCutoffMean = 0.0; // mean cutoff over usable cones (lower = wider rejection range)
};

class MeshletBuilder {
public:
    /// Build meshlets from an indexed triangle mesh.
    /// @param positions      Pointer to vertex positions (float3 per vertex, may be interleaved)
    /// @param vertexCount    Number of vertices
    /// @param vertexStride   Byte stride between consecutive vertex positions
    /// @param indices        Triangle index buffer
    /// @param indexCount     Number of indices (must be a multiple of 3)
    /// @param options        Must pass validateMeshletOptions (asserted)
    /// @return Packed meshlet data ready for GPU upload
    static MeshletBuildResult build(
        const float* positions, size_t vertexCount, size_t vertexStride,
        const u32* indices, size_t indexCount, const MeshletBuildOptions& options = {});

    /// Statistics of a build of a mesh with `vertexCount` unique vertices.
    static MeshletStats stats(const MeshletBuildResult& result, const MeshletBuildOptions& options);
};

} // namespace phosphor
