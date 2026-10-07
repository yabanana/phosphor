#pragma once

// F9 spikes (docs/opt-log.md, "F9 — Spike"): shared helpers on top of the
// bench/soc harness.  A MEASUREMENT TOOL, not engine code (same rules as
// bench/soc/harness.h: GPU objects only through soc::Context, output only
// through ctx.log(), every benchmark sets a negative control that can fail).
// Runner: bench/soc/soc_bench.cpp (--list, --only F9-S1, --runs N, --quick,
// --validate, --force-family apple9, --out).
//
// What lives here:
//   * the CPU reference: double-precision Moller-Trumbore and an exact BVH
//     (nearest hit and any hit), used to check every GPU hit;
//   * the scene corpus: procedural meshes (scene/procedural.h) and Sponza
//     (assets/sponza, loaded with the engine's GltfLoader into a GpuScene,
//     textures kept on the CPU for alpha tests);
//   * GPU helpers: the scene's GPUVertex/index buffers uploaded once, one
//     BLAS per mesh reading them in place (stride 48, mesh-local indices),
//     a TLAS of indirect instance descriptors, AS creation from the device or
//     from a placement heap, timed acceleration-structure encoders;
//   * the ray/hit layout shared with bench/f9_spike/shaders/f9_rt.h.

#include "harness.h"

#include "renderer/gpu_scene.h"
#include "renderer/gpu_types.h"

#include <Metal/MTL4AccelerationStructure.hpp>

#include <glm/glm.hpp>

#include <array>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace f9 {

using soc::u32;
using soc::u64;
using f64 = double;

// ---------------------------------------------------------------------------
// Shaders
// ---------------------------------------------------------------------------

/// Library compiled from bench/f9_spike/shaders/<file>; `#include "x"` is
/// inlined from bench/f9_spike/shaders/, then src/, then build/generated/.
MTL::Library* f9Library(soc::Context& ctx, const std::string& file, bool fastMath = true);

/// Silences stderr while alive: the engine's CPU code (GpuScene::uploadMesh,
/// GltfLoader) logs with LOG_INFO, and --validate counts every non-"[soc]"
/// line as a validation message.  Never hold it around Metal calls.
class QuietStderr {
public:
    QuietStderr();
    ~QuietStderr();
    QuietStderr(const QuietStderr&) = delete;
    QuietStderr& operator=(const QuietStderr&) = delete;

private:
    int saved_ = -1;
};

/// Compute pipeline of `fn` statically linking the intersection functions
/// `linked` (empty: a plain pipeline).  Kept until the benchmark ends.
MTL::ComputePipelineState* linkedPipeline(soc::Context& ctx, MTL::Library* lib, const std::string& fn,
                                          const std::vector<std::string>& linked,
                                          const MTL::FunctionConstantValues* constants = nullptr);

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------

struct V3 {
    f64 x = 0, y = 0, z = 0;
};
inline V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V3 operator*(V3 a, f64 s) { return {a.x * s, a.y * s, a.z * s}; }
inline V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline f64 dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline f64 length(V3 a) { return std::sqrt(dot(a, a)); }
inline V3 normalize(V3 a) { const f64 l = length(a); return l > 0 ? a * (1.0 / l) : a; }
inline V3 toV3(const glm::vec3& v) { return {v.x, v.y, v.z}; }

struct Ray {
    V3 o, d;           // d need not be normalised; t is in units of |d|
    f64 tmin = 0, tmax = 1e30;
};

/// Watertight ray/triangle test in double precision, no culling.  Returns
/// t in (tmin, tmax) or -1; u, v = barycentrics of vertices 1 and 2.
f64 rayTriangle(const Ray& r, V3 a, V3 b, V3 c, f64* u = nullptr, f64* v = nullptr);

/// Same, without the t range, accepting barycentrics within `eps` outside
/// the triangle (edge ties); returns the plane t or -1.
f64 rayTriangleLoose(const Ray& r, V3 a, V3 b, V3 c, f64 eps);

/// A world-space triangle soup with ids (the CPU mirror of a TLAS).
struct TriangleSoup {
    std::vector<V3> v;          // 3 per triangle
    std::vector<u32> instance;  // per triangle: TLAS instance id (userID / slot)
    std::vector<u32> primitive; // per triangle: primitive id inside its BLAS geometry
    std::vector<u32> geometry;  // per triangle: geometry index inside its BLAS
    [[nodiscard]] u32 count() const { return static_cast<u32>(instance.size()); }
    void add(V3 a, V3 b, V3 c, u32 inst, u32 prim, u32 geom = 0) {
        v.push_back(a); v.push_back(b); v.push_back(c);
        instance.push_back(inst); primitive.push_back(prim); geometry.push_back(geom);
    }
};

struct CpuHit {
    f64 t = -1;
    u32 tri = ~0u; // index in the soup
    f64 u = 0, v = 0;
    [[nodiscard]] bool hit() const { return t >= 0; }
};

/// Exact BVH over a TriangleSoup (median split, double-precision slabs with
/// a conservative epsilon, leaves of <= 4 triangles).  nearest() returns the
/// smallest t (ties: lowest triangle index); any() the first hit found.
/// `accept` (optional) filters candidate hits (alpha test, self-hit rules).
class CpuBvh {
public:
    using Filter = std::function<bool(u32 tri, f64 t, f64 u, f64 v)>;
    explicit CpuBvh(const TriangleSoup& soup);
    [[nodiscard]] CpuHit nearest(const Ray& r, const Filter& accept = {}) const;
    [[nodiscard]] bool any(const Ray& r, const Filter& accept = {}) const;
    [[nodiscard]] const TriangleSoup& soup() const { return soup_; }

private:
    struct Node {
        f64 lo[3], hi[3];
        u32 first = 0, count = 0; // leaf: triangles [first, first+count) of order_; inner: count = 0, first = right child
    };
    u32 build(u32 first, u32 count, u32 depth);
    const TriangleSoup& soup_;
    std::vector<Node> nodes_;
    std::vector<u32> order_;
};

/// parallelFor over [0, n) on all cores.
template <typename F> void parallelFor(u32 n, F&& fn) {
    const u32 nt = std::max(1u, std::min(n, std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    th.reserve(nt);
    for (u32 t = 0; t < nt; ++t)
        th.emplace_back([&, t] { for (u32 i = t; i < n; i += nt) fn(i); });
    for (auto& x : th) x.join();
}

inline u32 mix32(u32 h) { h ^= h >> 16; h *= 0x7FEB352Du; h ^= h >> 15; h *= 0x846CA68Bu; h ^= h >> 16; return h; }
inline float rnd01(u32 a, u32 b) { return float(mix32(a * 0x9E3779B1u ^ mix32(b + 0x85EBCA6Bu)) >> 8) * (1.0f / 16777216.0f); }

// ---------------------------------------------------------------------------
// Scene corpus
// ---------------------------------------------------------------------------

/// RGBA8 texture kept on the CPU (alpha tests, S5).
struct CpuTexture {
    u32 width = 0, height = 0;
    bool sRGB = true;
    std::vector<soc::u8> rgba;
};

/// A scene: meshes in a GpuScene (engine layout: GPUVertex, mesh-local
/// indices, GPUMeshInfo), instances (GPUInstance, world matrices), materials
/// (GPUMaterial; texture indices into `textures`).
struct SceneData {
    std::string name;
    phosphor::GpuScene scene;
    std::vector<phosphor::GPUInstance> instances;
    std::vector<phosphor::GPUMaterial> materials;
    std::vector<CpuTexture> textures;
    [[nodiscard]] u32 meshTriangles(u32 mesh) const { return scene.meshInfos()[mesh].indexCount / 3; }
    [[nodiscard]] u64 totalTriangles() const; // over instances
    /// World-space soup of every instance (instance id = position in `instances`).
    [[nodiscard]] TriangleSoup soup() const;
    /// Object-space vertex `i` of mesh `mesh` (mesh-local index).
    [[nodiscard]] V3 vertex(u32 mesh, u32 i) const;
    /// Bounds of all instances in world space.
    void bounds(V3& lo, V3& hi) const;
};

/// Sponza from assets/sponza (false if missing: the benchmark must then
/// report Status::Skipped, never invent data).  Uses the engine loader.
bool loadSponza(SceneData& out, std::string& error);

/// Procedural corpus: one mesh per entry, one identity instance each, laid
/// out side by side (sphere, torus, cube, plane, icosphere).
void proceduralScene(SceneData& out); // `out` must be empty (GpuScene is not copyable)

/// Mesh from raw arrays (normals/tangents/uvs computed if empty).
phosphor::MeshHandle addMesh(SceneData& s, const std::vector<glm::vec3>& pos, const std::vector<u32>& idx,
                             const std::vector<glm::vec2>& uv = {});

glm::mat4 instanceMatrix(const phosphor::GPUInstance& inst);

// ---------------------------------------------------------------------------
// GPU scene + acceleration structures
// ---------------------------------------------------------------------------

inline MTL4::BufferRange range(MTL::Buffer* b, u64 offset = 0, u64 length = 0) {
    return MTL4::BufferRange::Make(b->gpuAddress() + offset, length ? length : b->length() - offset);
}

/// Where acceleration structures are allocated.
enum class AsPlacement {
    Device, // device->newAccelerationStructure (standalone allocation)
    Heap,   // placement heap (heapAccelerationStructureSizeAndAlign), what GpuMemory would do
};

struct GpuGeometry {
    MTL::Buffer* vertices = nullptr; // GPUVertex[] of the whole scene
    MTL::Buffer* indices = nullptr;  // u32[] mesh-local
    u64 vertexBytes = 0, indexBytes = 0;
};

/// Upload the scene geometry (shared storage: the CPU may deform it).
GpuGeometry uploadGeometry(soc::Context& ctx, const SceneData& s);

struct Blas {
    MTL::AccelerationStructure* as = nullptr;
    MTL4::PrimitiveAccelerationStructureDescriptor* desc = nullptr;
    MTL::AccelerationStructureSizes sizes{};
    u32 triangles = 0;
};

struct BlasOptions {
    bool opaque = true;
    MTL::AccelerationStructureUsage usage = MTL::AccelerationStructureUsageNone;
    AsPlacement placement = AsPlacement::Device;
    u32 iftOffset = 0;      // geometry intersection function table offset
};

/// Descriptor of a BLAS over mesh `mesh` reading `g` in place (vertex
/// range starts at the mesh's vertexOffset, stride sizeof(GPUVertex)).
MTL4::PrimitiveAccelerationStructureDescriptor* blasDescriptor(soc::Context& ctx, const SceneData& s,
                                                                const GpuGeometry& g, u32 mesh,
                                                                const BlasOptions& o);

/// A new acceleration structure of `size` bytes (resident, released with
/// the benchmark).  Heap: one heap per call sized from
/// heapAccelerationStructureSizeAndAlign.
MTL::AccelerationStructure* newAccelerationStructure(soc::Context& ctx, u64 size, AsPlacement placement);

/// Allocate (no build) a BLAS for every mesh.
std::vector<Blas> allocateBlases(soc::Context& ctx, const SceneData& s, const GpuGeometry& g, const BlasOptions& o);

/// Scratch buffer large enough for every build/refit of `blases` (private).
MTL::Buffer* scratchFor(soc::Context& ctx, const std::vector<Blas>& blases);

/// Build every BLAS in one compute encoder (one shared scratch: a barrier
/// AccelerationStructure -> AccelerationStructure between builds).  Returns
/// the span in ms (CommandTimer).
double buildBlases(soc::Context& ctx, std::vector<Blas>& blases, MTL::Buffer* scratch);

#pragma pack(push, 1)
/// == MTL::IndirectAccelerationStructureInstanceDescriptor (72 B packed).
struct InstanceDesc {
    float m[12]; // column-major 4x3
    u32 options, mask, iftOffset, userID;
    MTL::ResourceID blas;
};
#pragma pack(pop)
static_assert(sizeof(InstanceDesc) == sizeof(MTL::IndirectAccelerationStructureInstanceDescriptor));

/// Fill a descriptor from a GPUInstance (column-major 4x4 -> 4x3).
InstanceDesc toInstanceDesc(const phosphor::GPUInstance& inst, MTL::ResourceID blas, u32 userID, u32 options,
                            u32 mask = 0xFF);

struct Tlas {
    MTL::AccelerationStructure* as = nullptr;
    MTL4::InstanceAccelerationStructureDescriptor* desc = nullptr;
    MTL::Buffer* instances = nullptr; // InstanceDesc[capacity], shared
    MTL::Buffer* scratch = nullptr;
    MTL::AccelerationStructureSizes sizes{};
    u32 count = 0;
};

/// TLAS over `count` indirect instance descriptors (buffer allocated here,
/// filled by the caller).  usage: None / Refit / PreferFastBuild ...
Tlas allocateTlas(soc::Context& ctx, u32 count, MTL::AccelerationStructureUsage usage, AsPlacement placement);

/// Timed single compute encoder (span ms, empty span subtracted).
double timeEncoder(soc::Context& ctx, const std::function<void(MTL4::ComputeCommandEncoder*)>& fn);

// ---------------------------------------------------------------------------
// Rays and hits (layout shared with shaders/f9_rt.h)
// ---------------------------------------------------------------------------

struct GpuRay {   // 32 B
    float o[3];
    float tmin;
    float d[3];
    float tmax;
};
static_assert(sizeof(GpuRay) == 32);

struct GpuHit {   // 32 B
    float t;      // < 0: miss
    u32 instance; // instance_id (TLAS) or 0
    u32 primitive;
    u32 geometry;
    float u, v;   // triangle barycentrics
    u32 front;    // 1 = front face
    u32 extra;    // kernel-specific (e.g. user id)
};
static_assert(sizeof(GpuHit) == 32);

inline GpuRay toGpu(const Ray& r) {
    return {{float(r.o.x), float(r.o.y), float(r.o.z)}, float(r.tmin), {float(r.d.x), float(r.d.y), float(r.d.z)},
            float(std::min(r.tmax, 3.0e38))};
}
inline Ray fromGpu(const GpuRay& g) {
    return {{g.o[0], g.o[1], g.o[2]}, {g.d[0], g.d[1], g.d[2]}, g.tmin, g.tmax};
}

/// Comparison of GPU nearest hits with the CPU reference.  A GPU hit is
/// correct if (a) hit/miss agree, (b) |t_gpu - t_cpu| <= tol * max(1, t),
/// (c) the triangle the GPU names, re-intersected on the CPU, gives t_gpu
/// within tol (adjacent triangles may tie on a shared edge).
struct HitCheck {
    u32 rays = 0, wrong = 0, missMismatch = 0, tMismatch = 0, idMismatch = 0;
    f64 maxRelErr = 0;
    std::string firstError;
    [[nodiscard]] bool ok() const { return wrong == 0; }
};

/// `soupIndex(hit)` maps a GPU hit to the soup triangle it names (~0u if
/// out of range).  `accept` filters CPU candidates like the GPU did.
HitCheck checkNearest(const CpuBvh& bvh, const std::vector<Ray>& rays, const std::vector<GpuHit>& gpu,
                      const std::function<u32(const GpuHit&)>& soupIndex, f64 tol = 2e-4,
                      const CpuBvh::Filter& accept = {});

/// Map (instance, geometry, primitive) -> soup index built from a soup.
struct SoupIndex {
    explicit SoupIndex(const TriangleSoup& soup);
    [[nodiscard]] u32 operator()(u32 instance, u32 geometry, u32 primitive) const;
    std::vector<std::vector<std::vector<u32>>> map; // [instance][geometry][primitive]
};

/// Trace `rays` (nearest hit, intersector) through `as` with
/// shaders/f9_trace.metal: instanced = TLAS (trace_nearest), else a BLAS.
/// Batches of <= 1M rays per command buffer.  Waits for completion.
std::vector<GpuHit> traceNearest(soc::Context& ctx, MTL::AccelerationStructure* as, bool instanced,
                                 const std::vector<Ray>& rays, u32 mask = 0xFF);

/// Pinhole camera primary rays (row-major pixels, NDC y up like the engine).
std::vector<Ray> cameraRays(V3 eye, V3 target, f64 vfovDeg, u32 width, u32 height, u32 stride = 1);

} // namespace f9
