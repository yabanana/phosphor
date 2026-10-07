#pragma once

#include "renderer/gpu_types.h"

#include <array>
#include <functional>
#include <memory>
#include <span>
#include <string>

namespace phosphor {

struct RtReferenceHit {
    double t = -1, u = 0, v = 0;
    double texU = 0, texV = 0;
    u32 slot = ~0u, mesh = ~0u, material = ~0u, primitive = ~0u, generation = 0;
    bool frontFacing = false;
    std::array<double, 3> normal{}; // unit geometric normal, WORLD winding
    [[nodiscard]] bool hit() const { return t >= 0; }
    [[nodiscard]] GPURtHit gpu() const;
};
struct RtHitCheck {
    bool ok = true, edgeTie = false;
    double relativeError = 0;
    std::string error;
};

// Diagnostic CPU reference. It owns copies, so caller readback slots can be
// recycled after these calls. Mesh BVHs are built only when geometry changes;
// setInstances builds a top-level BVH without expanding instanced triangles.
// CPU/GPU comparisons MUST use the same-frame GPUInstance world matrices.
class RtReference {
public:
    using Filter = std::function<bool(const GPURtRay&, const RtReferenceHit&)>;
    RtReference();
    ~RtReference();
    RtReference(RtReference&&) noexcept;
    RtReference& operator=(RtReference&&) noexcept;
    RtReference(const RtReference&) = delete;
    RtReference& operator=(const RtReference&) = delete;

    void setGeometry(std::span<const GPUVertex> vertices, std::span<const u32> indices,
                     std::span<const GPURtMesh> meshes);
    void setMaterials(std::span<const GPUMaterial> materials);
    // Optional masks override the default per-slot flags policy (same length).
    // With no override: primary+indirect for visible (bit 0), shadow for
    // castsShadows (bit 1), all gated by VALID. This ignores camera culling.
    // Primary rays cull backfaces unless their material is double-sided.
    // Invalid slots/mesh IDs never intersect.
    void setInstances(std::span<const GPUInstance> instances, std::span<const u32> masks = {});
    [[nodiscard]] RtReferenceHit nearest(const GPURtRay& ray, const Filter& accept = {}) const;
    [[nodiscard]] bool any(const GPURtRay& ray, const Filter& accept = {}) const;
    // Checks nearest distance AND the named triangle; accepts alternate
    // edge-tied IDs only when that triangle contains the ray within baryTolerance.
    [[nodiscard]] RtHitCheck check(const GPURtRay& ray, const GPURtHit& hit, const Filter& accept = {},
                                   double tTolerance = 2e-4, double baryTolerance = 1e-5) const;
    [[nodiscard]] size_t meshCount() const;
    [[nodiscard]] size_t instanceCount() const;
    [[nodiscard]] size_t triangleCount() const; // unique geometry, not instanced

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace phosphor
