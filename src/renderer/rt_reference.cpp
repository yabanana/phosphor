#include "renderer/rt_reference.h"

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <iomanip>
#include <locale>
#include <sstream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace phosphor {
namespace {
using V3 = glm::dvec3;
struct Ray { V3 o, d; double tmin, tmax; };
struct Bounds {
    V3 lo{std::numeric_limits<double>::infinity()};
    V3 hi{-std::numeric_limits<double>::infinity()};
    void add(V3 p) { lo = glm::min(lo, p); hi = glm::max(hi, p); }
    void add(const Bounds& b) { add(b.lo); add(b.hi); }
    void expand() {
        for (int k = 0; k < 3; ++k) {
            const double e = 1e-9 * std::max({1.0, std::abs(lo[k]), std::abs(hi[k])});
            lo[k] -= e;
            hi[k] += e;
        }
    }
};

bool finite(V3 v) { return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z); }
Ray fromGpu(const GPURtRay& r) { return {{r.ox, r.oy, r.oz}, {r.dx, r.dy, r.dz}, r.tmin, r.tmax}; }
bool valid(const Ray& r) {
    return finite(r.o) && finite(r.d) && glm::dot(r.d, r.d) > 0 &&
           std::isfinite(r.tmin) && !std::isnan(r.tmax) && r.tmax > r.tmin;
}

bool slab(const Bounds& b, const Ray& r, double limit) {
    double lo = r.tmin, hi = std::min(limit, r.tmax);
    for (int k = 0; k < 3; ++k) {
        if (r.d[k] == 0) {
            if (r.o[k] < b.lo[k] || r.o[k] > b.hi[k]) return false;
            continue;
        }
        double a = (b.lo[k] - r.o[k]) / r.d[k], c = (b.hi[k] - r.o[k]) / r.d[k];
        if (a > c) std::swap(a, c);
        lo = std::max(lo, a);
        hi = std::min(hi, c);
        if (hi < lo) return false;
    }
    return true;
}

// Woop/Benthin/Wald's sheared watertight test, as validated by F9-S0.
// No direction normalization: inverse-transforming a ray preserves world t.
double triangle(const Ray& r, V3 a, V3 b, V3 c, double& u, double& v) {
    int kz = 0;
    if (std::abs(r.d[1]) > std::abs(r.d[kz])) kz = 1;
    if (std::abs(r.d[2]) > std::abs(r.d[kz])) kz = 2;
    if (r.d[kz] == 0) return -1;
    int kx = (kz + 1) % 3, ky = (kx + 1) % 3;
    if (r.d[kz] < 0) std::swap(kx, ky);
    const double sx = r.d[kx] / r.d[kz], sy = r.d[ky] / r.d[kz], sz = 1.0 / r.d[kz];
    a -= r.o; b -= r.o; c -= r.o;
    const double ax = a[kx] - sx * a[kz], ay = a[ky] - sy * a[kz];
    const double bx = b[kx] - sx * b[kz], by = b[ky] - sy * b[kz];
    const double cx = c[kx] - sx * c[kz], cy = c[ky] - sy * c[kz];
    const double U = cx * by - cy * bx, V = ax * cy - ay * cx, W = bx * ay - by * ax;
    if ((U < 0 || V < 0 || W < 0) && (U > 0 || V > 0 || W > 0)) return -1;
    const double det = U + V + W;
    if (det == 0) return -1;
    const double t = (U * sz * a[kz] + V * sz * b[kz] + W * sz * c[kz]) / det;
    if (!(t > r.tmin && t < r.tmax)) return -1;
    u = V / det; v = W / det;
    return t;
}

// A named edge triangle may be the alternate float-hardware tie winner.
// Keep nearest distance strict, but allow its barycentrics the S0 tolerance.
double looseTriangle(const Ray& r, V3 a, V3 b, V3 c, double eps, double& u, double& v) {
    const V3 e1 = b - a, e2 = c - a, p = glm::cross(r.d, e2);
    const double det = glm::dot(e1, p);
    if (det == 0) return -1;
    const V3 s = r.o - a, q = glm::cross(s, e1);
    u = glm::dot(s, p) / det;
    v = glm::dot(r.d, q) / det;
    if (u < -eps || v < -eps || u + v > 1 + eps) return -1;
    const double t = glm::dot(e2, q) / det;
    return (t > r.tmin && t < r.tmax) ? t : -1;
}

struct Bvh {
    struct Node { Bounds bounds; u32 first = 0, count = 0; };
    std::vector<Node> nodes;
    std::vector<u32> order;
    void build(const std::vector<Bounds>& bounds) {
        nodes.clear();
        order.resize(bounds.size());
        std::iota(order.begin(), order.end(), 0u);
        if (!bounds.empty()) buildRange(bounds, 0, static_cast<u32>(bounds.size()), 0);
    }
    u32 buildRange(const std::vector<Bounds>& bounds, u32 first, u32 count, u32 depth) {
        const u32 index = static_cast<u32>(nodes.size());
        nodes.emplace_back();
        Node n;
        for (u32 i = first; i < first + count; ++i) n.bounds.add(bounds[order[i]]);
        n.bounds.expand();
        if (count <= 4 || depth >= 60) {
            n.first = first;
            n.count = count;
        } else {
            const V3 extent = n.bounds.hi - n.bounds.lo;
            int axis = extent.y > extent.x ? 1 : 0;
            if (extent.z > extent[axis]) axis = 2;
            const u32 mid = first + count / 2;
            std::nth_element(order.begin() + first, order.begin() + mid, order.begin() + first + count,
                [&](u32 a, u32 b) { return bounds[a].lo[axis] + bounds[a].hi[axis] <
                                          bounds[b].lo[axis] + bounds[b].hi[axis]; });
            buildRange(bounds, first, mid - first, depth + 1);
            n.first = buildRange(bounds, mid, first + count - mid, depth + 1);
        }
        nodes[index] = n;
        return index;
    }
    template<class Limit, class Visit> bool visit(const Ray& r, Limit limit, Visit visitLeaf) const {
        if (nodes.empty()) return true;
        std::array<u32, 128> stack{};
        u32 sp = 0;
        stack[sp++] = 0;
        while (sp) {
            const u32 index = stack[--sp];
            const auto& n = nodes[index];
            if (!slab(n.bounds, r, limit())) continue;
            if (n.count) {
                for (u32 i = n.first; i < n.first + n.count; ++i)
                    if (!visitLeaf(order[i])) return false;
            } else {
                if (sp + 2 > stack.size()) throw std::logic_error("RT reference BVH stack overflow");
                stack[sp++] = index + 1;
                stack[sp++] = n.first;
            }
        }
        return true;
    }
};
} // namespace

struct RtReference::Impl {
    struct Tri { std::array<u32, 3> vertices; };
    struct Mesh { std::vector<Tri> triangles; Bvh bvh; Bounds bounds; };
    struct Instance {
        GPUInstance gpu{};
        glm::dmat4 world{1}, inverse{1};
        u32 mask = 0;
    };
    std::vector<GPUVertex> vertices;
    std::vector<Mesh> meshes;
    std::vector<Instance> instances;
    std::vector<GPUMaterial> materials;
    std::vector<u32> topSlots;
    Bvh top;
    size_t triangleCount = 0;

    V3 position(u32 index) const {
        const auto& v = vertices[index];
        return {v.px, v.py, v.pz};
    }
    Ray local(const Ray& r, const Instance& i) const {
        return {V3(i.inverse * glm::dvec4(r.o, 1)), V3(i.inverse * glm::dvec4(r.d, 0)), r.tmin, r.tmax};
    }
    RtReferenceHit candidate(u32 slot, u32 prim, double t, double u, double v, const Ray& ray) const {
        const auto& i = instances[slot];
        const auto& tri = meshes[i.gpu.meshIndex].triangles[prim];
        RtReferenceHit h;
        h.t = t; h.u = u; h.v = v; h.slot = slot; h.mesh = i.gpu.meshIndex;
        h.material = i.gpu.materialIndex; h.primitive = prim; h.generation = i.gpu.generation;
        const auto& a = vertices[tri.vertices[0]];
        const auto& b = vertices[tri.vertices[1]];
        const auto& c = vertices[tri.vertices[2]];
        h.texU = (1-u-v) * a.u + u * b.u + v * c.u;
        h.texV = (1-u-v) * a.v + u * b.v + v * c.v;
        const V3 wa = V3(i.world * glm::dvec4(position(tri.vertices[0]), 1));
        const V3 wb = V3(i.world * glm::dvec4(position(tri.vertices[1]), 1));
        const V3 wc = V3(i.world * glm::dvec4(position(tri.vertices[2]), 1));
        const V3 n = glm::cross(wb-wa, wc-wa);
        h.frontFacing = glm::dot(n, ray.d) < 0;
        const double length = glm::length(n);
        if (length > 0) for (int k = 0; k < 3; ++k) h.normal[k] = n[k] / length;
        return h;
    }
    bool acceptFace(const GPURtRay& ray, const RtReferenceHit& hit) const {
        if (hit.material >= materials.size()) return false;
        if (ray.type != RT_PROBE_PRIMARY || (materials[hit.material].flags & MATERIAL_FLAG_DOUBLE_SIDED)) return true;
        // The raster/RT primary culling contract follows object winding. A
        // mirrored transform reverses world winding, reported separately in
        // frontFacing, but does not reverse this application-facing front.
        const auto& i = instances[hit.slot];
        const auto& tri = meshes[hit.mesh].triangles[hit.primitive];
        const V3 a = position(tri.vertices[0]), b = position(tri.vertices[1]), c = position(tri.vertices[2]);
        return glm::dot(glm::cross(b-a, c-a), local(fromGpu(ray), i).d) < 0;
    }
    RtReferenceHit trace(const GPURtRay& gpu, const Filter& accept, bool any) const {
        RtReferenceHit best;
        const Ray ray = fromGpu(gpu);
        if (!valid(ray) || !gpu.mask) return best;
        auto limit = [&] { return best.hit() ? best.t : ray.tmax; };
        top.visit(ray, limit, [&](u32 topIndex) {
            const u32 slot = topSlots[topIndex];
            const auto& i = instances[slot];
            if (!(i.mask & gpu.mask)) return true;
            const auto& mesh = meshes[i.gpu.meshIndex];
            const Ray r = local(ray, i);
            return mesh.bvh.visit(r, limit, [&](u32 primitive) {
                const auto& tri = mesh.triangles[primitive];
                double u = 0, v = 0;
                const double t = triangle(r, position(tri.vertices[0]), position(tri.vertices[1]),
                                          position(tri.vertices[2]), u, v);
                if (t < 0 || (best.hit() && (t > best.t || (t == best.t &&
                    (slot > best.slot || (slot == best.slot && primitive > best.primitive)))))) return true;
                auto h = candidate(slot, primitive, t, u, v, ray);
                if (!acceptFace(gpu, h) || (accept && !accept(gpu, h))) return true;
                best = h;
                return !any;
            });
        });
        return best;
    }
};

GPURtHit RtReferenceHit::gpu() const {
    return {static_cast<float>(t), static_cast<float>(u), static_cast<float>(v), slot, primitive,
            generation, frontFacing ? 1u : 0u, hit() ? 1u : 0u};
}
RtReference::RtReference() : impl_(std::make_unique<Impl>()) {}
RtReference::~RtReference() = default;
RtReference::RtReference(RtReference&&) noexcept = default;
RtReference& RtReference::operator=(RtReference&&) noexcept = default;

void RtReference::setGeometry(std::span<const GPUVertex> vertices, std::span<const u32> indices,
                              std::span<const GPURtMesh> meshes) {
    Impl replacement;
    replacement.vertices.assign(vertices.begin(), vertices.end());
    replacement.materials = impl_->materials;
    replacement.meshes.resize(meshes.size());
    for (size_t m = 0; m < meshes.size(); ++m) {
        const auto& src = meshes[m];
        if (src.indexCount % 3 || u64(src.indexOffset) + src.indexCount > indices.size())
            throw std::invalid_argument("RT reference mesh index range invalid");
        auto& dst = replacement.meshes[m];
        std::vector<Bounds> bounds;
        bounds.reserve(src.indexCount / 3);
        for (u32 j = 0; j < src.indexCount; j += 3) {
            Impl::Tri tri;
            Bounds b;
            for (u32 k = 0; k < 3; ++k) {
                const u64 index = u64(src.vertexOffset) + indices[src.indexOffset + j + k];
                if (index >= vertices.size()) throw std::invalid_argument("RT reference vertex index invalid");
                tri.vertices[k] = static_cast<u32>(index);
                const V3 p = replacement.position(tri.vertices[k]);
                if (!finite(p)) throw std::invalid_argument("RT reference vertex is not finite");
                b.add(p);
            }
            dst.triangles.push_back(tri);
            bounds.push_back(b);
            dst.bounds.add(b);
        }
        dst.bvh.build(bounds);
        replacement.triangleCount += dst.triangles.size();
    }
    *impl_ = std::move(replacement); // New geometry invalidates old instance snapshots.
}

void RtReference::setMaterials(std::span<const GPUMaterial> materials) {
    impl_->materials.assign(materials.begin(), materials.end());
}

void RtReference::setInstances(std::span<const GPUInstance> instances, std::span<const u32> masks) {
    if (!masks.empty() && masks.size() != instances.size())
        throw std::invalid_argument("RT reference mask count differs from instance count");
    impl_->instances.clear();
    impl_->instances.resize(instances.size());
    impl_->topSlots.clear();
    std::vector<Bounds> bounds;
    bounds.reserve(instances.size());
    for (u32 slot = 0; slot < instances.size(); ++slot) {
        auto& dst = impl_->instances[slot];
        dst.gpu = instances[slot];
        if (!(dst.gpu.flags & INSTANCE_FLAG_VALID) || dst.gpu.meshIndex >= impl_->meshes.size() ||
            dst.gpu.materialIndex >= impl_->materials.size()) continue;
        const auto& mesh = impl_->meshes[dst.gpu.meshIndex];
        if (mesh.triangles.empty()) continue;
        dst.mask = masks.empty() ? ((dst.gpu.flags & 1u ? RT_MASK_PRIMARY | RT_MASK_INDIRECT : 0u) |
                                   (dst.gpu.flags & 2u ? RT_MASK_SHADOW : 0u)) : masks[slot];
        if (!dst.mask) continue;
        dst.world = glm::dmat4(glm::make_mat4(dst.gpu.modelMatrix));
        bool validMatrix = true;
        for (int c = 0; c < 4; ++c) for (int r = 0; r < 4; ++r)
            validMatrix = validMatrix && std::isfinite(dst.world[c][r]);
        // Mirror the descriptor kernel's float determinant eligibility gate;
        // retain a double inverse/intersection for the independent reference.
        const float determinant = glm::determinant(glm::mat3(glm::make_mat4(dst.gpu.modelMatrix)));
        if (!validMatrix || !std::isfinite(determinant) || std::abs(determinant) <= 1e-20f ||
            dst.world[0][3] != 0 || dst.world[1][3] != 0 || dst.world[2][3] != 0 || dst.world[3][3] != 1) {
            dst.mask = 0;
            continue;
        }
        dst.inverse = glm::inverse(dst.world);
        Bounds world;
        for (u32 c = 0; c < 8; ++c) {
            V3 p;
            for (u32 k = 0; k < 3; ++k) p[k] = (c & (1u << k)) ? mesh.bounds.hi[k] : mesh.bounds.lo[k];
            world.add(V3(dst.world * glm::dvec4(p, 1)));
        }
        bounds.push_back(world);
        impl_->topSlots.push_back(slot);
    }
    impl_->top.build(bounds);
}

RtReferenceHit RtReference::nearest(const GPURtRay& ray, const Filter& accept) const {
    return impl_->trace(ray, accept, false);
}
bool RtReference::any(const GPURtRay& ray, const Filter& accept) const {
    return impl_->trace(ray, accept, true).hit();
}

RtHitCheck RtReference::check(const GPURtRay& gpuRay, const GPURtHit& gpu, const Filter& accept,
                             double tTolerance, double baryTolerance) const {
    RtHitCheck result;
    auto fail = [&](const char* reason) { result.ok = false; result.error = reason; return result; };
    if (!(tTolerance >= 0) || !(baryTolerance >= 0)) return fail("invalid comparison tolerance");
    const Ray ray = fromGpu(gpuRay);
    if (!valid(ray)) return fail("invalid ray");
    if (!std::isfinite(gpu.t) || gpu.hit > 1 || gpu.frontFacing > 1 ||
        (bool(gpu.hit) != (gpu.t >= 0))) return fail("inconsistent GPU hit state");
    const bool shadow = gpuRay.type == RT_PROBE_SHADOW;
    const auto expected = impl_->trace(gpuRay, accept, shadow);
    if (expected.hit() != bool(gpu.hit)) return fail("hit/miss mismatch");
    if (!expected.hit()) return result;
    if (!shadow) {
        result.relativeError = std::abs(expected.t - gpu.t) / std::max(1.0, std::abs(expected.t));
        if (result.relativeError > tTolerance) return fail("nearest distance mismatch");
    }
    if (gpu.slot >= impl_->instances.size()) return fail("instance slot out of range");
    const auto& instance = impl_->instances[gpu.slot];
    if (!(instance.mask & gpuRay.mask)) return fail("hit on masked/invalid instance");
    if (gpu.generation != instance.gpu.generation) return fail("instance generation mismatch");
    const auto& mesh = impl_->meshes[instance.gpu.meshIndex];
    if (gpu.primitive >= mesh.triangles.size()) return fail("primitive out of range");
    const auto& tri = mesh.triangles[gpu.primitive];
    double u = 0, v = 0;
    const double t = looseTriangle(impl_->local(ray, instance), impl_->position(tri.vertices[0]),
        impl_->position(tri.vertices[1]), impl_->position(tri.vertices[2]), baryTolerance, u, v);
    if (t < 0 || std::abs(t - gpu.t) > tTolerance * std::max(1.0, std::abs(t)))
        return fail("named triangle does not contain the reported hit");
    if (shadow) result.relativeError = std::abs(t - gpu.t) / std::max(1.0, std::abs(t));
    const auto named = impl_->candidate(gpu.slot, gpu.primitive, t, u, v, ray);
    if (!impl_->acceptFace(gpuRay, named) || (accept && !accept(gpuRay, named)))
        return fail("named triangle rejected by face/alpha rule");
    if (bool(gpu.frontFacing) != named.frontFacing) return fail("front-facing mismatch");
    if (!std::isfinite(gpu.u) || !std::isfinite(gpu.v) ||
        std::abs(gpu.u - u) > baryTolerance || std::abs(gpu.v - v) > baryTolerance) {
        // Diagnostics only: keep the original acceptance predicate above.
        // Barycentric coordinates are dimensionless and their conditioning
        // depends on triangle aspect ratio and ray incidence. These world-space
        // residuals distinguish a large geometric error from a parameter error
        // magnified by a tiny/skinny triangle or a grazing ray.
        const V3 a = V3(instance.world * glm::dvec4(impl_->position(tri.vertices[0]), 1));
        const V3 b = V3(instance.world * glm::dvec4(impl_->position(tri.vertices[1]), 1));
        const V3 c = V3(instance.world * glm::dvec4(impl_->position(tri.vertices[2]), 1));
        const V3 e1 = b - a, e2 = c - a;
        const double edge = std::max({glm::length(e1), glm::length(e2), glm::length(c - b)});
        const V3 normal = glm::cross(e1, e2);
        const double area = glm::length(normal), directionLength = glm::length(ray.d);
        const V3 cpuPoint = a + u * e1 + v * e2;
        const V3 gpuPoint = a + double(gpu.u) * e1 + double(gpu.v) * e2;
        const V3 rayPoint = ray.o + double(gpu.t) * ray.d;
        double scale = 0;
        for (const V3 p : {a, b, c, ray.o, rayPoint})
            for (int k = 0; k < 3; ++k) scale = std::max(scale, std::abs(p[k]));
        const float fpScale = static_cast<float>(scale);
        const double fpUlp = double(std::nextafter(fpScale, std::numeric_limits<float>::infinity())) - fpScale;
        std::ostringstream diagnostic;
        diagnostic.imbue(std::locale::classic());
        diagnostic << std::scientific << std::setprecision(12)
                   << "barycentric mismatch: slot=" << gpu.slot << " mesh=" << instance.gpu.meshIndex
                   << " primitive=" << gpu.primitive << " ray_type=" << gpuRay.type
                   << " cpu_uv=(" << u << ',' << v << ") gpu_uv=(" << gpu.u << ',' << gpu.v << ')'
                   << " abs_delta_uv=(" << std::abs(gpu.u-u) << ',' << std::abs(gpu.v-v) << ')'
                   << " bary_tolerance=" << baryTolerance
                   << " cpu_t=" << t << " gpu_t=" << gpu.t
                   << " edge_max_world=" << edge << " altitude_min_world=" << (edge > 0 ? area / edge : 0)
                   << " incidence_cos=" << (area > 0 ? std::abs(glm::dot(normal, ray.d)) / (area * directionLength) : 0)
                   << " spatial_residual_world=" << glm::length(gpuPoint-rayPoint)
                   << " bary_displacement_world=" << glm::length(gpuPoint-cpuPoint)
                   << " ray_perpendicular_residual_world=" << glm::length(glm::cross(gpuPoint-ray.o, ray.d)) / directionLength
                   << " coordinate_scale_world=" << scale << " fp32_ulp_world=" << fpUlp;
        result.ok = false;
        result.error = diagnostic.str();
        return result;
    }
    result.edgeTie = !shadow && (gpu.slot != expected.slot || gpu.primitive != expected.primitive);
    return result;
}

size_t RtReference::meshCount() const { return impl_->meshes.size(); }
size_t RtReference::instanceCount() const { return impl_->instances.size(); }
size_t RtReference::triangleCount() const { return impl_->triangleCount; }

} // namespace phosphor
