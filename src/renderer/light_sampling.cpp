#include "renderer/light_sampling.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace phosphor::di {
namespace {
constexpr double pi = 3.14159265358979323846;
glm::dvec3 v3(const float* v) { return {v[0], v[1], v[2]}; }
bool finite(glm::dvec3 v) { return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z); }
double luminance(glm::dvec3 c) { return glm::dot(c, glm::dvec3(0.2126, 0.7152, 0.0722)); }
double attenuation(double distance, double range) {
    const double r = distance / std::max(range, 1e-3);
    const double window = std::clamp(1.0 - r * r * r * r, 0.0, 1.0);
    return window * window / std::max(distance * distance, 1e-4);
}
glm::dvec3 viewPoint(const GPULightClusterParams& p, glm::dvec3 world) {
    return glm::dvec3(p.view[0], p.view[1], p.view[2]) * world.x +
           glm::dvec3(p.view[4], p.view[5], p.view[6]) * world.y +
           glm::dvec3(p.view[8], p.view[9], p.view[10]) * world.z +
           glm::dvec3(p.view[12], p.view[13], p.view[14]);
}
bool clusterIntersects(const GPULightClusterParams& p, u32 index, const GPUSampledLight& light) {
    // Range encloses receiving surfaces; adding emitter extent is conservative.
    // No range means unbounded, so keep the light in every cluster.
    if (!(light.range > 0) || !std::isfinite(light.range)) return true;
    double extent = 0;
    if (light.type == DI_LIGHT_RECTANGLE || light.type == DI_LIGHT_TRIANGLE)
        extent = glm::length(v3(light.axisU)) + glm::length(v3(light.axisV));
    if (light.type == DI_LIGHT_DISK) extent = light.radius * (glm::length(v3(light.axisU)) + glm::length(v3(light.axisV)));
    if (light.type == DI_LIGHT_TUBE) extent = glm::length(v3(light.axisU)) + light.radius;
    const double r = std::max(0.0, double(light.range)) + std::max(0.0, extent);
    const auto center = viewPoint(p, v3(light.position));
    if (!finite(center) || !std::isfinite(r)) return true;
    const u32 x = index % p.gridX, y = (index / p.gridX) % p.gridY, z = index / (p.gridX * p.gridY);
    const double z0 = p.nearPlane * std::pow(double(p.farPlane) / p.nearPlane, double(z) / p.gridZ);
    const double z1 = p.nearPlane * std::pow(double(p.farPlane) / p.nearPlane, double(z + 1u) / p.gridZ);
    const double depth = -center.z;
    if (depth + r < z0 || depth - r > z1) return false;
    const double left = (2.0 * x / p.gridX - 1.0) * p.tanHalfFovX;
    const double right = (2.0 * (x + 1u) / p.gridX - 1.0) * p.tanHalfFovX;
    const double bottom = (2.0 * y / p.gridY - 1.0) * p.tanHalfFovY;
    const double top = (2.0 * (y + 1u) / p.gridY - 1.0) * p.tanHalfFovY;
    return center.x - left * depth >= -r * std::sqrt(1.0 + left * left) &&
           right * depth - center.x >= -r * std::sqrt(1.0 + right * right) &&
           center.y - bottom * depth >= -r * std::sqrt(1.0 + bottom * bottom) &&
           top * depth - center.y >= -r * std::sqrt(1.0 + top * top);
}
} // namespace

void AliasTable::rebuild(std::span<const double> weights, u32 newRevision) {
    if (weights.size() > std::numeric_limits<u32>::max()) throw std::length_error("too many lights");
    entries.assign(weights.size(), {});
    revision = newRevision;
    if (entries.empty()) return;
    const u32 n = u32(entries.size());
    double maximum = 0;
    for (double w : weights) if (std::isfinite(w) && w > 0) maximum = std::max(maximum, w);
    std::vector<double> pdf(n, 1.0 / n), scaled(n);
    if (maximum > 0) {
        double sum = 0;
        for (u32 i = 0; i < n; ++i) { pdf[i] = std::isfinite(weights[i]) && weights[i] > 0 ? weights[i] / maximum : 0; sum += pdf[i]; }
        constexpr double supportMixture = 1e-6;
        for (double& w : pdf) w = (1.0 - supportMixture) * w / sum + supportMixture / n;
    }
    std::vector<u32> small, large;
    for (u32 i = 0; i < n; ++i) {
        scaled[i] = pdf[i] * n;
        entries[i] = {1.0f, float(pdf[i]), i, i};
        (scaled[i] < 1 ? small : large).push_back(i);
    }
    while (!small.empty() && !large.empty()) {
        const auto s = small.back(); small.pop_back();
        const auto l = large.back(); large.pop_back();
        entries[s].probability = float(scaled[s]);
        entries[s].alias = l;
        scaled[l] += scaled[s] - 1;
        (scaled[l] < 1 ? small : large).push_back(l);
    }
    // Floats are the upload representation. Recompute the actual distribution
    // of these thresholds so candidate PDFs match the quantized alias table.
    std::vector<double> actual(n, 0);
    for (u32 i = 0; i < n; ++i) {
        actual[i] += double(entries[i].probability) / n;
        actual[entries[i].alias] += (1.0 - double(entries[i].probability)) / n;
    }
    for (u32 i = 0; i < n; ++i) entries[i].selectionPdf = float(actual[i]);
}

u32 AliasTable::sample(double uniformColumn, double uniformThreshold) const {
    if (entries.empty() || !std::isfinite(uniformColumn) || !std::isfinite(uniformThreshold) ||
        uniformColumn < 0 || uniformColumn >= 1 || uniformThreshold < 0 || uniformThreshold >= 1) return ~0u;
    const u32 column = std::min(u32(uniformColumn * entries.size()), u32(entries.size() - 1));
    const u32 selected = uniformThreshold < entries[column].probability ? column : entries[column].alias;
    return entries[selected].lightIndex;
}

GPUSampledLight fromPunctual(const GPULight& l, u32 id, u32 generation) {
    GPUSampledLight out{};
    out.id = id; out.generation = generation; out.type = l.type;
    out.range = l.range; out.innerCone = l.innerCone; out.outerCone = l.outerCone;
    for (u32 i = 0; i < 3; ++i) {
        out.position[i] = l.position[i]; out.axisU[i] = l.direction[i];
        out.emission[i] = l.color[i] * l.intensity;
    }
    return out;
}

double area(const GPUSampledLight& l) {
    const auto u = v3(l.axisU), v = v3(l.axisV);
    double a = 0;
    if (l.type == DI_LIGHT_RECTANGLE) a = 4 * glm::length(glm::cross(u, v));
    if (l.type == DI_LIGHT_DISK && l.radius > 0) a = pi * l.radius * l.radius * glm::length(glm::cross(u, v));
    if (l.type == DI_LIGHT_TUBE && l.radius > 0) a = 4 * pi * l.radius * glm::length(u);
    if (l.type == DI_LIGHT_TRIANGLE) a = 0.5 * glm::length(glm::cross(u, v));
    return std::isfinite(a) && a > 0 ? a : 0;
}

double powerWeight(const GPUSampledLight& l) {
    const double e = std::max(0.0, luminance(glm::max(v3(l.emission), glm::dvec3(0))));
    if (!std::isfinite(e)) return 0;
    if (l.type == LIGHT_POINT) return 4 * pi * e;
    if (l.type == LIGHT_SPOT) return 2 * pi * (1 - std::cos(l.outerCone)) * e; // proposal heuristic only
    return pi * area(l) * e * ((l.flags & DI_LIGHT_TWO_SIDED) ? 2 : 1);
}

LightSample sampleLight(const GPUSampledLight& l, glm::dvec2 uv, glm::dvec3 receiver) {
    LightSample s;
    if (!std::isfinite(uv.x) || !std::isfinite(uv.y) || uv.x < 0 || uv.x >= 1 || uv.y < 0 || uv.y >= 1 || !finite(receiver)) return s;
    s.position = v3(l.position);
    s.radiance = glm::max(v3(l.emission), glm::dvec3(0));
    const auto u = v3(l.axisU), v = v3(l.axisV);
    s.delta = l.type == LIGHT_POINT || l.type == LIGHT_SPOT;
    if (!s.delta) {
        const double a = area(l);
        if (!(a > 0)) return s;
        s.pdfArea = 1.0 / a;
        if (l.type == DI_LIGHT_RECTANGLE) s.position += (2 * uv.x - 1) * u + (2 * uv.y - 1) * v;
        else if (l.type == DI_LIGHT_DISK) {
            const double r = l.radius * std::sqrt(uv.x), angle = 2 * pi * uv.y;
            s.position += r * (std::cos(angle) * u + std::sin(angle) * v);
        } else if (l.type == DI_LIGHT_TRIANGLE) {
            const double root = std::sqrt(uv.x);
            s.position += root * (1 - uv.y) * u + root * uv.y * v;
        } else if (l.type == DI_LIGHT_TUBE) {
            const auto axis = glm::normalize(u);
            const auto radial = v - axis * glm::dot(axis, v);
            if (!(glm::dot(radial, radial) > 1e-20)) return s;
            const auto x = glm::normalize(radial), y = glm::cross(axis, x);
            const double angle = 2 * pi * uv.y;
            s.normal = std::cos(angle) * x + std::sin(angle) * y;
            s.position += (2 * uv.x - 1) * u + double(l.radius) * s.normal;
        } else return s;
        if (l.type != DI_LIGHT_TUBE) s.normal = glm::normalize(glm::cross(u, v));
    }
    const auto delta = s.position - receiver;
    s.distance = glm::length(delta);
    if (!(s.distance > 1e-10) || !std::isfinite(s.distance)) return {};
    s.wi = delta / s.distance;
    if (s.delta) {
        s.pdfArea = 1; // discrete/delta endpoint; NOT an area density
        s.geometry = attenuation(s.distance, l.range);
        if (l.type == LIGHT_SPOT) {
            if (!(glm::dot(u, u) > 0)) return {};
            const double cosine = glm::dot(glm::normalize(u), -s.wi);
            const double inner = std::cos(l.innerCone), outer = std::cos(l.outerCone);
            const double spot = std::clamp((cosine - outer) / std::max(inner - outer, 1e-4), 0.0, 1.0);
            s.geometry *= spot * spot * (3 - 2 * spot); // identical smoothstep cone to resolve
        }
    } else {
        const double facing = glm::dot(s.normal, -s.wi);
        const double cosine = (l.flags & DI_LIGHT_TWO_SIDED) ? std::abs(facing) : std::max(facing, 0.0);
        s.geometry = cosine / (s.distance * s.distance);
        s.pdfSolidAngle = cosine > 0 ? s.pdfArea * s.distance * s.distance / cosine : 0;
        if (l.range > 0) {
            const double ratio = s.distance / l.range;
            const double window = std::clamp(1 - ratio * ratio * ratio * ratio, 0.0, 1.0);
            s.geometry *= window * window;
        }
    }
    s.valid = finite(s.position) && finite(s.wi) && finite(s.radiance) && std::isfinite(s.geometry) && s.geometry >= 0;
    return s;
}

glm::dvec3 incident(const LightSample& s) { return s.valid ? s.radiance * s.geometry : glm::dvec3(0); }

glm::dvec3 evaluateBRDF(const GPUDISurface& surface, const LightSample& s) {
    if (!surface.valid || !s.valid) return {};
    auto n = v3(surface.shadingNormal), v = v3(surface.viewDirection);
    if (!finite(n) || !finite(v) || !(glm::dot(n, n) > 0 && glm::dot(v, v) > 0)) return {};
    n = glm::normalize(n); v = glm::normalize(v);
    const double nl = std::clamp(glm::dot(n, s.wi), 0.0, 1.0), nv = std::max(glm::dot(n, v), 1e-4);
    const auto hv = v + s.wi;
    if (!(nl > 0) || glm::dot(hv, hv) <= 1e-20) return {};
    const auto h = glm::normalize(hv);
    const double nh = std::clamp(glm::dot(n, h), 0.0, 1.0), vh = std::clamp(glm::dot(v, h), 0.0, 1.0);
    const double metallic = std::clamp(double(surface.metallic), 0.0, 1.0);
    const auto albedo = glm::max(v3(surface.albedo), glm::dvec3(0));
    const double a = std::max(double(surface.roughness) * surface.roughness, 0.002), a2 = a * a;
    const double d = nh * nh * (a2 - 1) + 1;
    const double distribution = a2 / (pi * d * d);
    const double gv = nl * std::sqrt(nv * nv * (1 - a2) + a2), gl = nv * std::sqrt(nl * nl * (1 - a2) + a2);
    const double smith = 0.5 / std::max(gv + gl, 1e-5);
    const auto f0 = glm::mix(glm::dvec3(0.04), albedo, metallic);
    const auto fresnel = f0 + (glm::dvec3(1) - f0) * std::pow(1 - vh, 5.0);
    const auto specular = distribution * smith * fresnel;
    const auto diffuse = (glm::dvec3(1) - fresnel) * (1 - metallic) * albedo / pi;
    const auto result = (diffuse + specular) * incident(s) * nl;
    return finite(result) ? result : glm::dvec3(0);
}

double target(const GPUDISurface& surface, const LightSample& s, double floor) {
    if (!s.valid || !(floor > 0) || !std::isfinite(floor)) return 0;
    return std::max(luminance(evaluateBRDF(surface, s)), floor);
}

glm::dvec3 bruteForce(const GPUDISurface& surface, std::span<const GPUSampledLight> lights,
                     u32 sideSamples, Visibility visibility, void* user) {
    if (sideSamples == 0) return {};
    glm::dvec3 result(0);
    for (const auto& light : lights) {
        if (light.type == LIGHT_DIRECTIONAL) continue;
        const bool delta = light.type == LIGHT_POINT || light.type == LIGHT_SPOT;
        const u32 n = delta ? 1 : sideSamples;
        glm::dvec3 sum(0);
        for (u32 y = 0; y < n; ++y) for (u32 x = 0; x < n; ++x) {
            const auto s = sampleLight(light, {(x + 0.5) / n, (y + 0.5) / n}, v3(surface.position));
            if (!s.valid) continue;
            const double visible = visibility ? std::clamp(visibility(v3(surface.position), s, user), 0.0, 1.0) : 1;
            sum += evaluateBRDF(surface, s) * visible / s.pdfArea;
        }
        result += sum / (double(n) * n);
    }
    return result;
}

void ClusterGrid::rebuild(std::span<const GPUSampledLight> lights, const GPULightClusterParams& config) {
    params = config;
    if (!params.gridX || !params.gridY || !params.gridZ || !params.capacity ||
        !std::isfinite(params.nearPlane) || !std::isfinite(params.farPlane) ||
        !std::isfinite(params.tanHalfFovX) || !std::isfinite(params.tanHalfFovY) ||
        !(params.nearPlane > 0 && params.farPlane > params.nearPlane && params.tanHalfFovX > 0 && params.tanHalfFovY > 0))
        throw std::invalid_argument("invalid light cluster grid");
    const u64 count = u64(params.gridX) * params.gridY * params.gridZ;
    if (count > std::numeric_limits<u32>::max() || count * params.capacity > std::numeric_limits<u32>::max())
        throw std::length_error("light cluster grid overflow");
    if (lights.size() > std::numeric_limits<u32>::max()) throw std::length_error("too many lights");
    params.lightCount = u32(lights.size());
    cells.assign(size_t(count), {});
    indices.assign(size_t(count) * params.capacity, ~0u);
    for (u32 i = 0; i < cells.size(); ++i) {
        for (u32 l = 0; l < lights.size(); ++l) {
            if (lights[l].type == LIGHT_DIRECTIONAL || !clusterIntersects(params, i, lights[l])) continue;
            auto& cell = cells[i];
            if (cell.count < params.capacity) indices[size_t(i) * params.capacity + cell.count] = l;
            else cell.overflow = 1;
            ++cell.count;
        }
    }
}

u32 ClusterGrid::cell(glm::dvec3 position) const {
    const auto v = viewPoint(params, position);
    const double depth = -v.z;
    if (!finite(v) || depth < params.nearPlane || depth > params.farPlane || params.gridX == 0 || params.gridY == 0 || params.gridZ == 0) return ~0u;
    const double nx = v.x / (depth * params.tanHalfFovX) * 0.5 + 0.5, ny = v.y / (depth * params.tanHalfFovY) * 0.5 + 0.5;
    if (nx < 0 || nx > 1 || ny < 0 || ny > 1) return ~0u;
    const u32 x = std::min(u32(nx * params.gridX), params.gridX - 1), y = std::min(u32(ny * params.gridY), params.gridY - 1);
    const double slice = std::log(depth / params.nearPlane) / std::log(double(params.farPlane) / params.nearPlane);
    const u32 z = std::min(u32(slice * params.gridZ), params.gridZ - 1);
    return (z * params.gridY + y) * params.gridX + x;
}

bool ClusterGrid::requiresBruteForce(u32 i) const { return i >= cells.size() || cells[i].overflow != 0; }
std::span<const u32> ClusterGrid::list(u32 i) const {
    if (i >= cells.size() || cells[i].overflow) return {};
    return {indices.data() + size_t(i) * params.capacity, cells[i].count};
}
} // namespace phosphor::di
