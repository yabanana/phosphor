#include "renderer/probe_grid.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace phosphor {
namespace {
constexpr float pi = 3.14159265358979323846f;
bool finite(glm::vec3 v) { return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z); }
float signNonzero(float v) { return v < 0 ? -1.f : 1.f; }
glm::vec3 rayDirection(const GPUProbeRay& r) { return {r.direction[0], r.direction[1], r.direction[2]}; }
glm::vec3 rayRadiance(const GPUProbeRay& r) { return {r.radiance[0], r.radiance[1], r.radiance[2]}; }
template<class T>
T atlasSample(const std::vector<T>& atlas, u32 probe, u32 side, glm::vec3 direction) {
    const glm::vec2 xy = (probeOctEncode(direction) * 0.5f + 0.5f) * float(side) - 0.5f;
    const glm::ivec2 lo(glm::floor(xy));
    const glm::vec2 f = glm::fract(xy);
    // Octahedral seams: nearest interior value at an edge, same as explicit
    // border texels in the GPU implementation (seam folding occurs below).
    auto get = [&](int x, int y) {
        if (x < 0) { x = -x - 1; y = int(side) - 1 - y; }
        if (x >= int(side)) { x = 2 * int(side) - 1 - x; y = int(side) - 1 - y; }
        if (y < 0) { y = -y - 1; x = int(side) - 1 - x; }
        if (y >= int(side)) { y = 2 * int(side) - 1 - y; x = int(side) - 1 - x; }
        x = std::clamp(x, 0, int(side) - 1); y = std::clamp(y, 0, int(side) - 1);
        return atlas[(size_t(probe) * side + u32(y)) * side + u32(x)];
    };
    return glm::mix(glm::mix(get(lo.x, lo.y), get(lo.x + 1, lo.y), f.x),
                    glm::mix(get(lo.x, lo.y + 1), get(lo.x + 1, lo.y + 1), f.x), f.y);
}
}

bool validProbeGrid(const ProbeGridConfig& c) {
    const uint64_t probes = uint64_t(c.counts.x) * c.counts.y * c.counts.z;
    return finite(c.origin) && finite(c.spacing) && glm::all(glm::greaterThan(c.spacing, glm::vec3(0))) &&
        glm::all(glm::greaterThanEqual(c.counts, glm::uvec3(2))) &&
        glm::all(glm::lessThanEqual(c.counts, glm::uvec3(256))) && probes <= 65536 &&
        uint64_t(c.counts.x)*(std::max(c.irradianceTexels,c.distanceTexels)+2u)<=16384 &&
        uint64_t(c.counts.y)*c.counts.z*(std::max(c.irradianceTexels,c.distanceTexels)+2u)<=16384 &&
        c.raysPerProbe >= 8 && c.raysPerProbe <= 4096 && c.irradianceTexels >= 2 &&
        c.irradianceTexels <= 32 && c.distanceTexels >= 2 && c.distanceTexels <= 32 &&
        std::isfinite(c.maxDistance) && c.maxDistance > 0 && c.maxDistance < 1e6f &&
        std::isfinite(c.hysteresis) && c.hysteresis >= 0 && c.hysteresis < 1 &&
        std::isfinite(c.normalBias) && c.normalBias >= 0 && c.normalBias < std::min({c.spacing.x,c.spacing.y,c.spacing.z}) &&
        std::isfinite(c.backfaceThreshold) && c.backfaceThreshold >= 0 && c.backfaceThreshold < 1 &&
        std::isfinite(c.minFrontDistance) && c.minFrontDistance >= 0 &&
        std::isfinite(c.relocationStep) && c.relocationStep > 0 &&
        std::isfinite(c.maxRelocation) && c.maxRelocation >= 0 && c.maxRelocation <= 0.49f;
}
glm::vec2 probeOctEncode(glm::vec3 d) {
    if (!finite(d) || glm::dot(d, d) <= 1e-20f) return {};
    d /= std::abs(d.x) + std::abs(d.y) + std::abs(d.z);
    glm::vec2 uv(d.x, d.y);
    if (d.z < 0) uv = {(1 - std::abs(d.y)) * signNonzero(d.x), (1 - std::abs(d.x)) * signNonzero(d.y)};
    return uv;
}
glm::vec3 probeOctDecode(glm::vec2 uv) {
    glm::vec3 n(uv.x, uv.y, 1 - std::abs(uv.x) - std::abs(uv.y));
    if (n.z < 0) {
        const float x = n.x;
        n.x = (1 - std::abs(n.y)) * signNonzero(x);
        n.y = (1 - std::abs(x)) * signNonzero(n.y);
    }
    return glm::normalize(n);
}
glm::vec3 probeRayDirection(u32 ray, u32 count, u32 frame) {
    if (!count || ray >= count) return {};
    // Spherical Fibonacci equal-area quadrature, uniformly rotated per frame.
    // Generated locally; no downloaded mask and no claim of STBN equivalence.
    const float z = 1 - 2 * (float(ray) + 0.5f) / float(count);
    const float angle = 2 * pi * glm::fract(float(ray) * 0.61803398875f + float(frame % 4096) * 0.754877666f);
    const float r = std::sqrt(std::max(0.f, 1 - z * z));
    return {r * std::cos(angle), r * std::sin(angle), z};
}
float probeVisibility(float distance, float mean, float secondMoment) {
    if (!(std::isfinite(distance) && std::isfinite(mean) && std::isfinite(secondMoment)) || distance < 0 || mean < 0)
        return 0;
    if (distance <= mean) return 1;
    const float variance = std::max(0.f, secondMoment - mean * mean);
    const float delta = distance - mean;
    const float bound = variance / (variance + delta * delta + 1e-12f);
    return bound * bound * bound; // explicit leak-reduction bias, not exact visibility
}
ProbeGrid::ProbeGrid(ProbeGridConfig config) : config_(config) {
    if (!validProbeGrid(config_)) throw std::invalid_argument("invalid DDGI grid");
    states_.resize(size_t(config_.counts.x) * config_.counts.y * config_.counts.z);
    irradiance_.resize(states_.size() * config_.irradianceTexels * config_.irradianceTexels);
    moments_.resize(states_.size() * config_.distanceTexels * config_.distanceTexels);
    reset(1);
}
void ProbeGrid::reset(u32 generation) {
    for (auto& s : states_) { s = {}; s.generation = generation; }
    std::fill(irradiance_.begin(), irradiance_.end(), glm::vec3(0));
    std::fill(moments_.begin(), moments_.end(), glm::vec2(0));
}
void ProbeGrid::invalidateRadiance() {
    for(auto& s:states_)s.age=0;
    std::fill(irradiance_.begin(),irradiance_.end(),glm::vec3(0));
    std::fill(moments_.begin(),moments_.end(),glm::vec2(0));
}
u32 ProbeGrid::probeCount() const { return u32(states_.size()); }
glm::vec3 ProbeGrid::position(u32 p) const {
    if (p >= states_.size()) throw std::out_of_range("DDGI probe index");
    const glm::uvec3 cell{p % config_.counts.x, (p / config_.counts.x) % config_.counts.y,
                           p / (config_.counts.x * config_.counts.y)};
    return config_.origin + config_.spacing * glm::vec3(cell) +
        glm::vec3(states_[p].offset[0], states_[p].offset[1], states_[p].offset[2]);
}
GPUProbeGridParams ProbeGrid::parameters(u32 frame, u32 generation) const {
    GPUProbeGridParams p{};
    for (int i = 0; i < 3; ++i) { p.origin[i] = config_.origin[i]; p.spacing[i] = config_.spacing[i]; }
    p.maxDistance = config_.maxDistance; p.hysteresis = config_.hysteresis;
    p.countX = config_.counts.x; p.countY = config_.counts.y; p.countZ = config_.counts.z;
    p.raysPerProbe = config_.raysPerProbe; p.irradianceTexels = config_.irradianceTexels;
    p.distanceTexels = config_.distanceTexels; p.frameIndex = frame; p.generation = generation;
    p.normalBias = config_.normalBias; p.backfaceThreshold = config_.backfaceThreshold;
    p.minFrontDistance = config_.minFrontDistance; p.relocationStep = config_.relocationStep;
    p.maxRelocation = config_.maxRelocation; p.mode = GI_MODE_DDGI;
    return p;
}
ProbeUpdate ProbeGrid::update(u32 probe, std::span<const GPUProbeRay> rays) {
    if (probe >= states_.size()) throw std::out_of_range("DDGI probe index");
    if (rays.size() != config_.raysPerProbe) throw std::invalid_argument("DDGI ray count");
    u32 backfaces = 0;
    float closestBack = config_.maxDistance, closestFront = config_.maxDistance;
    glm::vec3 escape(0), away(0);
    for (const auto& r : rays) {
        const auto d = rayDirection(r), L = rayRadiance(r);
        if (!finite(d) || std::abs(glm::dot(d,d)-1) > 1e-3f || !finite(L) ||
            glm::any(glm::lessThan(L, glm::vec3(0))) || !std::isfinite(r.distance) ||
            std::abs(r.distance) > config_.maxDistance || (r.backface != 0) != (r.distance < 0))
            throw std::invalid_argument("invalid DDGI ray payload");
        if (r.backface) { ++backfaces; if (-r.distance < closestBack) { closestBack = -r.distance; escape = d; } }
        else if (r.distance < closestFront) { closestFront = r.distance; away = -d; }
    }
    auto& state = states_[probe];
    ProbeUpdate result;
    result.backfaceFraction = float(backfaces) / float(rays.size());
    const bool inside = result.backfaceFraction > config_.backfaceThreshold;
    const glm::vec3 old(state.offset[0], state.offset[1], state.offset[2]);
    glm::vec3 delta(0);
    if (inside && closestBack < config_.maxDistance)
        delta = escape * std::min(config_.relocationStep, closestBack + config_.minFrontDistance);
    else if (closestFront < config_.minFrontDistance)
        delta = away * std::min(config_.relocationStep, config_.minFrontDistance - closestFront);
    glm::vec3 next = old + delta;
    const glm::vec3 normalized = next / config_.spacing;
    const float length = glm::length(normalized);
    if (length > config_.maxRelocation && length > 0) next *= config_.maxRelocation / length;
    result.relocated = glm::length(next - old) > 1e-6f;
    state.relocationTravel += glm::length(next - old);
    for (int i = 0; i < 3; ++i) state.offset[i] = next[i];
    state.state = inside || result.relocated ? GI_PROBE_INACTIVE : GI_PROBE_ACTIVE;
    result.active = state.state == GI_PROBE_ACTIVE;
    if (!result.active) state.age = 0;
    const float history = state.age ? config_.hysteresis : 0;
    for (u32 y = 0; y < config_.irradianceTexels; ++y) for (u32 x = 0; x < config_.irradianceTexels; ++x) {
        const auto n = probeOctDecode((glm::vec2(x,y) + 0.5f) / float(config_.irradianceTexels) * 2.f - 1.f);
        glm::vec3 sum(0);
        if (result.active) for (const auto& r : rays) if (!r.backface)
            sum += rayRadiance(r) * std::max(0.f, glm::dot(n, rayDirection(r)));
        const auto index = (size_t(probe) * config_.irradianceTexels + y) * config_.irradianceTexels + x;
        irradiance_[index] = glm::mix(sum * (4 * pi / float(rays.size())), irradiance_[index], history);
    }
    for (u32 y = 0; y < config_.distanceTexels; ++y) for (u32 x = 0; x < config_.distanceTexels; ++x) {
        const auto n = probeOctDecode((glm::vec2(x,y) + 0.5f) / float(config_.distanceTexels) * 2.f - 1.f);
        glm::vec2 sum(0); float weights = 0;
        if (result.active) for (const auto& r : rays) {
            const float w = std::pow(std::max(0.f, glm::dot(n, rayDirection(r))), 50.f);
            const float d = std::max(0.f, r.distance);
            sum += glm::vec2(d, d*d) * w; weights += w;
        }
        const auto index = (size_t(probe) * config_.distanceTexels + y) * config_.distanceTexels + x;
        moments_[index] = glm::mix(weights > 1e-20f ? sum / weights : glm::vec2(0), moments_[index], history);
    }
    if (result.active && state.age != ~0u) ++state.age;
    return result;
}
glm::vec3 ProbeGrid::sampleIrradiance(u32 p, glm::vec3 n) const { return atlasSample(irradiance_,p,config_.irradianceTexels,n); }
glm::vec2 ProbeGrid::sampleMoments(u32 p, glm::vec3 n) const { return atlasSample(moments_,p,config_.distanceTexels,n); }
ProbeTexel ProbeGrid::texel(u32 p, glm::vec3 n) const {
    if (p >= states_.size()) throw std::out_of_range("DDGI probe index");
    return {sampleIrradiance(p,n),sampleMoments(p,n)};
}
glm::vec3 ProbeGrid::irradiance(glm::vec3 point, glm::vec3 normal) const {
    if (!finite(point) || !finite(normal) || glm::dot(normal,normal) < 1e-20f) return {};
    normal = glm::normalize(normal);
    const glm::vec3 biased = point + normal * config_.normalBias;
    const glm::vec3 coordinate = (biased-config_.origin)/config_.spacing;
    if (glm::any(glm::lessThan(coordinate,glm::vec3(0))) ||
        glm::any(glm::greaterThan(coordinate,glm::vec3(config_.counts)-1.f))) return {};
    const glm::ivec3 base = glm::min(glm::ivec3(glm::floor(coordinate)),glm::ivec3(config_.counts)-2);
    const glm::vec3 f = coordinate - glm::vec3(base);
    glm::vec3 result(0); float weightSum = 0;
    for (u32 corner = 0; corner < 8; ++corner) {
        const glm::ivec3 bit{int(corner&1),int((corner>>1)&1),int((corner>>2)&1)};
        const glm::ivec3 c = base+bit;
        const u32 p = u32((c.z * int(config_.counts.y)+c.y)*int(config_.counts.x)+c.x);
        if (states_[p].state != GI_PROBE_ACTIVE || !states_[p].age) continue;
        const glm::vec3 toPoint = biased-position(p);
        const float distance = glm::length(toPoint);
        const glm::vec3 direction = distance > 1e-8f ? toPoint/distance : normal;
        const auto m = sampleMoments(p,direction);
        const glm::vec3 tri = glm::mix(1.f-f,f,glm::vec3(bit));
        const float wrap = std::max(0.05f, (glm::dot(normal,-direction)+1.f)*0.5f);
        const float w = tri.x*tri.y*tri.z * wrap*wrap * probeVisibility(distance,m.x,m.y);
        result += sampleIrradiance(p,normal)*w; weightSum += w;
    }
    return weightSum > 1e-8f ? result/weightSum : glm::vec3(0);
}
} // namespace phosphor
