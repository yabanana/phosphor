#include "renderer/radiance_cache.h"
#include "renderer/probe_grid.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace phosphor {
namespace {
u32 mix(u32 x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; return x ^ (x >> 16); }
bool finite(glm::vec3 x) { return std::isfinite(x.x) && std::isfinite(x.y) && std::isfinite(x.z); }
u32 bin(glm::vec3 d) {
    const auto uv = glm::clamp(probeOctEncode(glm::normalize(d))*0.5f+0.5f,glm::vec2(0),glm::vec2(0.999999f));
    const glm::uvec2 b(uv*16.f);
    return b.x | (b.y<<4);
}
}
std::optional<RadianceCacheKey> radianceCacheKey(glm::vec3 p, glm::vec3 n, glm::vec3 d, float cellSize) {
    if (!finite(p) || !finite(n) || !finite(d) || !(std::isfinite(cellSize) && cellSize > 0) ||
        glm::dot(n,n) <= 1e-20f || glm::dot(d,d) <= 1e-20f) return std::nullopt;
    const glm::dvec3 cell = glm::floor(glm::dvec3(p)/double(cellSize));
    if (glm::any(glm::lessThan(cell,glm::dvec3(std::numeric_limits<i32>::min()))) ||
        glm::any(glm::greaterThan(cell,glm::dvec3(std::numeric_limits<i32>::max())))) return std::nullopt;
    return RadianceCacheKey{i32(cell.x),i32(cell.y),i32(cell.z),bin(n),bin(d)};
}
u32 radianceCacheHash(const RadianceCacheKey& k) {
    return mix(std::bit_cast<u32>(k.x)) ^ mix(std::bit_cast<u32>(k.y)+0x9e3779b9u) ^
        mix(std::bit_cast<u32>(k.z)+0x85ebca6bu) ^ mix(k.normal+0xc2b2ae35u) ^ mix(k.direction+0x27d4eb2fu);
}
RadianceCache::RadianceCache(RadianceCacheConfig c) : config_(c) {
    if (c.capacity < 1 || c.capacity > (1u<<24) || c.probeLimit < 1 || c.probeLimit > c.capacity ||
        c.maxAge >= 0x80000000u || c.maxSamples < 1 || !std::isfinite(c.cellSize) || c.cellSize <= 0)
        throw std::invalid_argument("invalid radiance cache");
    entries_.resize(c.capacity);
}
void RadianceCache::reset(u32 generation, RadianceRevisions revisions) {
    generation_ = generation; revisions_ = revisions;
    // Clear even on generation wrap or repeated generation; no stale value can
    // become current merely because a 32-bit epoch cycles.
    std::fill(entries_.begin(),entries_.end(),GPURadianceCacheEntry{}); evictions_ = 0;
}
bool RadianceCache::current(const GPURadianceCacheEntry& e, u32 f) const {
    return e.state == 1 && e.samples && e.generation == generation_ &&
        e.geometryRevision == revisions_.geometry && e.lightRevision == revisions_.lights &&
        e.materialRevision == revisions_.materials && u32(f-e.lastFrame) <= config_.maxAge;
}
bool RadianceCache::matches(const GPURadianceCacheEntry& e, const RadianceCacheKey& k) const {
    return e.cellX == std::bit_cast<u32>(k.x) && e.cellY == std::bit_cast<u32>(k.y) &&
        e.cellZ == std::bit_cast<u32>(k.z) && e.normalBin == k.normal && e.directionBin == k.direction;
}
std::optional<glm::vec3> RadianceCache::lookup(const RadianceCacheKey& k, u32 f) const {
    const u32 start = radianceCacheHash(k)%config_.capacity;
    for (u32 step=0; step<config_.probeLimit; ++step) {
        const auto& e = entries_[(start+step)%config_.capacity];
        if (current(e,f) && matches(e,k)) return glm::vec3(e.radiance[0],e.radiance[1],e.radiance[2]);
    }
    return std::nullopt;
}
bool RadianceCache::insert(const RadianceCacheKey& k, glm::vec3 L, u32 f) {
    if (!finite(L) || glm::any(glm::lessThan(L,glm::vec3(0)))) return false;
    const u32 start = radianceCacheHash(k)%config_.capacity;
    u32 victim=start, oldestAge=0;
    std::optional<u32> stale;
    for (u32 step=0; step<config_.probeLimit; ++step) {
        const u32 lane=(start+step)%config_.capacity;
        auto& e=entries_[lane];
        if (!current(e,f)) { if (!stale) stale=lane; continue; }
        if (matches(e,k)) {
            const float weight=1.f/float(std::min(e.samples,config_.maxSamples-1)+1);
            const glm::vec3 old(e.radiance[0],e.radiance[1],e.radiance[2]);
            const auto result=glm::mix(old,L,weight);
            for(int i=0;i<3;++i) e.radiance[i]=result[i];
            e.samples=std::min(e.samples+1,config_.maxSamples); e.lastFrame=f;
            return true;
        }
        const u32 age=f-e.lastFrame;
        if (step==0 || age>oldestAge) { oldestAge=age; victim=lane; }
    }
    if (stale) victim=*stale; else ++evictions_;
    auto& e=entries_[victim]; e={};
    e.cellX=std::bit_cast<u32>(k.x); e.cellY=std::bit_cast<u32>(k.y); e.cellZ=std::bit_cast<u32>(k.z);
    e.normalBin=k.normal; e.directionBin=k.direction; e.generation=generation_; e.lastFrame=f;
    e.samples=1; for(int i=0;i<3;++i) e.radiance[i]=L[i];
    e.geometryRevision=revisions_.geometry; e.lightRevision=revisions_.lights; e.materialRevision=revisions_.materials;
    e.state=1;
    return true;
}
u32 RadianceCache::size() const {
    return u32(std::count_if(entries_.begin(),entries_.end(),[&](const auto& e) {
        return e.state==1 && e.generation==generation_ && e.samples;
    }));
}
} // namespace phosphor
