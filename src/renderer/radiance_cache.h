#pragma once

#include "renderer/gpu_types.h"
#include <glm/glm.hpp>
#include <optional>
#include <vector>

namespace phosphor {
struct RadianceCacheConfig {
    u32 capacity = 16384, probeLimit = 8, maxAge = 60, maxSamples = 64;
    float cellSize = 0.5f;
};
struct RadianceRevisions { u32 geometry = 0, lights = 0, materials = 0; };
struct RadianceCacheKey {
    i32 x = 0, y = 0, z = 0;
    u32 normal = 0, direction = 0;
    bool operator==(const RadianceCacheKey&) const = default;
};
[[nodiscard]] std::optional<RadianceCacheKey> radianceCacheKey(glm::vec3 point, glm::vec3 normal,
                                                             glm::vec3 outgoing, float cellSize);
[[nodiscard]] u32 radianceCacheHash(const RadianceCacheKey& key);

// The cache stores outgoing RADIANCE for a directional bin at a surface cell.
// It NEVER stores irradiance disguised as radiance. A miss is explicit: the
// caller computes diffuse outgoing radiance from DDGI irradiance * albedo/pi.
// Open addressing probes a bounded window, compares the entire key and replaces
// the oldest lane in that window when full. Readers cannot observe partial data.
// CPU writes are single-threaded; GPU update uses one owner per table lane.
class RadianceCache {
public:
    explicit RadianceCache(RadianceCacheConfig config = {});
    void reset(u32 generation, RadianceRevisions revisions);
    [[nodiscard]] std::optional<glm::vec3> lookup(const RadianceCacheKey& key, u32 frame) const;
    [[nodiscard]] bool insert(const RadianceCacheKey& key, glm::vec3 radiance, u32 frame);
    [[nodiscard]] u32 size() const;
    [[nodiscard]] u32 evictions() const { return evictions_; }
    [[nodiscard]] const std::vector<GPURadianceCacheEntry>& entries() const { return entries_; }
    [[nodiscard]] const RadianceCacheConfig& config() const { return config_; }
private:
    [[nodiscard]] bool current(const GPURadianceCacheEntry& e, u32 frame) const;
    [[nodiscard]] bool matches(const GPURadianceCacheEntry& e, const RadianceCacheKey& key) const;
    RadianceCacheConfig config_;
    std::vector<GPURadianceCacheEntry> entries_;
    RadianceRevisions revisions_{};
    u32 generation_ = 1, evictions_ = 0;
};
} // namespace phosphor
