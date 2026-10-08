#pragma once

#include "renderer/gpu_types.h"
#include <glm/glm.hpp>
#include <span>
#include <vector>

namespace phosphor {

// These numbers are experimental presets, not measured quality/cost decisions.
struct ProbeGridConfig {
    glm::vec3 origin{-8, 0, -8}, spacing{2};
    glm::uvec3 counts{8, 4, 8};
    u32 raysPerProbe = 64, irradianceTexels = 6, distanceTexels = 14;
    float maxDistance = 100, hysteresis = 0.95f, normalBias = 0.02f;
    float backfaceThreshold = 0.25f, minFrontDistance = 0.1f;
    float relocationStep = 0.1f, maxRelocation = 0.45f; // fraction of spacing
};
[[nodiscard]] bool validProbeGrid(const ProbeGridConfig& c);
[[nodiscard]] glm::vec2 probeOctEncode(glm::vec3 direction);
[[nodiscard]] glm::vec3 probeOctDecode(glm::vec2 uv);
[[nodiscard]] glm::vec3 probeRayDirection(u32 ray, u32 count, u32 frame);
[[nodiscard]] float probeVisibility(float distance, float mean, float secondMoment);

struct ProbeTexel { glm::vec3 irradiance{}; glm::vec2 moments{}; };
struct ProbeUpdate { bool active = false, relocated = false; float backfaceFraction = 0; };

// CPU reference for atlas estimator and probe lifecycle. No graphics API and no
// GPU allocations. The backend owns atlases in GpuMemory and declares graph IO.
class ProbeGrid {
public:
    explicit ProbeGrid(ProbeGridConfig config = {});
    void reset(u32 generation);
    // Light/material/camera radiometric invalidation keeps learned geometric
    // offsets/classification. Topology/geometry reset() still clears everything.
    void invalidateRadiance();
    [[nodiscard]] u32 probeCount() const;
    [[nodiscard]] glm::vec3 position(u32 probe) const;
    [[nodiscard]] const ProbeGridConfig& config() const { return config_; }
    [[nodiscard]] const std::vector<GPUProbeState>& states() const { return states_; }
    [[nodiscard]] GPUProbeGridParams parameters(u32 frame, u32 generation) const;
    // Trace rays must correspond to the position BEFORE relocation. Relocation
    // clears this probe's atlas and requires a new trace before it can be used.
    ProbeUpdate update(u32 probe, std::span<const GPUProbeRay> rays);
    [[nodiscard]] glm::vec3 irradiance(glm::vec3 point, glm::vec3 geometricNormal) const;
    [[nodiscard]] ProbeTexel texel(u32 probe, glm::vec3 direction) const;

private:
    [[nodiscard]] glm::vec3 sampleIrradiance(u32 probe, glm::vec3 normal) const;
    [[nodiscard]] glm::vec2 sampleMoments(u32 probe, glm::vec3 direction) const;
    ProbeGridConfig config_;
    std::vector<GPUProbeState> states_;
    std::vector<glm::vec3> irradiance_;
    std::vector<glm::vec2> moments_;
};

} // namespace phosphor
