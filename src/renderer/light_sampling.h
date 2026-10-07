#pragma once

#include "renderer/gpu_types.h"
#include <glm/glm.hpp>
#include <span>
#include <vector>

namespace phosphor::di {

struct AliasTable {
    std::vector<GPUAliasEntry> entries;
    u32 revision = 0;
    // Invalid/negative/nonfinite weights are zero. All zero -> uniform.
    // Zero entries receive a tiny full-support mixture when others are lit.
    void rebuild(std::span<const double> weights, u32 newRevision);
    [[nodiscard]] u32 sample(double uniformColumn, double uniformThreshold) const;
};

struct LightSample {
    glm::dvec3 position{}, normal{}, wi{}, radiance{};
    double distance = 0, pdfArea = 0, pdfSolidAngle = 0, geometry = 0;
    bool delta = false, valid = false;
};

[[nodiscard]] double area(const GPUSampledLight& light);
[[nodiscard]] double powerWeight(const GPUSampledLight& light);
[[nodiscard]] GPUSampledLight fromPunctual(const GPULight& light, u32 id, u32 generation);
[[nodiscard]] LightSample sampleLight(const GPUSampledLight& light, glm::dvec2 uv, glm::dvec3 receiver);
[[nodiscard]] glm::dvec3 incident(const LightSample& sample);
[[nodiscard]] glm::dvec3 evaluateBRDF(const GPUDISurface& surface, const LightSample& sample);
[[nodiscard]] double target(const GPUDISurface& surface, const LightSample& sample, double positiveFloor);
// Per-light endpoint integration, uniform area sampling, independent of alias/RIS.
// Punctual lights are evaluated once. Visibility is optional and uncached.
using Visibility = double (*)(glm::dvec3 receiver, const LightSample&, void* user);
[[nodiscard]] glm::dvec3 bruteForce(const GPUDISurface& surface, std::span<const GPUSampledLight> lights,
                                  u32 sideSamples, Visibility visibility = nullptr, void* user = nullptr);

struct ClusterGrid {
    GPULightClusterParams params{};
    std::vector<GPULightCluster> cells;
    std::vector<u32> indices;
    void rebuild(std::span<const GPUSampledLight> lights, const GPULightClusterParams& config);
    [[nodiscard]] u32 cell(glm::dvec3 worldPosition) const;
    // A full cell must use all active lights: truncated lists lose energy.
    [[nodiscard]] bool requiresBruteForce(u32 cellIndex) const;
    [[nodiscard]] std::span<const u32> list(u32 cellIndex) const;
};

struct Preset {
    u32 candidates, neighbors, radius, maxHistoryM, maxHistoryAge;
    u32 clusterX, clusterY, clusterZ, clusterCapacity;
};
// Experimental, unmeasured presets. Both require effective Apple9 only.
inline constexpr Preset fullPreset{8, 4, 16, 64, 16, 16, 9, 24, 128};
inline constexpr Preset reducedPreset{2, 1, 8, 16, 8, 16, 9, 16, 32};

} // namespace phosphor::di
