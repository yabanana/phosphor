#pragma once
#include "renderer/gpu_types.h"
#include <glm/glm.hpp>
#include <optional>
#include <span>
namespace phosphor {
struct ReflectionSettings {
    float maxDistance = 100, ssrThickness = 0.1f, rtRoughnessLow = 0.2f, rtRoughnessHigh = 0.4f;
    u32 ssrSteps = 64, ssrBinarySteps = 5, flags = REFLECTION_ENABLE_RT|REFLECTION_ENABLE_SSR|REFLECTION_ENABLE_CACHE|REFLECTION_ENABLE_PROBES;
};
[[nodiscard]] bool validReflectionSettings(const ReflectionSettings&);
[[nodiscard]] float reflectionRTWeight(float roughness, const ReflectionSettings&);
struct SpecularDirection { glm::dvec3 direction{}, weight{}; double pdf = 0; bool valid = false; };
// GGX NDF importance sampling, not a falsely labelled VNDF. Rejected reflected
// hemispheres contribute black proposals, never resample until accepted.
[[nodiscard]] SpecularDirection sampleSpecular(const GPUDISurface&, glm::dvec2 uniform);
[[nodiscard]] glm::dvec3 specularBRDF(const GPUDISurface&, glm::dvec3 incoming);
using IncidentRadiance = glm::dvec3 (*)(glm::dvec3 direction, void* user);
[[nodiscard]] glm::dvec3 reflectionReference(const GPUDISurface&, u32 azimuthSteps, u32 elevationSteps,
                                          IncidentRadiance, void* user = nullptr);
struct SSRHit { u32 pixel = ~0u; float distance = 0; glm::vec3 position{}; };
[[nodiscard]] std::optional<SSRHit> reflectionSSR(glm::vec3 point, glm::vec3 direction,
    const glm::mat4& vp, const glm::mat4& inverseVP, const glm::mat4& view,
    u32 width, u32 height, std::span<const float> reverseDepth, const ReflectionSettings&);
struct AOSettings { float radius = 1, originBias = 0.001f, thickness = 0.1f; u32 rays = 4, slices = 4, steps = 8; };
[[nodiscard]] bool validAOSettings(const AOSettings&);
[[nodiscard]] glm::vec3 aoDirection(glm::vec3 normal, glm::vec2 uniform);
using Occluded = bool (*)(glm::vec3 direction, float radiusMetres, void* user);
[[nodiscard]] float aoReference(glm::vec3 normal, float radiusMetres, u32 sideSamples, Occluded, void* user = nullptr);
[[nodiscard]] float aoHorizonVisibility(glm::vec3 normal, glm::vec3 view, glm::vec3 sliceTangent,
    float positiveHorizon, float negativeHorizon, u32 angularSteps = 64);
// AO affects only residual ambient diffuse, never the already transported GI,
// direct light, emission or specular term.
[[nodiscard]] glm::vec3 composeSignalLighting(glm::vec3 direct, glm::vec3 gi, glm::vec3 specular,
    glm::vec3 emission, glm::vec3 ambientDiffuse, float aoVisibility, bool giEnabled);
} // namespace phosphor
