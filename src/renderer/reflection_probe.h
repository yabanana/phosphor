#pragma once
#include "renderer/gpu_types.h"
#include <glm/glm.hpp>
#include <optional>
namespace phosphor {
[[nodiscard]] bool validReflectionProbe(const GPUReflectionProbe&);
[[nodiscard]] float reflectionProbeWeight(const GPUReflectionProbe&,glm::vec3 point);
[[nodiscard]] std::optional<glm::vec3> reflectionProbeParallax(const GPUReflectionProbe&,glm::vec3 point,glm::vec3 direction);
[[nodiscard]] glm::vec3 reflectionCubeDirection(u32 face,glm::vec2 uv);
struct CubeCoordinate { u32 face; glm::vec2 uv; };
[[nodiscard]] std::optional<CubeCoordinate> reflectionCubeCoordinate(glm::vec3 direction);
[[nodiscard]] glm::mat4 reflectionProbeViewProjection(const GPUReflectionProbe&,u32 face,float nearPlane,float farPlane);
using ProbeRadiance = glm::vec3 (*)(glm::vec3 direction,void* user);
// GGX convolution divided by sum(NdotL), preserving constant radiance at
// every roughness. Capture and mip scheduling/lifetimes belong to the host.
[[nodiscard]] glm::vec3 reflectionProbePrefilter(glm::vec3 direction,float roughness,u32 samples,ProbeRadiance,void* user=nullptr);
} // namespace phosphor
