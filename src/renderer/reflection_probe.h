#pragma once
#include "renderer/gpu_types.h"
#include <glm/glm.hpp>
#include <optional>
#include <span>
#include <vector>
namespace phosphor {
// Static capture eligibility requires every ancestor to be valid, visible,
// declared static and without procedural motion. Child depth alone is not a
// reason to exclude geometry. Output is ascending slot order and reuses capacity.
void reflectionProbeStaticSlots(std::span<const GPUInstance>,std::span<const GPUTransformNode>,
                                std::span<const u32> motionSlots,std::vector<u32>& output);
struct ReflectionProbeBounds {glm::vec3 minimum,maximum;};
// Matrices MUST be resolved WORLD matrices for the same capture frame, never
// SceneStore's identity placeholders. Empty selection means all valid visible
// meshes. A caller with an intentionally empty static set should use nullopt.
[[nodiscard]] std::optional<ReflectionProbeBounds> reflectionProbeWorldBounds(
    std::span<const GPUInstance>,std::span<const GPUMeshInfo>,std::span<const float> worldMatrices,
    std::span<const u32> selection={});
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
[[nodiscard]] glm::vec3 reflectionProbeContribution(const GPUDISurface&,glm::vec3 prefilteredRadiance);
} // namespace phosphor
