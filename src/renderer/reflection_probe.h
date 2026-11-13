#pragma once
#include "renderer/gpu_types.h"
#include "renderer/mesh_bounds.h"
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
// The result is the union of the world AABBs of each mesh's local AABB (its
// eight transformed corners): exact for axis-aligned walls, and the box the
// probe projects onto (F13.5), so it must not inflate to culling spheres.
[[nodiscard]] std::optional<ReflectionProbeBounds> reflectionProbeWorldBounds(
    std::span<const GPUInstance>,std::span<const MeshLocalBounds>,std::span<const float> worldMatrices,
    std::span<const u32> selection={});
[[nodiscard]] bool validReflectionProbe(const GPUReflectionProbe&);
// F13.5: 1 inside the projection box (receivers on its faces included),
// fading linearly to 0 at blendDistance outside it, where the declared
// environment takes over. Shader mirror: reflectionProbeDirection.
[[nodiscard]] float reflectionProbeWeight(const GPUReflectionProbe&,glm::vec3 point);
[[nodiscard]] std::optional<glm::vec3> reflectionProbeParallax(const GPUReflectionProbe&,glm::vec3 point,glm::vec3 direction);
// F13.5: direction a prefiltered probe is read along. The prefilter assumes
// n = v = r; on rough receivers the GGX lobe's dominant direction bends from r
// toward the normal (Frostbite 2014, 4.9.3), keeping a wide lobe above the
// receiver's horizon. Shader mirror: reflection_probe_only.
[[nodiscard]] glm::vec3 reflectionProbeDominantDirection(glm::vec3 normal,glm::vec3 reflected,float roughness);
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
