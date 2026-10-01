#pragma once

// ---------------------------------------------------------------------------
// F5.5 / F5.3 -- portable references of the GPU cull + compaction + draw
// build kernels (shaders/gpu_scene.metal).  They define the expected output
// exactly; the unit tests check them against hand-made cases, and
// bench/f5_spike/k2_gpu_scene.cpp compares the real kernels with them.
// ---------------------------------------------------------------------------

#include "core/types.h"
#include "renderer/cull_math.h"
#include "renderer/gpu_types.h"

#include <glm/glm.hpp>

#include <array>
#include <span>
#include <vector>

namespace phosphor {

/// Result value of an invalid slot (INSTANCE_FLAG_VALID clear): never tested,
/// never visible.
inline constexpr u8 CULL_RESULT_INVALID = 255;

/// The 5 normalised planes (xyz normal pointing inwards, w distance) of the
/// reverse-Z infinite frustum of `viewProj` (Metal clip volume, depth 1 at
/// the near plane and 0 at infinity): left, right, bottom, top, near.  There
/// is no far plane.  Inside: dot(n, p) + w >= 0.
std::array<glm::vec4, 5> extractFrustumPlanesReverseZ(const glm::mat4& viewProj);

/// Fills GPUCullParams.  `proj11` = projection[1][1].  `cameraForward` is
/// documentation only: the size test uses the near plane's normal, which
/// equals it (checked by the unit tests), so no separate axis is stored.
/// groupCount = ceil(slotCount / SCENE_CULL_GROUP).
GPUCullParams makeCullParams(const glm::mat4& viewProj, float proj11, u32 viewportHeight, glm::vec3 cameraPos,
                             glm::vec3 cameraForward, float nearPlane, u32 flags, float maxDistance, float minPixels,
                             u32 slotCount);

/// Per slot: CULL_RESULT_INVALID, or cullSphere() of the instance's world
/// sphere (0 visible, 1 frustum, 2 distance, 3 size).  `instances` must have
/// at least params.slotCount entries.  `margins` (optional, one per slot): the
/// signed margin of cullSphereEval() (invalid slots: 0).
void cullReference(std::span<const GPUInstance> instances, std::span<const GPUMeshInfo> meshes, const GPUCullParams& params,
                   std::vector<u8>& result, std::vector<float>* margins = nullptr);

/// Stable compaction: `visible` = the slots whose result is 0 in slot order;
/// `prefix` (slots + 1 entries) = exclusive prefix of the visible flags, the
/// last entry is the total.
void compactReference(std::span<const u8> result, std::vector<u32>& visible, std::vector<u32>& prefix);

/// Marker in the commandBuckets array of a command that has no bucket (the
/// sentinel at the end of each cull class range).
inline constexpr u32 DRAW_COMMAND_SENTINEL = 0xFFFFFFFFu;

/// Per ICB command: {instanceCount, baseInstance} (2 u32).  A bucket's count
/// is prefix[firstSlot + capacity] - prefix[firstSlot] and its baseInstance
/// prefix[firstSlot] (index into the visible list); sentinel and empty
/// buckets give {0, 0} (their command is reset on the GPU).
void drawArgsReference(std::span<const GPUDrawBucket> buckets, std::span<const u32> commandBuckets, std::span<const u32> prefix,
                       std::vector<u32>& args);

} // namespace phosphor
