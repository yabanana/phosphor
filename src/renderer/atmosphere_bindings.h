#pragma once
#include "renderer/gpu_types.h"
namespace phosphor {
// Cloud composition contract: the incoming scene already includes atmospheric
// scattering along the camera ray. Shared optical LUTs can still exist when
// visible atmosphere is disabled; that case must not add front air twice.
PHOSPHOR_GPU_CONSTANT u32 CLOUD_SCENE_HAS_ATMOSPHERE = 1u << 5;
} // namespace phosphor
