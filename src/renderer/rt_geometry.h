#pragma once
#include "renderer/gpu_types.h"

namespace phosphor {
// F9 GPURtHit::frontFacing is WORLD winding. Reflection through a mirrored
// model reverses that winding while keeping the original material's front
// side. Use this conversion only for sided material/probe-classification
// decisions; never rewrite the shared hit's geometric WORLD facing.
[[nodiscard]] inline bool rtMaterialFrontFacing(u32 worldFrontFacing,u32 instanceFlags) {
    return (worldFrontFacing!=0u)!=((instanceFlags&INSTANCE_FLAG_MIRRORED)!=0u);
}
} // namespace phosphor
