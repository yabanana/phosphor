#pragma once
#include "renderer/gpu_types.h"
#ifndef __METAL_VERSION__
#include <cmath>
#endif
namespace phosphor {
// Shared CPU/MSL pre-conversion admission of the COMPLETE radiance sum.
// The caller marks rejected GPU output with alpha=-1; publication stays disabled.
inline bool reflectionProbeStorageAccepts(float red,float green,float blue,u32 flags) {
#ifdef __METAL_VERSION__
    const bool finite=metal::isfinite(red)&&metal::isfinite(green)&&metal::isfinite(blue);
#else
    const bool finite=std::isfinite(red)&&std::isfinite(green)&&std::isfinite(blue);
#endif
    return finite&&red>=0&&green>=0&&blue>=0&&
        (!(flags&REFLECTION_CAPTURE_HALF)||(red<=65504.f&&green<=65504.f&&blue<=65504.f));
}
} // namespace phosphor
