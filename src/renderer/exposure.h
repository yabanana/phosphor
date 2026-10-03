#pragma once
#include "core/types.h"
#include <array>

namespace phosphor {
constexpr u32 ExposureBins = 256;
constexpr float ExposureLogMin = -12.0f, ExposureLogRange = 28.0f;
u32 exposureBin(float luminance);
float exposureTarget(const std::array<u32, ExposureBins> &histogram, float low = 0.05f, float high = 0.95f);
float adaptExposure(float previous, float target, float dt, float speed = 3.0f);
} // namespace phosphor
