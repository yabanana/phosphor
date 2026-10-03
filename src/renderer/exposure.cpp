#include "renderer/exposure.h"
#include <algorithm>
#include <cmath>

namespace phosphor {
u32 exposureBin(float luminance) {
    if (!std::isfinite(luminance) || luminance <= std::exp2(ExposureLogMin))
        return 0;
    const float normalized = std::clamp((std::log2(luminance) - ExposureLogMin) / ExposureLogRange, 0.0f, 1.0f);
    return 1u + static_cast<u32>(normalized * 254.0f);
}
float exposureTarget(const std::array<u32, ExposureBins> &h, float low, float high) {
    u64 count = 0;
    for (u32 i = 1; i < ExposureBins; ++i)
        count += h[i];
    if (!count)
        return 1.0f;
    const double begin = double(count) * std::clamp(low, 0.0f, 1.0f);
    const double end = double(count) * std::clamp(high, low, 1.0f);
    double cursor = 0, sum = 0, used = 0;
    for (u32 i = 1; i < ExposureBins; ++i) {
        const double weight = std::max(0.0, std::min(cursor + h[i], end) - std::max(cursor, begin));
        const double logLuminance = ExposureLogMin + (double(i - 1) + 0.5) / 254.0 * ExposureLogRange;
        sum += weight * logLuminance;
        used += weight;
        cursor += h[i];
    }
    if (used == 0)
        return 1.0f;
    return std::clamp(0.18f / std::exp2(float(sum / used)), 1.0f / 1024.0f, 1024.0f);
}
float adaptExposure(float previous, float target, float dt, float speed) {
    if (!std::isfinite(target) || target <= 0)
        target = 1;
    if (!std::isfinite(previous) || previous <= 0)
        return target;
    return previous + (target - previous) * (1.0f - std::exp(-std::max(dt, 0.0f) * std::max(speed, 0.0f)));
}
} // namespace phosphor
