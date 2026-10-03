#include "renderer/history_registry.h"
#include <algorithm>
#include <stdexcept>

namespace phosphor {
const HistoryRegistry::View &HistoryRegistry::get(u32 view) const {
    return views_.at(view);
}
void HistoryRegistry::invalidate(u32 view, const char *reason) {
    auto &v = views_.at(view);
    v.valid = false;
    v.sample = 0;
    v.resetReason = reason;
    ++v.generation;
}
HistoryRegistry::Decision HistoryRegistry::begin(u32 view, Extent extent, u64 scene, bool cameraCut,
                                                 bool explicitReset) {
    if (!extent.inputWidth || !extent.inputHeight || !extent.outputWidth || !extent.outputHeight)
        throw std::invalid_argument("Temporal history extent must be positive");
    auto &v = views_.at(view);
    const char *reason = nullptr;
    if (v.extent != extent)
        reason = "resolution";
    if (v.scene != scene)
        reason = "scene";
    if (cameraCut)
        reason = "camera cut";
    if (explicitReset)
        reason = "explicit reset";
    if (reason)
        invalidate(view, reason);
    v.extent = extent;
    v.scene = scene;
    return {!v.valid, std::max(v.lastReader, v.lastWriter), v.generation, v.resetReason};
}
void HistoryRegistry::read(u32 view, u64 submission) {
    auto &v = views_.at(view);
    if (submission < v.lastWriter || submission < v.lastReader)
        throw std::logic_error("History read timeline moved backwards");
    v.lastReader = submission;
}
void HistoryRegistry::write(u32 view, u64 submission, const float *matrix) {
    auto &v = views_.at(view);
    if (submission < v.lastWriter || submission < v.lastReader)
        throw std::logic_error("History overwritten before its last reader");
    v.lastWriter = submission;
    v.valid = true;
    ++v.sample;
    std::copy_n(matrix, 16, v.previousViewProjection.begin());
}
void applyRasterJitter(float *output, const float *projection, float x, float y, u32 width, u32 height) {
    if (!width || !height)
        throw std::invalid_argument("Jitter extent must be positive");
    if (output != projection)
        std::copy_n(projection, 16, output);
    // Raster displacement in pixels, +X right and +Y down. The MetalFX
    // adapter convention is checked independently with a jitter sweep.
    for (u32 c = 0; c < 4; ++c) {
        output[c * 4] += (2.0f * x / float(width)) * projection[c * 4 + 3];
        output[c * 4 + 1] -= (2.0f * y / float(height)) * projection[c * 4 + 3];
    }
}
float halton(u32 index, u32 base) {
    if (base < 2)
        throw std::invalid_argument("Halton base must be at least two");
    float value = 0, scale = 1;
    while (index) {
        scale /= float(base);
        value += scale * float(index % base);
        index /= base;
    }
    return value;
}
std::array<float, 2> temporalJitter(u32 sample) {
    const u32 i = sample % 16 + 1;
    return {halton(i, 2) - 0.5f, halton(i, 3) - 0.5f};
}
} // namespace phosphor
