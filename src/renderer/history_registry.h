#pragma once
#include "core/types.h"
#include <array>

namespace phosphor {

// F8 baseline: one ordered graphics timeline, independent temporal state per
// view. Buffer slots are not views. The backend orders GPU reuse at or after
// requiredCompletion; the CPU never treats submission as GPU completion.
class HistoryRegistry {
  public:
    static constexpr u32 MaxViews = 4;
    struct Extent {
        u32 inputWidth = 0, inputHeight = 0, outputWidth = 0, outputHeight = 0;
        bool operator==(const Extent &) const = default;
    };
    struct View {
        Extent extent{};
        u64 scene = 0, generation = 0, lastReader = 0, lastWriter = 0;
        u32 sample = 0;
        bool valid = false;
        const char *resetReason = "initial";
        std::array<float, 16> previousViewProjection{};
    };
    struct Decision {
        bool reset;
        u64 requiredCompletion;
        u64 generation;
        const char *reason;
    };
    Decision begin(u32 view, Extent extent, u64 scene, bool cameraCut = false, bool explicitReset = false);
    void read(u32 view, u64 submission);
    // Each registry owns one signal across views: Hi-Z stores a jittered
    // projection, while motion/reconstruction stores an unjittered one.
    void write(u32 view, u64 submission, const float *viewProjection);
    void invalidate(u32 view, const char *reason);
    [[nodiscard]] const View &get(u32 view) const;

  private:
    std::array<View, MaxViews> views_{};
};

void applyRasterJitter(float *output, const float *projection, float x, float y, u32 width, u32 height);
float halton(u32 index, u32 base);
std::array<float, 2> temporalJitter(u32 sample);

} // namespace phosphor
