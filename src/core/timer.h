#pragma once

#include "core/types.h"
#include <SDL3/SDL_timer.h>
#include <array>

namespace phosphor {

class Timer {
public:
    Timer();

    void  tick();
    f32   getDeltaTime() const;
    f64   getTotalTime() const;
    u64   getFrameCount() const;
    f32   getFPS() const;

private:
    static constexpr u32 FPS_SAMPLE_COUNT = 60;

    u64 frequency_  = 0;
    u64 lastCounter_ = 0;
    f32 deltaTime_  = 0.0f;
    f64 totalTime_  = 0.0;
    u64 frameCount_ = 0;

    std::array<f32, FPS_SAMPLE_COUNT> frameTimes_{};
    u32 frameTimeIndex_ = 0;
};

} // namespace phosphor
