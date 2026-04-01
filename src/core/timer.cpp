#include "core/timer.h"
#include <numeric>

namespace phosphor {

Timer::Timer()
    : frequency_(SDL_GetPerformanceFrequency())
    , lastCounter_(SDL_GetPerformanceCounter())
{
    frameTimes_.fill(0.0f);
}

void Timer::tick() {
    u64 now = SDL_GetPerformanceCounter();
    u64 elapsed = now - lastCounter_;
    lastCounter_ = now;

    deltaTime_ = static_cast<f32>(elapsed) / static_cast<f32>(frequency_);
    totalTime_ += static_cast<f64>(elapsed) / static_cast<f64>(frequency_);
    ++frameCount_;

    frameTimes_[frameTimeIndex_] = deltaTime_;
    frameTimeIndex_ = (frameTimeIndex_ + 1) % FPS_SAMPLE_COUNT;
}

f32 Timer::getDeltaTime() const {
    return deltaTime_;
}

f64 Timer::getTotalTime() const {
    return totalTime_;
}

u64 Timer::getFrameCount() const {
    return frameCount_;
}

f32 Timer::getFPS() const {
    u32 samples = static_cast<u32>(std::min(frameCount_, static_cast<u64>(FPS_SAMPLE_COUNT)));
    if (samples == 0) return 0.0f;

    f32 sum = std::accumulate(frameTimes_.begin(),
                              frameTimes_.begin() + samples, 0.0f);
    if (sum <= 0.0f) return 0.0f;

    return static_cast<f32>(samples) / sum;
}

} // namespace phosphor
