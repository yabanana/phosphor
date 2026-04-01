#pragma once

#include "core/types.h"
#include <SDL3/SDL_events.h>
#include <SDL3/SDL_scancode.h>
#include <glm/glm.hpp>
#include <array>

namespace phosphor {

class Input {
public:
    void processEvent(const SDL_Event& event);
    void resetFrameState();

    bool      isKeyDown(SDL_Scancode key) const;
    bool      isKeyPressed(SDL_Scancode key) const;
    glm::vec2 getMouseDelta() const;
    glm::vec2 getMousePosition() const;
    bool      isMouseButtonDown(u32 button) const;
    f32       getScrollDelta() const;

private:
    static constexpr u32 KEY_COUNT = SDL_SCANCODE_COUNT;

    std::array<bool, KEY_COUNT> currentKeys_{};
    std::array<bool, KEY_COUNT> previousKeys_{};

    glm::vec2 mousePosition_{0.0f, 0.0f};
    glm::vec2 mouseDelta_{0.0f, 0.0f};
    f32       scrollDelta_ = 0.0f;
    u32       mouseButtons_ = 0;
};

} // namespace phosphor
