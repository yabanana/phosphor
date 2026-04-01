#include "core/input.h"
#include <SDL3/SDL_keyboard.h>
#include <SDL3/SDL_mouse.h>

namespace phosphor {

void Input::processEvent(const SDL_Event& event) {
    switch (event.type) {
    case SDL_EVENT_KEY_DOWN:
        if (event.key.scancode < KEY_COUNT) {
            currentKeys_[event.key.scancode] = true;
        }
        break;
    case SDL_EVENT_KEY_UP:
        if (event.key.scancode < KEY_COUNT) {
            currentKeys_[event.key.scancode] = false;
        }
        break;
    case SDL_EVENT_MOUSE_MOTION:
        mousePosition_.x = event.motion.x;
        mousePosition_.y = event.motion.y;
        mouseDelta_.x   += event.motion.xrel;
        mouseDelta_.y   += event.motion.yrel;
        break;
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
        mouseButtons_ |= (1u << event.button.button);
        break;
    case SDL_EVENT_MOUSE_BUTTON_UP:
        mouseButtons_ &= ~(1u << event.button.button);
        break;
    case SDL_EVENT_MOUSE_WHEEL:
        scrollDelta_ += event.wheel.y;
        break;
    default:
        break;
    }
}

void Input::resetFrameState() {
    previousKeys_ = currentKeys_;
    mouseDelta_   = {0.0f, 0.0f};
    scrollDelta_  = 0.0f;
}

bool Input::isKeyDown(SDL_Scancode key) const {
    if (static_cast<u32>(key) >= KEY_COUNT) return false;
    return currentKeys_[key];
}

bool Input::isKeyPressed(SDL_Scancode key) const {
    if (static_cast<u32>(key) >= KEY_COUNT) return false;
    return currentKeys_[key] && !previousKeys_[key];
}

glm::vec2 Input::getMouseDelta() const {
    return mouseDelta_;
}

glm::vec2 Input::getMousePosition() const {
    return mousePosition_;
}

bool Input::isMouseButtonDown(u32 button) const {
    return (mouseButtons_ & (1u << button)) != 0;
}

f32 Input::getScrollDelta() const {
    return scrollDelta_;
}

} // namespace phosphor
