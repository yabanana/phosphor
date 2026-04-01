#pragma once

#include "core/types.h"
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <vulkan/vulkan.h>
#include <vector>

namespace phosphor {

class Window {
public:
    Window(const char* title, u32 width, u32 height);
    ~Window();

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;
    Window(Window&& other) noexcept;
    Window& operator=(Window&& other) noexcept;

    void           pollEvents();
    VkSurfaceKHR   createVulkanSurface(VkInstance instance) const;
    bool           shouldClose() const;
    bool           wasResized();
    VkExtent2D     getExtent() const;
    u32            getWidth() const;
    u32            getHeight() const;
    SDL_Window*    getSDLWindow() const;

    const std::vector<SDL_Event>& getPendingEvents() const;

private:
    SDL_Window*           window_    = nullptr;
    u32                   width_     = 0;
    u32                   height_    = 0;
    bool                  closed_    = false;
    bool                  resized_   = false;
    std::vector<SDL_Event> pendingEvents_;
};

} // namespace phosphor
