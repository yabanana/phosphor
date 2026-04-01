#include "core/window.h"
#include "core/log.h"
#include <utility>

namespace phosphor {

Window::Window(const char* title, u32 width, u32 height)
    : width_(width), height_(height)
{
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        LOG_ERROR("SDL_Init failed: %s", SDL_GetError());
        return;
    }

    window_ = SDL_CreateWindow(title, static_cast<int>(width),
                               static_cast<int>(height),
                               SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
    if (!window_) {
        LOG_ERROR("SDL_CreateWindow failed: %s", SDL_GetError());
        return;
    }

    LOG_INFO("Window created: %s (%ux%u)", title, width, height);
}

Window::~Window() {
    if (window_) {
        SDL_DestroyWindow(window_);
        window_ = nullptr;
    }
    SDL_Quit();
}

Window::Window(Window&& other) noexcept
    : window_(std::exchange(other.window_, nullptr))
    , width_(other.width_)
    , height_(other.height_)
    , closed_(other.closed_)
    , resized_(other.resized_)
    , pendingEvents_(std::move(other.pendingEvents_))
{
}

Window& Window::operator=(Window&& other) noexcept {
    if (this != &other) {
        if (window_) {
            SDL_DestroyWindow(window_);
        }
        window_   = std::exchange(other.window_, nullptr);
        width_    = other.width_;
        height_   = other.height_;
        closed_   = other.closed_;
        resized_  = other.resized_;
        pendingEvents_ = std::move(other.pendingEvents_);
    }
    return *this;
}

void Window::pollEvents() {
    pendingEvents_.clear();
    resized_ = false;

    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        pendingEvents_.push_back(event);

        switch (event.type) {
        case SDL_EVENT_QUIT:
            closed_ = true;
            break;
        case SDL_EVENT_WINDOW_RESIZED:
            width_   = static_cast<u32>(event.window.data1);
            height_  = static_cast<u32>(event.window.data2);
            resized_ = true;
            LOG_DEBUG("Window resized: %ux%u", width_, height_);
            break;
        default:
            break;
        }
    }
}

VkSurfaceKHR Window::createVulkanSurface(VkInstance instance) const {
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (!SDL_Vulkan_CreateSurface(window_, instance, nullptr, &surface)) {
        LOG_ERROR("SDL_Vulkan_CreateSurface failed: %s", SDL_GetError());
        return VK_NULL_HANDLE;
    }
    return surface;
}

bool Window::shouldClose() const {
    return closed_;
}

bool Window::wasResized() {
    bool r = resized_;
    resized_ = false;
    return r;
}

VkExtent2D Window::getExtent() const {
    return {width_, height_};
}

u32 Window::getWidth() const {
    return width_;
}

u32 Window::getHeight() const {
    return height_;
}

SDL_Window* Window::getSDLWindow() const {
    return window_;
}

const std::vector<SDL_Event>& Window::getPendingEvents() const {
    return pendingEvents_;
}

} // namespace phosphor
