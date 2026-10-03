#pragma once
struct SDL_Window;
namespace CA {
class MetalLayer;
}
namespace phosphor {
struct DisplayOutputInfo {
    float headroom = 1.0f, potentialHeadroom = 1.0f;
    bool edr = false;
};
// mode 0 = SDR, 1 = EDR if physically supported, 2 = automatic.
DisplayOutputInfo configureDisplayOutput(SDL_Window *, CA::MetalLayer *, unsigned mode);
} // namespace phosphor
