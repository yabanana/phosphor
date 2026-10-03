#include "platform/metal/display_output.h"
#include <SDL3/SDL.h>
#import <AppKit/AppKit.h>
#import <QuartzCore/CAMetalLayer.h>
#include <algorithm>
#include <cmath>

namespace phosphor {
DisplayOutputInfo configureDisplayOutput(SDL_Window *window, CA::MetalLayer *layer, unsigned mode) {
    @autoreleasepool {
        void *pointer =
            SDL_GetPointerProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_COCOA_WINDOW_POINTER, nullptr);
        NSWindow *native = (__bridge NSWindow *)pointer;
        NSScreen *screen = native.screen ?: NSScreen.mainScreen;
        DisplayOutputInfo info;
        if (screen) {
            info.headroom = std::max(1.0f, float(screen.maximumExtendedDynamicRangeColorComponentValue));
            info.potentialHeadroom =
                std::max(1.0f, float(screen.maximumPotentialExtendedDynamicRangeColorComponentValue));
        }
        if (!std::isfinite(info.headroom))
            info.headroom = 1;
        if (!std::isfinite(info.potentialHeadroom))
            info.potentialHeadroom = 1;
        info.edr = mode != 0 && info.potentialHeadroom > 1.0001f;
        CAMetalLayer *metal = (__bridge CAMetalLayer *)static_cast<void *>(layer);
        const MTLPixelFormat format = info.edr ? MTLPixelFormatRGBA16Float : MTLPixelFormatBGRA8Unorm_sRGB;
        if (metal.pixelFormat != format || metal.wantsExtendedDynamicRangeContent != info.edr) {
            metal.wantsExtendedDynamicRangeContent = info.edr;
            metal.pixelFormat = format;
            CGColorSpaceRef colorSpace =
                CGColorSpaceCreateWithName(info.edr ? kCGColorSpaceExtendedLinearSRGB : kCGColorSpaceSRGB);
            metal.colorspace = colorSpace;
            CGColorSpaceRelease(colorSpace);
        }
        if (!info.edr)
            info.headroom = 1;
        return info;
    }
}
} // namespace phosphor
