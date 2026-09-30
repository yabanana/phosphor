// OPT-0 spike 3: CAMetalDisplayLink and presentation timing (B-26).
// A MEASUREMENT PROBE, not engine code.
//
// Opens a window with a CAMetalLayer, drives it with CAMetalDisplayLink for
// N seconds (clear-only frames) and records, per frame: callback time vs
// targetTimestamp, targetPresentationTimestamp, and the drawable's
// presentedTime (addPresentedHandler).  Input-to-photon latency needs a
// sensor and is not measured.
//
// Usage: display_probe [seconds] [preferred fps]

#import <AppKit/AppKit.h>
#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

struct FrameRec { double cb, target, targetPresent, presented; };
static std::vector<FrameRec> g_frames;
static double g_seconds = 3.0;

@interface Probe : NSObject <CAMetalDisplayLinkDelegate>
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) CAMetalDisplayLink* link;
@property(nonatomic) double start;
@end

@implementation Probe
- (void)metalDisplayLink:(CAMetalDisplayLink*)link needsUpdate:(CAMetalDisplayLinkUpdate*)update {
    const double now = CACurrentMediaTime();
    if (self.start == 0) self.start = now;
    const size_t idx = g_frames.size();
    g_frames.push_back({now, update.targetTimestamp, update.targetPresentationTimestamp, 0});
    id<CAMetalDrawable> d = update.drawable;
    MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = d.texture;
    rp.colorAttachments[0].loadAction = MTLLoadActionClear;
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    rp.colorAttachments[0].clearColor = MTLClearColorMake(0.1 * (idx % 10), 0.2, 0.3, 1);
    id<MTLCommandBuffer> cb = [self.queue commandBuffer];
    [[cb renderCommandEncoderWithDescriptor:rp] endEncoding];
    [d addPresentedHandler:^(id<MTLDrawable> dr) {
        dispatch_async(dispatch_get_main_queue(), ^{ g_frames[idx].presented = dr.presentedTime; });
    }];
    [cb presentDrawable:d];
    [cb commit];
    if (now - self.start > g_seconds) {
        [link invalidate];
        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 300 * NSEC_PER_MSEC), dispatch_get_main_queue(), ^{ [NSApp stop:nil];
            [NSApp postEvent:[NSEvent otherEventWithType:NSEventTypeApplicationDefined location:NSZeroPoint modifierFlags:0
                                                timestamp:0 windowNumber:0 context:nil subtype:0 data1:0 data2:0] atStart:YES]; });
    }
}
@end

static void summarize(const char* name, std::vector<double> v) {
    if (v.empty()) { std::printf("  %-34s n/a\n", name); return; }
    std::sort(v.begin(), v.end());
    double mean = 0;
    for (double x : v) mean += x;
    mean /= double(v.size());
    double sd = 0;
    for (double x : v) sd += (x - mean) * (x - mean);
    sd = std::sqrt(sd / double(v.size()));
    std::printf("  %-34s p50 %7.3f  p1 %7.3f  p99 %7.3f  sd %6.3f ms (n=%zu)\n", name, v[v.size() / 2],
                v[v.size() / 100], v[v.size() * 99 / 100], sd, v.size());
}

int main(int argc, char** argv) {
    @autoreleasepool {
        g_seconds = argc > 1 ? atof(argv[1]) : 3.0;
        const float fps = argc > 2 ? float(atof(argv[2])) : 120.0f;
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
        NSWindow* w = [[NSWindow alloc] initWithContentRect:NSMakeRect(100, 100, 640, 400)
                                                  styleMask:NSWindowStyleMaskTitled backing:NSBackingStoreBuffered defer:NO];
        w.title = @"display_probe";
        CAMetalLayer* layer = [CAMetalLayer layer];
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        layer.device = dev;
        layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
        w.contentView.wantsLayer = YES;
        w.contentView.layer = layer;
        layer.drawableSize = CGSizeMake(1280, 800);
        [w makeKeyAndOrderFront:nil];
        [NSApp activateIgnoringOtherApps:YES];
        Probe* p = [Probe new];
        p.queue = [dev newCommandQueue];
        p.link = [[CAMetalDisplayLink alloc] initWithMetalLayer:layer];
        p.link.delegate = p;
        p.link.preferredFrameRateRange = CAFrameRateRangeMake(fps, fps, fps);
        if (argc > 3) p.link.preferredFrameLatency = float(atof(argv[3]));
        std::printf("preferredFrameLatency %.1f\n", p.link.preferredFrameLatency);
        [p.link addToRunLoop:[NSRunLoop mainRunLoop] forMode:NSRunLoopCommonModes];
        [NSApp run];
        std::printf("display_probe: %zu frames in %.1f s at preferred %.0f fps, screen max %ld fps\n", g_frames.size(), g_seconds,
                    fps, long(w.screen.maximumFramesPerSecond));
        std::vector<double> interval, lead, presentErr, presentInterval;
        size_t missed = 0, presentedCount = 0;
        for (size_t i = 5; i < g_frames.size(); ++i) {   // skip the first frames (window appearing)
            const auto& f = g_frames[i];
            interval.push_back((f.cb - g_frames[i - 1].cb) * 1e3);
            lead.push_back((f.targetPresent - f.cb) * 1e3);
            if (f.presented > 0) {
                ++presentedCount;
                presentErr.push_back((f.presented - f.targetPresent) * 1e3);
                if (g_frames[i - 1].presented > 0) presentInterval.push_back((f.presented - g_frames[i - 1].presented) * 1e3);
            } else {
                ++missed;
            }
        }
        summarize("callback interval", interval);
        summarize("callback -> target presentation", lead);
        summarize("presented - target presentation", presentErr);
        summarize("presented interval", presentInterval);
        std::printf("  presented %zu, not presented %zu\n", presentedCount, missed);
    }
    return 0;
}
