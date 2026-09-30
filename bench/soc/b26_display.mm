// B-26: display pacing with CAMetalDisplayLink (needs --window: it opens
// windows and, for one test, switches a window to full screen).
//
// Per refresh rate (120 and 60 Hz) and window kind (titled window, borderless
// window covering the screen, full screen) the display link drives clear-only
// frames for a few seconds; per frame we record the callback time, the target
// presentation time, and the drawable's presentedTime (addPresentedHandler):
// callback interval jitter, presented interval jitter, presented - target
// error, frames not presented or late, callback -> presentation lead
// (spike: 41.6 ms constant in a composited window, independent of
// preferredFrameLatency).  Then a GPU load of known duration per frame
// (0, 4, 7, 9 ms at 120 Hz, frame budget 8.33 ms): frames must be missed only
// when the load exceeds the budget (negative control).
// Input-to-photon latency needs an external sensor and is NOT measured.
//
// Serves S-DISP-1 of docs/APPLE_SOC_PLAYBOOK.md (§14).

#import <AppKit/AppKit.h>
#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>

#include "harness.h"

#include <algorithm>
#include <cmath>

namespace soc {
namespace {

struct FrameRec {
    double cb = 0, target = 0, targetPresent = 0, presented = 0, gpuMs = -1;
};

NSString* const kLoadSrc = @"#include <metal_stdlib>\nusing namespace metal;\n"
                            "kernel void load(device float* out [[buffer(0)]], constant uint& iters [[buffer(1)]], uint i [[thread_position_in_grid]]) {\n"
                            "  float v = float(i & 15) * 0.0625f + 1.0f, w = 0.9990234375f;\n"
                            "  for (uint k = 0; k < iters; ++k) { v = fma(v, w, 0.0009765625f); }\n"
                            "  out[i] = v;\n}\n";
constexpr uint32_t kLoadThreads = 1u << 20;

} // namespace
} // namespace soc

@interface SocDisplayDriver : NSObject <CAMetalDisplayLinkDelegate>
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLComputePipelineState> loadPso;
@property(nonatomic, strong) id<MTLBuffer> loadOut;
@property(nonatomic) uint32_t loadIters;
@property(nonatomic) BOOL withLoad;
@property(nonatomic) double startTime;
@property(nonatomic) double seconds;
@property(nonatomic) BOOL finished;
- (soc::FrameRec*)records;
- (size_t)count;
@end

@implementation SocDisplayDriver {
    std::vector<soc::FrameRec> recs_; // capacity reserved up front: handlers write elements from other threads
}
- (instancetype)init {
    if ((self = [super init])) recs_.reserve(4096);
    return self;
}
- (soc::FrameRec*)records { return recs_.data(); }
- (size_t)count { return recs_.size(); }
- (void)metalDisplayLink:(CAMetalDisplayLink*)link needsUpdate:(CAMetalDisplayLinkUpdate*)update {
    const double now = CACurrentMediaTime();
    if (self.startTime == 0) self.startTime = now;
    if (self.finished || recs_.size() >= 4000) return;
    const size_t idx = recs_.size();
    recs_.push_back({now, update.targetTimestamp, update.targetPresentationTimestamp, 0, -1});
    soc::FrameRec* rec = &recs_[idx];
    if (self.withLoad) {
        id<MTLCommandBuffer> lcb = [self.queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [lcb computeCommandEncoder];
        uint32_t iters = self.loadIters;
        [e setComputePipelineState:self.loadPso];
        [e setBuffer:self.loadOut offset:0 atIndex:0];
        [e setBytes:&iters length:4 atIndex:1];
        [e dispatchThreads:MTLSizeMake(soc::kLoadThreads, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e endEncoding];
        [lcb addCompletedHandler:^(id<MTLCommandBuffer> cb) { rec->gpuMs = (cb.GPUEndTime - cb.GPUStartTime) * 1e3; }];
        [lcb commit];
    }
    id<CAMetalDrawable> d = update.drawable;
    MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = d.texture;
    rp.colorAttachments[0].loadAction = MTLLoadActionClear;
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    rp.colorAttachments[0].clearColor = MTLClearColorMake(0.1 * double(idx % 10), 0.2, 0.3, 1);
    id<MTLCommandBuffer> cb = [self.queue commandBuffer];
    [[cb renderCommandEncoderWithDescriptor:rp] endEncoding];
    [d addPresentedHandler:^(id<MTLDrawable> dr) { rec->presented = dr.presentedTime; }];
    [cb presentDrawable:d];
    [cb commit];
    if (now - self.startTime > self.seconds) self.finished = YES;
}
@end

namespace soc {
namespace {

void pump(double seconds) {
    NSDate* end = [NSDate dateWithTimeIntervalSinceNow:seconds];
    while ([end timeIntervalSinceNow] > 0) {
        @autoreleasepool {
            NSEvent* e;
            while ((e = [NSApp nextEventMatchingMask:NSEventMaskAny untilDate:nil inMode:NSDefaultRunLoopMode dequeue:YES])) [NSApp sendEvent:e];
            [[NSRunLoop currentRunLoop] runMode:NSDefaultRunLoopMode beforeDate:[NSDate dateWithTimeIntervalSinceNow:0.002]];
        }
    }
}

enum Kind { Titled, Borderless, FullScreen };

struct Win {
    NSWindow* window = nil;
    CAMetalLayer* layer = nil;
};

Win openWindow(Kind kind, id<MTLDevice> dev) {
    NSScreen* screen = NSScreen.mainScreen ? NSScreen.mainScreen : NSScreen.screens.firstObject;
    Win w;
    if (kind == Borderless) {
        w.window = [[NSWindow alloc] initWithContentRect:screen.frame styleMask:NSWindowStyleMaskBorderless backing:NSBackingStoreBuffered defer:NO];
        w.window.level = NSMainMenuWindowLevel + 1;
    } else {
        w.window = [[NSWindow alloc] initWithContentRect:NSMakeRect(100, 100, 640, 400)
                                               styleMask:(NSWindowStyleMaskTitled | NSWindowStyleMaskResizable)
                                                 backing:NSBackingStoreBuffered
                                                   defer:NO];
        w.window.title = @"soc_bench B-26";
        if (kind == FullScreen) w.window.collectionBehavior = NSWindowCollectionBehaviorFullScreenPrimary;
    }
    w.window.releasedWhenClosed = NO;
    w.layer = [CAMetalLayer layer];
    w.layer.device = dev;
    w.layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    w.window.contentView.wantsLayer = YES;
    w.window.contentView.layer = w.layer;
    [w.window makeKeyAndOrderFront:nil];
    [NSApp activateIgnoringOtherApps:YES];
    if (kind == FullScreen) {
        [w.window toggleFullScreen:nil];
        for (int i = 0; i < 400 && !(w.window.styleMask & NSWindowStyleMaskFullScreen); ++i) pump(0.01);
        pump(1.5); // the transition animation
    } else {
        pump(0.3);
    }
    const CGFloat scale = w.window.backingScaleFactor;
    const NSSize sz = w.window.contentView.bounds.size;
    w.layer.drawableSize = CGSizeMake(sz.width * scale, sz.height * scale);
    pump(0.1);
    return w;
}

void closeWindow(Win& w, Kind kind) {
    if (kind == FullScreen && (w.window.styleMask & NSWindowStyleMaskFullScreen)) {
        [w.window toggleFullScreen:nil];
        for (int i = 0; i < 400 && (w.window.styleMask & NSWindowStyleMaskFullScreen); ++i) pump(0.01);
        pump(1.0);
    }
    [w.window orderOut:nil];
    [w.window close];
    w.window = nil;
    w.layer = nil;
    pump(0.1);
}

struct Result {
    std::vector<double> cbInterval, presentInterval, lead, presentErr, gpuMs;
    size_t frames = 0, notPresented = 0, late = 0;
    double actualHz = 0;
};

/// Drives `layer` for `seconds`; skips the first `skip` frames.
Result drive(Win& w, SocDisplayDriver* drv, double fps, float latency, double seconds, bool withLoad) {
    CAMetalDisplayLink* link = [[CAMetalDisplayLink alloc] initWithMetalLayer:w.layer];
    link.delegate = drv;
    link.preferredFrameRateRange = CAFrameRateRangeMake(float(fps), float(fps), float(fps));
    if (latency > 0) link.preferredFrameLatency = latency;
    drv.seconds = seconds;
    drv.withLoad = withLoad;
    [link addToRunLoop:[NSRunLoop mainRunLoop] forMode:NSRunLoopCommonModes];
    const double t0 = CACurrentMediaTime();
    while (!drv.finished && CACurrentMediaTime() - t0 < seconds + 5.0) pump(0.005);
    [link invalidate];
    pump(0.4); // last presented handlers
    Result r;
    const FrameRec* f = drv.records;
    const size_t n = drv.count;
    const double period = 1.0 / fps;
    const size_t skip = 10;
    for (size_t i = skip; i < n; ++i) {
        ++r.frames;
        r.cbInterval.push_back((f[i].cb - f[i - 1].cb) * 1e3);
        r.lead.push_back((f[i].targetPresent - f[i].cb) * 1e3);
        if (f[i].gpuMs >= 0) r.gpuMs.push_back(f[i].gpuMs);
        if (f[i].presented > 0) {
            r.presentErr.push_back((f[i].presented - f[i].targetPresent) * 1e3);
            if (f[i - 1].presented > 0) r.presentInterval.push_back((f[i].presented - f[i - 1].presented) * 1e3);
            if (f[i].presented - f[i].targetPresent > 0.5 * period) ++r.late;
        } else if (i + 8 < n) { // the last frames may not have reported yet
            ++r.notPresented;
        }
    }
    if (n > skip + 1) r.actualHz = double(n - skip - 1) / (f[n - 1].cb - f[skip].cb);
    return r;
}

double sd(const std::vector<double>& v) {
    if (v.size() < 2) return 0;
    double m = 0;
    for (double x : v) m += x;
    m /= double(v.size());
    double s = 0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / double(v.size()));
}

void benchDisplay(Context& ctx, Report& rep) {
    if (!ctx.options().window) {
        rep.status(Status::Unsupported, "needs --window");
        return;
    }
    if (![NSThread isMainThread]) {
        rep.status(Status::Unsupported, "needs the main thread");
        return;
    }
    const double secs = ctx.quick() ? 1.0 : 3.0;
    [NSApplication sharedApplication];
    [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
    [NSApp finishLaunching];
    NSScreen* screen = NSScreen.mainScreen ? NSScreen.mainScreen : NSScreen.screens.firstObject;
    if (!screen) {
        rep.status(Status::Unsupported, "no display");
        return;
    }
    const long maxFps = long(screen.maximumFramesPerSecond);
    rep.note("screen: " + std::string(screen.localizedName.UTF8String) + ", maximumFramesPerSecond " + std::to_string(maxFps) + ", scale " +
             std::to_string(int(screen.backingScaleFactor)) + ", frame " + std::to_string(int(screen.frame.size.width)) + "x" +
             std::to_string(int(screen.frame.size.height)) + " pt");

    id<MTLDevice> dev = (__bridge id<MTLDevice>)ctx.device();
    SocDisplayDriver* drv = [SocDisplayDriver new];
    drv.queue = [dev newCommandQueue];
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:kLoadSrc options:nil error:&err];
    if (!lib) throw BenchError(std::string("load shader: ") + err.localizedDescription.UTF8String);
    drv.loadPso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"load"] error:&err];
    drv.loadOut = [dev newBufferWithLength:soc::kLoadThreads * 4 options:MTLResourceStorageModeShared];

    // GPU dispatch length calibration: iters for a target duration (ms) measured on its own.
    auto oneLoad = [&](uint32_t iters) {
        id<MTLCommandBuffer> cb = [drv.queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:drv.loadPso];
        [e setBuffer:drv.loadOut offset:0 atIndex:0];
        [e setBytes:&iters length:4 atIndex:1];
        [e dispatchThreads:MTLSizeMake(soc::kLoadThreads, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        return (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
    };
    auto calibrate = [&](double targetMs) {
        uint32_t iters = 1024;
        for (int round = 0; round < 4; ++round) {
            // Sustained load at the target so the clocks settle at what the frames will see.
            double t = 0;
            const double w0 = nowMs();
            while (nowMs() - w0 < 150.0) t = oneLoad(iters);
            iters = std::max<uint32_t>(16, uint32_t(double(iters) * targetMs / std::max(0.01, t)));
        }
        return iters;
    };

    struct Row { std::string name; Result r; double hz; };
    std::vector<Row> rows;
    auto reportRow = [&](const std::string& suffix, const Result& r) {
        const std::string s = suffix;
        if (r.presentInterval.empty()) {
            rep.note(s + ": no presented frames recorded");
            return;
        }
        const Stats pi = phosphor::soc::computeStats(r.presentInterval);
        rep.value("display.jitter_ms." + s, "ms", sd(r.presentInterval), {{"frames", double(r.frames)}}, false);
        rep.value("display.present_interval_ms." + s, "ms", pi.median, {}, false);
        rep.value("display.cb_jitter_ms." + s, "ms", sd(r.cbInterval), {}, false);
        rep.metric("display.lead_ms." + s, "ms", phosphor::soc::computeStats(r.lead), {}, false);
        rep.value("display.present_error_ms." + s, "ms", phosphor::soc::computeStats(r.presentErr).median, {}, false);
        rep.value("display.actual_hz." + s, "Hz", r.actualHz, {});
        rep.value("display.frames_not_presented." + s, "count", double(r.notPresented), {{"frames", double(r.frames)}}, false);
        rep.value("display.frames_late." + s, "count", double(r.late), {{"frames", double(r.frames)}}, false);
        ctx.log("%s: %zu frames, callback %.2f Hz, presented interval p50 %.3f sd %.3f ms, lead p50 %.2f ms, not presented %zu, late %zu", s.c_str(), r.frames,
                r.actualHz, pi.median, sd(r.presentInterval), phosphor::soc::computeStats(r.lead).median, r.notPresented, r.late);
    };

    // 1. Titled window (composited): 120 and 60 Hz, latency default.
    {
        Win w = openWindow(Titled, dev);
        {
            CAMetalDisplayLink* probe = [[CAMetalDisplayLink alloc] initWithMetalLayer:w.layer];
            rep.note("preferredFrameLatency default = " + std::to_string(probe.preferredFrameLatency));
            [probe invalidate];
        }
        for (double hz : {120.0, 60.0}) {
            SocDisplayDriver* d = [SocDisplayDriver new];
            d.queue = drv.queue;
            const Result r = drive(w, d, hz, 0, secs, false);
            reportRow(std::to_string(int(hz)) + "hz", r);
            rows.push_back({"titled", r, hz});
        }
        // preferredFrameLatency 1 and 3 at 120 Hz.
        for (float lat : ctx.quick() ? std::vector<float>{1.0f} : std::vector<float>{1.0f, 3.0f}) {
            SocDisplayDriver* d = [SocDisplayDriver new];
            d.queue = drv.queue;
            const Result r = drive(w, d, 120.0, lat, secs, false);
            reportRow("120hz.latency_" + std::to_string(int(lat)), r);
        }
        // 2. GPU load of known duration at 120 Hz.
        const double budget = 1e3 / 120.0;
        bool below = true, above = false, haveAbove = false, haveBelow = false;
        std::string detail;
        for (double load : {0.0, 4.0, 7.0, 9.0}) {
            SocDisplayDriver* d = [SocDisplayDriver new];
            d.queue = drv.queue;
            d.loadPso = drv.loadPso;
            d.loadOut = drv.loadOut;
            if (load > 0) d.loadIters = calibrate(load);
            const Result r = drive(w, d, 120.0, 0, secs, load > 0);
            const std::string s = "load_" + std::to_string(int(load)) + "ms.120hz";
            reportRow(s, r);
            const double gpu = r.gpuMs.empty() ? 0.0 : phosphor::soc::computeStats(r.gpuMs).median;
            const size_t missed = r.notPresented + r.late;
            const double pct = r.frames ? 100.0 * double(missed) / double(r.frames) : 0.0;
            rep.value("display." + s + ".gpu_ms", "ms", gpu, {{"target_ms", load}});
            rep.value("display." + s + ".missed_frames", "count", double(missed), {{"frames", double(r.frames)}, {"target_ms", load}}, false);
            rep.value("display." + s + ".missed_pct", "%", pct, {{"target_ms", load}}, false);
            detail += std::to_string(int(load)) + "ms(measured " + std::to_string(gpu).substr(0, 4) + ") missed " + std::to_string(pct).substr(0, 4) + "%; ";
            if (gpu > 0 && gpu < 0.85 * budget) { haveBelow = true; below &= pct <= 3.0; }
            if (gpu > 1.05 * budget) { haveAbove = true; above |= pct > 20.0; }
            if (load == 0.0) { haveBelow = true; below &= pct <= 3.0; }
        }
        closeWindow(w, Titled);
        // Control filled in at the end (needs the load rows only).
        rep.negative(haveBelow && haveAbove && below && above,
                     std::string("120 Hz budget 8.33 ms: loads well under budget miss <= 3% of frames, loads over budget miss > 20%: ") + detail +
                         (haveAbove ? "" : "no load exceeded the budget (calibration failed); ") + (below ? "" : "FRAMES MISSED UNDER BUDGET; ") + (haveAbove && !above ? "NO MISS ABOVE BUDGET" : ""));
    }
    // 3. Borderless window covering the screen.
    {
        Win w = openWindow(Borderless, dev);
        SocDisplayDriver* d = [SocDisplayDriver new];
        d.queue = drv.queue;
        const Result r = drive(w, d, 120.0, 0, secs, false);
        reportRow("borderless.120hz", r);
        closeWindow(w, Borderless);
    }
    // 4. Full screen window.
    {
        Win w = openWindow(FullScreen, dev);
        const bool isFull = (w.window.styleMask & NSWindowStyleMaskFullScreen) != 0;
        SocDisplayDriver* d = [SocDisplayDriver new];
        d.queue = drv.queue;
        const Result r = drive(w, d, 120.0, 0, secs, false);
        reportRow("fullscreen.120hz", r);
        rep.note(std::string("fullscreen window: NSWindowStyleMaskFullScreen ") + (isFull ? "reached" : "NOT reached (measured as a normal window)"));
        closeWindow(w, FullScreen);
    }
    rep.note("lead_ms = targetPresentationTimestamp - callback time; jitter_ms = standard deviation of the presented interval (drawable presentedTime); cb_jitter_ms = sd of the callback interval; frames late = presented > target + half a frame");
    rep.note("keys without a kind prefix (display.jitter_ms.120hz ...) are the titled (composited) window; borderless./fullscreen. and latency_N/load_Nms variants have their own keys");
    rep.status(Status::Partial, "input-to-photon latency needs an external sensor (photodiode/high-speed camera): not measured; only the software-visible presentation timing is");
}

} // namespace

SOC_BENCH("B-26", "display.pacing", "CAMetalDisplayLink pacing at 120/60 Hz: jitter, presentation error, lead, missed frames under load (--window)", benchDisplay);

} // namespace soc
