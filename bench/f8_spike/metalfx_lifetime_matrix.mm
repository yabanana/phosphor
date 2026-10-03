// F8.4 public-API lifetime comparison. No render graph or GPU submissions.
// Exit 0: probe ran without an observed live target (not a leak-scan verdict);
// --no-weak skips that observation and must be paired with a monitored run.
// Exit 1: live target; 2: invalid arguments;
// 77: unsupported/failed creation. Device allocation includes driver pools.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static NSUInteger liveCount(NSHashTable *weak) {
    @autoreleasepool {
        return [weak allObjects].count;
    }
}
static int number(const char *s, int min, int max) {
    char *end = nullptr;
    const long value = std::strtol(s, &end, 10);
    if (!*s || *end || value < min || value > max) {
        std::fprintf(stderr, "Invalid bounded numeric argument: %s\n", s);
        std::exit(2);
    }
    return static_cast<int>(value);
}
int main(int argc, char **argv) {
    std::string mode = "temporal4";
    int count = 8, width = 640, height = 360;
    bool dynamic = true, reactive = true, synchronous = true;
    bool autoExposure = false, reset = false, monitor = true;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--no-dynamic")
            dynamic = false;
        else if (arg == "--no-reactive")
            reactive = false;
        else if (arg == "--async")
            synchronous = false;
        else if (arg == "--auto-exposure")
            autoExposure = true;
        else if (arg == "--reset")
            reset = true;
        else if (arg == "--no-weak")
            monitor = false;
        else if (i + 1 < argc && arg == "--mode")
            mode = argv[++i];
        else if (i + 1 < argc && arg == "--count")
            count = number(argv[++i], 1, 16);
        else if (i + 1 < argc && arg == "--width")
            width = number(argv[++i], 32, 1920);
        else if (i + 1 < argc && arg == "--height")
            height = number(argv[++i], 32, 1080);
        else {
            std::fprintf(stderr, "Unknown/incomplete argument: %s\n", argv[i]);
            return 2;
        }
    }
    const bool temporal = mode == "temporal4" || mode == "temporal3";
    const bool denoised = mode == "denoised4" || mode == "denoised3";
    const bool metal3 = mode == "temporal3" || mode == "denoised3";
    if (!temporal && !denoised && mode != "spatial4")
        return 2;
    NSHashTable *weak = [[NSHashTable alloc] initWithOptions:NSHashTableWeakMemory capacity:count];
    @autoreleasepool {
        NSObject *control = [NSObject new];
        [weak addObject:control];
        [control release];
    }
    if (liveCount(weak))
        return 2;
    int failed = 0;
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        MTL4CompilerDescriptor *cd = [MTL4CompilerDescriptor new];
        NSError *error = nil;
        id<MTL4Compiler> compiler = metal3 ? nil : [device newCompilerWithDescriptor:cd error:&error];
        [cd release];
        if (!device || (!metal3 && !compiler))
            return 77;
        NSBundle *framework = [NSBundle bundleForClass:[MTLFXTemporalScalerDescriptor class]];
        std::printf(
            "CONFIG mode=%s count=%d size=%dx%d dynamic=%d reactive=%d sync=%d auto_exposure=%d reset=%d weak=%d\n",
            mode.c_str(), count, width, height, dynamic, reactive, synchronous, autoExposure, reset, monitor);
        std::printf("DEVICE %s | RUNTIME %s | FRAMEWORK %s\n", device.name.UTF8String,
                    [[NSProcessInfo processInfo] operatingSystemVersionString].UTF8String,
                    [[framework objectForInfoDictionaryKey:@"CFBundleVersion"] UTF8String]);
        const NSUInteger before = device.currentAllocatedSize;
        for (int i = 0; i < count; ++i) {
            @autoreleasepool {
                id effect = nil;
                if (temporal) {
                    MTLFXTemporalScalerDescriptor *d = [MTLFXTemporalScalerDescriptor new];
                    d.inputWidth = d.outputWidth = width;
                    d.inputHeight = d.outputHeight = height;
                    d.colorTextureFormat = d.outputTextureFormat = MTLPixelFormatRGBA16Float;
                    d.depthTextureFormat = MTLPixelFormatDepth32Float;
                    d.motionTextureFormat = MTLPixelFormatRG16Float;
                    d.requiresSynchronousInitialization = synchronous;
                    d.autoExposureEnabled = autoExposure;
                    d.reactiveMaskTextureEnabled = reactive;
                    d.reactiveMaskTextureFormat = MTLPixelFormatR8Unorm;
                    d.inputContentPropertiesEnabled = dynamic;
                    d.inputContentMinScale = 1;
                    d.inputContentMaxScale = 2;
                    if (metal3)
                        effect = [d newTemporalScalerWithDevice:device];
                    else
                        effect = [d newTemporalScalerWithDevice:device compiler:compiler];
                    if (reset)
                        [(id<MTLFXTemporalScalerBase>)effect setReset:YES];
                    [d release];
                } else if (denoised) {
                    MTLFXTemporalDenoisedScalerDescriptor *d = [MTLFXTemporalDenoisedScalerDescriptor new];
                    d.inputWidth = d.outputWidth = width;
                    d.inputHeight = d.outputHeight = height;
                    d.colorTextureFormat = d.outputTextureFormat = d.diffuseAlbedoTextureFormat =
                        d.specularAlbedoTextureFormat = d.normalTextureFormat = MTLPixelFormatRGBA16Float;
                    d.depthTextureFormat = MTLPixelFormatDepth32Float;
                    d.motionTextureFormat = MTLPixelFormatRG16Float;
                    d.roughnessTextureFormat = d.specularHitDistanceTextureFormat = MTLPixelFormatR16Float;
                    d.denoiseStrengthMaskTextureFormat = MTLPixelFormatR8Unorm;
                    d.transparencyOverlayTextureFormat = MTLPixelFormatRGBA16Float;
                    d.transparencyOverlayTextureEnabled = NO;
                    d.requiresSynchronousInitialization = synchronous;
                    d.autoExposureEnabled = autoExposure;
                    d.reactiveMaskTextureEnabled = reactive;
                    d.reactiveMaskTextureFormat = MTLPixelFormatR8Unorm;
                    if (metal3)
                        effect = [d newTemporalDenoisedScalerWithDevice:device];
                    else
                        effect = [d newTemporalDenoisedScalerWithDevice:device compiler:compiler];
                    if (reset)
                        [(id<MTLFXTemporalDenoisedScalerBase>)effect setShouldResetHistory:YES];
                    [d release];
                } else {
                    MTLFXSpatialScalerDescriptor *d = [MTLFXSpatialScalerDescriptor new];
                    d.inputWidth = d.outputWidth = width;
                    d.inputHeight = d.outputHeight = height;
                    d.colorTextureFormat = d.outputTextureFormat = MTLPixelFormatRGBA16Float;
                    d.colorProcessingMode = MTLFXSpatialScalerColorProcessingModeHDR;
                    effect = [d newSpatialScalerWithDevice:device compiler:compiler];
                    [d release];
                }
                if (effect) {
                    if (!i)
                        std::printf("CLASS %s\n", NSStringFromClass([effect class]).UTF8String);
                    if (monitor)
                        [weak addObject:effect];
                    [effect release];
                } else
                    ++failed;
            }
            // The sampling pool drains before the next creation: the NSArray
            // returned by allObjects must not extend an observed lifetime.
            std::printf("SAMPLE released=%d live=%ld allocated_delta=%lld\n", i + 1,
                        monitor ? static_cast<long>(liveCount(weak)) : -1L,
                        static_cast<long long>(device.currentAllocatedSize) - static_cast<long long>(before));
        }
        [compiler release];
        [device release];
    }
    @autoreleasepool {
        [[NSRunLoop currentRunLoop] runUntilDate:[NSDate dateWithTimeIntervalSinceNow:2]];
    }
    const NSUInteger live = liveCount(weak);
    std::printf("FINAL created=%d failed=%d live=%ld\n", count, failed, monitor ? static_cast<long>(live) : -1L);
    [weak release];
    return failed ? 77 : monitor && live ? 1 : 0;
}
