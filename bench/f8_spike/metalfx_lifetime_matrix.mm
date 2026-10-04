// F8.4 public-API lifetime comparison. No render graph or scaler encoding.
// --gpu-drain submits a checked blit before/after destruction to test deferred
// driver reclamation separately from a framework-owned reference cycle.
// --release-cycle releases standard temporal scalers through the engine's
// ownership record (src/platform/metal/metalfx_lifetime.cpp); build it with
// that file and metal_impl.cpp (see README). Without it: plain release.
// Exit 0: probe ran without an observed live target (not a leak-scan verdict);
// --no-weak skips that observation and must be paired with a monitored run.
// Exit 1: live target; 2: invalid arguments;
// 77: unsupported/failed creation. Device allocation includes driver pools.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#if defined(PHOSPHOR_RELEASE_CYCLE)
#include "platform/metal/metalfx_lifetime.h"
#endif
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
static void drainGpu(id<MTLDevice> device, id<MTLCommandQueue> queue) {
    @autoreleasepool {
        id<MTLBuffer> buffer = [device newBufferWithLength:16 options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> commands = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [commands blitCommandEncoder];
        if (!buffer || !commands || !blit)
            std::exit(77);
        [blit fillBuffer:buffer range:NSMakeRange(0, 16) value:0x5a];
        [blit endEncoding];
        [commands commit];
        [commands waitUntilCompleted]; // External runner bounds process lifetime.
        if (commands.status != MTLCommandBufferStatusCompleted ||
            static_cast<const unsigned char *>(buffer.contents)[0] != 0x5a)
            std::exit(78);
        [buffer release];
    }
}
int main(int argc, char **argv) {
    std::string mode = "temporal4";
    int count = 8, width = 640, height = 360;
    bool dynamic = true, reactive = true, synchronous = true;
    bool autoExposure = false, reset = false, monitor = true;
    bool gpuDrain = false, depthR32 = false, releaseCycle = false;
    int outputScale = 1;
    std::string format = "rgba16";
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
        else if (arg == "--gpu-drain")
            gpuDrain = true;
        else if (arg == "--depth-r32")
            depthR32 = true;
        else if (arg == "--release-cycle")
            releaseCycle = true;
        else if (i + 1 < argc && arg == "--output-scale")
            outputScale = number(argv[++i], 1, 3);
        else if (i + 1 < argc && arg == "--format")
            format = argv[++i];
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
#if !defined(PHOSPHOR_RELEASE_CYCLE)
    if (releaseCycle) {
        std::fprintf(stderr, "--release-cycle needs a build with metalfx_lifetime.cpp\n");
        return 2;
    }
#endif
    if (releaseCycle && !temporal)
        return 2;
    if (width * outputScale > 1920 || height * outputScale > 1080 ||
        (format != "rgba16" && format != "rgba8" && format != "rg11") ||
        (!temporal && (format != "rgba16" || depthR32 || outputScale != 1)))
        return 2;
    const MTLPixelFormat colorFormat = format == "rgba8" ? MTLPixelFormatRGBA8Unorm
                                       : format == "rg11" ? MTLPixelFormatRG11B10Float
                                                          : MTLPixelFormatRGBA16Float;
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
        id<MTLCommandQueue> drainQueue = gpuDrain ? [device newCommandQueue] : nil;
        if (gpuDrain) {
            if (!drainQueue)
                return 77;
            drainGpu(device, drainQueue);
        }
        NSBundle *framework = [NSBundle bundleForClass:[MTLFXTemporalScalerDescriptor class]];
        std::printf(
            "CONFIG mode=%s count=%d size=%dx%d dynamic=%d reactive=%d sync=%d auto_exposure=%d reset=%d weak=%d\n",
            mode.c_str(), count, width, height, dynamic, reactive, synchronous, autoExposure, reset, monitor);
        std::printf("FOLLOWUP output_scale=%d format=%s depth_r32=%d gpu_drain=%d release_cycle=%d\n", outputScale,
                    format.c_str(), depthR32, gpuDrain, releaseCycle);
        std::printf("DEVICE %s | RUNTIME %s | FRAMEWORK %s\n", device.name.UTF8String,
                    [[NSProcessInfo processInfo] operatingSystemVersionString].UTF8String,
                    [[framework objectForInfoDictionaryKey:@"CFBundleVersion"] UTF8String]);
        const NSUInteger before = device.currentAllocatedSize;
        for (int i = 0; i < count; ++i) {
            @autoreleasepool {
                id effect = nil;
                if (temporal) {
                    MTLFXTemporalScalerDescriptor *d = [MTLFXTemporalScalerDescriptor new];
                    d.inputWidth = width;
                    d.inputHeight = height;
                    d.outputWidth = width * outputScale;
                    d.outputHeight = height * outputScale;
                    d.colorTextureFormat = d.outputTextureFormat = colorFormat;
                    d.depthTextureFormat = depthR32 ? MTLPixelFormatR32Float : MTLPixelFormatDepth32Float;
                    d.motionTextureFormat = MTLPixelFormatRG16Float;
                    d.requiresSynchronousInitialization = synchronous;
                    d.autoExposureEnabled = autoExposure;
                    d.reactiveMaskTextureEnabled = reactive;
                    d.reactiveMaskTextureFormat = MTLPixelFormatR8Unorm;
                    d.inputContentPropertiesEnabled = dynamic;
                    d.inputContentMinScale = 1;
                    d.inputContentMaxScale = outputScale > 2 ? outputScale : 2;
                    @autoreleasepool {
                        if (metal3)
                            effect = [d newTemporalScalerWithDevice:device];
                        else
                            effect = [d newTemporalScalerWithDevice:device compiler:compiler];
                    }
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
#if defined(PHOSPHOR_RELEASE_CYCLE)
                    // Adopt before the monitor: NSHashTable insertion can leave
                    // an autoreleased reference (measured: 11 of 100 scalers).
                    std::shared_ptr<MTL4FX::TemporalScaler> owner;
                    if (releaseCycle) // same object model for the Metal 3 and Metal 4 scalers
                        owner = phosphor::metalfx::adoptTemporalScaler(reinterpret_cast<MTL4FX::TemporalScaler *>(effect));
#endif
                    if (monitor) {
                        @autoreleasepool {
                            [weak addObject:effect];
                        }
                    }
#if defined(PHOSPHOR_RELEASE_CYCLE)
                    if (releaseCycle)
                        owner.reset();
                    else
#endif
                        [effect release];
                } else
                    ++failed;
            }
            if (gpuDrain)
                drainGpu(device, drainQueue);
            // The sampling pool drains before the next creation: the NSArray
            // returned by allObjects must not extend an observed lifetime.
            std::printf("SAMPLE released=%d live=%ld allocated_delta=%lld\n", i + 1,
                        monitor ? static_cast<long>(liveCount(weak)) : -1L,
                        static_cast<long long>(device.currentAllocatedSize) - static_cast<long long>(before));
        }
        [drainQueue release];
        [compiler release];
        [device release];
    }
    @autoreleasepool {
        [[NSRunLoop currentRunLoop] runUntilDate:[NSDate dateWithTimeIntervalSinceNow:2]];
    }
    const NSUInteger live = liveCount(weak);
    std::printf("FINAL created=%d failed=%d live=%ld\n", count, failed, monitor ? static_cast<long>(live) : -1L);
#if defined(PHOSPHOR_RELEASE_CYCLE)
    if (releaseCycle) {
        const auto fx = phosphor::metalfx::counters();
        std::printf("RELEASE_CYCLE adopted=%llu released=%llu cycle_released=%llu retained=%llu unknown=%llu\n",
                    static_cast<unsigned long long>(fx.adopted), static_cast<unsigned long long>(fx.released),
                    static_cast<unsigned long long>(fx.cycleReleased), static_cast<unsigned long long>(fx.retained),
                    static_cast<unsigned long long>(fx.unknownSignature));
    }
#endif
    [weak release];
    return failed ? 77 : monitor && live ? 1 : 0;
}
