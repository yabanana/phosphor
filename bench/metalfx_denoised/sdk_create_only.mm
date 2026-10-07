// Standalone public-API Metal4FX ownership probe. Creation-only reduction.
// No Phosphor headers, private ivars, retain-count logic, F8 workaround,
// textures, command queues, command buffers, fences or encode calls.
// All N scalers share one MTL4Compiler and exist concurrently before release.
// One worker serializes factories and ordinary ARC releases; joining
// the worker drains every creation/release autorelease pool before teardown.
//
// Build using the local SDK:
// /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++ \
//   -std=c++20 -O0 -g -fobjc-arc -fblocks -pthread -mmacosx-version-min=27.0 \
//   -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk \
//   -framework Foundation -framework Metal -framework MetalFX \
//   sdk_create_only.mm -o sdk_create_only
// Run only by the GPU/SDK coordinator, in separate processes:
//   leaks --atExit -- ./sdk_create_only --count 1
//   leaks --atExit -- ./sdk_create_only --count 4

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <Metal/MTL4Compiler.h>
#import <MetalFX/MetalFX.h>
#import <MetalFX/MTL4FXTemporalDenoisedScaler.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>

struct Slot {
    id<MTL4FXTemporalDenoisedScaler> __strong owner = nil;
    id<MTL4FXTemporalDenoisedScaler> __weak witness = nil;
    bool created = false;
    bool weakAliveAfterCreation = false;
    bool weakNilAfterWorkerRelease = false;
    bool weakNilAfterTeardown = false;
    double creationMilliseconds = 0;
};

// Exact descriptor fields from the constant fixture's public SDK adapter.
// Generated source color is packed to RGBA16Float before SDK use in the
// renderer; these are SDK-facing formats, not generated-source formats.
static MTLFXTemporalDenoisedScalerDescriptor* descriptor() {
    MTLFXTemporalDenoisedScalerDescriptor* d = [MTLFXTemporalDenoisedScalerDescriptor new];
    d.inputWidth = d.outputWidth = 128;
    d.inputHeight = d.outputHeight = 96;
    d.colorTextureFormat = MTLPixelFormatRGBA16Float;
    d.depthTextureFormat = MTLPixelFormatDepth32Float;
    d.motionTextureFormat = MTLPixelFormatRG32Float;
    d.diffuseAlbedoTextureFormat = MTLPixelFormatRGBA16Float;
    d.specularAlbedoTextureFormat = MTLPixelFormatRGBA16Float;
    d.normalTextureFormat = MTLPixelFormatRGBA16Float;
    d.roughnessTextureFormat = MTLPixelFormatR16Float;
    d.outputTextureFormat = MTLPixelFormatRGBA16Float;
    d.reactiveMaskTextureEnabled = YES;
    d.reactiveMaskTextureFormat = MTLPixelFormatR8Unorm;
    d.specularHitDistanceTextureEnabled = NO;
    d.specularHitDistanceTextureFormat = MTLPixelFormatR32Float;
    d.denoiseStrengthMaskTextureEnabled = YES;
    d.denoiseStrengthMaskTextureFormat = MTLPixelFormatR8Unorm;
    d.transparencyOverlayTextureEnabled = NO;
    d.autoExposureEnabled = NO;
    d.requiresSynchronousInitialization = YES;
    return d;
}

static const char* jsonBool(bool value) { return value ? "true" : "false"; }

int main(int argc, char** argv) {
    if (argc != 3 || std::strcmp(argv[1], "--count") != 0 ||
        (std::strcmp(argv[2], "1") != 0 && std::strcmp(argv[2], "4") != 0)) {
        std::fprintf(stderr, "usage: sdk_create_only --count 1|4\n");
        return 2;
    }
    const unsigned count = unsigned(argv[2][0] - '0');
    std::array<Slot, 4> slots{};
    unsigned created = 0;
    bool supported = false, compilerCreated = false;
    bool descriptorMatched = true, workerException = false, workerJoined = false;
    long osMajor = 0, osMinor = 0, osPatch = 0;

    @autoreleasepool {
        const NSOperatingSystemVersion version = NSProcessInfo.processInfo.operatingSystemVersion;
        osMajor = version.majorVersion; osMinor = version.minorVersion; osPatch = version.patchVersion;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        supported = device && [MTLFXTemporalDenoisedScalerDescriptor supportsDevice:device] &&
                    [MTLFXTemporalDenoisedScalerDescriptor supportsMetal4FX:device];
        id<MTL4Compiler> compiler = nil;
        if (supported) {
            @autoreleasepool {
                MTL4CompilerDescriptor* d = [MTL4CompilerDescriptor new];
                d.label = @"Phosphor public create-only shared compiler";
                NSError* error = nil;
                compiler = [device newCompilerWithDescriptor:d error:&error];
                compilerCreated = compiler != nil;
                if (!compiler) std::fprintf(stderr, "compiler creation failed: %s\n", error ? error.localizedDescription.UTF8String : "no NSError");
            }
        }
        if (compilerCreated) {
            std::thread worker([&] {
                @try {
                    for (unsigned i = 0; i < count; ++i) {
                        const auto start = std::chrono::steady_clock::now();
                        @autoreleasepool {
                            MTLFXTemporalDenoisedScalerDescriptor* d = descriptor();
                            descriptorMatched = descriptorMatched && !d.autoExposureEnabled &&
                                d.reactiveMaskTextureEnabled && !d.specularHitDistanceTextureEnabled &&
                                d.denoiseStrengthMaskTextureEnabled && !d.transparencyOverlayTextureEnabled &&
                                d.requiresSynchronousInitialization;
                            // ARC owns the factory's +1 result. The slot keeps
                            // the only probe-owned strong reference after this
                            // creation scope and its autorelease pool drain.
                            id<MTL4FXTemporalDenoisedScaler> value =
                                [d newTemporalDenoisedScalerWithDevice:device compiler:compiler];
                            slots[i].owner = value;
                            slots[i].witness = value;
                            slots[i].created = value != nil;
                            created += slots[i].created;
                        }
                        slots[i].creationMilliseconds = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - start).count();
                        // A weak load may temporarily retain. Its entire scope
                        // and pool finish before any release observation.
                        @autoreleasepool { slots[i].weakAliveAfterCreation = slots[i].witness != nil; }
                    }
                } @catch (NSException* exception) {
                    workerException = true;
                    std::fprintf(stderr, "SDK exception: %s: %s\n", exception.name.UTF8String, exception.reason ? exception.reason.UTF8String : "no reason");
                }
                // Always release successfully created wrappers, even if a
                // later construction failed. No engine retirement scheduler.
                for (unsigned i = 0; i < count; ++i) {
                    @autoreleasepool { slots[i].owner = nil; }
                    @autoreleasepool { slots[i].weakNilAfterWorkerRelease = slots[i].witness == nil; }
                }
            });
            worker.join(); // Every serial creation/release and pool has drained.
            workerJoined = true;
        }
        compiler = nil;
        device = nil;
    } // Descriptor, compiler, device and main autorelease pools have drained.

    bool allAfterWorker = true, allAfterTeardown = true, allObservedAlive = true;
    @autoreleasepool {
        for (unsigned i = 0; i < count; ++i) {
            slots[i].weakNilAfterTeardown = slots[i].witness == nil;
            allObservedAlive &= slots[i].created && slots[i].weakAliveAfterCreation;
            allAfterWorker &= slots[i].created && slots[i].weakNilAfterWorkerRelease;
            allAfterTeardown &= slots[i].created && slots[i].weakNilAfterTeardown;
        }
    }
    const bool passed = supported && compilerCreated && created == count && descriptorMatched &&
                        !workerException && allObservedAlive && allAfterTeardown;
    std::printf("{\"schema\":\"phosphor.sdk-create-only.v1\",\"count\":%u,\"created\":%u,"
                "\"gpu_encodes\":0,\"shared_compilers\":%u,\"factory_serialized\":true,"
                "\"requires_synchronous_initialization\":true,\"extent\":[128,96,128,96],"
                "\"os_version\":[%ld,%ld,%ld],\"supported\":%s,\"compiler_created\":%s,"
                "\"descriptor_matched\":%s,\"worker_exception\":%s,\"worker_joined\":%s,"
                "\"weak_alive_after_creation\":%s,\"weak_nil_after_worker_release\":%s,"
                "\"weak_nil_after_teardown\":%s,\"objects\":[",
                count, created, compilerCreated ? 1u : 0u, osMajor, osMinor, osPatch, jsonBool(supported), jsonBool(compilerCreated),
                jsonBool(descriptorMatched), jsonBool(workerException), jsonBool(workerJoined), jsonBool(allObservedAlive),
                jsonBool(allAfterWorker), jsonBool(allAfterTeardown));
    for (unsigned i = 0; i < count; ++i) {
        std::printf("%s{\"index\":%u,\"created\":%s,\"create_ms\":%.6f,"
                    "\"weak_alive_after_creation\":%s,\"weak_nil_after_worker_release\":%s,"
                    "\"weak_nil_after_teardown\":%s}",
                    i ? "," : "", i, jsonBool(slots[i].created), slots[i].creationMilliseconds,
                    jsonBool(slots[i].weakAliveAfterCreation), jsonBool(slots[i].weakNilAfterWorkerRelease),
                    jsonBool(slots[i].weakNilAfterTeardown));
    }
    std::printf("],\"public_object_lifetime_passed\":%s,\"leak_free_proven\":false}\n", jsonBool(passed));
    const int code = passed ? 0 : 1;
    std::printf("EXIT %d\n", code);std::fflush(nullptr);
    return code;
}
