// Minimal ARC reproduction for an Apple Feedback report (see APPLE_FEEDBACK.md).
// Creates and releases MetalFX scalers; nothing in this program keeps them.
//   clang -fobjc-arc -framework Foundation -framework Metal -framework MetalFX \
//         apple_feedback_repro.m -o metalfx_temporal_leak && ./metalfx_temporal_leak
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>

static MTLFXTemporalScalerDescriptor *temporalDescriptor(void) {
    MTLFXTemporalScalerDescriptor *d = [MTLFXTemporalScalerDescriptor new];
    d.inputWidth = d.outputWidth = 640;
    d.inputHeight = d.outputHeight = 360;
    d.colorTextureFormat = d.outputTextureFormat = MTLPixelFormatRGBA16Float;
    d.depthTextureFormat = MTLPixelFormatDepth32Float;
    d.motionTextureFormat = MTLPixelFormatRG16Float;
    return d;
}

// Returns the number of scalers still alive after `count` create/release cycles.
static NSUInteger run(id<MTLDevice> device, id<MTL4Compiler> compiler, NSString *kind, int count) {
    NSHashTable *alive = [NSHashTable weakObjectsHashTable];
    const NSUInteger before = device.currentAllocatedSize;
    for (int i = 0; i < count; ++i) {
        @autoreleasepool {
            id scaler = nil;
            if ([kind isEqualToString:@"temporal (Metal 4)"])
                scaler = [temporalDescriptor() newTemporalScalerWithDevice:device compiler:compiler];
            else if ([kind isEqualToString:@"temporal (Metal 3)"])
                scaler = [temporalDescriptor() newTemporalScalerWithDevice:device];
            else {
                MTLFXSpatialScalerDescriptor *d = [MTLFXSpatialScalerDescriptor new];
                d.inputWidth = d.outputWidth = 640;
                d.inputHeight = d.outputHeight = 360;
                d.colorTextureFormat = d.outputTextureFormat = MTLPixelFormatRGBA16Float;
                scaler = [d newSpatialScalerWithDevice:device compiler:compiler];
            }
            if (i == 0)
                printf("%-20s class %s, retain count right after creation: %ld (one owner)\n", kind.UTF8String,
                       NSStringFromClass([scaler class]).UTF8String, (long)CFGetRetainCount((__bridge CFTypeRef)scaler));
            [alive addObject:scaler];
            scaler = nil; // the only strong reference held by this program
        }
    }
    @autoreleasepool {
        [[NSRunLoop currentRunLoop] runUntilDate:[NSDate dateWithTimeIntervalSinceNow:2]];
    }
    NSUInteger live = 0;
    @autoreleasepool {
        live = alive.allObjects.count;
    }
    printf("%-20s %d created and released -> %lu still alive, device allocation +%lld bytes\n", kind.UTF8String, count,
           (unsigned long)live, (long long)device.currentAllocatedSize - (long long)before);
    return live;
}

int main(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;
        id<MTL4Compiler> compiler = [device newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
        NSBundle *metalfx = [NSBundle bundleForClass:[MTLFXTemporalScalerDescriptor class]];
        printf("%s | %s | MetalFX %s\n", device.name.UTF8String,
               [NSProcessInfo processInfo].operatingSystemVersionString.UTF8String,
               [[metalfx objectForInfoDictionaryKey:@"CFBundleVersion"] UTF8String]);
        if (!device || !compiler || ![MTLFXTemporalScalerDescriptor supportsMetal4FX:device])
            return 77;
        const NSUInteger temporal4 = run(device, compiler, @"temporal (Metal 4)", 8);
        const NSUInteger temporal3 = run(device, compiler, @"temporal (Metal 3)", 8);
        const NSUInteger spatial = run(device, compiler, @"spatial (control)", 8);
        printf("%s\n", temporal4 || temporal3 ? "LEAK: released temporal scalers stay alive"
                                              : "OK: every released scaler was deallocated");
        return temporal4 || temporal3 || spatial ? 1 : 0;
    }
}
