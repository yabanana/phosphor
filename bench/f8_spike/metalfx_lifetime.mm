// Public-API reduction for the macOS 27.2 MetalFX lifetime residual.
// No renderer, render graph, textures, worker threads or GPU submissions.
// Build/run instructions and interpretation: README.md in this directory.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#include <cstdio>

int main() {
    NSHashTable *lifetimes = [[NSHashTable alloc] initWithOptions:NSHashTableWeakMemory capacity:1];
    int status = 0;
    @autoreleasepool {
        NSObject *control = [NSObject new];
        [lifetimes addObject:control];
        [control release];
    }
    @autoreleasepool {
        if ([lifetimes allObjects].count != 0) return 3;
    }
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        MTL4CompilerDescriptor *compilerDescriptor = [MTL4CompilerDescriptor new];
        NSError *error = nil;
        id<MTL4Compiler> compiler = [device newCompilerWithDescriptor:compilerDescriptor error:&error];
        [compilerDescriptor release];
        if (!compiler || ![MTLFXTemporalScalerDescriptor supportsMetal4FX:device]) {
            status = 77;
        } else {
            MTLFXTemporalScalerDescriptor *descriptor = [MTLFXTemporalScalerDescriptor new];
            descriptor.inputWidth = descriptor.outputWidth = 640;
            descriptor.inputHeight = descriptor.outputHeight = 360;
            descriptor.colorTextureFormat = descriptor.outputTextureFormat = MTLPixelFormatRGBA16Float;
            descriptor.depthTextureFormat = MTLPixelFormatDepth32Float;
            descriptor.motionTextureFormat = MTLPixelFormatRG16Float;
            descriptor.requiresSynchronousInitialization = YES;
            descriptor.autoExposureEnabled = NO;
            descriptor.reactiveMaskTextureEnabled = YES;
            descriptor.reactiveMaskTextureFormat = MTLPixelFormatR8Unorm;
            descriptor.inputContentPropertiesEnabled = YES;
            descriptor.inputContentMinScale = 1;
            descriptor.inputContentMaxScale = 2;
            id<MTL4FXTemporalScaler> scaler = [descriptor newTemporalScalerWithDevice:device compiler:compiler];
            if (!scaler)
                status = 2;
            else {
                [lifetimes addObject:scaler];
                [scaler release]; // balances the public new... factory's ownership
            }
            [descriptor release];
        }
        [compiler release];
        [device release];
    }
    // Allow framework completion work to drain, then check a zeroing weak
    // reference. This avoids mistaking a conservative leaks scan for proof
    // that the scaler was destroyed.
    @autoreleasepool {
        [[NSRunLoop currentRunLoop] runUntilDate:[NSDate dateWithTimeIntervalSinceNow:2]];
        const unsigned long live = [lifetimes allObjects].count;
        std::printf("METALFX-LIFETIME live scalers after release/pool drain: %lu | %s\n", live,
                    status ? "UNAVAILABLE"
                    : live ? "FAIL"
                           : "PASS");
        if (!status && live)
            status = 1;
    }
    [lifetimes release];
    return status;
}
