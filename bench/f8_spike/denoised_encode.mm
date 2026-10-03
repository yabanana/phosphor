// Public-API F8.4 denoised lifetime/extent probe. Not an engine backend.
// --mismatch requires Metal API validation and must abort before GPU encoding.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
static bool encode(id<MTLDevice> dev, id<MTL4FXTemporalDenoisedScaler> scaler, unsigned inputWidth) {
    const MTLPixelFormat formats[] = {MTLPixelFormatRGBA16Float, MTLPixelFormatDepth32Float, MTLPixelFormatRG16Float,
                                      MTLPixelFormatR8Unorm,     MTLPixelFormatRGBA16Float,  MTLPixelFormatR16Float,
                                      MTLPixelFormatRGBA16Float, MTLPixelFormatRGBA16Float,  MTLPixelFormatRGBA16Float,
                                      MTLPixelFormatR16Float,    MTLPixelFormatR16Float,     MTLPixelFormatR8Unorm};
    id<MTL4CommandQueue> q = [dev newMTL4CommandQueue];
    id<MTL4CommandAllocator> a = [dev newCommandAllocator];
    id<MTL4CommandBuffer> cb = [dev newCommandBuffer];
    id<MTLSharedEvent> done = [dev newSharedEvent];
    MTLResidencySetDescriptor *rd = [MTLResidencySetDescriptor new];
    id<MTLResidencySet> rs = [dev newResidencySetWithDescriptor:rd error:nil];
    [rd release];
    id<MTLTexture> t[12];
    for (unsigned i = 0; i < 12; ++i) {
        auto *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:formats[i]
                                                                     width:i == 5 ? 1 : (i == 4 ? 640 : inputWidth)
                                                                    height:i == 5 ? 1 : (i == 4 ? 360 : 180)
                                                                 mipmapped:NO];
        d.storageMode = MTLStorageModePrivate;
        d.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
        if (i != 1)
            d.usage |= MTLTextureUsageShaderWrite;
        t[i] = [dev newTextureWithDescriptor:d];
        [rs addAllocation:t[i]];
    }
    [rs commit];
    [q addResidencySet:rs];
    [cb beginCommandBufferWithAllocator:a];
    for (unsigned i = 0; i < 12; ++i) {
        auto *pass = [MTL4RenderPassDescriptor new];
        if (i == 1) {
            pass.depthAttachment.texture = t[i];
            pass.depthAttachment.loadAction = MTLLoadActionClear;
            pass.depthAttachment.storeAction = MTLStoreActionStore;
            pass.depthAttachment.clearDepth = .5;
        } else {
            auto *c = pass.colorAttachments[0];
            c.texture = t[i];
            c.loadAction = MTLLoadActionClear;
            c.storeAction = MTLStoreActionStore;
            c.clearColor = i == 0              ? MTLClearColorMake(.2, .3, .4, 1)
                           : i == 5 || i == 11 ? MTLClearColorMake(1, 1, 1, 1)
                           : i == 8            ? MTLClearColorMake(0, 0, 1, 1)
                           : i == 6            ? MTLClearColorMake(.5, .5, .5, 1)
                           : i == 7            ? MTLClearColorMake(.04, .04, .04, 1)
                           : i == 9            ? MTLClearColorMake(.5, 0, 0, 0)
                                               : MTLClearColorMake(0, 0, 0, 0);
        }
        id<MTL4RenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:pass];
        [enc endEncoding];
        [pass release];
    }
    [cb endCommandBuffer];
    id<MTL4CommandBuffer> batch[] = {cb};
    [q commit:batch count:1];
    [q signalEvent:done value:1];
    if (![done waitUntilSignaledValue:1 timeoutMS:10000])
        return false;
    [a reset];
    [cb beginCommandBufferWithAllocator:a];
    scaler.colorTexture = t[0];
    scaler.depthTexture = t[1];
    scaler.motionTexture = t[2];
    scaler.reactiveMaskTexture = t[3];
    scaler.outputTexture = t[4];
    scaler.exposureTexture = t[5];
    scaler.diffuseAlbedoTexture = t[6];
    scaler.specularAlbedoTexture = t[7];
    scaler.normalTexture = t[8];
    scaler.roughnessTexture = t[9];
    scaler.specularHitDistanceTexture = t[10];
    scaler.denoiseStrengthMaskTexture = t[11];
    scaler.preExposure = 1;
    scaler.motionVectorScaleX = scaler.motionVectorScaleY = 1;
    scaler.shouldResetHistory = YES;
    scaler.depthReversed = YES;
    scaler.worldToViewMatrix = matrix_identity_float4x4;
    scaler.viewToClipMatrix = matrix_identity_float4x4;
    [scaler encodeToCommandBuffer:cb];
    [cb endCommandBuffer];
    [q commit:batch count:1];
    [q signalEvent:done value:2];
    if (![done waitUntilSignaledValue:2 timeoutMS:10000])
        return false;
    [cb release];
    [a release];
    [q removeResidencySet:rs];
    [rs release];
    [q release];
    [done release];
    for (auto tex : t)
        [tex release];
    return true;
}
int main(int argc, char **argv) {
    const bool mismatch = argc == 2 && std::strcmp(argv[1], "--mismatch") == 0;
    if ((argc != 1 && !mismatch) || (mismatch && !std::getenv("MTL_DEBUG_LAYER"))) {
        std::fprintf(stderr, "Usage: denoised_encode [--mismatch with MTL_DEBUG_LAYER=1]\n");
        return 2;
    }
    NSHashTable *weak = [[NSHashTable alloc] initWithOptions:NSHashTableWeakMemory capacity:1];
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        auto *cd = [MTL4CompilerDescriptor new];
        id<MTL4Compiler> compiler = [dev newCompilerWithDescriptor:cd error:nil];
        [cd release];
        auto *d = [MTLFXTemporalDenoisedScalerDescriptor new];
        d.inputWidth = 320;
        d.inputHeight = 180;
        d.outputWidth = 640;
        d.outputHeight = 360;
        d.colorTextureFormat = d.outputTextureFormat = d.diffuseAlbedoTextureFormat = d.specularAlbedoTextureFormat =
            d.normalTextureFormat = MTLPixelFormatRGBA16Float;
        d.depthTextureFormat = MTLPixelFormatDepth32Float;
        d.motionTextureFormat = MTLPixelFormatRG16Float;
        d.roughnessTextureFormat = d.specularHitDistanceTextureFormat = MTLPixelFormatR16Float;
        d.denoiseStrengthMaskTextureFormat = MTLPixelFormatR8Unorm;
        d.transparencyOverlayTextureFormat = MTLPixelFormatRGBA16Float;
        d.transparencyOverlayTextureEnabled = NO;
        d.requiresSynchronousInitialization = YES;
        d.autoExposureEnabled = NO;
        d.reactiveMaskTextureEnabled = YES;
        d.reactiveMaskTextureFormat = MTLPixelFormatR8Unorm;
        id<MTL4FXTemporalDenoisedScaler> s = [d newTemporalDenoisedScalerWithDevice:dev compiler:compiler];
        [d release];
        [weak addObject:s];
        if (!encode(dev, s, mismatch ? 240 : 320))
            return 2;
        [s release];
        [compiler release];
        [dev release];
    }
    @autoreleasepool {
        [[NSRunLoop currentRunLoop] runUntilDate:[NSDate dateWithTimeIntervalSinceNow:2]];
    }
    NSUInteger live;
    @autoreleasepool {
        live = weak.allObjects.count;
    }
    printf("ENCODE FINAL live=%lu\n", (unsigned long)live);
    [weak release];
    return live ? 1 : 0;
}
