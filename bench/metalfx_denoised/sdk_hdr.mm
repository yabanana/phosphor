// Public SDK radiometry reduction. No Phosphor headers, pack shader or private ABI.
// clang++ -std=c++20 -fobjc-arc -framework Foundation -framework Metal -framework MetalFX sdk_hdr.mm -o sdk_hdr
// Run --constant first, then --wide, under API/shader validation. Exit 1 is a numeric failure.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

int main(int argc,char** argv) {
    if(argc!=2||(strcmp(argv[1],"--constant")&&strcmp(argv[1],"--wide")))return 2;
    const bool wide=!strcmp(argv[1],"--wide");
    @autoreleasepool {
        id<MTLDevice> dev=MTLCreateSystemDefaultDevice();
        NSError* error=nil;
        auto compiler=[dev newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
        if(!compiler){NSLog(@"%@",error);return 2;}
        auto d=[MTLFXTemporalDenoisedScalerDescriptor new];
        d.inputWidth=d.outputWidth=128;d.inputHeight=d.outputHeight=96;
        d.colorTextureFormat=d.outputTextureFormat=d.normalTextureFormat=d.diffuseAlbedoTextureFormat=d.specularAlbedoTextureFormat=MTLPixelFormatRGBA16Float;
        d.depthTextureFormat=MTLPixelFormatDepth32Float;d.motionTextureFormat=MTLPixelFormatRG32Float;
        d.roughnessTextureFormat=MTLPixelFormatR16Float;
        d.reactiveMaskTextureEnabled=YES;d.reactiveMaskTextureFormat=MTLPixelFormatR8Unorm;
        d.denoiseStrengthMaskTextureEnabled=YES;d.denoiseStrengthMaskTextureFormat=MTLPixelFormatR8Unorm;
        d.specularHitDistanceTextureEnabled=NO;d.specularHitDistanceTextureFormat=MTLPixelFormatR32Float;
        d.transparencyOverlayTextureEnabled=NO;d.autoExposureEnabled=NO;d.requiresSynchronousInitialization=YES;
        auto scaler=[d newTemporalDenoisedScalerWithDevice:dev compiler:compiler];
        if(!scaler)return 2;
        const MTLPixelFormat formats[]={MTLPixelFormatRGBA16Float,MTLPixelFormatDepth32Float,MTLPixelFormatRG32Float,
            MTLPixelFormatRGBA16Float,MTLPixelFormatRGBA16Float,MTLPixelFormatRGBA16Float,MTLPixelFormatR16Float,
            MTLPixelFormatR8Unorm,MTLPixelFormatR8Unorm,MTLPixelFormatR16Float,MTLPixelFormatRGBA16Float};
        id<MTLTexture> t[11];
        auto residency=[dev newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:&error];
        if(!residency)return 2;
        for(unsigned i=0;i<11;++i){
            auto td=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:formats[i] width:i==9?1:128 height:i==9?1:96 mipmapped:NO];
            td.storageMode=MTLStorageModePrivate;
            td.usage=MTLTextureUsageShaderRead|MTLTextureUsageRenderTarget;
            if(i!=1)td.usage|=MTLTextureUsageShaderWrite;
            t[i]=[dev newTextureWithDescriptor:td];
            if(!t[i])return 2;
            [residency addAllocation:t[i]];
        }
        auto readback=[dev newBufferWithLength:128*96*8 options:MTLResourceStorageModeShared];
        [residency addAllocation:readback];[residency commit];
        auto queue=[dev newMTL4CommandQueue];[queue addResidencySet:residency];
        auto allocator=[dev newCommandAllocator];auto cb=[dev newCommandBuffer];auto event=[dev newSharedEvent];
        id<MTL4CommandBuffer> batch[]={cb};uint64_t serial=0;
        auto submit=[&](){
            [cb endCommandBuffer];[queue commit:batch count:1];[queue signalEvent:event value:++serial];
            if(![event waitUntilSignaledValue:serial timeoutMS:120000])return false;
            [allocator reset];return true;
        };
        const float q=wide?1.f/64.f:1.f;
        const float expected[]={wide?368640.f:.5f,wide?128.f:.25f,wide?64.f:.125f};
        [cb beginCommandBufferWithAllocator:allocator];
        for(unsigned i=0;i<10;++i){
            auto rp=[MTL4RenderPassDescriptor new];
            if(i==1){rp.depthAttachment.texture=t[i];rp.depthAttachment.loadAction=MTLLoadActionClear;
                rp.depthAttachment.storeAction=MTLStoreActionStore;rp.depthAttachment.clearDepth=.025f;}
            else{
                auto a=rp.colorAttachments[0];a.texture=t[i];a.loadAction=MTLLoadActionClear;a.storeAction=MTLStoreActionStore;
                if(i==0)a.clearColor=MTLClearColorMake(expected[0]*q,expected[1]*q,expected[2]*q,1);
                else if(i==3)a.clearColor=MTLClearColorMake(.599609375,.599609375,.599609375,1);
                else if(i==4)a.clearColor=MTLClearColorMake(.03997802734375,.03997802734375,.03997802734375,1);
                else if(i==5)a.clearColor=MTLClearColorMake(0,0,1,0);
                else if(i==6)a.clearColor=MTLClearColorMake(.5,0,0,0);
                else if(i==9)a.clearColor=MTLClearColorMake(wide?.00008678436279296875:1.,0,0,0);
                else a.clearColor=MTLClearColorMake(0,0,0,0);
            }
            auto e=[cb renderCommandEncoderWithDescriptor:rp];[e endEncoding];
        }
        if(!submit())return 2;
        // Verify the actual packed color before the SDK consumes it.
        [cb beginCommandBufferWithAllocator:allocator];
        auto inputCopy=[cb computeCommandEncoder];
        [inputCopy copyFromTexture:t[0] sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0)
            sourceSize:MTLSizeMake(128,96,1) toBuffer:readback destinationOffset:0 destinationBytesPerRow:128*8 destinationBytesPerImage:0];
        [inputCopy endEncoding];if(!submit())return 2;
        const auto* input=static_cast<const _Float16*>(readback.contents);
        for(unsigned pixel=0;pixel<128*96;++pixel)for(unsigned c=0;c<3;++c)
            if(float(input[pixel*4+c])!=expected[c]*q)return 2;
        printf("{\"packed_color\":[%.9g,%.9g,%.9g],\"pre_exposure\":%.9g,\"input_readback_passed\":true}\n",
               float(input[0]),float(input[1]),float(input[2]),q);
        scaler.colorTexture=t[0];scaler.depthTexture=t[1];scaler.motionTexture=t[2];
        scaler.diffuseAlbedoTexture=t[3];scaler.specularAlbedoTexture=t[4];scaler.normalTexture=t[5];scaler.roughnessTexture=t[6];
        scaler.reactiveMaskTexture=t[7];scaler.denoiseStrengthMaskTexture=t[8];scaler.exposureTexture=t[9];scaler.outputTexture=t[10];
        scaler.preExposure=q;scaler.motionVectorScaleX=scaler.motionVectorScaleY=1;scaler.depthReversed=YES;
        scaler.jitterOffsetX=scaler.jitterOffsetY=0;
        if(scaler.preExposure!=q)return 2;
        scaler.worldToViewMatrix=matrix_identity_float4x4;
        const float f=1.7320508075688772f;
        matrix_float4x4 projection{};
        projection.columns[0]={f/(128.f/96.f),0,0,0};projection.columns[1]={0,f,0,0};
        projection.columns[2]={0,0,0,-1};projection.columns[3]={0,0,.1f,0};
        scaler.viewToClipMatrix=projection;
        double worst=0;unsigned steady=0;bool finite=true;
        for(unsigned frame=0;frame<96;++frame){
            [cb beginCommandBufferWithAllocator:allocator];scaler.shouldResetHistory=frame%48==0;
            [scaler encodeToCommandBuffer:cb];if(!submit())return 2;
            [cb beginCommandBufferWithAllocator:allocator];
            auto e=[cb computeCommandEncoder];
            [e copyFromTexture:t[10] sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0)
                sourceSize:MTLSizeMake(128,96,1) toBuffer:readback destinationOffset:0 destinationBytesPerRow:128*8 destinationBytesPerImage:0];
            [e endEncoding];if(!submit())return 2;
            const auto* pixels=static_cast<const _Float16*>(readback.contents);
            double error=0;
            for(unsigned pixel=0;pixel<128*96;++pixel)for(unsigned c=0;c<3;++c){
                const double value=double(pixels[pixel*4+c])/q;
                finite=finite&&std::isfinite(value)&&value>=0;
                error=std::max(error,std::abs(value-expected[c])/expected[c]);
            }
            const bool stable=frame%48>=8;
            if(stable){++steady;worst=std::max(worst,error);}
            printf("{\"frame\":%u,\"steady\":%s,\"restored_relative_error\":%.12g}\n",frame,stable?"true":"false",error);
        }
        const bool passed=finite&&steady==80&&worst<=.01;
        printf("{\"case\":\"%s\",\"encoded_frames\":96,\"steady_frames\":%u,\"max_steady_error\":%.12g,\"passed\":%s,\"phase_accepted\":false}\n",wide?"wide":"constant",steady,worst,passed?"true":"false");
        [queue removeResidencySet:residency];
        return passed?0:1;
    }
}
