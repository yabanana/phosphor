// Frozen F9 frame37/ray162 regression; uses the production MSL refinement helper.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "renderer/gpu_types.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
using namespace phosphor;
int main(int argc,char** argv){
 if(argc!=2)return 2;
 @autoreleasepool{
 auto device=MTLCreateSystemDefaultDevice();NSError* error=nil;
 auto library=[device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];
 auto pipeline=library?[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"surface_refinement_fixture"] error:&error]:nil;
 if(!pipeline){NSLog(@"%@",error);return 2;}
 GPURtRay ray{};ray.ox=-.994360625f;ray.oy=3.42155766f;ray.oz=-2.0064826f;
 ray.dx=.654699445f;ray.dy=.564253271f;ray.dz=.502978027f;ray.tmax=1e30f;ray.type=RT_PROBE_DIFFUSE;ray.mask=4;
 GPURtHit hit{};hit.hit=1;hit.t=6.90264368f;hit.u=.0141772246f;hit.v=.215880752f;hit.generation=1;hit.frontFacing=1;
 std::array<GPURtRay,3> rays{ray,ray,ray};rays[1].type=RT_PROBE_SHADOW;
 std::array<GPURtHit,3> hits{hit,hit,hit};hits[2].hit=0;hits[2].t=-1;
 GPUInstance instance{};instance.modelMatrix[0]=instance.modelMatrix[5]=instance.modelMatrix[10]=.0080000003799796104f;
 instance.modelMatrix[15]=1;instance.generation=1;
 GPURtMesh mesh{};mesh.indexCount=3;
 std::array<GPUVertex,3> vertices{};
 vertices[0].px=872.412109375f;vertices[0].py=914.64410400390625f;vertices[0].pz=183.17149353027344f;
 vertices[1].px=-1004.5604858398438f;vertices[1].py=907.74407958984375f;vertices[1].pz=vertices[0].pz;
 vertices[2].px=vertices[1].px;vertices[2].py=vertices[0].py;vertices[2].pz=vertices[0].pz;
 std::array<unsigned,3> indices{0,1,2};
 const void* input[]={rays.data(),hits.data(),&instance,&mesh,vertices.data(),indices.data()};
 const size_t sizes[]={sizeof(rays),sizeof(hits),sizeof(instance),sizeof(mesh),sizeof(vertices),sizeof(indices)};
 id<MTLBuffer> buffers[7];
 for(unsigned i=0;i<6;++i)buffers[i]=[device newBufferWithBytes:input[i] length:sizes[i] options:MTLResourceStorageModeShared];
 buffers[6]=[device newBufferWithLength:sizeof(hits) options:MTLResourceStorageModeShared];
 auto queue=[device newCommandQueue];auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];
 [e setComputePipelineState:pipeline];for(unsigned i=0;i<7;++i)[e setBuffer:buffers[i] offset:0 atIndex:i];
 [e dispatchThreads:MTLSizeMake(3,1,1) threadsPerThreadgroup:MTLSizeMake(3,1,1)];[e endEncoding];[cb commit];[cb waitUntilCompleted];
 if(cb.error){NSLog(@"%@",cb.error);return 2;}
 const auto* out=static_cast<const GPURtHit*>(buffers[6].contents);
 auto residual=[&](GPURtHit h){
  double sum=0;
  for(unsigned i=0;i<3;++i){
   const float* a=&vertices[0].px;const float* b=&vertices[1].px;const float* c=&vertices[2].px;
   const float* o=&ray.ox;const float* d=&ray.dx;
   double p=double(instance.modelMatrix[0])*(double(a[i])+double(h.u)*(double(b[i])-a[i])+double(h.v)*(double(c[i])-a[i]));
   double error=p-(double(o[i])+double(h.t)*d[i]);sum+=error*error;
  }
  return std::sqrt(sum);
 };
 const double before=residual(hit),after=residual(out[0]),cap=.000031480132954463841;
 const bool controls=!std::memcmp(&out[1],&hits[1],sizeof(hit))&&!std::memcmp(&out[2],&hits[2],sizeof(hit));
 printf("{\"before\":%.12g,\"after\":%.12g,\"cap\":%.12g,\"controls\":%s}\n",before,after,cap,controls?"true":"false");
 return before>cap&&after<=cap&&controls?0:1;
 }
}
