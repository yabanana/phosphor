#include <metal_stdlib>
#include "renderer/gpu_types.h"
#define PHOSPHOR_RT_NO_ALPHA_FUNCTION 1
#include "rt_common.h"
#include "renderer/visibility_math.h"
using namespace metal;
using namespace phosphor;
static float4x4 guideMatrix(constant float* p) {
    return float4x4(float4(p[0],p[1],p[2],p[3]),float4(p[4],p[5],p[6],p[7]),
                    float4(p[8],p[9],p[10],p[11]),float4(p[12],p[13],p[14],p[15]));
}
kernel void shadow_receiver_clear(constant GPUShadowParams& p [[buffer(1)]],
                                   device GPUShadowCounters& counters [[buffer(15)]],
                                   texture2d<uint, access::write> keys [[texture(10)]], uint2 pixel [[thread_position_in_grid]]) {
    if (all(pixel == uint2(0))) counters = GPUShadowCounters{};
    if (all(pixel < uint2(p.width,p.height))) keys.write(uint4(~0u),pixel);
}
kernel void shadow_receiver_pack(constant GPUShadowParams& p [[buffer(1)]],
                                  device GPUShadowSurface* surfaces [[buffer(2)]],
                                  const device GPUInstance* instances [[buffer(3)]],
                                  const device GPUMeshInfo* meshes [[buffer(5)]],
                                  const device GPUVertex* vertices [[buffer(6)]],
                                  device atomic_uint* counters [[buffer(15)]],
                                  const device uint* indices [[buffer(18)]],
                                  texture2d<float,access::write> world [[texture(5)]],
                                  texture2d<float,access::write> normals [[texture(6)]],
                                  texture2d<uint,access::read> keys [[texture(10)]],
                                  depth2d<float,access::read> depth [[texture(0)]], uint2 pixel [[thread_position_in_grid]]) {
    if (any(pixel >= uint2(p.width,p.height))) return;
    const uint4 key = keys.read(pixel); if (key.x == ~0u) return;
    GPUShadowSurface s{};
    float3 point=0,normal=0;bool valid=key.x<p.slotCount;
    if(valid){const device GPUInstance& instance=instances[key.x];
        valid=(instance.flags&INSTANCE_FLAG_VALID)&&instance.generation==key.y&&instance.meshIndex<p.meshCount;
        if(valid){const GPUMeshInfo mesh=meshes[instance.meshIndex];valid=key.z<mesh.indexCount/3u;
            if(valid){
                // Indexed ICB primitive_id is local to its original draw. The
                // index stream is mesh-local; meshlet vertices are global.
                // Decode the same three GPU positions as the V-buffer path,
                // without fixed-function interpolation of world coordinates.
                const uint base=mesh.indexOffset+key.z*3u;
                const float3 w0=rtWorldPoint(instance,rtPosition(vertices[mesh.vertexOffset+indices[base]]));
                const float3 w1=rtWorldPoint(instance,rtPosition(vertices[mesh.vertexOffset+indices[base+1u]]));
                const float3 w2=rtWorldPoint(instance,rtPosition(vertices[mesh.vertexOffset+indices[base+2u]]));
                const float4x4 vp=guideMatrix(p.viewProjection);
                const float4 c0=vp*float4(w0,1),c1=vp*float4(w1,1),c2=vp*float4(w2,1);
                const auto bary=visibilityBarycentrics(c0.x,c0.y,c0.w,c1.x,c1.y,c1.w,c2.x,c2.y,c2.w,
                    float(pixel.x)+0.5f,float(pixel.y)+0.5f,float(p.width),float(p.height));
                const float3 crossNormal=cross(w1-w0,w2-w0);const float z=depth.read(pixel);
                valid=z>0&&z<=1&&bary.valid&&dot(crossNormal,crossNormal)>1e-30f;
                if(valid){point=w0*bary.value[0]+w1*bary.value[1]+w2*bary.value[2];normal=normalize(crossNormal);
                    if(dot(normal,float3(p.cameraPosition[0],p.cameraPosition[1],p.cameraPosition[2])-point)<0)normal=-normal;
                    s.viewDepth=-(guideMatrix(p.view)*float4(point,1)).z;s.reverseDepth=z;
                    valid=all(isfinite(point))&&all(isfinite(normal))&&isfinite(s.viewDepth)&&s.viewDepth>0;}
            }
        }
    }
    if(valid){for(uint c=0;c<3;++c){s.position[c]=point[c];s.geometricNormal[c]=normal[c];}
        s.slot=key.x;s.generation=key.y;s.primitive=key.z;s.valid=1;}
    else{point=normal=0;atomic_fetch_add_explicit(counters+7,1u,memory_order_relaxed);}
    surfaces[pixel.y*p.width+pixel.x]=s;
    // Publish the canonical guides to DI/F13 and captures as well as F10.
    world.write(float4(point,float(s.valid)),pixel);normals.write(float4(normal,float(s.valid)),pixel);
}
kernel void lighting_zero(texture2d<float,access::write> output [[texture(0)]], uint2 p [[thread_position_in_grid]]) {
    if(all(p<uint2(output.get_width(),output.get_height())))output.write(float4(0),p);
}

kernel void di_receiver_pack(constant FrameConstants& frame [[buffer(0)]],constant GPUDIParams& p [[buffer(1)]],
    device GPUDISurface* surfaces [[buffer(2)]],texture2d<float,access::read> world [[texture(5)]],
    texture2d<float,access::read> geometric [[texture(6)]],texture2d<uint,access::read> keys [[texture(10)]],
    texture2d<float,access::read> shading [[texture(11)]],texture2d<float,access::read> albedo [[texture(12)]],texture2d<float,access::read> occlusion [[texture(13)]],
    uint2 pixel [[thread_position_in_grid]]) {
    if(any(pixel>=uint2(p.width,p.height)))return;const uint4 key=keys.read(pixel);if(key.x==~0u)return;
    const float4 pos=world.read(pixel),g=geometric.read(pixel),s=shading.read(pixel),a=albedo.read(pixel);
    if(dot(s.xyz,s.xyz)<=0)return;
    GPUDISurface out{};const float3 v=normalize(float3(frame.cameraPosition[0],frame.cameraPosition[1],frame.cameraPosition[2])-pos.xyz);
    for(uint c=0;c<3;++c){out.position[c]=pos[c];out.geometricNormal[c]=normalize(g.xyz)[c];out.shadingNormal[c]=normalize(s.xyz)[c];out.albedo[c]=a[c];out.viewDirection[c]=v[c];}
    out.depth=-(guideMatrix(frame.view)*float4(pos.xyz,1)).z;out.roughness=s.w;out.metallic=a.w;
    out.materialRevision=p.historyEpoch;out.instanceSlot=key.x;out.instanceGeneration=key.y;out.pad[0]=as_type<uint>(occlusion.read(pixel).x);out.valid=pos.w>0 && g.w>0 && dot(s.xyz,s.xyz)>0;
    surfaces[pixel.y*p.width+pixel.x]=out;
}

// Independent F12 comparison signal before the engine's artistic AO/specular
// environment/exposure. This is reflected INDIRECT diffuse radiance, not E.
kernel void gi_reference_diffuse(constant GPUProbeGridParams& p [[buffer(0)]],
    const device GPUDISurface* surfaces [[buffer(1)]],texture2d<float,access::read> irradiance [[texture(0)]],
    texture2d<float,access::write> output [[texture(1)]],uint2 pixel [[thread_position_in_grid]]) {
    if(any(pixel>=uint2(p.width,p.height)))return;const GPUDISurface s=surfaces[pixel.y*p.width+pixel.x];
    const float3 albedo(s.albedo[0],s.albedo[1],s.albedo[2]);
    output.write(float4(s.valid?irradiance.read(pixel).xyz*albedo*(1-s.metallic)/M_PI_F:float3(0),1),pixel);
}
