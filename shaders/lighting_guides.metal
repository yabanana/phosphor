#include <metal_stdlib>
#include "renderer/gpu_types.h"
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
                                  texture2d<float,access::read> world [[texture(5)]],
                                  texture2d<float,access::read> normals [[texture(6)]],
                                  texture2d<uint,access::read> keys [[texture(10)]],
                                  depth2d<float,access::read> depth [[texture(0)]], uint2 pixel [[thread_position_in_grid]]) {
    if (any(pixel >= uint2(p.width,p.height))) return;
    const uint4 key = keys.read(pixel); if (key.x == ~0u) return;
    const float4 point = world.read(pixel); const float4 n = normals.read(pixel);
    GPUShadowSurface s{};
    for (uint c=0;c<3;++c) { s.position[c]=point[c]; s.geometricNormal[c]=n[c]; }
    s.viewDepth=-(guideMatrix(p.view)*float4(point.xyz,1)).z; s.reverseDepth=depth.read(pixel);
    s.slot=key.x; s.generation=key.y; s.primitive=key.z;
    s.valid=point.w>0 && n.w>0 && all(isfinite(point.xyz)) && all(isfinite(n.xyz));
    surfaces[pixel.y*p.width+pixel.x]=s;
}
kernel void lighting_zero(texture2d<float,access::write> output [[texture(0)]], uint2 p [[thread_position_in_grid]]) {
    if(all(p<uint2(output.get_width(),output.get_height())))output.write(float4(0),p);
}

kernel void di_receiver_pack(constant FrameConstants& frame [[buffer(0)]],constant GPUDIParams& p [[buffer(1)]],
    device GPUDISurface* surfaces [[buffer(2)]],texture2d<float,access::read> world [[texture(5)]],
    texture2d<float,access::read> geometric [[texture(6)]],texture2d<uint,access::read> keys [[texture(10)]],
    texture2d<float,access::read> shading [[texture(11)]],texture2d<float,access::read> albedo [[texture(12)]],
    uint2 pixel [[thread_position_in_grid]]) {
    if(any(pixel>=uint2(p.width,p.height)))return;const uint4 key=keys.read(pixel);if(key.x==~0u)return;
    const float4 pos=world.read(pixel),g=geometric.read(pixel),s=shading.read(pixel),a=albedo.read(pixel);
    if(dot(s.xyz,s.xyz)<=0)return;
    GPUDISurface out{};const float3 v=normalize(float3(frame.cameraPosition[0],frame.cameraPosition[1],frame.cameraPosition[2])-pos.xyz);
    for(uint c=0;c<3;++c){out.position[c]=pos[c];out.geometricNormal[c]=g[c];out.shadingNormal[c]=s[c];out.albedo[c]=a[c];out.viewDirection[c]=v[c];}
    out.depth=-(guideMatrix(frame.view)*float4(pos.xyz,1)).z;out.roughness=s.w;out.metallic=a.w;
    out.materialRevision=p.historyEpoch;out.instanceSlot=key.x;out.instanceGeneration=key.y;out.valid=pos.w>0 && g.w>0 && dot(s.xyz,s.xyz)>0;
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
