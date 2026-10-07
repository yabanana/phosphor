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
