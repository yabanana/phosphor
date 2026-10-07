#include <metal_stdlib>
#include "renderer/metalfx_denoise_fixture_layout.h"
using namespace metal;
using namespace phosphor;

// Actual GPU-authored input guides: params0, color0/normal1/rough2/diffuse3/
// specular4/motion5/hit6/reactive7/strength8. No standard-scaler cast/compiler.
kernel void fx_fixture_generate(constant GPUFXFixtureParams& p [[buffer(0)]],
    texture2d<float,access::write> color [[texture(0)]],texture2d<float,access::write> normal [[texture(1)]],
    texture2d<float,access::write> rough [[texture(2)]],texture2d<float,access::write> diffuse [[texture(3)]],
    texture2d<float,access::write> specular [[texture(4)]],texture2d<float,access::write> motion [[texture(5)]],
    texture2d<float,access::write> hit [[texture(6)]],texture2d<float,access::write> reactive [[texture(7)]],
    texture2d<float,access::write> strength [[texture(8)]],uint2 xy [[thread_position_in_grid]]) {
    if(any(xy>=uint2(p.width,p.height)))return;
    float3 value(p.color[0],p.color[1],p.color[2]),n(0,0,1);float r=.5f;float2 mv(0);
    if(p.scenario==FX_FIXTURE_IMPULSE) {
        uint2 center(p.width/2u,p.height/2u);bool patch=all(abs(int2(xy)-int2(center))<=int2(3));
        value=patch?float3(p.impulseAmplitude):float3(0);
    } else if(p.scenario==FX_FIXTURE_CHANNELS) {
        bool right=xy.x>=p.width/2u;
        n=right?float3(.6f,0,.8f):float3(0,0,1);r=right?.9f:.04f;
        uint center=(p.width/4u+p.phase)%max(p.width,1u);
        bool patch=abs(int(xy.x)-int(center))<=3&&abs(int(xy.y)-int(p.height/2u))<=3;
        value=patch?float3(.75f,.5f,.25f):float3(.125f);
        mv=patch&&p.phase>0u?float2(-p.motionPixels,0):float2(0);
    }
    color.write(float4(value,1),xy);normal.write(float4(n,0),xy);rough.write(float4(r),xy);
    diffuse.write(float4(.6f,.6f,.6f,1),xy);specular.write(float4(.04f,.04f,.04f,1),xy);
    motion.write(float4(mv,0,0),xy);hit.write(float4(0),xy);reactive.write(float4(0),xy);strength.write(float4(0),xy);
}
struct FXDepthVertex {float4 position [[position]];};
vertex FXDepthVertex fx_fixture_depth_vs(uint vertex [[vertex_id]]) {
    float2 xy=vertex==0u?float2(-1,-1):vertex==1u?float2(3,-1):float2(-1,3);
    return {float4(xy,0,1)};
}
// Legitimate Depth32 writer; never assume compute storage writes to depth format.
fragment float fx_fixture_depth_fs(FXDepthVertex in [[stage_in]],constant GPUFXFixtureParams& p [[buffer(0)]]) [[depth(any)]] {
    (void)in;return p.nearPlane/p.planeDistance;
}
// params0, SDK0, physical1, actual authored guide textures2..8,
// full images buffer1(SDK RGB then physical RGB), samples buffer2.
kernel void fx_fixture_readback(constant GPUFXFixtureParams& p [[buffer(0)]],
    device float* images [[buffer(1)]],device GPUFXFixtureSample* samples [[buffer(2)]],
    texture2d<float,access::read> sdk [[texture(0)]],texture2d<float,access::read> physical [[texture(1)]],
    texture2d<float,access::read> color [[texture(2)]],texture2d<float,access::read> normal [[texture(3)]],
    texture2d<float,access::read> rough [[texture(4)]],texture2d<float,access::read> motion [[texture(5)]],
    texture2d<float,access::read> depth [[texture(6)]],texture2d<float,access::read> diffuse [[texture(7)]],
    texture2d<float,access::read> specular [[texture(8)]],uint tid [[thread_position_in_grid]]) {
    uint count=p.outputWidth*p.outputHeight;
    if(tid<count) {
        uint2 xy(tid%p.outputWidth,tid/p.outputWidth);
        float3 a=sdk.read(xy).rgb,b=physical.read(xy).rgb;
        for(uint c=0;c<3u;++c){images[tid*3u+c]=a[c];images[count*3u+tid*3u+c]=b[c];}
    }
    if(tid<4u) {
        uint2 point=tid==0u?uint2(p.width/4u,p.height/2u):tid==1u?uint2(3u*p.width/4u,p.height/2u):
            tid==2u?uint2(p.width/2u,p.height/2u):uint2(min(p.width/4u+p.phase,p.width-1u),p.height/2u);
        GPUFXFixtureSample s{};float3 rgb=color.read(point).rgb,n=normal.read(point).xyz;
        for(uint c=0;c<3u;++c){s.color[c]=rgb[c];s.normal[c]=n[c];}
        s.roughness=rough.read(point).x;s.depth=depth.read(point).x;
        float2 mv=motion.read(point).xy;s.motion[0]=mv.x;s.motion[1]=mv.y;
        s.diffuseR=diffuse.read(point).x;s.specularR=specular.read(point).x;samples[tid]=s;
    }
}
