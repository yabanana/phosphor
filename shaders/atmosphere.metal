#include "atmosphere_common.h"

kernel void volume_clear_counters(device atomic_uint* counters [[buffer(15)]],uint tid [[thread_position_in_grid]]) {
    if(tid<8)atomic_store_explicit(counters+tid,0u,memory_order_relaxed);
}
kernel void atmosphere_transmittance(constant GPUAtmosphereParams& p [[buffer(0)]],
    texture2d<float,access::write> output [[texture(5)]],device atomic_uint* counters [[buffer(15)]],uint2 pixel [[thread_position_in_grid]]) {
    if(pixel.x>=p.transmittanceWidth||pixel.y>=p.transmittanceHeight)return;
    const float2 uv=float2(pixel)/float2(p.transmittanceWidth-1,p.transmittanceHeight-1);
    const float H=sqrt((p.topRadius-p.bottomRadius)*(p.topRadius+p.bottomRadius)),rho=H*uv.y,r=sqrt(rho*rho+p.bottomRadius*p.bottomRadius);
    const float dmin=p.topRadius-r,d=dmin+uv.x*(rho+H-dmin),mu=d>1e-5f?clamp((p.topRadius*p.topRadius-r*r-d*d)/(2*r*d),-1.0f,1.0f):1.0f;
    const float3 world=atmoVec(p.planetCenter)+float3(0,r,0),ray=float3(sqrt(max(0.0f,1-mu*mu)),mu,0);
    const AtmoSegment segment=atmoSegment(world,ray,p);float3 tau=0;
    const uint steps=clamp(p.marchSteps,1u,512u);
    for(uint i=0;i<steps&&segment.valid;++i){const float a=float(i)/steps,b=float(i+1)/steps;
        const float t0=segment.begin+(segment.end-segment.begin)*a*a,t1=segment.begin+(segment.end-segment.begin)*b*b;
        tau+=atmoMedium(world+ray*((t0+t1)*0.5f),p).extinction*(t1-t0);}
    if(p.corruption==VOLUME_CORRUPT_UNITS)tau*=1000;
    const float3 T=segment.ground?float3(0):exp(-tau);if(!all(isfinite(T)))volumeCount(counters,0,1);
    output.write(float4(T,1),pixel);
}
kernel void atmosphere_multiscattering(constant GPUAtmosphereParams& p [[buffer(0)]],texture2d<float> trans [[texture(0)]],
    texture2d<float,access::write> output [[texture(5)]],device atomic_uint* counters [[buffer(15)]],uint2 pixel [[thread_position_in_grid]]) {
    if(pixel.x>=p.multiWidth||pixel.y>=p.multiHeight)return;
    const float2 uv=float2(pixel)/float2(p.multiWidth-1,p.multiHeight-1);const float mu=uv.x*2-1;
    const float3 point=atmoVec(p.planetCenter)+float3(0,mix(p.bottomRadius+0.5f,p.topRadius-0.5f,uv.y),0),sun=float3(sqrt(max(0.0f,1-mu*mu)),mu,0);
    const uint directions=clamp(p.multiDirections,1u,128u);float3 first=0,returnFactor=0;
    for(uint i=0;i<directions;++i){const float y=1-2*(float(i)+0.5f)/float(directions),phi=float(i)*2.39996322973f,r=sqrt(max(0.0f,1-y*y));
        const AtmoIntegral integral=atmoIntegrate(point,float3(r*cos(phi),y,r*sin(phi)),p,trans,trans,false,1e8f,true,sun,true);
        first+=integral.radiance/float(directions);returnFactor+=integral.scatteringFactor/float(directions);}
    // Hillaire isotropic higher-order closure: geometric series of return
    // scattering. Denominator guard is a declared extreme-density fallback.
    const float3 multiple=first/max(float3(1e-3f),1-returnFactor);
    if(!all(isfinite(multiple)))volumeCount(counters,0,1);output.write(float4(multiple,1),pixel);
}
kernel void atmosphere_sky_view(constant GPUAtmosphereParams& p [[buffer(0)]],texture2d<float> trans [[texture(0)]],texture2d<float> multi [[texture(1)]],
    texture2d<float,access::write> output [[texture(5)]],device atomic_uint* counters [[buffer(15)]],uint2 pixel [[thread_position_in_grid]]) {
    if(pixel.x>=p.skyWidth||pixel.y>=p.skyHeight)return;
    const float3 camera=atmoVec(p.cameraPosition),relative=camera-atmoVec(p.planetCenter);const float height=length(relative),upLength=max(height,1.0f);
    const float3 ray=atmoSkyDirection((float2(pixel)+0.5f)/float2(p.skyWidth,p.skyHeight),relative/upLength,height,p);
    const AtmoIntegral value=atmoIntegrate(camera,ray,p,trans,multi,true);
    if(!all(isfinite(value.radiance)))volumeCount(counters,0,1);output.write(float4(value.radiance,1),pixel);
}
kernel void atmosphere_apply(constant GPUAtmosphereParams& p [[buffer(0)]],texture2d<float> trans [[texture(0)]],texture2d<float> multi [[texture(1)]],
    texture2d<float> sky [[texture(2)]],texture2d<float,access::read> scene [[texture(3)]],texture2d<float,access::read> depth [[texture(4)]],
    texture2d<float,access::write> output [[texture(5)]],device atomic_uint* counters [[buffer(15)]],uint2 pixel [[thread_position_in_grid]]) {
    if(pixel.x>=p.outputWidth||pixel.y>=p.outputHeight)return;
    const float3 camera=atmoVec(p.cameraPosition),ray=atmoPixelRay(pixel,uint2(p.outputWidth,p.outputHeight),p.inverseViewProjection,camera);
    const float z=depth.read(pixel).x;float3 color;
    if(z>0){const float distance=atmoOpaqueDistance(pixel,uint2(p.outputWidth,p.outputHeight),depth,p.inverseViewProjection,camera,1e8f);
        // The resolved mesh supplies the endpoint radiance, including when it
        // coincides with the planet surface. Aerial perspective adds volume
        // scattering only; the sky/LUT paths still include planetary ground.
        const AtmoIntegral value=atmoIntegrate(camera,ray,p,trans,multi,true,distance,false,float3(0),false,false);
        color=scene.read(pixel).rgb*value.transmittance+value.radiance;}
    else {const float3 relative=camera-atmoVec(p.planetCenter),up=normalize(relative);const float height=length(relative);
        color=sky.sample(kAtmosphereSkySampler,atmoSkyUv(ray,up,height,p)).rgb;
        const AtmoSegment bounds=atmoSegment(camera,ray,p);
        if(!bounds.ground){const float3 T=atmoTransmittance(camera,ray,p,trans);
            color+=T*atmoSolarDisk(ray,p);
            color+=T*atmoLunarDisk(ray,p);
            color+=T*atmoStars(ray,p);
        }
    }
    if(!all(isfinite(color)))volumeCount(counters,0,1);output.write(float4(color,1),pixel);
}
