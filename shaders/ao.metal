#include "reflection_common.h"

// Raw AO = cosine-weighted visibility over a WORLD radius. It is a distinct
// scalar signal [0,1], never a second multiplier on transported GI or direct
// irradiance. Params0 surfaces1 depthtexture0 outputtexture1. RTAO adds
// TLAS2 OWN PSO IFT3 instances4; alphaLOD0, any-hit, RT_MASK_INDIRECT.
kernel void ao_rtao(constant GPUAOParams& p [[buffer(0)]],const device GPUDISurface* surfaces [[buffer(1)]],
    instance_acceleration_structure as [[buffer(2)]],intersection_function_table<triangle_data,instancing> ift [[buffer(3)]],
    const device GPUInstance* instances [[buffer(4)]],texture2d<float,access::write> output [[texture(1)]],uint tid [[thread_position_in_grid]]){
    if(tid>=p.width*p.height)return;const GPUDISurface s=surfaces[tid];float visibility=1;
    if(s.valid&&p.radius>0){const float3 n=normalize(diVec(s.geometricNormal)),t=reflectionTangent(n),b=cross(n,t),origin=rtOffsetRay(diVec(s.position),n)+n*p.originBias;
        uint seed=giHash(tid^giHash(p.frameIndex));float sum=0;const uint samples=clamp(p.samples,1u,64u);
        for(uint i=0;i<samples;++i){const float u=(float(i)+giRandom(seed))/float(samples),v=giRandom(seed),r=sqrt(u),phi=2*M_PI_F*v;
            const float3 d=t*(r*cos(phi))+b*(r*sin(phi))+n*sqrt(max(0.0f,1-u));GPURtRay ray{};
            ray.ox=origin.x;ray.oy=origin.y;ray.oz=origin.z;ray.dx=d.x;ray.dy=d.y;ray.dz=d.z;
            ray.tmax=p.radius;ray.mask=RT_MASK_INDIRECT;ray.type=RT_PROBE_SHADOW;ray.coneWidth=0;
            RtPayload payload{};const auto hit=rtTrace(ray,as,ift,instances,p.slotCount,payload);sum+=hit.hit||hit.t==-2.0f?0:1;}
        visibility=sum/float(samples);
    }output.write(float4(visibility,visibility,visibility,1),uint2(tid%p.width,tid/p.width));
}

inline float aoHorizonIntegral(float3 n,float3 v,float3 tangent,float positive,float negative){
    // Numerical cosine-weighted slice integral of the GTAO horizon model.
    // 64 fixed angular samples are a declared experimental discretization;
    // no timings/ground-truth equivalence are claimed for this source package.
    float total=0,visible=0;for(uint i=0;i<64;++i){const float theta=-M_PI_F+2*M_PI_F*(float(i)+.5f)/64.0f;
        const float3 d=v*cos(theta)+tangent*sin(theta);const float weight=max(0.0f,dot(n,d))*abs(sin(theta));
        total+=weight;if(abs(theta)<(theta>=0?positive:negative))visible+=weight;}
    return total>0?visible/total:1;
}
inline bool aoUnproject(uint2 pixel,float z,constant GPUAOParams& p,thread float3& world){
    if(!isfinite(z)||z<=0||z>1)return false;const float2 ndc=(float2(pixel)+.5f)*float2(2,-2)/float2(p.width,p.height)+float2(-1,1);
    const float4 h=reflectionMatrix(p.inverseViewProjection)*float4(ndc,z,1);if(!all(isfinite(h))||abs(h.w)<1e-20f)return false;world=h.xyz/h.w;return all(isfinite(world));
}
kernel void ao_gtao(constant GPUAOParams& p [[buffer(0)]],const device GPUDISurface* surfaces [[buffer(1)]],
    texture2d<float,access::read> depth [[texture(0)]],texture2d<float,access::write> output [[texture(1)]],uint tid [[thread_position_in_grid]]){
    if(tid>=p.width*p.height)return;const uint2 pixel(tid%p.width,tid/p.width);const GPUDISurface s=surfaces[tid];float visibility=1;
    if(s.valid&&p.radius>0){const float3 point=diVec(s.position),n=normalize(diVec(s.geometricNormal)),v=normalize(diVec(s.viewDirection));
        const float3 t=reflectionTangent(v),b=cross(v,t);uint seed=giHash(tid^giHash(p.frameIndex));float sum=0;
        // Projected sampling radius comes from metre radius and positive view
        // distance. Logical integer pixels avoid F8 backing-size sampling bugs.
        const float radiusPixels=min(float(max(p.width,p.height)),p.radius*p.pixelScale/max(s.depth,1e-5f));
        const uint slices=clamp(p.slices,1u,16u),steps=clamp(p.steps,1u,32u);const float rotation=giRandom(seed);
        for(uint slice=0;slice<slices;++slice){const float phi=M_PI_F*(float(slice)+rotation)/float(slices);const float3 axis=t*cos(phi)+b*sin(phi);
            const float4 clip=reflectionMatrix(p.viewProjection)*float4(point+axis,1),center=reflectionMatrix(p.viewProjection)*float4(point,1);
            float2 screen=(clip.xy/max(clip.w,1e-8f)-center.xy/max(center.w,1e-8f))*float2(p.width*.5f,-p.height*.5f);
            if(dot(screen,screen)<1e-12f)screen=float2(cos(phi),sin(phi));else screen=normalize(screen);
            float positive=M_PI_F,negative=M_PI_F;
            for(uint side=0;side<2;++side){float horizonCos=-1.0f,horizonDistance=0;
                for(uint step=1;step<=steps;++step){const float fraction=float(step)/float(steps);const float2 offset=screen*((side? -1.0f:1.0f)*max(1.0f,radiusPixels*fraction*fraction));
                    const int2 q=int2(round(float2(pixel)+offset));if(any(q<0)||any(q>=int2(p.width,p.height)))continue;
                    float3 neighbor;if(!aoUnproject(uint2(q),depth.read(uint2(q)).x,p,neighbor))continue;
                    const float3 delta=neighbor-point;const float distance=length(delta);
                    // Ignore coplanar/below-surface samples. Unknown/offscreen
                    // depth remains unoccluded, preventing silhouette halos.
                    if(!(distance>p.originBias&&distance<=p.radius))continue;
                    const float candidate=dot(delta,n)>p.originBias?clamp(dot(delta/distance,v),-1.0f,1.0f):-1.0f;
                    const float bounded=mix(candidate,-1.0f,saturate((distance-p.radius*.8f)/max(p.radius*.2f,1e-5f)));
                    if(bounded>=horizonCos){horizonCos=bounded;horizonDistance=distance;}
                    else if(distance>horizonDistance+p.thickness){
                        // Conservative height-field thin-feature erosion only
                        // after moving beyond a WORLD thickness interval. It
                        // cannot infer unseen back layers; compare against RTAO.
                        horizonCos=max(-1.0f,horizonCos-1.0f/float(steps));
                    }
                }const float horizon=acos(clamp(horizonCos,-1.0f,1.0f));if(side)negative=horizon;else positive=horizon;}
            sum+=aoHorizonIntegral(n,v,axis,positive,negative);
        }visibility=saturate(sum/float(slices));
    }output.write(float4(visibility,visibility,visibility,1),pixel);
}
