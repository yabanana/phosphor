#include "atmosphere_common.h"
#include "gi_common.h"

static float fogSlice(float normalized,constant GPUFogParams& p){return p.nearDistance*pow(p.farDistance/p.nearDistance,saturate(normalized));}
static uint fogIndex(uint3 c,constant GPUFogParams& p){return (c.z*p.gridY+c.y)*p.gridX+c.x;}
static float3 fogRay(uint2 cell,constant GPUFogParams& p){return atmoPixelRay(cell,uint2(p.gridX,p.gridY),p.inverseViewProjection,atmoVec(p.cameraPosition));}
static float3 fogPoint(uint3 cell,constant GPUFogParams& p,thread float3& ray,thread float& viewDepth) {
    ray=fogRay(cell.xy,p);viewDepth=fogSlice((float(cell.z)+0.5f)/float(p.gridZ),p);
    const float viewCos=max(1e-5f,-(atmoMatrix(p.view)*float4(ray,0)).z);
    return atmoVec(p.cameraPosition)+ray*(viewDepth/viewCos);
}
static float fogCsm(float3 point,constant GPUShadowParams& p,depth2d<float> m0,depth2d<float> m1,depth2d<float> m2,depth2d<float> m3) {
    constexpr sampler nearest(coord::normalized,filter::nearest,address::clamp_to_edge);
    for(uint c=0;c<4;++c){const float4 clip=atmoMatrix(p.cascades[c].viewProjection)*float4(point,1);const float2 uv=clip.xy*float2(0.5f,-0.5f)+0.5f;
        if(any(uv<0)||any(uv>1)||clip.z<0||clip.z>1)continue;
        float depth=0;switch(c){case 0:depth=m0.sample(nearest,uv);break;case 1:depth=m1.sample(nearest,uv);break;case 2:depth=m2.sample(nearest,uv);break;default:depth=m3.sample(nearest,uv);break;}
        return clip.z+p.cascades[c].biasWorld/max(p.cascades[c].depthRange,1e-6f)>=depth?1.0f:0.0f;}
    return 1; // explicit unshadowed CSM coverage fallback, never screen mask reuse
}
static float fogRtVisibility(float3 point,float3 wi,float distance,instance_acceleration_structure as,
    intersection_function_table<triangle_data,instancing> ift,const device GPUInstance* instances,constant GPUFogParams& p,
    device atomic_uint* counters) {
    GPURtRay ray{};ray.ox=point.x;ray.oy=point.y;ray.oz=point.z;ray.dx=wi.x;ray.dy=wi.y;ray.dz=wi.z;
    // A froxel is a volume point, NOT a geometric surface: do not invent a
    // normal for W&B. Full world origin and metre segment endpoints are used.
    ray.tmin=0;ray.tmax=max(0.0f,distance-1e-4f);ray.mask=RT_MASK_SHADOW;ray.type=RT_PROBE_SHADOW;ray.coneWidth=0;
    RtPayload payload{};const auto hit=rtTrace(ray,as,ift,instances,p.slotCount,payload);volumeCount(counters,3,1);
    if(hit.t==-2)volumeCount(counters,0,1);return hit.hit?0.0f:1.0f;
}
static float3 fogGi(float3 point,constant GPUProbeGridParams& p,const device GPUProbeState* states,texture2d<float> irradiance,texture2d<float> moments) {
    if(p.countX<2||p.countY<2||p.countZ<2)return 0;
    float3 sum=0;
    for(uint axis=0;axis<6;++axis){float3 normal=0;normal[axis/2]=(axis&1u)?-1.0f:1.0f;sum+=giIrradiance(point,normal,p,states,irradiance,moments,true);}
    // Six diffuse irradiance orientations approximate isotropic incident
    // radiance: E/pi for isotropic light. No surface cosine belongs to fog.
    return sum/(6*M_PI_F);
}
static DISample fogLocalSample(float3 point,const device GPUSampledLight* lights,const device GPUAliasEntry* alias,
    const device GPUEmissiveSurface* emitters,const device GPUMaterial* materials,const device DITextureHandle* textures,
    constant GPUFogParams& p,thread uint& seed,thread float& proposal) {
    DISample invalid{};proposal=0;if(!p.lightCount)return invalid;
    const uint column=min(uint(giRandom(seed)*float(p.lightCount)),p.lightCount-1);
    const GPUAliasEntry entry=alias[column];const uint selected=giRandom(seed)<entry.probability?column:entry.alias;
    if(selected>=p.lightCount)return invalid;const GPUAliasEntry selectedEntry=alias[selected];
    if(selectedEntry.lightIndex>=p.lightCount||!(selectedEntry.selectionPdf>0))return invalid;
    DISample sample=diSampleTexturedLight(lights[selectedEntry.lightIndex],selectedEntry.lightIndex,float2(giRandom(seed),giRandom(seed)),point,emitters,materials,textures);
    proposal=selectedEntry.selectionPdf*(sample.delta?1.0f:sample.pdfArea);return sample;
}
static GPUFogCell fogCell(uint3 cell,constant GPUFogParams& p,thread float3& ray) {
    GPUFogCell out{};float depth;const float3 point=fogPoint(cell,p,ray,depth);
    out.worldPosition[0]=point.x;out.worldPosition[1]=point.y;out.worldPosition[2]=point.z;out.viewDepth=depth;
    out.extinction=min(p.maxDensity,p.densityAtBase*exp(clamp(-(point.y-p.heightBase)*p.heightFalloff,-80.0f,80.0f)));
    if(p.corruption==VOLUME_CORRUPT_UNITS)out.extinction*=1000;
    out.viewID=p.viewID;out.generation=p.generation;out.age=1;out.valid=all(isfinite(point))&&isfinite(out.extinction)&&out.extinction>=0;
    return out;
}
#define FOG_COMMON_ARGS \
    constant GPUFogParams& p [[buffer(0)]],device GPUFogCell* cells [[buffer(1)]], \
    constant GPUProbeGridParams& probes [[buffer(3)]],const device GPUProbeState* states [[buffer(4)]], \
    const device GPUSampledLight* lights [[buffer(5)]],const device GPUAliasEntry* alias [[buffer(6)]], \
    const device GPUEmissiveSurface* emitters [[buffer(7)]],const device GPUMaterial* materials [[buffer(8)]], \
    const device DITextureHandle* textures [[buffer(9)]],constant GPUAtmosphereParams& atmosphere [[buffer(14)]], \
    device atomic_uint* counters [[buffer(15)]],texture2d<float> irradiance [[texture(0)]],texture2d<float> moments [[texture(1)]], \
    texture2d<float> transmittance [[texture(12)]],uint3 cell [[thread_position_in_grid]]
kernel void fog_inject(FOG_COMMON_ARGS,constant GPUShadowParams& shadows [[buffer(13)]],
    depth2d<float> csm0 [[texture(8)]],depth2d<float> csm1 [[texture(9)]],depth2d<float> csm2 [[texture(10)]],depth2d<float> csm3 [[texture(11)]]) {
    if(any(cell>=uint3(p.gridX,p.gridY,p.gridZ)))return;float3 ray;GPUFogCell out=fogCell(cell,p,ray);
    const float3 point=float3(out.worldPosition[0],out.worldPosition[1],out.worldPosition[2]);float3 incident=0;
    if(out.valid && out.extinction>0){const float3 sun=atmoVec(p.sunDirection),moon=atmoVec(p.moonDirection);
        const float visibility=(p.flags&VOLUME_SHADOW_CSM)?fogCsm(point,shadows,csm0,csm1,csm2,csm3):1.0f;
        incident+=atmoVec(p.sunIrradiance)*atmoTransmittance(point,sun,atmosphere,transmittance)*atmoHgPhase(dot(ray,sun),p.anisotropy)*visibility;
        incident+=atmoVec(p.moonIrradiance)*atmoTransmittance(point,moon,atmosphere,transmittance)*atmoHgPhase(dot(ray,moon),p.anisotropy);
        if(p.flags&VOLUME_ENABLE_GI)incident+=fogGi(point,probes,states,irradiance,moments);
        if((p.flags&VOLUME_ENABLE_LOCAL_LIGHTS)&&p.lightCount){uint seed=giHash(fogIndex(cell,p)^p.frameIndex*7919u);const uint samples=clamp(p.maxLocalLights,1u,16u);
            for(uint i=0;i<samples;++i){float proposal;const DISample sample=fogLocalSample(point,lights,alias,emitters,materials,textures,p,seed,proposal);
                if(sample.valid&&proposal>0)incident+=diIncident(sample)*atmoHgPhase(dot(ray,sample.wi),p.anisotropy)/(proposal*float(samples));}}
    }
    const float3 source=incident*out.extinction*atmoVec(p.albedo);out.source[0]=source.x;out.source[1]=source.y;out.source[2]=source.z;
    if(!all(isfinite(source))||!out.valid)volumeCount(counters,0,1);volumeCount(counters,4,1);cells[fogIndex(cell,p)]=out;
}
kernel void fog_inject_rt(FOG_COMMON_ARGS,const device GPUInstance* instances [[buffer(10)]],instance_acceleration_structure tlas [[buffer(11)]],
    intersection_function_table<triangle_data,instancing> ift [[buffer(12)]]) {
    if(any(cell>=uint3(p.gridX,p.gridY,p.gridZ)))return;float3 ray;GPUFogCell out=fogCell(cell,p,ray);
    const float3 point=float3(out.worldPosition[0],out.worldPosition[1],out.worldPosition[2]);float3 incident=0;
    if(out.valid&&out.extinction>0){const float3 sun=atmoVec(p.sunDirection),moon=atmoVec(p.moonDirection);
        const float3 solar=atmoVec(p.sunIrradiance)*atmoTransmittance(point,sun,atmosphere,transmittance);
        const float3 lunar=atmoVec(p.moonIrradiance)*atmoTransmittance(point,moon,atmosphere,transmittance);
        if(any(solar>0))incident+=solar*atmoHgPhase(dot(ray,sun),p.anisotropy)*fogRtVisibility(point,sun,p.maxTraceDistance,tlas,ift,instances,p,counters);
        if(any(lunar>0))incident+=lunar*atmoHgPhase(dot(ray,moon),p.anisotropy)*fogRtVisibility(point,moon,p.maxTraceDistance,tlas,ift,instances,p,counters);
        if(p.flags&VOLUME_ENABLE_GI)incident+=fogGi(point,probes,states,irradiance,moments);
        if((p.flags&VOLUME_ENABLE_LOCAL_LIGHTS)&&p.lightCount){uint seed=giHash(fogIndex(cell,p)^p.frameIndex*7919u);const uint samples=clamp(p.maxLocalLights,1u,16u);
            for(uint i=0;i<samples;++i){float proposal;const DISample sample=fogLocalSample(point,lights,alias,emitters,materials,textures,p,seed,proposal);
                if(sample.valid&&proposal>0)incident+=diIncident(sample)*atmoHgPhase(dot(ray,sample.wi),p.anisotropy)*
                    fogRtVisibility(point,sample.wi,sample.distance,tlas,ift,instances,p,counters)/(proposal*float(samples));}}
    }
    const float3 source=incident*out.extinction*atmoVec(p.albedo);out.source[0]=source.x;out.source[1]=source.y;out.source[2]=source.z;
    if(!all(isfinite(source))||!out.valid)volumeCount(counters,0,1);volumeCount(counters,4,1);cells[fogIndex(cell,p)]=out;
}
#undef FOG_COMMON_ARGS
kernel void fog_temporal(constant GPUFogParams& p [[buffer(0)]],device GPUFogCell* output [[buffer(1)]],
    const device GPUFogCell* current [[buffer(2)]],const device GPUFogCell* previous [[buffer(16)]],
    device atomic_uint* counters [[buffer(15)]],uint3 cell [[thread_position_in_grid]]) {
    if(any(cell>=uint3(p.gridX,p.gridY,p.gridZ)))return;const uint index=fogIndex(cell,p);GPUFogCell value=current[index];bool accepted=false;
    if((p.flags&VOLUME_HISTORY_VALID)&&value.valid){const float3 point(value.worldPosition[0],value.worldPosition[1],value.worldPosition[2]);
        const float4 clip=atmoMatrix(p.previousViewProjection)*float4(point,1);const float oldDepth=-(atmoMatrix(p.previousView)*float4(point,1)).z;
        const float2 uv=clip.xy/max(clip.w,1e-6f)*float2(0.5f,-0.5f)+0.5f;const float z=log(max(oldDepth,p.nearDistance)/p.nearDistance)/log(p.farDistance/p.nearDistance);
        if(clip.w>0&&all(uv>=0)&&all(uv<1)&&z>=0&&z<1){const uint3 oldCell(uint2(uv*float2(p.gridX,p.gridY)),uint(z*p.gridZ));GPUFogCell h=previous[fogIndex(oldCell,p)];
            if(p.corruption==VOLUME_CORRUPT_HISTORY)h.generation^=1u;
            const float3 oldPoint(h.worldPosition[0],h.worldPosition[1],h.worldPosition[2]);
            const bool identity=h.viewID==p.viewID&&h.generation==p.generation&&p.previousViewID==p.viewID;
            accepted=h.valid&&(identity||p.corruption==VOLUME_CORRUPT_HISTORY)&&h.age>0&&h.age<=p.maxHistoryAge&&
                length(oldPoint-point)<=p.positionThreshold&&abs(h.extinction-value.extinction)<=p.depthRelativeThreshold*max(value.extinction,1e-6f);
            if(accepted){float3 low=float3(value.source[0],value.source[1],value.source[2]),high=low;
                if(!identity)volumeCount(counters,2,1); // actual stale identity consumed by the negative control
                for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx){const int2 q=int2(cell.xy)+int2(dx,dy);if(any(q<0)||any(q>=int2(p.gridX,p.gridY)))continue;
                    const GPUFogCell n=current[fogIndex(uint3(uint2(q),cell.z),p)];const float3 source(n.source[0],n.source[1],n.source[2]);low=min(low,source);high=max(high,source);}
                const float3 filtered=mix(float3(value.source[0],value.source[1],value.source[2]),clamp(float3(h.source[0],h.source[1],h.source[2]),low,high),p.historyWeight);
                value.source[0]=filtered.x;value.source[1]=filtered.y;value.source[2]=filtered.z;value.age=min(h.age+1,p.maxHistoryAge);}
        }
        volumeCount(counters,accepted?6u:7u,1u);
    }
    output[index]=value;
}
kernel void fog_integrate(constant GPUFogParams& p [[buffer(0)]],device GPUFogIntegrated* integrated [[buffer(1)]],
    const device GPUFogCell* cells [[buffer(2)]],uint2 column [[thread_position_in_grid]]) {
    if(column.x>=p.gridX||column.y>=p.gridY)return;float3 L=0;float T=1;
    const float3 ray=fogRay(column,p);const float viewCos=max(1e-5f,-(atmoMatrix(p.view)*float4(ray,0)).z);
    for(uint z=0;z<p.gridZ&&z<128;++z){const GPUFogCell cell=cells[fogIndex(uint3(column,z),p)];
        const float ds=(fogSlice(float(z+1)/p.gridZ,p)-fogSlice(float(z)/p.gridZ,p))/viewCos;
        const float segment=exp(-cell.extinction*ds),factor=cell.extinction>1e-8f?-expm1(-cell.extinction*ds)/cell.extinction:ds;
        L+=T*float3(cell.source[0],cell.source[1],cell.source[2])*factor;T*=segment;
        GPUFogIntegrated out{};out.radiance[0]=L.x;out.radiance[1]=L.y;out.radiance[2]=L.z;out.transmittance=T;integrated[fogIndex(uint3(column,z),p)]=out;}
}
static GPUFogIntegrated fogPrefix(uint2 column,float viewDepth,constant GPUFogParams& p,
    const device GPUFogIntegrated* integrated,const device GPUFogCell* cells) {
    float3 L=0;float T=1;
    const float3 ray=fogRay(column,p);const float viewCos=max(1e-5f,-(atmoMatrix(p.view)*float4(ray,0)).z);
    if(viewDepth>p.nearDistance){const uint z=min(uint(log(viewDepth/p.nearDistance)/log(p.farDistance/p.nearDistance)*p.gridZ),p.gridZ-1u);
        if(z>0){const GPUFogIntegrated previous=integrated[fogIndex(uint3(column,z-1),p)];L=float3(previous.radiance[0],previous.radiance[1],previous.radiance[2]);T=previous.transmittance;}
        const GPUFogCell cell=cells[fogIndex(uint3(column,z),p)];const float ds=max(0.0f,(viewDepth-fogSlice(float(z)/p.gridZ,p))/viewCos);
        const float segment=exp(-cell.extinction*ds),factor=cell.extinction>1e-8f?-expm1(-cell.extinction*ds)/cell.extinction:ds;
        L+=T*float3(cell.source[0],cell.source[1],cell.source[2])*factor;T*=segment;}
    GPUFogIntegrated out{};out.radiance[0]=L.x;out.radiance[1]=L.y;out.radiance[2]=L.z;out.transmittance=T;return out;
}
kernel void fog_apply(constant GPUFogParams& p [[buffer(0)]],const device GPUFogIntegrated* integrated [[buffer(1)]],
    const device GPUFogCell* cells [[buffer(2)]],texture2d<float,access::read> scene [[texture(2)]],texture2d<float,access::read> depth [[texture(3)]],
    texture2d<float,access::write> output [[texture(4)]],uint2 pixel [[thread_position_in_grid]]) {
    if(pixel.x>=p.outputWidth||pixel.y>=p.outputHeight)return;const float3 camera=atmoVec(p.cameraPosition);
    const float3 ray=atmoPixelRay(pixel,uint2(p.outputWidth,p.outputHeight),p.inverseViewProjection,camera);
    const float viewCos=max(1e-5f,-(atmoMatrix(p.view)*float4(ray,0)).z);
    const float rayDistance=atmoOpaqueDistance(pixel,uint2(p.outputWidth,p.outputHeight),depth,p.inverseViewProjection,camera,p.farDistance/viewCos);
    const float viewDepth=min(p.farDistance,rayDistance*viewCos);
    const float2 coordinate=(float2(pixel)+0.5f)/float2(p.outputWidth,p.outputHeight)*float2(p.gridX,p.gridY)-0.5f;
    const int2 base=int2(floor(coordinate));const float2 f=fract(coordinate);float3 L=0;float T=0;
    for(uint corner=0;corner<4;++corner){const uint2 bit(corner&1u,(corner>>1u)&1u);
        const uint2 column=uint2(clamp(base+int2(bit),int2(0),int2(p.gridX-1,p.gridY-1)));
        const float weight=(bit.x?f.x:1-f.x)*(bit.y?f.y:1-f.y);const GPUFogIntegrated value=fogPrefix(column,viewDepth,p,integrated,cells);
        L+=weight*float3(value.radiance[0],value.radiance[1],value.radiance[2]);T+=weight*value.transmittance;}
    output.write(float4(L+T*scene.read(pixel).rgb,1),pixel);
}
