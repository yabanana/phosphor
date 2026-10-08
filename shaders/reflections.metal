#include "reflection_common.h"

// RT and screen-space kernels share params0/surfaces1, cache13/GIparams14,
// probeMetadata18, specOutput17. Textures: baseRadiance0, depth1, cubeAtlas2,
// previous/current GIirradiance3/moments4, rawLoOUT5, worldHitDistanceOUT6.
// RT additionally TLAS2/OWN IFT3, instances4/meshes5/vertices6/indices7,
// materials8/DITextureHandles9/GPULights10/sampled11/emitterRecords12,
// GIprobeStates15/traceExtra16. EVERY RT PSO owns its own linked alpha IFT.
inline bool reflectionMaterial(GPURtHit hit,float3 point,float3 geometric,float3 outgoing,
    constant GPUReflectionParams& p,const device GPUInstance* instances,const device GPURtMesh* meshes,
    const device GPUVertex* vertices,const device uint* indices,const device GPUMaterial* materials,
    const device DITextureHandle* textures,thread GPUDISurface& surface,thread float3& emission,thread bool& numericError){
    if(hit.slot>=p.slotCount)return false;const GPUInstance instance=instances[hit.slot];
    if(instance.meshIndex>=p.meshCount||instance.materialIndex>=p.materialCount)return false;
    const GPURtMesh mesh=meshes[instance.meshIndex];if(hit.primitive>=mesh.indexCount/3u)return false;
    const uint base=mesh.indexOffset+3u*hit.primitive;const GPUVertex a=vertices[mesh.vertexOffset+indices[base]],b=vertices[mesh.vertexOffset+indices[base+1]],c=vertices[mesh.vertexOffset+indices[base+2]];
    const float3 bary(1-hit.u-hit.v,hit.u,hit.v);const float2 uv=float2(a.u,a.v)*bary.x+float2(b.u,b.v)*bary.y+float2(c.u,c.v)*bary.z;
    // GPURtHit.frontFacing is WORLD geometric facing. The root F9 helper
    // converts it to material/object facing, including a mirrored instance.
    const GPUMaterial m=materials[instance.materialIndex];
    if(!rtMaterialFrontFacing(hit,instances,p.slotCount)&&!(m.flags&MATERIAL_FLAG_DOUBLE_SIDED))return false;
    float4x4 model=float4x4(float4(instance.modelMatrix[0],instance.modelMatrix[1],instance.modelMatrix[2],instance.modelMatrix[3]),float4(instance.modelMatrix[4],instance.modelMatrix[5],instance.modelMatrix[6],instance.modelMatrix[7]),float4(instance.modelMatrix[8],instance.modelMatrix[9],instance.modelMatrix[10],instance.modelMatrix[11]),float4(instance.modelMatrix[12],instance.modelMatrix[13],instance.modelMatrix[14],instance.modelMatrix[15]));
    float3 n=surfaceNormal(model,float3(a.nx,a.ny,a.nz)*bary.x+float3(b.nx,b.ny,b.nz)*bary.y+float3(c.nx,c.ny,c.nz)*bary.z);
    if(dot(n,outgoing)<0)n=-n;n=normalize(n);if(dot(geometric,outgoing)<0)geometric=-geometric;
    const float3 tangent=(float3x3(model[0].xyz,model[1].xyz,model[2].xyz))*(float3(a.tx,a.ty,a.tz)*bary.x+float3(b.tx,b.ty,b.tz)*bary.y+float3(c.tx,c.ty,c.tz)*bary.z);
    float3 t=tangent-n*dot(n,tangent);if(m.normalTex!=INVALID_TEXTURE_INDEX&&dot(t,t)>1e-8f){t=normalize(t);
        float3 tn=float3(half3(textures[m.normalTex].tex.sample(kGiMaterialSampler,uv,level(0)).rgb))*2-1;tn.xy*=m.normalScale;
        float handed=(a.tw*bary.x+b.tw*bary.y+c.tw*bary.z)*((instance.flags&INSTANCE_FLAG_MIRRORED)?-1.0f:1.0f);n=normalize(t*tn.x+cross(n,t)*handed*tn.y+n*tn.z);}
    float3 albedo=float3(m.baseColor[0],m.baseColor[1],m.baseColor[2]);if(m.baseColorTex!=INVALID_TEXTURE_INDEX)albedo*=float3(half3(textures[m.baseColorTex].tex.sample(kGiMaterialSampler,uv,level(0)).rgb));
    float rough=m.roughness,metal=m.metallic;if(m.metallicRoughnessTex!=INVALID_TEXTURE_INDEX){half4 mr=half4(textures[m.metallicRoughnessTex].tex.sample(kGiMaterialSampler,uv,level(0)));rough*=float(mr.g);metal*=float(mr.b);}
    emission=float3(m.emissive[0],m.emissive[1],m.emissive[2]);if(m.emissiveTex!=INVALID_TEXTURE_INDEX)emission*=float3(half3(textures[m.emissiveTex].tex.sample(kGiMaterialSampler,uv,level(0)).rgb));
    surface={};for(uint i=0;i<3;++i){surface.position[i]=point[i];surface.geometricNormal[i]=geometric[i];surface.shadingNormal[i]=n[i];surface.albedo[i]=albedo[i];surface.viewDirection[i]=outgoing[i];}
    surface.roughness=clamp(rough,.04f,1.0f);surface.metallic=saturate(metal);surface.instanceSlot=hit.slot;surface.instanceGeneration=hit.generation;surface.valid=1;
    const bool finite=all(isfinite(n))&&all(isfinite(albedo))&&all(isfinite(emission));if(!finite)numericError=true;return finite;
}
inline float3 reflectionSecondary(GPURtHit hit,float3 direction,
    instance_acceleration_structure as,intersection_function_table<triangle_data,instancing> ift,
    constant GPUReflectionParams& p,const device GPUInstance* instances,const device GPURtMesh* meshes,
    const device GPUVertex* vertices,const device uint* indices,const device GPUMaterial* materials,
    const device DITextureHandle* textures,const device GPULight* lights,const device GPUSampledLight* sampled,
    const device GPUEmissiveSurface* emitters,constant GPUProbeGridParams& gi,const device GPUProbeState* states,
    constant GPUProbeTraceExtra& extra,texture2d<float> irradiance,texture2d<float> moments,thread uint& seed,thread bool& numericError){
    float3 point,normal,emission;GPUDISurface surface;
    if(!rtSurface(hit,instances,meshes,vertices,indices,p.slotCount,p.meshCount,point,normal)||
        !reflectionMaterial(hit,point,normal,-direction,p,instances,meshes,vertices,indices,materials,textures,surface,emission,numericError))return 0;
    float3 result=max(emission,0.0f);const float3 n=diVec(surface.shadingNormal),geometric=diVec(surface.geometricNormal);
    for(uint i=0;i<p.lightCount;++i){const GPULight light=lights[i];if(extra.sampledLightCount&&light.type!=LIGHT_DIRECTIONAL)continue;
        DISample sample{};sample.valid=true;sample.delta=true;sample.distance=p.maxDistance;sample.radiance=float3(light.color[0],light.color[1],light.color[2])*light.intensity;
        if(light.type==LIGHT_DIRECTIONAL){sample.wi=-normalize(float3(light.direction[0],light.direction[1],light.direction[2]));sample.geometry=1;
            if(extra.sunAngularRadius>0){const float cosCone=cos(extra.sunAngularRadius),z=1-giRandom(seed)*(1-cosCone),r=sqrt(max(0.0f,1-z*z)),phi=2*M_PI_F*giRandom(seed);
                const float3 t=reflectionTangent(sample.wi),b=cross(sample.wi,t);sample.wi=t*(r*cos(phi))+b*(r*sin(phi))+sample.wi*z;sample.geometry=2/(1+cosCone);}
            sample.position=point+sample.wi*p.maxDistance;}
        else{GPUSampledLight l{};l.type=light.type;l.range=light.range;l.innerCone=light.innerCone;l.outerCone=light.outerCone;
            for(uint c=0;c<3;++c){l.position[c]=light.position[c];l.axisU[c]=light.direction[c];l.emission[c]=light.color[c]*light.intensity;}sample=diSampleLight(l,float2(.5f),point);}
        const float3 contribution=diBRDF(surface,sample);if(!all(isfinite(contribution))||any(contribution<0))numericError=true;
        else if(sample.valid&&any(contribution>0)&&diEndpointVisible(surface,sample,as,ift,instances,p.slotCount))result+=contribution;
    }
    if(extra.sampledLightCount){const uint i=min(uint(giRandom(seed)*extra.sampledLightCount),extra.sampledLightCount-1u);
        const DISample sample=diSampleTexturedLight(sampled[i],i,float2(giRandom(seed),giRandom(seed)),point,emitters,materials,textures);
        if(sample.valid&&sample.pdfArea>0){const float3 contribution=diBRDF(surface,sample)*float(extra.sampledLightCount)/sample.pdfArea;
            if(!all(isfinite(contribution))||any(contribution<0))numericError=true;else if(diEndpointVisible(surface,sample,as,ift,instances,p.slotCount))result+=contribution;}}
    // GI atlas is INDIRECT irradiance. Add once; AO never modulates it.
    if((p.flags&REFLECTION_ENABLE_GI)&&gi.countX>=2&&gi.countY>=2&&gi.countZ>=2)
        result+=giIrradiance(point,geometric,gi,states,irradiance,moments,true)*diVec(surface.albedo)*(1-surface.metallic)/M_PI_F;
    if(!all(isfinite(result))||any(result<0)){numericError=true;return 0;}return result;
}

kernel void reflection_ssr(constant GPUReflectionParams& p [[buffer(0)]],const device GPUDISurface* surfaces [[buffer(1)]],
    const device GPURadianceCacheEntry* cache [[buffer(13)]],constant GPUProbeGridParams& gi [[buffer(14)]],
    device GPUSpecularSample* metadata [[buffer(17)]],const device GPUReflectionProbe* probes [[buffer(18)]],
    texture2d<float,access::read> baseRadiance [[texture(0)]],texture2d<float,access::read> depth [[texture(1)]],texturecube_array<float> atlas [[texture(2)]],
    texture2d<float,access::write> output [[texture(5)]],texture2d<float,access::write> distance [[texture(6)]],uint tid [[thread_position_in_grid]]){
    if(tid>=p.width*p.height)return;uint seed=giHash(tid^giHash(p.frameIndex)^p.seed);const GPUDISurface s=surfaces[tid];
    const auto dir=reflectionDirection(s,float2(giRandom(seed),giRandom(seed)));const auto sample=reflectionFallback(s,dir,p,surfaces,cache,gi,probes,baseRadiance,depth,atlas);
    metadata[tid]=sample;uint2 pixel(tid%p.width,tid/p.width);output.write(float4(sample.radiance[0],sample.radiance[1],sample.radiance[2],1),pixel);distance.write(float4(sample.hitDistance),pixel);
}

kernel void reflection_rt(constant GPUReflectionParams& p [[buffer(0)]],const device GPUDISurface* surfaces [[buffer(1)]],
    instance_acceleration_structure as [[buffer(2)]],intersection_function_table<triangle_data,instancing> ift [[buffer(3)]],
    const device GPUInstance* instances [[buffer(4)]],const device GPURtMesh* meshes [[buffer(5)]],const device GPUVertex* vertices [[buffer(6)]],const device uint* indices [[buffer(7)]],
    const device GPUMaterial* materials [[buffer(8)]],const device DITextureHandle* textures [[buffer(9)]],const device GPULight* lights [[buffer(10)]],
    const device GPUSampledLight* sampled [[buffer(11)]],const device GPUEmissiveSurface* emitters [[buffer(12)]],const device GPURadianceCacheEntry* cache [[buffer(13)]],
    constant GPUProbeGridParams& gi [[buffer(14)]],const device GPUProbeState* states [[buffer(15)]],constant GPUProbeTraceExtra& extra [[buffer(16)]],
    device GPUSpecularSample* metadata [[buffer(17)]],const device GPUReflectionProbe* probes [[buffer(18)]],
    texture2d<float,access::read> baseRadiance [[texture(0)]],texture2d<float,access::read> depth [[texture(1)]],texturecube_array<float> atlas [[texture(2)]],
    texture2d<float> irradiance [[texture(3)]],texture2d<float> moments [[texture(4)]],texture2d<float,access::write> output [[texture(5)]],
    texture2d<float,access::write> distance [[texture(6)]],uint tid [[thread_position_in_grid]]){
    if(tid>=p.width*p.height)return;const GPUDISurface s=surfaces[tid];uint seed=giHash(tid^giHash(p.frameIndex)^p.seed);
    const auto dir=reflectionDirection(s,float2(giRandom(seed),giRandom(seed)));GPUSpecularSample sample=reflectionEmpty();
    if(s.valid&&dir.valid&&any(dir.weight>0)&&giRandom(seed)<reflectionRTWeight(s.roughness,p)){
        float3 n=normalize(diVec(s.geometricNormal));if(dot(n,dir.direction)<0)n=-n;const float3 origin=rtOffsetRay(diVec(s.position),n);GPURtRay ray{};
        ray.ox=origin.x;ray.oy=origin.y;ray.oz=origin.z;ray.dx=dir.direction.x;ray.dy=dir.direction.y;ray.dz=dir.direction.z;ray.tmax=p.maxDistance;ray.mask=RT_MASK_INDIRECT;ray.type=RT_PROBE_DIFFUSE;
        RtPayload payload{};const auto hit=rtTrace(ray,as,ift,instances,p.slotCount,payload);float3 L;
        sample.flags=SPECULAR_SAMPLE_VALID;sample.proposalSolidAngle=dir.pdf;for(uint i=0;i<3;++i)sample.direction[i]=dir.direction[i];
        bool numericError=dir.error;
        if(hit.hit){L=reflectionSecondary(hit,dir.direction,as,ift,p,instances,meshes,vertices,indices,materials,textures,lights,sampled,emitters,gi,states,extra,irradiance,moments,seed,numericError);
            sample.path=REFLECTION_PATH_RT;sample.hitDistance=length(origin+dir.direction*hit.t-diVec(s.position));sample.secondarySlot=hit.slot;sample.secondaryGeneration=hit.generation;}
        else L=reflectionProbeRadiance(diVec(s.position),dir.direction,s.roughness,false,p,probes,atlas,sample.path);
        L*=dir.weight;if(!all(isfinite(L))||any(L<0)){numericError=true;L=0;}if(numericError)sample.flags|=SPECULAR_SAMPLE_ERROR;for(uint i=0;i<3;++i)sample.radiance[i]=L[i];
    }else sample=reflectionFallback(s,dir,p,surfaces,cache,gi,probes,baseRadiance,depth,atlas);
    metadata[tid]=sample;uint2 pixel(tid%p.width,tid/p.width);output.write(float4(sample.radiance[0],sample.radiance[1],sample.radiance[2],1),pixel);distance.write(float4(sample.hitDistance),pixel);
}

// Optional STATIC actual scene capture from each probe face, not a constant
// environment posing as a scene. Same RT resource ABI above, no surfaces/cache/
// SSR reads; output texture5 is a 2D view of the selected face/mip0. Own IFT.
kernel void reflection_capture_rt(constant GPUReflectionParams& p [[buffer(0)]],instance_acceleration_structure as [[buffer(2)]],
    intersection_function_table<triangle_data,instancing> ift [[buffer(3)]],const device GPUInstance* instances [[buffer(4)]],
    const device GPURtMesh* meshes [[buffer(5)]],const device GPUVertex* vertices [[buffer(6)]],const device uint* indices [[buffer(7)]],
    const device GPUMaterial* materials [[buffer(8)]],const device DITextureHandle* textures [[buffer(9)]],const device GPULight* lights [[buffer(10)]],
    const device GPUSampledLight* sampled [[buffer(11)]],const device GPUEmissiveSurface* emitters [[buffer(12)]],constant GPUProbeGridParams& gi [[buffer(14)]],
    const device GPUProbeState* states [[buffer(15)]],constant GPUProbeTraceExtra& extra [[buffer(16)]],texture2d<float> irradiance [[texture(3)]],texture2d<float> moments [[texture(4)]],
    texture2d<float,access::write> output [[texture(5)]],uint tid [[thread_position_in_grid]]){
    if(tid>=p.width*p.height)return;const uint2 pixel(tid%p.width,tid/p.width);const float2 ndc=(float2(pixel)+.5f)*float2(2,-2)/float2(p.width,p.height)+float2(-1,1);
    const float4 near=reflectionMatrix(p.inverseViewProjection)*float4(ndc,1,1);const float3 origin=reflectionConstantVec(p.cameraPosition),direction=normalize(near.xyz/near.w-origin);
    GPURtRay ray{};ray.ox=origin.x;ray.oy=origin.y;ray.oz=origin.z;ray.dx=direction.x;ray.dy=direction.y;ray.dz=direction.z;ray.tmax=p.maxDistance;ray.mask=RT_MASK_INDIRECT;ray.type=RT_PROBE_DIFFUSE;
    RtPayload payload{};const auto hit=rtTrace(ray,as,ift,instances,p.slotCount,payload);uint seed=giHash(tid^p.seed);bool numericError=false;const float3 L=hit.hit?
        reflectionSecondary(hit,direction,as,ift,p,instances,meshes,vertices,indices,materials,textures,lights,sampled,emitters,gi,states,extra,irradiance,moments,seed,numericError):reflectionConstantVec(p.environment);
    const bool valid=!numericError&&reflectionProbeStorageAccepts(L.x,L.y,L.z,p.flags);
    output.write(float4(valid?L:float3(0),valid?1.0f:-1.0f),pixel);
}
