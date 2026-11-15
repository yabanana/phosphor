#include "reflection_common.h"
#include "texture_transform_core.h" // F15.1 (ticket 22): per-slot texture transforms at the reflection hits
#include "renderer/denoise_endpoint.h"

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
    // F15.4 (ticket 23): a layer stack's blended base parameters, normal and emission (the helper of gi_common.h, one source with GI and probes).
    LayerSurface layered;const bool stack=layerStackSurface<DITextureHandle,LayerLod0Samples>(materials[instance.materialIndex],p.materialExtensions,materials,textures,uv,float2(0.0f),float2(0.0f),layered);
    const float3 nShade=n;
    float3 t=tangent-n*dot(n,tangent);if(m.normalTex!=INVALID_TEXTURE_INDEX&&dot(t,t)>1e-8f){t=normalize(t);
        // F15.1 (ticket 22): the normal slot's KHR_texture_transform (the texel axes follow the transformed uv); unchanged without one.
        float3 tn=float3(half3(textures[m.normalTex].tex.sample(kGiMaterialSampler,texXfSlotUv(p.materialExtensions,m.flags,m.textureTransformIndex,TT_SLOT_NORMAL,uv),level(0)).rgb))*2-1;tn.xy*=m.normalScale;
        float handed=(a.tw*bary.x+b.tw*bary.y+c.tw*bary.z)*((instance.flags&INSTANCE_FLAG_MIRRORED)?-1.0f:1.0f);
        if((m.flags&MATERIAL_FLAG_TEXTURE_TRANSFORM)!=0u){float3 tt=t,bb=cross(n,t)*handed;texXfFrameOn(texXfForOn(p.materialExtensions,m.flags,m.textureTransformIndex),TT_SLOT_NORMAL,tt,bb);n=normalize(tt*tn.x+bb*tn.y+n*tn.z);}
        else n=normalize(t*tn.x+cross(n,t)*handed*tn.y+n*tn.z);}
    if(stack){const float3 ts=tangent-nShade*dot(nShade,tangent);if(dot(ts,ts)>1e-8f){const float3 tu=normalize(ts);
        const float handedL=(a.tw*bary.x+b.tw*bary.y+c.tw*bary.z)*((instance.flags&INSTANCE_FLAG_MIRRORED)?-1.0f:1.0f);
        n=normalize(tu*layered.tn.x+cross(nShade,tu)*handedL*layered.tn.y+nShade*layered.tn.z);}else n=nShade;}
    float3 albedo=float3(m.baseColor[0],m.baseColor[1],m.baseColor[2]);if(m.baseColorTex!=INVALID_TEXTURE_INDEX)albedo*=float3(half3(textures[m.baseColorTex].tex.sample(kGiMaterialSampler,texXfSlotUv(p.materialExtensions,m.flags,m.textureTransformIndex,TT_SLOT_BASE_COLOR,uv),level(0)).rgb));
    // F15 closure (COLOR_0): base colour = factor x texture x vertex colour (RT hit surface); an unlit material shows it through its emission.
    float4 vertexColor=float4(1.0f);if(p.vertexColors!=nullptr){vertexColor=p.vertexColors[mesh.vertexOffset+indices[base]]*bary.x+p.vertexColors[mesh.vertexOffset+indices[base+1]]*bary.y+p.vertexColors[mesh.vertexOffset+indices[base+2]]*bary.z;albedo*=vertexColor.rgb;}
    float rough=m.roughness,metal=m.metallic;if(m.metallicRoughnessTex!=INVALID_TEXTURE_INDEX){half4 mr=half4(textures[m.metallicRoughnessTex].tex.sample(kGiMaterialSampler,texXfSlotUv(p.materialExtensions,m.flags,m.textureTransformIndex,TT_SLOT_METAL_ROUGH,uv),level(0)));rough*=float(mr.g);metal*=float(mr.b);}
    emission=float3(m.emissive[0],m.emissive[1],m.emissive[2]);if(m.emissiveTex!=INVALID_TEXTURE_INDEX)emission*=float3(half3(textures[m.emissiveTex].tex.sample(kGiMaterialSampler,texXfSlotUv(p.materialExtensions,m.flags,m.textureTransformIndex,TT_SLOT_EMISSIVE,uv),level(0)).rgb));if(stack){albedo=layered.baseColor;rough=layered.roughness;metal=layered.metallic;emission+=layered.emission;}
    if(p.vertexColors!=nullptr&&(m.flags&MATERIAL_FLAG_UNLIT)!=0u)emission*=vertexColor.rgb;
    surface={};for(uint i=0;i<3;++i){surface.position[i]=point[i];surface.geometricNormal[i]=geometric[i];surface.shadingNormal[i]=n[i];surface.albedo[i]=albedo[i];surface.viewDirection[i]=outgoing[i];}
    surface.roughness=clamp(rough,.04f,1.0f);surface.metallic=saturate(metal);surface.instanceSlot=hit.slot;surface.instanceGeneration=hit.generation;surface.valid=1;
    const bool finite=all(isfinite(n))&&all(isfinite(albedo))&&all(isfinite(emission));if(!finite)numericError=true;return finite;
}
// F13.1 residue found in F15: a transport ray that meets the BACK of a single-sided surface.  glTF single-sided means invisible from behind (the
// rasteriser culls it), but reflectionMaterial rejects a back face, so the ray ended there and returned BLACK.  The ray now goes on past the surface,
// at most kReflectionBackfaceSkips times per trace (a bound: more ends in the probes/environment like a miss) and counted in the numerical
// counter words 16 (continuations) and 17 (bound exhausted, the ray ended).  `glassInterface`: the refraction walk treats the back face of a
// transmissive single-sided mesh as an interface (the inside of a closed glass volume), so only the reflection estimators cull it.
constant uint kReflectionBackfaceSkips=4u;
inline bool reflectionCulledBackFace(GPURtHit hit,constant GPUReflectionParams& p,const device GPUInstance* instances,const device GPUMaterial* materials,bool glassInterface){
    if(!hit.hit||hit.slot>=p.slotCount)return false;const GPUInstance instance=instances[hit.slot];
    if(instance.materialIndex>=p.materialCount)return false;const GPUMaterial m=materials[instance.materialIndex];
    if(m.flags&MATERIAL_FLAG_DOUBLE_SIDED)return false;
    if(glassInterface&&(m.flags&(MATERIAL_FEATURE_TRANSMISSION<<MATERIAL_FEATURE_SHIFT)))return false;
    return !rtMaterialFrontFacing(hit,instances,p.slotCount);
}
template<typename Payload>
inline GPURtHit reflectionTraceVisible(GPURtRay ray,instance_acceleration_structure as,intersection_function_table<triangle_data,instancing> ift,
    constant GPUReflectionParams& p,const device GPUInstance* instances,const device GPURtMesh* meshes,const device GPUVertex* vertices,
    const device uint* indices,const device GPUMaterial* materials,thread Payload& payload,device atomic_uint* counts){
    float travelled=0;GPURtHit hit{};const float tmax=ray.tmax;
    for(uint k=0;k<=kReflectionBackfaceSkips;++k){
        hit=rtTraceSurface(ray,as,ift,instances,p.slotCount,payload,meshes,vertices,indices,p.meshCount);
        if(!hit.hit)return hit;
        const float segment=hit.t;hit.t+=travelled;
        if(!reflectionCulledBackFace(hit,p,instances,materials,false))return hit;
        if(counts)atomic_fetch_add_explicit(counts+16,1u,memory_order_relaxed);
        if(k==kReflectionBackfaceSkips)break;
        const float3 from(ray.ox,ray.oy,ray.oz),d(ray.dx,ray.dy,ray.dz),next=rtOffsetRay(from+d*segment,d);
        travelled+=length(next-from);ray.ox=next.x;ray.oy=next.y;ray.oz=next.z;ray.tmax=max(tmax-travelled,0.0f);
    }
    if(counts)atomic_fetch_add_explicit(counts+17,1u,memory_order_relaxed);
    hit.hit=0u;hit.t=-1.0f;hit.slot=~0u;return hit;
}
// F15.1 (ticket 22, C22-A1): a secondary hit that is itself a near-delta specular reflector.  reflectionSecondary reports it so that the ray can go on:
// a surface whose clamped roughness is at most kReflectionMirrorRoughness (alpha <= 0.01, the lobe of a polished surface: the ideal mirror direction
// stands for it; every smooth surface, metal or dielectric: no F0 floor, no per-hit Fresnel floor: the chain stops on the cumulative
// throughput, the product of the Fresnel weights along it, below kReflectionMirrorMinThroughput).  `weight` is that
// Fresnel term F(F0, n.v) of the base BSDF (F0 = mix(0.04, baseColor, metallic)), the directional albedo of a smooth specular lobe; the diffuse
// and light terms of the hit are shaded as before (a hit that is not mirror-like keeps the single-bounce result bit for bit).
constant float kReflectionMirrorRoughness=0.1f;
constant float kReflectionMirrorMinThroughput=0.01f;
struct ReflectionMirror {bool mirror;float3 point,normal,geometric,weight;float roughness;};
template<typename Payload=RtPayload>
inline float3 reflectionSecondary(GPURtHit hit,float3 direction,
    instance_acceleration_structure as,intersection_function_table<triangle_data,instancing> ift,
    constant GPUReflectionParams& p,const device GPUInstance* instances,const device GPURtMesh* meshes,
    const device GPUVertex* vertices,const device uint* indices,const device GPUMaterial* materials,
    const device DITextureHandle* textures,const device GPULight* lights,const device GPUSampledLight* sampled,
    const device GPUEmissiveSurface* emitters,constant GPUProbeGridParams& gi,const device GPUProbeState* states,
    constant GPUProbeTraceExtra& extra,texture2d<float> irradiance,texture2d<float> moments,thread uint& seed,thread bool& numericError,
    float minRoughness=0.0f,thread float* emissionLuminance=nullptr,thread ReflectionMirror* mirror=nullptr){
    float3 point,normal,emission;GPUDISurface surface;
    if(!rtSurface(hit,instances,meshes,vertices,indices,p.slotCount,p.meshCount,point,normal)||
        !reflectionMaterial(hit,point,normal,-direction,p,instances,meshes,vertices,indices,materials,textures,surface,emission,numericError))return 0;
    float3 result=max(emission,0.0f);const float3 n=diVec(surface.shadingNormal),geometric=diVec(surface.geometricNormal);
    if(mirror){mirror->mirror=false;
        if(surface.roughness<=kReflectionMirrorRoughness){const float3 F=bsdfFSchlick(bsdfBaseF0(max(diVec(surface.albedo),0.0f),surface.metallic),saturate(dot(n,-direction)));
            if(all(isfinite(F))){mirror->mirror=true;mirror->point=point;mirror->normal=n;mirror->geometric=geometric;mirror->weight=F;mirror->roughness=surface.roughness;}}}
    if(emissionLuminance)*emissionLuminance=dot(result,float3(.2126f,.7152f,.0722f));
    // Path regularisation (declared in the F13 reference): the light connection
    // at the vertex after a sampled specular lobe uses at least that lobe's
    // roughness, alpha = max(alpha_secondary, alpha_camera). It bounds the
    // variance of specular-specular-light paths through a near-delta lobe.
    surface.roughness=max(surface.roughness,minRoughness);
    for(uint i=0;i<p.lightCount;++i){const GPULight light=lights[i];if(extra.sampledLightCount&&light.type!=LIGHT_DIRECTIONAL)continue;
        DISample sample{};sample.valid=true;sample.delta=true;sample.distance=p.maxDistance;sample.radiance=float3(light.color[0],light.color[1],light.color[2])*light.intensity;
        if(light.type==LIGHT_DIRECTIONAL){sample.wi=-normalize(float3(light.direction[0],light.direction[1],light.direction[2]));sample.geometry=1;
            if(extra.sunAngularRadius>0){const float cosCone=cos(extra.sunAngularRadius),z=1-giRandom(seed)*(1-cosCone),r=sqrt(max(0.0f,1-z*z)),phi=2*M_PI_F*giRandom(seed);
                const float3 t=reflectionTangent(sample.wi),b=cross(sample.wi,t);sample.wi=t*(r*cos(phi))+b*(r*sin(phi))+sample.wi*z;sample.geometry=2/(1+cosCone);}
            sample.position=point+sample.wi*p.maxDistance;}
        else{GPUSampledLight l{};l.type=light.type;l.range=light.range;l.innerCone=light.innerCone;l.outerCone=light.outerCone;
            for(uint c=0;c<3;++c){l.position[c]=light.position[c];l.axisU[c]=light.direction[c];l.emission[c]=light.color[c]*light.intensity;}sample=diSampleLight(l,float2(.5f),point);}
        const float3 contribution=diBRDF(surface,sample);if(!all(isfinite(contribution))||any(contribution<0))numericError=true;
        else if(sample.valid&&any(contribution>0))result+=contribution*diEndpointTransmittance<Payload>(surface,sample,as,ift,instances,p.slotCount);
    }
    if(extra.sampledLightCount){const uint i=min(uint(giRandom(seed)*extra.sampledLightCount),extra.sampledLightCount-1u);
        const DISample sample=diSampleTexturedLight(sampled[i],i,float2(giRandom(seed),giRandom(seed)),point,emitters,materials,textures,p.materialExtensions,p.vertexColors);
        if(sample.valid&&sample.pdfArea>0){const float3 contribution=diBRDF(surface,sample)*float(extra.sampledLightCount)/sample.pdfArea;
            if(!all(isfinite(contribution))||any(contribution<0))numericError=true;else result+=contribution*diEndpointTransmittance<Payload>(surface,sample,as,ift,instances,p.slotCount);}}
    // GI atlas is INDIRECT irradiance. Add once; AO never modulates it.
    if((p.flags&REFLECTION_ENABLE_GI)&&gi.countX>=2&&gi.countY>=2&&gi.countZ>=2)
        result+=giIrradiance(point,geometric,gi,states,irradiance,moments,true)*diVec(surface.albedo)*(1-surface.metallic)/M_PI_F;
    // F15.1 (ticket 22, C22-D4; F13.1 residue found in F15): wherever the probe atlas does not supply the indirect light (transported GI off), the
    // secondary hit receives the declared environment diffuse term the camera-visible shading adds (environment x albedo x (1 - metallic), no
    // occlusion).  The radiance a surface sends toward a viewer does not depend on who looks: a surface the sun does not reach (the wall in the
    // shadow of an opaque panel) must be as ambient-lit in a mirror, a refraction walk or a probe capture as it is seen directly.  Without the
    // term a shadowed hit was black in every transport estimator (the reflection estimators had kept the F13 behaviour until ticket 22's
    // fixture showed the asymmetry with the refraction walk).
    else result+=float3(p.environment[0],p.environment[1],p.environment[2])*diVec(surface.albedo)*(1-surface.metallic);
    if(!all(isfinite(result))||any(result<0)){numericError=true;return 0;}return result;
}

// F15.1 (ticket 22, C22-A1): the radiance of a reflection ray that met `hit`, following up to `maxBounces` further ideal specular reflections while the
// surface it meets is mirror-like (ReflectionMirror).  Every bounce multiplies the throughput by the Fresnel weight of the mirror it left, shades
// the next hit with the same secondary-hit model (reflectionSecondary) and ends in the probes/environment on a miss, like the first ray.  Hard
// bounds: maxBounces <= 3 traces, the throughput below kReflectionMirrorMinThroughput ends the chain (negligible, not counted), a bound that ends a
// chain at a still mirror-like hit drops that hit's own reflection (counter word 19).  Counter word 18 counts the extra traces.  A hit that is not
// mirror-like, or maxBounces == 0, returns exactly what reflectionSecondary returns.
template<typename Payload=RtPayload>
inline float3 reflectionSecondaryChain(GPURtHit hit,float3 direction,uint maxBounces,
    instance_acceleration_structure as,intersection_function_table<triangle_data,instancing> ift,
    constant GPUReflectionParams& p,const device GPUInstance* instances,const device GPURtMesh* meshes,
    const device GPUVertex* vertices,const device uint* indices,const device GPUMaterial* materials,
    const device DITextureHandle* textures,const device GPULight* lights,const device GPUSampledLight* sampled,
    const device GPUEmissiveSurface* emitters,constant GPUProbeGridParams& gi,const device GPUProbeState* states,
    constant GPUProbeTraceExtra& extra,texture2d<float> irradiance,texture2d<float> moments,
    const device GPUReflectionProbe* probes,texturecube_array<float> atlas,thread uint& seed,thread bool& numericError,
    float minRoughness,device atomic_uint* counts){
    ReflectionMirror m{};
    float3 L=reflectionSecondary<Payload>(hit,direction,as,ift,p,instances,meshes,vertices,indices,materials,textures,lights,sampled,emitters,gi,states,extra,irradiance,moments,
        seed,numericError,minRoughness,nullptr,maxBounces?&m:nullptr);
    float3 throughput=1;
    for(uint bounce=0;bounce<maxBounces&&m.mirror;++bounce){
        throughput*=m.weight;if(!(bsdfMax3(throughput)>=kReflectionMirrorMinThroughput))break;
        const float3 r=reflect(direction,m.normal);if(!all(isfinite(r))||dot(r,m.geometric)<=0)break;
        GPURtRay ray{};const float3 origin=rtOffsetRay(m.point,m.geometric);
        ray.ox=origin.x;ray.oy=origin.y;ray.oz=origin.z;ray.dx=r.x;ray.dy=r.y;ray.dz=r.z;ray.tmax=p.maxDistance;ray.mask=RT_MASK_INDIRECT;ray.type=RT_PROBE_DIFFUSE;
        Payload payload{};const GPURtHit next=reflectionTraceVisible<Payload>(ray,as,ift,p,instances,meshes,vertices,indices,materials,payload,counts);
        if(counts)atomic_fetch_add_explicit(counts+18,1u,memory_order_relaxed);
        const float roughness=m.roughness;direction=normalize(r);ReflectionMirror following{};
        if(!next.hit){uint path;L+=throughput*reflectionProbeRadiance(m.point,direction,roughness,false,p,probes,atlas,path);break;}
        L+=throughput*reflectionSecondary<Payload>(next,direction,as,ift,p,instances,meshes,vertices,indices,materials,textures,lights,sampled,emitters,gi,states,extra,irradiance,moments,
            seed,numericError,minRoughness,nullptr,&following);
        m=following;
        if(bounce+1u==maxBounces&&m.mirror&&counts)atomic_fetch_add_explicit(counts+19,1u,memory_order_relaxed);
    }
    return L;
}
#ifdef PHOSPHOR_REFL_LOBE
// ---------------------------------------------------------------------------------------------------------------------------------------
// F15.1 (ticket 13): ray-traced KHR_materials_transmission + KHR_materials_volume (ADR-0004, decisions 06/09).
//
// A refraction sample of a transmissive pixel crosses at most kTransmissionMaxCrossings interfaces (the camera-visible refraction is the first) and
// meets at most kTransmissionMaxEvents interface events (reflections included: total internal reflection in a cube).
// Every interface is a GGX dielectric of the glass material it belongs to: the half vector is drawn from D(h) n.h, the Fresnel factor chooses
// reflection or refraction, the weight is the separable-Smith Walter weight (transmission_math.h) times the baseColor tint: once per closed solid at the ENTERING interface (including the
// camera-visible one, through the albedo) and at every thin-walled surface, never at an exit; the transmission factor applies at the camera-visible surface only: interior crossings are pure dielectric interfaces.  A medium
// entered through a front face of a closed volume absorbs along the REAL inside segment (Beer-Lambert c^(x/d)) and ends at ANY hit; the inside
// back faces of single-sided closed meshes are interfaces like the front faces (the reflection estimator rejects them: reflectionMaterial).
// Thin-walled glass (thickness 0) keeps the direction on transmission, no medium.  A ray that leaves the glass ends at the first non-glass
// surface (lit radiance: emitters hit by refracted rays are owned by the transport) or at the probes/environment; the walk never exceeds
// kTransmissionMaxEvents traces; a fifth crossing ends in the probes/environment along the refracted direction.
// ---------------------------------------------------------------------------------------------------------------------------------------
// Visible-normal sampling of the isotropic GGX distribution (Heitz 2018): the half vector is drawn from D(h) G1(v) |v.h| / |n.v|, so the one-sample
// estimator of a reflection or a refraction through it weighs G1(o) <= 1 (bounded: the D(h) n.h proposal weighs |i.h| G / (|i.n| |h.n|), unbounded for
// rough interfaces, and made the rough refraction of a three-interface walk fire).  Returns the world half vector for the unit view direction `v`.
inline float3 reflectionSampleVisibleNormal(float3 n,float3 v,float alpha,float2 u){
    const float3 t=reflectionTangent(n),b=cross(n,t);const float3 ve=float3(dot(v,t),dot(v,b),dot(v,n));
    const float3 vh=normalize(float3(alpha*ve.x,alpha*ve.y,ve.z));const float lensq=vh.x*vh.x+vh.y*vh.y;
    const float3 t1=lensq>1e-12f?float3(-vh.y,vh.x,0.0f)*rsqrt(lensq):float3(1,0,0),t2=cross(vh,t1);
    const float r=sqrt(u.x),phi=2*M_PI_F*u.y,p1=r*cos(phi);float p2=r*sin(phi);const float s=.5f*(1+vh.z);
    p2=(1-s)*sqrt(max(0.0f,1-p1*p1))+s*p2;
    const float3 nh=p1*t1+p2*t2+sqrt(max(0.0f,1-p1*p1-p2*p2))*vh;
    const float3 ne=normalize(float3(alpha*nh.x,alpha*nh.y,max(0.0f,nh.z)));
    return t*ne.x+b*ne.y+n*ne.z;
}
struct ReflectionGlassHit {
    bool glass;           // a transmissive interface the walk refracts through
    bool thin;            // thin-walled: no bend, no medium
    bool entering;        // the ray goes from the outside into the closed volume
    float3 point, normal, geometric; // world point, shading normal and geometric normal on the side of the INCOMING ray (unit)
    float alpha, ior;
    float3 tint, attenuationColor;
    float attenuationDistance;
};
inline ReflectionGlassHit reflectionGlassHit(GPURtHit hit,float3 direction,constant GPUReflectionParams& p,const device GPUInstance* instances,
    const device GPURtMesh* meshes,const device GPUVertex* vertices,const device uint* indices,const device GPUMaterial* materials,
    const device DITextureHandle* textures,const device GPUMaterialExtRecord* extRecords){
    ReflectionGlassHit g{};
    if(hit.slot>=p.slotCount)return g;const GPUInstance instance=instances[hit.slot];
    if(instance.meshIndex>=p.meshCount||instance.materialIndex>=p.materialCount)return g;
    const GPUMaterial m=materials[instance.materialIndex];
    if(!(m.flags&(MATERIAL_FEATURE_TRANSMISSION<<MATERIAL_FEATURE_SHIFT))||m.extensionIndex==MATERIAL_EXTENSION_INVALID)return g;
    const GPUMaterialExt0 e0=*reinterpret_cast<const device GPUMaterialExt0*>(extRecords+m.extensionIndex);
    const GPUMaterialExt1 e1=*reinterpret_cast<const device GPUMaterialExt1*>(extRecords+m.extensionIndex+1u);
    // The transmission factor itself only matters at the camera-visible surface; an interface needs a factor above 0 to be glass at all.
    if(!(e1.transmissionFactor>0.0f)||!(e0.ior>=1.0f))return g;
    const GPURtMesh mesh=meshes[instance.meshIndex];if(hit.primitive>=mesh.indexCount/3u)return g;
    float3 point,geometric;if(!rtSurface(hit,instances,meshes,vertices,indices,p.slotCount,p.meshCount,point,geometric))return g;
    const uint base=mesh.indexOffset+3u*hit.primitive;const GPUVertex a=vertices[mesh.vertexOffset+indices[base]],b=vertices[mesh.vertexOffset+indices[base+1]],c=vertices[mesh.vertexOffset+indices[base+2]];
    const float3 bary(1-hit.u-hit.v,hit.u,hit.v);const float2 uv=float2(a.u,a.v)*bary.x+float2(b.u,b.v)*bary.y+float2(c.u,c.v)*bary.z;
    const float4x4 model=float4x4(float4(instance.modelMatrix[0],instance.modelMatrix[1],instance.modelMatrix[2],instance.modelMatrix[3]),float4(instance.modelMatrix[4],instance.modelMatrix[5],instance.modelMatrix[6],instance.modelMatrix[7]),float4(instance.modelMatrix[8],instance.modelMatrix[9],instance.modelMatrix[10],instance.modelMatrix[11]),float4(instance.modelMatrix[12],instance.modelMatrix[13],instance.modelMatrix[14],instance.modelMatrix[15]));
    float3 outward=surfaceNormal(model,float3(a.nx,a.ny,a.nz)*bary.x+float3(b.nx,b.ny,b.nz)*bary.y+float3(c.nx,c.ny,c.nz)*bary.z);
    if(!all(isfinite(outward))||dot(outward,outward)<=0)return g;outward=normalize(outward);
    if(dot(geometric,outward)<0)geometric=-geometric; // the geometric normal agrees with the outward shading normal (a mirrored winding flips the cross product)
    const bool volume=(m.flags&(MATERIAL_FEATURE_VOLUME<<MATERIAL_FEATURE_SHIFT))!=0u&&e1.volumeThickness>0.0f;
    g.glass=true;g.thin=!volume;g.entering=dot(direction,geometric)<0;
    const float side=g.thin?(dot(direction,geometric)<0?1.0f:-1.0f):(g.entering?1.0f:-1.0f); // the side the incoming ray is on
    g.normal=side*outward;g.geometric=side*geometric;g.point=point;
    float rough=m.roughness;if(m.metallicRoughnessTex!=INVALID_TEXTURE_INDEX)rough*=float(half4(textures[m.metallicRoughnessTex].tex.sample(kGiMaterialSampler,texXfSlotUv(extRecords,m.flags,m.textureTransformIndex,TT_SLOT_METAL_ROUGH,uv),level(0))).g);
    g.alpha=bsdfAlpha(clamp(rough,.04f,1.0f));g.ior=e0.ior;
    float3 tint=float3(m.baseColor[0],m.baseColor[1],m.baseColor[2]);if(m.baseColorTex!=INVALID_TEXTURE_INDEX)tint*=float3(half3(textures[m.baseColorTex].tex.sample(kGiMaterialSampler,texXfSlotUv(extRecords,m.flags,m.textureTransformIndex,TT_SLOT_BASE_COLOR,uv),level(0)).rgb));
    g.tint=tint;g.attenuationColor=float3(e1.attenuationColor[0],e1.attenuationColor[1],e1.attenuationColor[2]);g.attenuationDistance=e1.attenuationDistance;
    return g;
}
// The medium the ray travels in: the closed volume entered through a front face (decision: it ends at any hit).
struct ReflectionMedium {bool active;float3 color;float distance;};
// One interface event: draws the half vector, chooses reflection or refraction by Fresnel, updates the ray and the weight.  `normal` faces the
// incoming direction `d`.  Returns 0 when the sample dies (a facet that cannot reflect or refract the ray, an invalid weight), 1 to continue, 2 when the
// event crossed the interface beyond the bound (kTransmissionMaxCrossings): the ray ends in the probes/environment along its new direction.
inline uint reflectionGlassEvent(thread float3& d,thread float3& weight,thread ReflectionMedium& medium,ReflectionGlassHit g,thread uint& seed,
    thread float3& side,thread uint& crossings,thread bool& numericError,bool tintInterior){
    const float3 n=g.normal,i=-d;const float cosN=dot(i,n);if(!(cosN>0))return 0;
    const float2 u=float2(giRandom(seed),giRandom(seed));
    // A material at the roughness floor (alpha <= kTransmissionSmoothAlpha: roughness 0 clamps to it) is a smooth interface: h = n, weight 1, the delta
    // the reference's `dielectric` is (the floor's own GGX tail would blur the refracted image the reference keeps sharp).
    const bool smooth=g.alpha<=phosphor::kTransmissionSmoothAlpha;
    const float3 h=smooth?n:reflectionSampleVisibleNormal(n,i,g.alpha,u);const float iH=dot(i,h);if(!(iH>0))return 0;
    const float eta=(g.thin||g.entering)?g.ior:1.0f/g.ior;
    const TransmissionFresnel fr=phosphor::transmissionFresnel(iH,eta);
    float3 o=2.0f*iH*h-i;float oN=dot(o,n);float wk;
    if(giRandom(seed)<fr.f){ // reflection: stays on the incoming side, inside the same medium
        if(!(oN>0))return 0;
        wk=phosphor::transmissionG1(oN,dot(o,h),g.alpha);side=g.geometric;
    }else{
        float3 refracted;
        if(g.thin){ // thin-walled: the reflected direction mirrored through the surface keeps the ray going (no bend)
            if(!(oN>0))return 0;wk=phosphor::transmissionG1(oN,dot(o,h),g.alpha);refracted=o-2.0f*oN*n;
        }else{
            const float ratio=1.0f/eta;refracted=-ratio*i+(ratio*iH-fr.cosT)*h;const float rN=dot(refracted,n);if(!(rN<0))return 0;
            wk=phosphor::transmissionG1(rN,dot(refracted,h),g.alpha);
            // entering a closed volume starts its medium, leaving ends it
            if(g.entering){medium.active=true;medium.color=g.attenuationColor;medium.distance=g.attenuationDistance;}else medium.active=false;
        }
        o=refracted;side=-g.geometric;++crossings;
        // F15 closure (ticket 13): baseColor tints a closed solid ONCE, when the ray enters it (the exit is a pure dielectric interface: the medium does the
        // rest) and every thin-walled surface (each is an independent pane); the camera-visible surface carries its own tint in the weight (albedo).
        if(tintInterior&&(g.thin||g.entering))weight*=g.tint;
    }
    if(!(isfinite(wk))||!all(isfinite(o))||dot(o,o)<=0){numericError=true;return 0;}
    weight*=wk;d=normalize(o);
    if(!any(weight>0))return 0;
    return crossings>kTransmissionMaxCrossings?2u:1u;
}
// The radiance a ray of weight `weight` brings back after the refraction at the camera-visible surface, walked through the glass it meets.
// `origin` already stands on the far side of the camera-visible surface.  distance = the first segment, slot/generation = the first non-glass hit.
inline float3 reflectionRefractWalk(float3 origin,float3 direction,float3 weight,ReflectionMedium medium,
    instance_acceleration_structure as,intersection_function_table<triangle_data,instancing> ift,
    constant GPUReflectionParams& p,const device GPUInstance* instances,const device GPURtMesh* meshes,
    const device GPUVertex* vertices,const device uint* indices,const device GPUMaterial* materials,
    const device DITextureHandle* textures,const device GPUMaterialExtRecord* extRecords,const device GPULight* lights,const device GPUSampledLight* sampled,
    const device GPUEmissiveSurface* emitters,constant GPUProbeGridParams& gi,const device GPUProbeState* states,
    constant GPUProbeTraceExtra& extra,texture2d<float> irradiance,texture2d<float> moments,
    const device GPUReflectionProbe* probes,texturecube_array<float> atlas,float floorRoughness,
    thread uint& seed,thread bool& numericError,thread float& firstDistance,thread uint& hitSlot,thread uint& hitGeneration,device atomic_uint* counts){
    float3 radiance=0,o=origin,d=direction;uint crossings=1u;uint path;firstDistance=0;
    for(uint step=0;step<kTransmissionMaxEvents;++step){
        GPURtRay ray{};ray.ox=o.x;ray.oy=o.y;ray.oz=o.z;ray.dx=d.x;ray.dy=d.y;ray.dz=d.z;ray.tmax=p.maxDistance;ray.mask=RT_MASK_INDIRECT;ray.type=RT_PROBE_DIFFUSE;
        RtPayload payload{};const auto hit=rtTraceSurface(ray,as,ift,instances,p.slotCount,payload,meshes,vertices,indices,p.meshCount);
        if(!hit.hit){radiance+=weight*reflectionProbeRadiance(o,d,0.0f,false,p,probes,atlas,path);break;}
        if(step==0)firstDistance=hit.t;
        if(medium.active)weight*=float3(phosphor::transmissionBeerLambert(medium.color.x,medium.distance,hit.t),phosphor::transmissionBeerLambert(medium.color.y,medium.distance,hit.t),
                                        phosphor::transmissionBeerLambert(medium.color.z,medium.distance,hit.t));
        const ReflectionGlassHit g=reflectionGlassHit(hit,d,p,instances,meshes,vertices,indices,materials,textures,extRecords);
        if(!g.glass&&reflectionCulledBackFace(hit,p,instances,materials,true)){ // the back of a single-sided opaque surface: culled, the ray goes on (one more event)
            if(counts)atomic_fetch_add_explicit(counts+16,1u,memory_order_relaxed);
            o=rtOffsetRay(o+d*hit.t,d);
            if(step+1u==kTransmissionMaxEvents){if(counts)atomic_fetch_add_explicit(counts+17,1u,memory_order_relaxed);radiance+=weight*reflectionProbeRadiance(o,d,0.0f,false,p,probes,atlas,path);}
            continue;
        }
        if(!g.glass){
            radiance+=weight*reflectionSecondary(hit,d,as,ift,p,instances,meshes,vertices,indices,materials,textures,lights,sampled,emitters,gi,states,extra,irradiance,moments,seed,numericError,floorRoughness);
            hitSlot=hit.slot;hitGeneration=hit.generation;break;
        }
        float3 side;
        const uint status=reflectionGlassEvent(d,weight,medium,g,seed,side,crossings,numericError,(p.flags&REFLECTION_DEBUG_TINT_CAMERA_ONLY)==0u);
        if(status==0u)break;
        if(status==2u){radiance+=weight*reflectionProbeRadiance(g.point,d,0.0f,false,p,probes,atlas,path);break;}
        o=rtOffsetRay(g.point,side);
        if(step+1u==kTransmissionMaxEvents)radiance+=weight*reflectionProbeRadiance(o,d,0.0f,false,p,probes,atlas,path); // the event bound ends the ray in the environment
    }
    return radiance;
}
// F15.1 (ticket 13, temporal clip): the ENDPOINT of the refracted path of a transmissive pixel, for the endpoint-matched denoiser history
// (renderer/denoise_endpoint.h).  A DETERMINISTIC walk (no random numbers, no shading): the camera-visible surface refracts at h = n (thin-walled: the
// ray stays straight), every glass interface after it refracts (reflects only under total internal reflection), back faces of single-sided opaque
// surfaces are skipped like in the transport, and the walk ends at the first non-glass hit (key = slot + 1, point = the OBJECT-space position of the
// hit from the vertices and the barycentric coordinates: a moving panel and an orbiting camera keep the same point) or in the environment (key 0, point
// = the final ray direction: a miss, the crossing bound or the event bound, like the transport).  At most kTransmissionMaxEvents traces.
struct ReflectionEndpoint {bool valid;uint key;float3 point;};
inline ReflectionEndpoint reflectionRefractEndpoint(GPUDISurface s,ReflectionLobe lobe,
    instance_acceleration_structure as,intersection_function_table<triangle_data,instancing> ift,
    constant GPUReflectionParams& p,const device GPUInstance* instances,const device GPURtMesh* meshes,
    const device GPUVertex* vertices,const device uint* indices,const device GPUMaterial* materials,
    const device DITextureHandle* textures,const device GPUMaterialExtRecord* extRecords){
    ReflectionEndpoint e{};float3 n=diVec(s.shadingNormal),v=diVec(s.viewDirection);
    if(!s.valid||!all(isfinite(n))||!all(isfinite(v))||dot(n,n)<=0||dot(v,v)<=0)return e;
    n=normalize(n);v=normalize(v);float3 d;
    if(!(lobe.transThickness>0))d=-v;else{d=refract(-v,n,1.0f/lobe.transIor);if(!(dot(d,d)>0)||!all(isfinite(d)))return e;}
    d=normalize(d);float3 ng=normalize(diVec(s.geometricNormal));if(dot(ng,d)<0)ng=-ng;float3 o=rtOffsetRay(diVec(s.position),ng);uint crossings=1u;
    for(uint step=0;step<kTransmissionMaxEvents;++step){
        GPURtRay ray{};ray.ox=o.x;ray.oy=o.y;ray.oz=o.z;ray.dx=d.x;ray.dy=d.y;ray.dz=d.z;ray.tmax=p.maxDistance;ray.mask=RT_MASK_INDIRECT;ray.type=RT_PROBE_DIFFUSE;
        RtPayload payload{};const auto hit=rtTraceSurface(ray,as,ift,instances,p.slotCount,payload,meshes,vertices,indices,p.meshCount);
        if(!hit.hit)break;
        const ReflectionGlassHit g=reflectionGlassHit(hit,d,p,instances,meshes,vertices,indices,materials,textures,extRecords);
        if(!g.glass&&reflectionCulledBackFace(hit,p,instances,materials,true)){o=rtOffsetRay(o+d*hit.t,d);continue;}
        if(!g.glass){
            if(hit.slot>=p.slotCount)return e;const GPUInstance instance=instances[hit.slot];if(instance.meshIndex>=p.meshCount)return e;
            const GPURtMesh mesh=meshes[instance.meshIndex];if(hit.primitive>=mesh.indexCount/3u)return e;
            const uint base=mesh.indexOffset+3u*hit.primitive;const GPUVertex a=vertices[mesh.vertexOffset+indices[base]],b=vertices[mesh.vertexOffset+indices[base+1]],c=vertices[mesh.vertexOffset+indices[base+2]];
            const float3 bary(1-hit.u-hit.v,hit.u,hit.v);
            e.point=float3(a.px,a.py,a.pz)*bary.x+float3(b.px,b.py,b.pz)*bary.y+float3(c.px,c.py,c.pz)*bary.z;if(!all(isfinite(e.point)))return ReflectionEndpoint{};
            e.key=denoiseEndpointKeyOfSlot(hit.slot);e.valid=true;return e;
        }
        float3 side=-g.geometric;
        if(!g.thin){const float eta=g.entering?g.ior:1.0f/g.ior;const float3 r=refract(d,g.normal,1.0f/eta);
            if(dot(r,r)>0&&all(isfinite(r))){d=normalize(r);++crossings;}else{d=reflect(d,g.normal);side=g.geometric;}}
        else ++crossings;
        if(crossings>kTransmissionMaxCrossings)break;
        o=rtOffsetRay(g.point,side);
    }
    e.valid=all(isfinite(d));e.key=0u;e.point=d;return e;
}
// One refraction sample of a transmissive pixel (the single Fresnel-chosen sample of ADR-0004): the camera-visible surface refracts through a
// half vector drawn like the reflection proposal (the same random numbers), weight (1 - max F) t (1 - metallic) tint G |v.h| / (|n.v| n.h) over the
// branch probability 1 - pReflect.  A refraction is ALWAYS traced, whatever the reflection roughness band.
inline GPUSpecularSample reflectionRefractSample(GPUDISurface s,float2 u,ReflectionLobe lobe,
    instance_acceleration_structure as,intersection_function_table<triangle_data,instancing> ift,
    constant GPUReflectionParams& p,const device GPUInstance* instances,const device GPURtMesh* meshes,
    const device GPUVertex* vertices,const device uint* indices,const device GPUMaterial* materials,
    const device DITextureHandle* textures,const device GPUMaterialExtRecord* extRecords,const device GPULight* lights,const device GPUSampledLight* sampled,
    const device GPUEmissiveSurface* emitters,constant GPUProbeGridParams& gi,const device GPUProbeState* states,
    constant GPUProbeTraceExtra& extra,texture2d<float> irradiance,texture2d<float> moments,
    const device GPUReflectionProbe* probes,texturecube_array<float> atlas,thread uint& seed,device atomic_uint* counts){
    GPUSpecularSample out=reflectionEmpty();if(!s.valid)return out;
    float3 n=diVec(s.shadingNormal),v=diVec(s.viewDirection);if(!all(isfinite(n))||!all(isfinite(v))||dot(n,n)<=0||dot(v,v)<=0){out.flags|=SPECULAR_SAMPLE_ERROR;return out;}
    n=normalize(n);v=normalize(v);out.flags|=SPECULAR_SAMPLE_VALID|SPECULAR_SAMPLE_STOCHASTIC;
    const float alpha=bsdfAlpha(s.roughness);const bool smooth=alpha<=phosphor::kTransmissionSmoothAlpha; // a smooth interface: h = n (see reflectionGlassEvent)
    const float3 h=smooth?n:reflectionSampleVisibleNormal(n,v,alpha,u);const float c=dot(n,h),vh=dot(v,h),nv=dot(n,v);
    if(!(vh>0)||!(nv>0))return out;
    const bool thin=!(lobe.transThickness>0);float3 direction;float wk;
    if(thin){const float3 mirror=2.0f*vh*h-v;const float mN=dot(mirror,n);if(!(mN>0))return out;wk=phosphor::transmissionG1(mN,dot(mirror,h),alpha);direction=mirror-2.0f*mN*n;}
    else{const float ratio=1.0f/lobe.transIor;const TransmissionFresnel fr=phosphor::transmissionFresnel(vh,lobe.transIor);
        direction=-ratio*v+(ratio*vh-fr.cosT)*h;const float dN=dot(direction,n);if(fr.tir||!(dN<0))return out;wk=phosphor::transmissionG1(dN,dot(direction,h),alpha);}
    // The Fresnel layer of the glTF composition keeps (1 - max F) of the base (diffuse mixed with the tinted BTDF) for the transmitted lobe.
    const float layer=1.0f-bsdfMax3(bsdfSpecFresnel(lobe.spec,saturate(vh)));
    float3 weight=(layer*lobe.transFactor*(1.0f-saturate(s.metallic))*wk/max(1.0f-lobe.pReflect,1e-4f))*max(diVec(s.albedo),0.0f);
    bool numericError=false;
    if(!all(isfinite(weight))||!all(isfinite(direction))||dot(direction,direction)<=0){out.flags|=SPECULAR_SAMPLE_ERROR;return out;}
    direction=normalize(direction);for(uint i=0;i<3;++i)out.direction[i]=direction[i];out.proposalSolidAngle=bsdfDGGX(c,alpha)*c/(4*vh);
    float3 ng=normalize(diVec(s.geometricNormal));if(dot(ng,direction)<0)ng=-ng;const float3 origin=rtOffsetRay(diVec(s.position),ng);
    ReflectionMedium medium{};
    if(!thin){medium.active=true;medium.color=lobe.transAttenuationColor;medium.distance=lobe.transAttenuationDistance;}
    float distance=0;uint hitSlot=~0u,hitGeneration=0;
    const float floorRoughness=sqrt(min(1.0f,max(p.regularizationScale,1.0f)*bsdfAlpha(s.roughness)));
    float3 L=reflectionRefractWalk(origin,direction,weight,medium,as,ift,p,instances,meshes,vertices,indices,materials,textures,extRecords,lights,sampled,emitters,gi,states,extra,irradiance,moments,
        probes,atlas,floorRoughness,seed,numericError,distance,hitSlot,hitGeneration,counts);
    out.path=REFLECTION_PATH_RT;out.hitDistance=distance;out.secondarySlot=hitSlot;out.secondaryGeneration=hitGeneration;
    if(!all(isfinite(L))||any(L<0)){numericError=true;L=0;}if(numericError)out.flags|=SPECULAR_SAMPLE_ERROR;
    for(uint i=0;i<3;++i)out.radiance[i]=L[i];return out;
}
#endif // PHOSPHOR_REFL_LOBE

kernel void REFL_KERNEL(reflection_ssr)(constant GPUReflectionParams& p [[buffer(0)]],const device GPUDISurface* surfaces [[buffer(1)]],
    const device GPURadianceCacheEntry* cache [[buffer(13)]],constant GPUProbeGridParams& gi [[buffer(14)]],
    device GPUSpecularSample* metadata [[buffer(17)]],const device GPUReflectionProbe* probes [[buffer(18)]],device atomic_uint* counts [[buffer(19)]]REFL_GUIDE_PARAM(20),
    texture2d<float,access::read> baseRadiance [[texture(0)]],texture2d<float,access::read> depth [[texture(1)]],texturecube_array<float> atlas [[texture(2)]],
    texture2d<float,access::write> output [[texture(5)]],texture2d<float,access::write> distance [[texture(6)]],uint tid [[thread_position_in_grid]]){
    if(tid>=p.width*p.height)return;uint seed=giHash(tid^giHash(p.frameIndex)^p.seed);const GPUDISurface s=surfaces[tid];
#ifdef PHOSPHOR_REFL_LOBE
    auto lobe=REFL_LOBE(p.flags,REFLECTION_LOBE_GUIDE_VALID,tid,s);const float2 u=float2(giRandom(seed),giRandom(seed));
    if(lobe.coat)reflectionLobePick(lobe,giRandom(seed)); // the coat or the base lobe by albedo weight (one-sample MIS)
    const auto dir=REFL_DIRECTION(s,u,lobe);const GPUDISurface sl=lobe.coat?reflectionOwnerSurface(s,lobe,dir.coatShare,giRandom(seed)):s;
#else
    const auto dir=reflectionDirection(s,float2(giRandom(seed),giRandom(seed)));const GPUDISurface sl=s;
#endif
    auto sample=reflectionFallback(sl,dir,p,surfaces,cache,gi,probes,baseRadiance,depth,atlas);
#ifdef PHOSPHOR_REFL_LOBE
    // F15.1 (ticket 13): without RT the transmitted radiance is the declared probe/environment term (never SSR: it cannot see behind the glass).
    if(lobe.trans&&(sample.flags&SPECULAR_SAMPLE_VALID)){uint transmittedPath;const float3 Lt=reflectionDeclaredTransmission(s,lobe,p,probes,atlas,transmittedPath);
        if(!all(isfinite(Lt))||any(Lt<0))sample.flags|=SPECULAR_SAMPLE_ERROR;else for(uint i=0;i<3;++i)sample.radiance[i]+=Lt[i];}
    REFL_TAG(sample,lobe);
#endif
    reflectionStoreSample(p.flags,tid,p.width,sample,metadata,counts,output,distance);
}

// F15 ticket 24: the base reflection_rt serves a scene without material features (no transmissive material), so it traces with the 16-byte
// RtPayloadLite and links rt_alpha_lite (shadow rays of the secondary hits: transmittance 1, exact there); the 124-byte payload slowed every
// traversal of the pass, multiplied by the back-face continuation (26fe229).  The lobe kernels keep RtPayload (glass needs the transmittance).
#ifdef PHOSPHOR_REFL_LOBE
typedef RtPayload ReflectionRtPayload;
#else
typedef RtPayloadLite ReflectionRtPayload;
#endif
kernel void REFL_KERNEL(reflection_rt)(constant GPUReflectionParams& p [[buffer(0)]],const device GPUDISurface* surfaces [[buffer(1)]],
    instance_acceleration_structure as [[buffer(2)]],intersection_function_table<triangle_data,instancing> ift [[buffer(3)]],
    const device GPUInstance* instances [[buffer(4)]],const device GPURtMesh* meshes [[buffer(5)]],const device GPUVertex* vertices [[buffer(6)]],const device uint* indices [[buffer(7)]],
    const device GPUMaterial* materials [[buffer(8)]],const device DITextureHandle* textures [[buffer(9)]],const device GPULight* lights [[buffer(10)]],
    const device GPUSampledLight* sampled [[buffer(11)]],const device GPUEmissiveSurface* emitters [[buffer(12)]],const device GPURadianceCacheEntry* cache [[buffer(13)]],
    constant GPUProbeGridParams& gi [[buffer(14)]],const device GPUProbeState* states [[buffer(15)]],constant GPUProbeTraceExtra& extra [[buffer(16)]],
    device GPUSpecularSample* metadata [[buffer(17)]],const device GPUReflectionProbe* probes [[buffer(18)]],device atomic_uint* counts [[buffer(19)]]REFL_GUIDE_PARAM(20)REFL_EXT_PARAM(21),
    texture2d<float,access::read> baseRadiance [[texture(0)]],texture2d<float,access::read> depth [[texture(1)]],texturecube_array<float> atlas [[texture(2)]],
    texture2d<float> irradiance [[texture(3)]],texture2d<float> moments [[texture(4)]],texture2d<float,access::write> output [[texture(5)]],
    texture2d<float,access::write> distance [[texture(6)]],uint tid [[thread_position_in_grid]]){
    if(tid>=p.width*p.height)return;const GPUDISurface s=surfaces[tid];
    // F13.6 start-up boost (single-sample graph only): for the first frames after
    // a specular history reset, surfaces in the RT band take boostSamples
    // independent samples, reduced in registers exactly like reflection_reduce.
    // Sample 0 keeps the single-sample seed, so count 1 is the plain path.
#ifdef PHOSPHOR_REFL_LOBE
    auto lobe=REFL_LOBE(p.flags,REFLECTION_LOBE_GUIDE_VALID,tid,s);
    // a coated pixel is in the boost band when its sharper lobe is
    const float boostRoughness=lobe.coat?min(s.roughness,lobe.coatRoughness):s.roughness;
    const uint baseLobeId=lobe.id;
#else
    const float boostRoughness=s.roughness;
#endif
    uint count=(p.boostSamples>1u&&(p.flags&REFLECTION_SINGLE_SAMPLE)&&s.valid&&boostRoughness<p.boostRoughness)?min(p.boostSamples,8u):1u;
#ifdef PHOSPHOR_REFL_LOBE
    // VERIFICATION ONLY (--debug-transmission-both-branches): a transmissive pixel evaluates both Fresnel branches (k = 0 refraction, k = 1 reflection),
    // each scaled back to its unweighted contribution (x (1 - pReflect) / x pReflect) and by the count the reduction divides by: the same expectation.
    const bool both=(p.flags&REFLECTION_DEBUG_BOTH_BRANCHES)!=0u&&lobe.trans&&s.valid;if(both)count=2u;
#endif
    auto reduction=reflectionReduceBegin();GPUSpecularSample sample=reflectionEmpty();
    for(uint k=0;k<count;++k){uint seed=giHash(tid^giHash(p.frameIndex)^(p.seed+k*0x9e3779b9u));
#ifdef PHOSPHOR_REFL_LOBE
    const float2 u=float2(giRandom(seed),giRandom(seed));
    // F15.1 (ticket 13): a transmissive pixel draws the branch first: reflection (today's lobes, weight / pReflect) or refraction (always traced).
    const float branchDraw=lobe.trans?giRandom(seed):0.0f; /* drawn only by transmissive pixels: every other pixel keeps its random sequence */ lobe.sampleRefract=lobe.trans&&(both?(k==0u):branchDraw>=lobe.pReflect);lobe.id=lobe.sampleRefract?SPECULAR_LOBE_TRANSMISSION:baseLobeId;lobe.refract=lobe.sampleRefract;
    if(lobe.sampleRefract){
        sample=reflectionRefractSample(s,u,lobe,as,ift,p,instances,meshes,vertices,indices,materials,textures,extRecords,lights,sampled,emitters,gi,states,extra,irradiance,moments,probes,atlas,seed,counts);
        if(both)for(uint i=0;i<3;++i)sample.radiance[i]*=float(count)*(1.0f-lobe.pReflect);
        REFL_TAG(sample,lobe);if(count>1u)reflectionReduceAdd(reduction,sample,count);continue;
    }
    if(lobe.coat)reflectionLobePick(lobe,giRandom(seed)); // the coat or the base lobe by albedo weight (one-sample MIS)
    auto dir=REFL_DIRECTION(s,u,lobe);sample=reflectionEmpty();
    if(lobe.trans)dir.weight/=lobe.pReflect; // one-sample estimator over the reflection and refraction branches
    const GPUDISurface sl=lobe.coat?reflectionOwnerSurface(s,lobe,dir.coatShare,giRandom(seed)):s;
#else
    const auto dir=reflectionDirection(s,float2(giRandom(seed),giRandom(seed)));sample=reflectionEmpty();const GPUDISurface sl=s;
#endif
    if(s.valid&&dir.valid&&any(dir.weight>0)&&giRandom(seed)<reflectionRTWeight(sl.roughness,p)){
        float3 n=normalize(diVec(s.geometricNormal));if(dot(n,dir.direction)<0)n=-n;const float3 origin=rtOffsetRay(diVec(s.position),n);GPURtRay ray{};
        ray.ox=origin.x;ray.oy=origin.y;ray.oz=origin.z;ray.dx=dir.direction.x;ray.dy=dir.direction.y;ray.dz=dir.direction.z;ray.tmax=p.maxDistance;ray.mask=RT_MASK_INDIRECT;ray.type=RT_PROBE_DIFFUSE;
        ReflectionRtPayload payload{};const auto hit=reflectionTraceVisible(ray,as,ift,p,instances,meshes,vertices,indices,materials,payload,counts);float3 L;
        sample.flags=SPECULAR_SAMPLE_VALID|SPECULAR_SAMPLE_STOCHASTIC;sample.proposalSolidAngle=dir.pdf;for(uint i=0;i<3;++i)sample.direction[i]=dir.direction[i];
        bool numericError=dir.error;
        // Declared regularisation (F13 reference): alpha' >= k * alpha_camera.
        const float floorRoughness=sqrt(min(1.0f,max(p.regularizationScale,1.0f)*bsdfAlpha(sl.roughness)));
        if(hit.hit){L=reflectionSecondaryChain<ReflectionRtPayload>(hit,dir.direction,(p.flags>>REFLECTION_BOUNCE_SHIFT)&REFLECTION_BOUNCE_MASK,as,ift,p,instances,meshes,vertices,indices,materials,textures,lights,sampled,emitters,gi,states,extra,irradiance,moments,
                probes,atlas,seed,numericError,floorRoughness,counts);
            sample.path=REFLECTION_PATH_RT;sample.hitDistance=length(origin+dir.direction*hit.t-diVec(s.position));sample.secondarySlot=hit.slot;sample.secondaryGeneration=hit.generation;}
        else L=reflectionProbeRadiance(diVec(s.position),dir.direction,sl.roughness,false,p,probes,atlas,sample.path);
        L*=dir.weight;if(!all(isfinite(L))||any(L<0)){numericError=true;L=0;}if(numericError)sample.flags|=SPECULAR_SAMPLE_ERROR;for(uint i=0;i<3;++i)sample.radiance[i]=L[i];
    }else sample=reflectionFallback(sl,dir,p,surfaces,cache,gi,probes,baseRadiance,depth,atlas);
#ifdef PHOSPHOR_REFL_LOBE
    if(both)for(uint i=0;i<3;++i)sample.radiance[i]*=float(count)*lobe.pReflect;
    REFL_TAG(sample,lobe);
#endif
    if(count>1u)reflectionReduceAdd(reduction,sample,count);}
#ifdef PHOSPHOR_REFL_LOBE
    // F15.1 (ticket 13, temporal clip): the refracted-path endpoint of a transmissive pixel, whichever Fresnel branch the frame drew and also for a boosted
    // multi-sample pixel (the start-up frames: their history must carry the endpoint too), for the endpoint-matched denoiser history.  Not in the
    // verification both-branch mode (its control never reaches a denoiser).
    ReflectionEndpoint endpoint{};
    if(lobe.trans&&!both&&s.valid)endpoint=reflectionRefractEndpoint(s,lobe,as,ift,p,instances,meshes,vertices,indices,materials,textures,extRecords);
    const bool endpointOk=endpoint.valid;
#endif
    if(count==1u){
#ifdef PHOSPHOR_REFL_LOBE
        if(endpointOk&&(sample.flags&SPECULAR_SAMPLE_VALID)&&!(sample.flags&SPECULAR_SAMPLE_ERROR)){for(uint i=0;i<3;++i)sample.direction[i]=endpoint.point[i];
            sample.proposalSolidAngle=denoiseEndpointKeyEncode(endpoint.key);sample.flags|=SPECULAR_SAMPLE_ENDPOINT;}
#endif
        reflectionStoreSample(p.flags,tid,p.width,sample,metadata,counts,output,distance);
    }else{auto reduced=reflectionReduceEnd(reduction,counts);reduced.flags|=count<<SPECULAR_SAMPLE_COUNT_SHIFT;
#ifdef PHOSPHOR_REFL_LOBE
        if(endpointOk&&(reduced.flags&SPECULAR_SAMPLE_VALID)&&!(reduced.flags&SPECULAR_SAMPLE_ERROR)){for(uint i=0;i<3;++i)reduced.direction[i]=endpoint.point[i];
            reduced.proposalSolidAngle=denoiseEndpointKeyEncode(endpoint.key);reduced.flags|=SPECULAR_SAMPLE_ENDPOINT;}
#endif
        reflectionWriteSample(tid,p.width,reduced,metadata,output,distance);}
}

#ifndef PHOSPHOR_REFL_LOBE
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
    RtPayload payload{};const auto hit=reflectionTraceVisible(ray,as,ift,p,instances,meshes,vertices,indices,materials,payload,nullptr);uint seed=giHash(tid^p.seed);bool numericError=false;
    float emitted=0;const float3 L=hit.hit?
        reflectionSecondary(hit,direction,as,ift,p,instances,meshes,vertices,indices,materials,textures,lights,sampled,emitters,gi,states,extra,irradiance,moments,seed,numericError,
            0.0f,&emitted):reflectionConstantVec(p.environment);
    const bool valid=!numericError&&reflectionProbeStorageAccepts(L.x,L.y,L.z,p.flags)&&isfinite(emitted);
    // Alpha: emission luminance (F13.1), never negative; -1 marks a rejected texel.
    const float alpha=(p.flags&REFLECTION_CAPTURE_HALF)?min(max(emitted,0.0f),65504.0f):max(emitted,0.0f);
    output.write(float4(valid?L:float3(0),valid?alpha:-1.0f),pixel);
}
#endif
