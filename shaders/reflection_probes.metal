#include "reflection_common.h"
#include "texture_transform_core.h" // F15.1 (ticket 22)
#include "renderer/dfg_table.h"

#ifndef PHOSPHOR_REFL_LOBE
// Actual base-compatible STATIC scene capture. Host draws the full selected
// static instance list for EACH face with a depth attachment; no camera cull
// list reused. This baseline is explicitly unshadowed local/direct capture.
// For shadow-correct capture use reflection_capture_rt and its own consumer
// IFT. Neither variant is a constant environment pretending to be a scene.
// Raster buffers: captureParams0,vertices1,instances2,materials3,lights4,
// DITextureHandles5,staticSlots6,sampledLights7,emitterRecords8.
struct ProbeVertexOut {float4 position [[position]];float3 world,normal,tangent;float handed;float2 uv;
    uint slot [[flat]],material [[flat]];};
vertex ProbeVertexOut reflection_probe_capture_vs(uint vertexID [[vertex_id]],uint instanceID [[instance_id]],
    constant GPUProbeCaptureParams& p [[buffer(0)]],const device GPUVertex* vertices [[buffer(1)]],
    const device GPUInstance* instances [[buffer(2)]],const device uint* staticSlots [[buffer(6)]]){
    ProbeVertexOut out{};const uint slot=staticSlots[instanceID];const GPUInstance i=instances[slot];const GPUVertex v=vertices[vertexID];
    float4x4 m=float4x4(float4(i.modelMatrix[0],i.modelMatrix[1],i.modelMatrix[2],i.modelMatrix[3]),float4(i.modelMatrix[4],i.modelMatrix[5],i.modelMatrix[6],i.modelMatrix[7]),float4(i.modelMatrix[8],i.modelMatrix[9],i.modelMatrix[10],i.modelMatrix[11]),float4(i.modelMatrix[12],i.modelMatrix[13],i.modelMatrix[14],i.modelMatrix[15]));
    out.world=(m*float4(v.px,v.py,v.pz,1)).xyz;out.position=reflectionMatrix(p.viewProjection)*float4(out.world,1);
    out.normal=surfaceNormal(m,float3(v.nx,v.ny,v.nz));out.tangent=float3x3(m[0].xyz,m[1].xyz,m[2].xyz)*float3(v.tx,v.ty,v.tz);
    out.handed=v.tw*((i.flags&INSTANCE_FLAG_MIRRORED)?-1.0f:1.0f);out.uv=float2(v.u,v.v);out.slot=slot;out.material=i.materialIndex;return out;
}
// F15.1 (ticket 22): a material texture of the capture surface sampled through its slot's KHR_texture_transform (the plain sample for a material
// without one: the values handed to the sampler are untouched).
static half4 probeSlotSample(const device DITextureHandle* textures,uint index,const device GPUMaterialExtRecord* extensions,const GPUMaterial m,
    uint slot,float2 uv,float2 dx,float2 dy){
    if(m.flags&MATERIAL_FLAG_TEXTURE_TRANSFORM)texXfUvOn(texXfForOn(extensions,m.flags,m.textureTransformIndex),slot,uv,dx,dy);
    return half4(textures[index].tex.sample(kGiMaterialSampler,uv,gradient2d(dx,dy)));
}
struct LayerGradSamples {
    template <typename T>
    static half4 sample(const device T* table,uint index,float2 uv,half4 fallback,float2 dx,float2 dy){return index==INVALID_TEXTURE_INDEX?fallback:half4(table[index].tex.sample(kGiMaterialSampler,uv,gradient2d(dx,dy)));}
    template <typename T>
    static half4 mask(const device T* table,uint index,float2 uv,half4 fallback,float2 dx,float2 dy){return index==INVALID_TEXTURE_INDEX?fallback:half4(table[index].tex.sample(kGiMaskSampler,uv,gradient2d(dx,dy)));}
};
fragment float4 reflection_probe_capture_fs(ProbeVertexOut in [[stage_in]],bool frontFacing [[front_facing]],
    constant GPUProbeCaptureParams& p [[buffer(0)]],const device GPUInstance* instances [[buffer(2)]],
    const device GPUMaterial* materials [[buffer(3)]],const device GPULight* lights [[buffer(4)]],const device DITextureHandle* textures [[buffer(5)]],
    const device GPUSampledLight* sampled [[buffer(7)]],const device GPUEmissiveSurface* emitters [[buffer(8)]]){
    if(in.slot>=p.slotCount||in.material>=p.materialCount)discard_fragment();const GPUMaterial m=materials[in.material];
    const float2 dx=dfdx(in.uv),dy=dfdy(in.uv);const half4 baseTex=m.baseColorTex==INVALID_TEXTURE_INDEX?half4(1):probeSlotSample(textures,m.baseColorTex,p.materialExtensions,m,TT_SLOT_BASE_COLOR,in.uv,dx,dy);
    if(m.baseColor[3]*float(baseTex.a)*layerCoverage<DITextureHandle,LayerAlphaGrad>(materials[in.material],materials,textures,in.uv,dx,dy,p.materialExtensions)<m.alphaCutoff)discard_fragment(); // F15.4 (ticket 23): a stack's coverage
    const bool front=frontFacing!=((instances[in.slot].flags&INSTANCE_FLAG_MIRRORED)!=0);float3 n=normalize(front?in.normal:-in.normal);
    const float3 geometric=normalize(cross(dfdx(in.world),dfdy(in.world)));float3 t=in.tangent-n*dot(n,in.tangent);
    // F15.4 (ticket 23): a layer stack's blended base parameters, normal and emission (gradient sampling of the capture, clamped mask).
    LayerSurface layered;const bool stack=layerStackSurface<DITextureHandle,LayerGradSamples>(materials[in.material],p.materialExtensions,materials,textures,in.uv,dx,dy,layered);
    const float3 nShade=n;const float3 tShade=t;
    if(m.normalTex!=INVALID_TEXTURE_INDEX&&dot(t,t)>1e-8f){t=normalize(t);float3 tn=float3(probeSlotSample(textures,m.normalTex,p.materialExtensions,m,TT_SLOT_NORMAL,in.uv,dx,dy).rgb)*2-1;tn.xy*=m.normalScale;
        if(m.flags&MATERIAL_FLAG_TEXTURE_TRANSFORM){float3 tt=t,bb=cross(n,t)*in.handed;texXfFrameOn(texXfForOn(p.materialExtensions,m.flags,m.textureTransformIndex),TT_SLOT_NORMAL,tt,bb);n=normalize(tt*tn.x+bb*tn.y+n*tn.z);}
        else n=normalize(t*tn.x+cross(n,t)*in.handed*tn.y+n*tn.z);}
    if(stack){if(dot(tShade,tShade)>1e-8f){const float3 tu=normalize(tShade);n=normalize(tu*layered.tn.x+cross(nShade,tu)*in.handed*layered.tn.y+nShade*layered.tn.z);}else n=nShade;}
    const half4 mr=m.metallicRoughnessTex==INVALID_TEXTURE_INDEX?half4(1):probeSlotSample(textures,m.metallicRoughnessTex,p.materialExtensions,m,TT_SLOT_METAL_ROUGH,in.uv,dx,dy);
    GPUDISurface s{};float3 base=float3(m.baseColor[0],m.baseColor[1],m.baseColor[2])*float3(baseTex.rgb);if(stack)base=layered.baseColor;const float3 v=normalize(reflectionConstantVec(p.capturePosition)-in.world);
    for(uint c=0;c<3;++c){s.position[c]=in.world[c];s.geometricNormal[c]=geometric[c];s.shadingNormal[c]=n[c];s.albedo[c]=base[c];s.viewDirection[c]=v[c];}
    s.valid=1;s.roughness=clamp(m.roughness*float(mr.g),.04f,1.0f);s.metallic=saturate(m.metallic*float(mr.b));if(stack){s.roughness=layered.roughness;s.metallic=layered.metallic;}float3 result=0;
    for(uint i=0;i<p.lightCount;++i){GPULight light=lights[i];if(p.sampledLightCount&&light.type!=LIGHT_DIRECTIONAL)continue;DISample sample{};
        if(light.type==LIGHT_DIRECTIONAL){sample.valid=true;sample.delta=true;sample.geometry=1;sample.wi=-normalize(float3(light.direction[0],light.direction[1],light.direction[2]));sample.radiance=float3(light.color[0],light.color[1],light.color[2])*light.intensity;}
        else{GPUSampledLight local{};local.type=light.type;local.range=light.range;local.innerCone=light.innerCone;local.outerCone=light.outerCone;
            for(uint c=0;c<3;++c){local.position[c]=light.position[c];local.axisU[c]=light.direction[c];local.emission[c]=light.color[c]*light.intensity;}sample=diSampleLight(local,float2(.5f),in.world);}
        result+=diBRDF(s,sample);
    }
    // Capture is loading-time work: one stratified 4x4 area quadrature per light
    // avoids depending on a noisy one-endpoint static cube map.
    for(uint i=0;i<p.sampledLightCount;++i)for(uint y=0;y<4;++y)for(uint x=0;x<4;++x){DISample sample=diSampleTexturedLight(sampled[i],i,(float2(x,y)+.5f)/4.0f,in.world,emitters,materials,textures,p.materialExtensions,p.vertexColors);
        if(sample.valid&&sample.pdfArea>0)result+=diBRDF(s,sample)/(16*sample.pdfArea);}
    float3 emission=float3(m.emissive[0],m.emissive[1],m.emissive[2]);if(m.emissiveTex!=INVALID_TEXTURE_INDEX)emission*=float3(probeSlotSample(textures,m.emissiveTex,p.materialExtensions,m,TT_SLOT_EMISSIVE,in.uv,dx,dy).rgb);if(stack)emission+=layered.emission;
    result+=emission+reflectionConstantVec(p.environment)*base*(1-s.metallic);
    // Validate the TOTAL physical sum before the attachment's narrowing store.
    // Negative alpha survives black repair and disables GPU probe publication.
    const bool finite=reflectionProbeStorageAccepts(result.x,result.y,result.z,p.storageFlags);
    // Alpha: emission luminance (F13.1), never negative; -1 marks a rejected texel.
    const float emitted=max(reflectionProbeLuminance(emission),0.0f),alpha=(p.storageFlags&REFLECTION_CAPTURE_HALF)?min(emitted,65504.0f):emitted;
    return float4(finite?result:float3(0),finite&&isfinite(alpha)?alpha:-1.0f);
}

// F15 closure (COLOR_0): the vertex-colour twins of the static probe capture (separate entry points: a function-constant interface member cannot be lowered in
// a constant-free descriptor).  Alpha test, base colour and the unlit emission multiply the interpolated vertex colour.
struct ProbeVertexOutVC {float4 position [[position]];float3 world,normal,tangent;float handed;float2 uv;
    uint slot [[flat]],material [[flat]];float4 color;};
vertex ProbeVertexOutVC reflection_probe_capture_vs_vc(uint vertexID [[vertex_id]],uint instanceID [[instance_id]],
    constant GPUProbeCaptureParams& p [[buffer(0)]],const device GPUVertex* vertices [[buffer(1)]],
    const device GPUInstance* instances [[buffer(2)]],const device uint* staticSlots [[buffer(6)]]){
    ProbeVertexOutVC out{};out.color=p.vertexColors[vertexID];const uint slot=staticSlots[instanceID];const GPUInstance i=instances[slot];const GPUVertex v=vertices[vertexID];
    float4x4 m=float4x4(float4(i.modelMatrix[0],i.modelMatrix[1],i.modelMatrix[2],i.modelMatrix[3]),float4(i.modelMatrix[4],i.modelMatrix[5],i.modelMatrix[6],i.modelMatrix[7]),float4(i.modelMatrix[8],i.modelMatrix[9],i.modelMatrix[10],i.modelMatrix[11]),float4(i.modelMatrix[12],i.modelMatrix[13],i.modelMatrix[14],i.modelMatrix[15]));
    out.world=(m*float4(v.px,v.py,v.pz,1)).xyz;out.position=reflectionMatrix(p.viewProjection)*float4(out.world,1);
    out.normal=surfaceNormal(m,float3(v.nx,v.ny,v.nz));out.tangent=float3x3(m[0].xyz,m[1].xyz,m[2].xyz)*float3(v.tx,v.ty,v.tz);
    out.handed=v.tw*((i.flags&INSTANCE_FLAG_MIRRORED)?-1.0f:1.0f);out.uv=float2(v.u,v.v);out.slot=slot;out.material=i.materialIndex;return out;
}
fragment float4 reflection_probe_capture_fs_vc(ProbeVertexOutVC in [[stage_in]],bool frontFacing [[front_facing]],
    constant GPUProbeCaptureParams& p [[buffer(0)]],const device GPUInstance* instances [[buffer(2)]],
    const device GPUMaterial* materials [[buffer(3)]],const device GPULight* lights [[buffer(4)]],const device DITextureHandle* textures [[buffer(5)]],
    const device GPUSampledLight* sampled [[buffer(7)]],const device GPUEmissiveSurface* emitters [[buffer(8)]]){
    if(in.slot>=p.slotCount||in.material>=p.materialCount)discard_fragment();const GPUMaterial m=materials[in.material];
    const float2 dx=dfdx(in.uv),dy=dfdy(in.uv);const half4 baseTex=m.baseColorTex==INVALID_TEXTURE_INDEX?half4(1):probeSlotSample(textures,m.baseColorTex,p.materialExtensions,m,TT_SLOT_BASE_COLOR,in.uv,dx,dy);
    if(m.baseColor[3]*float(baseTex.a)*in.color.a*layerCoverage<DITextureHandle,LayerAlphaGrad>(materials[in.material],materials,textures,in.uv,dx,dy,p.materialExtensions)<m.alphaCutoff)discard_fragment(); // F15.4 (ticket 23): a stack's coverage
    const bool front=frontFacing!=((instances[in.slot].flags&INSTANCE_FLAG_MIRRORED)!=0);float3 n=normalize(front?in.normal:-in.normal);
    const float3 geometric=normalize(cross(dfdx(in.world),dfdy(in.world)));float3 t=in.tangent-n*dot(n,in.tangent);
    // F15.4 (ticket 23): a layer stack's blended base parameters, normal and emission (gradient sampling of the capture, clamped mask).
    LayerSurface layered;const bool stack=layerStackSurface<DITextureHandle,LayerGradSamples>(materials[in.material],p.materialExtensions,materials,textures,in.uv,dx,dy,layered);
    const float3 nShade=n;const float3 tShade=t;
    if(m.normalTex!=INVALID_TEXTURE_INDEX&&dot(t,t)>1e-8f){t=normalize(t);float3 tn=float3(probeSlotSample(textures,m.normalTex,p.materialExtensions,m,TT_SLOT_NORMAL,in.uv,dx,dy).rgb)*2-1;tn.xy*=m.normalScale;
        if(m.flags&MATERIAL_FLAG_TEXTURE_TRANSFORM){float3 tt=t,bb=cross(n,t)*in.handed;texXfFrameOn(texXfForOn(p.materialExtensions,m.flags,m.textureTransformIndex),TT_SLOT_NORMAL,tt,bb);n=normalize(tt*tn.x+bb*tn.y+n*tn.z);}
        else n=normalize(t*tn.x+cross(n,t)*in.handed*tn.y+n*tn.z);}
    if(stack){if(dot(tShade,tShade)>1e-8f){const float3 tu=normalize(tShade);n=normalize(tu*layered.tn.x+cross(nShade,tu)*in.handed*layered.tn.y+nShade*layered.tn.z);}else n=nShade;}
    const half4 mr=m.metallicRoughnessTex==INVALID_TEXTURE_INDEX?half4(1):probeSlotSample(textures,m.metallicRoughnessTex,p.materialExtensions,m,TT_SLOT_METAL_ROUGH,in.uv,dx,dy);
    GPUDISurface s{};float3 base=float3(m.baseColor[0],m.baseColor[1],m.baseColor[2])*float3(baseTex.rgb)*in.color.rgb;if(stack)base=layered.baseColor;const float3 v=normalize(reflectionConstantVec(p.capturePosition)-in.world);
    for(uint c=0;c<3;++c){s.position[c]=in.world[c];s.geometricNormal[c]=geometric[c];s.shadingNormal[c]=n[c];s.albedo[c]=base[c];s.viewDirection[c]=v[c];}
    s.valid=1;s.roughness=clamp(m.roughness*float(mr.g),.04f,1.0f);s.metallic=saturate(m.metallic*float(mr.b));if(stack){s.roughness=layered.roughness;s.metallic=layered.metallic;}float3 result=0;
    for(uint i=0;i<p.lightCount;++i){GPULight light=lights[i];if(p.sampledLightCount&&light.type!=LIGHT_DIRECTIONAL)continue;DISample sample{};
        if(light.type==LIGHT_DIRECTIONAL){sample.valid=true;sample.delta=true;sample.geometry=1;sample.wi=-normalize(float3(light.direction[0],light.direction[1],light.direction[2]));sample.radiance=float3(light.color[0],light.color[1],light.color[2])*light.intensity;}
        else{GPUSampledLight local{};local.type=light.type;local.range=light.range;local.innerCone=light.innerCone;local.outerCone=light.outerCone;
            for(uint c=0;c<3;++c){local.position[c]=light.position[c];local.axisU[c]=light.direction[c];local.emission[c]=light.color[c]*light.intensity;}sample=diSampleLight(local,float2(.5f),in.world);}
        result+=diBRDF(s,sample);
    }
    // Capture is loading-time work: one stratified 4x4 area quadrature per light
    // avoids depending on a noisy one-endpoint static cube map.
    for(uint i=0;i<p.sampledLightCount;++i)for(uint y=0;y<4;++y)for(uint x=0;x<4;++x){DISample sample=diSampleTexturedLight(sampled[i],i,(float2(x,y)+.5f)/4.0f,in.world,emitters,materials,textures,p.materialExtensions,p.vertexColors);
        if(sample.valid&&sample.pdfArea>0)result+=diBRDF(s,sample)/(16*sample.pdfArea);}
    float3 emission=float3(m.emissive[0],m.emissive[1],m.emissive[2]);if(m.emissiveTex!=INVALID_TEXTURE_INDEX)emission*=float3(probeSlotSample(textures,m.emissiveTex,p.materialExtensions,m,TT_SLOT_EMISSIVE,in.uv,dx,dy).rgb);if(stack)emission+=layered.emission;
    if(m.flags&MATERIAL_FLAG_UNLIT)emission*=in.color.rgb; // KHR_materials_unlit shows base colour x COLOR_0 through its emission
    result+=emission+reflectionConstantVec(p.environment)*base*(1-s.metallic);
    // Validate the TOTAL physical sum before the attachment's narrowing store.
    // Negative alpha survives black repair and disables GPU probe publication.
    const bool finite=reflectionProbeStorageAccepts(result.x,result.y,result.z,p.storageFlags);
    // Alpha: emission luminance (F13.1), never negative; -1 marks a rejected texel.
    const float emitted=max(reflectionProbeLuminance(emission),0.0f),alpha=(p.storageFlags&REFLECTION_CAPTURE_HALF)?min(emitted,65504.0f):emitted;
    return float4(finite?result:float3(0),finite&&isfinite(alpha)?alpha:-1.0f);
}

inline float3 reflectionCubeDirection(uint face,float2 uv){const float2 p=uv*2-1;switch(face){case 0:return normalize(float3(1,-p.y,-p.x));case 1:return normalize(float3(-1,-p.y,p.x));case 2:return normalize(float3(p.x,1,p.y));case 3:return normalize(float3(p.x,-1,-p.y));case 4:return normalize(float3(p.x,-p.y,1));default:return normalize(float3(-p.x,-p.y,-1));}}
inline float reflectionRadicalInverse(uint v){v=(v<<16)|(v>>16);v=((v&0x55555555u)<<1)|((v&0xaaaaaaaau)>>1);v=((v&0x33333333u)<<2)|((v&0xccccccccu)>>2);v=((v&0x0f0f0f0fu)<<4)|((v&0xf0f0f0f0u)>>4);v=((v&0x00ff00ffu)<<8)|((v&0xff00ff00u)>>8);return float(v)*2.3283064365386963e-10f;}
// Params0; raw cubeArray texture0; selected destination mip viewed as 2DArray
// texture1. Dispatch side x side x6; write slice=6*cubeIndex+face. Every mip
// reads mip0 of a DIFFERENT source texture; never read/write the same resource.
kernel void reflection_probe_prefilter(constant GPUProbeFilterParams& p [[buffer(0)]],texturecube_array<float> raw [[texture(0)]],
    texture2d_array<float,access::write> filtered [[texture(1)]],uint3 pixel [[thread_position_in_grid]]){
    if(pixel.x>=p.side||pixel.y>=p.side||pixel.z>=6)return;const float3 n=reflectionCubeDirection(pixel.z,(float2(pixel.xy)+.5f)/float(p.side));float3 L=0;float weight=0;
    float emitted=0;
    if(p.roughness<=0){const float4 t=raw.sample(kReflectionProbeSampler,n,p.cubeIndex,level(0));L=t.rgb;emitted=t.a;weight=1;}
    else{const float3 t=reflectionTangent(n),b=cross(n,t);const float a=bsdfAlpha(p.roughness),a2=a*a;
        const uint count=clamp(p.sampleCount,1u,4096u);
        for(uint i=0;i<count;++i){float u=(float(i)+.5f)/float(count),v=reflectionRadicalInverse(i),c=sqrt((1-u)/(1+(a2-1)*u)),r=sqrt(max(0.0f,1-c*c)),phi=2*M_PI_F*v;
            const float3 h=t*(r*cos(phi))+b*(r*sin(phi))+n*c,l=reflect(-n,h);const float nl=max(0.0f,dot(n,l));const float4 t=raw.sample(kReflectionProbeSampler,l,p.cubeIndex,level(0));L+=t.rgb*nl;emitted+=t.a*nl;weight+=nl;}}
    const float3 value=weight>0?L/weight:float3(0);const float alpha=weight>0?emitted/weight:0.0f;
    // A rejected raw texel (alpha -1) cannot pass: raw validation disables the probe.
    const bool valid=reflectionProbeStorageAccepts(value.x,value.y,value.z,p.storageFlags)&&isfinite(alpha)&&alpha>=0;
    filtered.write(float4(valid?value:float3(0),valid?alpha:-1.0f),pixel.xy,6*p.cubeIndex+pixel.z);
}
#endif // !PHOSPHOR_REFL_LOBE
// Deterministic prefiltered-probe fallback uses split-sum receiver BRDF. It is
// a separate mode, not a prefilter applied AGAIN to an already GGX sampled ray.
kernel void REFL_KERNEL(reflection_probe_only)(constant GPUReflectionParams& p [[buffer(0)]],const device GPUDISurface* surfaces [[buffer(1)]],
    device GPUSpecularSample* metadata [[buffer(17)]],const device GPUReflectionProbe* probes [[buffer(18)]],device atomic_uint* counts [[buffer(19)]]REFL_GUIDE_PARAM(20),texturecube_array<float> atlas [[texture(2)]],
    texture2d<float,access::write> output [[texture(5)]],texture2d<float,access::write> distance [[texture(6)]],uint tid [[thread_position_in_grid]]){
    if(tid>=p.width*p.height)return;const GPUDISurface s=surfaces[tid];GPUSpecularSample sample=reflectionEmpty();float3 result=0;
    if(s.valid){const float3 n=normalize(diVec(s.shadingNormal)),v=normalize(diVec(s.viewDirection)),r=reflect(-v,n);uint path;
        // F13.5: the prefilter assumes n = v = r, so a rough receiver seen at a
        // grazing angle would gather a lobe half below its own horizon. Look the
        // probe up along the GGX dominant direction instead (Frostbite 2014,
        // 4.9.3), which bends toward the normal as roughness grows.
        const float da=s.roughness*s.roughness,dk=(1-da)*(sqrt(1-da)+da);
#ifdef PHOSPHOR_REFL_LOBE
        // F15.1 (ticket 11): an anisotropic lobe fetches along the README's bent normal (the dominant direction about it).
        float3 nFetch=n;
        {const auto fetchLobe=REFL_LOBE(p.flags,REFLECTION_LOBE_GUIDE_VALID,tid,s);
            if(fetchLobe.aniso)nFetch=bsdfAnisoBentNormal(n,v,bsdfAnisoFrame(n,fetchLobe.anisoDirection,s.roughness,fetchLobe.anisoStrength),s.roughness,fetchLobe.anisoStrength);}
        const float3 dominant=normalize(mix(nFetch,reflect(-v,nFetch),dk));
#else
        const float3 dominant=normalize(mix(n,r,dk));
#endif
        const float3 L=reflectionProbeRadiance(diVec(s.position),dominant,s.roughness,true,p,probes,atlas,path),f0=bsdfBaseF0(diVec(s.albedo),s.metallic);
        // F13.5: tabulated split-sum DFG of the engine BRDF (tools/dfg_table.py);
        // the former analytic mobile fit was 14% low at roughness 0.3-0.6.
        // The table can leave the physical [0,1] BRDF-integral domain (roughness=1
        // gives B=-.0024 for black metal): bsdfEnvSpecular bounds ONLY a finite
        // weight, never radiance or invalid numerical input (shared module, F15).
        const float nv=max(0.0f,dot(n,v));
#ifdef PHOSPHOR_REFL_LOBE
        // F15.1 (ticket 08): extended pixels weight the probe/environment by the split-sum of their specular lobe.
        float3 fitWeight=bsdfEnvSpecular(f0,s.roughness,nv,phosphor::DfgTable);float3 coatResult=0;
        {const auto lobe=REFL_LOBE(p.flags,REFLECTION_LOBE_GUIDE_VALID,tid,s);
            if(lobe.extended){const float3 e=bsdfSpecEnvIrid(lobe.spec,lobe.irid,diVec(s.albedo),saturate(s.metallic),lobe.envRoughness,nv,phosphor::DfgTable).specular;if(isfinite(nv)&&isfinite(s.roughness)&&all(isfinite(diVec(s.albedo))))fitWeight=e*lobe.sheenScale;}} // invalid input stays visible: the base weight is non-finite then
        // F15.1 (ticket 09): the coat term of the split sum, (1 - c F_c) material + c F_c clamp(A_c + B_c, 0, 1) at the coat roughness,
        // the coat radiance filtered at the coat roughness along the coat's dominant direction.
        {const auto lobe=REFL_LOBE(p.flags,REFLECTION_LOBE_GUIDE_VALID,tid,s);
            if(lobe.coat){const float3 nc=lobe.coatNormal,rc=reflect(-v,nc);const float fc=bsdfCoatFresnel(abs(dot(nc,v)));
                const float dac=lobe.coatRoughness*lobe.coatRoughness,dkc=(1-dac)*(sqrt(1-dac)+dac);uint coatPath;
                const float3 Lc=reflectionProbeRadiance(diVec(s.position),normalize(mix(nc,rc,dkc)),lobe.coatRoughness,true,p,probes,atlas,coatPath);
                fitWeight*=1.0f-lobe.coatFactor*fc;
                coatResult=Lc*(lobe.coatFactor*fc*bsdfCoatEnvAlbedo(lobe.coatRoughness,max(dot(nc,v),1e-4f),phosphor::DfgTable));}}
#else
        const float3 fitWeight=bsdfEnvSpecular(f0,s.roughness,nv,phosphor::DfgTable);
#endif
const bool finiteFit=all(isfinite(fitWeight));
        result=L*fitWeight;
#ifdef PHOSPHOR_REFL_LOBE
        result+=coatResult;
        // F15.1 (ticket 10, decision 06): the sheen environment term S E L(N) takes the PROBE radiance at N (the roughest filtered level), with the
        // coat attenuation, where the composite does not own it (GI on: the GI irradiance / pi stands for L(N) there).
        if(!(p.flags&REFLECTION_ENABLE_GI)){const auto sheenLobe=REFL_LOBE(p.flags,REFLECTION_LOBE_GUIDE_VALID,tid,s);
            if(sheenLobe.sheen){uint sheenPath;float sheenCoatAtt=1.0f;
                if(sheenLobe.coat)sheenCoatAtt=1.0f-sheenLobe.coatFactor*bsdfCoatFresnel(abs(dot(sheenLobe.coatNormal,v)));
                result+=reflectionProbeRadiance(diVec(s.position),n,1.0f,true,p,probes,atlas,sheenPath)*sheenLobe.sheenColor*(sheenLobe.sheenAlbedo*sheenCoatAtt);}}
        // F15.1 (ticket 13): the declared transmitted term of a transmissive pixel (probes/environment along the refracted direction).
        {const auto transLobe=REFL_LOBE(p.flags,REFLECTION_LOBE_GUIDE_VALID,tid,s);uint transmittedPath;
            if(transLobe.trans){const float3 Lt=reflectionDeclaredTransmission(s,transLobe,p,probes,atlas,transmittedPath);if(!all(isfinite(Lt))||any(Lt<0))result=as_type<float>(0x7fc00000u);else result+=Lt;}}
#endif
        sample.flags=SPECULAR_SAMPLE_VALID|SPECULAR_SAMPLE_PREFILTERED;
        if(!finiteFit||!all(isfinite(L))||any(L<0)||!all(isfinite(result))||any(result<0))sample.flags|=SPECULAR_SAMPLE_ERROR;
        sample.path=path;for(uint i=0;i<3;++i){sample.direction[i]=r[i];sample.radiance[i]=result[i];}
#ifdef PHOSPHOR_REFL_LOBE
        {const auto taggedLobe=REFL_LOBE(p.flags,REFLECTION_LOBE_GUIDE_VALID,tid,s);REFL_TAG(sample,taggedLobe);}
#endif
    }
    // The record carries result and distance 0, the values written before the
    // shared store (multi-sample reduction reads them back from the record).
    sample.hitDistance=0;reflectionStoreSample(p.flags,tid,p.width,sample,metadata,counts,output,distance);
}
