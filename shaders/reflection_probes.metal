#include "reflection_common.h"

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
fragment float4 reflection_probe_capture_fs(ProbeVertexOut in [[stage_in]],bool frontFacing [[front_facing]],
    constant GPUProbeCaptureParams& p [[buffer(0)]],const device GPUInstance* instances [[buffer(2)]],
    const device GPUMaterial* materials [[buffer(3)]],const device GPULight* lights [[buffer(4)]],const device DITextureHandle* textures [[buffer(5)]],
    const device GPUSampledLight* sampled [[buffer(7)]],const device GPUEmissiveSurface* emitters [[buffer(8)]]){
    if(in.slot>=p.slotCount||in.material>=p.materialCount)discard_fragment();const GPUMaterial m=materials[in.material];
    const float2 dx=dfdx(in.uv),dy=dfdy(in.uv);const half4 baseTex=m.baseColorTex==INVALID_TEXTURE_INDEX?half4(1):half4(textures[m.baseColorTex].tex.sample(kGiMaterialSampler,in.uv,gradient2d(dx,dy)));
    if(m.baseColor[3]*float(baseTex.a)<m.alphaCutoff)discard_fragment();
    const bool front=frontFacing!=((instances[in.slot].flags&INSTANCE_FLAG_MIRRORED)!=0);float3 n=normalize(front?in.normal:-in.normal);
    const float3 geometric=normalize(cross(dfdx(in.world),dfdy(in.world)));float3 t=in.tangent-n*dot(n,in.tangent);
    if(m.normalTex!=INVALID_TEXTURE_INDEX&&dot(t,t)>1e-8f){t=normalize(t);float3 tn=float3(half3(textures[m.normalTex].tex.sample(kGiMaterialSampler,in.uv,gradient2d(dx,dy)).rgb))*2-1;tn.xy*=m.normalScale;n=normalize(t*tn.x+cross(n,t)*in.handed*tn.y+n*tn.z);}
    const half4 mr=m.metallicRoughnessTex==INVALID_TEXTURE_INDEX?half4(1):half4(textures[m.metallicRoughnessTex].tex.sample(kGiMaterialSampler,in.uv,gradient2d(dx,dy)));
    GPUDISurface s{};const float3 base=float3(m.baseColor[0],m.baseColor[1],m.baseColor[2])*float3(baseTex.rgb),v=normalize(reflectionConstantVec(p.capturePosition)-in.world);
    for(uint c=0;c<3;++c){s.position[c]=in.world[c];s.geometricNormal[c]=geometric[c];s.shadingNormal[c]=n[c];s.albedo[c]=base[c];s.viewDirection[c]=v[c];}
    s.valid=1;s.roughness=clamp(m.roughness*float(mr.g),.04f,1.0f);s.metallic=saturate(m.metallic*float(mr.b));float3 result=0;
    for(uint i=0;i<p.lightCount;++i){GPULight light=lights[i];if(p.sampledLightCount&&light.type!=LIGHT_DIRECTIONAL)continue;DISample sample{};
        if(light.type==LIGHT_DIRECTIONAL){sample.valid=true;sample.delta=true;sample.geometry=1;sample.wi=-normalize(float3(light.direction[0],light.direction[1],light.direction[2]));sample.radiance=float3(light.color[0],light.color[1],light.color[2])*light.intensity;}
        else{GPUSampledLight local{};local.type=light.type;local.range=light.range;local.innerCone=light.innerCone;local.outerCone=light.outerCone;
            for(uint c=0;c<3;++c){local.position[c]=light.position[c];local.axisU[c]=light.direction[c];local.emission[c]=light.color[c]*light.intensity;}sample=diSampleLight(local,float2(.5f),in.world);}
        result+=diBRDF(s,sample);
    }
    // Capture is loading-time work: one stratified 4x4 area quadrature per light
    // avoids depending on a noisy one-endpoint static cube map.
    for(uint i=0;i<p.sampledLightCount;++i)for(uint y=0;y<4;++y)for(uint x=0;x<4;++x){DISample sample=diSampleTexturedLight(sampled[i],i,(float2(x,y)+.5f)/4.0f,in.world,emitters,materials,textures);
        if(sample.valid&&sample.pdfArea>0)result+=diBRDF(s,sample)/(16*sample.pdfArea);}
    float3 emission=float3(m.emissive[0],m.emissive[1],m.emissive[2]);if(m.emissiveTex!=INVALID_TEXTURE_INDEX)emission*=float3(half3(textures[m.emissiveTex].tex.sample(kGiMaterialSampler,in.uv,gradient2d(dx,dy)).rgb));
    result+=emission+reflectionConstantVec(p.environment)*base*(1-s.metallic);return float4(all(isfinite(result))?max(result,0.0f):float3(0),1);
}

inline float3 reflectionCubeDirection(uint face,float2 uv){const float2 p=uv*2-1;switch(face){case 0:return normalize(float3(1,-p.y,-p.x));case 1:return normalize(float3(-1,-p.y,p.x));case 2:return normalize(float3(p.x,1,p.y));case 3:return normalize(float3(p.x,-1,-p.y));case 4:return normalize(float3(p.x,-p.y,1));default:return normalize(float3(-p.x,-p.y,-1));}}
inline float reflectionRadicalInverse(uint v){v=(v<<16)|(v>>16);v=((v&0x55555555u)<<1)|((v&0xaaaaaaaau)>>1);v=((v&0x33333333u)<<2)|((v&0xccccccccu)>>2);v=((v&0x0f0f0f0fu)<<4)|((v&0xf0f0f0f0u)>>4);v=((v&0x00ff00ffu)<<8)|((v&0xff00ff00u)>>8);return float(v)*2.3283064365386963e-10f;}
// Params0; raw cubeArray texture0; selected destination mip viewed as 2DArray
// texture1. Dispatch side x side x6; write slice=6*cubeIndex+face. Every mip
// reads mip0 of a DIFFERENT source texture; never read/write the same resource.
kernel void reflection_probe_prefilter(constant GPUProbeFilterParams& p [[buffer(0)]],texturecube_array<float> raw [[texture(0)]],
    texture2d_array<float,access::write> filtered [[texture(1)]],uint3 pixel [[thread_position_in_grid]]){
    if(pixel.x>=p.side||pixel.y>=p.side||pixel.z>=6)return;const float3 n=reflectionCubeDirection(pixel.z,(float2(pixel.xy)+.5f)/float(p.side));float3 L=0;float weight=0;
    if(p.roughness<=0){L=raw.sample(kReflectionProbeSampler,n,p.cubeIndex,level(0)).rgb;weight=1;}
    else{const float3 t=reflectionTangent(n),b=cross(n,t);const float a=max(p.roughness*p.roughness,.002f),a2=a*a;
        const uint count=clamp(p.sampleCount,1u,4096u);
        for(uint i=0;i<count;++i){float u=(float(i)+.5f)/float(count),v=reflectionRadicalInverse(i),c=sqrt((1-u)/(1+(a2-1)*u)),r=sqrt(max(0.0f,1-c*c)),phi=2*M_PI_F*v;
            const float3 h=t*(r*cos(phi))+b*(r*sin(phi))+n*c,l=reflect(-n,h);const float nl=max(0.0f,dot(n,l));L+=raw.sample(kReflectionProbeSampler,l,p.cubeIndex,level(0)).rgb*nl;weight+=nl;}}
    filtered.write(float4(weight>0?L/weight:float3(0),1),pixel.xy,6*p.cubeIndex+pixel.z);
}
// Deterministic prefiltered-probe fallback uses split-sum receiver BRDF. It is
// a separate mode, not a prefilter applied AGAIN to an already GGX sampled ray.
kernel void reflection_probe_only(constant GPUReflectionParams& p [[buffer(0)]],const device GPUDISurface* surfaces [[buffer(1)]],
    device GPUSpecularSample* metadata [[buffer(17)]],const device GPUReflectionProbe* probes [[buffer(18)]],texturecube_array<float> atlas [[texture(2)]],
    texture2d<float,access::write> output [[texture(5)]],texture2d<float,access::write> distance [[texture(6)]],uint tid [[thread_position_in_grid]]){
    if(tid>=p.width*p.height)return;const GPUDISurface s=surfaces[tid];GPUSpecularSample sample=reflectionEmpty();float3 result=0;
    if(s.valid){const float3 n=normalize(diVec(s.shadingNormal)),v=normalize(diVec(s.viewDirection)),r=reflect(-v,n);uint path;
        const float3 L=reflectionProbeRadiance(diVec(s.position),r,s.roughness,true,p,probes,atlas,path),f0=mix(float3(.04f),diVec(s.albedo),s.metallic);
        const float4 c0(-1,-.0275f,-.572f,.022f),c1(1,.0425f,1.04f,-.04f),fit=s.roughness*c0+c1;
        const float a004=min(fit.x*fit.x,exp2(-9.28f*max(0.0f,dot(n,v))))*fit.x+fit.y;const float2 ab=float2(-1.04f,1.04f)*a004+fit.zw;
        result=L*(f0*ab.x+ab.y);sample.flags=SPECULAR_SAMPLE_VALID|SPECULAR_SAMPLE_PREFILTERED;sample.path=path;for(uint i=0;i<3;++i){sample.direction[i]=r[i];sample.radiance[i]=result[i];}}
    metadata[tid]=sample;const uint2 pixel(tid%p.width,tid/p.width);output.write(float4(result,1),pixel);distance.write(float4(0),pixel);
}
