#pragma once
#define PHOSPHOR_RT_NO_ALPHA_FUNCTION
#include "rt_common.h"
#include "restir_common.h"
#include "texture_transform_core.h" // F15.1 (ticket 22)
#include "renderer/layer_blend.h"

// F12 irradiance atlas tiles: x = probe % countX; y = probe / countX.
// Tile stride is texels + 2 (explicit seam-folded 1-texel border).
// No uninitialised history is sampled: epoch/reset and active/age are checked.
constexpr sampler kGiAtlasSampler(filter::linear, address::clamp_to_edge);
constexpr sampler kGiMaterialSampler(filter::linear, mip_filter::linear, address::repeat);
// F15.4 (ticket 23): the layer mask is clamped to its edge texels at every secondary hit too (the primary resolve does the same).
constexpr sampler kGiMaskSampler(filter::linear, mip_filter::linear, address::clamp_to_edge);
inline uint giHash(uint x) { x ^= x>>16; x *= 0x7feb352du; x ^= x>>15; x *= 0x846ca68bu; return x^(x>>16); }
inline float giRandom(thread uint& seed) { seed=giHash(seed+0x9e3779b9u); return float(seed>>8)*(1.0f/16777216.0f); }
inline float giSign(float x) { return x < 0.0f ? -1.0f : 1.0f; }
inline float2 giOctEncode(float3 d) {
    d /= max(abs(d.x)+abs(d.y)+abs(d.z),1e-20f);
    return d.z<0.0f ? float2((1.0f-abs(d.y))*giSign(d.x),(1.0f-abs(d.x))*giSign(d.y)) : d.xy;
}
inline float3 giOctDecode(float2 uv) {
    float3 n(uv,1.0f-abs(uv.x)-abs(uv.y));
    if(n.z<0.0f) n.xy=float2((1.0f-abs(n.y))*giSign(n.x),(1.0f-abs(n.x))*giSign(n.y));
    return normalize(n);
}
inline float3 giProbePosition(uint id,constant GPUProbeGridParams& p,const device GPUProbeState* states) {
    uint3 c(id%p.countX,(id/p.countX)%p.countY,id/(p.countX*p.countY));
    const device GPUProbeState& s=states[id];
    float3 offset=s.generation!=p.generation?float3(0):float3(s.offset[0],s.offset[1],s.offset[2]);
    return float3(p.origin[0],p.origin[1],p.origin[2])+float3(p.spacing[0],p.spacing[1],p.spacing[2])*float3(c)+offset;
}
inline float3 giProbeDirection(uint ray,uint count,uint frame) {
    float z=1.0f-2.0f*(float(ray)+0.5f)/float(count);
    float a=2.0f*M_PI_F*fract(float(ray)*0.61803398875f+float(frame%4096u)*0.754877666f);
    float r=sqrt(max(0.0f,1.0f-z*z));
    return float3(r*cos(a),r*sin(a),z);
}
inline float giVisibility(float distance,float2 moments) {
    if(!all(isfinite(moments)) || moments.x<0.0f) return 0.0f;
    if(distance<=moments.x) return 1.0f;
    float variance=max(0.0f,moments.y-moments.x*moments.x), delta=distance-moments.x;
    float p=variance/(variance+delta*delta+1e-12f);
    return p*p*p;
}
inline float2 giAtlasUV(uint id,uint side,float3 direction,constant GPUProbeGridParams& p) {
    const uint stride=side+2u;
    float2 pixel=float2(id%p.countX,id/p.countX)*float(stride)+1.0f+(giOctEncode(direction)*0.5f+0.5f)*float(side);
    return pixel/float2(p.countX*stride,p.countY*p.countZ*stride);
}
inline float3 giIrradiance(float3 point,float3 normal,constant GPUProbeGridParams& p,
                           const device GPUProbeState* states,texture2d<float> irradiance,texture2d<float> moments,bool currentAtlas=false) {
    if((p.reset && !currentAtlas) || !all(isfinite(point)) || !all(isfinite(normal)) || dot(normal,normal)<1e-20f) return 0.0f;
    normal=normalize(normal);
    float3 biased=point+normal*p.normalBias;
    float3 coordinate=(biased-float3(p.origin[0],p.origin[1],p.origin[2]))/float3(p.spacing[0],p.spacing[1],p.spacing[2]);
    if(any(coordinate<0.0f) || any(coordinate>float3(p.countX,p.countY,p.countZ)-1.0f)) return 0.0f;
    int3 base=min(int3(floor(coordinate)),int3(p.countX,p.countY,p.countZ)-2);
    float3 f=coordinate-float3(base), result=0.0f; float weights=0.0f;
    for(uint corner=0;corner<8u;++corner) {
        int3 bit(int(corner&1u),int((corner>>1u)&1u),int((corner>>2u)&1u)), c=base+bit;
        uint probe=uint((c.z*int(p.countY)+c.y)*int(p.countX)+c.x);
        const device GPUProbeState& s=states[probe];
        if(s.generation!=p.generation || s.state!=GI_PROBE_ACTIVE || s.age==0u) continue;
        float3 delta=biased-giProbePosition(probe,p,states);
        float distance=length(delta), directionLength=max(distance,1e-8f);
        float3 direction=distance>1e-8f?delta/directionLength:normal;
        float2 m=moments.sample(kGiAtlasSampler,giAtlasUV(probe,p.distanceTexels,direction,p)).xy;
        float3 tri=mix(1.0f-f,f,float3(bit));
        float wrap=max(0.05f,(dot(normal,-direction)+1.0f)*0.5f);
        // A physical ablation: keep real rays/moments/classification and finite
        // output so the independent image oracle, not a forced counter, detects leaks.
        float visibility=(p.debugFlags&GI_DEBUG_NO_VISIBILITY)?1.0f:giVisibility(distance,m);
        float w=tri.x*tri.y*tri.z*wrap*wrap*visibility;
        result+=irradiance.sample(kGiAtlasSampler,giAtlasUV(probe,p.irradianceTexels,normal,p)).rgb*w;
        weights+=w;
    }
    return weights>1e-8f ? result/weights : float3(0.0f);
}

// F15 (F13.1 residue found in F15; owner decision 2026-11-14): ENVIRONMENT OWNERSHIP with a GI mode on.
//
//   GI off      the resolve / composite adds the analytic diffuse term  environment x albedo x (1 - metallic)  (declared, unoccluded);
//               reflectionSecondary adds the same term to every secondary RT hit.
//   GI on       the TRANSPORT owns the environment.  A ray that leaves the scene (no hit within maxDistance) returns the declared radiance:
//               - ddgi_trace probe rays and gi_candidates cosine rays return it on a miss, so the atlas / reservoir irradiance of a surface
//                 already contains the sky it can see, occluded by geometry (an open surface reads albedo x L, a slot reads albedo x L x V);
//               - a secondary hit receives it through the atlas lookup like every other indirect light (giSecondaryRadiance has no explicit
//                 sky ray: with the atlas carrying the sky it would count it twice);
//               - the GI-off analytic term is NOT added: useGI (resolve) and REFLECT_COMPOSE_GI (composite) strip it, and reflectionSecondary
//                 takes the atlas branch when REFLECTION_ENABLE_GI is set.
//   Specular environment / probe radiance is a different lobe (ambientSpecular, the reflection estimators) and is untouched.
// Emitters follow the F13.1 band rule: this function never returns emission.
inline float3 giEscapeRadiance(constant GPUProbeGridParams& p) {
    return max(float3(p.skyRadiance[0],p.skyRadiance[1],p.skyRadiance[2]),0.0f);
}
// A reservoir sample that escaped the scene references no instance: giReusable and the independent state check accept this slot.
constant uint GI_SLOT_ESCAPED=~0u;

// F15.4 (ticket 23): the layer stack's blended base parameters at a secondary hit.  ONE helper (shaders/layer_resolve.h, the code of the primary
// resolve) for GI, reflection and probe-capture hits: level-0 sampling of the mask (clamped) and of every layer texture at uv * uvScale.  A
// stack's feature blocks are not evaluated at secondary hits (no material's are: declared reduction of ADR-0002), its emission is.
struct LayerLod0Samples {
    template <typename T>
    static half4 sample(const device T* table,uint index,float2 uv,half4 fallback,float2 dx,float2 dy) {
        return index==INVALID_TEXTURE_INDEX?fallback:half4(table[index].tex.sample(kGiMaterialSampler,uv,level(0.0f)));
    }
    template <typename T>
    static half4 mask(const device T* table,uint index,float2 uv,half4 fallback,float2 dx,float2 dy) {
        return index==INVALID_TEXTURE_INDEX?fallback:half4(table[index].tex.sample(kGiMaskSampler,uv,level(0.0f)));
    }
};
#include "layer_resolve.h"
// True (and `out` filled) when material `m` is a layer stack and the extension buffer is bound.
template <typename T,typename Samples>
inline bool layerStackSurface(const device GPUMaterial& m,GPUMaterialExtPtr extensions,const device GPUMaterial* materials,const device T* textures,
                              float2 uv,float2 dx,float2 dy,thread LayerSurface& out) {
    if((m.flags&(MATERIAL_FEATURE_LAYERS<<MATERIAL_FEATURE_SHIFT))==0u||extensions==nullptr||m.extensionIndex==MATERIAL_EXTENSION_INVALID) return false;
    const GPUMaterialExt3 e3=*reinterpret_cast<const device GPUMaterialExt3*>(extensions+m.extensionIndex+3u);
    out=resolveLayerBase<T,Samples>(e3,materials,textures,uv,dx,dy,true,extensions,m.flags,m.textureTransformIndex);
    return true;
}

struct GiMaterial { float3 diffuse,emissive; bool doubleSided; };
inline bool giMaterial(GPURtHit hit,constant GPUProbeGridParams& p,const device GPUInstance* instances,
                       const device GPURtMesh* meshes,const device GPUVertex* vertices,const device uint* indices,
                       const device GPUMaterial* materials,const device DITextureHandle* textures,
                       thread GiMaterial& out) {
    if(hit.slot>=p.slotCount) return false;
    const device GPUInstance& i=instances[hit.slot];
    if(i.meshIndex>=p.meshCount || i.materialIndex>=p.materialCount) return false;
    const device GPURtMesh& mesh=meshes[i.meshIndex];
    if(hit.primitive>=mesh.indexCount/3u) return false;
    const uint b=mesh.indexOffset+3u*hit.primitive;
    const device GPUVertex& a=vertices[mesh.vertexOffset+indices[b]], &v=vertices[mesh.vertexOffset+indices[b+1]],
        &w=vertices[mesh.vertexOffset+indices[b+2]];
    float2 uv=float2(a.u,a.v)*(1.0f-hit.u-hit.v)+float2(v.u,v.v)*hit.u+float2(w.u,w.v)*hit.v;
    const device GPUMaterial& m=materials[i.materialIndex];
    float3 base=float3(m.baseColor[0],m.baseColor[1],m.baseColor[2]);
    if(m.baseColorTex!=INVALID_TEXTURE_INDEX) base*=float3(half3(textures[m.baseColorTex].tex.sample(kGiMaterialSampler,texXfSlotUv(p.materialExtensions,m.flags,m.textureTransformIndex,TT_SLOT_BASE_COLOR,uv),level(0.0f)).rgb));
    // F15 closure (COLOR_0): base colour = factor x texture x vertex colour (GI hit surface); the pointer is 0 without a colour stream.
    if(p.vertexColors!=nullptr) base*=(p.vertexColors[mesh.vertexOffset+indices[b]]*(1.0f-hit.u-hit.v)+p.vertexColors[mesh.vertexOffset+indices[b+1]]*hit.u+p.vertexColors[mesh.vertexOffset+indices[b+2]]*hit.v).rgb;
    float metallic=m.metallic;
    if(m.metallicRoughnessTex!=INVALID_TEXTURE_INDEX) metallic*=float(half(textures[m.metallicRoughnessTex].tex.sample(kGiMaterialSampler,texXfSlotUv(p.materialExtensions,m.flags,m.textureTransformIndex,TT_SLOT_METAL_ROUGH,uv),level(0.0f)).b));
    out.diffuse=max(base,0.0f)*(1.0f-saturate(metallic));
    out.emissive=float3(m.emissive[0],m.emissive[1],m.emissive[2]);
    if(m.emissiveTex!=INVALID_TEXTURE_INDEX) out.emissive*=float3(half3(textures[m.emissiveTex].tex.sample(kGiMaterialSampler,texXfSlotUv(p.materialExtensions,m.flags,m.textureTransformIndex,TT_SLOT_EMISSIVE,uv),level(0.0f)).rgb));
    LayerSurface layered;
    if(layerStackSurface<DITextureHandle,LayerLod0Samples>(m,p.materialExtensions,materials,textures,uv,float2(0.0f),float2(0.0f),layered)) {
        out.diffuse=max(layered.baseColor,0.0f)*(1.0f-saturate(layered.metallic));
        out.emissive+=layered.emission;
    }
    out.emissive=max(out.emissive,0.0f);
    if(m.flags&MATERIAL_FLAG_UNLIT)out.emissive=float3(0.0f);
    // F12.1 residue found in F15: the atlas ignores transmission (declared: secondary_gi_and_probe_hits_ignore_transmission), so a glass pane is a
    // white diffuse surface seen from either side.  Its BACK face must not read as the inside of a solid: a probe in the gap between a single-sided
    // pane and the wall behind it counted > 25 % back faces, went INACTIVE, and the lookups of everything behind the glass found no probe (0).
    out.doubleSided=(m.flags&(MATERIAL_FLAG_DOUBLE_SIDED|(MATERIAL_FEATURE_TRANSMISSION<<MATERIAL_FEATURE_SHIFT)))!=0u;
    return all(isfinite(out.diffuse)) && all(isfinite(out.emissive));
}
// F15.1 (ticket 14): the RGB transmittance of the shadow segment (0 = blocked by an opaque caster, product of the crossed glass factors otherwise).
inline float3 giShadowTransmittance(float3 point,float3 geometricNormal,float3 wi,float distance,
                            instance_acceleration_structure as,intersection_function_table<triangle_data,instancing> ift,
                            const device GPUInstance* instances,constant GPUProbeGridParams& p) {
    float3 origin=rtOffsetRay(point,dot(geometricNormal,wi)>=0.0f?geometricNormal:-geometricNormal);
    GPURtRay ray{}; ray.ox=origin.x;ray.oy=origin.y;ray.oz=origin.z;ray.dx=wi.x;ray.dy=wi.y;ray.dz=wi.z;
    ray.tmax=max(0.0f,distance-length(origin-point)-1e-5f);ray.mask=RT_MASK_SHADOW;ray.type=RT_PROBE_SHADOW_TRANSMITTED;
    RtPayload payload{};
    return rtTrace(ray,as,ift,instances,p.slotCount,payload).hit==0u?rtShadowTransmittance(payload,ray.tmax):float3(0);
}
// Secondary diffuse radiance. Area integrals include both emitter cosine and
// the 1/(selection-PMF * area-PDF). Emissive hit contributes Le directly once.
inline float3 giSecondaryRadiance(float3 point,float3 normal,GiMaterial material,
                                  instance_acceleration_structure as,intersection_function_table<triangle_data,instancing> ift,
                                  const device GPUInstance* instances,constant GPUProbeGridParams& p,
                                  const device GPULight* lights,const device GPUMaterial* materials,const device DITextureHandle* textures,
                                  const device GPUSampledLight* sampled,const device GPUEmissiveSurface* emissive,
                                  constant GPUProbeTraceExtra& extra,thread uint& seed,
                                  const device GPUProbeState* states,texture2d<float> irradiance,texture2d<float> moments) {
    // The atlas carries the sky (escaped probe rays return it, giEscapeRadiance): no explicit sky ray here (it would count it twice).
    float3 E=giIrradiance(point,normal,p,states,irradiance,moments);
    for(uint l=0;l<p.lightCount;++l) {
        const device GPULight& light=lights[l];
        if(extra.sampledLightCount && light.type!=LIGHT_DIRECTIONAL) continue;
        float3 wi;float distance=p.maxDistance,att=1.0f;
        if(light.type==LIGHT_DIRECTIONAL) {
            float3 axis=-normalize(float3(light.direction[0],light.direction[1],light.direction[2]));
            wi=axis;
            if(extra.sunAngularRadius>0.0f) {
                float coneCos=cos(extra.sunAngularRadius),z=1.0f-giRandom(seed)*(1.0f-coneCos);
                float phi=2.0f*M_PI_F*giRandom(seed),r=sqrt(max(0.0f,1.0f-z*z));
                float3 t=normalize(cross(abs(axis.z)<0.999f?float3(0,0,1):float3(0,1,0),axis)),b=cross(axis,t);
                wi=t*(r*cos(phi))+b*(r*sin(phi))+axis*z;
                // GPULight intensity remains perpendicular irradiance. Uniform
                // cone PDF cancels Le*Omega with 2/(1+cos angularRadius).
                att=2.0f/(1.0f+coneCos);
            }
        }
        else {
            float3 delta=float3(light.position[0],light.position[1],light.position[2])-point;
            distance=length(delta);wi=delta/max(distance,1e-8f);
            float t=distance/max(light.range,1e-3f), window=saturate(1.0f-t*t*t*t);
            att=window*window/max(distance*distance,1e-4f);
            if(light.type==LIGHT_SPOT) att*=smoothstep(cos(light.outerCone),cos(light.innerCone),
                dot(-wi,normalize(float3(light.direction[0],light.direction[1],light.direction[2]))));
        }
        float cosN=max(0.0f,dot(normal,wi));
        if(cosN>0.0f && att>0.0f) {
            const float3 transmittance=giShadowTransmittance(point,normal,wi,distance,as,ift,instances,p);
            if(any(transmittance>0.0f)) E+=float3(light.color[0],light.color[1],light.color[2])*light.intensity*att*cosN*transmittance;
        }
    }
    if(extra.sampledLightCount) {
        uint index=min(uint(giRandom(seed)*float(extra.sampledLightCount)),extra.sampledLightCount-1u);
        DISample sample=diSampleTexturedLight(sampled[index],index,float2(giRandom(seed),giRandom(seed)),point,emissive,materials,textures,p.materialExtensions,p.vertexColors);
        float cosN=max(0.0f,dot(normal,sample.wi));
        if(sample.valid && cosN>0.0f) {
            float inversePdf=sample.delta?float(extra.sampledLightCount):float(extra.sampledLightCount)/max(sample.pdfArea,1e-20f);
            const float3 transmittance=giShadowTransmittance(point,normal,sample.wi,sample.distance,as,ift,instances,p);
            if(any(transmittance>0.0f)) E+=diIncident(sample)*cosN*inversePdf*transmittance;
        }
    }
    return material.emissive+material.diffuse*max(E,0.0f)/M_PI_F;
}
