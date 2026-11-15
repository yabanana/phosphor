#pragma once
#include <metal_stdlib>
#include "renderer/gpu_types.h"
#include "materials/bsdf.h"
#include "renderer/lobe_guide_pack.h"
#include "texture_transform_core.h" // F15.1 (ticket 22): per-slot texture transforms of emitter surfaces
#include "renderer/layer_blend.h"   // F15.4 (ticket 23): the coverage of a layer stack that is an emitter
#include "layer_resolve.h"
// F15.1 (ticket 07): the lobe-guide kernels are separate entry points ("<name>_lobe", compiled by
// restir_di_lobe.metal / light_cluster_lobe.metal with PHOSPHOR_DI_LOBE, linked into the first material module) that always carry the reserved function
// constant MATERIAL_FEATURES (22, shaders/variants.def).  Everything else, the base kernels included, never
// references the constant: a compute function with an unresolved constant cannot be lowered into the pipeline archive
// (metal-tt, measured), so the base DI descriptors stay constant-free and archived.
#ifdef PHOSPHOR_DI_LOBE
#define DI_KERNEL(name) name##_lobe
#define DI_FN(name) name##_lobe // helpers whose body differs between the two modes: distinct symbols (no ODR merge)
#include "pipeline/forward_variants.generated.metal.h" // kMaterialFeatures (constant 22); shared with the material module
#else
#define DI_KERNEL(name) name
#define DI_FN(name) name
#endif
using namespace metal;
using namespace phosphor;

// All light integrands are with respect to AREA endpoints plus punctual
// discrete endpoints. Solid-angle PDF is exposed for consumer/MIS diagnostics,
// never accidentally mixed into area-measure reservoir candidate weights.
struct DISample {
    float3 position, normal, wi, radiance;
    float distance, pdfArea, pdfSolidAngle, geometry;
    bool delta, valid;
};
struct DITextureHandle { texture2d<float> tex; };
static_assert(sizeof(DITextureHandle) == 8, "DI bindless texture handle ABI");
constexpr sampler kDIEmitterSampler(filter::linear, mip_filter::linear, address::repeat);
inline float3 diVec(const thread float* p) { return float3(p[0], p[1], p[2]); }
inline float diLuminance(float3 c) { return dot(c, float3(0.2126f, 0.7152f, 0.0722f)); }
inline float diArea(GPUSampledLight l) {
    const float3 u = diVec(l.axisU), v = diVec(l.axisV);
    float a = 0;
    if (l.type == DI_LIGHT_RECTANGLE) a = 4.0f * length(cross(u, v));
    if (l.type == DI_LIGHT_DISK && l.radius > 0) a = M_PI_F * l.radius * l.radius * length(cross(u, v));
    if (l.type == DI_LIGHT_TUBE && l.radius > 0) a = 4.0f * M_PI_F * l.radius * length(u);
    if (l.type == DI_LIGHT_TRIANGLE) a = 0.5f * length(cross(u, v));
    return isfinite(a) && a > 0 ? a : 0;
}

inline DISample diSampleLight(GPUSampledLight l, float2 uv, float3 receiver) {
    DISample s{};
    if (!all(isfinite(uv)) || any(uv < 0) || any(uv >= 1) || !all(isfinite(receiver))) return s;
    s.position = diVec(l.position);
    if (!all(isfinite(diVec(l.emission)))) return s;
    s.radiance = max(diVec(l.emission), 0.0f);
    const float3 u = diVec(l.axisU), v = diVec(l.axisV);
    s.delta = l.type == LIGHT_POINT || l.type == LIGHT_SPOT;
    if (!s.delta) {
        const float area = diArea(l);
        if (!(area > 0)) return s;
        s.pdfArea = 1.0f / area;
        if (l.type == DI_LIGHT_RECTANGLE) s.position += (2.0f * uv.x - 1.0f) * u + (2.0f * uv.y - 1.0f) * v;
        else if (l.type == DI_LIGHT_DISK) {
            const float r = l.radius * sqrt(uv.x), angle = 2.0f * M_PI_F * uv.y;
            s.position += r * (cos(angle) * u + sin(angle) * v);
        } else if (l.type == DI_LIGHT_TRIANGLE) {
            const float root = sqrt(uv.x);
            s.position += root * (1.0f - uv.y) * u + root * uv.y * v;
        } else if (l.type == DI_LIGHT_TUBE) {
            const float3 axis = normalize(u), radial = v - axis * dot(axis, v);
            if (!(dot(radial, radial) > 1e-20f)) return s;
            const float3 x = normalize(radial), y = cross(axis, x);
            const float angle = 2.0f * M_PI_F * uv.y;
            s.normal = cos(angle) * x + sin(angle) * y;
            s.position += (2.0f * uv.x - 1.0f) * u + l.radius * s.normal;
        } else return s;
        if (l.type != DI_LIGHT_TUBE) s.normal = normalize(cross(u, v));
        if (l.type == DI_LIGHT_TRIANGLE && (l.flags & DI_LIGHT_MIRRORED)) s.normal = -s.normal;
    }
    const float3 delta = s.position - receiver;
    s.distance = length(delta);
    if (!(s.distance > 1e-10f) || !isfinite(s.distance)) { s.valid = false; return s; }
    s.wi = delta / s.distance;
    if (s.delta) {
        s.pdfArea = 1.0f; // discrete endpoint measure, NOT m^-2
        const float ratio = s.distance / max(l.range, 1e-3f);
        const float window = saturate(1.0f - ratio * ratio * ratio * ratio);
        s.geometry = window * window / max(s.distance * s.distance, 1e-4f);
        if (l.type == LIGHT_SPOT) {
            if (!(dot(u, u) > 0)) return s;
            const float spot = saturate((dot(normalize(u), -s.wi) - cos(l.outerCone)) /
                                        max(cos(l.innerCone) - cos(l.outerCone), 1e-4f));
            s.geometry *= spot * spot * (3.0f - 2.0f * spot);
        }
    } else {
        const float facing = dot(s.normal, -s.wi);
        const float cosine = (l.flags & DI_LIGHT_TWO_SIDED) ? abs(facing) : max(facing, 0.0f);
        s.geometry = cosine / (s.distance * s.distance);
        s.pdfSolidAngle = cosine > 0 ? s.pdfArea * s.distance * s.distance / cosine : 0;
        if (l.range > 0) {
            const float ratio = s.distance / l.range;
            const float window = saturate(1.0f - ratio * ratio * ratio * ratio);
            s.geometry *= window * window;
        }
    }
    s.valid = all(isfinite(s.position)) && all(isfinite(s.wi)) && all(isfinite(s.radiance)) &&
              isfinite(s.geometry) && s.geometry >= 0;
    return s;
}

// The full-scene emissive extraction pass supplies transformed geometry, UVs
// and materialIndex. Constant/nontextured lights never read these buffers.
// Emissive RGB and MASK alpha use explicit LOD0, with the raster's half texel
// arithmetic. No camera derivatives exist on a sampled light surface.
inline DISample diSampleTexturedLight(GPUSampledLight l, uint lightIndex, float2 uv, float3 receiver,
                                      const device GPUEmissiveSurface* emitters,
                                      const device GPUMaterial* materials, const device DITextureHandle* textures,
                                      const device GPUMaterialExtRecord* extensions = nullptr) { // F15.1 (ticket 22): per-slot texture transforms
    DISample s = diSampleLight(l, uv, receiver);
    if (!s.valid || l.type != DI_LIGHT_TRIANGLE || !(l.flags & DI_LIGHT_TEXTURED_EMISSION)) return s;
    const GPUEmissiveSurface emitter = emitters[lightIndex];
    if (!emitter.valid) { s.valid = false; return s; }
    const GPUMaterial m = materials[emitter.materialIndex]; // host/extraction bounds-checked
    const float root = sqrt(uv.x);
    const float2 lightUV = (1.0f - root) * float2(emitter.uv0[0], emitter.uv0[1]) +
                           root * (1.0f - uv.y) * float2(emitter.uv1[0], emitter.uv1[1]) +
                           root * uv.y * float2(emitter.uv2[0], emitter.uv2[1]);
    const half3 texel = m.emissiveTex == INVALID_TEXTURE_INDEX ? half3(1) :
                        half3(textures[m.emissiveTex].tex.sample(kDIEmitterSampler,
                              texXfSlotUv(extensions, m.flags, m.textureTransformIndex, TT_SLOT_EMISSIVE, lightUV), level(0.0f)).rgb);
    s.radiance = float3(m.emissive[0], m.emissive[1], m.emissive[2]) * float3(texel);
    if (m.alphaCutoff > 0) {
        const half alphaTexel = m.baseColorTex == INVALID_TEXTURE_INDEX ? half(1) :
                                half(textures[m.baseColorTex].tex.sample(kDIEmitterSampler,
                                     texXfSlotUv(extensions, m.flags, m.textureTransformIndex, TT_SLOT_BASE_COLOR, lightUV), level(0.0f)).a);
        // F15.4 (ticket 23): an emissive layer stack cuts the emitter out by its blended layer alpha, not by its own record's.
        const float alpha = m.baseColor[3] * float(alphaTexel) *
                            layerCoverage<DITextureHandle, LayerAlphaLevel0>(materials[emitter.materialIndex], materials, textures, lightUV,
                                                                             float2(0.0f), float2(0.0f), extensions);
        if (!isfinite(alpha)) { s.valid = false; return s; }
        if (alpha < m.alphaCutoff) s.radiance = float3(0);
    }
    s.valid = all(isfinite(s.radiance)) && all(s.radiance >= 0);
    return s;
}

inline float3 diIncident(DISample s) { return s.valid ? s.radiance * s.geometry : float3(0); }
inline float3 diBRDFWeighted(GPUDISurface surface, DISample s, float specularWeight) {
    if (!surface.valid || !s.valid) return float3(0);
    float3 n = diVec(surface.shadingNormal), v = diVec(surface.viewDirection);
    if (!all(isfinite(n)) || !all(isfinite(v)) || !(dot(n, n) > 0 && dot(v, v) > 0)) return float3(0);
    n = normalize(n); v = normalize(v);
    const float nl = saturate(dot(n, s.wi)), nv = max(dot(n, v), 1e-4f);
    if (!(nl > 0) || dot(v + s.wi, v + s.wi) <= 1e-20f) return float3(0);
    const float3 h = normalize(v + s.wi);
    const float nh = saturate(dot(n, h)), vh = saturate(dot(v, h));
    const float metallic = saturate(surface.metallic);
    const float3 albedo = max(diVec(surface.albedo), 0.0f);
    const BsdfBaseLobes lobes =
        bsdfBaseEval(nl, nv, nh, vh, albedo, metallic, bsdfAlpha(surface.roughness), specularWeight);
    const float3 result = (lobes.diffuse + lobes.specular) * diIncident(s) * nl;
    return result; // caller MUST record nonfinite BEFORE any safe-output sanitization
}
// F15.1 (ticket 07): lobe parameters of the destination pixel read from the lobe guide.  `extended` is false for
// base pixels and, in the base kernels (no PHOSPHOR_DI_LOBE), always: the lobe code is not compiled in, so scenes
// without features keep the base expressions bit for bit.
struct DILobe {
    float ior, specularFactor;
    float3 specularColor;
    bool extended;
    // F15.1 (ticket 09): KHR_materials_clearcoat.  The coat normal is the guide's (the shading normal without the base normal map).
    bool coat;
    float coatFactor, coatRoughness;
    float3 coatNormal;
    // F15.1 (ticket 10): KHR_materials_sheen (roughness already clamped to [0.07, 1] by the producer).
    bool sheen;
    float3 sheenColor;
    float sheenRoughness;
    // F15.1 (ticket 11): KHR_materials_anisotropy.  The lobe direction T' is the guide's world direction (orthogonal to N up to the
    // oct quantisation: bsdfAnisoFrame re-orthogonalises), the bitangent B' = cross(N, T') is built where the normal is known.
    bool aniso;
    float3 anisoDirection;
    float anisoStrength;
    // F15.1 (ticket 12): KHR_materials_iridescence.  The film (strength, ior, thickness in nm) replaces the Fresnel of the specular lobe at
    // the V.H of each light sample (target and shading alike).
    bool irid;
    BsdfIrid iridParams;
    // F15.1 (ticket 16): subsurface thin mode.  Lights on the back hemisphere light the pixel through the transmitted lobe (target and
    // shading); the visibility ray already leaves from the light's side (diEndpointVisible flips the geometric normal towards the light).
    bool thin;
    BsdfThin thinParams;
    // F15.1 (ticket 13): KHR_materials_transmission.  The diffuse base is mixed by 1 - t; a thin-walled surface (no volume thickness) also takes the
    // delta lights behind it through the glTF thin BTDF (the area emitters behind it belong to the transport's straight/refracted rays).
    bool trans;
    float transFactor;
    bool transThin;
    // F15.1 (ticket 17, RT tier): the weight of a volumetric-subsurface pixel while the walk is effective (DI_SSS_WALK), 0 otherwise.
    float sssWalk;
};
inline DILobe DI_FN(diLoadLobe)(constant GPUDIParams& p, const device GPULobeGuide* guides, uint index) {
    DILobe l{};
#ifdef PHOSPHOR_DI_LOBE
    if ((kMaterialFeatures & LOBE_GUIDE_RESOLVED_FEATURES) != 0u && (p.flags & DI_LOBE_GUIDE_VALID) != 0u) {
        const GPULobeGuide g = guides[index];
        if ((g.features & LOBE_GUIDE_RESOLVED_FEATURES) != 0u) {
            const LobeParams u = unpackLobeGuide(g); // neutral values (ior 1.5, specular 1) for a clear feature
            l.ior = u.ior; l.specularFactor = u.specularFactor;
            l.specularColor = float3(u.specularColor[0], u.specularColor[1], u.specularColor[2]);
            l.extended = true;
            if ((kMaterialFeatures & MATERIAL_FEATURE_CLEARCOAT) != 0u && (g.features & MATERIAL_FEATURE_CLEARCOAT) != 0u) {
                l.coat = true;
                l.coatFactor = u.clearcoatFactor; l.coatRoughness = u.clearcoatRoughness;
                l.coatNormal = float3(u.clearcoatNormal[0], u.clearcoatNormal[1], u.clearcoatNormal[2]);
            }
            if ((kMaterialFeatures & MATERIAL_FEATURE_SHEEN) != 0u && (g.features & MATERIAL_FEATURE_SHEEN) != 0u) {
                l.sheen = true;
                l.sheenColor = float3(u.sheenColor[0], u.sheenColor[1], u.sheenColor[2]);
                l.sheenRoughness = u.sheenRoughness;
            }
            if ((kMaterialFeatures & MATERIAL_FEATURE_ANISOTROPY) != 0u && (g.features & MATERIAL_FEATURE_ANISOTROPY) != 0u) {
                l.aniso = true;
                l.anisoStrength = u.anisotropyStrength;
                l.anisoDirection = float3(u.anisotropyDirection[0], u.anisotropyDirection[1], u.anisotropyDirection[2]);
            }
            if ((kMaterialFeatures & MATERIAL_FEATURE_IRIDESCENCE) != 0u && (g.features & MATERIAL_FEATURE_IRIDESCENCE) != 0u) {
                l.iridParams.strength = u.iridescenceFactor;
                l.iridParams.ior = u.iridescenceIor;
                l.iridParams.thickness = u.iridescenceThickness;
                l.irid = bsdfIridActive(l.iridParams);
            }
            if ((kMaterialFeatures & MATERIAL_FEATURE_SUBSURFACE) != 0u && (g.features & MATERIAL_FEATURE_SUBSURFACE) != 0u &&
                (g.features & LOBE_GUIDE_FLAG_SUBSURFACE_VOLUMETRIC) == 0u) { // ticket 17: the volumetric mode is the screen-space diffusion, not a thin lobe
                l.thinParams.weight = u.subsurfaceWeight;
                l.thinParams.color = float3(u.subsurfaceColor[0], u.subsurfaceColor[1], u.subsurfaceColor[2]);
                l.thin = bsdfThinActive(l.thinParams);
            }
            if ((kMaterialFeatures & MATERIAL_FEATURE_SUBSURFACE) != 0u && (g.features & MATERIAL_FEATURE_SUBSURFACE) != 0u &&
                (g.features & LOBE_GUIDE_FLAG_SUBSURFACE_VOLUMETRIC) != 0u && (p.flags & DI_SSS_WALK) != 0u)
                l.sssWalk = saturate(u.subsurfaceWeight); // F15.1 (ticket 17, RT tier)
            if ((kMaterialFeatures & MATERIAL_FEATURE_TRANSMISSION) != 0u && (g.features & MATERIAL_FEATURE_TRANSMISSION) != 0u && u.transmissionFactor > 0.0f) {
                l.trans = true;
                l.transFactor = u.transmissionFactor;
                l.transThin = !((g.features & MATERIAL_FEATURE_VOLUME) != 0u && u.volumeThickness > 0.0f);
            }
        }
    }
#endif
    return l;
}
// Same clamps and conventions as diBRDFWeighted; the emitter specular weight scales the specular lobe only.
// F15.1 (ticket 09): `coatWeight` is the emitter share of the coat lobe (diEmitterCoatWeight; 1 for punctual lights).
inline float3 DI_FN(diBRDFLobe)(GPUDISurface surface, DISample s, float specularWeight, DILobe lobe, float coatWeight = 1.0f) {
#ifdef PHOSPHOR_DI_LOBE
    if ((kMaterialFeatures & LOBE_GUIDE_RESOLVED_FEATURES) != 0u && lobe.extended) {
        if (!surface.valid || !s.valid) return float3(0);
        float3 n = diVec(surface.shadingNormal), v = diVec(surface.viewDirection);
        if (!all(isfinite(n)) || !all(isfinite(v)) || !(dot(n, n) > 0 && dot(v, v) > 0)) return float3(0);
        n = normalize(n); v = normalize(v);
        const float nl = saturate(dot(n, s.wi)), nv = max(dot(n, v), 1e-4f);
        const float3 incident = diIncident(s);
        float3 result = float3(0);
        if (nl > 0 && dot(v + s.wi, v + s.wi) > 1e-20f) {
            const float3 h = normalize(v + s.wi);
            const float nh = saturate(dot(n, h)), vh = saturate(dot(v, h));
            const float metallic = saturate(surface.metallic);
            const float3 albedo = max(diVec(surface.albedo), 0.0f);
            const BsdfDielectricSpec spec = bsdfSpecularParams(lobe.ior, lobe.specularFactor, lobe.specularColor);
            // F15.1 (ticket 11): the anisotropic D V of the guide's direction replaces the isotropic GGX.
            BsdfBaseLobes lobes;
            if ((kMaterialFeatures & MATERIAL_FEATURE_IRIDESCENCE) != 0u && lobe.irid) {
                // F15.1 (ticket 12): the thin-film Fresnel at this sample's V.H with the (anisotropic or isotropic) D V of the lobe.
                const float a = bsdfAlpha(surface.roughness);
                const float dv = ((kMaterialFeatures & MATERIAL_FEATURE_ANISOTROPY) != 0u && lobe.aniso)
                    ? bsdfAnisoDV(bsdfAnisoFrame(n, lobe.anisoDirection, surface.roughness, lobe.anisoStrength), v, s.wi, h, nl, nv, nh)
                    : bsdfDGGX(nh, a) * bsdfVSmithGGXCorrelated(nv, nl, a);
                lobes = bsdfSpecEvalIrid(spec, lobe.iridParams, dv, vh, albedo, metallic);
            } else
            lobes = ((kMaterialFeatures & MATERIAL_FEATURE_ANISOTROPY) != 0u && lobe.aniso)
                ? bsdfSpecEvalAniso(spec, bsdfAnisoFrame(n, lobe.anisoDirection, surface.roughness, lobe.anisoStrength), v, s.wi, h, nl, nv, nh, vh,
                                    albedo, metallic)
                : bsdfSpecEval(spec, nl, nv, nh, vh, albedo, metallic, bsdfAlpha(surface.roughness));
            if ((kMaterialFeatures & MATERIAL_FEATURE_SUBSURFACE) != 0u && lobe.thin) lobes.diffuse *= bsdfThinReflectScale(lobe.thinParams);
            // F15.1 (ticket 17, RT tier): the walk's next-event estimation owns the diffuse of the weight share for the analytic (delta) lights.
            if ((kMaterialFeatures & MATERIAL_FEATURE_SUBSURFACE) != 0u && lobe.sssWalk > 0.0f && s.delta) lobes.diffuse *= 1.0f - lobe.sssWalk;
            if ((kMaterialFeatures & MATERIAL_FEATURE_TRANSMISSION) != 0u && lobe.trans) lobes.diffuse *= 1.0f - lobe.transFactor; // ticket 13
            result = (lobes.diffuse + specularWeight * lobes.specular) * incident * nl;
            // F15.1 (ticket 10): the sheen layer between the material and the coat.  The transport does not sample sheen: the
            // emitter share of the sheen lobe is 1 (DI owns it), the material keeps its own emitter weights; the albedo
            // scaling is the min form of the notes.
            if ((kMaterialFeatures & MATERIAL_FEATURE_SHEEN) != 0u && lobe.sheen) {
                const float scale = bsdfSheenScaleLight(lobe.sheenColor, lobe.sheenRoughness, nv, nl, phosphor::SheenTable);
                result = result * scale + bsdfSheenLobe(lobe.sheenColor, lobe.sheenRoughness, nl, nv, nh) * incident * nl;
            }
        }
        // Ticket 16: a light behind the surface (N.L < 0) lights it through the transmitted lobe, weighted by 1 - F(V.H) and |N.L|.
        if ((kMaterialFeatures & MATERIAL_FEATURE_SUBSURFACE) != 0u && lobe.thin) {
            const float nls = dot(n, s.wi);
            if (nls < 0 && dot(v + s.wi, v + s.wi) > 1e-20f)
                result = bsdfThinTransmitted(lobe.thinParams, saturate(dot(v, normalize(v + s.wi))), saturate(surface.metallic)) * incident * (-nls);
        }
        // Ticket 13: a DELTA light behind a thin-walled transmissive surface lights it through the glTF thin BTDF (bsdfThinGlassTransmitted).
        if ((kMaterialFeatures & MATERIAL_FEATURE_TRANSMISSION) != 0u && lobe.trans && lobe.transThin && s.delta && dot(n, s.wi) < 0) {
            const BsdfDielectricSpec spec = bsdfSpecularParams(lobe.ior, lobe.specularFactor, lobe.specularColor);
            result = bsdfThinGlassTransmitted(spec, lobe.transFactor, saturate(surface.metallic), max(diVec(surface.albedo), 0.0f), n, v, s.wi,
                                              bsdfAlpha(surface.roughness)) * incident;
        }
        if ((kMaterialFeatures & MATERIAL_FEATURE_CLEARCOAT) != 0u && lobe.coat) {
            float3 nc = lobe.coatNormal;
            if (!all(isfinite(nc)) || !(dot(nc, nc) > 0)) return float3(0);
            nc = normalize(nc);
            const float fc = bsdfCoatFresnel(abs(dot(nc, v))), ncl = saturate(dot(nc, s.wi)), ncv = max(dot(nc, v), 1e-4f);
            result *= 1.0f - lobe.coatFactor * fc; // the material under the coat (emitter weights stay inside the lobes)
            if (ncl > 0 && dot(v + s.wi, v + s.wi) > 1e-20f) {
                const float3 h = normalize(v + s.wi);
                result += (coatWeight * lobe.coatFactor * fc * bsdfCoatSpecular(ncl, ncv, saturate(dot(nc, h)), bsdfAlpha(lobe.coatRoughness)) * ncl) * incident;
            }
        }
        return result; // caller MUST record nonfinite BEFORE any safe-output sanitization
    }
#endif
    return diBRDFWeighted(surface, s, specularWeight);
}
inline float3 diBRDF(GPUDISurface surface, DISample s) { return diBRDFWeighted(surface, s, 1.0f); }
// Share of a light's specular lobe that primary DI owns (DI_REFLECTED_EMITTER_SPECULAR).
// Same smoothstep as reflectionRTWeight on the same packed roughness, so the
// DI share plus the RT reflection probability is exactly one.
inline float diEmitterSpecularWeight(constant GPUDIParams& p, GPUSampledLight l, GPUDISurface surface) {
    if (!(p.flags & DI_REFLECTED_EMITTER_SPECULAR) || l.type != DI_LIGHT_TRIANGLE) return 1.0f;
    return smoothstep(p.emitterSpecularLow, p.emitterSpecularHigh, surface.roughness);
}
// F15.1 (ticket 09): the coat lobe's share of a sampled emitter (decision 06: the coat lobe's DI/RT emitter share uses the
// coat roughness, the same smoothstep as the base lobe on the surface roughness).
inline float DI_FN(diEmitterCoatWeight)(constant GPUDIParams& p, GPUSampledLight l, DILobe lobe) {
#ifdef PHOSPHOR_DI_LOBE
    if (lobe.coat && (p.flags & DI_REFLECTED_EMITTER_SPECULAR) && l.type == DI_LIGHT_TRIANGLE)
        return smoothstep(p.emitterSpecularLow, p.emitterSpecularHigh, lobe.coatRoughness);
#endif
    return 1.0f;
}
inline float DI_FN(diTarget)(GPUDISurface surface, DISample sample, float positiveFloor, DILobe lobe) {
    if (!sample.valid || !isfinite(positiveFloor) || !(positiveFloor > 0)) return 0;
    const float3 value=DI_FN(diBRDFLobe)(surface,sample,1.0f,lobe);
    if(!all(isfinite(value)))return as_type<float>(0x7fc00000u);
    return max(diLuminance(value),positiveFloor);
}

inline uint diHash(uint v) {
    v ^= v >> 16; v *= 0x7feb352du; v ^= v >> 15; v *= 0x846ca68bu; v ^= v >> 16;
    return v;
}
inline float diWhite(uint x, uint y, uint frame, uint dimension, uint seed) {
    const uint bits = diHash(seed ^ diHash(x) ^ diHash(y + 0x9e3779b9u) ^
                             diHash(frame + 0x632be5abu) ^ diHash(dimension + 0x85157af5u));
    return float(bits >> 8) * (1.0f / 16777216.0f);
}
inline float diRandom(uint2 pixel, uint dimension, constant GPUDIParams& p, const device uint* ranks) {
    if (!(p.flags & DI_USE_STBN) || !p.stbnWidth || !p.stbnHeight || !p.stbnFrames || !p.stbnDimensions)
        return diWhite(pixel.x, pixel.y, p.frameIndex, dimension, p.stbnSeed);
    const uint n = p.stbnWidth * p.stbnHeight * p.stbnFrames;
    const uint index = (((dimension % p.stbnDimensions) * p.stbnFrames + p.frameIndex % p.stbnFrames) *
                        p.stbnHeight + pixel.y % p.stbnHeight) * p.stbnWidth + pixel.x % p.stbnWidth;
    const float rank = (float(ranks[index]) + 0.5f) / float(n);
    return fract(rank + diWhite(0, 0, p.frameIndex / p.stbnFrames, dimension, p.stbnSeed ^ 0xa511e9b3u));
}

inline GPUDIReservoir diEmpty(constant GPUDIParams& p) {
    GPUDIReservoir r{};
    r.lightIndex = ~0u; r.viewID = p.viewID; r.historyEpoch = p.historyEpoch; r.lightRevision = p.lightRevision;
    return r;
}
inline bool diStream(thread GPUDIReservoir& r, GPUDIReservoir sample, float weight, uint m, float random) {
    if (!isfinite(weight) || weight < 0 || !isfinite(random) || random < 0 || random >= 1 ||
        m == 0 || m > ~0u - r.M) { r.pad[0] |= DI_ERROR_WEIGHT; return false; }
    const float sum = r.weightSum + weight;
    if (!isfinite(sum)) { r.pad[0] |= DI_ERROR_WEIGHT; return false; }
    r.M += m;r.pad[1]|=DI_PROPOSAL_VALID;
    if (weight > 0 && random * sum < weight) {
        r.lightIndex = sample.lightIndex; r.lightID = sample.lightID; r.lightGeneration = sample.lightGeneration;
        r.u = sample.u; r.v = sample.v; r.target = sample.target; r.age = sample.age; r.valid = 1;
    }
    r.weightSum = sum; r.normalization = 0;
    return true;
}
inline void diFinalize(thread GPUDIReservoir& r) {
    if(!isfinite(r.weightSum)||!isfinite(r.target)||r.weightSum<0||r.target<0){
        r.pad[0]|=DI_ERROR_WEIGHT;r.pad[1]&=~DI_PROPOSAL_VALID;r.valid=0;r.normalization=0;return;
    }
    // Divide successively to avoid overflow of M*target even when W is finite.
    const float w = r.M > 0 && r.target > 0 ? (r.weightSum / float(r.M)) / r.target : 0;
    r.valid = r.valid && r.pad[0] == 0 && isfinite(w) && w > 0;
    r.normalization = r.valid ? w : 0;
}
inline bool diValid(GPUDIReservoir r, GPUSampledLight light, constant GPUDIParams& p) {
    return r.valid && r.pad[0] == 0 && r.M > 0 && r.viewID == p.viewID &&
           r.historyEpoch == p.historyEpoch && r.lightRevision == p.lightRevision &&
           r.lightID == light.id && r.lightGeneration == light.generation && isfinite(r.target) && r.target > 0 &&
           isfinite(r.normalization) && r.normalization > 0 && isfinite(r.u) && isfinite(r.v) &&
           r.u >= 0 && r.u < 1 && r.v >= 0 && r.v < 1;
}
inline bool diZeroProposal(GPUDIReservoir r,constant GPUDIParams& p) {
    return (r.pad[1]&DI_PROPOSAL_VALID) && !r.valid && r.M>0 && r.pad[0]==0 &&
        r.weightSum==0 && r.normalization==0 && isfinite(r.target) && r.age<p.maxHistoryAge &&
        r.viewID==p.viewID && r.historyEpoch==p.historyEpoch && r.lightRevision==p.lightRevision;
}
inline bool diReusable(GPUDIReservoir r, GPUSampledLight light, constant GPUDIParams& p) {
    return diZeroProposal(r,p) || (diValid(r, light, p) && r.age < p.maxHistoryAge);
}
inline bool diCompatible(GPUDISurface a, GPUDISurface b, constant GPUDIParams& p, bool temporal) {
    if (!a.valid || !b.valid || !isfinite(a.depth) || !isfinite(b.depth) || a.depth <= 0 || b.depth <= 0) return false;
    if (temporal && (a.instanceSlot != b.instanceSlot || a.instanceGeneration != b.instanceGeneration ||
                     a.materialRevision != b.materialRevision)) return false;
    const float3 an = diVec(a.geometricNormal), bn = diVec(b.geometricNormal);
    const float3 ap = diVec(a.position), bp = diVec(b.position);
    if (!all(isfinite(an)) || !all(isfinite(bn)) || !all(isfinite(ap)) || !all(isfinite(bp)) ||
        !(dot(an, an) > 0 && dot(bn, bn) > 0) || dot(normalize(an), normalize(bn)) < p.normalThreshold) return false;
    const float tolerance = p.depthRelativeThreshold * max(a.depth, b.depth);
    return abs(a.depth - b.depth) <= tolerance && abs(dot(bp - ap, normalize(an))) <= tolerance;
}
inline bool DI_FN(diMerge)(thread GPUDIReservoir& r, GPUDIReservoir source, GPUDISurface destination,
                     GPUSampledLight light, constant GPUDIParams& p, float random, bool advanceAge,
                     const device GPUEmissiveSurface* emitters,
                     const device GPUMaterial* materials, const device DITextureHandle* textures, DILobe lobe) {
    const bool zero = diZeroProposal(source,p);
    if (!zero && !diReusable(source, light, p)) return false;
    // Bounds the complete history chain independently of which endpoint wins.
    // Both validity predicates guarantee source.age < maxHistoryAge, so +1
    // cannot overflow even when the configured upper bound is UINT_MAX.
    const uint mergedAge = max(r.age, source.age + uint(advanceAge));
    const uint m = min(source.M, p.maxHistoryM);
    if (zero) {
        const bool accepted = diStream(r,source,0,m,random);
        if (accepted) r.age = mergedAge;
        return accepted;
    }
    const DISample sample = diSampleTexturedLight(light, source.lightIndex, float2(source.u, source.v),
                                                 diVec(destination.position), emitters, materials, textures, p.materialExtensions);
    source.target = DI_FN(diTarget)(destination, sample, p.targetFloor, lobe);
    const bool accepted = diStream(r, source, source.target * source.normalization * float(m), m, random);
    if (accepted) r.age = mergedAge; // A fresh selection must not rejuvenate old mass.
    return accepted;
}

// Persistent diagnostics survive sanitization; this is independent of output
// texture consistency and does not claim an energetic reference comparison.
inline void diRecordNonfinite(float3 value,device atomic_uint* errors,uint word) {
    if(!all(isfinite(value)))atomic_fetch_add_explicit(errors+word,1u,memory_order_relaxed);
}
inline bool diLightFinite(GPUSampledLight light) {
    return all(isfinite(diVec(light.emission))) && all(isfinite(diVec(light.position))) &&
           all(isfinite(diVec(light.axisU))) && all(isfinite(diVec(light.axisV))) &&
           isfinite(light.range) && isfinite(light.radius) && isfinite(light.innerCone) && isfinite(light.outerCone);
}
