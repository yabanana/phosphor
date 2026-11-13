#pragma once
#include "restir_common.h"
#include "materials/bsdf.h"
#include "renderer/denoise_history_layout.h"
// F15.1 (ticket 08): lobe-guide variants.  The "_lobe" kernels (reflections.metal, reflection_probes.metal and
// reflection_composite.metal compiled with PHOSPHOR_REFL_LOBE by reflection_lobe.metal, linked into the material
// module like the DI lobe kernels) are separate entry points that carry the reserved MATERIAL_FEATURES constant;
// base kernels never reference it (metal-tt cannot lower an unresolved constant), so scenes without features keep
// the base expressions bit for bit.  The guide has the layout and index of GPUDISurface (width = p.width).
#ifdef PHOSPHOR_REFL_LOBE
#define REFL_KERNEL(name) name##_lobe
#define REFL_GUIDE_PARAM(index) , const device GPULobeGuide* guides [[buffer(index)]]
#define REFL_LOBE(flags, flag, i, surface) reflectionLoadLobe(flags, flag, guides, i, surface)
#define REFL_DIRECTION(s, u, lobe) reflectionDirectionLobe(s, u, lobe)
// The sampled lobe's (id, axis) key into the sample flags read by the "_lobe" denoiser kernels (denoise.metal, lobe contract).
#define REFL_TAG(sample, lobe) if (((sample).flags & SPECULAR_SAMPLE_VALID) != 0u) (sample).flags |= specularSampleLobeFlags((lobe).id, (lobe).refract)
#else
#define REFL_KERNEL(name) name
#define REFL_GUIDE_PARAM(index)
#define REFL_TAG(sample, lobe)
#endif
#ifdef PHOSPHOR_REFL_LOBE
#ifndef PHOSPHOR_DI_LOBE
#error "PHOSPHOR_REFL_LOBE needs the lobe-mode restir_common.h (kMaterialFeatures)"
#endif
// The specular lobe set of one pixel: the base specular lobe (ior/specular, KHR_materials_specular) and, ticket 09, the
// clearcoat lobe.  reflectionLobePick draws the lobe with probability proportional to its albedo weight (renderer/lobe_select.h) and
// reflectionLobeWeight divides by the mixture pdf over the lobes (one-sample MIS).
struct ReflectionLobe {
    bool extended;  // an ior/specular or clearcoat feature is resolved for this pixel
    uint id;        // the denoiser key (SPECULAR_LOBE_*): STABLE per pixel, the coat lobe when it is significant (lobe_select.h)
    bool refract;   // the sampled axis: false = reflection (ticket 13 samples refraction)
    BsdfDielectricSpec spec;
    bool coat;      // clearcoat resolved
    bool sampleCoat; // this sample's lobe (set by reflectionLobePick)
    float coatFactor, coatRoughness;
    float3 coatNormal;
    float pCoat;    // probability of sampling the coat lobe
    // F15.1 (ticket 10): KHR_materials_sheen.  The transport does NOT sample sheen (decision 06): the weight of the sampled base
    // specular lobe carries the view-only albedo scaling sheenScale = 1 - max3(S) E(N.V), and the composite adds the sheen
    // environment term S E L(N).  sheenScale is 1 for a pixel without sheen.
    bool sheen;
    float3 sheenColor;
    float sheenRoughness, sheenAlbedo, sheenScale;
    // F15.1 (ticket 11): KHR_materials_anisotropy.  The base specular lobe is sampled by anisotropic visible-normal sampling (the coat
    // lobe, when present, stays isotropic): the weight is the exact f cos / pdf of that proposal.  anisoDirection is the guide's T'
    // (frame rebuilt with the primary normal, bsdfAnisoFrame); envRoughness is the roughness of every split-sum term (the effective
    // isotropic roughness of the lobe, the base roughness without anisotropy).
    bool aniso;
    float3 anisoDirection;
    float anisoStrength;
    float envRoughness;
    // F15.1 (ticket 12): KHR_materials_iridescence.  The sample weight uses the thin-film Fresnel at the sampled V.H (bsdfSpecColorIrid);
    // every split-sum term (probe weight, composite diffuse complement, coat select) uses it at N.V (bsdfSpecEnvIrid).  strength 0 = no film.
    BsdfIrid irid;
    // F15.1 (ticket 16): subsurface thin mode.  The composite scales the front diffuse by 1 - w and adds the transmitted environment/GI term
    // (bsdfThinEnvWeight; GI: the pixel's irradiance stands for E(-N), declared).  The specular transport is unchanged.
    bool thin;
    BsdfThin thinParams;
    float diffuseRoughness; // roughness of the diffuse complement in the composite (bsdfAnisoDiffuseRoughness), the base roughness without anisotropy
};
inline ReflectionLobe reflectionLoadLobe(uint flags, uint validFlag, const device GPULobeGuide* guides, uint index, GPUDISurface s) {
    ReflectionLobe l{};
    l.sheenScale = 1.0f;
    l.envRoughness = s.roughness;
    l.diffuseRoughness = s.roughness;
    if ((kMaterialFeatures & LOBE_GUIDE_RESOLVED_FEATURES) != 0u && (flags & validFlag) != 0u) {
        const GPULobeGuide g = guides[index];
        if ((g.features & LOBE_GUIDE_RESOLVED_FEATURES) != 0u) {
            const LobeParams u = unpackLobeGuide(g); // neutral ior 1.5 / specular 1 for a clear feature: f0 0.04, f90 1
            l.spec = bsdfSpecularParams(u.ior, u.specularFactor, float3(u.specularColor[0], u.specularColor[1], u.specularColor[2]));
            l.extended = true;
            if ((kMaterialFeatures & MATERIAL_FEATURE_SHEEN) != 0u && (g.features & MATERIAL_FEATURE_SHEEN) != 0u) {
                float3 n = diVec(s.shadingNormal), v = diVec(s.viewDirection);
                if (s.valid && all(isfinite(n)) && all(isfinite(v)) && dot(n, n) > 0 && dot(v, v) > 0) {
                    n = normalize(n); v = normalize(v);
                    l.sheen = true;
                    l.sheenColor = float3(u.sheenColor[0], u.sheenColor[1], u.sheenColor[2]);
                    l.sheenRoughness = u.sheenRoughness;
                    l.sheenAlbedo = bsdfSheenE(l.sheenRoughness, dot(n, v), phosphor::SheenTable);
                    l.sheenScale = 1.0f - bsdfMax3(l.sheenColor) * l.sheenAlbedo;
                }
            }
            if ((kMaterialFeatures & MATERIAL_FEATURE_IRIDESCENCE) != 0u && (g.features & MATERIAL_FEATURE_IRIDESCENCE) != 0u) {
                l.irid.strength = u.iridescenceFactor;
                l.irid.ior = u.iridescenceIor;
                l.irid.thickness = u.iridescenceThickness;
            }
            if ((kMaterialFeatures & MATERIAL_FEATURE_SUBSURFACE) != 0u && (g.features & MATERIAL_FEATURE_SUBSURFACE) != 0u) {
                l.thinParams.weight = u.subsurfaceWeight;
                l.thinParams.color = float3(u.subsurfaceColor[0], u.subsurfaceColor[1], u.subsurfaceColor[2]);
                l.thin = bsdfThinActive(l.thinParams);
            }
            if ((kMaterialFeatures & MATERIAL_FEATURE_ANISOTROPY) != 0u && (g.features & MATERIAL_FEATURE_ANISOTROPY) != 0u) {
                const float3 ad = float3(u.anisotropyDirection[0], u.anisotropyDirection[1], u.anisotropyDirection[2]);
                float3 n = diVec(s.shadingNormal), v = diVec(s.viewDirection);
                if (s.valid && all(isfinite(n)) && all(isfinite(v)) && dot(n, n) > 0 && dot(v, v) > 0 && all(isfinite(ad)) && dot(ad, ad) > 0 &&
                    isfinite(u.anisotropyStrength) && isfinite(s.roughness)) {
                    l.aniso = true;
                    l.anisoStrength = u.anisotropyStrength;
                    l.anisoDirection = ad;
                    l.envRoughness = bsdfAnisoEnvRoughness(s.roughness, l.anisoStrength);
                    l.diffuseRoughness = bsdfAnisoDiffuseRoughness(s.roughness, l.anisoStrength);
                }
            }
            if ((kMaterialFeatures & MATERIAL_FEATURE_CLEARCOAT) != 0u && (g.features & MATERIAL_FEATURE_CLEARCOAT) != 0u) {
                const float3 nc = float3(u.clearcoatNormal[0], u.clearcoatNormal[1], u.clearcoatNormal[2]);
                float3 n = diVec(s.shadingNormal), v = diVec(s.viewDirection);
                if (s.valid && all(isfinite(n)) && all(isfinite(v)) && dot(n, n) > 0 && dot(v, v) > 0 && all(isfinite(nc)) && dot(nc, nc) > 0) {
                    n = normalize(n); v = normalize(v);
                    l.coat = true;
                    l.coatFactor = u.clearcoatFactor;
                    l.coatRoughness = u.clearcoatRoughness;
                    l.coatNormal = normalize(nc);
                    const float baseAlbedo = bsdfMax3(bsdfSpecEnvIrid(l.spec, l.irid, diVec(s.albedo), saturate(s.metallic), l.envRoughness,
                                                                      max(dot(n, v), 1e-4f), phosphor::DfgTable).specular);
                    const CoatLobeSelect sel = coatLobeSelect(l.coatFactor, abs(dot(l.coatNormal, v)), baseAlbedo);
                    l.pCoat = sel.pCoat;
                    l.id = sel.significant ? SPECULAR_LOBE_CLEARCOAT : SPECULAR_LOBE_BASE;
                }
            }
        }
    }
    return l;
}
// The lobe of this sample: the coat with probability pCoat (uSelect uniform in [0,1)).
inline void reflectionLobePick(thread ReflectionLobe& lobe, float uSelect) { lobe.sampleCoat = lobe.coat && uSelect < lobe.pCoat; }
// The surface as the sampled lobe sees it: the coat lobe has the coat roughness and the coat normal (everything
// downstream, RT band, emitter share, SSR steps, probe filter, regularisation, then follows the coat roughness).
inline GPUDISurface reflectionLobeSurface(GPUDISurface s, ReflectionLobe lobe) {
    if (lobe.coat && lobe.sampleCoat) {
        s.roughness = lobe.coatRoughness;
        for (uint i = 0; i < 3; ++i) s.shadingNormal[i] = lobe.coatNormal[i];
    }
    return s;
}
#endif
