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
#define REFL_LOBE(flags, flag, i) reflectionLoadLobe(flags, flag, guides, i)
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
// The specular lobe of one pixel.  Today a single lobe (ior/specular, KHR_materials_specular); ticket 09 (clearcoat)
// turns this into a lobe set: reflectionDirectionLobe then picks the lobe with probability proportional to its albedo
// weight and reflectionLobeWeight divides by the mixture pdf over the lobes instead of the single-lobe pdf.
struct ReflectionLobe {
    bool extended;
    uint id;       // the sampled lobe (SPECULAR_LOBE_*): 0, the base specular lobe, until ticket 09 samples a second one
    bool refract;  // the sampled axis: false = reflection (ticket 13 samples refraction)
    BsdfDielectricSpec spec;
};
inline ReflectionLobe reflectionLoadLobe(uint flags, uint validFlag, const device GPULobeGuide* guides, uint index) {
    ReflectionLobe l{};
    if ((kMaterialFeatures & MATERIAL_FEATURE_SPECULAR_IOR) != 0u && (flags & validFlag) != 0u) {
        const GPULobeGuide g = guides[index];
        if ((g.features & MATERIAL_FEATURE_SPECULAR_IOR) != 0u) {
            const LobeParams u = unpackLobeGuide(g);
            l.spec = bsdfSpecularParams(u.ior, u.specularFactor, float3(u.specularColor[0], u.specularColor[1], u.specularColor[2]));
            l.extended = true;
        }
    }
    return l;
}
#endif
