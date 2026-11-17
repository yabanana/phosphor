#pragma once
// ---------------------------------------------------------------------------
// F15.1 (ticket 09): clearcoat lobe selection of the specular transport and the stable denoiser key.
// Compiled by C++ (unit tests) and MSL (reflection kernels) from this single source; scalars only.
//
// The coated specular BSDF has two lobes: the coat (weight w_c = c F_c(|V.Nc|)) and the material under it
// (weight w_b = (1 - w_c) * E_b, E_b the directional albedo of the base specular lobe).  The transport traces one
// ray per pixel and picks the lobe by its albedo weight:
//   P_coat = clamp(w_c / (w_c + w_b), kCoatSelectFloor, 1 - kCoatSelectFloor)      (0 when w_c = 0)
// The sample weight divides by the MIXTURE pdf P_c pdf_c(wi) + (1 - P_c) pdf_b(wi) (one-sample MIS, unbiased for any
// P in (0,1)); the floor only bounds the variance when the albedo weights misjudge a lobe.
//
// The per-sample lobe flips from frame to frame, so the denoiser must not key its history on the sampled lobe.  The
// KEY recorded in the sample flags is a pure function of the pixel (its guide and view): the coat lobe when it is
// significant (w_c >= kCoatSignificantShare * (w_c + w_b)), else the base lobe.  For a significant coat the denoiser
// takes the SHARPER of the two roughnesses as the edge-stopping roughness (the smooth coat reflection must survive, the
// rough base lobe is filtered no wider than the coat image tolerates): denoiseLobeRoughness in denoise_settings.cpp.
// ---------------------------------------------------------------------------
#include "renderer/lobe_guide_pack.h"

namespace phosphor {

PHOSPHOR_GPU_CONSTANT float kCoatSelectFloor = 1.0f / 32.0f;
PHOSPHOR_GPU_CONSTANT float kCoatSignificantShare = 0.03f;

/// F_c = 0.04 + 0.96 (1 - |V.Nc|)^5 (coat IOR 1.5).
PHOSPHOR_LOBE_FN float lobeCoatFresnel(float nv) {
    const float x = 1.0f - (nv < 0.0f ? 0.0f : (nv > 1.0f ? 1.0f : nv));
    const float x2 = x * x;
    return 0.04f + 0.96f * (x2 * x2 * x);
}

struct CoatLobeSelect {
    float coatWeight;  // w_c
    float baseWeight;  // w_b
    float pCoat;       // probability of sampling the coat lobe
    bool significant;  // the stable key is the coat lobe
};

/// `factor` the coat factor, `nv` = |V.Nc|, `baseAlbedo` = the directional albedo of the base specular lobe (>= 0).
PHOSPHOR_LOBE_FN CoatLobeSelect coatLobeSelect(float factor, float nv, float baseAlbedo) {
    CoatLobeSelect s;
    const float c = factor < 0.0f ? 0.0f : (factor > 1.0f ? 1.0f : factor);
    s.coatWeight = c * lobeCoatFresnel(nv);
    const float b = baseAlbedo > 0.0f ? baseAlbedo : 0.0f;
    s.baseWeight = (1.0f - s.coatWeight) * b;
    const float sum = s.coatWeight + s.baseWeight;
    float p = sum > 0.0f ? s.coatWeight / sum : 0.0f;
    if (s.coatWeight > 0.0f) p = p < kCoatSelectFloor ? kCoatSelectFloor : (p > 1.0f - kCoatSelectFloor ? 1.0f - kCoatSelectFloor : p);
    else p = 0.0f;
    s.pCoat = p;
    s.significant = s.coatWeight > 0.0f && s.coatWeight >= kCoatSignificantShare * sum;
    return s;
}

} // namespace phosphor
