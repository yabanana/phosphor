#pragma once
// ---------------------------------------------------------------------------
// F15.1 (ticket 13): scalar math of ray-traced KHR_materials_transmission + KHR_materials_volume, compiled by C++ (unit tests) and MSL
// (the specular transport, shaders/reflection_transmission.h) from this single source; scalars only.
//
// The model (ADR-0004, decisions 06/09, tools/f15_transmission_reference.py):
//  - every interface is a GGX microfacet dielectric sampled by the D(h) n.h proposal.  The one-sample estimator of reflection AND
//    refraction through a sampled half vector h is the Walter et al. weight |i.h| G(i,o,h) / (|i.n| |h.n|) with the SEPARABLE Smith G
//    (glTF's G_T, Mitsuba's roughdielectric); the Fresnel factor cancels because the branch is chosen with probability F (reflect) or
//    1 - F (refract).  F is glTF's Schlick form with the TRANSMITTED cosine on the leaving side and F = 1 under total internal reflection.
//  - the camera-visible surface keeps the glTF layering: reflection is the existing specular lobe, refraction carries
//    (1 - F) transmission baseColor (1 - metallic); the branch is chosen with the probability transmissionSelect (floor 1/32), the
//    weight of each branch divides by its probability (unbiased for any probability in (0, 1)).
//  - the medium of a closed volume is homogeneous and purely absorbing: T = c^(x / d) = exp(-sigma_t x), sigma_t = -ln(c) / d.
// ---------------------------------------------------------------------------
#include "renderer/lobe_guide_pack.h"
#ifdef __METAL_VERSION__
#define PHOSPHOR_LOBE_EXP(x) metal::precise::exp(x)
#define PHOSPHOR_LOBE_LOG(x) metal::precise::log(x)
#define PHOSPHOR_LOBE_ABS(x) metal::abs(x)
#else
#include <cmath>
#define PHOSPHOR_LOBE_EXP(x) std::exp(x)
#define PHOSPHOR_LOBE_LOG(x) std::log(x)
#define PHOSPHOR_LOBE_ABS(x) std::fabs(x)
#endif

namespace phosphor {

PHOSPHOR_GPU_CONSTANT float kTransmissionSelectFloor = 1.0f / 32.0f;
/// The declared bound of interface CROSSINGS (transmissions) along one transport sample, the camera-visible refraction included: a path that would
/// cross a fifth interface ends in the probes/environment along the refracted direction.  Reflections (total internal reflection inside a cube)
/// do not cross an interface; the hard bound kTransmissionMaxEvents on all interface events keeps every loop finite (GPU safety).
PHOSPHOR_GPU_CONSTANT unsigned kTransmissionMaxCrossings = 4u;
PHOSPHOR_GPU_CONSTANT unsigned kTransmissionMaxEvents = 8u;
/// An interface whose GGX alpha is at the engine's floor (bsdfAlpha: max(roughness^2, 0.002); roughness 0 clamps to it) is a SMOOTH interface for the
/// transmission transport: h = n and weight 1 (the delta of the reference's `dielectric`).
PHOSPHOR_GPU_CONSTANT float kTransmissionSmoothAlpha = 0.00201f;
/// Smallest attenuation colour channel: sigma_t = -ln(max(c, kTransmissionMinColor)) / d (the reference clips at 1e-6).
PHOSPHOR_GPU_CONSTANT float kTransmissionMinColor = 1.0e-6f;

PHOSPHOR_LOBE_FN float transmissionClamp01(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }
PHOSPHOR_LOBE_FN float transmissionMax(float a, float b) { return a > b ? a : b; }

struct TransmissionFresnel {
    float f;       // reflectance
    float cosT;    // cosine of the transmitted angle (0 under total internal reflection)
    bool tir;
};

/// Schlick Fresnel of one interface for the incident cosine `cosI` (>= 0) and `eta` = n_transmitted / n_incident: f0 = ((eta - 1) /
/// (eta + 1))^2, F = f0 + (1 - f0)(1 - c)^5 with c = cosI entering (eta >= 1) and the TRANSMITTED cosine leaving (KHR_materials_volume
/// F_TR-); total internal reflection gives F = 1.
PHOSPHOR_LOBE_FN TransmissionFresnel transmissionFresnel(float cosI, float eta) {
    TransmissionFresnel r;
    const float c = transmissionClamp01(cosI);
    const float sin2T = (1.0f - c * c) / (eta * eta);
    if (!(sin2T < 1.0f)) {
        r.f = 1.0f;
        r.cosT = 0.0f;
        r.tir = true;
        return r;
    }
    r.cosT = PHOSPHOR_LOBE_SQRT(1.0f - sin2T);
    const float f0n = (eta - 1.0f) / (eta + 1.0f);
    const float f0 = f0n * f0n;
    const float x = 1.0f - (eta >= 1.0f ? c : r.cosT);
    const float x2 = x * x;
    r.f = f0 + (1.0f - f0) * (x2 * x2 * x);
    r.tir = false;
    return r;
}

/// Probability of choosing the REFLECTION branch of the camera-visible surface: wR / (wR + wT) clamped to [floor, 1 - floor] (1 when the
/// surface transmits nothing).  `reflectWeight` = the albedo of the specular layer at N.V, `transmitWeight` = (1 - F) t (1 - metallic) max(tint).
PHOSPHOR_LOBE_FN float transmissionSelect(float reflectWeight, float transmitWeight) {
    const float r = transmissionMax(reflectWeight, 0.0f), t = transmissionMax(transmitWeight, 0.0f);
    if (!(t > 0.0f)) return 1.0f;
    const float sum = r + t;
    float p = sum > 0.0f ? r / sum : 0.5f;
    p = p < kTransmissionSelectFloor ? kTransmissionSelectFloor : (p > 1.0f - kTransmissionSelectFloor ? 1.0f - kTransmissionSelectFloor : p);
    return p;
}

/// Beer-Lambert transmittance of one channel over `length` metres of a homogeneous medium: c^(length / distance) = exp(-sigma_t length).
/// A medium without attenuation (distance <= 0 or infinite, colour 1) transmits everything.
PHOSPHOR_LOBE_FN float transmissionBeerLambert(float color, float distance, float length) {
    if (!(distance > 0.0f) || !(distance < 3.0e38f) || !(length > 0.0f)) return 1.0f;
    const float c = color < kTransmissionMinColor ? kTransmissionMinColor : (color > 1.0f ? 1.0f : color);
    return PHOSPHOR_LOBE_EXP(PHOSPHOR_LOBE_LOG(c) * (length / distance));
}

/// Smith G1 of the GGX distribution for the direction with cosine `cosN` to the macro normal, as seen from the microfacet with
/// v.h = `vDotH`: 0 when the direction is on the other side of the microfacet than of the macro surface (the facet is back-facing for it).
PHOSPHOR_LOBE_FN float transmissionG1(float cosN, float vDotH, float alpha) {
    if (!(cosN * vDotH > 0.0f)) return 0.0f;
    const float c2 = cosN * cosN;
    const float tan2 = (1.0f - c2) / transmissionMax(c2, 1.0e-12f);
    return 2.0f / (1.0f + PHOSPHOR_LOBE_SQRT(1.0f + alpha * alpha * tan2));
}

/// Walter et al. (eq. 41) weight of a direction sampled through the D(h) n.h half-vector proposal, reflection and refraction alike:
/// |i.h| G1(i) G1(o) / (|i.n| |h.n|).  `iN`, `oN` the cosines of the incident and the outgoing direction with the macro normal
/// (signed: the outgoing direction is on the same side as the incident one for a reflection), `iH` = i.h, `oH` = o.h, `hN` = h.n.
PHOSPHOR_LOBE_FN float transmissionMicrofacetWeight(float iN, float oN, float iH, float oH, float hN, float alpha) {
    const float g = transmissionG1(iN, iH, alpha) * transmissionG1(oN, oH, alpha);
    const float d = transmissionMax(PHOSPHOR_LOBE_ABS(iN) * PHOSPHOR_LOBE_ABS(hN), 1.0e-12f);
    return PHOSPHOR_LOBE_ABS(iH) * g / d;
}

/// Smith G1 of the GGX distribution without the facet test (|cosN| the cosine to the macro normal): 2 |c| / (|c| + sqrt(a^2 + (1 - a^2) c^2)).
PHOSPHOR_LOBE_FN float transmissionG1Plain(float cosN, float alpha) {
    const float c = PHOSPHOR_LOBE_ABS(cosN);
    const float a2 = alpha * alpha;
    return 2.0f * c / transmissionMax(c + PHOSPHOR_LOBE_SQRT(a2 + (1.0f - a2) * c * c), 1.0e-12f);
}

/// KHR_materials_transmission thin-walled specular_btdf (README, non-normative): the GGX D of the half vector H_T = normalize(V + L')
/// with L' the light mirrored to the front side, times the SEPARABLE Smith G_T / (4 |N.L| N.V) (= nu_T): the value of f_t without the Fresnel
/// factor, the tint and the cosine.  `nlAbs` = |N.L|, `nv` = N.V, `nh` = N.H_T (all > 0, else 0).
PHOSPHOR_LOBE_FN float transmissionThinBtdf(float nlAbs, float nv, float nh, float alpha) {
    if (!(nlAbs > 0.0f && nv > 0.0f && nh > 0.0f)) return 0.0f;
    const float a2 = alpha * alpha;
    const float d = nh * nh * (a2 - 1.0f) + 1.0f;
    const float D = a2 / (3.14159265358979323846f * d * d);
    return D * transmissionG1Plain(nlAbs, alpha) * transmissionG1Plain(nv, alpha) / (4.0f * nlAbs * nv);
}


// ---------------------------------------------------------------------------
// F15.1 (ticket 14): attenuated coloured shadows (ADR-0004 amendment).  A visibility ray crossing transmissive casters multiplies, in
// log space (so the product is order independent and a zero payload is transmittance 1), per crossed SURFACE the factor (1 - F) with F the
// Schlick Fresnel of the STRAIGHT ray at the geometric normal (transmissionFresnel with eta = ior), per medium the Beer-Lambert factor of the
// signed distance sum (an entry surface adds +sigma_t t, an exit surface -sigma_t t: the medium length is the sum of the (exit - entry)
// distances along the straight ray), and ONCE per caster the transmission factor x (1 - metallic) x tint at the ENTRY surface (a thin-walled
// single surface counts as the entry).  Caustics are excluded: the ray never bends.
// ---------------------------------------------------------------------------
PHOSPHOR_GPU_CONSTANT float kShadowLogFloor = 1.0e-30f; // a zero factor is ln(1e-30) = -69: exp() of the sum is 0 at float precision
PHOSPHOR_GPU_CONSTANT float kShadowDefaultIor = 1.5f;   // a transmissive material without the ior block (glTF default)

/// ln(1 - F) of one crossed interface for the ray cosine `cosI` (the |cosine| with the geometric normal); ior <= 0 (the special no-Fresnel mode) gives 0.
PHOSPHOR_LOBE_FN float transmissionShadowLnPass(float cosI, float ior) {
    if (!(ior > 0.0f)) return 0.0f;
    const TransmissionFresnel f = transmissionFresnel(PHOSPHOR_LOBE_ABS(cosI), ior);
    const float pass = 1.0f - f.f;
    return PHOSPHOR_LOBE_LOG(pass < kShadowLogFloor ? kShadowLogFloor : pass);
}

/// ln of a non-negative factor, floored.
PHOSPHOR_LOBE_FN float transmissionShadowLn(float factor) {
    return PHOSPHOR_LOBE_LOG(factor < kShadowLogFloor ? kShadowLogFloor : factor);
}

/// sigma_t of one channel of a homogeneous medium (1 / metres): -ln(c) / distance, 0 without attenuation (colour 1, distance <= 0 or infinite).
PHOSPHOR_LOBE_FN float transmissionShadowSigma(float color, float distance) {
    if (!(distance > 0.0f) || !(distance < 3.0e38f)) return 0.0f;
    const float c = color < kTransmissionMinColor ? kTransmissionMinColor : (color > 1.0f ? 1.0f : color);
    return -PHOSPHOR_LOBE_LOG(c) / distance;
}

} // namespace phosphor
