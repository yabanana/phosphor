#pragma once

// ---------------------------------------------------------------------------
// F15.1 (ticket 17/18, RT tier) -- scalar math of the volumetric-subsurface random walk (shaders/subsurface_walk.metal), compiled by C++ (tests)
// and by MSL from this single source.  Scalars only; the vector parts of the walk live in the shader.
//
// The boundary of the medium is the rough dielectric the ground truth uses (Mitsuba 3.9.1 `roughdielectric`): a single-scattering GGX
// microfacet BSDF with the separable Smith masking G(i, o, m) = G1(i, m) G1(o, m), ior 1.5 over vacuum; the shape is
// the engine's roughness (alpha = max(roughness^2, 0.002)).  The medium is homogeneous: extinction 1 / (radius x radiusScale) per colour channel,
// single-scatter albedo from the observed-albedo table (renderer/subsurface_walk_table.h, tools/f15_subsurface_lut.py: the albedo of a
// semi-infinite slab under THIS boundary equals the material's subsurface colour, the OpenPBR definition), Henyey-Greenstein phase.
// ---------------------------------------------------------------------------

#include "renderer/gpu_types.h"
#include "renderer/subsurface_walk_table.h"

#ifdef __METAL_VERSION__
#define PHOSPHOR_SW_THREAD thread
#define PHOSPHOR_SW_CONSTANT_PTR const device
#else
#include <algorithm>
#include <cmath>
#define PHOSPHOR_SW_THREAD
#define PHOSPHOR_SW_CONSTANT_PTR const
#endif

namespace phosphor {

PHOSPHOR_GPU_CONSTANT float kSubsurfaceIor = 1.5f; // the boundary (the albedo table is measured behind this interface)

/// Unpolarised dielectric Fresnel reflectance; cosI >= 0 measured on the side of the incident ray, eta = n_transmitted / n_incident.
/// Total internal reflection returns 1.
inline float subsurfaceFresnel(float cosI, float eta) {
    cosI = cosI < 0.0f ? 0.0f : (cosI > 1.0f ? 1.0f : cosI);
    const float sin2T = (1.0f - cosI * cosI) / (eta * eta);
    if (sin2T >= 1.0f) return 1.0f;
    const float cosT = sqrt(1.0f - sin2T);
    const float rs = (cosI - eta * cosT) / (cosI + eta * cosT);
    const float rp = (eta * cosI - cosT) / (eta * cosI + cosT);
    return 0.5f * (rs * rs + rp * rp);
}

/// GGX normal distribution, cosM = m . n (> 0), isotropic alpha.
inline float subsurfaceGgxD(float cosM, float alpha) {
    if (cosM <= 0.0f) return 0.0f;
    const float a2 = alpha * alpha;
    const float d = cosM * cosM * (a2 - 1.0f) + 1.0f;
    return a2 / (3.14159265358979f * d * d);
}

/// Smith masking of one direction: cosV = v . n, vm = v . m (the sign test of Mitsuba's smith_g1: the microfacet must face the direction).
inline float subsurfaceGgxG1(float cosV, float vm, float alpha) {
    if (vm * cosV <= 0.0f) return 0.0f;
    const float c2 = cosV * cosV;
    if (c2 <= 0.0f) return 0.0f;
    const float tan2 = (1.0f - c2) / c2;
    return 2.0f / (1.0f + sqrt(1.0f + alpha * alpha * tan2));
}

/// Transmission BTDF of the rough boundary times |cos theta_i| (Walter et al. 2007, eq. 21 in radiance form: n_o^2 with n_o the index on the
/// side of the outgoing direction o = the direction towards the sensor; i = the light direction on the side n_i):
///   f_t |i.n| = |i.m| |o.m| / |o.n| n_o^2 (1 - F) G1(i) G1(o) D(m) / (n_i |i.m| - n_o |o.m|)^2
/// (i.m and o.m have opposite signs across the interface, so the denominator of Walter's eq. 21 is the difference of the magnitudes).
/// cosIm = |i.m|, cosOm = |o.m|, cosOn = |o.n|, cosM = m.n.
inline float subsurfaceBtdfCos(float cosIm, float cosOm, float cosOn, float cosM, float alpha, float ni, float no, float fresnel, float g1i,
                               float g1o) {
    const float walter = ni * cosIm - no * cosOm;
    if (cosOn <= 0.0f || walter == 0.0f) return 0.0f;
    return cosIm * cosOm / cosOn * no * no * (1.0f - fresnel) * g1i * g1o * subsurfaceGgxD(cosM, alpha) / (walter * walter);
}

/// Henyey-Greenstein: cos of the scattering angle relative to the propagation direction for a uniform u in [0, 1).
inline float subsurfaceHgCos(float g, float u) {
    if (g > -1e-3f && g < 1e-3f) return 1.0f - 2.0f * u;
    const float s = (1.0f - g * g) / (1.0f - g + 2.0f * g * u);
    return (1.0f + g * g - s * s) / (2.0f * g);
}

/// Single-scatter albedo whose semi-infinite slab under the rough ior 1.5 boundary reflects the observed albedo `colour`
/// (OpenPBR subsurface_color).  The table stores sqrt(1 - albedo) over the colour axis; roughness and g are interpolated linearly and
/// clamped to the measured range (kSubsurfaceTable*).  A colour above the largest albedo the boundary lets through returns 1 (saturated).
inline float subsurfaceSingleScatterAlbedo(PHOSPHOR_SW_CONSTANT_PTR float* table, float roughness, float g, float colour) {
    const float c = colour < 0.0f ? 0.0f : (colour > 1.0f ? 1.0f : colour);
    float rf = 0.0f;
    u32 r0 = 0u, r1 = 0u;
    if (roughness <= subsurfaceTableRoughness(0u)) { r0 = r1 = 0u; }
    else if (roughness >= subsurfaceTableRoughness(kSubsurfaceTableRoughnessCount - 1u)) { r0 = r1 = kSubsurfaceTableRoughnessCount - 1u; }
    else {
        for (u32 i = 0u; i + 1u < kSubsurfaceTableRoughnessCount; ++i) {
            if (roughness <= subsurfaceTableRoughness(i + 1u)) {
                r0 = i; r1 = i + 1u;
                rf = (roughness - subsurfaceTableRoughness(i)) / (subsurfaceTableRoughness(i + 1u) - subsurfaceTableRoughness(i));
                break;
            }
        }
    }
    float gf = 0.0f;
    u32 g0 = 0u, g1 = 0u;
    if (g <= subsurfaceTableG(0u)) { g0 = g1 = 0u; }
    else if (g >= subsurfaceTableG(kSubsurfaceTableGCount - 1u)) { g0 = g1 = kSubsurfaceTableGCount - 1u; }
    else {
        for (u32 i = 0u; i + 1u < kSubsurfaceTableGCount; ++i) {
            if (g <= subsurfaceTableG(i + 1u)) {
                g0 = i; g1 = i + 1u;
                gf = (g - subsurfaceTableG(i)) / (subsurfaceTableG(i + 1u) - subsurfaceTableG(i));
                break;
            }
        }
    }
    const float x = c * float(kSubsurfaceTableColourCount - 1u);
    u32 c0 = u32(x);
    if (c0 >= kSubsurfaceTableColourCount - 1u) c0 = kSubsurfaceTableColourCount - 2u;
    const float cf = x - float(c0);
    float s = 0.0f;
    for (u32 ri = 0u; ri < 2u; ++ri) {
        const u32 r = ri == 0u ? r0 : r1;
        const float rw = ri == 0u ? 1.0f - rf : rf;
        if (rw == 0.0f) continue;
        for (u32 gi = 0u; gi < 2u; ++gi) {
            const u32 gg = gi == 0u ? g0 : g1;
            const float gw = gi == 0u ? 1.0f - gf : gf;
            if (gw == 0.0f) continue;
            const u32 base = (r * kSubsurfaceTableGCount + gg) * kSubsurfaceTableColourCount + c0;
            s += rw * gw * (table[base] * (1.0f - cf) + table[base + 1u] * cf);
        }
    }
    return 1.0f - s * s; // table value = sqrt(1 - albedo)
}

} // namespace phosphor
