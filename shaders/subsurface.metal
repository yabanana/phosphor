// F15.1 (ticket 17): volumetric subsurface, screen-space Burley normalised diffusion (ADR-0005, decision 10).
//
// The resolve (and the indexed overflow fallback) of a scene with volumetric subsurface strips the diffuse term of those pixels from the
// colour and writes it, premultiplied by weight x colour, to the diffuse-irradiance target together with the material index + 1
// (shadeSurface, kSssWrite).  subsurface_diffuse spreads that irradiance with the Christensen-Burley profile
//   R(r) = a (exp(-a r) + exp(-a r / 3)) / (8 pi r),   a = s / l',  s = 1.85 - A + 7 |A - 0.8|^3
// (unit integral per channel; A = the multiple-scattering colour, l' = the mean free path radius x radiusScale, widened by the similarity
// relation sigma_s' = sigma_s (1 - g) for an anisotropic phase function) and adds the result to the colour.  Samples follow a golden-angle
// spiral; their radii are the equal-mass quantiles of the mixture of the three channel profiles (21 samples for a small footprint,
// 55 otherwise), so a channel's weight is P_c / P_mix and bounded.  A sample counts when it is inside the image, has the SAME material and
// its 3D distance (planar radius and depth difference) is finite; its weight carries the surface-area factor (z_n / z)^2 / cos(theta_n).
// The estimator is self-normalised over the accepted samples.  A footprint below half a pixel is skipped: colour + the pixel's own term.
#include <metal_stdlib>
#include "renderer/gpu_types.h"
using namespace metal;
using namespace phosphor;

kernel void subsurface_clear(constant GPUSubsurfaceParams& p [[buffer(0)]], texture2d<float, access::write> sss [[texture(0)]],
                             uint2 pixel [[thread_position_in_grid]]) {
    if (pixel.x >= p.outputWidth || pixel.y >= p.outputHeight) return;
    sss.write(float4(0.0f), pixel);
}

constant float kGoldenAngle = 2.39996323f;

// Burley shape factor of a channel with multiple-scattering albedo A.
static float burleyScale(float A) {
    const float a = saturate(A);
    return 1.85f - a + 7.0f * abs(a - 0.8f) * abs(a - 0.8f) * abs(a - 0.8f);
}
// The HG anisotropy g widens the mean free path through the single-scattering albedo that gives the multiple-scattering albedo A
// (the draft's Kulla-Conty inversion): l' = l / (1 - rho g).
static float anisotropyWiden(float A, float g) {
    if (g == 0.0f) return 1.0f;
    const float a = saturate(A);
    const float s = 4.09712f + 4.20863f * a - sqrt(9.59217f + 41.6808f * a + 17.7126f * a * a);
    const float rho = (1.0f - s * s) / max(1.0f - g * s * s, 1e-3f);
    return 1.0f / max(1.0f - clamp(rho, 0.0f, 1.0f) * g, 0.05f);
}
// Radial CDF of one channel: 1 - exp(-a r) / 4 - 3 exp(-a r / 3) / 4.
static float profileCdf(float a, float r) { return 1.0f - 0.25f * exp(-a * r) - 0.75f * exp(-a * r / 3.0f); }
// Profile (without 1 / (8 pi r)): a (exp(-a r) + exp(-a r / 3)).
static float profileShape(float a, float r) { return a * (exp(-a * r) + exp(-a * r / 3.0f)); }

kernel void subsurface_diffuse(constant GPUSubsurfaceParams& p [[buffer(0)]], constant FrameConstants& frame [[buffer(1)]],
                               const device GPUMaterial* materials [[buffer(2)]],
                               texture2d<float, access::read> color [[texture(0)]], texture2d<float, access::read> sss [[texture(1)]],
                               depth2d<float, access::read> depth [[texture(2)]], texture2d<float, access::read> normalRoughness [[texture(3)]],
                               texture2d<float, access::write> result [[texture(4)]], uint2 pixel [[thread_position_in_grid]]) {
    if (pixel.x >= p.outputWidth || pixel.y >= p.outputHeight) return;
    const float4 base = color.read(pixel);
    const float4 own = sss.read(pixel);
    const uint material = uint(own.w + 0.5f);
    if (material == 0u || pixel.x >= p.width || pixel.y >= p.height) {
        result.write(base, pixel);
        return;
    }
    const device GPUMaterial& m = materials[material - 1u];
    const GPUMaterialExt2 e2 = *reinterpret_cast<const device GPUMaterialExt2*>(frame.materialExtensions + m.extensionIndex + 2u);
    const float d0 = depth.read(pixel);
    const float z0 = p.nearPlane / max(d0, 1e-12f);
    const float mpp = 2.0f * z0 * p.tanHalfFovX / float(p.width); // metres per pixel at the pixel's depth
    float a[3], scatter[3];
    float aMin = 1e30f, dMax = 0.0f;
    for (uint c = 0; c < 3; ++c) {
        const float A = e2.color[c];
        const float mfp = max(e2.radius * e2.radiusScale[c], 1e-6f) * anisotropyWiden(A, e2.anisotropy);
        a[c] = burleyScale(A) / mfp;
        scatter[c] = 1.0f / a[c];
        aMin = min(aMin, a[c]);
        dMax = max(dMax, scatter[c]);
    }
    // Sub-pixel footprint: the diffusion length is below half a pixel: nothing to spread.
    if (dMax < 0.5f * mpp) {
        result.write(float4(base.rgb + own.rgb, base.a), pixel);
        return;
    }
    const uint count = dMax < 2.0f * mpp ? 21u : 55u;
    // Equal-mass radii (metres) of the mixture of the channel profiles, by bisection on the monotone CDF.
    const float rHi = 28.0f / aMin;
    float3 accumulated = float3(0.0f), weights = float3(0.0f);
    const float3x3 viewRotation = float3x3(float3(frame.view[0], frame.view[1], frame.view[2]), float3(frame.view[4], frame.view[5], frame.view[6]),
                                           float3(frame.view[8], frame.view[9], frame.view[10]));
    for (uint i = 0; i < count; ++i) {
        const float u = (float(i) + 0.5f) / float(count);
        float lo = 0.0f, hi = rHi;
        for (uint k = 0; k < 16; ++k) {
            const float mid = 0.5f * (lo + hi);
            const float cdf = (profileCdf(a[0], mid) + profileCdf(a[1], mid) + profileCdf(a[2], mid)) * (1.0f / 3.0f);
            if (cdf < u) lo = mid; else hi = mid;
        }
        const float r = max(0.5f * (lo + hi), 1e-7f);
        const float angle = float(i) * kGoldenAngle;
        const float2 offset = (r / mpp) * float2(cos(angle), sin(angle));
        const int2 target = int2(floor(float2(pixel) + 0.5f + offset));
        if (target.x < 0 || target.y < 0 || target.x >= int(p.width) || target.y >= int(p.height)) continue;
        const uint2 neighbour = uint2(target);
        const float4 value = sss.read(neighbour);
        if (uint(value.w + 0.5f) != material) continue;
        const float dn = depth.read(neighbour);
        if (!(dn > 0.0f)) continue;
        const float zn = p.nearPlane / dn;
        const float dz = zn - z0;
        const float distance3 = sqrt(r * r + dz * dz);
        // View-space ray and normal of the neighbour: the surface area one of its pixels covers.
        const float2 ndc = (float2(neighbour) + 0.5f) / float2(float(p.width), float(p.height)) * 2.0f - 1.0f;
        const float3 ray = normalize(float3(ndc.x * p.tanHalfFovX, -ndc.y * p.tanHalfFovY, -1.0f));
        const float3 normal = viewRotation * normalRoughness.read(neighbour).xyz;
        const float cosine = max(abs(dot(normalize(normal), ray)), 0.05f);
        const float area = (zn / z0) * (zn / z0) / cosine;
        // Weight of channel c: P_c(distance3) area / P_mix(r); 8 pi cancels, the 1 / r factors stay.
        const float mixture = (profileShape(a[0], r) + profileShape(a[1], r) + profileShape(a[2], r)) * (1.0f / 3.0f) / r;
        const float3 shape = float3(profileShape(a[0], distance3), profileShape(a[1], distance3), profileShape(a[2], distance3)) / distance3;
        const float3 weight = shape * (area / max(mixture, 1e-20f));
        accumulated += weight * value.rgb;
        weights += weight;
    }
    float3 spread = own.rgb;
    for (uint c = 0; c < 3; ++c)
        if (weights[c] > 1e-20f) spread[c] = accumulated[c] / weights[c];
    result.write(float4(base.rgb + spread, base.a), pixel);
}
