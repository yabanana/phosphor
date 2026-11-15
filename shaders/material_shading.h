#pragma once
#include <metal_stdlib>
#include "renderer/gpu_types.h"
#include "renderer/lobe_guide_pack.h"
#include "renderer/layer_blend.h"
#include "pipeline/forward_variants.generated.metal.h"
using namespace metal;
using namespace phosphor;
#include "surface_geometry.h"
#include "materials/bsdf.h"

constant bool kHasDirectional = (kLightTypes & 1u) != 0u;
constant bool kHasPoint = (kLightTypes & 2u) != 0u;
constant bool kHasSpot = (kLightTypes & 4u) != 0u;
// 0..3: normal-map/emissive feature bits, 4: generic reference.
constant uint FC_RESOLVE_CLASS [[function_constant(21)]];
constant uint kResolveClass = is_function_constant_defined(FC_RESOLVE_CLASS) ? FC_RESOLVE_CLASS : 4u;
// F15.1 (ticket 17): the volumetric-subsurface resolve (reserved constant SSS_OUTPUT, variants.def): the diffuse irradiance goes to its own target.
constant bool kSssWrite = kSssOutput != 0u;
struct TextureHandle {
    texture2d<float> tex;
};
struct SurfaceInput {
    float3 worldPos;
    float3 normal;
    float4 tangent;
    float2 uv;
    float2 uvDx;
    float2 uvDy;
    uint materialIndex;
    uint mirrored;
    bool frontFacing;
    // F15.1 (ticket 13): the instance scale along its thinnest axis (min column norm of the model matrix): KHR_materials_volume's
    // thicknessFactor is mesh-local, the lobe guide and the declared --rt off model carry the world thickness.  1 where unknown.
    float volumeScale = 1.0f;
};
struct ShadingResult {
    // F15.1 (ticket 17): the diffuse irradiance of a volumetric-subsurface pixel (premultiplied by weight x colour, see shadeSurface) and its
    // material index + 1 (0 = none): written to the diffuse-irradiance target when kSssOutput; the colour then excludes that diffuse term.
    float3 sssIrradiance;
    uint sssMaterial;
    float3 color;
    float3 normal;
    float3 baseColor;
    float3 diffuseAlbedo;
    float3 specularAlbedo;
    float roughness;
    float metallic;
    float occlusion;
    float alpha;
    float3 emission; // the emissive term already included in color (F13.1)
    // F15.1 (ticket 06): the resolved feature parameters the lobe guide stores (renderer/lobe_guide_pack.h);
    // zero for base materials and whenever kMaterialFeatures has no resolved feature.  One source for the
    // visibility resolve, the guide kernel and the indexed receivers.
    LobeParams lobes;
};

// --- BRDF --------------------------------------------------------------------

constexpr sampler kMaterialSampler(filter::linear, mip_filter::linear, address::repeat, max_anisotropy(8));

static half4 sampleOr(const device TextureHandle *table, uint index, float2 uv, half4 fallback, float2 dx, float2 dy) {
    if (index == INVALID_TEXTURE_INDEX) {
        return fallback;
    }
    return half4(table[index].tex.sample(kMaterialSampler, uv, gradient2d(dx, dy)));
}

#include "texture_transform.h"

static float3 evaluateLight(float3 N, float3 V, float3 L, float3 radiance, float3 albedo, float metallic,
                            float roughness) {
    const float3 H = normalize(V + L);
    const float NdotL = saturate(dot(N, L));
    const float NdotV = max(dot(N, V), 1e-4);
    const float NdotH = saturate(dot(N, H));
    const float VdotH = saturate(dot(V, H));
    if (NdotL <= 0.0) {
        return float3(0.0);
    }

    const BsdfBaseLobes lobes =
        bsdfBaseEval(NdotL, NdotV, NdotH, VdotH, albedo, metallic, bsdfAlpha(roughness), 1.0f);
    return (lobes.diffuse + lobes.specular) * radiance * NdotL;
}

// F13: the sun is a disk of uniform radiance and angular radius r (the same
// disk shadows, reflections and the reference use). Returns the irradiance-
// normalised specular term, (1/Omega) * integral over the disk of f_s cos.
// Where the GGX lobe is wide or far from the reflected disk, the point
// evaluation already equals the integral; elsewhere a deterministic uniform
// solid-angle quadrature (golden-angle spiral) is used at the pixel centre.
static float3 sunDiskSpecular(float3 N, float3 V, float3 L, float3 albedo, float metallic, float roughness,
                              float angularRadius) {
    const float a = bsdfAlpha(roughness);
    const float NdotV = max(dot(N, V), 1e-4);
    const float3 f0 = bsdfBaseF0(albedo, metallic);
    const float3 R = reflect(-V, N);
    const float separation = acos(clamp(dot(R, L), -1.0f, 1.0f));
    const bool narrow = a < 0.05f && separation < 4.0f * angularRadius + 16.0f * a;
    const uint count = !narrow ? 1u : (a < 2.0f * angularRadius ? 128u : 32u);
    const float3 t = normalize(cross(abs(L.z) < 0.999f ? float3(0, 0, 1) : float3(0, 1, 0), L)), b = cross(L, t);
    const float oneMinusCos = 1.0f - cos(angularRadius);
    float3 sum = float3(0.0);
    for (uint k = 0; k < count; ++k) {
        float3 Lk = L;
        if (count > 1u) {
            const float z = 1.0f - (float(k) + 0.5f) / float(count) * oneMinusCos, s = sqrt(max(0.0f, 1.0f - z * z));
            const float phi = 2.39996323f * float(k);
            Lk = t * (s * cos(phi)) + b * (s * sin(phi)) + L * z;
        }
        const float NdotL = saturate(dot(N, Lk));
        if (NdotL <= 0.0f) continue;
        const float3 H = normalize(V + Lk);
        sum += bsdfBaseSpecular(NdotL, NdotV, saturate(dot(N, H)), saturate(dot(V, H)), f0, a) * NdotL;
    }
    return sum / float(count);
}

// F15.1: the same two functions for a material with KHR_materials_ior/specular (bsdf.h, section
// "KHR_materials_ior + KHR_materials_specular"); the base versions above stay untouched.
static float3 evaluateLightSpec(float3 N, float3 V, float3 L, float3 radiance, float3 albedo, float metallic,
                                float roughness, BsdfDielectricSpec spec, BsdfIrid irid = bsdfIridNone(), float diffuseScale = 1.0f) {
    const float3 H = normalize(V + L);
    const float NdotL = saturate(dot(N, L));
    const float NdotV = max(dot(N, V), 1e-4);
    const float NdotH = saturate(dot(N, H));
    const float VdotH = saturate(dot(V, H));
    if (NdotL <= 0.0) {
        return float3(0.0);
    }
    // F15.1 (ticket 12): with a film the Fresnel is evaluated at this sample's V.H (bsdfSpecEvalIrid; no film = bsdfSpecEval's expression).
    const float a = bsdfAlpha(roughness);
    const BsdfBaseLobes lobes = bsdfSpecEvalIrid(spec, irid, bsdfDGGX(NdotH, a) * bsdfVSmithGGXCorrelated(NdotV, NdotL, a), VdotH, albedo, metallic);
    return (lobes.diffuse * diffuseScale + lobes.specular) * radiance * NdotL; // ticket 13: the diffuse base of a transmissive material is mixed by 1 - t
}

static float3 sunDiskSpecularSpec(float3 N, float3 V, float3 L, float3 albedo, float metallic, float roughness,
                                  float angularRadius, BsdfDielectricSpec spec, BsdfIrid irid = bsdfIridNone()) {
    const float a = bsdfAlpha(roughness);
    const float NdotV = max(dot(N, V), 1e-4);
    const float3 R = reflect(-V, N);
    const float separation = acos(clamp(dot(R, L), -1.0f, 1.0f));
    const bool narrow = a < 0.05f && separation < 4.0f * angularRadius + 16.0f * a;
    const uint count = !narrow ? 1u : (a < 2.0f * angularRadius ? 128u : 32u);
    const float3 t = normalize(cross(abs(L.z) < 0.999f ? float3(0, 0, 1) : float3(0, 1, 0), L)), b = cross(L, t);
    const float oneMinusCos = 1.0f - cos(angularRadius);
    float3 sum = float3(0.0);
    for (uint k = 0; k < count; ++k) {
        float3 Lk = L;
        if (count > 1u) {
            const float z = 1.0f - (float(k) + 0.5f) / float(count) * oneMinusCos, s = sqrt(max(0.0f, 1.0f - z * z));
            const float phi = 2.39996323f * float(k);
            Lk = t * (s * cos(phi)) + b * (s * sin(phi)) + L * z;
        }
        const float NdotL = saturate(dot(N, Lk));
        if (NdotL <= 0.0f) continue;
        const float3 H = normalize(V + Lk);
        const float NdotH = saturate(dot(N, H)), VdotH = saturate(dot(V, H));
        sum += bsdfDGGX(NdotH, a) * bsdfVSmithGGXCorrelated(NdotV, NdotL, a) *
               bsdfSpecColorIrid(spec, irid, bsdfSpecFresnel(spec, VdotH), albedo, metallic, VdotH) * NdotL;
    }
    return sum / float(count);
}

// F15.1 (ticket 16): subsurface thin mode (bsdf.h).  A light on the FRONT hemisphere sees the reflected lobes with the diffuse scaled
// by 1 - w; a light on the BACK hemisphere lights the visible face through the transmitted lobe w (1 - F(V.H)) color / pi |N.L|.
// `spec` is the dielectric of the extended path (the default f0 0.04, f90 1 for a thin-only material).
static float3 evaluateLightThin(float3 N, float3 V, float3 L, float3 radiance, float3 albedo, float metallic, float roughness,
                                BsdfDielectricSpec spec, BsdfThin thin) {
    const float3 H = normalize(V + L);
    const float NdotLs = dot(N, L);
    const float NdotV = max(dot(N, V), 1e-4);
    if (NdotLs > 0.0f) {
        const float NdotL = saturate(NdotLs);
        const float NdotH = saturate(dot(N, H));
        const float VdotH = saturate(dot(V, H));
        const float a = bsdfAlpha(roughness);
        const BsdfBaseLobes lobes = bsdfSpecEval(spec, NdotL, NdotV, NdotH, VdotH, albedo, metallic, a);
        return (lobes.diffuse * bsdfThinReflectScale(thin) + lobes.specular) * radiance * NdotL;
    }
    if (NdotLs < 0.0f && dot(V + L, V + L) > 1e-20f)
        return bsdfThinTransmitted(thin, saturate(dot(V, H)), metallic) * radiance * (-NdotLs);
    return float3(0.0);
}
// The sun of a thin material: reflected and transmitted terms at the disk centre (the diffuse rule), the specular lobe over the disk.
static float3 sunThinDiffuse(float3 N, float3 V, float3 L, float3 albedo, float metallic, BsdfDielectricSpec spec, BsdfThin thin) {
    const float NdotLs = dot(N, L);
    if (!(dot(V + L, V + L) > 1e-20f)) return float3(0.0);
    const float VdotH = saturate(dot(V, normalize(V + L)));
    if (NdotLs > 0.0f) return bsdfSpecDiffuse(bsdfSpecFresnel(spec, VdotH), metallic, albedo) * bsdfThinReflectScale(thin) * NdotLs;
    if (NdotLs < 0.0f) return bsdfThinTransmitted(thin, VdotH, metallic) * (-NdotLs);
    return float3(0.0);
}

// F15.1 (ticket 11): KHR_materials_anisotropy.  The ior/specular dielectric (or the metal) with the anisotropic D V of bsdf.h in place
// of the isotropic GGX; the same two functions as evaluateLightSpec / sunDiskSpecularSpec otherwise.
static float3 evaluateLightAniso(float3 N, float3 V, float3 L, float3 radiance, float3 albedo, float metallic,
                                 BsdfDielectricSpec spec, BsdfAniso an, BsdfIrid irid = bsdfIridNone(), float diffuseScale = 1.0f) {
    const float3 H = normalize(V + L);
    const float NdotL = saturate(dot(N, L));
    const float NdotV = max(dot(N, V), 1e-4);
    const float NdotH = saturate(dot(N, H));
    const float VdotH = saturate(dot(V, H));
    if (NdotL <= 0.0) {
        return float3(0.0);
    }
    const BsdfBaseLobes lobes = bsdfSpecEvalIrid(spec, irid, bsdfAnisoDV(an, V, L, H, NdotL, NdotV, NdotH), VdotH, albedo, metallic);
    return (lobes.diffuse * diffuseScale + lobes.specular) * radiance * NdotL;
}
// Sun disk of an anisotropic lobe: the disk rule keys on the NARROWEST axis alpha_b = alpha and on the reflection axis of the
// isotropic lobe about N (the anisotropic lobe is only wider along T'): alpha >= 0.05 evaluates the lobe at the disk centre (the
// reference's rule); a narrower lobe near the reflection axis uses the same golden-angle quadrature as the isotropic function.
static float3 sunDiskSpecularAniso(float3 N, float3 V, float3 L, float3 albedo, float metallic, float angularRadius,
                                   BsdfDielectricSpec spec, BsdfAniso an, BsdfIrid irid = bsdfIridNone()) {
    const float a = an.ab;
    const float NdotV = max(dot(N, V), 1e-4);
    const float3 R = reflect(-V, N);
    const float separation = acos(clamp(dot(R, L), -1.0f, 1.0f));
    const bool narrow = a < 0.05f && separation < 4.0f * angularRadius + 16.0f * a;
    const uint count = !narrow ? 1u : (a < 2.0f * angularRadius ? 128u : 32u);
    const float3 t = normalize(cross(abs(L.z) < 0.999f ? float3(0, 0, 1) : float3(0, 1, 0), L)), b = cross(L, t);
    const float oneMinusCos = 1.0f - cos(angularRadius);
    float3 sum = float3(0.0);
    for (uint k = 0; k < count; ++k) {
        float3 Lk = L;
        if (count > 1u) {
            const float z = 1.0f - (float(k) + 0.5f) / float(count) * oneMinusCos, s = sqrt(max(0.0f, 1.0f - z * z));
            const float phi = 2.39996323f * float(k);
            Lk = t * (s * cos(phi)) + b * (s * sin(phi)) + L * z;
        }
        const float NdotL = saturate(dot(N, Lk));
        if (NdotL <= 0.0f) continue;
        const float3 H = normalize(V + Lk);
        const float NdotH = saturate(dot(N, H)), VdotH = saturate(dot(V, H));
        sum += bsdfAnisoDV(an, V, Lk, H, NdotL, NdotV, NdotH) *
               bsdfSpecColorIrid(spec, irid, bsdfSpecFresnel(spec, VdotH), albedo, metallic, VdotH) * NdotL;
    }
    return sum / float(count);
}

// F15.1 (ticket 09): KHR_materials_clearcoat.  The coat lobe c F_c D Vis (bsdf.h) over the layered material: the
// material terms carry the pixel-wide attenuation 1 - c F_c(|V.Nc|), the coat term its own cosine max(Nc.L, 0).
struct CoatParams {
    float factor;    // c
    float roughness; // clamped [0.04, 1]
    float3 normal;   // Nc
};
inline float coatAttenuation(CoatParams coat, float3 V) {
    return 1.0f - coat.factor * bsdfCoatFresnel(abs(dot(coat.normal, V)));
}
static float3 evaluateLightCoat(float3 N, float3 V, float3 L, float3 radiance, float3 albedo, float metallic, float roughness,
                                BsdfDielectricSpec spec, CoatParams coat, BsdfIrid irid = bsdfIridNone(), float diffuseScale = 1.0f) {
    float3 result = evaluateLightSpec(N, V, L, radiance, albedo, metallic, roughness, spec, irid, diffuseScale) * coatAttenuation(coat, V);
    const float ncl = saturate(dot(coat.normal, L));
    if (ncl > 0.0f) {
        const float3 H = normalize(V + L);
        const float ncv = max(dot(coat.normal, V), 1e-4f);
        const float fc = bsdfCoatFresnel(abs(dot(coat.normal, V)));
        result += (coat.factor * fc * bsdfCoatSpecular(ncl, ncv, saturate(dot(coat.normal, H)), bsdfAlpha(coat.roughness)) * ncl) * radiance;
    }
    return result;
}
// Irradiance-normalised coat term of the sun disk: the declared disk rule of sunDiskSpecular with the coat's own alpha
// and reflection axis (about Nc), (1/Omega) * integral of c F_c D Vis max(Nc.L, 0).
static float sunDiskCoat(float3 V, float3 L, float angularRadius, CoatParams coat) {
    const float a = bsdfAlpha(coat.roughness);
    const float3 Nc = coat.normal;
    const float ncv = max(dot(Nc, V), 1e-4f);
    const float3 R = reflect(-V, Nc);
    const float separation = acos(clamp(dot(R, L), -1.0f, 1.0f));
    const bool narrow = a < 0.05f && separation < 4.0f * angularRadius + 16.0f * a;
    const uint count = !narrow ? 1u : (a < 2.0f * angularRadius ? 128u : 32u);
    const float3 t = normalize(cross(abs(L.z) < 0.999f ? float3(0, 0, 1) : float3(0, 1, 0), L)), b = cross(L, t);
    const float oneMinusCos = 1.0f - cos(angularRadius);
    const float weight = coat.factor * bsdfCoatFresnel(abs(dot(Nc, V)));
    float sum = 0.0f;
    for (uint k = 0; k < count; ++k) {
        float3 Lk = L;
        if (count > 1u) {
            const float z = 1.0f - (float(k) + 0.5f) / float(count) * oneMinusCos, s = sqrt(max(0.0f, 1.0f - z * z));
            const float phi = 2.39996323f * float(k);
            Lk = t * (s * cos(phi)) + b * (s * sin(phi)) + L * z;
        }
        const float ncl = saturate(dot(Nc, Lk));
        if (ncl <= 0.0f) continue;
        const float3 H = normalize(V + Lk);
        sum += weight * bsdfCoatSpecular(ncl, ncv, saturate(dot(Nc, H)), a) * ncl;
    }
    return sum / float(count);
}

// F15.1 (ticket 13): the thin-walled transmitted term of the sun disk (a light behind a transmissive surface), irradiance-normalised:
// (1 / Omega) * integral over the disk of f_t |N.L| (bsdfThinGlassTransmitted without the radiance).  The lobe peaks at L = -V (straight
// through): the disk rule of sunDiskSpecular with that axis.
static float3 sunDiskThinGlass(float3 N, float3 V, float3 L, float3 tint, float metallic, float t, float roughness, float angularRadius,
                               BsdfDielectricSpec spec) {
    const float a = bsdfAlpha(roughness);
    const float separation = acos(clamp(dot(-V, L), -1.0f, 1.0f));
    const bool narrow = a < 0.05f && separation < 4.0f * angularRadius + 16.0f * a;
    const uint count = !narrow ? 1u : (a < 2.0f * angularRadius ? 128u : 32u);
    const float3 tt = normalize(cross(abs(L.z) < 0.999f ? float3(0, 0, 1) : float3(0, 1, 0), L)), b = cross(L, tt);
    const float oneMinusCos = 1.0f - cos(angularRadius);
    float3 sum = float3(0.0);
    for (uint k = 0; k < count; ++k) {
        float3 Lk = L;
        if (count > 1u) {
            const float z = 1.0f - (float(k) + 0.5f) / float(count) * oneMinusCos, s = sqrt(max(0.0f, 1.0f - z * z));
            const float phi = 2.39996323f * float(k);
            Lk = tt * (s * cos(phi)) + b * (s * sin(phi)) + L * z;
        }
        sum += bsdfThinGlassTransmitted(spec, t, metallic, tint, N, V, Lk, a);
    }
    return sum / float(count);
}

// F15.1 (ticket 10): KHR_materials_sheen.  The sheen layer sits between the material and the coat: f = S D V + f_material scaling
// (bsdf.h).  `value` is the material term for one light, irradiance-normalised (f cos, without the radiance): the analytic lights
// and the sun disk centre use the min form of the albedo scaling, the sheen lobe is point-evaluated (the lobe is wide against the sun).
struct SheenParams {
    float3 color;    // S
    float roughness; // clamped [0.07, 1]
};
static float3 sheenLayerLight(float3 value, float3 N, float3 V, float3 L, SheenParams sheen, const constant float* table) {
    const float NdotL = saturate(dot(N, L));
    if (NdotL <= 0.0f) return float3(0.0);
    const float3 H = normalize(V + L);
    const float NdotV = max(dot(N, V), 1e-4f), NdotH = saturate(dot(N, H));
    const float scale = bsdfSheenScaleLight(sheen.color, sheen.roughness, NdotV, NdotL, table);
    return value * scale + bsdfSheenLobe(sheen.color, sheen.roughness, NdotL, NdotV, NdotH) * NdotL;
}
// The coat lobe alone for one light, irradiance-normalised: c F_c D Vis max(Nc.L, 0) (the same expression as evaluateLightCoat).
static float coatLobeLight(float3 V, float3 L, CoatParams coat) {
    const float ncl = saturate(dot(coat.normal, L));
    if (!(ncl > 0.0f)) return 0.0f;
    const float3 H = normalize(V + L);
    const float ncv = max(dot(coat.normal, V), 1e-4f);
    const float fc = bsdfCoatFresnel(abs(dot(coat.normal, V)));
    return coat.factor * fc * bsdfCoatSpecular(ncl, ncv, saturate(dot(coat.normal, H)), bsdfAlpha(coat.roughness)) * ncl;
}

static float distanceAttenuation(float dist, float range) {
    // Inverse square with a smooth window to zero at `range`.
    const float r = dist / max(range, 1e-3);
    const float window = saturate(1.0 - r * r * r * r);
    return window * window / max(dist * dist, 1e-4);
}

// Hemispheric sky/ground gradient, the only ambient light until the GI of F12.
static float3 hemisphere(float3 dir) {
    const float3 sky = float3(0.30, 0.36, 0.45);
    const float3 ground = float3(0.10, 0.09, 0.08);
    return mix(ground, sky, dir.y * 0.5 + 0.5);
}

// Filmic tonemap (Narkowicz ACES fit); the swapchain format is sRGB so the
// hardware applies the transfer function on write.
static float3 tonemapACES(float3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

// F15.4 (ticket 23): the layer stack (record 3 of the material's extension group).  The mask is sampled at the primitive's uv; every
// layer with a non-zero weight is an ordinary material of the library whose base parameters are sampled at uv * uvScale with the
// same expressions as the single-material path (so a weight vector e_j reproduces layer j exactly), blended in parameter space
// (renderer/layer_blend.h) and shaded ONCE by the base model.  The layer materials' feature blocks, emissive and alpha are not
// blended: they belong to the stack's own description (the coat finish).
struct LayerSurface {
    float3 baseColor;
    float roughness, metallic, occlusion;
    float3 tn; // the tangent-space normal vector for normalize(T a + B b + N c)
};
static LayerSurface resolveLayerStack(GPUMaterialExt3 e3, const device GPUMaterial *materials, const device TextureHandle *textures,
                                      float2 uv, float2 dx, float2 dy) {
    const uint count = min(e3.layerCount, LAYER_STACK_MAX);
    const float4 mask = float4(sampleOr(textures, e3.maskTex, uv, half4(1.0h), dx, dy));
    LayerWeights raw{{0.0f, 0.0f, 0.0f, 0.0f}};
    bool heightOn = false;
    for (uint i = 0; i < LAYER_STACK_MAX; ++i) {
        if (i < count) {
            raw.w[i] = mask[(e3.packedMaskChannels >> (2u * i)) & 3u];
            heightOn = heightOn || e3.heightContrast[i] >= 0.0f;
        }
    }
    LayerWeights w = layerNormalise(raw, count);
    if (heightOn) {
        LayerHeights heights{{0.0f, 0.0f, 0.0f, 0.0f}};
        for (uint i = 0; i < LAYER_STACK_MAX; ++i) {
            if (i < count && w.w[i] > 0.0f && e3.heightContrast[i] >= 0.0f) {
                const float s = e3.uvScale[i];
                const float texel = float(sampleOr(textures, e3.heightTex[i], uv * s, half4(0.5h), dx * s, dy * s).r);
                heights.h[i] = layerHeight(e3.heightContrast[i], texel);
            }
        }
        w = layerHeightBlend(w, heights, count);
    }
    float3 color[4] = {float3(0.0f), float3(0.0f), float3(0.0f), float3(0.0f)};
    float rough[4] = {0.0f, 0.0f, 0.0f, 0.0f}, metal[4] = {0.0f, 0.0f, 0.0f, 0.0f}, occ[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    LayerNormals normals{{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f, 1.0f}};
    for (uint i = 0; i < LAYER_STACK_MAX; ++i) {
        if (i < count && w.w[i] > 0.0f) {
            const device GPUMaterial &lm = materials[e3.layerMaterial[i]];
            const float s = e3.uvScale[i];
            const float2 uvi = uv * s, dxi = dx * s, dyi = dy * s;
            const half4 baseTex = sampleOr(textures, lm.baseColorTex, uvi, half4(1.0h), dxi, dyi);
            const half4 mrTex = sampleOr(textures, lm.metallicRoughnessTex, uvi, half4(1.0h), dxi, dyi);
            const half4 aoTex = sampleOr(textures, lm.occlusionTex, uvi, half4(1.0h), dxi, dyi);
            const half4 nTex = sampleOr(textures, lm.normalTex, uvi, half4(0.5h, 0.5h, 1.0h, 1.0h), dxi, dyi);
            color[i] = float3(lm.baseColor[0], lm.baseColor[1], lm.baseColor[2]) * float3(baseTex.rgb);
            rough[i] = clamp(lm.roughness * float(mrTex.g), 0.04, 1.0);
            metal[i] = saturate(lm.metallic * float(mrTex.b));
            occ[i] = mix(1.0, float(aoTex.r), lm.occlusionStrength);
            float3 tn = float3(nTex.xyz) * 2.0 - 1.0;
            tn.xy *= lm.normalScale;
            normals.tx[i] = tn.x;
            normals.ty[i] = tn.y;
            normals.tz[i] = tn.z;
        }
    }
    LayerSurface out;
    out.baseColor = float3(layerBlendScalar(w, color[0].x, color[1].x, color[2].x, color[3].x),
                           layerBlendScalar(w, color[0].y, color[1].y, color[2].y, color[3].y),
                           layerBlendScalar(w, color[0].z, color[1].z, color[2].z, color[3].z));
    out.roughness = layerBlendScalar(w, rough[0], rough[1], rough[2], rough[3]);
    out.metallic = layerBlendScalar(w, metal[0], metal[1], metal[2], metal[3]);
    out.occlusion = layerBlendScalar(w, occ[0], occ[1], occ[2], occ[3]);
    const LayerNormalResult n = layerNormalBlend(w, normals);
    out.tn = float3(n.a, n.b, n.c);
    return out;
}

static ShadingResult shadeSurface(SurfaceInput in, constant FrameConstants &frame, const device GPUMaterial *materials,
                                  const device GPULight *lights, const device TextureHandle *textures,
                                  bool alphaTest = true, float3 sunVisibility = float3(1.0f), uint shadowLight = ~0u,
                                  bool replaceLocal = false, float3 localDirect = float3(0),
                                  bool useGI = false, float3 indirectIrradiance = float3(0),
                                  bool omitSpecularAmbient = false, bool physicalGI = false, bool externalDiffuse = false,
                                  float3 environment = float3(0), float sunAngularRadius = 0.0f) {
    const device GPUMaterial &m = materials[in.materialIndex];
    // F15.1 (ticket 22): the per-slot texture transforms of the material (compiled out without kTextureTransform).
    const TexXf xf = texXfFor(frame.materialExtensions, m.flags, m.textureTransformIndex);

    const half4 baseTex = sampleSlot(textures, m.baseColorTex, TT_SLOT_BASE_COLOR, xf, in.uv, half4(1.0h), in.uvDx, in.uvDy);
    const half4 mrTex = sampleSlot(textures, m.metallicRoughnessTex, TT_SLOT_METAL_ROUGH, xf, in.uv, half4(1.0h), in.uvDx, in.uvDy);
    const half4 aoTex = sampleSlot(textures, m.occlusionTex, TT_SLOT_OCCLUSION, xf, in.uv, half4(1.0h), in.uvDx, in.uvDy);
    // EMISSIVE == false: no material of the scene emits, so the texture fetch
    // is skipped.  The multiply-add below stays unconditional on purpose: a
    // branch around it changes the compiler's contraction of the generic
    // pipeline by 1 LSB in some pixels (measured), and adding factor(0) * 1
    // leaves the result exact.
    const half4 emTex = (kEmissive && (kResolveClass == 4u || (kResolveClass & 2u) != 0u))
                            ? sampleSlot(textures, m.emissiveTex, TT_SLOT_EMISSIVE, xf, in.uv, half4(1.0h), in.uvDx, in.uvDy)
                            : half4(1.0h);
    const half4 nTex =
        (kResolveClass == 4u || (kResolveClass & 1u) != 0u)
            ? sampleSlot(textures, m.normalTex, TT_SLOT_NORMAL, xf, in.uv, half4(0.5h, 0.5h, 1.0h, 1.0h), in.uvDx, in.uvDy)
            : (m.normalTex == INVALID_TEXTURE_INDEX ? half4(0.5h, 0.5h, 1.0h, 1.0h)
                                                    : half4(half(128.0f / 255.0f), half(128.0f / 255.0f), 1.0h, 1.0h));

    float4 baseColor = float4(m.baseColor[0], m.baseColor[1], m.baseColor[2], m.baseColor[3]) * float4(baseTex);
    ShadingResult out{};
    out.alpha = baseColor.a;
    if (alphaTest && baseColor.a < m.alphaCutoff)
        return out;

    // glTF: G = roughness, B = metallic.
    float roughness = clamp(m.roughness * float(mrTex.g), 0.04, 1.0);
    float metallic = saturate(m.metallic * float(mrTex.b));
    float occlusion = mix(1.0, float(aoTex.r), m.occlusionStrength);
    // F15.4 (ticket 23): the layer stack replaces the base parameters and the normal map of the pixel (compiled only when the scene's
    // feature union has the bit, run only for materials that carry it).
    const bool layersOn = (kMaterialFeatures & MATERIAL_FEATURE_LAYERS) != 0u &&
                          (m.flags & (MATERIAL_FEATURE_LAYERS << MATERIAL_FEATURE_SHIFT)) != 0u;
    float3 layerTn = float3(0.0f, 0.0f, 1.0f);
    if (layersOn) {
        const LayerSurface ls = resolveLayerStack(*reinterpret_cast<const device GPUMaterialExt3*>(frame.materialExtensions + m.extensionIndex + 3u),
                                                  materials, textures, in.uv, in.uvDx, in.uvDy);
        baseColor.rgb = ls.baseColor;
        roughness = ls.roughness;
        metallic = ls.metallic;
        occlusion = ls.occlusion;
        layerTn = ls.tn;
    }

    // F15.1: KHR_materials_ior + KHR_materials_specular.  Compiled only when the scene's feature
    // union has the bit (kMaterialFeatures, undefined => 0) and run only for materials that carry it.
    const bool specExt = (kMaterialFeatures & MATERIAL_FEATURE_SPECULAR_IOR) != 0u &&
                         (m.flags & (MATERIAL_FEATURE_SPECULAR_IOR << MATERIAL_FEATURE_SHIFT)) != 0u;
    BsdfDielectricSpec spec;
    spec.f0 = float3(0.04f);
    spec.f90 = 1.0f;
    float lobeIor = LobeNeutralIor, lobeSpecularWeight = 1.0f;
    float3 lobeSpecularColor = float3(1.0f);
    // F15.1 (ticket 09): KHR_materials_clearcoat, same gating.  A coated material without the ior/specular block
    // keeps the default dielectric (f0 0.04, f90 1) through the extended path.
    const bool coatOn = (kMaterialFeatures & MATERIAL_FEATURE_CLEARCOAT) != 0u &&
                        (m.flags & (MATERIAL_FEATURE_CLEARCOAT << MATERIAL_FEATURE_SHIFT)) != 0u;
    // F15.1 (ticket 11): KHR_materials_anisotropy, same gating.  An anisotropic material without the ior/specular block keeps the default
    // dielectric (f0 0.04, f90 1) through the extended path; anisoActive below says whether the lobe is resolved for this pixel.
    const bool anisoOn = (kMaterialFeatures & MATERIAL_FEATURE_ANISOTROPY) != 0u &&
                         (m.flags & (MATERIAL_FEATURE_ANISOTROPY << MATERIAL_FEATURE_SHIFT)) != 0u;
    // F15.1 (ticket 12): KHR_materials_iridescence, same gating.  An iridescent material without the ior/specular block keeps the default
    // dielectric (f0 0.04, f90 1) through the extended path; the film replaces the Fresnel of the specular lobe (bsdf.h).
    const bool iridOn = (kMaterialFeatures & MATERIAL_FEATURE_IRIDESCENCE) != 0u &&
                        (m.flags & (MATERIAL_FEATURE_IRIDESCENCE << MATERIAL_FEATURE_SHIFT)) != 0u;
    // F15.1 (ticket 13): KHR_materials_transmission (+ volume), same gating.  A transmissive material without the ior/specular block keeps the
    // default dielectric (ior 1.5: f0 0.04, f90 1) through the extended path; the diffuse base is mixed by 1 - t, the volume decides
    // thin-walled (thickness 0) or refracting (thickness > 0: RT traces it, probes/environment declare it).
    const bool transOn = (kMaterialFeatures & MATERIAL_FEATURE_TRANSMISSION) != 0u &&
                         (m.flags & (MATERIAL_FEATURE_TRANSMISSION << MATERIAL_FEATURE_SHIFT)) != 0u;
    const bool volOn = transOn && (kMaterialFeatures & MATERIAL_FEATURE_VOLUME) != 0u &&
                       (m.flags & (MATERIAL_FEATURE_VOLUME << MATERIAL_FEATURE_SHIFT)) != 0u;
    const bool useSpec = specExt || coatOn || anisoOn || iridOn || transOn;
    // F15.1 (ticket 10): KHR_materials_sheen, same gating.  A sheen-only material keeps the base dielectric path below.
    const bool sheenOn = (kMaterialFeatures & MATERIAL_FEATURE_SHEEN) != 0u &&
                         (m.flags & (MATERIAL_FEATURE_SHEEN << MATERIAL_FEATURE_SHIFT)) != 0u;
    GPUMaterialExt0 e0{};
    if (specExt || coatOn || sheenOn) e0 = *reinterpret_cast<const device GPUMaterialExt0*>(frame.materialExtensions + m.extensionIndex);
    GPUMaterialExt1 e1{};
    if (anisoOn || iridOn || transOn) e1 = *reinterpret_cast<const device GPUMaterialExt1*>(frame.materialExtensions + m.extensionIndex + 1u);
    // t = saturate(factor x texture.r); the volume thickness (mesh-local x the instance scale = world) and the Beer-Lambert medium.
    float transFactor = 0.0f, volThickness = 0.0f, volAttenuationDistance = INFINITY;
    float3 volAttenuationColor = float3(1.0f);
    if (transOn) {
        transFactor = saturate(e1.transmissionFactor * float(sampleSlot(textures, e1.transmissionTex, TT_SLOT_TRANSMISSION, xf, in.uv, half4(1.0h), in.uvDx, in.uvDy).r));
        if (volOn) {
            volThickness = e1.volumeThickness * float(sampleSlot(textures, e1.volumeThicknessTex, TT_SLOT_VOLUME_THICKNESS, xf, in.uv, half4(1.0h), in.uvDx, in.uvDy).g) * in.volumeScale;
            volAttenuationColor = float3(e1.attenuationColor[0], e1.attenuationColor[1], e1.attenuationColor[2]);
            volAttenuationDistance = e1.attenuationDistance;
        }
    }
    const float diffuseT = 1.0f - transFactor; // the diffuse base weight of the transmissive mix
    // Thickness = mix(min, max, texture.g) (the maximum without a texture: min (1 - t) + max t is exact at t = 1); thickness 0 disables the
    // film (strength 0 = the base model, the orchestrator's interpretation of the unspecified d = 0).
    BsdfIrid irid = bsdfIridNone();
    if (iridOn) {
        const float t = float(sampleSlot(textures, e1.iridescenceThicknessTex, TT_SLOT_IRIDESCENCE_THICKNESS, xf, in.uv, half4(1.0h), in.uvDx, in.uvDy).g);
        irid.thickness = e1.iridescenceThicknessMin * (1.0f - t) + e1.iridescenceThicknessMax * t;
        irid.ior = e1.iridescenceIor;
        irid.strength = irid.thickness > 0.0f
                            ? saturate(e1.iridescenceFactor * float(sampleSlot(textures, e1.iridescenceFactorTex, TT_SLOT_IRIDESCENCE, xf, in.uv, half4(1.0h), in.uvDx, in.uvDy).r))
                            : 0.0f;
    }
    // F15.1 (ticket 16/17): subsurface (record 2), same gating.  Mode 0 = thin; mode 1 = volumetric: its weight, colour and profile
    // parameters reach the lobe guide (the thin consumers ignore the volumetric flag) and, when the resolve writes the diffuse-irradiance
    // target (kSssWrite, resolve lighting only), the diffuse term of the pixel is replaced by its screen-space diffusion
    // (subsurface.metal).  weight = saturate(w x texture.a), colour = color x texture.rgb.
    BsdfThin thin = bsdfThinNone();
    const bool thinOn = (kMaterialFeatures & MATERIAL_FEATURE_SUBSURFACE) != 0u &&
                        (m.flags & (MATERIAL_FEATURE_SUBSURFACE << MATERIAL_FEATURE_SHIFT)) != 0u;
    bool thinResolved = false, volResolved = false;
    GPUMaterialExt2 e2{};
    if (thinOn) {
        e2 = *reinterpret_cast<const device GPUMaterialExt2*>(frame.materialExtensions + m.extensionIndex + 2u);
        if (e2.mode == 0u || e2.mode == 1u) {
            thin.weight = saturate(e2.weight * float(sampleSlot(textures, e2.weightTex, TT_SLOT_SUBSURFACE_WEIGHT, xf, in.uv, half4(1.0h), in.uvDx, in.uvDy).a));
            thin.color = float3(e2.color[0], e2.color[1], e2.color[2]) *
                         float3(sampleSlot(textures, e2.colorTex, TT_SLOT_SUBSURFACE_COLOR, xf, in.uv, half4(1.0h), in.uvDx, in.uvDy).rgb);
            thinResolved = e2.mode == 0u;
            volResolved = e2.mode == 1u;
        }
    }
    CoatParams coat{0.0f, 0.04f, float3(0.0f)};
    if (coatOn) {
        coat.factor = saturate(e0.clearcoatFactor * float(sampleSlot(textures, e0.clearcoatFactorTex, TT_SLOT_COAT, xf, in.uv, half4(1.0h), in.uvDx, in.uvDy).r));
        coat.roughness = clamp(e0.clearcoatRoughness * float(sampleSlot(textures, e0.clearcoatRoughnessTex, TT_SLOT_COAT_ROUGHNESS, xf, in.uv, half4(1.0h), in.uvDx, in.uvDy).g), 0.04f, 1.0f);
    }
    SheenParams sheen{float3(0.0f), 0.07f};
    if (sheenOn) {
        sheen.color = float3(e0.sheenColor[0], e0.sheenColor[1], e0.sheenColor[2]) *
                      float3(sampleSlot(textures, e0.sheenColorTex, TT_SLOT_SHEEN_COLOR, xf, in.uv, half4(1.0h), in.uvDx, in.uvDy).rgb);
        sheen.roughness = bsdfSheenRoughness(e0.sheenRoughness * float(sampleSlot(textures, e0.sheenRoughnessTex, TT_SLOT_SHEEN_ROUGHNESS, xf, in.uv, half4(1.0h), in.uvDx, in.uvDy).a));
    }
    if (specExt) {
        const float specularWeight =
            e0.specularFactor * float(sampleSlot(textures, e0.specularTex, TT_SLOT_SPECULAR, xf, in.uv, half4(1.0h), in.uvDx, in.uvDy).a);
        const float3 specularColor = float3(e0.specularColor[0], e0.specularColor[1], e0.specularColor[2]) *
                                     float3(sampleSlot(textures, e0.specularColorTex, TT_SLOT_SPECULAR_COLOR, xf, in.uv, half4(1.0h), in.uvDx, in.uvDy).rgb);
        spec = bsdfSpecularParams(e0.ior, specularWeight, specularColor);
        lobeIor = e0.ior;
        lobeSpecularWeight = specularWeight;
        lobeSpecularColor = specularColor;
    }

    // Two-sided lighting.  A mirrored model matrix reverses the winding, so
    // the rasteriser's facing is inverted for those instances.
    const bool front = in.frontFacing != (in.mirrored != 0);
    float3 N = normalize(front ? in.normal : -in.normal);
    const float3 N0 = N; // the shading normal before the base normal map: the coat normal without a coat normal texture
    const float3 T = in.tangent.xyz - N * dot(N, in.tangent.xyz);
    // F15.1 (ticket 11): the lobe direction T' = Tn dr.x + B dr.y, dr = the README's rotation of the texture direction (default +X) by
    // anisotropyRotation, counter-clockwise from the tangent; B = cross(N, Tn) w with w the vertex handedness times the mirrored-instance
    // sign (the vertex stage folds it into tangent.w): a mirrored instance is the exact mirror image of the plain one.
    float3 anisoDir = float3(0.0f);
    float anisoStrength = 0.0f;
    if (dot(T, T) > 1e-8) {
        const float3 Tn = normalize(T);
        const float3 B = cross(N, Tn) * in.tangent.w;
        if (anisoOn) {
            const float4 at = float4(sampleSlot(textures, e1.anisotropyTex, TT_SLOT_ANISOTROPY, xf, in.uv, half4(1.0h, 0.5h, 1.0h, 1.0h), in.uvDx, in.uvDy));
            float2 d = at.xy * 2.0f - 1.0f;
            const float dl = length(d);
            d = dl > 1e-6f ? d / dl : float2(1.0f, 0.0f);
            const float cs = cos(e1.anisotropyRotation), sn = sin(e1.anisotropyRotation);
            float3 Ta = Tn, Ba = B; // the frame of the anisotropy texture slot (rotated with its KHR_texture_transform)
            texXfFrame(xf, TT_SLOT_ANISOTROPY, Ta, Ba);
            anisoDir = Ta * (cs * d.x - sn * d.y) + Ba * (sn * d.x + cs * d.y);
            anisoStrength = saturate(e1.anisotropyStrength * at.z);
        }
        float3 tn = float3(nTex.xyz) * 2.0 - 1.0;
        tn.xy *= m.normalScale;
        if (layersOn) tn = layerTn;
        float3 Tm = Tn, Bm = B; // the frame of the normal texture slot (rotated with its KHR_texture_transform)
        texXfFrame(xf, TT_SLOT_NORMAL, Tm, Bm);
        N = normalize(Tm * tn.x + Bm * tn.y + N * tn.z);
        if (coatOn && e0.clearcoatNormalTex != INVALID_TEXTURE_INDEX) {
            float3 cn = float3(sampleSlot(textures, e0.clearcoatNormalTex, TT_SLOT_COAT_NORMAL, xf, in.uv, half4(0.5h, 0.5h, 1.0h, 1.0h), in.uvDx, in.uvDy).xyz) * 2.0 - 1.0;
            cn.xy *= e0.clearcoatNormalScale;
            float3 Tc = Tn, Bc = B;
            texXfFrame(xf, TT_SLOT_COAT_NORMAL, Tc, Bc);
            coat.normal = normalize(Tc * cn.x + Bc * cn.y + N0 * cn.z);
        } else coat.normal = N0;
    } else coat.normal = N0;

    const float3 camPos = float3(frame.cameraPosition[0], frame.cameraPosition[1], frame.cameraPosition[2]);
    const float3 V = normalize(camPos - in.worldPos);
    const bool transThin = transOn && transFactor > 0.0f && !(volThickness > 0.0f); // thin-walled: two surfaces back to back, no bend
    // The lobe is resolved when a tangent frame exists; a primitive without tangents resolves it as strength 0 (the isotropic lobe).
    const bool anisoActive = anisoOn && anisoStrength > 0.0f && dot(anisoDir, anisoDir) > 1e-12f;
    const BsdfAniso aniso = bsdfAnisoFrame(N, anisoDir, roughness, anisoStrength);

    // F15.1 (ticket 17): the volumetric diffusion replaces the diffuse of the base model (and the ior/specular dielectric) only: another
    // lobe that modifies the diffuse (coat, sheen, anisotropy, iridescence, transmission) keeps the base diffuse of that pixel.
    const bool volDiffuse = kSssWrite && volResolved && thin.weight > 0.0f && !(coatOn || sheenOn || anisoOn || iridOn || transOn);
    float3 sssUnit = float3(0.0f); // sum over the lights of the diffuse weight (albedo 1) x radiance x N.L, then the ambient term
    float3 color = float3(0.0);
    for (uint i = 0; i < frame.lightCount; ++i) {
        const device GPULight &light = lights[i];
        const float3 lightColor = float3(light.color[0], light.color[1], light.color[2]) * light.intensity;
        float3 L = float3(0.0);
        float attenuation = 1.0;
        // Generic (all three types present): light.type == LIGHT_DIRECTIONAL.
        // Directional-only scenes skip the test; scenes without directional
        // lights never take the branch.
        const bool isDirectional = kHasDirectional && (!(kHasPoint || kHasSpot) || light.type == LIGHT_DIRECTIONAL);
        if (isDirectional) {
            L = -normalize(float3(light.direction[0], light.direction[1], light.direction[2]));
        } else if (kHasPoint || kHasSpot) {
            const float3 toLight = float3(light.position[0], light.position[1], light.position[2]) - in.worldPos;
            const float dist = length(toLight);
            L = toLight / max(dist, 1e-4);
            attenuation = distanceAttenuation(dist, light.range);
            // Generic: light.type == LIGHT_SPOT; spot-only scenes skip the test,
            // point-only scenes drop the cone code.
            if (kHasSpot && (!kHasPoint || light.type == LIGHT_SPOT)) {
                const float3 spotDir = normalize(float3(light.direction[0], light.direction[1], light.direction[2]));
                const float cosOuter = cos(light.outerCone);
                const float cosInner = cos(light.innerCone);
                attenuation *= smoothstep(cosOuter, cosInner, dot(-L, spotDir));
            }
        }
        if (!isDirectional && replaceLocal) continue;
        // F15.1 (ticket 14): the RGB transmittance of the sun's shadow ray (1 on every channel without transmissive casters).
        const float3 visibility = i == shadowLight ? sunVisibility : float3(1.0f);
        if (volDiffuse) {
            const float volNL = saturate(dot(N, L));
            if (volNL > 0.0f) {
                const float volVH = saturate(dot(V, normalize(V + L)));
                const float3 du = useSpec ? bsdfSpecDiffuse(bsdfSpecFresnel(spec, volVH), metallic, float3(1.0f))
                                          : bsdfBaseDiffuse(bsdfDielectricFresnel(volVH), metallic, float3(1.0f));
                sssUnit += du * lightColor * attenuation * visibility * volNL;
            }
        }
        // F15.1 (ticket 13): a light BEHIND a thin-walled transmissive surface lights it through the transmitted lobe (delta lights only: the area
        // emitters a refracted or straight ray hits belong to the transport, decision 09).  The visible face keeps the front-hemisphere terms below.
        if (transThin && dot(N, L) < 0.0f) {
            if (physicalGI && isDirectional && i == shadowLight && sunAngularRadius > 0.0f)
                color += sunDiskThinGlass(N, V, L, baseColor.rgb, metallic, transFactor, roughness, sunAngularRadius, spec) * lightColor * attenuation * visibility;
            else
                color += bsdfThinGlassTransmitted(spec, transFactor, metallic, baseColor.rgb, N, V, L, bsdfAlpha(roughness)) * lightColor * attenuation * visibility;
            continue;
        }
        if (thinResolved) {
            // F15.1 (ticket 16): lights on the back hemisphere light the visible face through the transmitted lobe (no N.L clamp to the front).
            // F15.1 (ticket 19): a coat above the thin material attenuates the reflected lobes and the transmitted lobe alike by
            // 1 - c F_c(|V.Nc|) and adds its own lobe (research/khr-materials.md section 12; the base sits under the coat).
            float3 thinSum;
            if (physicalGI && isDirectional && i == shadowLight && sunAngularRadius > 0.0f) {
                thinSum = sunThinDiffuse(N, V, L, baseColor.rgb, metallic, spec, thin) +
                          sunDiskSpecularSpec(N, V, L, baseColor.rgb, metallic, roughness, sunAngularRadius, spec);
                if (coatOn) thinSum = thinSum * coatAttenuation(coat, V) + sunDiskCoat(V, L, sunAngularRadius, coat);
                color += thinSum * lightColor * attenuation * visibility;
            } else {
                thinSum = evaluateLightThin(N, V, L, float3(1.0f), baseColor.rgb, metallic, roughness, spec, thin);
                if (coatOn) thinSum = thinSum * coatAttenuation(coat, V) + coatLobeLight(V, L, coat);
                color += thinSum * lightColor * attenuation * visibility;
            }
            continue;
        }
        if (physicalGI && isDirectional && i == shadowLight && sunAngularRadius > 0.0f) {
            // Diffuse at the disk centre (exact to O(r^2)), specular over the disk.
            const float NdotL = saturate(dot(N, L));
            if (useSpec) {
                const float sunVH = saturate(dot(V, normalize(V + L)));
                const float3 Fd = bsdfSpecFresnel(spec, sunVH);
                const float3 diffuse = NdotL > 0.0f ? bsdfSpecDiffuseIrid(spec, irid, Fd, metallic, baseColor.rgb, sunVH) * NdotL * diffuseT : float3(0.0);
                float3 sunSum = diffuse + (anisoActive ? sunDiskSpecularAniso(N, V, L, baseColor.rgb, metallic, sunAngularRadius, spec, aniso, irid)
                                                       : sunDiskSpecularSpec(N, V, L, baseColor.rgb, metallic, roughness, sunAngularRadius, spec, irid));
                if (sheenOn) sunSum = sheenLayerLight(sunSum, N, V, L, sheen, frame.sheenTable);
                if (coatOn) sunSum = sunSum * coatAttenuation(coat, V) + sunDiskCoat(V, L, sunAngularRadius, coat);
                color += sunSum * lightColor * attenuation * visibility;
            } else {
                const float3 Fd = bsdfDielectricFresnel(saturate(dot(V, normalize(V + L))));
                const float3 diffuse = NdotL > 0.0f ? bsdfBaseDiffuse(Fd, metallic, baseColor.rgb) * NdotL : float3(0.0);
                float3 sunSum = diffuse + sunDiskSpecular(N, V, L, baseColor.rgb, metallic, roughness, sunAngularRadius);
                if (sheenOn) sunSum = sheenLayerLight(sunSum, N, V, L, sheen, frame.sheenTable);
                color += sunSum * lightColor * attenuation * visibility;
            }
        } else if (sheenOn) {
            // The material under the sheen (ior/specular when present), the sheen layer, then the coat above (decision 06).
            float3 value = anisoActive ? evaluateLightAniso(N, V, L, float3(1.0f), baseColor.rgb, metallic, spec, aniso, irid, diffuseT)
                         : useSpec ? evaluateLightSpec(N, V, L, float3(1.0f), baseColor.rgb, metallic, roughness, spec, irid, diffuseT)
                                   : evaluateLight(N, V, L, float3(1.0f), baseColor.rgb, metallic, roughness);
            value = sheenLayerLight(value, N, V, L, sheen, frame.sheenTable);
            if (coatOn) value = value * coatAttenuation(coat, V) + coatLobeLight(V, L, coat);
            color += value * lightColor * attenuation * visibility;
        } else if (anisoActive && coatOn) {
            const float3 value = evaluateLightAniso(N, V, L, float3(1.0f), baseColor.rgb, metallic, spec, aniso, irid, diffuseT) * coatAttenuation(coat, V) +
                                 coatLobeLight(V, L, coat);
            color += value * lightColor * attenuation * visibility;
        } else if (anisoActive)
            color += evaluateLightAniso(N, V, L, lightColor * attenuation, baseColor.rgb, metallic, spec, aniso, irid, diffuseT) * visibility;
        else if (coatOn)
            color += evaluateLightCoat(N, V, L, lightColor * attenuation, baseColor.rgb, metallic, roughness, spec, coat, irid, diffuseT) * visibility;
        else if (specExt || transOn || bsdfIridActive(irid))
            color += evaluateLightSpec(N, V, L, lightColor * attenuation, baseColor.rgb, metallic, roughness, spec, irid, diffuseT) * visibility;
        else
            color += evaluateLight(N, V, L, lightColor * attenuation, baseColor.rgb, metallic, roughness) * visibility;
    }

    if (replaceLocal && !externalDiffuse) color += localDirect;

    // Ambient: diffuse for dielectrics only, specular from the hemisphere in
    // the reflected direction, widened towards N as the lobe gets rougher.
    const float NdotV = max(dot(N, V), 1e-4);
    const float3 f0 = bsdfBaseF0(baseColor.rgb, metallic);
    // F15.1 (ticket 11): anisotropy fetches along the README's bent normal.
    const float3 envNormal = anisoActive ? bsdfAnisoBentNormal(N, V, aniso, roughness, anisoStrength) : N;
    const float3 specularDir = normalize(mix(reflect(-V, envNormal), envNormal, roughness * roughness));
    // F13 path (physicalGI): the scene's single declared isotropic environment,
    // the same radiance GI and reflection misses see; legacy keeps the gradient.
    float envDiffuse = bsdfEnvDiffuseWeight(roughness, NdotV, frame.dfgTable); // ADR-0006
    float3 envSpecularWeight = bsdfEnvSpecular(f0, roughness, NdotV, frame.dfgTable);
    if (useSpec) { // F15.1: f0 * A + f90 * B per lobe, diffuse weight 1 - max(E_dielectric)
        // F15.1 (ticket 11): the split sum (diffuse weight included) at the effective isotropic roughness of an anisotropic lobe.
        // F15.1 (ticket 12): the film at N.V (bsdfSpecEnvIrid; no film = bsdfSpecEnv).
        const BsdfSpecEnv env = bsdfSpecEnvIrid(spec, irid, baseColor.rgb, metallic, anisoActive ? bsdfAnisoEnvRoughness(roughness, anisoStrength) : roughness,
                                                NdotV, frame.dfgTable);
        envDiffuse = env.diffuseWeight;
        envSpecularWeight = env.specular;
    }
    // F15.1 (ticket 13): the diffuse base of a transmissive material is mixed by 1 - t; the transmitted lobe takes (1 - E) t (1 - metallic) tint T
    // of the radiance along the refracted direction (declared: the environment, or the legacy hemisphere along the refraction of V through N,
    // with the volume absorption over the thickness hint, decision 09).  Added below where the resolve owns the ambient specular, i.e. where
    // the specular transport (which traces or fetches the refracted radiance itself) does not run.
    float3 transWeight = float3(0.0f);
    float3 transDirection = -V;
    if (transOn && transFactor > 0.0f) {
        const float3 absorption = volThickness > 0.0f ? bsdfVolumeAbsorption(volAttenuationColor, volAttenuationDistance, volThickness) : float3(1.0f);
        transWeight = bsdfTransmissionEnvWeight(transFactor, metallic, envDiffuse, baseColor.rgb, absorption);
        const float3 refracted = refract(-V, N, 1.0f / max(lobeIor, 1.0f));
        if (lobeIor >= 1.0f && volThickness > 0.0f && dot(refracted, refracted) > 0.0f) transDirection = normalize(refracted);
        envDiffuse *= diffuseT;
    }
    // F15.1 (ticket 09): the coat term of the environment, (1 - c F_c) material + c F_c clamp(A_c + B_c, 0, 1) at the coat roughness.
    float coatAtt = 1.0f;
    float3 coatEnvironment = float3(0.0);
    if (coatOn) {
        const float coatNV = max(dot(coat.normal, V), 1e-4);
        const float fc = bsdfCoatFresnel(abs(dot(coat.normal, V)));
        coatAtt = 1.0f - coat.factor * fc;
        envDiffuse *= coatAtt;
        envSpecularWeight *= coatAtt;
        const float3 coatDir = normalize(mix(reflect(-V, coat.normal), coat.normal, coat.roughness * coat.roughness));
        coatEnvironment = (physicalGI ? environment : hemisphere(coatDir)) *
                          (coat.factor * fc * bsdfCoatEnvAlbedo(coat.roughness, coatNV, frame.dfgTable));
    }
    // F15.1 (ticket 10): the sheen layer scales every indirect material term by 1 - max3(S) E(N.V, r) and adds its own environment
    // term S E L(N): L(N) is the GI irradiance / pi (useGI), the declared environment (physicalGI) or the legacy hemisphere at N.
    // The term travels with the diffuse ambient (the composite adds it when the resolve strips the external diffuse); the coat
    // attenuates it like the material.
    float3 sheenEnvironment = float3(0.0);
    if (sheenOn) {
        const float E = bsdfSheenE(sheen.roughness, NdotV, frame.sheenTable);
        const float scale = 1.0f - bsdfMax3(sheen.color) * E;
        envDiffuse *= scale;
        envSpecularWeight *= scale;
        sheenEnvironment = (useGI ? indirectIrradiance / M_PI_F : (physicalGI ? environment : hemisphere(N))) * sheen.color * E;
    }
    // F15.1 (ticket 16): the front diffuse weight is scaled by 1 - w and the transmitted lobe takes w color (1 - E_d) (1 - 0.16 N.V) of the
    // radiance at -N: the declared environment (physicalGI), the legacy hemisphere at -N, or (GI) the irradiance of this pixel, which stands
    // for E(-N) (declared approximation: no probe lookup at -N).  The term travels with the diffuse ambient.
    float3 thinWeight = float3(0.0);
    if (thinResolved) {
        thinWeight = bsdfThinEnvWeight(thin, metallic, envDiffuse, NdotV);
        envDiffuse *= bsdfThinReflectScale(thin);
    }
    const float3 ambientDiffuse = (physicalGI ? environment : hemisphere(N)) * baseColor.rgb * (1.0 - metallic) * envDiffuse;
    const float3 ambientSpecular = (physicalGI ? environment : hemisphere(specularDir)) * envSpecularWeight + coatEnvironment;
    // F13 assembles these positive signals once outside this residual base.
    // Never subtract full-float lighting from a quantized summed color later.
    if(!externalDiffuse)color += (useGI ? indirectIrradiance * baseColor.rgb * (1.0f - metallic) * envDiffuse / M_PI_F * (physicalGI?1.0f:occlusion) : ambientDiffuse * occlusion);
    if(!externalDiffuse && thinResolved)color += useGI ? indirectIrradiance * thinWeight / M_PI_F * (physicalGI?1.0f:occlusion)
                                                       : (physicalGI ? environment : hemisphere(-N)) * thinWeight * occlusion;
    if (volDiffuse) {
        // F15.1 (ticket 17): the diffuse of the pixel (the resolve's lights + ambient, base colour) is stripped from the colour; weight x colour x the same
        // unit-albedo irradiance goes to the target, where the diffusion profile spreads it (and the recombination adds it back).
        // The ambient diffuse joins only where the resolve owns it (an external composite adds its own, not diffused: declared).
        if (!externalDiffuse)
            sssUnit += useGI ? indirectIrradiance * (1.0f - metallic) * envDiffuse / M_PI_F * (physicalGI ? 1.0f : occlusion)
                             : (physicalGI ? environment : hemisphere(N)) * (1.0f - metallic) * envDiffuse * occlusion;
        color = max(color - thin.weight * baseColor.rgb * sssUnit, float3(0.0f));
        out.sssIrradiance = thin.weight * thin.color * sssUnit;
        out.sssMaterial = in.materialIndex + 1u;
    }
    if(!externalDiffuse && sheenOn)color += sheenEnvironment * coatAtt * (useGI ? (physicalGI?1.0f:occlusion) : occlusion);
    if(!omitSpecularAmbient)color += ambientSpecular * occlusion;
    if (!omitSpecularAmbient && transOn && transFactor > 0.0f && lobeIor >= 1.0f) color += (physicalGI ? environment : hemisphere(transDirection)) * transWeight;

    // The emission under a coat is attenuated like the material (the DI emitter share stays uncoated: declared).
    const float3 emission = float3(m.emissive[0], m.emissive[1], m.emissive[2]) * float3(emTex.rgb) * coatAtt;
    color += emission;

    out.color = color;
    out.emission = emission;
    out.normal = N;
    out.baseColor = baseColor.rgb;
    out.diffuseAlbedo = baseColor.rgb * (1.0f - metallic) * envDiffuse + thinWeight;
    out.specularAlbedo = useSpec ? mix(bsdfSpecFresnel(spec, NdotV), bsdfFSchlick(baseColor.rgb, NdotV), metallic)
                                 : bsdfFSchlick(f0, NdotV);
    if (bsdfIridActive(irid)) out.specularAlbedo = bsdfSpecColorIrid(spec, irid, bsdfSpecFresnel(spec, NdotV), baseColor.rgb, metallic, NdotV);
    if (sheenOn) out.specularAlbedo *= bsdfSheenScaleView(sheen.color, sheen.roughness, NdotV, frame.sheenTable);
    if (coatOn) out.specularAlbedo = out.specularAlbedo * coatAtt + coat.factor * bsdfCoatFresnel(abs(dot(coat.normal, V)));
    out.roughness = roughness;
    out.metallic = metallic;
    out.occlusion = occlusion;
    // F15.1: only the features the resolve resolves today (LOBE_GUIDE_RESOLVED_FEATURES) and the scene's
    // union reach the guide; every other LobeParams field stays 0 (packLobeGuide ignores clear features).
    out.lobes.features = (kMaterialFeatures != 0u)
                             ? ((m.flags >> MATERIAL_FEATURE_SHIFT) & kMaterialFeatures & LOBE_GUIDE_RESOLVED_FEATURES)
                             : 0u;
    // An anisotropy bit without a resolved lobe (no tangent frame, or a texture strength of 0) is not published: the guide then
    // describes the isotropic lobe, as the resolve shaded it.
    if (anisoOn && !anisoActive) out.lobes.features &= ~MATERIAL_FEATURE_ANISOTROPY;
    // The subsurface bit is published for both modes (the volumetric one carries the flag and the profile parameters, ticket 17).
    if (thinOn && !thinResolved && !volResolved) out.lobes.features &= ~MATERIAL_FEATURE_SUBSURFACE;
    if (thinResolved || volResolved) {
        out.lobes.subsurfaceWeight = thin.weight;
        out.lobes.subsurfaceColor[0] = thin.color.x;
        out.lobes.subsurfaceColor[1] = thin.color.y;
        out.lobes.subsurfaceColor[2] = thin.color.z;
    }
    if (volResolved) {
        out.lobes.features |= LOBE_GUIDE_FLAG_SUBSURFACE_VOLUMETRIC;
        out.lobes.subsurfaceRadius = e2.radius;
        out.lobes.subsurfaceAnisotropy = e2.anisotropy;
        for (uint c = 0; c < 3; ++c) out.lobes.subsurfaceRadiusScale[c] = e2.radiusScale[c];
    }
    // F15.1 (ticket 13): the volume bit travels with the transmission bit only ("a volume without transmission has no effect"); the factor, the
    // world thickness (mesh-local x instance scale) and the medium.
    if (!transOn) out.lobes.features &= ~(MATERIAL_FEATURE_TRANSMISSION | MATERIAL_FEATURE_VOLUME);
    else if (!volOn) out.lobes.features &= ~MATERIAL_FEATURE_VOLUME;
    if (transOn) {
        out.lobes.transmissionFactor = transFactor;
        if (volOn) {
            out.lobes.volumeThickness = volThickness;
            for (uint c = 0; c < 3; ++c) out.lobes.attenuationColor[c] = volAttenuationColor[c];
            out.lobes.attenuationDistance = volAttenuationDistance;
        }
        out.lobes.ior = lobeIor;
    }
    if (specExt) {
        out.lobes.ior = lobeIor;
        out.lobes.specularFactor = lobeSpecularWeight;
        out.lobes.specularColor[0] = lobeSpecularColor.x;
        out.lobes.specularColor[1] = lobeSpecularColor.y;
        out.lobes.specularColor[2] = lobeSpecularColor.z;
    }
    if (sheenOn) {
        out.lobes.sheenColor[0] = sheen.color.x;
        out.lobes.sheenColor[1] = sheen.color.y;
        out.lobes.sheenColor[2] = sheen.color.z;
        out.lobes.sheenRoughness = sheen.roughness;
    }
    if (anisoActive) {
        out.lobes.anisotropyStrength = anisoStrength;
        out.lobes.anisotropyDirection[0] = aniso.t.x;
        out.lobes.anisotropyDirection[1] = aniso.t.y;
        out.lobes.anisotropyDirection[2] = aniso.t.z;
    }
    if (iridOn) {
        out.lobes.iridescenceFactor = irid.strength;
        out.lobes.iridescenceIor = irid.ior;
        out.lobes.iridescenceThickness = irid.thickness;
    }
    if (coatOn) {
        out.lobes.clearcoatFactor = coat.factor;
        out.lobes.clearcoatRoughness = coat.roughness;
        out.lobes.clearcoatNormal[0] = coat.normal.x;
        out.lobes.clearcoatNormal[1] = coat.normal.y;
        out.lobes.clearcoatNormal[2] = coat.normal.z;
    }
    return out;
}

// F15.1 (ticket 06): the guide record of a resolved surface (renderer/lobe_guide_pack.h).  DI_LOBE_GUIDE_CORRUPT
// is the negative control of --debug-lobe-guide: +0.25 on the stored ior of ior/specular pixels and (ticket 10) on the stored
// sheen roughness of sheen pixels, (ticket 11) the lobe direction of anisotropic pixels (set to the shading normal) and (ticket 12)
// the film ior of iridescent pixels and (ticket 16) the weight of thin subsurface pixels.
static GPULobeGuide lobeGuideRecord(ShadingResult value, uint diFlags) {
    if ((diFlags & DI_LOBE_GUIDE_CORRUPT) != 0u && (value.lobes.features & MATERIAL_FEATURE_SPECULAR_IOR) != 0u)
        value.lobes.ior += 0.25f;
    if ((diFlags & DI_LOBE_GUIDE_CORRUPT) != 0u && (value.lobes.features & MATERIAL_FEATURE_SHEEN) != 0u)
        value.lobes.sheenRoughness += 0.25f;
    // Ticket 12: an iridescent pixel stores a film ior shifted by 0.25 (the half-rounded value differs from the expectation).
    if ((diFlags & DI_LOBE_GUIDE_CORRUPT) != 0u && (value.lobes.features & MATERIAL_FEATURE_IRIDESCENCE) != 0u)
        value.lobes.iridescenceIor += 0.25f;
    // Ticket 16: a thin subsurface pixel stores a weight shifted by 0.25 (a transmission colour channel would not move for a zero channel).
    if ((diFlags & DI_LOBE_GUIDE_CORRUPT) != 0u && (value.lobes.features & MATERIAL_FEATURE_SUBSURFACE) != 0u)
        value.lobes.subsurfaceWeight += 0.25f;
    // Ticket 13: a transmissive pixel stores a factor shifted by 0.25 (and a volume pixel its thickness).
    if ((diFlags & DI_LOBE_GUIDE_CORRUPT) != 0u && (value.lobes.features & MATERIAL_FEATURE_TRANSMISSION) != 0u)
        value.lobes.transmissionFactor += 0.25f;
    if ((diFlags & DI_LOBE_GUIDE_CORRUPT) != 0u && (value.lobes.features & MATERIAL_FEATURE_VOLUME) != 0u)
        value.lobes.volumeThickness += 0.25f;
    // Ticket 11: an anisotropic pixel stores its shading normal as the lobe direction (outside the tangent plane).
    if ((diFlags & DI_LOBE_GUIDE_CORRUPT) != 0u && (value.lobes.features & MATERIAL_FEATURE_ANISOTROPY) != 0u) {
        value.lobes.anisotropyDirection[0] = value.normal.x;
        value.lobes.anisotropyDirection[1] = value.normal.y;
        value.lobes.anisotropyDirection[2] = value.normal.z;
    }
    return packLobeGuide(value.lobes);
}
