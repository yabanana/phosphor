#pragma once
#include <metal_stdlib>
#include "renderer/gpu_types.h"
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
};
struct ShadingResult {
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
};

// --- BRDF --------------------------------------------------------------------

constexpr sampler kMaterialSampler(filter::linear, mip_filter::linear, address::repeat, max_anisotropy(8));

static half4 sampleOr(const device TextureHandle *table, uint index, float2 uv, half4 fallback, float2 dx, float2 dy) {
    if (index == INVALID_TEXTURE_INDEX) {
        return fallback;
    }
    return half4(table[index].tex.sample(kMaterialSampler, uv, gradient2d(dx, dy)));
}

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
                                float roughness, BsdfDielectricSpec spec) {
    const float3 H = normalize(V + L);
    const float NdotL = saturate(dot(N, L));
    const float NdotV = max(dot(N, V), 1e-4);
    const float NdotH = saturate(dot(N, H));
    const float VdotH = saturate(dot(V, H));
    if (NdotL <= 0.0) {
        return float3(0.0);
    }
    const BsdfBaseLobes lobes = bsdfSpecEval(spec, NdotL, NdotV, NdotH, VdotH, albedo, metallic, bsdfAlpha(roughness));
    return (lobes.diffuse + lobes.specular) * radiance * NdotL;
}

static float3 sunDiskSpecularSpec(float3 N, float3 V, float3 L, float3 albedo, float metallic, float roughness,
                                  float angularRadius, BsdfDielectricSpec spec) {
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
               bsdfSpecLobeColor(bsdfSpecFresnel(spec, VdotH), albedo, metallic, VdotH) * NdotL;
    }
    return sum / float(count);
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

static ShadingResult shadeSurface(SurfaceInput in, constant FrameConstants &frame, const device GPUMaterial *materials,
                                  const device GPULight *lights, const device TextureHandle *textures,
                                  bool alphaTest = true, float sunVisibility = 1.0f, uint shadowLight = ~0u,
                                  bool replaceLocal = false, float3 localDirect = float3(0),
                                  bool useGI = false, float3 indirectIrradiance = float3(0),
                                  bool omitSpecularAmbient = false, bool physicalGI = false, bool externalDiffuse = false,
                                  float3 environment = float3(0), float sunAngularRadius = 0.0f) {
    const device GPUMaterial &m = materials[in.materialIndex];

    const half4 baseTex = sampleOr(textures, m.baseColorTex, in.uv, half4(1.0h), in.uvDx, in.uvDy);
    const half4 mrTex = sampleOr(textures, m.metallicRoughnessTex, in.uv, half4(1.0h), in.uvDx, in.uvDy);
    const half4 aoTex = sampleOr(textures, m.occlusionTex, in.uv, half4(1.0h), in.uvDx, in.uvDy);
    // EMISSIVE == false: no material of the scene emits, so the texture fetch
    // is skipped.  The multiply-add below stays unconditional on purpose: a
    // branch around it changes the compiler's contraction of the generic
    // pipeline by 1 LSB in some pixels (measured), and adding factor(0) * 1
    // leaves the result exact.
    const half4 emTex = (kEmissive && (kResolveClass == 4u || (kResolveClass & 2u) != 0u))
                            ? sampleOr(textures, m.emissiveTex, in.uv, half4(1.0h), in.uvDx, in.uvDy)
                            : half4(1.0h);
    const half4 nTex =
        (kResolveClass == 4u || (kResolveClass & 1u) != 0u)
            ? sampleOr(textures, m.normalTex, in.uv, half4(0.5h, 0.5h, 1.0h, 1.0h), in.uvDx, in.uvDy)
            : (m.normalTex == INVALID_TEXTURE_INDEX ? half4(0.5h, 0.5h, 1.0h, 1.0h)
                                                    : half4(half(128.0f / 255.0f), half(128.0f / 255.0f), 1.0h, 1.0h));

    const float4 baseColor = float4(m.baseColor[0], m.baseColor[1], m.baseColor[2], m.baseColor[3]) * float4(baseTex);
    ShadingResult out{};
    out.alpha = baseColor.a;
    if (alphaTest && baseColor.a < m.alphaCutoff)
        return out;

    // glTF: G = roughness, B = metallic.
    const float roughness = clamp(m.roughness * float(mrTex.g), 0.04, 1.0);
    const float metallic = saturate(m.metallic * float(mrTex.b));
    const float occlusion = mix(1.0, float(aoTex.r), m.occlusionStrength);

    // F15.1: KHR_materials_ior + KHR_materials_specular.  Compiled only when the scene's feature
    // union has the bit (kMaterialFeatures, undefined => 0) and run only for materials that carry it.
    const bool specExt = (kMaterialFeatures & MATERIAL_FEATURE_SPECULAR_IOR) != 0u &&
                         (m.flags & (MATERIAL_FEATURE_SPECULAR_IOR << MATERIAL_FEATURE_SHIFT)) != 0u;
    BsdfDielectricSpec spec;
    spec.f0 = float3(0.04f);
    spec.f90 = 1.0f;
    if (specExt) {
        const GPUMaterialExt0 e0 = *reinterpret_cast<const device GPUMaterialExt0*>(frame.materialExtensions + m.extensionIndex);
        const float specularWeight =
            e0.specularFactor * float(sampleOr(textures, e0.specularTex, in.uv, half4(1.0h), in.uvDx, in.uvDy).a);
        const float3 specularColor = float3(e0.specularColor[0], e0.specularColor[1], e0.specularColor[2]) *
                                     float3(sampleOr(textures, e0.specularColorTex, in.uv, half4(1.0h), in.uvDx, in.uvDy).rgb);
        spec = bsdfSpecularParams(e0.ior, specularWeight, specularColor);
    }

    // Two-sided lighting.  A mirrored model matrix reverses the winding, so
    // the rasteriser's facing is inverted for those instances.
    const bool front = in.frontFacing != (in.mirrored != 0);
    float3 N = normalize(front ? in.normal : -in.normal);
    const float3 T = in.tangent.xyz - N * dot(N, in.tangent.xyz);
    if (dot(T, T) > 1e-8) {
        const float3 Tn = normalize(T);
        const float3 B = cross(N, Tn) * in.tangent.w;
        float3 tn = float3(nTex.xyz) * 2.0 - 1.0;
        tn.xy *= m.normalScale;
        N = normalize(Tn * tn.x + B * tn.y + N * tn.z);
    }

    const float3 camPos = float3(frame.cameraPosition[0], frame.cameraPosition[1], frame.cameraPosition[2]);
    const float3 V = normalize(camPos - in.worldPos);

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
        const float visibility = i == shadowLight ? sunVisibility : 1.0f;
        if (physicalGI && isDirectional && i == shadowLight && sunAngularRadius > 0.0f) {
            // Diffuse at the disk centre (exact to O(r^2)), specular over the disk.
            const float NdotL = saturate(dot(N, L));
            if (specExt) {
                const float3 Fd = bsdfSpecFresnel(spec, saturate(dot(V, normalize(V + L))));
                const float3 diffuse = NdotL > 0.0f ? bsdfSpecDiffuse(Fd, metallic, baseColor.rgb) * NdotL : float3(0.0);
                color += (diffuse + sunDiskSpecularSpec(N, V, L, baseColor.rgb, metallic, roughness, sunAngularRadius, spec)) *
                         lightColor * attenuation * visibility;
            } else {
                const float3 Fd = bsdfDielectricFresnel(saturate(dot(V, normalize(V + L))));
                const float3 diffuse = NdotL > 0.0f ? bsdfBaseDiffuse(Fd, metallic, baseColor.rgb) * NdotL : float3(0.0);
                color += (diffuse + sunDiskSpecular(N, V, L, baseColor.rgb, metallic, roughness, sunAngularRadius)) *
                         lightColor * attenuation * visibility;
            }
        } else if (specExt)
            color += evaluateLightSpec(N, V, L, lightColor * attenuation, baseColor.rgb, metallic, roughness, spec) * visibility;
        else
            color += evaluateLight(N, V, L, lightColor * attenuation, baseColor.rgb, metallic, roughness) * visibility;
    }

    if (replaceLocal && !externalDiffuse) color += localDirect;

    // Ambient: diffuse for dielectrics only, specular from the hemisphere in
    // the reflected direction, widened towards N as the lobe gets rougher.
    const float NdotV = max(dot(N, V), 1e-4);
    const float3 f0 = bsdfBaseF0(baseColor.rgb, metallic);
    const float3 specularDir = normalize(mix(reflect(-V, N), N, roughness * roughness));
    // F13 path (physicalGI): the scene's single declared isotropic environment,
    // the same radiance GI and reflection misses see; legacy keeps the gradient.
    float envDiffuse = bsdfEnvDiffuseWeight(roughness, NdotV, frame.dfgTable); // ADR-0006
    float3 envSpecularWeight = bsdfEnvSpecular(f0, roughness, NdotV, frame.dfgTable);
    if (specExt) { // F15.1: f0 * A + f90 * B per lobe, diffuse weight 1 - max(E_dielectric)
        const BsdfSpecEnv env = bsdfSpecEnv(spec, baseColor.rgb, metallic, roughness, NdotV, frame.dfgTable);
        envDiffuse = env.diffuseWeight;
        envSpecularWeight = env.specular;
    }
    const float3 ambientDiffuse = (physicalGI ? environment : hemisphere(N)) * baseColor.rgb * (1.0 - metallic) * envDiffuse;
    const float3 ambientSpecular = (physicalGI ? environment : hemisphere(specularDir)) * envSpecularWeight;
    // F13 assembles these positive signals once outside this residual base.
    // Never subtract full-float lighting from a quantized summed color later.
    if(!externalDiffuse)color += (useGI ? indirectIrradiance * baseColor.rgb * (1.0f - metallic) * envDiffuse / M_PI_F * (physicalGI?1.0f:occlusion) : ambientDiffuse * occlusion);
    if(!omitSpecularAmbient)color += ambientSpecular * occlusion;

    const float3 emission = float3(m.emissive[0], m.emissive[1], m.emissive[2]) * float3(emTex.rgb);
    color += emission;

    out.color = color;
    out.emission = emission;
    out.normal = N;
    out.baseColor = baseColor.rgb;
    out.diffuseAlbedo = baseColor.rgb * (1.0f - metallic) * envDiffuse;
    out.specularAlbedo = specExt ? mix(bsdfSpecFresnel(spec, NdotV), bsdfFSchlick(baseColor.rgb, NdotV), metallic)
                                 : bsdfFSchlick(f0, NdotV);
    out.roughness = roughness;
    out.metallic = metallic;
    out.occlusion = occlusion;
    return out;
}
