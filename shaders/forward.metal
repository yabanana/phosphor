// forward.metal -- F0 forward pass: indexed draws, PBR (GGX), direct lights.
//
// Bindings (Metal 4 argument table, buffers by GPU address):
//   buffer(0)  FrameConstants
//   buffer(1)  GPUVertex[]      (global vertex buffer)
//   buffer(2)  GPUInstance[]    (F5: persistent, by scene store slot)
//   buffer(3)  GPUMaterial[]
//   buffer(4)  GPULight[]
//   buffer(5)  TextureHandle[]  (bindless texture table of resource IDs)
//   buffer(6)  uint[]           (F5: visible instance slots; instance_id
//                               indexes it, identity in --gpu-driven off)
//
// The visibility-buffer / mesh-shader pipeline replaces this pass in F2; the
// forward pass stays as the reference path and for debugging.

#include <metal_stdlib>
#include "renderer/gpu_types.h"

// Function constants (F3.3, generated from shaders/variants.def).  With none
// defined -- the generic pipeline -- every k<Name> takes its generic value, so
// the generic pipeline behaves exactly like the unspecialised shader.
#include "pipeline/forward_variants.generated.metal.h"

using namespace metal;
using namespace phosphor;

// Light types present in the scene (bitmask 1 directional, 2 point, 4 spot).
// A type absent from the mask has its code path removed at pipeline creation.
constant bool kHasDirectional = (kLightTypes & 1u) != 0u;
constant bool kHasPoint       = (kLightTypes & 2u) != 0u;
constant bool kHasSpot        = (kLightTypes & 4u) != 0u;

struct TextureHandle {
    texture2d<float> tex;
};

struct VertexOut {
    float4 position [[position]];
    float3 worldPos;
    float3 normal;
    float4 tangent;
    float2 uv;
    uint   materialIndex [[flat]];
    uint   mirrored [[flat]]; // 1 if INSTANCE_FLAG_MIRRORED
};

static float4x4 loadMatrix(const device float* m) {
    return float4x4(float4(m[0],  m[1],  m[2],  m[3]),
                    float4(m[4],  m[5],  m[6],  m[7]),
                    float4(m[8],  m[9],  m[10], m[11]),
                    float4(m[12], m[13], m[14], m[15]));
}

static float4x4 loadMatrix(constant float* m) {
    return float4x4(float4(m[0],  m[1],  m[2],  m[3]),
                    float4(m[4],  m[5],  m[6],  m[7]),
                    float4(m[8],  m[9],  m[10], m[11]),
                    float4(m[12], m[13], m[14], m[15]));
}

vertex VertexOut forward_vs(uint vertexId                          [[vertex_id]],
                            uint instanceId                        [[instance_id]],
                            constant FrameConstants& frame         [[buffer(0)]],
                            const device GPUVertex* vertices       [[buffer(1)]],
                            const device GPUInstance* instances    [[buffer(2)]],
                            const device uint* visible             [[buffer(6)]])
{
    // vertex_id includes the draw's base vertex and instance_id includes the
    // base instance: instance_id indexes the visible list, which names the slot.
    const device GPUVertex& v    = vertices[vertexId];
    const device GPUInstance& gi = instances[visible[instanceId]];

    const float4x4 model = loadMatrix(gi.modelMatrix);
    const float3x3 normalMatrix = float3x3(model[0].xyz, model[1].xyz, model[2].xyz);

    const float4 world = model * float4(v.px, v.py, v.pz, 1.0);

    VertexOut out;
    out.position      = loadMatrix(frame.viewProjection) * world;
    out.worldPos      = world.xyz;
    // Non-uniform scale is rare in the test benches; renormalise in the FS.
    out.normal        = normalMatrix * float3(v.nx, v.ny, v.nz);
    out.tangent       = float4(normalMatrix * float3(v.tx, v.ty, v.tz), v.tw);
    out.uv            = float2(v.u, v.v);
    out.materialIndex = gi.materialIndex;
    out.mirrored      = (gi.flags & INSTANCE_FLAG_MIRRORED) != 0 ? 1u : 0u;
    return out;
}

// --- BRDF --------------------------------------------------------------------

constexpr sampler kMaterialSampler(filter::linear, mip_filter::linear,
                                   address::repeat, max_anisotropy(8));

static half4 sampleOr(const device TextureHandle* table, uint index, float2 uv, half4 fallback) {
    if (index == INVALID_TEXTURE_INDEX) {
        return fallback;
    }
    return half4(table[index].tex.sample(kMaterialSampler, uv));
}

static float D_GGX(float NdotH, float a) {
    const float a2 = a * a;
    const float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (M_PI_F * d * d);
}

static float V_SmithGGXCorrelated(float NdotV, float NdotL, float a) {
    const float a2 = a * a;
    const float gv = NdotL * sqrt(NdotV * NdotV * (1.0 - a2) + a2);
    const float gl = NdotV * sqrt(NdotL * NdotL * (1.0 - a2) + a2);
    return 0.5 / max(gv + gl, 1e-5);
}

static float3 F_Schlick(float3 f0, float VdotH) {
    return f0 + (1.0 - f0) * pow(1.0 - VdotH, 5.0);
}

static float3 evaluateLight(float3 N, float3 V, float3 L, float3 radiance,
                            float3 albedo, float metallic, float roughness) {
    const float3 H = normalize(V + L);
    const float NdotL = saturate(dot(N, L));
    const float NdotV = max(dot(N, V), 1e-4);
    const float NdotH = saturate(dot(N, H));
    const float VdotH = saturate(dot(V, H));
    if (NdotL <= 0.0) {
        return float3(0.0);
    }

    const float a = max(roughness * roughness, 0.002);
    const float3 f0 = mix(float3(0.04), albedo, metallic);
    const float3 F = F_Schlick(f0, VdotH);
    const float3 specular = D_GGX(NdotH, a) * V_SmithGGXCorrelated(NdotV, NdotL, a) * F;
    const float3 diffuse = (1.0 - F) * (1.0 - metallic) * albedo / M_PI_F;
    return (diffuse + specular) * radiance * NdotL;
}

static float distanceAttenuation(float dist, float range) {
    // Inverse square with a smooth window to zero at `range`.
    const float r = dist / max(range, 1e-3);
    const float window = saturate(1.0 - r * r * r * r);
    return window * window / max(dist * dist, 1e-4);
}

// Karis, "Physically Based Shading on Mobile" (2014): analytic fit of the
// split-sum environment BRDF, so ambient light keeps a specular term (and
// metals stay metallic) without an irradiance/prefiltered-environment map.
static float3 envBRDFApprox(float3 f0, float roughness, float NdotV) {
    const float4 c0 = float4(-1.0, -0.0275, -0.572, 0.022);
    const float4 c1 = float4(1.0, 0.0425, 1.04, -0.04);
    const float4 r = roughness * c0 + c1;
    const float a004 = min(r.x * r.x, exp2(-9.28 * NdotV)) * r.x + r.y;
    const float2 ab = float2(-1.04, 1.04) * a004 + r.zw;
    return f0 * ab.x + ab.y;
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

fragment half4 forward_fs(VertexOut in                              [[stage_in]],
                          bool frontFacing                          [[front_facing]],
                          constant FrameConstants& frame            [[buffer(0)]],
                          const device GPUMaterial* materials       [[buffer(3)]],
                          const device GPULight* lights             [[buffer(4)]],
                          const device TextureHandle* textures      [[buffer(5)]])
{
#ifdef PHOSPHOR_HOT_RELOAD_PROBE
    // F3.6 self-test (--debug-hot-reload): the probe library, built by CMake
    // with this define, draws opaque magenta so the engine can check exactly
    // that every forward pipeline was swapped.  Absent from normal builds.
    return half4(1.0h, 0.0h, 1.0h, 1.0h);
#endif
    // Salt (F3.1 cold-compile measurements): a defined salt embeds its value in
    // the binary so the OS shader cache cannot serve the specialisation.  The
    // guard depends on a runtime value (a light count of 0xFFFFFFFF cannot
    // occur: the light buffer would be 96 GB), so the compiler cannot fold the
    // branch away, yet it is never taken and the output cannot change.  With
    // the salt undefined (every normal pipeline) the branch is removed.  It
    // sits first, before any shading maths: placed later, its control flow
    // perturbed the floating-point contraction of the generic pipeline by 1 LSB.
    if (is_function_constant_defined(FC_SALT) && frame.lightCount == 0xFFFFFFFFu) {
        return half4(half(float(FC_SALT)));
    }

    const device GPUMaterial& m = materials[in.materialIndex];

    const half4 baseTex = sampleOr(textures, m.baseColorTex, in.uv, half4(1.0h));
    const half4 mrTex   = sampleOr(textures, m.metallicRoughnessTex, in.uv, half4(1.0h));
    const half4 aoTex   = sampleOr(textures, m.occlusionTex, in.uv, half4(1.0h));
    // EMISSIVE == false: no material of the scene emits, so the texture fetch
    // is skipped.  The multiply-add below stays unconditional on purpose: a
    // branch around it changes the compiler's contraction of the generic
    // pipeline by 1 LSB in some pixels (measured), and adding factor(0) * 1
    // leaves the result exact.
    const half4 emTex   = kEmissive ? sampleOr(textures, m.emissiveTex, in.uv, half4(1.0h)) : half4(1.0h);
    const half4 nTex    = sampleOr(textures, m.normalTex, in.uv, half4(0.5h, 0.5h, 1.0h, 1.0h));

    const float4 baseColor = float4(m.baseColor[0], m.baseColor[1], m.baseColor[2], m.baseColor[3]) * float4(baseTex);
    if (baseColor.a < m.alphaCutoff) {
        discard_fragment();
    }

    // glTF: G = roughness, B = metallic.
    const float roughness = clamp(m.roughness * float(mrTex.g), 0.04, 1.0);
    const float metallic  = saturate(m.metallic * float(mrTex.b));
    const float occlusion = mix(1.0, float(aoTex.r), m.occlusionStrength);

    // Two-sided lighting.  A mirrored model matrix reverses the winding, so
    // the rasteriser's facing is inverted for those instances.
    const bool front = frontFacing != (in.mirrored != 0);
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
        const device GPULight& light = lights[i];
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
        color += evaluateLight(N, V, L, lightColor * attenuation, baseColor.rgb, metallic, roughness);
    }

    // Ambient: diffuse for dielectrics only, specular from the hemisphere in
    // the reflected direction, widened towards N as the lobe gets rougher.
    const float NdotV = max(dot(N, V), 1e-4);
    const float3 f0 = mix(float3(0.04), baseColor.rgb, metallic);
    const float3 specularDir = normalize(mix(reflect(-V, N), N, roughness * roughness));
    const float3 ambientDiffuse = hemisphere(N) * baseColor.rgb * (1.0 - metallic);
    const float3 ambientSpecular = hemisphere(specularDir) * envBRDFApprox(f0, roughness, NdotV);
    color += (ambientDiffuse + ambientSpecular) * occlusion;

    color += float3(m.emissive[0], m.emissive[1], m.emissive[2]) * float3(emTex.rgb);

    // Salt (F3.1 cold-compile measurements): a defined salt embeds its value in
    // the binary so the OS shader cache cannot serve the specialisation.  The
    // guard depends on a runtime value (a light count of 0xFFFFFFFF cannot
    // occur: the light buffer would be 96 GB), so the compiler cannot fold the
    // branch away, yet it is never taken and the output cannot change (an
    // early return leaves the colour arithmetic untouched, unlike an add).  With
    // the salt undefined (every normal pipeline) the branch is removed.
    // DEBUG_MODE specialised: the debug output is chosen at pipeline creation;
    // generic: the runtime FrameConstants::debugMode, as before.
    const uint debugMode = kDebugModeSpecialised ? kDebugMode : frame.debugMode;
    if (debugMode == 1) {
        return half4(half3(N * 0.5 + 0.5), 1.0h);
    }
    if (debugMode == 2) {
        return half4(half3(baseColor.rgb), 1.0h);
    }

    return half4(half3(tonemapACES(color * frame.exposure)), 1.0h);
}
