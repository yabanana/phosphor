#pragma once
// F15.4 (ticket 23): the layer stack's base resolve, shared by the primary resolve (material_shading.h, one sampler policy with
// derivative gradients) and every secondary-hit consumer (reflection hits, GI probe and reservoir hits, the reflection probe capture:
// level-0 policies).  The mask is sampled at the primitive uv with a clamp-to-edge sampler; every layer with a non-zero weight is an
// ordinary material of the library whose base parameters are sampled at uv * uvScale with the same expressions as the single-material
// path (so a weight vector e_j reproduces layer j exactly), blended in parameter space (renderer/layer_blend.h) and shaded ONCE by the
// base model.  Includers: `using namespace metal; using namespace phosphor;`, gpu_types.h and layer_blend.h before this file.
//
// The sampler policy is a struct with two static member templates (T = the bindless table element, TextureHandle or DITextureHandle, with
// a `tex` member):
//   half4 sample(const device T *table, uint index, float2 uv, half4 fallback, float2 dx, float2 dy)   a repeat-addressed layer texture
//   half4 mask(const device T *table, uint index, float2 uv, half4 fallback, float2 dx, float2 dy)     the clamp-to-edge mask

struct LayerSurface {
    LayerWeights w;     // the final weights (normalised, height contrast applied)
    float3 baseColor;
    float roughness, metallic, occlusion;
    float3 tn;          // the tangent-space normal vector for normalize(T a + B b + N c)
    float3 emission;    // sum w_i emissive_i x emissive texture(uv_i); 0 when `wantEmission` is false or no layer emits
    float3 emitFactor[4], emitTexel[4]; // the per-layer factor and texel of that product (the primary resolve evaluates a sole layer's product inline, like a single material)
};

template <typename T, typename Samples>
static LayerSurface resolveLayerBase(GPUMaterialExt3 e3, const device GPUMaterial *materials, const device T *textures, float2 uv, float2 dx,
                                     float2 dy, bool wantEmission) {
    const uint count = min(e3.layerCount, LAYER_STACK_MAX);
    const float4 mask = float4(Samples::mask(textures, e3.maskTex, uv, half4(1.0h), dx, dy));
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
                const float texel = float(Samples::sample(textures, e3.heightTex[i], uv * s, half4(0.5h), dx * s, dy * s).r);
                heights.h[i] = layerHeight(e3.heightContrast[i], texel);
            }
        }
        w = layerHeightBlend(w, heights, count);
    }
    float3 color[4] = {float3(0.0f), float3(0.0f), float3(0.0f), float3(0.0f)};
    float3 emit[4] = {float3(0.0f), float3(0.0f), float3(0.0f), float3(0.0f)};
    float3 emitF[4] = {float3(0.0f), float3(0.0f), float3(0.0f), float3(0.0f)}, emitT[4] = {float3(1.0f), float3(1.0f), float3(1.0f), float3(1.0f)};
    float rough[4] = {0.0f, 0.0f, 0.0f, 0.0f}, metal[4] = {0.0f, 0.0f, 0.0f, 0.0f}, occ[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    LayerNormals normals{{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f, 1.0f}};
    for (uint i = 0; i < LAYER_STACK_MAX; ++i) {
        if (i < count && w.w[i] > 0.0f) {
            const device GPUMaterial &lm = materials[e3.layerMaterial[i]];
            const float s = e3.uvScale[i];
            const float2 uvi = uv * s, dxi = dx * s, dyi = dy * s;
            const half4 baseTex = Samples::sample(textures, lm.baseColorTex, uvi, half4(1.0h), dxi, dyi);
            const half4 mrTex = Samples::sample(textures, lm.metallicRoughnessTex, uvi, half4(1.0h), dxi, dyi);
            const half4 aoTex = Samples::sample(textures, lm.occlusionTex, uvi, half4(1.0h), dxi, dyi);
            const half4 nTex = Samples::sample(textures, lm.normalTex, uvi, half4(0.5h, 0.5h, 1.0h, 1.0h), dxi, dyi);
            color[i] = float3(lm.baseColor[0], lm.baseColor[1], lm.baseColor[2]) * float3(baseTex.rgb);
            rough[i] = clamp(lm.roughness * float(mrTex.g), 0.04, 1.0);
            metal[i] = saturate(lm.metallic * float(mrTex.b));
            occ[i] = mix(1.0, float(aoTex.r), lm.occlusionStrength);
            float3 tn = float3(nTex.xyz) * 2.0 - 1.0;
            tn.xy *= lm.normalScale;
            normals.tx[i] = tn.x;
            normals.ty[i] = tn.y;
            normals.tz[i] = tn.z;
            // The layer's own emission (a material of the library: factor x texture), the same expression as the single material's.
            if (wantEmission && (lm.emissive[0] > 0.0f || lm.emissive[1] > 0.0f || lm.emissive[2] > 0.0f)) {
                const half4 emTex = Samples::sample(textures, lm.emissiveTex, uvi, half4(1.0h), dxi, dyi);
                emitF[i] = float3(lm.emissive[0], lm.emissive[1], lm.emissive[2]);
                emitT[i] = float3(emTex.rgb);
                emit[i] = emitF[i] * emitT[i];
            }
        }
    }
    LayerSurface out;
    out.w = w;
    out.baseColor = float3(layerBlendScalar(w, color[0].x, color[1].x, color[2].x, color[3].x),
                           layerBlendScalar(w, color[0].y, color[1].y, color[2].y, color[3].y),
                           layerBlendScalar(w, color[0].z, color[1].z, color[2].z, color[3].z));
    out.roughness = layerBlendScalar(w, rough[0], rough[1], rough[2], rough[3]);
    out.metallic = layerBlendScalar(w, metal[0], metal[1], metal[2], metal[3]);
    out.occlusion = layerBlendScalar(w, occ[0], occ[1], occ[2], occ[3]);
    const LayerNormalResult n = layerNormalBlend(w, normals);
    out.tn = float3(n.a, n.b, n.c);
    for (uint k = 0; k < LAYER_STACK_MAX; ++k) {
        out.emitFactor[k] = emitF[k];
        out.emitTexel[k] = emitT[k];
    }
    out.emission = float3(layerBlendScalar(w, emit[0].x, emit[1].x, emit[2].x, emit[3].x),
                          layerBlendScalar(w, emit[0].y, emit[1].y, emit[2].y, emit[3].y),
                          layerBlendScalar(w, emit[0].z, emit[1].z, emit[2].z, emit[3].z));
    return out;
}
